from __future__ import annotations

from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
from typing import Any

from query_parser import SearchIntent
from class_mapper import SemanticClassMapper


@dataclass(slots=True)
class DetectionRecord:
    serial_no: int
    frame_no: int
    timestamp: float
    object_class: str
    class_id: int
    track_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_width: int
    frame_height: int


@dataclass(slots=True)
class SearchEvidence:
    found: bool
    object_class: str | None = None
    track_id: int | None = None
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    best_frame: int | None = None
    confidence: float | None = None
    bbox: dict[str, float] | None = None
    related_object: str | None = None
    related_track_id: int | None = None
    relation: str | None = None
    match_score: float | None = None
    frames_seen: int = 0


@dataclass(slots=True)
class SearchCandidate:
    record: DetectionRecord
    crop_filename: str | None = None
    related_track_id: int | None = None
    relation_score: float | None = None


@dataclass(slots=True)
class SearchResult:
    found: bool
    evidence: SearchEvidence
    images: list[dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    notes: list[str] = field(default_factory=list)


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def _center(record: DetectionRecord) -> tuple[float, float]:
    return ((record.x1 + record.x2) / 2.0, (record.y1 + record.y2) / 2.0)


def _distance(a: DetectionRecord, b: DetectionRecord) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def _frame_diagonal(record: DetectionRecord) -> float:
    return math.sqrt(record.frame_width * record.frame_width + record.frame_height * record.frame_height)


def _bbox_from_record(record: DetectionRecord) -> dict[str, float]:
    return {
        "x1": round(record.x1, 2),
        "y1": round(record.y1, 2),
        "x2": round(record.x2, 2),
        "y2": round(record.y2, 2),
    }


class VisualSearchEngine:
    def __init__(
        self,
        intent: SearchIntent,
        job_id: str,
        result_dir: str | Path,
        available_classes: set[str] | None = None,
    ) -> None:
        self.intent = intent
        self.job_id = job_id
        self.result_dir = Path(result_dir)
        self.available_classes: set[str] = available_classes or set()

        # Resolve target classes via SemanticClassMapper
        self.target_classes: list[str] = SemanticClassMapper.resolve(
            intent.target.type, self.available_classes
        )
        if not self.target_classes and intent.target.type:
            self.target_classes = [_normalize(intent.target.type)]

        # Resolve related classes
        related_type = intent.related_entities[0].type if intent.related_entities else None
        self.related_classes: list[str] = SemanticClassMapper.resolve(
            related_type, self.available_classes
        )
        if not self.related_classes and related_type:
            self.related_classes = [_normalize(related_type)]

        self.relation = _normalize(intent.relations[0].type if intent.relations else None)
        self.unsupported_attributes: list[str] = intent.unsupported_attributes

        self.records: list[DetectionRecord] = []
        self.target_candidates: list[SearchCandidate] = []
        self.related_records: list[DetectionRecord] = []

    def _is_target(self, record: DetectionRecord) -> bool:
        if not self.target_classes:
            return False
        normalized_class = _normalize(record.object_class)
        return any(tc in normalized_class or normalized_class in tc for tc in self.target_classes)

    def _is_related(self, record: DetectionRecord) -> bool:
        if not self.related_classes:
            return False
        normalized_class = _normalize(record.object_class)
        return any(rc in normalized_class or normalized_class in rc for rc in self.related_classes)

    def _save_crop(self, frame, record: DetectionRecord) -> str | None:
        import cv2

        h, w = frame.shape[:2]
        pad = 10

        ix1 = max(0, int(record.x1) - pad)
        iy1 = max(0, int(record.y1) - pad)
        ix2 = min(w, int(record.x2) + pad)
        iy2 = min(h, int(record.y2) + pad)

        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size <= 0:
            return None

        filename = f"{self.job_id}_frame_{record.frame_no}_track_{record.track_id}_serial_{record.serial_no}.jpg"
        image_path = self.result_dir / filename
        cv2.imwrite(str(image_path), crop)
        return filename

    def consider_detection(self, frame, record: DetectionRecord) -> None:
        if self._is_target(record):
            crop_filename = self._save_crop(frame, record)
            self.target_candidates.append(
                SearchCandidate(
                    record=record,
                    crop_filename=crop_filename,
                )
            )

        if self._is_related(record):
            self.related_records.append(record)

        if self._is_target(record) or self._is_related(record):
            self.records.append(record)

    def _choose_best_candidate(self) -> tuple[SearchCandidate | None, int | None, float | None]:
        if not self.target_candidates:
            return None, None, None

        if self.relation != "near" or not self.related_classes:
            best_candidate = max(
                self.target_candidates,
                key=lambda candidate: candidate.record.confidence,
            )
            return best_candidate, None, None

        related_by_frame: dict[int, list[DetectionRecord]] = {}
        for record in self.related_records:
            related_by_frame.setdefault(record.frame_no, []).append(record)

        best_candidate: SearchCandidate | None = None
        best_related_track: int | None = None
        best_score: float | None = None

        for candidate in self.target_candidates:
            related_records = related_by_frame.get(candidate.record.frame_no, [])
            if not related_records:
                continue

            frame_diagonal = _frame_diagonal(candidate.record) or 1.0
            nearest_distance = None
            nearest_related = None

            for related_record in related_records:
                distance = _distance(candidate.record, related_record)
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_related = related_record

            if nearest_distance is None or nearest_related is None:
                continue

            normalized_distance = nearest_distance / frame_diagonal
            relation_score = max(0.0, 1.0 - normalized_distance)

            if normalized_distance > 0.35:
                continue

            total_score = candidate.record.confidence + relation_score
            if best_score is None or total_score > best_score:
                best_candidate = candidate
                best_related_track = nearest_related.track_id
                best_score = total_score

        return best_candidate, best_related_track, best_score

    def finalize(self) -> SearchResult:
        best_candidate, related_track_id, relation_score = self._choose_best_candidate()

        images = sorted(
            self.target_candidates,
            key=lambda candidate: candidate.record.confidence,
            reverse=True,
        )[:2]

        image_payload = [
            {
                "filename": candidate.crop_filename,
                "frame_no": candidate.record.frame_no,
                "class_name": candidate.record.object_class,
                "confidence": round(candidate.record.confidence, 3),
                "track_id": candidate.record.track_id,
            }
            for candidate in images
            if candidate.crop_filename
        ]

        # Populate notes from unsupported_attributes
        notes: list[str] = []
        if self.unsupported_attributes:
            notes.append(
                "Note: the following attributes could not be visually verified: "
                + ", ".join(self.unsupported_attributes)
            )

        target_class_str = self.target_classes[0] if self.target_classes else None
        related_class_str = self.related_classes[0] if self.related_classes else None

        if best_candidate is None:
            return SearchResult(
                found=False,
                evidence=SearchEvidence(
                    found=False,
                    object_class=target_class_str,
                    related_object=related_class_str,
                    relation=self.relation or None,
                    frames_seen=len(self.target_candidates),
                ),
                images=image_payload,
                candidate_count=len(self.target_candidates),
                notes=notes,
            )

        track_records = [
            candidate.record
            for candidate in self.target_candidates
            if candidate.record.track_id == best_candidate.record.track_id
        ]
        if not track_records:
            track_records = [best_candidate.record]

        first_record = min(track_records, key=lambda item: item.timestamp)
        last_record = max(track_records, key=lambda item: item.timestamp)
        best_record = max(track_records, key=lambda item: item.confidence)

        evidence = SearchEvidence(
            found=True,
            object_class=best_record.object_class,
            track_id=best_record.track_id,
            first_timestamp=round(first_record.timestamp, 3),
            last_timestamp=round(last_record.timestamp, 3),
            best_frame=best_record.frame_no,
            confidence=round(best_record.confidence, 4),
            bbox=_bbox_from_record(best_record),
            related_object=related_class_str,
            related_track_id=related_track_id,
            relation=self.relation or None,
            match_score=round(relation_score, 4) if relation_score is not None else round(best_record.confidence, 4),
            frames_seen=len(track_records),
        )

        return SearchResult(
            found=True,
            evidence=evidence,
            images=image_payload,
            candidate_count=len(self.target_candidates),
            notes=notes,
        )


def read_detections_csv(detections_csv: str | Path) -> list[DetectionRecord]:
    path = Path(detections_csv)
    if not path.exists():
        return []

    records: list[DetectionRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                records.append(
                    DetectionRecord(
                        serial_no=int(row.get("serial_no", 0) or 0),
                        frame_no=int(row.get("frame_no", 0) or 0),
                        timestamp=float(row.get("timestamp", 0.0) or 0.0),
                        object_class=str(row.get("object_class", "") or ""),
                        class_id=int(row.get("class_id", 0) or 0),
                        track_id=int(row.get("track_id", 0) or 0),
                        confidence=float(row.get("confidence", 0.0) or 0.0),
                        x1=float(row.get("x1", 0.0) or 0.0),
                        y1=float(row.get("y1", 0.0) or 0.0),
                        x2=float(row.get("x2", 0.0) or 0.0),
                        y2=float(row.get("y2", 0.0) or 0.0),
                        frame_width=int(float(row.get("frame_width", 1) or 1)),
                        frame_height=int(float(row.get("frame_height", 1) or 1)),
                    )
                )
            except (TypeError, ValueError):
                continue

    return records


def summarize_search_result(search_result: SearchResult) -> dict[str, Any]:
    return {
        "found": search_result.found,
        "evidence": {
            "found": search_result.evidence.found,
            "object_class": search_result.evidence.object_class,
            "track_id": search_result.evidence.track_id,
            "first_timestamp": search_result.evidence.first_timestamp,
            "last_timestamp": search_result.evidence.last_timestamp,
            "best_frame": search_result.evidence.best_frame,
            "confidence": search_result.evidence.confidence,
            "bbox": search_result.evidence.bbox,
            "related_object": search_result.evidence.related_object,
            "related_track_id": search_result.evidence.related_track_id,
            "relation": search_result.evidence.relation,
            "match_score": search_result.evidence.match_score,
            "frames_seen": search_result.evidence.frames_seen,
        },
        "images": search_result.images,
        "candidate_count": search_result.candidate_count,
        "notes": search_result.notes,
    }
