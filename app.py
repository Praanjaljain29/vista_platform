from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict

import cv2
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from llama_service import model_status
from query_parser import parse_query
from response_service import generate_response
from search_service import (
    DetectionRecord,
    SearchEvidence,
    SearchResult,
    VisualSearchEngine,
    summarize_search_result,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "yolo26n.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)

# Maximum upload size = 500 MB.
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm",
}

CONFIDENCE = 0.35

print("Loading YOLO26 model...")
model = YOLO(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has an allowed video extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _empty_search_result() -> SearchResult:
    return SearchResult(
        found=False,
        evidence=SearchEvidence(found=False),
        images=[],
        candidate_count=0,
    )


def _result_conversation(query: str, assistant_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "text": query,
        },
        {
            "role": "assistant",
            "text": assistant_text,
        },
    ]


def process_video(video_path: str, query: str, job_id: str) -> dict[str, object]:
    """
    Run YOLO26 + ByteTrack on the uploaded video and keep the search logic
    separate from Llama by using the structured query parser and search engine.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detections_csv = os.path.join(RESULT_DIR, f"{job_id}_detections.csv")
    classes_csv = os.path.join(RESULT_DIR, f"{job_id}_classes.csv")

    parse_result = parse_query(query)
    intent = parse_result.intent
    search_engine = None

    if parse_result.source != "missing" and (
        intent.target_object or intent.related_object or intent.action
    ):
        search_engine = VisualSearchEngine(intent, job_id, RESULT_DIR)

    classes_seen: dict[int, str] = {}
    frame_no = 0
    serial_no = 1
    start_time = time.time()

    with open(detections_csv, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
                "serial_no",
                "frame_no",
                "timestamp",
                "object_class",
                "class_id",
                "track_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "frame_width",
                "frame_height",
            ]
        )

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_no += 1
            frame_height, frame_width = frame.shape[:2]

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE,
                verbose=False,
            )

            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue

                boxes = result.boxes

                for index in range(len(boxes)):
                    class_id = int(boxes.cls[index])
                    class_name = result.names[class_id]
                    confidence = float(boxes.conf[index])
                    track_id = int(boxes.id[index])
                    x1, y1, x2, y2 = boxes.xyxy[index].tolist()
                    timestamp = frame_no / fps

                    classes_seen[class_id] = class_name

                    detection = DetectionRecord(
                        serial_no=serial_no,
                        frame_no=frame_no,
                        timestamp=round(timestamp, 3),
                        object_class=class_name,
                        class_id=class_id,
                        track_id=track_id,
                        confidence=round(confidence, 4),
                        x1=round(x1, 2),
                        y1=round(y1, 2),
                        x2=round(x2, 2),
                        y2=round(y2, 2),
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )

                    writer.writerow(
                        [
                            detection.serial_no,
                            detection.frame_no,
                            detection.timestamp,
                            detection.object_class,
                            detection.class_id,
                            detection.track_id,
                            detection.confidence,
                            detection.x1,
                            detection.y1,
                            detection.x2,
                            detection.y2,
                            detection.frame_width,
                            detection.frame_height,
                        ]
                    )

                    serial_no += 1

                    if search_engine is not None:
                        search_engine.consider_detection(frame, detection)

            if frame_no % 300 == 0:
                elapsed = time.time() - start_time
                progress = (frame_no / total_frames) * 100 if total_frames else 0
                print(
                    f"Processed {frame_no}/{total_frames} frames ({progress:.1f}%) | "
                    f"Detections: {serial_no - 1} | Time: {elapsed:.1f}s"
                )

    cap.release()

    with open(classes_csv, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["class_id", "object_class"])
        for class_id in sorted(classes_seen):
            writer.writerow([class_id, classes_seen[class_id]])

    search_result = search_engine.finalize() if search_engine is not None else _empty_search_result()
    response_result = generate_response(query, parse_result, search_result)

    elapsed = time.time() - start_time
    conversation = _result_conversation(query, response_result.text)

    return {
        "frames_processed": frame_no,
        "detections": serial_no - 1,
        "unique_classes": len(classes_seen),
        "time_minutes": round(elapsed / 60, 2),
        "images": search_result.images,
        "detections_csv": os.path.basename(detections_csv),
        "classes_csv": os.path.basename(classes_csv),
        "intent": asdict(intent),
        "parse_result": {
            "ok": parse_result.ok,
            "source": parse_result.source,
            "error": parse_result.error,
            "warnings": parse_result.warnings,
            "raw_output": parse_result.raw_output,
        },
        "search_result": summarize_search_result(search_result),
        "assistant_message": response_result.text,
        "assistant_mode": response_result.mode,
        "assistant_ok": response_result.ok,
        "assistant_error": response_result.error,
        "conversation": conversation,
        "llm_status": model_status(),
    }


@app.route("/", methods=["GET"])
def index():
    return render_home(
        result=None,
        error=None,
        query="",
    )


@app.route("/process", methods=["POST"])
def process():
    video_file = request.files.get("video")
    query = request.form.get("query", "").strip()

    if not video_file or video_file.filename == "":
        return render_home(
            result=None,
            error="Please choose a video.",
            query=query,
        )

    if not allowed_file(video_file.filename):
        return render_home(
            result=None,
            error="Please upload an MP4, AVI, MOV, MKV, or WEBM video.",
            query=query,
        )

    if not query:
        return render_home(
            result=None,
            error="Enter a natural-language query such as: Find the person near the car.",
            query=query,
        )

    filename = secure_filename(video_file.filename)
    job_id = f"job_{int(time.time())}"
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")
    video_file.save(video_path)

    try:
        result = process_video(video_path, query, job_id)
        return render_home(
            result=result,
            error=None,
            query=query,
        )
    except Exception as exc:
        return render_home(
            result=None,
            error=str(exc),
            query=query,
        )


@app.route("/results/<path:filename>")
def result_file(filename: str):
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)


@app.route("/download/<path:filename>")
def download_file(filename: str):
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)


def render_home(result: dict[str, object] | None, error: str | None, query: str):
    runtime_status = model_status()
    return render_template(
        "index.html",
        result=result,
        error=error,
        query=query,
        runtime_status=runtime_status,
    )


if __name__ == "__main__":
    app.run(debug=True)
