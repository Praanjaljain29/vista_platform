from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from llama_service import get_local_llama_service, is_mock_mode, model_status

SUPPORTED_OBJECTS = {
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "bike",
    "motorbike",
    "motorcycle",
    "chair",
    "table",
    "cup",
    "bottle",
    "phone",
    "laptop",
    "tv",
    "monitor",
    "dog",
    "cat",
    "backpack",
    "bag",
}

RELATION_KEYWORDS = {
    "near": ["near", "next to", "beside", "close to", "around"],
    "inside": ["inside", "in the"],
    "on": ["on", "on top of", "over"],
    "behind": ["behind"],
    "in_front_of": ["in front of", "ahead of"],
}

ACTION_KEYWORDS = [
    "find",
    "show",
    "track",
    "follow",
    "detect",
    "watch",
    "search",
    "locate",
]


@dataclass
class TargetEntity:
    type: str | None = None
    attributes: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RelatedEntity:
    type: str | None = None
    attributes: list[str] = field(default_factory=list)


@dataclass
class Relation:
    type: str | None = None


@dataclass
class SearchIntent:
    target: TargetEntity = field(default_factory=TargetEntity)
    related_entities: list[RelatedEntity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    action: str | None = None
    temporal_constraint: str | None = None
    original_query: str = ""
    unsupported_attributes: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    ok: bool
    intent: SearchIntent
    source: str
    raw_output: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _pick_first_matching_phrase(query: str, phrases: list[str]) -> bool:
    for phrase in phrases:
        if " " in phrase:
            if phrase in query:
                return True
        elif re.search(rf"\b{re.escape(phrase)}\b", query):
            return True
    return False


def _find_supported_object(query: str, exclude: str | None = None) -> str | None:
    for candidate in sorted(SUPPORTED_OBJECTS, key=len, reverse=True):
        if exclude and candidate == exclude:
            continue
        if re.search(rf"\b{re.escape(candidate)}\b", query):
            return candidate
    return None


def _fallback_intent(query: str) -> SearchIntent:
    normalized = _normalize_text(query)
    target_type = _find_supported_object(normalized)
    related_type = None

    if target_type:
        related_type = _find_supported_object(normalized, exclude=target_type)

    relation: str | None = None
    for relation_name, phrases in RELATION_KEYWORDS.items():
        if _pick_first_matching_phrase(normalized, phrases):
            relation = relation_name
            break

    action: str | None = None
    for keyword in ACTION_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            action = keyword
            break

    attribute_words = [
        word
        for word in normalized.split()
        if word in {"red", "blue", "green", "black", "white", "yellow", "small", "large", "big"}
    ]

    intent = SearchIntent(
        target=TargetEntity(
            type=target_type,
            attributes=attribute_words,
            description="",
        ),
        related_entities=(
            [RelatedEntity(type=related_type)] if related_type else []
        ),
        relations=(
            [Relation(type=relation)] if relation else []
        ),
        action=action,
        original_query=query,
        unsupported_attributes=attribute_words,
    )
    return intent


def _intent_to_dict(intent: SearchIntent) -> dict[str, Any]:
    return {
        "target": {
            "type": intent.target.type,
            "attributes": intent.target.attributes,
            "description": intent.target.description,
        },
        "related_entities": [
            {"type": ent.type, "attributes": ent.attributes}
            for ent in intent.related_entities
        ],
        "relations": [
            {"type": r.type}
            for r in intent.relations
        ],
        "action": intent.action,
        "temporal_constraint": intent.temporal_constraint,
        "original_query": intent.original_query,
        "unsupported_attributes": intent.unsupported_attributes,
    }


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_intent(data: dict[str, Any], raw_query: str) -> SearchIntent:
    intent = SearchIntent(original_query=raw_query)

    # Parse target
    target_data = data.get("target")
    if isinstance(target_data, dict):
        t_type = target_data.get("type")
        intent.target = TargetEntity(
            type=str(t_type).strip().lower() if isinstance(t_type, str) and t_type.strip() else None,
            attributes=_coerce_list(target_data.get("attributes")),
            description=str(target_data.get("description", "") or ""),
        )
    elif isinstance(target_data, str) and target_data.strip():
        # Graceful fallback: Llama returned a bare string for target
        intent.target = TargetEntity(type=target_data.strip().lower())

    # Parse related_entities
    related_raw = data.get("related_entities")
    if isinstance(related_raw, list):
        for item in related_raw:
            if isinstance(item, dict):
                re_type = item.get("type")
                intent.related_entities.append(
                    RelatedEntity(
                        type=str(re_type).strip().lower() if isinstance(re_type, str) and re_type.strip() else None,
                        attributes=_coerce_list(item.get("attributes")),
                    )
                )

    # Parse relations
    relations_raw = data.get("relations")
    if isinstance(relations_raw, list):
        for item in relations_raw:
            if isinstance(item, dict):
                r_type = item.get("type")
                intent.relations.append(
                    Relation(
                        type=str(r_type).strip().lower() if isinstance(r_type, str) and r_type.strip() else None,
                    )
                )

    # Scalar fields
    if isinstance(data.get("action"), str):
        intent.action = data["action"].strip().lower() or None

    if isinstance(data.get("temporal_constraint"), str):
        intent.temporal_constraint = data["temporal_constraint"].strip() or None

    intent.unsupported_attributes = _coerce_list(data.get("unsupported_attributes"))

    if isinstance(data.get("original_query"), str) and data["original_query"].strip():
        intent.original_query = data["original_query"].strip()

    if not intent.target.type:
        intent = _fallback_intent(raw_query)

    return intent


def _extract_json_candidates(text: str) -> list[str]:
    candidates = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    code_block_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(match.strip() for match in code_block_matches if match.strip())

    brace_start = text.find("{")
    while brace_start != -1:
        candidates.append(text[brace_start:].strip())
        brace_start = text.find("{", brace_start + 1)

    return candidates


def _parse_json_output(text: str, raw_query: str) -> tuple[SearchIntent | None, str | None]:
    last_error = None
    for candidate in _extract_json_candidates(text):
        try:
            decoded = json.loads(candidate)
            if isinstance(decoded, dict):
                return _validate_intent(decoded, raw_query), None
            last_error = "Llama output was not a JSON object."
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    return None, last_error


def build_parser_prompt(query: str) -> str:
    schema = {
        "target": {"type": "person", "attributes": [], "description": ""},
        "related_entities": [{"type": "car", "attributes": []}],
        "relations": [{"type": "near"}],
        "action": None,
        "temporal_constraint": None,
        "unsupported_attributes": [],
        "original_query": query,
    }
    instructions = (
        "Convert the user's natural-language VISTA query into JSON only.\n"
        "Return a single JSON object and nothing else.\n"
        "Do not invent detections or evidence.\n"
        "Use null when a field is unknown.\n\n"
        "YOLO class vocabulary: person, car, truck, bus, bicycle, motorcycle, "
        "chair, table, cup, bottle, phone, laptop, tv, monitor, dog, cat, "
        "backpack, bag.\n\n"
        "Rules:\n"
        "- Map synonyms to YOLO class names: "
        "automobile\u2192car, vehicle\u2192car/truck/bus/motorcycle/bicycle, "
        "two-wheeler\u2192motorcycle/bicycle, pedestrian/human\u2192person.\n"
        "- If a concept maps to multiple classes, pick the most likely one for target.type.\n"
        "- Put unverifiable visual attributes (color, clothing, size, activity) "
        "in unsupported_attributes.\n"
        "- Return ONLY valid JSON matching the schema below.\n\n"
        f"Schema example: {json.dumps(schema)}\n\n"
        f"User query: {query}"
    )
    return instructions


def parse_query(query: str) -> ParseResult:
    cleaned_query = query.strip()
    if not cleaned_query:
        return ParseResult(
            ok=False,
            intent=SearchIntent(original_query=query),
            source="empty",
            error="Query is empty.",
        )

    if is_mock_mode():
        intent = _fallback_intent(cleaned_query)
        return ParseResult(
            ok=True,
            intent=intent,
            source="mock",
            raw_output=json.dumps(_intent_to_dict(intent), indent=2),
        )

    status = model_status()
    if not status["available"]:
        return ParseResult(
            ok=False,
            intent=SearchIntent(original_query=cleaned_query),
            source="missing",
            error=str(status["message"]),
        )

    llm = get_local_llama_service()
    result = llm.generate_json(
        user_prompt=build_parser_prompt(cleaned_query),
        system_prompt=(
            "You are a strict JSON generator for a computer-vision query parser. "
            "Only output valid JSON matching the requested schema."
        ),
        max_tokens=256,
        temperature=0.0,
    )

    if not result.ok:
        return ParseResult(
            ok=False,
            intent=_fallback_intent(cleaned_query),
            source=result.mode,
            error=result.error,
        )

    parsed_intent, parse_error = _parse_json_output(result.text, cleaned_query)
    if parsed_intent is None:
        return ParseResult(
            ok=False,
            intent=_fallback_intent(cleaned_query),
            source="invalid-json",
            raw_output=result.text,
            error=f"Invalid JSON returned by Llama: {parse_error}",
        )

    return ParseResult(
        ok=True,
        intent=parsed_intent,
        source="llama",
        raw_output=result.text,
    )
