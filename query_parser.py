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


@dataclass(slots=True)
class SearchIntent:
    target_object: str | None = None
    attributes: list[str] = field(default_factory=list)
    related_object: str | None = None
    relation: str | None = None
    action: str | None = None
    motion: str | None = None
    event: str | None = None
    time_window_seconds: float | None = None
    raw_query: str = ""


@dataclass(slots=True)
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
    target = _find_supported_object(normalized)
    related = None

    if target:
        related = _find_supported_object(normalized, exclude=target)

    relation = None
    for relation_name, phrases in RELATION_KEYWORDS.items():
        if _pick_first_matching_phrase(normalized, phrases):
            relation = relation_name
            break

    action = None
    for keyword in ACTION_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            action = keyword
            break

    attributes = [
        word
        for word in normalized.split()
        if word in {"red", "blue", "green", "black", "white", "yellow", "small", "large", "big"}
    ]

    return SearchIntent(
        target_object=target,
        attributes=attributes,
        related_object=related,
        relation=relation,
        action=action,
        raw_query=query,
    )


def _intent_to_dict(intent: SearchIntent) -> dict[str, Any]:
    return {
        "target_object": intent.target_object,
        "attributes": intent.attributes,
        "related_object": intent.related_object,
        "relation": intent.relation,
        "action": intent.action,
        "motion": intent.motion,
        "event": intent.event,
        "time_window_seconds": intent.time_window_seconds,
        "raw_query": intent.raw_query,
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
    intent = SearchIntent(raw_query=raw_query)

    if isinstance(data.get("target_object"), str):
        intent.target_object = data["target_object"].strip().lower() or None

    intent.attributes = _coerce_list(data.get("attributes"))

    if isinstance(data.get("related_object"), str):
        intent.related_object = data["related_object"].strip().lower() or None

    if isinstance(data.get("relation"), str):
        intent.relation = data["relation"].strip().lower() or None

    if isinstance(data.get("action"), str):
        intent.action = data["action"].strip().lower() or None

    if isinstance(data.get("motion"), str):
        intent.motion = data["motion"].strip().lower() or None

    if isinstance(data.get("event"), str):
        intent.event = data["event"].strip().lower() or None

    intent.time_window_seconds = _coerce_float(data.get("time_window_seconds"))

    if not intent.target_object:
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
        "target_object": "person",
        "attributes": [],
        "related_object": None,
        "relation": None,
        "action": None,
        "motion": None,
        "event": None,
        "time_window_seconds": None,
    }

    return (
        "Convert the user's natural-language VISTA query into JSON only. "
        "Return a single JSON object and nothing else. "
        "Do not invent detections or evidence. "
        "Use null when a field is unknown.\n\n"
        f"Schema: {json.dumps(schema)}\n\n"
        f"User query: {query}"
    )


def parse_query(query: str) -> ParseResult:
    cleaned_query = query.strip()
    if not cleaned_query:
        return ParseResult(
            ok=False,
            intent=SearchIntent(raw_query=query),
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
            intent=SearchIntent(raw_query=cleaned_query),
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