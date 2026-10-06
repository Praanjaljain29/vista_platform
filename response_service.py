from __future__ import annotations

from dataclasses import dataclass
import json

from llama_service import get_local_llama_service, is_mock_mode, model_status
from query_parser import ParseResult
from search_service import SearchResult, summarize_search_result


@dataclass(slots=True)
class ResponseResult:
    ok: bool
    text: str
    mode: str
    raw_output: str | None = None
    error: str | None = None


def _build_response_prompt(query: str, parse_result: ParseResult, search_result: SearchResult) -> str:
    evidence = summarize_search_result(search_result)
    intent = parse_result.intent

    payload = {
        "query": query,
        "intent": {
            "target": intent.target.type,
            "related_entity": intent.related_entities[0].type if intent.related_entities else None,
            "relation": intent.relations[0].type if intent.relations else None,
            "action": intent.action,
            "temporal_constraint": intent.temporal_constraint,
            "unsupported_attributes": intent.unsupported_attributes,
        },
        "search_result": evidence,
    }

    return json.dumps(payload, indent=2)


def _mock_response(search_result: SearchResult) -> ResponseResult:
    evidence = search_result.evidence
    if search_result.found:
        timestamp = evidence.first_timestamp if evidence.first_timestamp is not None else 0.0
        text = (
            f"[MOCK] Yes, I found {evidence.object_class or 'the target object'} "
            f"around {timestamp:.1f} seconds into the video."
        )
    else:
        text = "[MOCK] No matching object was found in the uploaded video."

    return ResponseResult(ok=True, text=text, mode="mock")


def generate_response(query: str, parse_result: ParseResult, search_result: SearchResult) -> ResponseResult:
    if is_mock_mode():
        return _mock_response(search_result)

    status = model_status()
    if not status["available"]:
        return ResponseResult(
            ok=False,
            text=str(status["message"]),
            mode="missing",
            error=str(status["message"]),
        )

    llm = get_local_llama_service()
    response = llm.generate(
        user_prompt=_build_response_prompt(query, parse_result, search_result),
        system_prompt=(
            "You are VISTA, a concise video-search assistant. "
            "Use ONLY the provided structured evidence. "
            "Do not invent track IDs, timestamps, frames, confidence scores, or bounding boxes. "
            "Refer to objects by their class name and time only (e.g. 'around 18 seconds'). "
            "Do NOT show track IDs or confidence numbers in the answer. "
            "If unsupported_attributes is non-empty, acknowledge those attributes "
            "could not be visually verified. "
            "Give a short, human-friendly answer."
        ),
        max_tokens=192,
        temperature=0.2,
    )

    if not response.ok:
        return ResponseResult(
            ok=False,
            text=response.error or "Failed to generate a local response.",
            mode=response.mode,
            error=response.error,
            raw_output=None,
        )

    return ResponseResult(
        ok=True,
        text=response.text,
        mode=response.mode,
        raw_output=response.raw if isinstance(response.raw, str) else None,
    )
