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
    payload = {
        "query": query,
        "intent": {
            "target_object": parse_result.intent.target_object,
            "attributes": parse_result.intent.attributes,
            "related_object": parse_result.intent.related_object,
            "relation": parse_result.intent.relation,
            "action": parse_result.intent.action,
            "motion": parse_result.intent.motion,
            "event": parse_result.intent.event,
            "time_window_seconds": parse_result.intent.time_window_seconds,
        },
        "search_result": evidence,
    }

    return json.dumps(payload, indent=2)


def _mock_response(search_result: SearchResult) -> ResponseResult:
    evidence = search_result.evidence
    if search_result.found:
        timestamp = evidence.first_timestamp if evidence.first_timestamp is not None else 0.0
        track_id = evidence.track_id if evidence.track_id is not None else "unknown"
        frame_no = evidence.best_frame if evidence.best_frame is not None else "unknown"
        text = (
            f"[MOCK LLM] Yes, I found {evidence.object_class or 'the target object'} "
            f"around {timestamp:.1f} seconds. Track ID {track_id}, frame {frame_no}."
        )
    else:
        text = "[MOCK LLM] No matching object was found in the uploaded video."

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
            "Use only the provided structured evidence. "
            "Do not invent track IDs, timestamps, frames, confidence scores, or boxes. "
            "Give a short answer and mention whether the object was found."
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