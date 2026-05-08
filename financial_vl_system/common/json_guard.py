from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common.io_utils import extract_first_json_object, normalize_answer, normalize_scale

_ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _loose_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parser for common LLM formatting mistakes.

    This function is intentionally conservative: it fixes wrappers/fences,
    full-width quotes, Python booleans/None, and trailing commas, then returns
    a dict only when the resulting text can be parsed as valid JSON.
    """
    if not text or not text.strip():
        return None

    candidates: List[str] = []
    base = _strip_markdown_fence(text)
    candidates.append(base)

    # Try the existing balanced-brace extractor first.
    extracted = extract_first_json_object(base)
    if isinstance(extracted, dict):
        return extracted

    # Build a repaired candidate for common non-JSON outputs.
    repaired = base
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    candidates.append(repaired)

    # If extra prose surrounds the JSON, try balanced substrings again after repair.
    for start in [m.start() for m in re.finditer(r"{", repaired)]:
        depth = 0
        for end, ch in enumerate(repaired[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                candidates.append(repaired[start : end + 1])
                break

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是", "支持", "abstain"}


def _coerce_source_pages(value: Any, allowed_pages: Optional[Sequence[int]] = None) -> List[int]:
    allowed = set(int(x) for x in allowed_pages) if allowed_pages else None
    if value is None:
        pages: List[Any] = []
    elif isinstance(value, list):
        pages = value
    else:
        # Accept strings like "[3, 4]" or "3,4".
        text = str(value)
        pages = re.findall(r"\d+", text)

    cleaned: List[int] = []
    for item in pages:
        try:
            page = int(item)
        except Exception:
            continue
        if allowed is not None and page not in allowed:
            continue
        if page not in cleaned:
            cleaned.append(page)
    return cleaned


def coerce_answer_json(
    raw_text: str,
    *,
    allowed_pages: Optional[Sequence[int]] = None,
    default_source_pages: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Parse, validate, and normalize the answer JSON schema used by task2.

    Returns a stable dict with:
      answer, scale, source_pages, abstain, confidence, json_ok, schema_ok,
      schema_errors, raw_output, parsed
    """
    obj = _loose_json_loads(raw_text)
    errors: List[str] = []

    if obj is None:
        obj = {}
        errors.append("json_parse_failed")

    required = {"answer", "scale", "source_pages", "abstain", "confidence"}
    missing = sorted(required - set(obj.keys()))
    if missing:
        errors.append("missing_keys:" + ",".join(missing))

    source_pages = _coerce_source_pages(obj.get("source_pages"), allowed_pages=allowed_pages)
    if not source_pages and default_source_pages:
        source_pages = _coerce_source_pages(default_source_pages, allowed_pages=allowed_pages)
        if source_pages:
            errors.append("source_pages_defaulted_from_retrieval")

    confidence = str(obj.get("confidence", "low")).strip().lower()
    if confidence not in _ALLOWED_CONFIDENCE:
        confidence = "low"
        errors.append("confidence_normalized_to_low")

    payload = {
        "answer": normalize_answer(obj.get("answer", "")),
        "scale": normalize_scale(obj.get("scale", "")),
        "source_pages": source_pages,
        "abstain": _to_bool(obj.get("abstain", False)),
        "confidence": confidence,
        "json_ok": obj != {} and "json_parse_failed" not in errors,
        "schema_ok": not errors,
        "schema_errors": errors,
        "raw_output": raw_text,
        "parsed": obj,
    }

    # If the model abstains, enforce an empty answer for downstream metrics/UI.
    if payload["abstain"]:
        payload["answer"] = ""
    return payload


def build_json_schema_instruction(allowed_pages: Iterable[int]) -> str:
    pages = sorted({int(p) for p in allowed_pages})
    return (
        "Return JSON only. Required schema: "
        '{"answer": string, "scale": string, "source_pages": int[], '
        '"abstain": boolean, "confidence": "high|medium|low"}. '
        f"source_pages must be selected from {pages}. Do not add markdown fences or explanations."
    )
