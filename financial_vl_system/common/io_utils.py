from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# 正则表达式: 匹配一个或多个连续的 <image>
IMAGE_TOKEN_RE = re.compile(r"(?:<image>)+")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    # indent=2：缩进 2 个空格，方便人读
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
    return records


def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# 模型输出可能不是纯 JSON，而是“解释文字 + JSON + 后续补充”，
# 这个函数尽量从中找出第一个合法 JSON 对象。
def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object from model output."""
    text = text.strip()
    if not text:
        return None

    # Fast path: whole string is JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: scan for balanced braces.
    start_positions = [i for i, ch in enumerate(text) if ch == "{"]
    for start in start_positions:
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
    return None


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, (int, float)):
        return str(text)
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_scale(scale: Any) -> str:
    s = normalize_text(scale).lower()
    return "" if s in {"", "none", "null"} else s


def normalize_answer(answer: Any) -> str:
    return normalize_text(answer)


def safe_get_question(sample: Dict[str, Any]) -> str:
    """
    Heuristic to extract the user question text from a sample, handling various formats.
    """
    if "question" in sample:
        return normalize_text(sample["question"])
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                match = IMAGE_TOKEN_RE.match(content)
                if match:
                    return normalize_text(content[match.end() :])
                return normalize_text(content)
    return ""


def swift_style_to_qwen_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert ms-swift style training sample into Qwen3-VL chat-format messages.

    Input sample example:
        {
            "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "<image><image>Question"},
            {"role": "assistant", "content": "{...}"}
            ],
            "images": ["/abs/path/1.png", "/abs/path/2.png"]
        }
    Output sample example:
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/data/a.png"},
                {"type": "image", "image": "/data/b.png"},
                {"type": "text", "text": "Compare the two charts."}
            ]
        }
    """
    messages = sample["messages"]
    image_paths = sample.get("images", [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    converted: List[Dict[str, Any]] = []
    used = 0
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role != "user" or not isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        match = IMAGE_TOKEN_RE.match(content)
        image_count = 0
        question = content
        if match:
            image_count = match.group(0).count("<image>")
            question = content[match.end() :]

        user_content: List[Dict[str, str]] = []
        for i in range(image_count):
            if used + i >= len(image_paths):
                raise ValueError(
                    f"Found {image_count} <image> tokens but only {len(image_paths)} images. Sample meta={sample.get('meta')}"
                )
            user_content.append({"type": "image", "image": str(image_paths[used + i])})
        used += image_count
        if question.strip():
            user_content.append({"type": "text", "text": question.strip()})
        converted.append({"role": role, "content": user_content})
    return converted


def build_answer_schema_output(raw_text: str) -> Dict[str, Any]:
    """Parse model output into a structured schema with 'answer', 'scale', and 'json_ok' fields."""
    obj = extract_first_json_object(raw_text)
    if obj is None:
        return {
            "answer": "",
            "scale": "",
            "json_ok": False,
            "raw_output": raw_text,
        }
    return {
        "answer": normalize_answer(obj.get("answer", "")),
        "scale": normalize_scale(obj.get("scale", "")),
        "json_ok": True,
        "raw_output": raw_text,
        "parsed": obj,
    }
