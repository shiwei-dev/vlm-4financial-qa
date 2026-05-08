from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common.io_utils import build_answer_schema_output, extract_first_json_object, load_jsonl, save_jsonl
from common.json_guard import build_json_schema_instruction, coerce_answer_json
from common.modeling import generate_from_messages, load_qwen_vl_model
from task2_rag.schemas import ANSWER_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer questions from retrieved pages using Qwen3-VL")
    parser.add_argument("--retrieval_file", type=str, required=True, help="reranked retrieval jsonl or raw retrieval jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="optional fine-tuned adapter checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def build_messages(question: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, str]] = []
    page_refs: List[int] = []
    for page in pages:
        page_refs.append(int(page["page_num"]))
        content.append({"type": "image", "image": page["image_path"]})

    text_lines = [
        f"Question: {question}",
        f"Available pages: {page_refs}",
        build_json_schema_instruction(page_refs),
        "For each page, brief metadata:",
    ]
    for page in pages:
        summary = page.get("structured", {}).get("page_summary", "")
        native = page.get("native_text", "")[:800]
        text_lines.append(f"- page {page['page_num']}: summary={summary} | native_text={native}")
    content.append({"type": "text", "text": "\n".join(text_lines)})
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.retrieval_file)
    processor, model = load_qwen_vl_model(
        model_name_or_path=args.base_model,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )

    outputs: List[Dict[str, Any]] = []
    for record in records:
        pages = record.get("top_pages_reranked") or record.get("top_pages") or []
        pages = pages[: args.top_k]
        page_nums = [int(p["page_num"]) for p in pages]
        messages = build_messages(record["question"], pages)
        raw = generate_from_messages(
            processor=processor,
            model=model,
            messages=messages,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        # parsed = extract_first_json_object(raw) or {}
        # answer_payload = build_answer_schema_output(raw)
        answer_payload = coerce_answer_json(
            raw,
            allowed_pages=page_nums,
            default_source_pages=page_nums,
        )
        outputs.append(
            {
                "id": record.get("id", ""),
                "question": record["question"],
                "answer": answer_payload["answer"],
                "scale": answer_payload["scale"],
                "source_pages": answer_payload["source_pages"],
                "abstain": answer_payload["abstain"],
                "confidence": answer_payload["confidence"],
                "json_ok": answer_payload["json_ok"],
                "schema_ok": answer_payload["schema_ok"],
                "schema_errors": answer_payload["schema_errors"],
                "raw_output": raw,
                "parsed_output": answer_payload["parsed"],
                "retrieved_pages": page_nums,
            }
        )

    save_jsonl(outputs, args.output_file)
    print(f"Saved QA outputs to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
