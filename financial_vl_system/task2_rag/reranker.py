"""
Reranker for retrieved candidate pages using a multimodal Qwen3-VL scorer.
使用多模态Qwen3-VL模型对检索到的候选页面进行重新排序。
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common.io_utils import extract_first_json_object, load_jsonl, save_jsonl
from common.modeling import generate_from_messages, load_qwen_vl_model
from task2_rag.schemas import RERANK_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank retrieved candidate pages with a multimodal Qwen3-VL scorer")
    parser.add_argument("--retrieval_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--top_k", type=int, default=10, help="Only rerank the first top_k retrieved pages")
    parser.add_argument("--keep_k", type=int, default=5, help="Keep this many pages after reranking")
    return parser.parse_args()


def score_candidate(processor: Any, model: Any, question: str, candidate: Dict[str, Any], device: str) -> Dict[str, Any]:
    page_info = candidate.get("structured", {})
    text_hint = candidate.get("native_text", "")
    prompt = (
        f"Question: {question}\n"
        f"Candidate page number: {candidate.get('page_num')}\n"
        f"Page summary: {page_info.get('page_summary', '')}\n"
        f"Native text snippet: {text_hint[:1500]}\n"
        "Return JSON only."
    )
    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": candidate["image_path"]},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    raw = generate_from_messages(
        processor=processor,
        model=model,
        messages=messages,
        device=device,
        max_new_tokens=128,
        temperature=0.0,
    )
    obj = extract_first_json_object(raw) or {}
    candidate = dict(candidate)
    candidate["rerank_raw_output"] = raw
    candidate["rerank_score"] = int(obj.get("relevance", 0)) if str(obj.get("relevance", "")).isdigit() else 0
    candidate["rerank_reason"] = str(obj.get("reason", ""))
    return candidate


if __name__ == "__main__":
    args = parse_args()
    records = load_jsonl(args.retrieval_file)
    processor, model = load_qwen_vl_model(
        model_name_or_path=args.model,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )

    reranked_records: List[Dict[str, Any]] = []
    for record in records:
        candidates = record.get("top_pages", [])[: args.top_k]
        scored = [score_candidate(processor, model, record["question"], c, args.device) for c in candidates]
        scored = sorted(scored, key=lambda x: x.get("rerank_score", 0), reverse=True)
        record = dict(record)
        record["top_pages_reranked"] = scored[: args.keep_k]
        reranked_records.append(record)

    save_jsonl(reranked_records, args.output_file)
    print(f"Saved reranked results to {args.output_file}")
