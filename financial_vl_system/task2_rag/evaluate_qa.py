from __future__ import annotations

import argparse
from typing import Any, Dict

from common.io_utils import build_answer_schema_output, load_jsonl, save_json
from task1_compare_eval.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QA outputs against swift-style gold file")
    parser.add_argument("--gold_file", type=str, required=True, help="swift-style JSONL with gold assistant content")
    parser.add_argument("--pred_file", type=str, required=True, help="QA outputs jsonl from answerer.py")
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gold = load_jsonl(args.gold_file)
    pred = load_jsonl(args.pred_file)
    pred_by_id = {p.get("id", f"pred_{i}"): p for i, p in enumerate(pred)}

    rows = []
    for i, sample in enumerate(gold):
        sample_id = sample.get("id") or sample.get("meta", {}).get("sample_id", f"sample_{i}")
        gold_obj = build_answer_schema_output(sample["messages"][-1]["content"])
        p = pred_by_id.get(sample_id, {})
        rows.append(
            {
                "id": sample_id,
                "question": sample.get("question", ""),
                "gold_answer": gold_obj["answer"],
                "gold_scale": gold_obj["scale"],
                "pred_answer": p.get("answer", ""),
                "pred_scale": p.get("scale", ""),
                "json_ok": bool(p.get("answer") is not None),
                "answer_type": sample.get("meta", {}).get("answer_type", "unknown"),
            }
        )

    metrics = compute_metrics(rows)
    save_json(metrics, args.output_file)
    print(f"Saved QA metrics to {args.output_file}")
