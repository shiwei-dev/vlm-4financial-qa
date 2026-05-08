from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from common.io_utils import (
    build_answer_schema_output,
    ensure_dir,
    load_jsonl,
    save_json,
    save_jsonl,
    safe_get_question,
    swift_style_to_qwen_messages,
)
from common.modeling import generate_from_messages, load_qwen_vl_model
from task1_compare_eval.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base Qwen3-VL vs fine-tuned checkpoint on same samples")
    parser.add_argument("--data_file", type=str, required=True, help="swift-style JSONL test/dev file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True, help="best checkpoint adapter dir, e.g. checkpoint-1100")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def run_model(
    samples: List[Dict[str, Any]],
    processor: Any,
    model: Any,
    model_tag: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        messages = swift_style_to_qwen_messages(sample)
        raw_text = generate_from_messages(
            processor=processor,
            model=model,
            messages=messages[:-1],  # use prompt only, not gold assistant
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        parsed = build_answer_schema_output(raw_text)
        gold_obj = build_answer_schema_output(sample["messages"][-1]["content"])
        outputs.append(
            {
                "sample_idx": idx,
                "model": model_tag,
                "id": sample.get("id") or sample.get("meta", {}).get("sample_id", f"sample_{idx}"),
                "question": safe_get_question(sample),
                "gold_answer": gold_obj["answer"],
                "gold_scale": gold_obj["scale"],
                "pred_answer": parsed["answer"],
                "pred_scale": parsed["scale"],
                "json_ok": parsed["json_ok"],
                "raw_output": parsed["raw_output"],
                "answer_type": sample.get("meta", {}).get("answer_type", "unknown"),
                "req_comparison": sample.get("meta", {}).get("req_comparison", False),
            }
        )
    return outputs


def build_analysis_table(base_rows: List[Dict[str, Any]], ft_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_by_id = {row["id"]: row for row in base_rows}
    ft_by_id = {row["id"]: row for row in ft_rows}
    rows: List[Dict[str, Any]] = []
    for sample_id, ft_row in ft_by_id.items():
        base_row = base_by_id[sample_id]
        base_joint = int(
            base_row["pred_answer"] == base_row["gold_answer"]
            and base_row["pred_scale"] == base_row["gold_scale"]
        )
        ft_joint = int(
            ft_row["pred_answer"] == ft_row["gold_answer"]
            and ft_row["pred_scale"] == ft_row["gold_scale"]
        )
        rows.append(
            {
                "id": sample_id,
                "question": ft_row["question"],
                "gold_answer": ft_row["gold_answer"],
                "gold_scale": ft_row["gold_scale"],
                "base_answer": base_row["pred_answer"],
                "base_scale": base_row["pred_scale"],
                "base_json_ok": base_row["json_ok"],
                "base_joint_ok": base_joint,
                "ft_answer": ft_row["pred_answer"],
                "ft_scale": ft_row["pred_scale"],
                "ft_json_ok": ft_row["json_ok"],
                "ft_joint_ok": ft_joint,
                "improved": int(ft_joint > base_joint),
                "regressed": int(ft_joint < base_joint),
                "same": int(ft_joint == base_joint),
                "answer_type": ft_row["answer_type"],
            }
        )
    return rows


def save_analysis_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    samples = load_jsonl(args.data_file)
    if args.limit > 0:
        samples = samples[: args.limit]

    base_processor, base_model = load_qwen_vl_model(
        model_name_or_path=args.base_model,
        adapter_path=None,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    ft_processor, ft_model = load_qwen_vl_model(
        model_name_or_path=args.base_model,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )

    base_rows = run_model(
        samples,
        base_processor,
        base_model,
        "base_qwen3_vl_8b",
        args.device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    ft_rows = run_model(
        samples,
        ft_processor,
        ft_model,
        "ft_qwen3_vl_8b_checkpoint",
        args.device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    save_jsonl(base_rows, output_dir / "predictions_base.jsonl")
    save_jsonl(ft_rows, output_dir / "predictions_ft.jsonl")

    metrics = {
        "base": compute_metrics(base_rows),
        "ft": compute_metrics(ft_rows),
    }
    save_json(metrics, output_dir / "evaluation_metrics.json")

    analysis_rows = build_analysis_table(base_rows, ft_rows)
    save_analysis_csv(analysis_rows, output_dir / "comparison_analysis.csv")
    save_jsonl(analysis_rows, output_dir / "comparison_analysis.jsonl")

    summary = {
        "base_joint_em": metrics["base"]["joint_em"],
        "ft_joint_em": metrics["ft"]["joint_em"],
        "delta_joint_em": metrics["ft"]["joint_em"] - metrics["base"]["joint_em"],
        "base_answer_f1": metrics["base"]["answer_f1"],
        "ft_answer_f1": metrics["ft"]["answer_f1"],
        "delta_answer_f1": metrics["ft"]["answer_f1"] - metrics["base"]["answer_f1"],
        "base_json_parse_rate": metrics["base"]["json_parse_rate"],
        "ft_json_parse_rate": metrics["ft"]["json_parse_rate"],
        "delta_json_parse_rate": metrics["ft"]["json_parse_rate"] - metrics["base"]["json_parse_rate"],
        "improved_count": sum(r["improved"] for r in analysis_rows),
        "regressed_count": sum(r["regressed"] for r in analysis_rows),
        "same_count": sum(r["same"] for r in analysis_rows),
    }
    save_json(summary, output_dir / "result_analysis_summary.json")
    print(f"Saved comparison results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
