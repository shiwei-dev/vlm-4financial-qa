from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from common.io_utils import normalize_answer, normalize_scale

# total：总样本数
# json_ok：JSON 解析成功的样本数
# answer_em：答案完全匹配的样本数
# scale_em：scale 完全匹配的样本数
# joint_em：答案和 scale 都匹配的样本数
def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    json_ok = 0
    answer_em = 0
    scale_em = 0
    joint_em = 0
    by_type = defaultdict(lambda: {"total": 0, "answer_em": 0, "scale_em": 0, "joint_em": 0, "json_ok": 0})

    for row in records:
        total += 1
        gold_answer = normalize_answer(row.get("gold_answer", ""))
        gold_scale = normalize_scale(row.get("gold_scale", ""))
        pred_answer = normalize_answer(row.get("pred_answer", ""))
        pred_scale = normalize_scale(row.get("pred_scale", ""))
        ok = bool(row.get("json_ok", False))

        ans_hit = pred_answer == gold_answer
        scale_hit = pred_scale == gold_scale
        joint_hit = ans_hit and scale_hit

        if ok:
            json_ok += 1
        if ans_hit:
            answer_em += 1
        if scale_hit:
            scale_em += 1
        if joint_hit:
            joint_em += 1

        t = row.get("answer_type", "unknown") or "unknown"
        bucket = by_type[t]
        bucket["total"] += 1
        if ok:
            bucket["json_ok"] += 1
        if ans_hit:
            bucket["answer_em"] += 1
        if scale_hit:
            bucket["scale_em"] += 1
        if joint_hit:
            bucket["joint_em"] += 1

    def ratio(x: int, n: int) -> float:
        return 0.0 if n == 0 else x / n

    metrics = {
        "total": total,
        "json_parse_rate": ratio(json_ok, total),
        "answer_em": ratio(answer_em, total),
        "scale_em": ratio(scale_em, total),
        "joint_em": ratio(joint_em, total),
        "by_answer_type": {},
    }
    for t, bucket in by_type.items():
        n = bucket["total"]
        metrics["by_answer_type"][t] = {
            "total": n,
            "json_parse_rate": ratio(bucket["json_ok"], n),
            "answer_em": ratio(bucket["answer_em"], n),
            "scale_em": ratio(bucket["scale_em"], n),
            "joint_em": ratio(bucket["joint_em"], n),
        }
    return metrics
