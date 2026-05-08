from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Dict, Iterable, List

from common.io_utils import normalize_answer, normalize_scale


def _ratio(x: float, n: int) -> float:
    return 0.0 if n == 0 else float(x) / n


def _answer_tokens(text: Any) -> List[str]:
    """Tokenize answer text for a soft F1 diagnostic metric.

    EM/joint EM should remain the primary metric for financial numeric answers;
    this token F1 is useful for partial-credit analysis on list/span answers.
    """
    text = normalize_answer(text).lower()
    # Keep decimals, signs, percent signs and common word tokens.
    return re.findall(r"[-+]?\d+(?:\.\d+)?%?|[a-zA-Z]+|[\u4e00-\u9fff]+", text)


def answer_f1_score(pred_answer: Any, gold_answer: Any) -> float:
    pred_tokens = _answer_tokens(pred_answer)
    gold_tokens = _answer_tokens(gold_answer)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# total：总样本数
# json_ok：JSON 解析成功的样本数
# answer_em：答案完全匹配的样本数
# answer_f1：答案 token-level F1，提供部分匹配诊断
# scale_em：scale 完全匹配的样本数
# joint_em：答案和 scale 都匹配的样本数
def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    json_ok = 0
    answer_em = 0
    answer_f1_sum = 0.0
    scale_em = 0
    joint_em = 0

    by_type = defaultdict(
        lambda: {
            "total": 0,
            "answer_em": 0,
            "answer_f1_sum": 0.0,
            "scale_em": 0,
            "joint_em": 0,
            "json_ok": 0,
        }
    )

    for row in records:
        total += 1
        gold_answer = normalize_answer(row.get("gold_answer", ""))
        gold_scale = normalize_scale(row.get("gold_scale", ""))
        pred_answer = normalize_answer(row.get("pred_answer", ""))
        pred_scale = normalize_scale(row.get("pred_scale", ""))

        ok = bool(row.get("json_ok", False))
        ans_hit = pred_answer == gold_answer
        f1 = answer_f1_score(pred_answer, gold_answer)
        scale_hit = pred_scale == gold_scale
        joint_hit = ans_hit and scale_hit

        if ok:
            json_ok += 1
        if ans_hit:
            answer_em += 1
        answer_f1_sum += f1
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
        bucket["answer_f1_sum"] += f1
        if scale_hit:
            bucket["scale_em"] += 1
        if joint_hit:
            bucket["joint_em"] += 1

    metrics = {
        "total": total,
        "json_parse_rate": _ratio(json_ok, total),
        "answer_em": _ratio(answer_em, total),
        "answer_f1": _ratio(answer_f1_sum, total),
        "scale_em": _ratio(scale_em, total),
        "joint_em": _ratio(joint_em, total),
        "by_answer_type": {},
    }

    for t, bucket in by_type.items():
        n = bucket["total"]
        metrics["by_answer_type"][t] = {
            "total": n,
            "json_parse_rate": _ratio(bucket["json_ok"], n),
            "answer_em": _ratio(bucket["answer_em"], n),
            "answer_f1": _ratio(bucket["answer_f1_sum"], n),
            "scale_em": _ratio(bucket["scale_em"], n),
            "joint_em": _ratio(bucket["joint_em"], n),
        }
    return metrics
