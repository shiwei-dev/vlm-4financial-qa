"""
该脚本实现了一个基于密集向量检索的系统，用于从预先构建的页面嵌入索引中检索与输入问题最相关的页面。主要功能包括：
1. 加载预先构建的页面嵌入索引和页面元数据。
2. 从输入的JSONL文件中提取问题文本。
3. 使用与索引构建时相同的文本嵌入模型将问题转换为向量表示。
4. 计算问题向量与页面嵌入之间的相似度，并返回相似度最高的前K个页面作为检索结果。
5. 将检索结果保存到指定的输出文件中，包含每个问题的检索到的页面信息和相似度分数。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from common.io_utils import load_json, load_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense retrieve top-k pages for each question")
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--questions_file", type=str, required=True, help="jsonl with question or swift-style messages")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-VL-Embedding-4B")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def extract_question(record: Dict[str, Any]) -> str:
    if "question" in record:
        return str(record["question"]).strip()
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            content = str(msg.get("content", ""))
            return content.replace("<image>", "").strip()
    return ""


if __name__ == "__main__":
    args = parse_args()
    index_dir = Path(args.index_dir)
    metadata = load_jsonl(index_dir / "page_metadata.jsonl")
    embeddings = np.load(index_dir / "page_embeddings.npy")
    questions = load_jsonl(args.questions_file)

    model_kwargs = {"trust_remote_code": True}
    if args.device:
        model_kwargs["device"] = args.device
    embedder = SentenceTransformer(args.embedding_model, **model_kwargs)

    queries = [extract_question(q) for q in questions]
    query_embeds = embedder.encode(
        queries,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results: List[Dict[str, Any]] = []
    sims = query_embeds @ embeddings.T
    for q_idx, query in enumerate(queries):
        top_idx = np.argsort(-sims[q_idx])[: args.top_k]
        candidates = []
        for idx in top_idx:
            page = metadata[int(idx)]
            candidates.append(
                {
                    "doc_id": page["doc_id"],
                    "page_num": page["page_num"],
                    "image_path": page["image_path"],
                    "native_text": page.get("native_text", "")[:1000],
                    "score": float(sims[q_idx, idx]),
                    "structured": page.get("structured", {}),
                }
            )
        results.append(
            {
                "id": questions[q_idx].get("id", questions[q_idx].get("meta", {}).get("sample_id", f"q_{q_idx}")),
                "question": query,
                "top_pages": candidates,
                "raw_record": questions[q_idx],
            }
        )

    save_jsonl(results, args.output_file)
    print(f"Saved retrieval results to {args.output_file}")
