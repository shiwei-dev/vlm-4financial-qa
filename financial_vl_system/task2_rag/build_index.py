"""
读取已解析的页面记录（JSONL 格式），为每一页生成一个密集向量（embedding），
并保存为 NumPy 数组，同时保留页面元数据。生成的索引可用于后续的语义检索（RAG）系统。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from common.io_utils import ensure_dir, load_jsonl, save_json, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense page index from parsed page records")
    parser.add_argument("--parsed_pages", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-VL-Embedding-4B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--text_only", action="store_true", help="Embed only text, ignore page images")
    return parser.parse_args()


def build_document_text(page: Dict[str, Any]) -> str:
    """将页面的结构化信息和原始文本内容组合成一个文本块，用于生成文本嵌入。"""
    s = page.get("structured", {})
    chunks = page.get("chunks", [])
    chunk_text = "\n".join(c.get("text", "") for c in chunks[:20])
    parts = [
        f"doc_id: {page.get('doc_id', '')}",
        f"page_num: {page.get('page_num', '')}",
        f"page_type: {s.get('page_type', '')}",
        f"page_title: {s.get('page_title', '')}",
        f"page_summary: {s.get('page_summary', '')}",
        f"native_text: {page.get('native_text', '')[:3000]}",
        f"chunks: {chunk_text[:3000]}",
    ]
    return "\n".join(p for p in parts if p.strip())


if __name__ == "__main__":
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    pages = load_jsonl(args.parsed_pages)

    texts = [build_document_text(p) for p in pages]
    image_paths = [p.get("image_path", None) for p in pages]

    model_kwargs = {"trust_remote_code": True}
    if args.device:
        model_kwargs["device"] = args.device
    embedder = SentenceTransformer(args.embedding_model, **model_kwargs)

    if args.text_only:
        embeddings = embedder.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    else:
        embeddings = embedder.encode(
            texts,
            images=image_paths,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    np.save(output_dir / "page_embeddings.npy", embeddings)
    save_jsonl(pages, output_dir / "page_metadata.jsonl") # 页面元数据（副本）
    save_json(
        {
            "embedding_model": args.embedding_model,
            "num_pages": len(pages),
            "embedding_dim": int(embeddings.shape[1]),
            "text_only": args.text_only,
        },
        output_dir / "index_meta.json",
    ) # 索引元信息，记录使用的模型、页面数量、嵌入维度等信息，便于后续使用和维护。
    print(f"Saved dense index to {output_dir}")
