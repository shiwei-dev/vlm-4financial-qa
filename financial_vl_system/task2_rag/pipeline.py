from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end retrieval + QA pipeline for financial reports")
    parser.add_argument("--reports_dir", type=str, required=True, help="Directory of PDF reports")
    parser.add_argument("--questions_file", type=str, required=True, help="JSONL of questions or swift-style samples")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--parse_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen3-VL-Embedding-8B")
    parser.add_argument("--rerank_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--answer_base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--answer_adapter_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--retrieval_top_k", type=int, default=10)
    parser.add_argument("--rerank_keep_k", type=int, default=5)
    parser.add_argument("--answer_top_k", type=int, default=5)
    parser.add_argument("--native_text_only", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    args = parse_args()
    work_dir = ensure_dir(args.work_dir)
    parsed_dir = ensure_dir(work_dir / "parsed")
    index_dir = ensure_dir(work_dir / "index")
    retrieval_dir = ensure_dir(work_dir / "retrieval")
    answers_dir = ensure_dir(work_dir / "answers")

    parsed_pages = parsed_dir / "parsed_pages.jsonl"
    raw_retrieval = retrieval_dir / "retrieved_topk.jsonl"
    reranked_retrieval = retrieval_dir / "reranked_topk.jsonl"
    final_answers = answers_dir / "qa_outputs.jsonl"

    if not parsed_pages.exists():
        cmd = [
            "python",
            "task2_rag/doc_parser.py",
            "--input_dir", args.reports_dir,
            "--output_dir", str(parsed_dir),
            "--model", args.parse_model,
            "--device", args.device,
            "--dtype", args.dtype,
            "--attn_implementation", args.attn_implementation,
        ]
        if args.native_text_only:
            cmd.append("--native_text_only")
        run(cmd)

    if not (index_dir / "page_embeddings.npy").exists():
        run([
            "python",
            "task2_rag/build_index.py",
            "--parsed_pages", str(parsed_pages),
            "--output_dir", str(index_dir),
            "--embedding_model", args.embed_model,
        ])

    run([
        "python",
        "task2_rag/dense_retriever.py",
        "--index_dir", str(index_dir),
        "--questions_file", args.questions_file,
        "--output_file", str(raw_retrieval),
        "--embedding_model", args.embed_model,
        "--top_k", str(args.retrieval_top_k),
    ])

    run([
        "python",
        "task2_rag/reranker.py",
        "--retrieval_file", str(raw_retrieval),
        "--output_file", str(reranked_retrieval),
        "--model", args.rerank_model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn_implementation", args.attn_implementation,
        "--top_k", str(args.retrieval_top_k),
        "--keep_k", str(args.rerank_keep_k),
    ])

    answer_cmd = [
        "python",
        "task2_rag/answerer.py",
        "--retrieval_file", str(reranked_retrieval),
        "--output_file", str(final_answers),
        "--base_model", args.answer_base_model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn_implementation", args.attn_implementation,
        "--top_k", str(args.answer_top_k),
    ]
    if args.answer_adapter_path:
        answer_cmd.extend(["--adapter_path", args.answer_adapter_path])
    run(answer_cmd)

    print(f"Pipeline complete. Final answers: {final_answers}")
