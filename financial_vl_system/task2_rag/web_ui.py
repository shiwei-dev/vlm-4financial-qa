from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import gradio as gr
from PIL import Image, ImageDraw

from common.io_utils import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = PROJECT_ROOT / ".webui_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio WebUI for task2 financial RAG QA")
    parser.add_argument("--work_dir", type=str, default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--parse_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen3-VL-Embedding-8B")
    parser.add_argument("--rerank_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--answer_base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--answer_adapter_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--native_text_only", action="store_true", help="Skip Qwen page parsing for faster debugging")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _uploaded_path(file_obj: Any) -> Path:
    if file_obj is None:
        raise ValueError("请先上传 PDF 财报。")
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    if hasattr(file_obj, "name"):
        return Path(file_obj.name)
    raise ValueError("无法识别上传文件路径。")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _run_step(cmd: List[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + proc.stdout)
    return proc.stdout


def _write_question_file(question: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "webui_q0", "question": question}, ensure_ascii=False) + "\n")


def _load_first_jsonl(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    if not rows:
        raise RuntimeError(f"没有读到输出文件: {path}")
    return rows[0]


def _collect_bbox_blocks(page_record: Dict[str, Any]) -> List[Tuple[str, Sequence[float]]]:
    """Collect optional bbox blocks if parser output contains them.

    Current parser chunks are text-only; this function supports future enhanced
    parser outputs such as structured.layout_blocks[*].bbox or chunks[*].bbox.
    """
    blocks: List[Tuple[str, Sequence[float]]] = []
    structured = page_record.get("structured", {}) or {}
    for container in (structured.get("layout_blocks", []) or [], page_record.get("chunks", []) or []):
        if isinstance(container, dict) and isinstance(container.get("bbox"), (list, tuple)):
            label = str(container.get("block_type") or container.get("chunk_type") or "evidence")
            blocks.append((label, container["bbox"]))
    return blocks


def _draw_evidence_overlay(image_path: Path, page_record: Optional[Dict[str, Any]]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Always emphasize the cited page. If bbox exists, also emphasize the block.
    draw.rectangle([4, 4, w - 5, h - 5], outline=(255, 0, 0), width=max(4, w // 250))

    if page_record:
        for label, bbox in _collect_bbox_blocks(page_record):
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(x) for x in bbox]
            # Accept normalized [0, 1] or pixel coordinates.
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(3, w // 350))
            draw.text((x1 + 4, y1 + 4), label[:30], fill=(255, 0, 0))
    return img


def _build_gallery(answer: Dict[str, Any], work_dir: Path) -> List[Tuple[Image.Image, str]]:
    parsed_file = work_dir / "parsed" / "parsed_pages.jsonl"
    pages = load_jsonl(parsed_file) if parsed_file.exists() else []
    by_num = {int(p.get("page_num", -1)): p for p in pages if "page_num" in p}
    source_pages = answer.get("source_pages") or answer.get("retrieved_pages") or []
    gallery: List[Tuple[Image.Image, str]] = []
    for page_num in source_pages[:6]:
        try:
            page_num = int(page_num)
        except Exception:
            continue
        page_record = by_num.get(page_num)
        if not page_record:
            continue
        image_path = Path(page_record.get("image_path", ""))
        if not image_path.exists():
            continue
        summary = (page_record.get("structured", {}) or {}).get("page_summary", "")
        gallery.append((_draw_evidence_overlay(image_path, page_record), f"Page {page_num}: {summary}"))
    return gallery


def _json_tree(answer: Dict[str, Any], retrieval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer_json": {
            "answer": answer.get("answer", ""),
            "scale": answer.get("scale", ""),
            "source_pages": answer.get("source_pages", []),
            "abstain": answer.get("abstain", False),
            "confidence": answer.get("confidence", "low"),
            "json_ok": answer.get("json_ok", False),
            "schema_ok": answer.get("schema_ok", False),
            "schema_errors": answer.get("schema_errors", []),
            "parsed_output": answer.get("parsed_output", {}),
        },
        "retrieval": {
            "retrieved_pages": [p.get("page_num") for p in retrieval.get("top_pages", [])],
            "reranked_pages": [p.get("page_num") for p in retrieval.get("top_pages_reranked", [])],
        },
        "raw_output": answer.get("raw_output", ""),
    }


def run_financial_qa(
    pdf_file: Any,
    question: str,
    parse_model: str,
    embed_model: str,
    rerank_model: str,
    answer_base_model: str,
    answer_adapter_path: str,
    device: str,
    dtype: str,
    top_k: int,
    native_text_only: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, Dict[str, Any], List[Tuple[Image.Image, str]], str]:
    try:
        if not question or not question.strip():
            raise ValueError("请输入自然语言问题。")
        src_pdf = _uploaded_path(pdf_file)
        if src_pdf.suffix.lower() != ".pdf":
            raise ValueError("请上传 PDF 文件。")

        pdf_hash = _file_sha256(src_pdf)
        q_hash = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:8]
        work_dir = Path(run_financial_qa.work_dir) / f"{pdf_hash}_{q_hash}"
        reports_dir = work_dir / "reports"
        parsed_dir = work_dir / "parsed"
        index_dir = work_dir / "index"
        retrieval_dir = work_dir / "retrieval"
        answers_dir = work_dir / "answers"
        for d in [reports_dir, retrieval_dir, answers_dir]:
            d.mkdir(parents=True, exist_ok=True)

        dst_pdf = reports_dir / src_pdf.name
        if not dst_pdf.exists():
            shutil.copy2(src_pdf, dst_pdf)
        questions_file = work_dir / "questions.jsonl"
        _write_question_file(question.strip(), questions_file)

        logs: List[str] = []
        py = sys.executable
        progress(0.05, desc="解析 PDF 页面")
        doc_cmd = [
            py,
            "task2_rag/doc_parser.py",
            "--input_dir",
            str(reports_dir),
            "--output_dir",
            str(parsed_dir),
            "--model",
            parse_model,
            "--device",
            device,
            "--dtype",
            dtype,
            "--attn_implementation",
            run_financial_qa.attn_implementation,
        ]
        if native_text_only:
            doc_cmd.append("--native_text_only")
        logs.append(_run_step(doc_cmd, PROJECT_ROOT))

        progress(0.30, desc="构建页面向量索引")
        logs.append(
            _run_step(
                [
                    py,
                    "task2_rag/build_index.py",
                    "--parsed_pages",
                    str(parsed_dir / "parsed_pages.jsonl"),
                    "--output_dir",
                    str(index_dir),
                    "--embedding_model",
                    embed_model,
                    "--device",
                    device,
                ],
                PROJECT_ROOT,
            )
        )

        progress(0.50, desc="Dense Retrieval")
        retrieved_file = retrieval_dir / "retrieved_topk.jsonl"
        logs.append(
            _run_step(
                [
                    py,
                    "task2_rag/dense_retriever.py",
                    "--index_dir",
                    str(index_dir),
                    "--questions_file",
                    str(questions_file),
                    "--output_file",
                    str(retrieved_file),
                    "--embedding_model",
                    embed_model,
                    "--top_k",
                    str(max(top_k, 10)),
                    "--device",
                    device,
                ],
                PROJECT_ROOT,
            )
        )

        progress(0.68, desc="Rerank 证据页")
        reranked_file = retrieval_dir / "reranked_topk.jsonl"
        logs.append(
            _run_step(
                [
                    py,
                    "task2_rag/reranker.py",
                    "--retrieval_file",
                    str(retrieved_file),
                    "--output_file",
                    str(reranked_file),
                    "--model",
                    rerank_model,
                    "--device",
                    device,
                    "--dtype",
                    dtype,
                    "--top_k",
                    str(max(top_k, 10)),
                    "--keep_k",
                    str(top_k),
                ],
                PROJECT_ROOT,
            )
        )

        progress(0.84, desc="生成答案 JSON")
        qa_file = answers_dir / "qa_outputs.jsonl"
        answer_cmd = [
            py,
            "task2_rag/answerer.py",
            "--retrieval_file",
            str(reranked_file),
            "--output_file",
            str(qa_file),
            "--base_model",
            answer_base_model,
            "--device",
            device,
            "--dtype",
            dtype,
            "--top_k",
            str(top_k),
        ]
        if answer_adapter_path.strip():
            answer_cmd.extend(["--adapter_path", answer_adapter_path.strip()])
        logs.append(_run_step(answer_cmd, PROJECT_ROOT))

        answer = _load_first_jsonl(qa_file)
        retrieval = _load_first_jsonl(reranked_file)
        result_md = (
            f"### 答案\n\n{answer.get('answer', '') or '（模型选择拒答/未找到可支持答案）'}\n\n"
            f"**Scale**: `{answer.get('scale', '')}`  \n"
            f"**Confidence**: `{answer.get('confidence', 'low')}`  \n"
            f"**Source pages**: `{answer.get('source_pages', [])}`  \n"
            f"**JSON OK / Schema OK**: `{answer.get('json_ok')}` / `{answer.get('schema_ok')}`\n"
        )
        return result_md, _json_tree(answer, retrieval), _build_gallery(answer, work_dir), "\n".join(logs)[-12000:]
    except Exception as exc:
        return (
            f"### 运行失败\n\n`{exc}`",
            {"error": str(exc), "traceback": traceback.format_exc()},
            [],
            traceback.format_exc(),
        )


def build_demo(args: argparse.Namespace) -> gr.Blocks:
    run_financial_qa.work_dir = args.work_dir
    run_financial_qa.attn_implementation = args.attn_implementation

    with gr.Blocks(title="Financial VL RAG WebUI") as demo:
        gr.Markdown(
            "# 基于 Qwen3-VL 的金融财报多模态问答 WebUI\n"
            "上传 PDF 财报并输入自然语言问题，系统会执行解析、检索、重排与证据约束生成；"
            "右侧展示答案、完整 JSON 解析树，以及带红框标注的溯源页面。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(label="上传 PDF 财报", file_types=[".pdf"])
                question = gr.Textbox(label="问题", lines=3, placeholder="例如：2023 年净销售额同比变化是多少？")
                with gr.Accordion("模型与运行参数", open=False):
                    parse_model = gr.Textbox(label="Parse model", value=args.parse_model)
                    embed_model = gr.Textbox(label="Embedding model", value=args.embed_model)
                    rerank_model = gr.Textbox(label="Rerank model", value=args.rerank_model)
                    answer_base_model = gr.Textbox(label="Answer base model", value=args.answer_base_model)
                    answer_adapter_path = gr.Textbox(label="Answer LoRA adapter path", value=args.answer_adapter_path)
                    device = gr.Textbox(label="Device", value=args.device)
                    dtype = gr.Dropdown(label="DType", choices=["bf16", "fp16", "fp32"], value=args.dtype)
                    top_k = gr.Slider(label="Top-K evidence pages", minimum=1, maximum=10, value=args.top_k, step=1)
                    native_text_only = gr.Checkbox(label="Native text only（调试用：跳过页面 VLM 解析）", value=args.native_text_only)
                run_btn = gr.Button("运行问答", variant="primary")
            with gr.Column(scale=2):
                result_md = gr.Markdown(label="推理结果")
                json_view = gr.JSON(label="完整 JSON 解析树")
                gallery = gr.Gallery(label="溯源页面 / 图表区块高亮", columns=2, height="auto")
                logs = gr.Textbox(label="运行日志", lines=10)

        run_btn.click(
            fn=run_financial_qa,
            inputs=[
                pdf_file,
                question,
                parse_model,
                embed_model,
                rerank_model,
                answer_base_model,
                answer_adapter_path,
                device,
                dtype,
                top_k,
                native_text_only,
            ],
            outputs=[result_md, json_view, gallery, logs],
        )
    return demo


if __name__ == "__main__":
    cli_args = parse_args()
    build_demo(cli_args).launch(
        server_name=cli_args.server_name,
        server_port=cli_args.server_port,
        share=cli_args.share,
    )
