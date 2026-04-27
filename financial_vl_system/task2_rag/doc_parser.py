"""
从指定目录中的PDF文件中提取每一页的内容，并使用Qwen VL模型将每页内容解析成结构化的JSON格式。对于每一页，脚本会执行以下步骤：
1. 使用PyMuPDF库将PDF页面渲染成图像，并保存到指定的输出目录。
2. 提取页面的原始文本内容。
3. 将页面图像和原始文本输入到Qwen VL模型中，生成结构化的页面信息，包括页面类型、标题、摘要、段落、表格、图 

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from common.io_utils import ensure_dir, extract_first_json_object, save_jsonl
from common.modeling import generate_from_messages, load_qwen_vl_model
from task2_rag.schemas import PARSER_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PDF reports into page-level structured JSONL")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save page images and parsed JSONL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit_files", type=int, default=0)
    parser.add_argument("--native_text_only", action="store_true", help="Skip Qwen parsing and only extract native PDF text")
    return parser.parse_args()


def render_page(page: fitz.Page, out_path: Path, dpi: int) -> None:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(out_path.as_posix())


def parse_page_with_model(
    processor: Any,
    model: Any,
    image_path: str,
    native_text: str,
    page_num: int,
    device: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """
    Use Qwen VL model to parse page content into structured JSON. If parsing fails, return a default structure with raw text.
    """
    user_text = (
        f"Parse this report page into structured JSON. Page number: {page_num}.\n"
        f"Native extracted text (may be incomplete/noisy):\n{native_text[:4000]}"
    )
    messages = [
        {"role": "system", "content": PARSER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    raw = generate_from_messages(
        processor=processor,
        model=model,
        messages=messages,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    parsed = extract_first_json_object(raw)
    if parsed is None:
        parsed = {
            "page_type": "other",
            "page_title": "",
            "page_summary": native_text[:500],
            "paragraphs": [p.strip() for p in native_text.split("\n") if p.strip()][:10],
            "tables": [],
            "charts": [],
            "key_numbers": [],
            "entities": [],
            "raw_output": raw,
            "json_ok": False,
        }
    else:
        parsed["json_ok"] = True
        parsed["raw_output"] = raw
    return parsed


def naive_chunk_page(page_struct: Dict[str, Any], native_text: str) -> List[Dict[str, Any]]:
    """Convert the parsed page structure into a list of text chunks for retrieval. 
    Prioritize structured paragraphs, tables, and charts. If none, fallback to splitting native text.
    """
    chunks: List[Dict[str, Any]] = []
    for para in page_struct.get("paragraphs", []) or []:
        para = str(para).strip()
        if para:
            chunks.append({"chunk_type": "paragraph", "text": para})
    for tbl in page_struct.get("tables", []) or []:
        txt = tbl if isinstance(tbl, str) else str(tbl)
        txt = txt.strip()
        if txt:
            chunks.append({"chunk_type": "table", "text": txt})
    for chart in page_struct.get("charts", []) or []:
        txt = chart if isinstance(chart, str) else str(chart)
        txt = txt.strip()
        if txt:
            chunks.append({"chunk_type": "chart", "text": txt})

    # 如果没有任何结构化内容，退回到原始文本分块，按照空行进行硬切分。
    if not chunks:
        paras = [p.strip() for p in native_text.split("\n\n") if p.strip()]
        for para in paras[:20]:
            chunks.append({"chunk_type": "paragraph", "text": para})
    return chunks


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)
    page_image_dir = ensure_dir(output_dir / "page_images")

    processor = model = None
    if not args.native_text_only:
        processor, model = load_qwen_vl_model(
            model_name_or_path=args.model,
            adapter_path=None,
            dtype=args.dtype,
            device=args.device,
            attn_implementation=args.attn_implementation,
        )

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if args.limit_files > 0:
        pdf_files = pdf_files[: args.limit_files]

    page_records: List[Dict[str, Any]] = []
    for pdf_path in pdf_files:
        doc_id = pdf_path.stem  # 不带扩展名的文件名，作为文档ID
        with fitz.open(pdf_path) as doc:
            for page_idx in range(len(doc)):
                # 使用 PyMuPDF 的 get_text("text") 提取的纯文本
                page = doc.load_page(page_idx)
                page_num = page_idx + 1
                image_path = page_image_dir / f"{doc_id}_page{page_num}.png"
                # 如果图片不存在，才渲染并保存。避免重复渲染已经存在的页面图片。
                if not image_path.exists():
                    render_page(page, image_path, dpi=args.dpi)

                # 提取原始文本和结构化解析并生成页面记录
                native_text = page.get_text("text") or ""
                if args.native_text_only:
                    parsed = {
                        "page_type": "unknown",
                        "page_title": "",
                        "page_summary": native_text[:500],
                        "paragraphs": [p.strip() for p in native_text.split("\n") if p.strip()][:20],
                        "tables": [],
                        "charts": [],
                        "key_numbers": [],
                        "entities": [],
                        "json_ok": False,
                        "raw_output": "",
                    }
                else:
                    parsed = parse_page_with_model(
                        processor=processor,
                        model=model,
                        image_path=str(image_path),
                        native_text=native_text,
                        page_num=page_num,
                        device=args.device,
                        max_new_tokens=args.max_new_tokens,
                    )
                # 构建页面记录，包含文档ID（来自PDF文件名)、页码、PDF路径、图片路径、原始文本 和 模型结构化解析结果
                record = {
                    "doc_id": doc_id, 
                    "page_num": page_num, 
                    "pdf_path": str(pdf_path), 
                    "image_path": str(image_path), 
                    "native_text": native_text, 
                    "structured": parsed, # Qwen解析得到的结构化内容，包括页面类型、标题、摘要、段落、表格、图表、关键数字和实体等
                }
                # 基于模型解析的结构化内容，生成用于检索的文本块列表。优先使用段落、表格和图表等结构化内容，如果没有，则退回到原始文本的分块。
                record["chunks"] = naive_chunk_page(parsed, native_text)
                page_records.append(record)

    save_jsonl(page_records, output_dir / "parsed_pages.jsonl")
    print(f"Saved {len(page_records)} page records to {output_dir / 'parsed_pages.jsonl'}")
