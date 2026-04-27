#!/usr/bin/env python3
"""
Convert TAT-DQA to Qwen3-VL training JSONL.

Supported output formats:
1) ms-swift standard multimodal format: messages + images
2) qwen-vl-finetune format: image + conversations

Expected dataset layout (common community layout):
  <data_root>/tat_docs/
  <data_root>/tatdqa_dataset_train.json
  <data_root>/tatdqa_dataset_dev.json
  <data_root>/tatdqa_dataset_test.json

Each PDF filename should match doc.uid, e.g. tat_docs/<uid>.pdf.

Example:
  python convert_tat_dqa_to_qwen3vl.py \
      --data-root /path/to/dataset_tatdqa \
      --split train \
      --images-dir /path/to/rendered_pages \
      --output /path/to/tatdqa_train_swift.jsonl \
      --format swift

python convert_tat_dqa_to_qwen3vl.py \
  --data-root dataset_tatdqa/TAT-DQA \
  --split train \
  --images-dir outputs/rendered_pages \
  --output outputs/tatdqa_train_swift.jsonl \
  --format swift \
  --skip-missing-pdf
  
python convert_tat_dqa_to_qwen3vl.py \
  --data-root dataset_tatdqa/TAT-DQA \
  --split dev \
  --images-dir outputs/rendered_pages \
  --output outputs/tatdqa_dev_swift.jsonl \
  --format swift \
  --skip-missing-pdf
  
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    fitz = None
    FITZ_IMPORT_ERROR = exc
else:
    FITZ_IMPORT_ERROR = None


SYSTEM_PROMPT = (
    "You are a financial document QA assistant. "
    "Answer only from the provided document page images. "
    "Return a compact JSON object with keys 'answer' and 'scale'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TAT-DQA to Qwen3-VL JSONL")
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder of TAT-DQA")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        required=True,
        help="Dataset split to convert",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory where rendered page PNGs will be stored",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--format",
        choices=["swift", "qwen_vl_finetune"],
        default="swift",
        help="Output format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=144,
        help="Rendering DPI for PDF pages (default: 144)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only convert first N documents for debugging; 0 means all",
    )
    parser.add_argument(
        "--question-prefix",
        type=str,
        default="",
        help="Optional prefix before each question",
    )
    parser.add_argument(
        "--include-derivation-in-meta",
        action="store_true",
        help="Keep derivation/facts/block_mapping in an extra meta field",
    )
    parser.add_argument(
        "--skip-missing-pdf",
        action="store_true",
        help="Skip samples whose PDF is missing instead of exiting",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Use at most the first N pages from each PDF; 0 means all pages",
    )
    return parser.parse_args()


def normalize_scale(scale: Any) -> str:
    if scale is None:
        return ""
    s = str(scale).strip()
    if s.lower() == "none":
        return ""
    return s


_number_cleanup_re = re.compile(r"\s+")


def normalize_answer(answer: Any) -> str:
    """Turn TAT-DQA answer into a stable string for JSON supervision."""
    if isinstance(answer, list):
        parts = [normalize_answer(x) for x in answer]
        return "; ".join([p for p in parts if p])
    if isinstance(answer, (int, float)):
        if isinstance(answer, float):
            # Avoid scientific notation when unnecessary.
            text = format(answer, ".15g")
            return text
        return str(answer)
    if answer is None:
        return ""
    text = str(answer).strip()
    text = _number_cleanup_re.sub(" ", text)
    return text


def load_split_json(data_root: Path, split: str) -> List[Dict[str, Any]]:
    name_map = {
        "train": "tatdqa_dataset_train.json",
        "dev": "tatdqa_dataset_dev.json",
        "test": "tatdqa_dataset_test.json",
    }
    split_file = data_root / name_map[split]
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {split_file}, got {type(data).__name__}")
    return data


def render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int, max_pages: int = 0) -> List[Path]:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for rendering PDFs. Install it with `pip install pymupdf`. "
            f"Original import error: {FITZ_IMPORT_ERROR}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Path] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        if max_pages > 0:
            page_count = min(page_count, max_pages)
        for page_idx in range(page_count):
            page = doc.load_page(page_idx)
            img_path = out_dir / f"{pdf_path.stem}_page{page_idx + 1}.png"
            if not img_path.exists():
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(img_path.as_posix())
            rendered.append(img_path)
    return rendered


def get_doc_uid(item: Dict[str, Any]) -> str:
    doc = item.get("doc")
    if not isinstance(doc, dict) or "uid" not in doc:
        raise KeyError(f"Unexpected record structure, missing doc.uid: {item.keys()}")
    return str(doc["uid"])


def build_answer_json(answer: Any, scale: Any) -> str:
    payload = {
        "answer": normalize_answer(answer),
        "scale": normalize_scale(scale),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(question: str, num_images: int, prefix: str = "") -> str:
    img_tokens = "<image>" * num_images
    q = question.strip()
    if prefix:
        q = f"{prefix.strip()} {q}".strip()
    return f"{img_tokens}{q}"


def convert_record_to_samples(
    item: Dict[str, Any],
    image_paths: List[Path],
    fmt: str,
    question_prefix: str,
    include_meta: bool,
) -> Iterable[Dict[str, Any]]:
    doc = item["doc"]
    doc_uid = str(doc["uid"])
    questions = item.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError(f"Unexpected questions type for doc {doc_uid}: {type(questions).__name__}")

    images_str = [p.as_posix() for p in image_paths]

    for q in questions:
        question = str(q.get("question", "")).strip()
        if not question:
            continue
        assistant_content = build_answer_json(q.get("answer"), q.get("scale"))
        user_content = build_user_prompt(question, len(images_str), question_prefix)

        sample: Dict[str, Any]
        if fmt == "swift":
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": images_str,
            }
        elif fmt == "qwen_vl_finetune":
            sample = {
                "image": images_str if len(images_str) > 1 else images_str[0],
                "conversations": [
                    {
                        "from": "human",
                        "value": user_content,
                    },
                    {
                        "from": "gpt",
                        "value": assistant_content,
                    },
                ],
            }
        else:  # pragma: no cover
            raise ValueError(f"Unsupported format: {fmt}")

        meta = {
            "sample_id": q.get("uid"),
            "doc_id": doc_uid,
            "source": doc.get("source"),
            "page": doc.get("page"),
            "answer_type": q.get("answer_type"),
            "req_comparison": q.get("req_comparison"),
        }
        if include_meta:
            meta["derivation"] = q.get("derivation")
            meta["facts"] = q.get("facts")
            meta["block_mapping"] = q.get("block_mapping")
        sample["meta"] = meta
        yield sample


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    images_root = args.images_dir.resolve()
    output_path = args.output.resolve()

    items = load_split_json(data_root, args.split)
    if args.limit > 0:
        items = items[: args.limit]

    tat_docs_dir = data_root / "tat_docs" / args.split
    if not tat_docs_dir.exists():
        raise FileNotFoundError(f"tat_docs directory not found: {tat_docs_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    skipped_docs = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for item in items:
            doc_uid = get_doc_uid(item)
            pdf_path = tat_docs_dir / f"{doc_uid}.pdf"
            if not pdf_path.exists():
                msg = f"Missing PDF for doc {doc_uid}: {pdf_path}"
                if args.skip_missing_pdf:
                    skipped_docs += 1
                    print(f"[WARN] {msg}", file=sys.stderr)
                    continue
                raise FileNotFoundError(msg)

            rendered_dir = images_root / args.split
            image_paths = render_pdf_pages(pdf_path, rendered_dir, dpi=args.dpi, max_pages=args.max_pages)
            if not image_paths:
                print(f"[WARN] No pages rendered for {pdf_path}", file=sys.stderr)
                skipped_docs += 1
                continue

            for sample in convert_record_to_samples(
                item=item,
                image_paths=image_paths,
                fmt=args.format,
                question_prefix=args.question_prefix,
                include_meta=args.include_derivation_in_meta,
            ):
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_samples += 1

    print(f"Done. Wrote {total_samples} samples to: {output_path}")
    if skipped_docs:
        print(f"Skipped documents: {skipped_docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
