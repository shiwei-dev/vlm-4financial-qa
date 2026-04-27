#!/usr/bin/env python3
"""
Pure Python / Transformers fine-tuning script for Qwen3-VL on local JSONL data.

Designed for datasets in the "ms-swift style" that you already generated, e.g.:
{"messages": [...], "images": ["/abs/path/page1.png", ...]}

This script converts each sample on the fly into the official Qwen3-VL conversational
multimodal format expected by AutoProcessor.apply_chat_template().

Features
--------
- Hugging Face Transformers Trainer (no ms-swift)
- Optional LoRA via PEFT
- Assistant-only loss masking (default)
- Train / eval on local JSONL files
- Works with Qwen3-VL chat-format multimodal inputsWSAZXD

Example
-------
python pure_transformers_qwen3_vl_sft.py \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --train_file ./vlm/outputs/tatdqa_train_swift.jsonl \
  --eval_file ./vlm/outputs/tatdqa_dev_swift.jsonl \
  --output_dir ./vlm/outputs/qwen3_vl_4b_hf_lora \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --max_length 4096 \
  --attn_implementation sdpa \
  --use_lora \
  --bf16 \
  --gradient_checkpointing \
  --logging_steps 5 \
  --eval_steps 100 \
  --save_steps 100

Multi-GPU example:
  torchrun --nproc_per_node=2 pure_transformers_qwen3_vl_sft.py ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import ImageFile
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger("qwen3_vl_hf_sft")
IMAGE_TOKEN_RE = re.compile(r"(?:<image>)+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL with pure Transformers/PEFT")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    # Optimization
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    # Precision / runtime
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )

    # Logging / saving
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", nargs="*", default=None)
    parser.add_argument("--run_name", type=str, default=None)

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Language-side modules to receive LoRA adapters."
    )

    # Behavior
    parser.add_argument(
        "--assistant_only_loss",
        action="store_true",
        default=True,
        help="Mask loss on system/user tokens and train only on the final assistant reply.",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Disable PEFT and train the full model. Overrides --use_lora.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite existing output_dir contents.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a saved checkpoint directory.",
    )
    return parser.parse_args()


class SwiftStyleJsonlDataset(Dataset):
    """Reads the JSONL you already generated for ms-swift style training.

    Expected example:
    {
      "messages": [
         {"role": "system", "content": "..."},
         {"role": "user", "content": "<image><image>Question?"},
         {"role": "assistant", "content": "{\"answer\":\"...\"}"}
      ],
      "images": ["/path/a.png", "/path/b.png"],
      "meta": {...}
    }
    """

    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.samples: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {self.path}: {exc}") from exc
                self.samples.append(item)
        if not self.samples:
            raise ValueError(f"No samples found in {self.path}")
        LOGGER.info("Loaded %d samples from %s", len(self.samples), self.path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]



def _extract_question_and_image_count(user_text: str) -> tuple[str, int]:
    match = IMAGE_TOKEN_RE.match(user_text)
    image_count = 0
    text = user_text
    if match:
        prefix = match.group(0)
        image_count = prefix.count("<image>")
        text = user_text[match.end():]
    return text.strip(), image_count



def _to_qwen3vl_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert local swift-style JSONL sample into official Qwen3-VL chat-format messages.

    Official format wants `content` to be either a string or a list of content items,
    such as {"type": "image", "image": "path/to/file.png"} + {"type": "text", "text": "..."}.
    """
    messages = sample["messages"]
    image_paths = sample.get("images", [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    converted: List[Dict[str, Any]] = []
    used_images = 0

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role != "user":
            converted.append({"role": role, "content": content})
            continue

        if not isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        question_text, image_count = _extract_question_and_image_count(content)
        if image_count > len(image_paths):
            raise ValueError(
                f"Sample has {image_count} <image> tokens but only {len(image_paths)} image paths. "
                f"Meta: {sample.get('meta')}"
            )

        user_content: List[Dict[str, str]] = []
        for i in range(image_count):
            user_content.append({"type": "image", "image": str(image_paths[used_images + i])})
        used_images += image_count

        if question_text:
            user_content.append({"type": "text", "text": question_text})

        converted.append({"role": role, "content": user_content})

    if used_images not in {0, len(image_paths)}:
        LOGGER.warning(
            "Not all images were consumed by <image> placeholders. consumed=%d total=%d meta=%s",
            used_images,
            len(image_paths),
            sample.get("meta"),
        )
    return converted


@dataclass
class Qwen3VLDataCollator:
    processor: Any
    max_length: int
    assistant_only_loss: bool = True

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_messages = [_to_qwen3vl_messages(f) for f in features]

        # Tokenize the full conversation, including the assistant answer.
        batch_inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch_inputs.pop("token_type_ids", None)

        labels = batch_inputs["input_ids"].clone()

        # Always ignore padding in the loss.
        if "attention_mask" in batch_inputs:
            labels[batch_inputs["attention_mask"] == 0] = -100

        if self.assistant_only_loss:
            # Build prompt-only messages: everything before the final assistant message.
            prompt_only_messages: List[List[Dict[str, Any]]] = []
            for msgs in batch_messages:
                if not msgs or msgs[-1]["role"] != "assistant":
                    raise ValueError("Each training sample must end with an assistant message.")
                prompt_only_messages.append(msgs[:-1])

            prompt_inputs = self.processor.apply_chat_template(
                prompt_only_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            prompt_inputs.pop("token_type_ids", None)
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()

            for row_idx, prompt_len in enumerate(prompt_lens):
                labels[row_idx, : int(prompt_len)] = -100

        batch_inputs["labels"] = labels
        return batch_inputs



def count_trainable_parameters(model: torch.nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
    }



def maybe_wrap_with_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.full_finetune:
        LOGGER.info("Running full fine-tuning (LoRA disabled).")
        return model
    if not args.use_lora:
        LOGGER.info("LoRA disabled. Running full fine-tuning.")
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model



def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    set_seed(args.seed)

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16")

    train_dataset = SwiftStyleJsonlDataset(args.train_file)
    eval_dataset = SwiftStyleJsonlDataset(args.eval_file) if args.eval_file else None

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    # Right padding is standard for training causal LM losses.
    processor.tokenizer.padding_side = "right"

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    LOGGER.info("Loading model: %s", args.model_name_or_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False

    model = maybe_wrap_with_lora(model, args)

    param_stats = count_trainable_parameters(model)
    LOGGER.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        f"{param_stats['trainable']:,}",
        f"{param_stats['total']:,}",
        100.0 * param_stats["trainable"] / max(param_stats["total"], 1),
    )

    data_collator = Qwen3VLDataCollator(
        processor=processor,
        max_length=args.max_length,
        assistant_only_loss=args.assistant_only_loss,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=True,
        do_eval=eval_dataset is not None,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=args.report_to if args.report_to is not None else [],
        run_name=args.run_name,
        greater_is_better=False,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        load_best_model_at_end=eval_dataset is not None,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    LOGGER.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if eval_dataset is not None:
        LOGGER.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    LOGGER.info("Done. Model and processor saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
