"""
加载一个 Qwen-VL 多模态模型（图文模型），并基于聊天消息 messages 生成回复文本。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def _resolve_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def load_qwen_vl_model(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    dtype: str = "bf16",
    device: str = "cuda",
    attn_implementation: str = "sdpa",
) -> Tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=_resolve_dtype(dtype),
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    if device.startswith("cuda") and torch.cuda.is_available():
        model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def generate_from_messages(
    processor: Any,
    model: Any,
    messages: List[Dict[str, Any]],
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs.pop("token_type_ids", None)
    if device.startswith("cuda") and torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {k: v for k, v in inputs.items()}

    do_sample = temperature > 0
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()
