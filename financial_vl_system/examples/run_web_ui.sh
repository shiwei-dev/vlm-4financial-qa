#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python task2_rag/web_ui.py \
  --work_dir ./.webui_runs \
  --parse_model Qwen/Qwen3-VL-8B-Instruct \
  --embed_model Qwen/Qwen3-VL-Embedding-8B \
  --rerank_model Qwen/Qwen3-VL-8B-Instruct \
  --answer_base_model Qwen/Qwen3-VL-8B-Instruct \
  --answer_adapter_path ./vlm/outputs/qwen3_vl_8b_tatdqa_lora/v2-20260416-152316/checkpoint-1100 \
  --device cuda \
  --dtype bf16 \
  --top_k 5 \
  --server_name 0.0.0.0 \
  --server_port 7860
