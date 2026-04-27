#!/usr/bin/env bash
set -e
python task2_rag/pipeline.py \
  --reports_dir /path/to/report_pdfs \
  --questions_file /home/wangrui/shiwei/vlm/outputs/tatdqa_test_swift.jsonl \
  --work_dir /path/to/work_dir \
  --parse_model Qwen/Qwen3-VL-4B-Instruct \
  --embed_model Qwen/Qwen3-VL-Embedding-4B \
  --rerank_model Qwen/Qwen3-VL-4B-Instruct \
  --answer_base_model Qwen/Qwen3-VL-4B-Instruct \
  --answer_adapter_path /home/wangrui/shiwei/vlm/outputs/qwen3_vl_4b_tatdqa_lora/v2-20260416-152316/checkpoint-1100 \
  --device cuda \
  --dtype bf16
