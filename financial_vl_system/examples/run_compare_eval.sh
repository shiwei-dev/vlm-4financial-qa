#!/usr/bin/env bash
set -e
python task1_compare_eval/run_compare_eval.py \
  --data_file /home/wangrui/shiwei/vlm/outputs/tatdqa_test_swift.jsonl \
  --base_model Qwen/Qwen3-VL-4B-Instruct \
  --adapter_path /home/wangrui/shiwei/vlm/outputs/qwen3_vl_4b_tatdqa_lora/v2-20260416-152316/checkpoint-1100 \
  --output_dir /home/wangrui/shiwei/vlm/outputs/task1_compare_results \
  --dtype bf16 \
  --device cuda
