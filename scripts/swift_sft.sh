#!/bin/bash
# Run from project root: bash scripts/swift_sft.sh
# Loads secrets from .env in project root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: ${ENV_FILE} not found. Copy .env.example to .env and fill in your secrets." >&2
    exit 1
fi

set -a
source "${ENV_FILE}"
set +a

if [ -z "${SWANLAB_API_KEY}" ]; then
    echo "ERROR: SWANLAB_API_KEY not set in ${ENV_FILE}" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export IMAGE_MAX_TOKEN_NUM=1024
# 单卡训练
# export CUDA_VISIBLE_DEVICES=0
# export NPROC_PER_NODE=1
# 多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NPROC_PER_NODE=6

swift sft \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset ./outputs/tatdqa_train_swift.jsonl \
  --val_dataset ./outputs/tatdqa_dev_swift.jsonl \
  --load_from_cache_file false \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --eval_strategy steps \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 3 \
  --load_best_model_at_end true \
  --logging_steps 5 \
  --max_length 4096 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --output_dir ./outputs/qwen3vl_4b_cot_lora \
  --report_to swanlab \
  --swanlab_project tatdqa-qwen3-vl \
  --swanlab_exp_name qwen3-vl-4b-lora-cot-v1 \
  --predict_with_generate false \
  --metric_for_best_model loss \
  --greater_is_better false
