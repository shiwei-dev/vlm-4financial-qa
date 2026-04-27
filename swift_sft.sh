export SWANLAB_API_KEY=""
# export SWANLAB_WORKSPACE="你的_swanlab_workspace"   # 可选
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
# 单卡训练
# CUDA_VISIBLE_DEVICES=0 \
# NPROC_PER_NODE=1 \
# 多卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift sft \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset ./vlm/outputs/tatdqa_train_swift.jsonl \
  --val_dataset ./vlm/outputs/tatdqa_dev_swift.jsonl \
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
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 4096 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --output_dir ./vlm/outputs/qwen3_vl_4b_tatdqa_lora \
  --report_to swanlab \
  --swanlab_project tatdqa-qwen3-vl \
  --swanlab_exp_name qwen3-vl-4b-lora-v1 \
  --predict_with_generate false \
  --metric_for_best_model loss \
  --greater_is_better false