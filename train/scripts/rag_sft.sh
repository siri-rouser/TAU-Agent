#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODEL_SEQ_LEN=12800   # max length of visual token(10x10 patch)
export FPS_MAX_FRAMES=100
export VIDEO_MAX_TOKEN_NUM=256 # 16x16 patch, 256 tokens

# RAG Config
export AICITY_BASE_FPS=2.0 # Base FPS for slow-fast sampling (2 FPS)
export AICITY_DENSE_MULT=2.0 # Dense sampling multiplier (2x base FPS = 4 FPS)
export AICITY_EVIDENCE_DROPOUT=0 # Dropout percentage for retrieved evidence

# DeepSpeed configuration
deepspeed=./scripts/zero2.json 

# Model configuration
llm=Qwen/Qwen3-VL-8B-Instruct 

# Training hyperparameters
lr=5e-5
batch_size=2
grad_accum_steps=2  # As we have 2 GPUs, this results in an effective batch size of 2 * 2 * 2 = 8. NOTE: If you change the number of GPUs, keep effective batch size =8 to replicate our work.

# LoRA configuration
lora_r=128
lora_alpha=256
dropout=0.03

# Training entry point
entry_file=qwenvl/train/train_qwen_rag.py

# All 10 tasks (RAG per-video tree: slow-fast sampling + retrieved evidence)
datasets="aicity_rag_mcq,aicity_rag_mcq_openended,aicity_rag_bcq,aicity_rag_bcq_openended,aicity_rag_temporal_localization,aicity_rag_video_summarization,aicity_rag_temporal_description,aicity_rag_scene_description,aicity_rag_causal_linkage,aicity_rag_open_qa,psi_vqa_train_bcq,psi_vqa_train_mcq,psi_vqa_train_open_qa,psi_vqa_train_temporal_localization"
# Output configuration
run_name="qwen3vl-8b-aicity-challenge-rag-sft"
output_dir=/output/aicity_new_rag2_stage1_dp0_last_update1

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --lora_enable True \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${dropout} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels $((256*28*28)) \
    --min_pixels $((24*28*28)) \
    --video_max_pixels $((256*28*28)) \
    --video_min_pixels $((24*28*28)) \
    --video_max_frames 100 \
    --warmup_steps 100 \
    --save_strategy steps \
    --save_steps 1060 \
    --save_total_limit 15 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb \
    "

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
