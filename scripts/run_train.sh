#!/bin/bash
OMP_NUM_THREADS=1 \
torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  training/train_vlm.py \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True
