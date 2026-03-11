#!/bin/bash
echo "===== TRAINING VLM ====="

python -m training.train_vlm \
  --model_config configs/model.yaml \
  --train_config configs/training.yaml \
  --output_dir outputs
