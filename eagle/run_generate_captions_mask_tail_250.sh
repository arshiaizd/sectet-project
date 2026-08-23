#!/usr/bin/env bash

# Generate Qwen2.5-VL-7B captions for the optimized mask-tail subset on 4 GPUs.
# Avoid `set -u`: Conda activation scripts may reference unset variables.
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

INPUT="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_mask_tail_seed_search_100k_all_val/best_subset_250.json"
IMAGE_ROOT="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
MODEL="/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"
OUTPUT="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_mask_tail_seed_search_100k_all_val/qwen25vl7b_captions"

TIMEFORMAT=$'Caption generation wall time: %3lR\nUser CPU time: %3lU\nSystem CPU time: %3lS'

time torchrun \
  --standalone \
  --nproc-per-node=4 \
  generate_qwen_captions_250.py \
  --input "$INPUT" \
  --image-root "$IMAGE_ROOT" \
  --model-id "$MODEL" \
  --output-dir "$OUTPUT" \
  --max-new-tokens 128 \
  --attention-implementation auto
