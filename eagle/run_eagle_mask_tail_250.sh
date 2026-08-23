#!/usr/bin/env bash

# EAGLE attribution on the shared, manually audited mask-tail benchmark.
# Safe to rerun: completed per-image JSON/NPY pairs are skipped atomically.
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/eagle"
EVAL_LIST="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_mask_tail_250_benchmark.json"
OUTPUT_DIR="$PROJECT_DIR/results/eagle_mask_tail_250"

cd "$PROJECT_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TIMEFORMAT=$'Total wall time: %3lR\nUser CPU time: %3lU\nSystem CPU time: %3lS'

time torchrun \
  --standalone \
  --nproc-per-node=4 \
  Qwen25-VL-7B-coco-object.py \
  --Datasets /mnt/vilab/scratch/arshia/datasets/coco/val2017 \
  --eval-list "$EVAL_LIST" \
  --begin 0 \
  --end 250 \
  --division-number 64 \
  --attention-implementation auto \
  --save-dir "$OUTPUT_DIR"
