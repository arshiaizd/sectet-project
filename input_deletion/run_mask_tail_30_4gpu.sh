#!/usr/bin/env bash

# Input-space insertion+deletion baseline on the first 30 benchmark cases.
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/input_deletion"
MODEL_DIR="/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"
COCO_DIR="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
EVAL_LIST="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_mask_tail_250_benchmark.json"
OUTPUT_DIR="$PROJECT_DIR/results/input_deletion_mask_tail_30"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

begins=(0 8 16 23)
ends=(8 16 23 30)
pids=()

for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$rank" python -u our_method_insertion_deletion_input_online.py \
    --model-id "$MODEL_DIR" \
    --coco-root "$COCO_DIR" \
    --eval-list "$EVAL_LIST" \
    --output-csv "$OUTPUT_DIR/rank${rank}.csv" \
    --begin "${begins[$rank]}" \
    --end "${ends[$rank]}" \
    --neighbor-radius 1 \
    --score-mode both \
    --combine-mode insertion_plus_necessity \
    --baseline white \
    --resume &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
