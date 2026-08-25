#!/usr/bin/env bash

# Our attribution method on the shared, manually audited mask-tail benchmark.
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/ours"
MODEL_DIR="/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"
COCO_DIR="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
EVAL_LIST="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_mask_tail_250_benchmark.json"
OUTPUT_DIR="$PROJECT_DIR/results/ours_mask_tail_250_old_prompt"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

begins=(0 63 126 188)
ends=(63 126 188 250)
pids=()

for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$rank" python -u our_method_insertion_online_encoder.py \
    --model-id "$MODEL_DIR" \
    --coco-root "$COCO_DIR" \
    --eval-list "$EVAL_LIST" \
    --output-csv "$OUTPUT_DIR/rank${rank}.csv" \
    --question-template "Is there a {object_label} in the image or not? Answer with exactly one word: yes or no." \
    --begin "${begins[$rank]}" \
    --end "${ends[$rank]}" \
    --start-layer 0 \
    --end-layer -1 \
    --canvas-mode full \
    --inject-mode norm_preserve \
    --neighbor-mode square \
    --baseline white \
    --deletion-mode none \
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
