#!/usr/bin/env bash

# Patch-8 insertion/deletion evaluation for our first 30 mask-tail cases.
# Safe to rerun: complete per-image JSON/NPY outputs are skipped.
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/ours"
MODEL_DIR="/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"
COCO_DIR="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
EVAL_LIST="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_mask_tail_30_benchmark.json"
ATTRIBUTION_DIR="$PROJECT_DIR/results/ours_mask_tail_30_old_prompt"
OUTPUT_DIR="$ATTRIBUTION_DIR/evaluation_patch8"
POINT_GAME="$PROJECT_DIR/../shared/eval_point_game_coco.py"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for rank in 0 1 2 3; do
  if [ ! -s "$ATTRIBUTION_DIR/rank${rank}.csv" ]; then
    echo "Missing attribution CSV: $ATTRIBUTION_DIR/rank${rank}.csv" >&2
    exit 2
  fi
done

pids=()
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$rank" python -u evaluate_patch8_faithfulness.py \
    --model-id "$MODEL_DIR" \
    --coco-root "$COCO_DIR" \
    --input-csv "$ATTRIBUTION_DIR/rank${rank}.csv" \
    --output-dir "$OUTPUT_DIR" \
    --patches-per-step 8 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

python validate_patch8_evaluation.py \
  --eval-list "$EVAL_LIST" \
  --output-dir "$OUTPUT_DIR" \
  --patches-per-step 8

python -u "$PROJECT_DIR/../eagle/eval_AUC_faithfulness.py" \
  --explanation-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/evaluation_metrics.txt"

python "$POINT_GAME" \
  --map-source patch \
  --pg-criterion centroid \
  --patch-mode centroid \
  --csv \
    "$ATTRIBUTION_DIR/rank0.csv" \
    "$ATTRIBUTION_DIR/rank1.csv" \
    "$ATTRIBUTION_DIR/rank2.csv" \
    "$ATTRIBUTION_DIR/rank3.csv" \
  --eval-list "$EVAL_LIST" \
  --coco-root "$COCO_DIR" \
  --out "$OUTPUT_DIR/point_game_per_image.csv" \
  --out-summary "$OUTPUT_DIR/point_game_summary.json"
