#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/ours"
MODEL_DIR="/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"
COCO_DIR="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
EVAL_LIST="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_eval_250_seed_20260815.json"
ATTRIBUTION_DIR="$PROJECT_DIR/results/encoder_250_seed_20260815"
OUTPUT_DIR="$ATTRIBUTION_DIR/evaluation_patch8"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 python -u evaluate_patch8_faithfulness.py --model-id "$MODEL_DIR" --coco-root "$COCO_DIR" --input-csv "$ATTRIBUTION_DIR/rank0.csv" "$ATTRIBUTION_DIR/rank2.csv" --output-dir "$OUTPUT_DIR" --patches-per-step 8 &
pid0="$!"

CUDA_VISIBLE_DEVICES=1 python -u evaluate_patch8_faithfulness.py --model-id "$MODEL_DIR" --coco-root "$COCO_DIR" --input-csv "$ATTRIBUTION_DIR/rank1.csv" "$ATTRIBUTION_DIR/rank3.csv" --output-dir "$OUTPUT_DIR" --patches-per-step 8 &
pid1="$!"

status=0
if ! wait "$pid0"; then status=1; fi
if ! wait "$pid1"; then status=1; fi
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

python validate_patch8_evaluation.py --eval-list "$EVAL_LIST" --output-dir "$OUTPUT_DIR" --patches-per-step 8
