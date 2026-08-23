#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/ours"
MODEL_DIR="/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"
COCO_DIR="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
EVAL_LIST="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_eval_250_seed_20260815.json"
OUTPUT_CSV="$PROJECT_DIR/results/self_check_1image/sample0.csv"

mkdir -p "$(dirname "$OUTPUT_CSV")"
cd "$PROJECT_DIR"

if [ -s "$OUTPUT_CSV" ]; then
  echo "Refusing to append duplicate rows to existing output: $OUTPUT_CSV" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

python -u our_method_insertion_online_encoder.py --model-id "$MODEL_DIR" --coco-root "$COCO_DIR" --eval-list "$EVAL_LIST" --output-csv "$OUTPUT_CSV" --begin 0 --end 1 --start-layer 0 --end-layer -1 --canvas-mode full --inject-mode norm_preserve --neighbor-mode square --baseline white --self-check
