#!/usr/bin/env bash

# Aggregate the completed EAGLE-30 insertion/deletion curves.
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/eagle"
EXPLANATION_DIR="$PROJECT_DIR/results/eagle_mask_tail_30/slico-1.0-1.0-division-number-64"
OUTPUT_LOG="$PROJECT_DIR/results/eagle_mask_tail_30/evaluation_metrics.txt"
POINT_GAME="$PROJECT_DIR/../shared/eval_point_game_coco.py"
POINT_GAME_DIR="$PROJECT_DIR/results/eagle_mask_tail_30/point_game"

cd "$PROJECT_DIR"

if [ "$(find "$EXPLANATION_DIR/json" -maxdepth 1 -type f -name '*.json' | wc -l)" -ne 30 ]; then
  echo "Expected exactly 30 EAGLE JSON outputs before evaluation." >&2
  exit 2
fi

python -u eval_AUC_faithfulness.py \
  --explanation-dir "$EXPLANATION_DIR" \
  2>&1 | tee "$OUTPUT_LOG"

mkdir -p "$POINT_GAME_DIR"
python "$POINT_GAME" \
  --map-source eagle \
  --pg-criterion centroid \
  --explanation-dir "$EXPLANATION_DIR" \
  --coco-root /mnt/vilab/scratch/arshia/datasets/coco/val2017 \
  --out "$POINT_GAME_DIR/point_game_per_image.csv" \
  --out-summary "$POINT_GAME_DIR/point_game_summary.json"

echo "Saved aggregate evaluation log to: $OUTPUT_LOG"
