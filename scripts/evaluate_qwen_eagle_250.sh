#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 2 || $# -gt 2 ]]; then
  echo "Usage: $0 COCO_DIR EAGLE_OUTPUT_DIR" >&2
  exit 2
fi

REPO="$(repo_root)"
PYTHON="$(resolve_python)"
COCO_ROOT="$(normalize_coco_root "$1")"
OUTPUT="$(resolve_output_path "$2")"
MANIFEST="$REPO/shared/coco_mask_tail_250_benchmark.json"
EXPLANATION="$OUTPUT/slico-1.0-1.0-division-number-64"

validate_benchmark_images "$PYTHON" "$MANIFEST" "$COCO_ROOT"
"$PYTHON" "$REPO/shared/validate_eagle_outputs.py" \
  --eval-list "$MANIFEST" --output-dir "$EXPLANATION"
"$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" \
  --explanation-dir "$EXPLANATION" 2>&1 | tee "$OUTPUT/evaluation_metrics.txt"
"$PYTHON" "$REPO/shared/eval_point_game_coco.py" \
  --map-source eagle --pg-criterion centroid --explanation-dir "$EXPLANATION" \
  --eval-list "$MANIFEST" --coco-root "$COCO_ROOT" \
  --out "$OUTPUT/point_game_per_image.csv" \
  --out-summary "$OUTPUT/point_game_summary.json"
