#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 {ours|input_level|eagle|tam|llavacam} ATTRIBUTION_PATH [OUTPUT_DIR] [MANIFEST]" >&2
  exit 2
fi

METHOD="$1"
REPO="$(repo_root)"
PYTHON="$(resolve_python)"
ATTRIBUTION_PATH="$(resolve_output_path "$2")"
OUTPUT="$(resolve_output_path "${3:-$ATTRIBUTION_PATH/evaluation_segmentation_ap}")"
MANIFEST="${4:-$REPO/shared/coco_mask_tail_250_benchmark.json}"

[[ -f "$MANIFEST" ]] || die "Manifest does not exist: $MANIFEST"
[[ -d "$ATTRIBUTION_PATH" ]] || die "Attribution path does not exist: $ATTRIBUTION_PATH"
mkdir -p "$OUTPUT"

COMMON=(
  --eval-list "$MANIFEST"
  --resolution 224
  --out "$OUTPUT/segmentation_auprc_per_image.csv"
  --out-summary "$OUTPUT/segmentation_auprc_summary.json"
)

case "$METHOD" in
  ours|input_level)
    mapfile -d '' CSV_FILES < <(
      find "$ATTRIBUTION_PATH" -maxdepth 1 -type f -name '*.csv' -print0 | sort -z
    )
    [[ ${#CSV_FILES[@]} -gt 0 ]] || die "No attribution CSV files found in $ATTRIBUTION_PATH"
    "$PYTHON" "$REPO/shared/eval_segmentation_average_precision_coco.py" \
      --map-source patch --score-column attribution_score \
      --csv "${CSV_FILES[@]}" "${COMMON[@]}"
    ;;
  tam|llavacam)
    "$PYTHON" "$REPO/shared/eval_segmentation_average_precision_coco.py" \
      --map-source dense --explanation-dir "$ATTRIBUTION_PATH" \
      "${COMMON[@]}"
    ;;
  eagle)
    [[ -d "$ATTRIBUTION_PATH/json" && -d "$ATTRIBUTION_PATH/npy" ]] || \
      die "EAGLE attribution path must contain json/ and npy/: $ATTRIBUTION_PATH"
    "$PYTHON" "$REPO/shared/eval_segmentation_average_precision_coco.py" \
      --map-source eagle --explanation-dir "$ATTRIBUTION_PATH" \
      "${COMMON[@]}"
    ;;
  *)
    die "Method must be one of: ours, input_level, eagle, tam, llavacam"
    ;;
esac
