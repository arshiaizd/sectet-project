#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 EAGLE_OUTPUT_DIR [OUTPUT_DIR] [MANIFEST]" >&2
  exit 2
fi

REPO="$(repo_root)"
EAGLE_OUTPUT="$(resolve_output_path "$1")"
DEFAULT_SLICO="$EAGLE_OUTPUT/slico-1.0-1.0-division-number-64"
if [[ -d "$DEFAULT_SLICO/json" && -d "$DEFAULT_SLICO/npy" ]]; then
  EXPLANATION_DIR="$DEFAULT_SLICO"
else
  EXPLANATION_DIR="$EAGLE_OUTPUT"
fi
OUTPUT="${2:-$EAGLE_OUTPUT/evaluation_segmentation_ap_224}"
MANIFEST="${3:-$REPO/shared/coco_mask_tail_250_benchmark.json}"

exec "$SCRIPT_DIR/evaluate_segmentation_ap_coco.sh" \
  eagle "$EXPLANATION_DIR" "$OUTPUT" "$MANIFEST"
