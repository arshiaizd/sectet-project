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
if [[ -f "$EAGLE_OUTPUT" && "$EAGLE_OUTPUT" == *.zip ]]; then
  EXPLANATION_DIR="$EAGLE_OUTPUT"
elif [[ -d "$DEFAULT_SLICO/json" && -d "$DEFAULT_SLICO/npy" ]]; then
  EXPLANATION_DIR="$DEFAULT_SLICO"
else
  EXPLANATION_DIR="$EAGLE_OUTPUT"
fi
if [[ -n "${2:-}" ]]; then
  OUTPUT="$2"
elif [[ -f "$EAGLE_OUTPUT" ]]; then
  OUTPUT="${EAGLE_OUTPUT%.zip}_evaluation_segmentation_ap_224"
else
  OUTPUT="$EAGLE_OUTPUT/evaluation_segmentation_ap_224"
fi
MANIFEST="${3:-$REPO/shared/coco_mask_tail_250_benchmark.json}"

exec "$SCRIPT_DIR/evaluate_segmentation_ap_coco.sh" \
  eagle "$EXPLANATION_DIR" "$OUTPUT" "$MANIFEST"
