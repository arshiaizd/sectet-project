#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="${1:-$REPO/reports/segmentation_ap_mask_tail_250_224}"
MANIFEST="${2:-$REPO/shared/coco_mask_tail_250_benchmark.json}"

mkdir -p "$OUTPUT_ROOT"

"$SCRIPT_DIR/evaluate_segmentation_ap_coco.sh" \
  ours "$REPO/ours/results/ours_mask_tail_250_old_prompt" \
  "$OUTPUT_ROOT/ours" "$MANIFEST"

"$SCRIPT_DIR/evaluate_segmentation_ap_coco.sh" \
  input_level "$REPO/input_deletion/results/input_deletion_mask_tail_250" \
  "$OUTPUT_ROOT/input_level" "$MANIFEST"

"$SCRIPT_DIR/evaluate_eagle_segmentation_ap_coco_250.sh" \
  "$REPO/eagle/results/bundles/qwen_coco_mask_tail_250_segmentation_auprc.zip" \
  "$OUTPUT_ROOT/eagle" "$MANIFEST"

"$SCRIPT_DIR/evaluate_segmentation_ap_coco.sh" \
  tam "$REPO/tam/results/mask_tail_250_portable/TAM" \
  "$OUTPUT_ROOT/tam" "$MANIFEST"

"$SCRIPT_DIR/evaluate_segmentation_ap_coco.sh" \
  llavacam "$REPO/llavacam/results/mask_tail_250_portable/LLaVACAM" \
  "$OUTPUT_ROOT/llavacam" "$MANIFEST"

echo "Comparable 224x224 segmentation AUPRC summaries written under: $OUTPUT_ROOT"
