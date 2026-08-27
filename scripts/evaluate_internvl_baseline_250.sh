#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 {eagle|tam|llavacam|igos_pp} COCO_DIR [MODEL_ID_OR_PATH] [OUTPUT_DIR]" >&2
  exit 2
fi

METHOD="$1"; REPO="$(repo_root)"; PYTHON="$(resolve_python)"
COCO_ROOT="$(normalize_coco_root "$2")"
MODEL_ARG="$(normalize_model_id "${3:-${MODEL_ID:-OpenGVLab/InternVL3_5-8B-HF}}")"
MODEL_TAG="$(printf '%s' "$MODEL_ARG" | sha256sum | cut -c1-12)"
MANIFEST="$REPO/shared/generated/coco_mask_tail_250_internvl35_8b_${MODEL_TAG}.json"
SOURCE_MANIFEST="$REPO/shared/coco_mask_tail_250_benchmark.json"

case "$METHOD" in
  eagle) DEFAULT_OUTPUT="$REPO/eagle/results/internvl35_8b_mask_tail_250/EAGLE" ;;
  tam) DEFAULT_OUTPUT="$REPO/tam/results/internvl35_8b_mask_tail_250/TAM" ;;
  llavacam) DEFAULT_OUTPUT="$REPO/llavacam/results/internvl35_8b_mask_tail_250/LLaVACAM" ;;
  igos_pp) DEFAULT_OUTPUT="$REPO/igos_pp/results/internvl35_8b_mask_tail_250/IGOS_PP" ;;
  *) die "Unknown InternVL baseline: $METHOD" ;;
esac
OUTPUT="$(resolve_output_path "${4:-${OUTPUT_DIR:-$DEFAULT_OUTPUT}}")"
configure_runtime
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$(dirname "$MANIFEST")"
"$PYTHON" -m shared.prepare_internvl35_manifest --input "$SOURCE_MANIFEST" \
  --output "$MANIFEST" --coco-root "$COCO_ROOT" --model-id "$MODEL_ARG"

if [[ "$METHOD" == eagle ]]; then
  EXPLANATION="$OUTPUT/slico-1.0-1.0-division-number-64"
  "$PYTHON" "$REPO/shared/validate_eagle_outputs.py" --eval-list "$MANIFEST" --output-dir "$EXPLANATION"
  "$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" --explanation-dir "$EXPLANATION" 2>&1 | tee "$OUTPUT/evaluation_metrics.txt"
  "$PYTHON" "$REPO/shared/eval_point_game_coco.py" --map-source eagle --pg-criterion centroid \
    --explanation-dir "$EXPLANATION" --coco-root "$COCO_ROOT" \
    --out "$OUTPUT/point_game_per_image.csv" --out-summary "$OUTPUT/point_game_summary.json"
  exit 0
fi

"$PYTHON" "$REPO/shared/validate_baseline_outputs.py" --eval-list "$MANIFEST" --output-dir "$OUTPUT/npy" --kind npy
WORKERS="$(detect_worker_count "$PYTHON")"
SHARDS="$OUTPUT/.evaluation_runtime_shards/${WORKERS}gpu"
prepare_runtime_shards "$PYTHON" "$REPO" "$MANIFEST" "$SHARDS" "$WORKERS"
cd "$REPO"
pids=()
for ((rank=0; rank<WORKERS; rank++)); do
  device="$(worker_cuda_device "$rank")"
  extra=(); [[ "$METHOD" == igos_pp ]] && extra+=(--invert-map)
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u shared/internvl35_8b_faithfulness.py \
    --model-id "$MODEL_ARG" --Datasets "$COCO_ROOT" \
    --eval-list "$SHARDS/rank${rank}-of-${WORKERS}.json" --division-number 64 \
    --eval-dir "$OUTPUT" "${extra[@]}" &
  pids+=("$!")
done
wait_for_workers "${pids[@]}"
"$PYTHON" "$REPO/shared/validate_baseline_outputs.py" --eval-list "$MANIFEST" --output-dir "$OUTPUT/json" --kind json
"$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" --explanation-dir "$OUTPUT" 2>&1 | tee "$OUTPUT/evaluation_metrics.txt"
"$PYTHON" "$REPO/shared/eval_point_game_coco.py" --map-source dense --pg-criterion centroid \
  --explanation-dir "$OUTPUT" --eval-list "$MANIFEST" --coco-root "$COCO_ROOT" \
  --out "$OUTPUT/point_game_per_image.csv" --out-summary "$OUTPUT/point_game_summary.json"
