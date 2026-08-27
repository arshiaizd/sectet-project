#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: $0 {eagle|tam|llavacam|igos_pp} COCO_DIR [MODEL_ID_OR_PATH] [NUM_IMAGES] [OUTPUT_DIR]" >&2
  echo "Set RUN_EVALUATION=1 to also smoke-test dense faithfulness evaluation." >&2
  exit 2
fi

METHOD="$1"; REPO="$(repo_root)"; PYTHON="$(resolve_python)"
COCO_ROOT="$(normalize_coco_root "$2")"
MODEL_ARG="$(normalize_model_id "${3:-${MODEL_ID:-OpenGVLab/InternVL3_5-8B-HF}}")"
COUNT="${4:-1}"
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || die "NUM_IMAGES must be a positive integer"
(( COUNT <= 250 )) || die "NUM_IMAGES cannot exceed 250"
SOURCE_MANIFEST="$REPO/shared/coco_mask_tail_250_benchmark.json"

case "$METHOD" in
  eagle)
    PROJECT_DIR="$REPO/eagle"; ENTRYPOINT="internvl35_8b_coco_object.py"
    DEFAULT_OUTPUT="$REPO/eagle/results/internvl35_8b_smoke_$COUNT/EAGLE" ;;
  tam)
    PROJECT_DIR="$REPO/tam"; ENTRYPOINT="internvl35_8b_coco_object_tam.py"
    DEFAULT_OUTPUT="$REPO/tam/results/internvl35_8b_smoke_$COUNT/TAM" ;;
  llavacam)
    PROJECT_DIR="$REPO/llavacam"; ENTRYPOINT="internvl35_8b_coco_object_llavacam.py"
    DEFAULT_OUTPUT="$REPO/llavacam/results/internvl35_8b_smoke_$COUNT/LLaVACAM" ;;
  igos_pp)
    PROJECT_DIR="$REPO/igos_pp"; ENTRYPOINT="internvl35_8b_coco_object_igos_pp.py"
    DEFAULT_OUTPUT="$REPO/igos_pp/results/internvl35_8b_smoke_$COUNT/IGOS_PP" ;;
  *) die "Unknown InternVL baseline: $METHOD" ;;
esac

OUTPUT="$(resolve_output_path "${5:-${OUTPUT_DIR:-$DEFAULT_OUTPUT}}")"
FULL_MANIFEST="$OUTPUT/smoke_manifest_${COUNT}.json"
SUBSET="$FULL_MANIFEST"
configure_runtime
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
validate_benchmark_images "$PYTHON" "$SOURCE_MANIFEST" "$COCO_ROOT"
mkdir -p "$OUTPUT"
"$PYTHON" -m shared.prepare_internvl35_manifest --input "$SOURCE_MANIFEST" \
  --output "$FULL_MANIFEST" --coco-root "$COCO_ROOT" --model-id "$MODEL_ARG" \
  --limit "$COUNT"

WORKERS="$(detect_worker_count "$PYTHON")"
DEVICE="$(worker_cuda_device 0)"
echo "Smoke test: method=$METHOD images=$COUNT GPU=$DEVICE output=$OUTPUT"
cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON" -u "$ENTRYPOINT" \
  --model-id "$MODEL_ARG" --Datasets "$COCO_ROOT" \
  --eval-list "$SUBSET" --save-dir "$OUTPUT"

if [[ "$METHOD" == eagle ]]; then
  EXPLANATION="$OUTPUT/slico-1.0-1.0-division-number-64"
  "$PYTHON" "$REPO/shared/validate_eagle_outputs.py" \
    --eval-list "$SUBSET" --output-dir "$EXPLANATION"
else
  "$PYTHON" "$REPO/shared/validate_baseline_outputs.py" \
    --eval-list "$SUBSET" --output-dir "$OUTPUT/npy" --kind npy
fi

if [[ "${RUN_EVALUATION:-0}" == 1 ]]; then
  if [[ "$METHOD" == eagle ]]; then
    "$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" \
      --explanation-dir "$EXPLANATION" | tee "$OUTPUT/smoke_evaluation_metrics.txt"
  else
    extra=(); [[ "$METHOD" == igos_pp ]] && extra+=(--invert-map)
    cd "$REPO"
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON" -u shared/internvl35_8b_faithfulness.py \
      --model-id "$MODEL_ARG" --Datasets "$COCO_ROOT" --eval-list "$SUBSET" \
      --division-number 64 --eval-dir "$OUTPUT" "${extra[@]}"
    "$PYTHON" "$REPO/shared/validate_baseline_outputs.py" \
      --eval-list "$SUBSET" --output-dir "$OUTPUT/json" --kind json
    "$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" \
      --explanation-dir "$OUTPUT" | tee "$OUTPUT/smoke_evaluation_metrics.txt"
  fi
fi

echo "InternVL smoke test passed: $METHOD ($COUNT image(s))"
