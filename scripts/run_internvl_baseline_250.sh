#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 {eagle|tam|llavacam|igos_pp} COCO_DIR [MODEL_ID_OR_PATH] [OUTPUT_DIR]" >&2
  exit 2
fi

METHOD="$1"
REPO="$(repo_root)"
PYTHON="$(resolve_python)"
COCO_ROOT="$(normalize_coco_root "$2")"
MODEL_ARG="$(normalize_model_id "${3:-${MODEL_ID:-OpenGVLab/InternVL3_5-8B-HF}}")"
SOURCE_MANIFEST="$REPO/shared/coco_mask_tail_250_benchmark.json"
MODEL_TAG="$(printf '%s' "$MODEL_ARG" | sha256sum | cut -c1-12)"
MANIFEST="$REPO/shared/generated/coco_mask_tail_250_internvl35_8b_${MODEL_TAG}.json"

case "$METHOD" in
  eagle)
    PROJECT_DIR="$REPO/eagle"; ENTRYPOINT="internvl35_8b_coco_object.py"
    DEFAULT_OUTPUT="$REPO/eagle/results/internvl35_8b_mask_tail_250/EAGLE" ;;
  tam)
    PROJECT_DIR="$REPO/tam"; ENTRYPOINT="internvl35_8b_coco_object_tam.py"
    DEFAULT_OUTPUT="$REPO/tam/results/internvl35_8b_mask_tail_250/TAM" ;;
  llavacam)
    PROJECT_DIR="$REPO/llavacam"; ENTRYPOINT="internvl35_8b_coco_object_llavacam.py"
    DEFAULT_OUTPUT="$REPO/llavacam/results/internvl35_8b_mask_tail_250/LLaVACAM" ;;
  igos_pp)
    PROJECT_DIR="$REPO/igos_pp"; ENTRYPOINT="internvl35_8b_coco_object_igos_pp.py"
    DEFAULT_OUTPUT="$REPO/igos_pp/results/internvl35_8b_mask_tail_250/IGOS_PP" ;;
  *) die "Unknown InternVL baseline: $METHOD" ;;
esac

OUTPUT="$(resolve_output_path "${4:-${OUTPUT_DIR:-$DEFAULT_OUTPUT}}")"
WORKERS="$(detect_worker_count "$PYTHON")"
SHARDS="$OUTPUT/.runtime_shards/${WORKERS}gpu"
configure_runtime
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
validate_benchmark_images "$PYTHON" "$SOURCE_MANIFEST" "$COCO_ROOT"
mkdir -p "$(dirname "$MANIFEST")"
"$PYTHON" -m shared.prepare_internvl35_manifest \
  --input "$SOURCE_MANIFEST" --output "$MANIFEST" \
  --coco-root "$COCO_ROOT" --model-id "$MODEL_ARG"
prepare_runtime_shards "$PYTHON" "$REPO" "$MANIFEST" "$SHARDS" "$WORKERS"
print_run_configuration "InternVL3.5 $METHOD" "$COCO_ROOT" "$MODEL_ARG" "$OUTPUT" "$WORKERS"

cd "$PROJECT_DIR"
pids=()
for ((rank=0; rank<WORKERS; rank++)); do
  device="$(worker_cuda_device "$rank")"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u "$ENTRYPOINT" \
    --model-id "$MODEL_ARG" --Datasets "$COCO_ROOT" \
    --eval-list "$SHARDS/rank${rank}-of-${WORKERS}.json" --save-dir "$OUTPUT" &
  pids+=("$!")
done
wait_for_workers "${pids[@]}"

if [[ "$METHOD" == eagle ]]; then
  "$PYTHON" "$REPO/shared/validate_eagle_outputs.py" --eval-list "$MANIFEST" \
    --output-dir "$OUTPUT/slico-1.0-1.0-division-number-64"
else
  "$PYTHON" "$REPO/shared/validate_baseline_outputs.py" --eval-list "$MANIFEST" \
    --output-dir "$OUTPUT/npy" --kind npy
fi
