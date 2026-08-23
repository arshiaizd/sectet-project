#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 COCO_DIR [MODEL_ID_OR_PATH] [OUTPUT_DIR]" >&2
  exit 2
fi

REPO="$(repo_root)"
PYTHON="$(resolve_python)"
COCO_ROOT="$(normalize_coco_root "$1")"
MODEL_ARG="${2:-${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}}"
MANIFEST="$REPO/shared/coco_mask_tail_250_benchmark.json"
OUTPUT="${3:-${OUTPUT_DIR:-$REPO/llavacam/results/mask_tail_250_portable/LLaVACAM}}"
WORKERS="$(detect_worker_count "$PYTHON")"
SHARDS="$OUTPUT/.runtime_shards/${WORKERS}gpu"

configure_runtime
validate_benchmark_images "$PYTHON" "$MANIFEST" "$COCO_ROOT"
prepare_runtime_shards "$PYTHON" "$REPO" "$MANIFEST" "$SHARDS" "$WORKERS"
print_run_configuration "LLaVA-CAM" "$COCO_ROOT" "$MODEL_ARG" "$OUTPUT" "$WORKERS"

cd "$REPO/llavacam"
pids=()
for ((rank=0; rank<WORKERS; rank++)); do
  device="$(worker_cuda_device "$rank")"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u qwen25vl7b_coco_object_llavacam.py \
    --model-id "$MODEL_ARG" \
    --Datasets "$COCO_ROOT" \
    --eval-list "$SHARDS/rank${rank}-of-${WORKERS}.json" \
    --save-dir "$OUTPUT" &
  pids+=("$!")
done
wait_for_workers "${pids[@]}"

"$PYTHON" "$REPO/shared/validate_baseline_outputs.py" \
  --eval-list "$MANIFEST" \
  --output-dir "$OUTPUT/npy" \
  --kind npy
