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
MODEL_ARG="$(normalize_model_id "${2:-${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}}")"
MANIFEST="$REPO/shared/coco_mask_tail_250_benchmark.json"
OUTPUT="$(resolve_output_path "${3:-${OUTPUT_DIR:-$REPO/eagle/results/eagle_mask_tail_250_portable}}")"
WORKERS="$(detect_worker_count "$PYTHON")"

configure_runtime
validate_benchmark_images "$PYTHON" "$MANIFEST" "$COCO_ROOT"
print_run_configuration "EAGLE" "$COCO_ROOT" "$MODEL_ARG" "$OUTPUT" "$WORKERS"

cd "$REPO/eagle"
TIMEFORMAT=$'EAGLE wall time: %3lR'
time "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$WORKERS" \
  Qwen25-VL-7B-coco-object.py \
  --model-id "$MODEL_ARG" \
  --Datasets "$COCO_ROOT" \
  --eval-list "$MANIFEST" \
  --begin 0 \
  --end 250 \
  --division-number 64 \
  --attention-implementation auto \
  --save-dir "$OUTPUT"

"$PYTHON" "$REPO/shared/validate_eagle_outputs.py" \
  --eval-list "$MANIFEST" \
  --output-dir "$OUTPUT/slico-1.0-1.0-division-number-64"
