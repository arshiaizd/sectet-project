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
OUTPUT="${3:-${OUTPUT_DIR:-$REPO/ours/results/ours_mask_tail_250_portable}}"
WORKERS="$(detect_worker_count "$PYTHON")"
RUN_DIR="$OUTPUT/${WORKERS}gpu"

configure_runtime
validate_benchmark_images "$PYTHON" "$MANIFEST" "$COCO_ROOT"
mkdir -p "$RUN_DIR"
print_run_configuration "Ours" "$COCO_ROOT" "$MODEL_ARG" "$RUN_DIR" "$WORKERS"

cd "$REPO/ours"
pids=()
csv_files=()
for ((rank=0; rank<WORKERS; rank++)); do
  begin=$((rank * 250 / WORKERS))
  end=$(((rank + 1) * 250 / WORKERS))
  device="$(worker_cuda_device "$rank")"
  output_csv="$RUN_DIR/rank${rank}-of-${WORKERS}.csv"
  csv_files+=("$output_csv")
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u our_method_insertion_online_encoder.py \
    --model-id "$MODEL_ARG" \
    --coco-root "$COCO_ROOT" \
    --eval-list "$MANIFEST" \
    --output-csv "$output_csv" \
    --begin "$begin" \
    --end "$end" \
    --start-layer 0 \
    --end-layer -1 \
    --canvas-mode full \
    --inject-mode norm_preserve \
    --neighbor-mode square \
    --baseline white \
    --deletion-mode none \
    --resume &
  pids+=("$!")
done
wait_for_workers "${pids[@]}"

"$PYTHON" "$REPO/shared/validate_patch_csv_outputs.py" \
  --eval-list "$MANIFEST" \
  --csv "${csv_files[@]}"
