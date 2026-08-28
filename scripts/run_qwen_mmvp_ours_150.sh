#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 MMVP_IMAGES_DIR [MODEL_ID_OR_PATH] [OUTPUT_DIR]" >&2
  exit 2
fi

REPO="$(repo_root)"
PYTHON="$(resolve_python)"
IMAGES_DIR="$(cd "$1" 2>/dev/null && pwd)" || die "MMVP image directory does not exist: $1"
MODEL_ARG="$(normalize_model_id "${2:-${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}}")"
OUTPUT="$(resolve_output_path "${3:-${OUTPUT_DIR:-$REPO/ours/results/ours_mmvp_forced_choice_150}}")"
MANIFEST="$REPO/shared/mmvp_forced_choice_150_seed_20260828.json"
WORKERS="$(detect_worker_count "$PYTHON")"
begins=(0 38 76 113)
ends=(38 76 113 150)

configure_runtime
mkdir -p "$OUTPUT"
cd "$REPO/ours"
echo "Method: Ours"
echo "Dataset: forced-choice MMVP-150"
echo "MMVP images: $IMAGES_DIR"
echo "Model: $MODEL_ARG"
echo "Output: $OUTPUT"
echo "Concurrent workers: $WORKERS"

for ((wave=0; wave<4; wave+=WORKERS)); do
  pids=()
  for ((slot=0; slot<WORKERS; slot++)); do
    rank=$((wave + slot))
    (( rank < 4 )) || break
    device="$(worker_cuda_device "$slot")"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u mmvp/our_method_mmvp_forced_choice.py \
      --model-id "$MODEL_ARG" --images-dir "$IMAGES_DIR" --eval-list "$MANIFEST" \
      --output-csv "$OUTPUT/rank${rank}.csv" --begin "${begins[$rank]}" \
      --end "${ends[$rank]}" --start-layer 0 --end-layer -1 \
      --neighbor-mode square --baseline white --inject-mode norm_preserve --resume &
    pids+=("$!")
  done
  wait_for_workers "${pids[@]}"
done

"$PYTHON" -u mmvp/validate_mmvp_forced_choice_attribution.py \
  --manifest "$MANIFEST" --csv "$OUTPUT/rank0.csv" "$OUTPUT/rank1.csv" \
  "$OUTPUT/rank2.csv" "$OUTPUT/rank3.csv"
