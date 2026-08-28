#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 MMVP_IMAGES_DIR MODEL_ID_OR_PATH ATTRIBUTION_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

REPO="$(repo_root)"
PYTHON="$(resolve_python)"
IMAGES_DIR="$(cd "$1" 2>/dev/null && pwd)" || die "MMVP image directory does not exist: $1"
MODEL_ARG="$(normalize_model_id "$2")"
ATTR_DIR="$(resolve_output_path "$3")"
OUTPUT="$(resolve_output_path "${4:-$ATTR_DIR/evaluation_patch8}")"
WORKERS="$(detect_worker_count "$PYTHON")"
MANIFEST="$REPO/shared/mmvp_forced_choice_150_seed_20260828.json"
begins=(0 38 76 113)
ends=(38 76 113 150)
csvs=("$ATTR_DIR/rank0.csv" "$ATTR_DIR/rank1.csv" "$ATTR_DIR/rank2.csv" "$ATTR_DIR/rank3.csv")

configure_runtime
for path in "${csvs[@]}"; do [[ -s "$path" ]] || die "Missing attribution file: $path"; done
"$PYTHON" "$REPO/ours/mmvp/validate_mmvp_forced_choice_attribution.py" \
  --manifest "$MANIFEST" --csv "${csvs[@]}"
mkdir -p "$OUTPUT"
cd "$REPO/ours"

for ((wave=0; wave<4; wave+=WORKERS)); do
  pids=()
  for ((slot=0; slot<WORKERS; slot++)); do
    rank=$((wave + slot))
    (( rank < 4 )) || break
    device="$(worker_cuda_device "$slot")"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u mmvp/evaluate_mmvp_patch8.py \
      --model-id "$MODEL_ARG" --images-dir "$IMAGES_DIR" --input-csv "${csvs[@]}" \
      --output-dir "$OUTPUT" --begin "${begins[$rank]}" --end "${ends[$rank]}" \
      --patches-per-step 8 &
    pids+=("$!")
  done
  wait_for_workers "${pids[@]}"
done

json_count="$(find "$OUTPUT/json" -maxdepth 1 -type f -name '*.json' | wc -l)"
[[ "$json_count" -eq 150 ]] || die "Expected 150 evaluation JSON files, found $json_count"
"$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" \
  --explanation-dir "$OUTPUT" --sensitiveity 0.2 2>&1 | tee "$OUTPUT/evaluation_metrics.txt"
