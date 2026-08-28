#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 {ours|input_level} COCO_DIR MODEL_ID_OR_PATH ATTRIBUTION_CSV_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

METHOD="$1"
REPO="$(repo_root)"
PYTHON="$(resolve_python)"
COCO_ROOT="$(normalize_coco_root "$2")"
MODEL_ARG="$(normalize_model_id "$3")"
ATTR_DIR="$(resolve_output_path "$4")"
MANIFEST="$REPO/shared/coco_mask_tail_250_benchmark.json"

case "$METHOD" in
  ours)
    PROJECT_DIR="$REPO/ours"
    SCORE_ARGS=()
    ;;
  input_level)
    PROJECT_DIR="$REPO/input_deletion"
    SCORE_ARGS=(--score-column attribution_score)
    ;;
  *) die "Patch method must be ours or input_level: $METHOD" ;;
esac

OUTPUT="$(resolve_output_path "${5:-$ATTR_DIR/evaluation_patch8}")"
WORKERS="$(detect_worker_count "$PYTHON")"
configure_runtime
validate_benchmark_images "$PYTHON" "$MANIFEST" "$COCO_ROOT"
[[ -d "$ATTR_DIR" ]] || die "Attribution CSV directory does not exist: $ATTR_DIR"

mapfile -d '' CSV_FILES < <(find "$ATTR_DIR" -maxdepth 1 -type f -name '*.csv' -print0 | sort -z)
if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  mapfile -d '' CSV_FILES < <(find "$ATTR_DIR" -mindepth 2 -maxdepth 2 -type f -name 'rank*-of-*.csv' -print0 | sort -z)
fi
[[ ${#CSV_FILES[@]} -gt 0 ]] || die "No attribution CSV files found under $ATTR_DIR"

"$PYTHON" "$REPO/shared/validate_patch_csv_outputs.py" \
  --eval-list "$MANIFEST" --csv "${CSV_FILES[@]}"
mkdir -p "$OUTPUT"
print_run_configuration "$METHOD patch-8 evaluation" "$COCO_ROOT" "$MODEL_ARG" "$OUTPUT" "$WORKERS"

cd "$PROJECT_DIR"
for ((wave=0; wave<${#CSV_FILES[@]}; wave+=WORKERS)); do
  pids=()
  for ((slot=0; slot<WORKERS; slot++)); do
    index=$((wave + slot))
    (( index < ${#CSV_FILES[@]} )) || break
    device="$(worker_cuda_device "$slot")"
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u evaluate_patch8_faithfulness.py \
      --model-id "$MODEL_ARG" --coco-root "$COCO_ROOT" \
      --input-csv "${CSV_FILES[$index]}" --output-dir "$OUTPUT" \
      --patches-per-step 8 &
    pids+=("$!")
  done
  wait_for_workers "${pids[@]}"
done

"$PYTHON" validate_patch8_evaluation.py \
  --eval-list "$MANIFEST" --output-dir "$OUTPUT" --patches-per-step 8
"$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" \
  --explanation-dir "$OUTPUT" 2>&1 | tee "$OUTPUT/evaluation_metrics.txt"
"$PYTHON" "$REPO/shared/eval_point_game_coco.py" \
  --map-source patch --pg-criterion centroid --patch-mode centroid \
  "${SCORE_ARGS[@]}" --csv "${CSV_FILES[@]}" --eval-list "$MANIFEST" \
  --coco-root "$COCO_ROOT" --out "$OUTPUT/point_game_per_image.csv" \
  --out-summary "$OUTPUT/point_game_summary.json"
