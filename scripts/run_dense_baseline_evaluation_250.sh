#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 {tam|llavacam} COCO_DIR [MODEL_ID_OR_PATH] [OUTPUT_DIR]" >&2
  exit 2
fi

METHOD="$1"
REPO="$(repo_root)"
PYTHON="$(resolve_python)"
COCO_ROOT="$(normalize_coco_root "$2")"
MODEL_ARG="$(normalize_model_id "${3:-${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}}")"
MANIFEST="$REPO/shared/coco_mask_tail_250_benchmark.json"

case "$METHOD" in
  tam)
    PROJECT_DIR="$REPO/tam"
    DEFAULT_OUTPUT="$REPO/tam/results/mask_tail_250_portable/TAM"
    ;;
  llavacam)
    PROJECT_DIR="$REPO/llavacam"
    DEFAULT_OUTPUT="$REPO/llavacam/results/mask_tail_250_portable/LLaVACAM"
    ;;
  *)
    die "Dense evaluation method must be tam or llavacam: $METHOD"
    ;;
esac

OUTPUT="$(resolve_output_path "${4:-${OUTPUT_DIR:-$DEFAULT_OUTPUT}}")"
WORKERS="$(detect_worker_count "$PYTHON")"
SHARDS="$OUTPUT/.evaluation_runtime_shards/${WORKERS}gpu"
EVALUATOR="$PROJECT_DIR/qwen25vl7b_faithfulness.py"

configure_runtime
validate_benchmark_images "$PYTHON" "$MANIFEST" "$COCO_ROOT"
"$PYTHON" "$REPO/shared/validate_baseline_outputs.py" \
  --eval-list "$MANIFEST" \
  --output-dir "$OUTPUT/npy" \
  --kind npy
prepare_runtime_shards "$PYTHON" "$REPO" "$MANIFEST" "$SHARDS" "$WORKERS"
print_run_configuration "${METHOD} dense faithfulness" "$COCO_ROOT" "$MODEL_ARG" "$OUTPUT" "$WORKERS"

cd "$PROJECT_DIR"
pids=()
for ((rank=0; rank<WORKERS; rank++)); do
  device="$(worker_cuda_device "$rank")"
  CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u "$EVALUATOR" \
    --model-id "$MODEL_ARG" \
    --Datasets "$COCO_ROOT" \
    --eval-list "$SHARDS/rank${rank}-of-${WORKERS}.json" \
    --division-number 64 \
    --eval-dir "$OUTPUT" &
  pids+=("$!")
done
wait_for_workers "${pids[@]}"

"$PYTHON" "$REPO/shared/validate_baseline_outputs.py" \
  --eval-list "$MANIFEST" \
  --output-dir "$OUTPUT/json" \
  --kind json

"$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" \
  --explanation-dir "$OUTPUT" \
  2>&1 | tee "$OUTPUT/evaluation_metrics.txt"

"$PYTHON" "$REPO/shared/eval_point_game_coco.py" \
  --map-source dense \
  --pg-criterion centroid \
  --explanation-dir "$OUTPUT" \
  --eval-list "$MANIFEST" \
  --coco-root "$COCO_ROOT" \
  --out "$OUTPUT/point_game_per_image.csv" \
  --out-summary "$OUTPUT/point_game_summary.json"
