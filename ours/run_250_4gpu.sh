#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/ours"
MODEL_DIR="/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"
COCO_DIR="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
EVAL_LIST="/mnt/vilab/scratch/arshia/projects/izadi/shared/coco_eval_250_seed_20260815.json"
OUTPUT_DIR="$PROJECT_DIR/results/encoder_250_seed_20260815"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

begins=(0 63 126 188)
ends=(63 126 188 250)
pids=()

for rank in 0 1 2 3; do
  output_csv="$OUTPUT_DIR/rank${rank}.csv"
  if [ -s "$output_csv" ]; then
    echo "Refusing to append duplicate rows to existing output: $output_csv" >&2
    echo "The uploaded method has no resume/skip mechanism." >&2
    exit 2
  fi
done

for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$rank" python -u our_method_insertion_online_encoder.py --model-id "$MODEL_DIR" --coco-root "$COCO_DIR" --eval-list "$EVAL_LIST" --output-csv "$OUTPUT_DIR/rank${rank}.csv" --begin "${begins[$rank]}" --end "${ends[$rank]}" --start-layer 0 --end-layer -1 --canvas-mode full --inject-mode norm_preserve --neighbor-mode square --baseline white &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
