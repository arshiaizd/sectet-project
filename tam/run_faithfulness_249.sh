#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/tam"
SHARED_DIR="/mnt/vilab/scratch/arshia/projects/izadi/shared"
DATASET_DIR="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
OUTPUT_DIR="$PROJECT_DIR/results/qwen25vl7b_coco_object_250_seed_20260815/TAM_249_POSITIONAL"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

run_workers() {
  local status=0
  local pids=()
  for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$rank" python -u qwen25vl7b_faithfulness.py \
      --Datasets "$DATASET_DIR" \
      --eval-list "$SHARED_DIR/eval_shards_249_excluding_324614/coco_target_eval_249_excluding_324614.rank${rank}-of-4.json" \
      --division-number 64 \
      --eval-dir "$OUTPUT_DIR" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  return "$status"
}

TIMEFORMAT=$'TAM 249 positional faithfulness wall time: %3lR'
time run_workers
python "$SHARED_DIR/validate_baseline_outputs.py" \
  --eval-list "$SHARED_DIR/coco_target_eval_249_excluding_324614_seed_20260815.json" \
  --output-dir "$OUTPUT_DIR/json" \
  --kind json
