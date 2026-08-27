#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

ROOT="/mnt/vilab/scratch/arshia/projects/izadi"
PROJECT_DIR="$ROOT/ours"
MODEL_DIR="${MODEL_DIR:-/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct}"
IMAGES_DIR="${MMVP_IMAGES_DIR:-/mnt/vilab/scratch/arshia/datasets/MMVP/MMVP Images}"
ATTR_DIR="${ATTR_DIR:-$PROJECT_DIR/results/ours_mmvp_300}"
EVAL_DIR="${EVAL_DIR:-$PROJECT_DIR/results/ours_mmvp_300_patch8_eval}"

cd "$PROJECT_DIR"
mkdir -p "$EVAL_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

available_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( available_gpus < 1 )); then
  echo "No CUDA GPU is visible inside this job." >&2
  exit 1
fi
concurrency="$available_gpus"
if (( concurrency > 4 )); then
  concurrency=4
fi
begins=(0 75 150 225)
ends=(75 150 225 300)
csvs=("$ATTR_DIR/rank0.csv" "$ATTR_DIR/rank1.csv" "$ATTR_DIR/rank2.csv" "$ATTR_DIR/rank3.csv")

for csv_path in "${csvs[@]}"; do
  test -s "$csv_path" || { echo "Missing attribution file: $csv_path" >&2; exit 1; }
done

echo "MMVP-300 patch-8 evaluation with $concurrency concurrent GPU worker(s)."
for (( wave=0; wave<4; wave+=concurrency )); do
  pids=()
  for (( slot=0; slot<concurrency; slot++ )); do
    rank=$(( wave + slot ))
    if (( rank >= 4 )); then
      break
    fi
    echo "eval-rank=$rank gpu=$slot range=[${begins[$rank]},${ends[$rank]})"
    CUDA_VISIBLE_DEVICES="$slot" python -u mmvp/evaluate_mmvp_patch8.py \
      --model-id "$MODEL_DIR" \
      --images-dir "$IMAGES_DIR" \
      --input-csv "${csvs[@]}" \
      --output-dir "$EVAL_DIR" \
      --begin "${begins[$rank]}" \
      --end "${ends[$rank]}" \
      --patches-per-step 8 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "An evaluation worker failed. Re-submit to resume completed JSONs." >&2
    exit "$status"
  fi
done

json_count="$(find "$EVAL_DIR/json" -maxdepth 1 -name '*.json' -type f | wc -l)"
if (( json_count != 300 )); then
  echo "Expected 300 evaluation JSON files, found $json_count" >&2
  exit 1
fi

python -u "$ROOT/eagle/eval_AUC_faithfulness.py" \
  --explanation-dir "$EVAL_DIR" \
  --sensitiveity 0.2 | tee "$EVAL_DIR/metrics.txt"
