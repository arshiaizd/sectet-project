#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

ROOT="/mnt/vilab/scratch/arshia/projects/izadi"
PROJECT_DIR="$ROOT/ours"
MODEL_DIR="${MODEL_DIR:-/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct}"
IMAGES_DIR="${MMVP_IMAGES_DIR:-/mnt/vilab/scratch/arshia/datasets/MMVP/MMVP Images}"
MANIFEST="${MMVP_MANIFEST:-$ROOT/shared/mmvp_forced_choice_150_seed_20260828.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/ours_mmvp_forced_choice_150}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"
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

begins=(0 38 76 113)
ends=(38 76 113 150)
echo "Forced-choice MMVP-150: four fixed shards, $concurrency concurrent worker(s)."
for (( wave=0; wave<4; wave+=concurrency )); do
  pids=()
  for (( slot=0; slot<concurrency; slot++ )); do
    rank=$(( wave + slot ))
    if (( rank >= 4 )); then
      break
    fi
    echo "rank=$rank gpu=$slot range=[${begins[$rank]},${ends[$rank]})"
    CUDA_VISIBLE_DEVICES="$slot" python -u mmvp/our_method_mmvp_forced_choice.py \
      --model-id "$MODEL_DIR" \
      --images-dir "$IMAGES_DIR" \
      --eval-list "$MANIFEST" \
      --output-csv "$OUTPUT_DIR/rank${rank}.csv" \
      --begin "${begins[$rank]}" \
      --end "${ends[$rank]}" \
      --start-layer 0 \
      --end-layer -1 \
      --neighbor-mode square \
      --baseline white \
      --inject-mode norm_preserve \
      --resume &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "A worker failed. Submit the identical job again to resume." >&2
    exit "$status"
  fi
done

python -u mmvp/validate_mmvp_forced_choice_attribution.py \
  --manifest "$MANIFEST" \
  --csv "$OUTPUT_DIR/rank0.csv" "$OUTPUT_DIR/rank1.csv" \
        "$OUTPUT_DIR/rank2.csv" "$OUTPUT_DIR/rank3.csv"
