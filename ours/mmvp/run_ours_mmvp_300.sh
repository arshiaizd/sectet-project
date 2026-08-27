#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/ours"
MODEL_DIR="${MODEL_DIR:-/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct}"
IMAGES_DIR="${MMVP_IMAGES_DIR:-/mnt/vilab/scratch/arshia/datasets/MMVP/MMVP Images}"
MANIFEST="${MMVP_MANIFEST:-$PROJECT_DIR/mmvp/official_eagle/Qwen2.5-VL-7B-MMVP-VQA.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/ours_mmvp_300}"
TOTAL=300

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

available_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( available_gpus < 1 )); then
  echo "No CUDA GPU is visible inside this job." >&2
  exit 1
fi
workers="$available_gpus"
if (( workers > 4 )); then
  workers=4
fi
chunk=$(( (TOTAL + workers - 1) / workers ))

echo "Running MMVP-300 attribution with $workers GPU worker(s)."
echo "Existing complete samples in each rank CSV will be resumed safely."
pids=()
for (( rank=0; rank<workers; rank++ )); do
  begin=$(( rank * chunk ))
  end=$(( begin + chunk ))
  if (( begin >= TOTAL )); then
    break
  fi
  if (( end > TOTAL )); then
    end=$TOTAL
  fi
  echo "rank=$rank gpu=$rank range=[$begin,$end)"
  CUDA_VISIBLE_DEVICES="$rank" python -u mmvp/our_method_mmvp.py \
    --model-id "$MODEL_DIR" \
    --images-dir "$IMAGES_DIR" \
    --eval-list "$MANIFEST" \
    --output-csv "$OUTPUT_DIR/rank${rank}.csv" \
    --begin "$begin" \
    --end "$end" \
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
  echo "At least one MMVP worker failed. Re-submit the same job to resume." >&2
  exit "$status"
fi

mapfile -t csv_files < <(find "$OUTPUT_DIR" -maxdepth 1 -name 'rank*.csv' -type f -print | sort)
python -u mmvp/validate_mmvp_attribution.py \
  --manifest "$MANIFEST" \
  --csv "${csv_files[@]}"
