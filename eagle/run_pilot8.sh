#!/usr/bin/env bash

# Eight-image, four-GPU EAGLE timing run.
# Do not enable `set -u`: some Conda activation scripts reference unset variables.
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TIMEFORMAT=$'Total wall time: %3lR\nUser CPU time: %3lU\nSystem CPU time: %3lS'

time torchrun \
  --standalone \
  --nproc-per-node=4 \
  Qwen25-VL-7B-coco-object.py \
  --Datasets /mnt/vilab/scratch/arshia/datasets/coco/val2017 \
  --eval-list coco_single_target_once_qwen25vl-7B.json \
  --begin 0 \
  --end 8 \
  --division-number 64 \
  --attention-implementation eager \
  --save-dir results/pilot8_parallel
