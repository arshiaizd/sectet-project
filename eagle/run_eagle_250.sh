#!/usr/bin/env bash

# Canonical 250-image EAGLE run on four GPUs.
# Avoid `set -u`: Conda activation scripts may reference unset variables.
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
  --eval-list /mnt/vilab/scratch/arshia/projects/izadi/shared/coco_eval_250_seed_20260815.json \
  --begin 0 \
  --end 250 \
  --division-number 64 \
  --attention-implementation eager \
  --save-dir results/eagle_250_seed_20260815
