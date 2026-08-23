#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/vilab/scratch/arshia/conda-envs/izadi

PROJECT_DIR="/mnt/vilab/scratch/arshia/projects/izadi/tam"
SHARED_DIR="/mnt/vilab/scratch/arshia/projects/izadi/shared"
DATASET_DIR="/mnt/vilab/scratch/arshia/datasets/coco/val2017"
OUTPUT_DIR="$PROJECT_DIR/results/qwen25vl7b_coco_object_250_seed_20260815/TAM"

cd "$PROJECT_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# The missing image is in rank 2's shard. Existing readable maps are skipped,
# so this retries only 000000324614.jpg and preserves canonical outputs.
python -u qwen25vl7b_coco_object_tam.py --Datasets "$DATASET_DIR" --eval-list "$SHARED_DIR/eval_shards/coco_target_eval_250_seed_20260815.rank2-of-4.json" --save-dir "$OUTPUT_DIR"

python "$SHARED_DIR/validate_baseline_outputs.py" --eval-list "$SHARED_DIR/coco_eval_250_seed_20260815.json" --output-dir "$OUTPUT_DIR/npy" --kind npy
