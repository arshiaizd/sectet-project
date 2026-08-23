#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh

SOURCE_ENV=/mnt/vilab/scratch/arshia/conda-envs/main
TARGET_ENV=/mnt/vilab/scratch/arshia/conda-envs/izadi

if [[ ! -x "${TARGET_ENV}/bin/python" ]]; then
    conda create --prefix "${TARGET_ENV}" --clone "${SOURCE_ENV}" --yes
fi

conda activate "${TARGET_ENV}"

# Repair duplicate registrations left by the interrupted clone.
python -m pip uninstall --yes transformers tokenizers safetensors || true
python -m pip uninstall --yes transformers tokenizers safetensors || true

# The source environment contains competing OpenCV wheels. This project needs
# the contrib build because its SLICO/SEEDS implementation uses cv2.ximgproc.
python -m pip uninstall --yes \
    opencv-python \
    opencv-python-headless \
    opencv-contrib-python \
    opencv-contrib-python-headless || true

python -m pip install --no-cache-dir \
    "transformers==4.49.0" \
    "tokenizers==0.21.4" \
    "safetensors==0.4.5" \
    "opencv-contrib-python-headless==4.11.0.86" \
    "qwen-vl-utils==0.0.8" \
    "scipy==1.11.4"

python - <<'PY'
import cv2
import qwen_vl_utils
import scipy
import torch
import torchvision
import transformers

assert hasattr(cv2, "ximgproc"), "OpenCV contrib ximgproc is unavailable"
assert hasattr(
    transformers, "Qwen2_5_VLForConditionalGeneration"
), "Installed Transformers does not support Qwen2.5-VL"

print("Environment verification passed")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("opencv:", cv2.__version__)
print("scipy:", scipy.__version__)
print("environment:", __import__("sys").prefix)
PY
