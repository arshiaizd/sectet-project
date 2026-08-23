#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Run one or all attribution methods on the audited COCO-250 benchmark.

Required:
  --method NAME           eagle, ours, input_deletion, tam, llavacam,
                          igos_pp, or all (repeatable; comma lists accepted)
  --coco-dir PATH         COCO root containing val2017/, or val2017/ itself

Model:
  --model ID_OR_PATH      Local checkpoint or Hugging Face repository
                          (default: Qwen/Qwen2.5-VL-7B-Instruct)
  --download-model-to DIR Download --model into DIR before running
  --model-revision REV    Hugging Face revision used with --download-model-to
                          (default: main; use a commit hash to pin weights)
  --offline               Require a local model and disable network model access

Runtime:
  --num-gpus N            Maximum workers, 1-4 (default: 4, capped by visibility)
  --python PATH           Python executable from the desired environment
  --output-root PATH      Put method outputs below this directory
  --install-deps          Install requirements.txt into --python environment
  --help                  Show this help

Examples:
  ./scripts/run_benchmark_250.sh \
    --method eagle \
    --coco-dir /datasets/coco \
    --model /models/Qwen2.5-VL-7B-Instruct

  ./scripts/run_benchmark_250.sh \
    --method ours --method eagle \
    --coco-dir /datasets/coco/val2017 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --download-model-to /models/qwen25vl7b \
    --model-revision COMMIT_HASH \
    --num-gpus 1 \
    --output-root /experiments/coco250
EOF
}

methods_raw=()
coco_dir=""
model="Qwen/Qwen2.5-VL-7B-Instruct"
download_model_to=""
model_revision="main"
num_gpus=4
python_bin="${PYTHON_BIN:-python}"
output_root=""
install_deps=0
offline=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      [[ $# -ge 2 ]] || { echo "--method requires a value" >&2; exit 2; }
      methods_raw+=("$2"); shift 2 ;;
    --coco-dir)
      [[ $# -ge 2 ]] || { echo "--coco-dir requires a value" >&2; exit 2; }
      coco_dir="$2"; shift 2 ;;
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 2; }
      model="$2"; shift 2 ;;
    --download-model-to)
      [[ $# -ge 2 ]] || { echo "--download-model-to requires a value" >&2; exit 2; }
      download_model_to="$2"; shift 2 ;;
    --model-revision)
      [[ $# -ge 2 ]] || { echo "--model-revision requires a value" >&2; exit 2; }
      model_revision="$2"; shift 2 ;;
    --num-gpus)
      [[ $# -ge 2 ]] || { echo "--num-gpus requires a value" >&2; exit 2; }
      num_gpus="$2"; shift 2 ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      python_bin="$2"; shift 2 ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "--output-root requires a value" >&2; exit 2; }
      output_root="$2"; shift 2 ;;
    --install-deps)
      install_deps=1; shift ;;
    --offline)
      offline=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ ${#methods_raw[@]} -gt 0 ]] || { echo "At least one --method is required." >&2; usage >&2; exit 2; }
[[ -n "$coco_dir" ]] || { echo "--coco-dir is required." >&2; usage >&2; exit 2; }
[[ "$num_gpus" =~ ^[1-4]$ ]] || { echo "--num-gpus must be between 1 and 4." >&2; exit 2; }
command -v "$python_bin" >/dev/null 2>&1 || { echo "Python not found: $python_bin" >&2; exit 2; }

if [[ "$install_deps" -eq 1 ]]; then
  "$python_bin" -m pip install --requirement "$REPO/requirements.txt"
fi

"$python_bin" - <<'PY'
import importlib
import sys

required = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "cv2": "opencv-contrib-python-headless",
    "qwen_vl_utils": "qwen-vl-utils",
    "numpy": "numpy",
    "PIL": "Pillow",
}
missing = []
for module, package in required.items():
    try:
        importlib.import_module(module)
    except Exception as error:
        missing.append(f"{package} ({error})")
if missing:
    print("Missing or broken runtime packages:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(2)

import torch
if torch.cuda.device_count() < 1:
    raise SystemExit("No CUDA GPU is visible to PyTorch.")
print(f"Runtime preflight passed: torch={torch.__version__}, visible_gpus={torch.cuda.device_count()}")
PY

if [[ "$model_revision" != "main" && -z "$download_model_to" ]]; then
  echo "--model-revision requires --download-model-to." >&2
  exit 2
fi

if [[ -n "$download_model_to" ]]; then
  [[ "$offline" -eq 0 ]] || { echo "--download-model-to cannot be combined with --offline." >&2; exit 2; }
  if [[ -d "$model" || "$model" == /* || "$model" == ./* || "$model" == ../* ]]; then
    echo "--download-model-to expects --model to be a Hugging Face repository ID." >&2
    exit 2
  fi
  mkdir -p "$download_model_to"
  "$python_bin" - "$model" "$model_revision" "$download_model_to" <<'PY'
import pathlib
import sys
from huggingface_hub import snapshot_download

repo_id, revision, destination = sys.argv[1:]
print(f"Downloading {repo_id}@{revision} to {destination}")
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=pathlib.Path(destination),
)
PY
  model="$download_model_to"
fi

if [[ "$offline" -eq 1 ]]; then
  [[ -d "$model" ]] || { echo "--offline requires --model to be an existing local directory." >&2; exit 2; }
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

if [[ -d "$model" || "$model" == /* || "$model" == ./* || "$model" == ../* ]]; then
  [[ -d "$model" ]] || { echo "Local model directory does not exist: $model" >&2; exit 2; }
  [[ -f "$model/config.json" ]] || { echo "Local model lacks config.json: $model" >&2; exit 2; }
  model="$(cd "$model" && pwd)"
fi

if [[ -n "$output_root" ]]; then
  mkdir -p "$output_root"
  output_root="$(cd "$output_root" && pwd)"
fi

methods=()
for raw in "${methods_raw[@]}"; do
  IFS=',' read -r -a split <<< "$raw"
  methods+=("${split[@]}")
done

expanded=()
for method in "${methods[@]}"; do
  if [[ "$method" == "all" ]]; then
    expanded+=(eagle ours input_deletion tam llavacam igos_pp)
  else
    expanded+=("$method")
  fi
done

allowed=" eagle ours input_deletion tam llavacam igos_pp "
unique=()
seen=" "
for method in "${expanded[@]}"; do
  [[ "$allowed" == *" $method "* ]] || { echo "Unknown method: $method" >&2; exit 2; }
  if [[ "$seen" != *" $method "* ]]; then
    unique+=("$method")
    seen+="$method "
  fi
done

declare -A launcher=(
  [eagle]="$SCRIPT_DIR/run_eagle_250.sh"
  [ours]="$SCRIPT_DIR/run_ours_250.sh"
  [input_deletion]="$SCRIPT_DIR/run_input_deletion_250.sh"
  [tam]="$SCRIPT_DIR/run_tam_250.sh"
  [llavacam]="$SCRIPT_DIR/run_llavacam_250.sh"
  [igos_pp]="$SCRIPT_DIR/run_igos_pp_250.sh"
)

declare -A output_name=(
  [eagle]="eagle"
  [ours]="ours"
  [input_deletion]="input_deletion"
  [tam]="tam"
  [llavacam]="llavacam"
  [igos_pp]="igos_pp"
)

echo "Selected methods: ${unique[*]}"
echo "COCO input: $coco_dir"
echo "Qwen model: $model"
echo "Maximum GPUs: $num_gpus"
[[ -z "$output_root" ]] || echo "Output root: $output_root"

for method in "${unique[@]}"; do
  echo
  echo "========== $method =========="
  command=("${launcher[$method]}" "$coco_dir" "$model")
  if [[ -n "$output_root" ]]; then
    method_output="$output_root/${output_name[$method]}"
    mkdir -p "$method_output"
    command+=("$method_output")
  fi
  NUM_GPUS="$num_gpus" PYTHON_BIN="$python_bin" "${command[@]}"
done

echo
echo "All requested attribution runs completed and passed 250-image validation."
