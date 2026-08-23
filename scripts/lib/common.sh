#!/usr/bin/env bash

# Shared helpers for the portable COCO-250 attribution launchers.

PORTABLE_ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

die() {
  echo "Error: $*" >&2
  exit 2
}

normalize_coco_root() {
  local supplied="$1"
  if [[ -d "$supplied/val2017" ]]; then
    printf '%s\n' "$(cd "$supplied/val2017" && pwd)"
  elif [[ -d "$supplied" ]]; then
    printf '%s\n' "$(cd "$supplied" && pwd)"
  else
    die "COCO directory does not exist: $supplied"
  fi
}

normalize_model_id() {
  local supplied="
resolve_python() {"
  if [[ -d "" ]]; then
    printf '%s\n' "$(cd "" && pwd)"
  else
    printf '%s\n' ""
  fi
}

resolve_output_path() {
  local supplied="
resolve_python() {"
  if [[ "" == /* ]]; then
    printf '%s\n' ""
  else
    printf '%s\n' "$(pwd)/"
  fi
}

resolve_python() {
  local candidate="${PYTHON_BIN:-python}"
  command -v "$candidate" >/dev/null 2>&1 || die "Python not found: $candidate"
  printf '%s\n' "$candidate"
}

detect_worker_count() {
  local python_bin="$1"
  local available requested workers

  available="$($python_bin - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
  [[ "$available" =~ ^[0-9]+$ ]] || available=0
  (( available > 0 )) || die "No CUDA GPU is visible to PyTorch."

  requested="${NUM_GPUS:-4}"
  [[ "$requested" =~ ^[1-9][0-9]*$ ]] || die "NUM_GPUS must be a positive integer."

  workers="$requested"
  (( workers > 4 )) && workers=4
  (( workers > available )) && workers="$available"
  printf '%s\n' "$workers"
}

worker_cuda_device() {
  local rank="$1"
  if [[ -n "$PORTABLE_ORIGINAL_CUDA_VISIBLE_DEVICES" ]]; then
    local devices
    IFS="," read -r -a devices <<< "$PORTABLE_ORIGINAL_CUDA_VISIBLE_DEVICES"
    [[ "$rank" -lt "${#devices[@]}" ]] || die "GPU rank $rank is not visible"
    printf '%s\n' "${devices[$rank]}"
  else
    printf '%s\n' "$rank"
  fi
}

validate_benchmark_images() {
  local python_bin="$1"
  local manifest="$2"
  local coco_root="$3"
  "$python_bin" - "$manifest" "$coco_root" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
coco_root = pathlib.Path(sys.argv[2])
records = json.loads(manifest.read_text(encoding="utf-8"))
if len(records) != 250:
    raise SystemExit(f"Expected 250 benchmark records, found {len(records)}")
names = [pathlib.Path(record["image_path"]).name for record in records]
if len(names) != len(set(names)):
    raise SystemExit("Benchmark contains duplicate image paths")
missing = [name for name in names if not (coco_root / name).is_file()]
if missing:
    raise SystemExit(
        f"COCO directory is missing {len(missing)} benchmark images; "
        f"first missing: {', '.join(missing[:10])}"
    )
print(f"Validated {len(records)} benchmark images in {coco_root}")
PY
}

prepare_runtime_shards() {
  local python_bin="$1"
  local repo="$2"
  local manifest="$3"
  local output_dir="$4"
  local workers="$5"
  "$python_bin" "$repo/shared/make_runtime_shards.py" \
    --input "$manifest" \
    --output-dir "$output_dir" \
    --workers "$workers"
}

configure_runtime() {
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTHONUNBUFFERED=1
  # Local checkpoints still work with these set to 0. Users can explicitly set
  # either variable to 1 when they require a fully offline run.
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
}

print_run_configuration() {
  local method="$1" coco_root="$2" model_id="$3" output_dir="$4" workers="$5"
  echo "Method: $method"
  echo "COCO val2017: $coco_root"
  echo "Model: $model_id"
  echo "Output: $output_dir"
  echo "Visible workers used: $workers"
}

wait_for_workers() {
  local status=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}
