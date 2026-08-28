#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Run or report one cell of the 5-method x 2-dataset x 2-model matrix.

Required:
  --method NAME       ours, input_level, eagle, tam, or llavacam
  --dataset NAME      coco or mmvp
  --model NAME        qwen or internvl

For runnable experiments:
  --data-dir PATH     COCO root/val2017, or the MMVP Images directory
  --model-path VALUE  Local checkpoint or Hugging Face ID
  --output-dir PATH   Experiment output directory (default: runs/MODEL/DATASET/METHOD)
  --stage NAME        attribution, evaluation, or all (default: all)
  --num-gpus N        Maximum visible GPUs to use, 1-4 (default: 4)
  --python PATH       Python executable (default: active python)

Result reuse:
  Completed cells print their tracked final numbers and exit without GPU work.
  --force-rerun       Actually rerun a completed cell.
  --help              Show this help.
EOF
}

method=""; dataset=""; model=""; data_dir=""; model_path=""; output_dir=""
stage="all"; num_gpus=4; python_bin="${PYTHON_BIN:-python}"; force=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) method="${2:-}"; shift 2 ;;
    --dataset) dataset="${2:-}"; shift 2 ;;
    --model) model="${2:-}"; shift 2 ;;
    --data-dir) data_dir="${2:-}"; shift 2 ;;
    --model-path) model_path="${2:-}"; shift 2 ;;
    --output-dir) output_dir="${2:-}"; shift 2 ;;
    --stage) stage="${2:-}"; shift 2 ;;
    --num-gpus) num_gpus="${2:-}"; shift 2 ;;
    --python) python_bin="${2:-}"; shift 2 ;;
    --force-rerun) force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ " ours input_level eagle tam llavacam " == *" $method "* ]] || { echo "Invalid --method: $method" >&2; exit 2; }
[[ " coco mmvp " == *" $dataset "* ]] || { echo "Invalid --dataset: $dataset" >&2; exit 2; }
[[ " qwen internvl " == *" $model "* ]] || { echo "Invalid --model: $model" >&2; exit 2; }
[[ " attribution evaluation all " == *" $stage "* ]] || { echo "Invalid --stage: $stage" >&2; exit 2; }
[[ "$num_gpus" =~ ^[1-4]$ ]] || { echo "--num-gpus must be 1-4" >&2; exit 2; }
command -v "$python_bin" >/dev/null 2>&1 || { echo "Python not found: $python_bin" >&2; exit 2; }

key="$model.$dataset.$method"
readarray -t registry_values < <("$python_bin" - "$REPO/benchmark/results.json" "$key" <<'PY'
import json, sys
entry = json.load(open(sys.argv[1], encoding="utf-8"))["experiments"].get(sys.argv[2])
if entry is None:
    raise SystemExit(f"Unknown experiment: {sys.argv[2]}")
print(entry["status"])
print("1" if entry.get("runner_available") else "0")
print(entry.get("notes", ""))
PY
)
status="${registry_values[0]}"; runner_available="${registry_values[1]}"; notes="${registry_values[2]:-}"

if [[ "$status" == completed && "$force" -eq 0 ]]; then
  echo "$key is already complete. Reusing tracked final results; no GPU job was started."
  "$python_bin" "$REPO/benchmark/show_results.py" --model "$model" --dataset "$dataset" --method "$method"
  echo "Use --force-rerun only if you intentionally want to regenerate it."
  exit 0
fi
if [[ "$runner_available" != 1 ]]; then
  echo "$key is not runnable yet: $status" >&2
  [[ -z "$notes" ]] || echo "$notes" >&2
  exit 3
fi

[[ -n "$data_dir" ]] || { echo "--data-dir is required when a run will execute." >&2; exit 2; }
if [[ -z "$model_path" ]]; then
  if [[ "$model" == qwen ]]; then model_path="Qwen/Qwen2.5-VL-7B-Instruct"
  else model_path="OpenGVLab/InternVL3_5-8B-HF"; fi
fi
output_dir="${output_dir:-$REPO/runs/$model/$dataset/$method}"
mkdir -p "$output_dir"
export NUM_GPUS="$num_gpus" PYTHON_BIN="$python_bin"

run_attribution() {
  if [[ "$model.$dataset" == qwen.coco ]]; then
    case "$method" in
      ours) "$SCRIPT_DIR/run_ours_250.sh" "$data_dir" "$model_path" "$output_dir/attribution" ;;
      input_level) "$SCRIPT_DIR/run_input_deletion_250.sh" "$data_dir" "$model_path" "$output_dir/attribution" ;;
      eagle) "$SCRIPT_DIR/run_eagle_250.sh" "$data_dir" "$model_path" "$output_dir/attribution" ;;
      tam) "$SCRIPT_DIR/run_tam_250.sh" "$data_dir" "$model_path" "$output_dir/attribution" ;;
      llavacam) "$SCRIPT_DIR/run_llavacam_250.sh" "$data_dir" "$model_path" "$output_dir/attribution" ;;
    esac
  elif [[ "$dataset" == mmvp ]]; then
    "$SCRIPT_DIR/run_mmvp_150.sh" "$method" "$model" "$data_dir" \
      "$model_path" "$output_dir/attribution"
  elif [[ "$model.$dataset" == internvl.coco ]]; then
    "$SCRIPT_DIR/run_internvl_baseline_250.sh" "$method" "$data_dir" "$model_path" "$output_dir/attribution"
  else
    echo "Internal error: no attribution dispatch for $key" >&2; exit 3
  fi
}

run_evaluation() {
  if [[ "$model.$dataset" == qwen.coco ]]; then
    case "$method" in
      ours|input_level)
        "$SCRIPT_DIR/evaluate_qwen_patch_250.sh" "$method" "$data_dir" "$model_path" \
          "$output_dir/attribution" "$output_dir/evaluation"
        ;;
      eagle)
        "$SCRIPT_DIR/evaluate_qwen_eagle_250.sh" "$data_dir" "$output_dir/attribution"
        ;;
      tam|llavacam)
        "$SCRIPT_DIR/run_dense_baseline_evaluation_250.sh" "$method" "$data_dir" \
          "$model_path" "$output_dir/attribution"
        ;;
    esac
  elif [[ "$dataset" == mmvp ]]; then
    "$SCRIPT_DIR/evaluate_mmvp_150.sh" "$method" "$model" "$data_dir" \
      "$model_path" "$output_dir/attribution" "$output_dir/evaluation"
  elif [[ "$model.$dataset" == internvl.coco ]]; then
    "$SCRIPT_DIR/evaluate_internvl_baseline_250.sh" "$method" "$data_dir" \
      "$model_path" "$output_dir/attribution"
  else
    echo "Internal error: no evaluation dispatch for $key" >&2; exit 3
  fi
}

echo "Experiment: $key"
echo "Registry status before run: $status"
echo "Stage: $stage"
echo "Output: $output_dir"
[[ "$stage" == evaluation ]] || run_attribution
[[ "$stage" == attribution ]] || run_evaluation
