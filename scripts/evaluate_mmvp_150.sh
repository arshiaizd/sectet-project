#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)";source "$SCRIPT_DIR/lib/common.sh"
if [[ $# -lt 5 || $# -gt 6 ]];then echo "Usage: $0 METHOD {qwen|internvl} MMVP_IMAGES_DIR MODEL_ID_OR_PATH ATTRIBUTION_DIR [EVALUATION_DIR]" >&2;exit 2;fi
METHOD="$1";FAMILY="$2";REPO="$(repo_root)";PYTHON="$(resolve_python)";IMAGES="$(cd "$3"&&pwd)";MODEL="$(normalize_model_id "$4")";ATTR="$(resolve_output_path "$5")";EVAL="$(resolve_output_path "${6:-$ATTR/evaluation}")";WORKERS="$(detect_worker_count "$PYTHON")";configure_runtime;export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}";mkdir -p "$EVAL"
TAG="$(printf '%s' "$FAMILY:$MODEL"|sha256sum|cut -c1-12)";MANIFEST="$REPO/shared/generated/mmvp_forced_choice_150_${FAMILY}_${TAG}.json";[[ -s "$MANIFEST" ]]||die "Missing shared target manifest: run attribution first"
if [[ "$FAMILY" == qwen && " ours input_level " == *" $METHOD "* ]];then "$SCRIPT_DIR/evaluate_qwen_mmvp_ours_150.sh" "$IMAGES" "$MODEL" "$ATTR" "$EVAL";exit 0;fi
if [[ "$METHOD" == eagle ]];then ROOT="$ATTR/slico-1.0-1.0-division-number-64";"$PYTHON" "$REPO/shared/validate_eagle_outputs.py" --eval-list "$MANIFEST" --output-dir "$ROOT";"$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" --explanation-dir "$ROOT" --sensitiveity 0.2 2>&1|tee "$EVAL/evaluation_metrics.txt";exit 0;fi
[[ " tam llavacam " == *" $METHOD "* ]]||die "Unsupported MMVP evaluation: $FAMILY.$METHOD"
"$PYTHON" "$REPO/shared/validate_baseline_outputs.py" --eval-list "$MANIFEST" --output-dir "$ATTR/npy" --kind npy
SHARDS="$EVAL/.runtime_shards/${WORKERS}gpu";prepare_runtime_shards "$PYTHON" "$REPO" "$MANIFEST" "$SHARDS" "$WORKERS";pids=()
for((rank=0;rank<WORKERS;rank++));do device="$(worker_cuda_device "$rank")";if [[ "$FAMILY" == qwen ]];then ENTRY="$REPO/shared/qwen_mmvp_dense_faithfulness.py";else ENTRY="$REPO/shared/internvl_mmvp_dense_faithfulness.py";fi;CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u "$ENTRY" --model-id "$MODEL" --Datasets "$IMAGES" --eval-list "$SHARDS/rank${rank}-of-${WORKERS}.json" --eval-dir "$ATTR" --division-number 64 & pids+=("$!");done;wait_for_workers "${pids[@]}"
"$PYTHON" "$REPO/shared/validate_baseline_outputs.py" --eval-list "$MANIFEST" --output-dir "$ATTR/json" --kind json
"$PYTHON" "$REPO/eagle/eval_AUC_faithfulness.py" --explanation-dir "$ATTR" --sensitiveity 0.2 2>&1|tee "$EVAL/evaluation_metrics.txt"
