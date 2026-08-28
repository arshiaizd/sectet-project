#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)"; source "$SCRIPT_DIR/lib/common.sh"
if [[ $# -lt 4 || $# -gt 5 ]];then echo "Usage: $0 {ours|input_level|eagle|tam|llavacam} {qwen|internvl} MMVP_IMAGES_DIR MODEL_ID_OR_PATH [OUTPUT_DIR]" >&2;exit 2;fi
METHOD="$1";FAMILY="$2";REPO="$(repo_root)";PYTHON="$(resolve_python)";IMAGES="$(cd "$3" 2>/dev/null&&pwd)"||die "MMVP images directory does not exist: $3";MODEL="$(normalize_model_id "$4")";OUTPUT="$(resolve_output_path "${5:-$REPO/runs/$FAMILY/mmvp/$METHOD/attribution}")";WORKERS="$(detect_worker_count "$PYTHON")"
[[ " ours input_level eagle tam llavacam " == *" $METHOD "* ]]||die "Unknown method: $METHOD";[[ " qwen internvl " == *" $FAMILY "* ]]||die "Unknown model family: $FAMILY"
if [[ "$FAMILY" == internvl && " ours input_level " == *" $METHOD "* ]];then die "$METHOD is not implemented for InternVL";fi
configure_runtime;export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}";mkdir -p "$OUTPUT" "$REPO/shared/generated"
TAG="$(printf '%s' "$FAMILY:$MODEL"|sha256sum|cut -c1-12)";MANIFEST="$REPO/shared/generated/mmvp_forced_choice_150_${FAMILY}_${TAG}.json";SOURCE="$REPO/shared/mmvp_forced_choice_150_seed_20260828.json"
DEVICE="$(worker_cuda_device 0)";CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON" -u "$REPO/shared/prepare_mmvp_forced_choice_targets.py" --model-family "$FAMILY" --model-id "$MODEL" --images-dir "$IMAGES" --input "$SOURCE" --output "$MANIFEST"
echo "Method=$METHOD model=$FAMILY dataset=MMVP-150 workers=$WORKERS";echo "Shared answer targets=$MANIFEST";echo "Output=$OUTPUT"

run_fixed_patch(){ local entry="$1";local begins=(0 38 76 113);local ends=(38 76 113 150);cd "$(dirname "$entry")";for((wave=0;wave<4;wave+=WORKERS));do pids=();for((slot=0;slot<WORKERS;slot++));do rank=$((wave+slot));((rank<4))||break;device="$(worker_cuda_device "$slot")";CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u "$(basename "$entry")" --model-id "$MODEL" --images-dir "$IMAGES" --eval-list "$MANIFEST" --output-csv "$OUTPUT/rank${rank}.csv" --begin "${begins[$rank]}" --end "${ends[$rank]}" --resume & pids+=("$!");done;wait_for_workers "${pids[@]}";done;"$PYTHON" "$REPO/ours/mmvp/validate_mmvp_forced_choice_attribution.py" --manifest "$SOURCE" --csv "$OUTPUT/rank0.csv" "$OUTPUT/rank1.csv" "$OUTPUT/rank2.csv" "$OUTPUT/rank3.csv";}
run_dense(){ local project="$1" entry="$2";local shards="$OUTPUT/.runtime_shards/${WORKERS}gpu";prepare_runtime_shards "$PYTHON" "$REPO" "$MANIFEST" "$shards" "$WORKERS";cd "$project";pids=();for((rank=0;rank<WORKERS;rank++));do device="$(worker_cuda_device "$rank")";CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u "$entry" --model-id "$MODEL" --Datasets "$IMAGES" --eval-list "$shards/rank${rank}-of-${WORKERS}.json" --save-dir "$OUTPUT" & pids+=("$!");done;wait_for_workers "${pids[@]}";"$PYTHON" "$REPO/shared/validate_baseline_outputs.py" --eval-list "$MANIFEST" --output-dir "$OUTPUT/npy" --kind npy;}
run_eagle_sharded(){ local project="$1" entry="$2";local shards="$OUTPUT/.runtime_shards/${WORKERS}gpu";prepare_runtime_shards "$PYTHON" "$REPO" "$MANIFEST" "$shards" "$WORKERS";cd "$project";pids=();for((rank=0;rank<WORKERS;rank++));do device="$(worker_cuda_device "$rank")";CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -u "$entry" --model-id "$MODEL" --Datasets "$IMAGES" --eval-list "$shards/rank${rank}-of-${WORKERS}.json" --save-dir "$OUTPUT" --division-number 64 & pids+=("$!");done;wait_for_workers "${pids[@]}";"$PYTHON" "$REPO/shared/validate_eagle_outputs.py" --eval-list "$MANIFEST" --output-dir "$OUTPUT/slico-1.0-1.0-division-number-64";}

if [[ "$FAMILY.$METHOD" == qwen.ours ]];then run_fixed_patch "$REPO/ours/mmvp/our_method_mmvp_forced_choice_prepared.py"
elif [[ "$FAMILY.$METHOD" == qwen.input_level ]];then run_fixed_patch "$REPO/input_deletion/qwen25vl7b_mmvp_forced_choice_input.py"
elif [[ "$FAMILY.$METHOD" == qwen.eagle ]];then cd "$REPO/eagle";"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" qwen25vl7b_mmvp_forced_choice_eagle.py --model-id "$MODEL" --Datasets "$IMAGES" --eval-list "$MANIFEST" --save-dir "$OUTPUT" --division-number 64 --attention-implementation auto;"$PYTHON" "$REPO/shared/validate_eagle_outputs.py" --eval-list "$MANIFEST" --output-dir "$OUTPUT/slico-1.0-1.0-division-number-64"
elif [[ "$FAMILY.$METHOD" == qwen.tam ]];then run_dense "$REPO/tam" qwen25vl7b_mmvp_forced_choice_tam.py
elif [[ "$FAMILY.$METHOD" == qwen.llavacam ]];then run_dense "$REPO/llavacam" qwen25vl7b_mmvp_forced_choice_llavacam.py
elif [[ "$FAMILY.$METHOD" == internvl.eagle ]];then run_eagle_sharded "$REPO/eagle" internvl35_8b_mmvp_forced_choice.py
elif [[ "$FAMILY.$METHOD" == internvl.tam ]];then run_dense "$REPO/tam" internvl35_8b_mmvp_forced_choice.py
elif [[ "$FAMILY.$METHOD" == internvl.llavacam ]];then run_dense "$REPO/llavacam" internvl35_8b_mmvp_forced_choice.py
fi
