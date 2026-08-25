#!/usr/bin/env bash

# Complete, resumable old-prompt run: attribution followed by all evaluations.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_ours_mask_tail_250_4gpu.sh"
bash "$SCRIPT_DIR/run_evaluation_patch8_mask_tail_250_4gpu.sh"
