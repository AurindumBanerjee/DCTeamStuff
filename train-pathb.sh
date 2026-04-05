#!/bin/bash
# PATH B launcher: TensorFlow 2.10.1 + CUDA 11.2 runtime libs
# Default behavior: run in background with nohup and write logs.

set -e
set -o pipefail

SCRIPT_DIR="/DATA/anikde/Aurindum/DCTeam/DC_VIT"
PY_SCRIPT="$SCRIPT_DIR/dc-aug-3april-pathb.py"
ENV_NAME="${TRAIN_PATHB_ENV:-bob}"
RUNS_DIR="${TRAIN_PATHB_RUNS_DIR:-$SCRIPT_DIR/runs}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${TRAIN_PATHB_RUN_DIR:-$RUNS_DIR/$RUN_ID}"
LOG_FILE="${TRAIN_PATHB_LOG:-$RUN_DIR/train.log}"

mkdir -p "$RUN_DIR"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TRAIN_OUTPUT_DIR="$RUN_DIR"

if [[ "${1:-}" == "--foreground" ]]; then
	shift
	conda run -n "$ENV_NAME" python "$PY_SCRIPT" "$@" 2>&1 | tee -a "$LOG_FILE"
	exit ${PIPESTATUS[0]}
fi

nohup conda run -n "$ENV_NAME" python "$PY_SCRIPT" "$@" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Started training in background"
echo "PID: $PID"
echo "Run dir: $RUN_DIR"
echo "Log: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"
