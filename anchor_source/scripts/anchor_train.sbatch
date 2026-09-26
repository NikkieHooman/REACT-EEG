#!/bin/bash
set -euo pipefail

RUN="$1"
ORIG="$2"
PY="/users/nikkieh/bibm_env/bin/python"

cd "$RUN/code"
export PYTHONPATH="$PWD"

echo "job=$SLURM_JOB_ID index=$SLURM_ARRAY_TASK_ID host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -u -m fresh.anchor_diag \
    --orig-study "$ORIG" \
    --out "$RUN" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --device cuda
