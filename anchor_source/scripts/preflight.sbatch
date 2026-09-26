#!/usr/bin/env bash
set -euo pipefail
RUN="$1"
PYTHON="${READER_PYTHON:-/users/nikkieh/bibm_env/bin/python}"
export PYTHONPATH="$RUN/code"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=2025
cd "$RUN/code"
echo "Job=$SLURM_JOB_ID Host=$(hostname) NEW STUDY TRAINING-ONLY SMOKE"
"$PYTHON" -u -m fresh.smoke --study "$RUN"
