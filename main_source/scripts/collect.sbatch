#!/usr/bin/env bash
set -euo pipefail
RUN="$1"
PYTHON="${READER_PYTHON:-/users/nikkieh/bibm_env/bin/python}"
export PYTHONPATH="$RUN/code"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
cd "$RUN/code"
"$PYTHON" -u -m fresh.report --study "$RUN"
