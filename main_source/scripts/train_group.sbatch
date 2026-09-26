#!/usr/bin/env bash
set -euo pipefail
RUN="$1"
PYTHON="${READER_PYTHON:-/users/nikkieh/bibm_env/bin/python}"
export PYTHONPATH="$RUN/code"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=2025
cd "$RUN/code"
"$PYTHON" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'smoke.json'
if json.loads(p.read_text()).get('status')!='SMOKE_PASSED': raise SystemExit('Fresh-study smoke did not pass')
PY
echo "Job=$SLURM_JOB_ID Task=$SLURM_ARRAY_TASK_ID Host=$(hostname) NEW MATCHED TRAINING"
"$PYTHON" -u -m fresh.run --study "$RUN" --index "$SLURM_ARRAY_TASK_ID" --device cuda
