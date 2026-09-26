#!/usr/bin/env bash
# New study only. No edits to /users/nikkieh/bibm or existing weights.
set -euo pipefail
umask 077
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${READER_PYTHON:-/users/nikkieh/bibm_env/bin/python}"
PARENT="${READER_OUTPUT_PARENT:-/lustre/smuexa01/client/users/nikkieh/reader_fresh_runs}"
STAGE="${READER_STAGE:-/users/nikkieh/reader_followup/session_B1VqC0}"
CONCURRENT="${READER_CONCURRENT_GPUS:-4}"
case "$CONCURRENT" in 1|2|3|4|5|6|7|8) ;; *) echo 'READER_CONCURRENT_GPUS must be 1..8'; exit 1;; esac
[ -x "$PYTHON" ] || { echo "Python missing: $PYTHON"; exit 1; }
command -v sbatch >/dev/null || { echo 'Run launch.sh on the HPC login node'; exit 1; }
sinfo -h -o '%P' | sed 's/\*$//' | grep -Fx batch >/dev/null || { echo 'The batch partition is not exposed; stop rather than guess'; exit 1; }
mkdir -p "$PARENT" "$STAGE"
RUN="$(mktemp -d "$PARENT/study_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
mkdir -p "$RUN/code" "$RUN/logs" "$RUN/runs" "$RUN/failures"
cp -R "$SOURCE"/. "$RUN/code/"
export PYTHONPATH="$RUN/code"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$RUN/code"
printf '%s\n' "$RUN" > "$STAGE/latest_fresh_study.txt"
"$PYTHON" -m fresh.setup --study "$RUN" 2>&1 | tee "$RUN/logs/setup.out"
COMMON=(--parsable --partition=batch --account=eclarson_ehr_fair_0003 --chdir="$RUN/code")
record_job() {
  "$PYTHON" - "$RUN/jobs.json" "$1" "$2" <<'PYJOB'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);value=sys.argv[3]
if not value.isdigit():raise SystemExit('Invalid returned Slurm job ID: '+value)
data=json.loads(path.read_text()) if path.exists() else {}
data[sys.argv[2]]=value
path.write_text(json.dumps(data,indent=2)+'\n')
PYJOB
}
trap 'echo "Submission stopped. Preserve $RUN; inspect jobs.json before any retry. Already accepted jobs are not silently cancelled."' ERR
PREFLIGHT_RAW="$(sbatch "${COMMON[@]}" --job-name=reader_fresh_smoke --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 --output="$RUN/logs/preflight.out" --error="$RUN/logs/preflight.out" scripts/preflight.sbatch "$RUN")"
PREFLIGHT="${PREFLIGHT_RAW%%;*}"
record_job preflight "$PREFLIGHT"
ARRAY_RAW="$(sbatch "${COMMON[@]}" --job-name=reader_fresh --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=12:00:00 --array="0-83%$CONCURRENT" --dependency="afterok:$PREFLIGHT" --kill-on-invalid-dep=yes --output="$RUN/logs/task_%A_%a.out" --error="$RUN/logs/task_%A_%a.out" scripts/train_group.sbatch "$RUN")"
ARRAY="${ARRAY_RAW%%;*}"
record_job array "$ARRAY"
COLLECT_RAW="$(sbatch "${COMMON[@]}" --job-name=reader_report --cpus-per-task=2 --mem=12G --time=00:30:00 --dependency="afterany:$ARRAY" --output="$RUN/logs/collect.out" --error="$RUN/logs/collect.out" scripts/collect.sbatch "$RUN")"
COLLECT="${COLLECT_RAW%%;*}"
record_job collector "$COLLECT"
"$PYTHON" - "$RUN" "$PREFLIGHT" "$ARRAY" "$COLLECT" "$CONCURRENT" <<'PY'
import json,sys
from pathlib import Path
run,smoke,array,collect,concurrent=sys.argv[1:]
p=Path(run)/'jobs.json';p.write_text(json.dumps(dict(preflight=smoke,array=array,collector=collect,concurrent_gpus=int(concurrent)),indent=2)+'\n')
PY
printf '\nFRESH STUDY SUBMITTED\nRun: %s\nPreflight: %s\nTraining array: %s\nCollector: %s\n' "$RUN" "$PREFLIGHT" "$ARRAY" "$COLLECT"
printf '84 subject/seed tasks, 5 models per task, 600 epochs each. At most %s concurrent GPUs.\n' "$CONCURRENT"
printf 'Old runs are untouched. Real training begins only after training-role smoke checks pass.\n'
printf 'Watch: bash "%s/code/status.sh" "%s"\n' "$RUN" "$RUN"
printf 'Return archive after collection: %s/return_outputs.zip\n' "$RUN"
