#!/usr/bin/env bash
# Explicit retry ONLY after original array has ended; resumes compatible checkpoints.
set -euo pipefail
RUN="${1:?Pass the exact study directory}"
PY=/users/nikkieh/bibm_env/bin/python
export PYTHONPATH="$RUN/code"
"$PY" - "$RUN" <<'PYCHECK'
import json,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]);smoke=json.loads((r/'smoke.json').read_text())
if smoke.get('status')!='SMOKE_PASSED':raise SystemExit('Preflight did not pass; do not retry training.')
jobs=json.loads((r/'jobs.json').read_text());ids=[jobs[k] for k in ('array','collector') if k in jobs]
if (r/'retry_jobs.json').exists():
 for j in json.loads((r/'retry_jobs.json').read_text()):ids.extend([j['array'],j['collector']])
for j in ids:
 c=subprocess.run(['squeue','-h','-j',j],capture_output=True,text=True)
 if c.returncode and 'Invalid job id' not in c.stderr and 'Invalid job_id' not in c.stderr:
  raise SystemExit('Cannot verify old job status: '+c.stderr)
 if c.stdout.strip():raise SystemExit('A training or collection job is still active: '+j)
PYCHECK
INDICES="$($PY - "$RUN" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);p=json.loads((r/'study.json').read_text());missing=[]
for i,c in enumerate(p['tasks']):
 g=r/'runs'/('%s_S%02d_seed%d'%(c['dataset'],c['subject'],c['seed']))
 if not all((g/m/'result.json').is_file() for m in p['models']):missing.append(str(i))
print(','.join(missing))
PY
)"
[ -n "$INDICES" ] || { echo 'No incomplete measured-model tasks'; exit 0; }
cd "$RUN/code"
JRAW="$(sbatch --parsable --partition=batch --account=eclarson_ehr_fair_0003 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=12:00:00 --array="$INDICES%4" --job-name=reader_retry --chdir="$RUN/code" --output="$RUN/logs/retry_%A_%a.out" --error="$RUN/logs/retry_%A_%a.out" scripts/train_group.sbatch "$RUN")"
J="${JRAW%%;*}"
CRAW="$(sbatch --parsable --partition=batch --account=eclarson_ehr_fair_0003 --cpus-per-task=2 --mem=12G --time=00:30:00 --dependency="afterany:$J" --job-name=reader_report_retry --chdir="$RUN/code" --output="$RUN/logs/collect_retry_%j.out" scripts/collect.sbatch "$RUN")"
C="${CRAW%%;*}"
"$PY" - "$RUN" "$J" "$C" <<'PYSAVE'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/"retry_jobs.json";rows=json.loads(p.read_text()) if p.exists() else []
rows.append(dict(array=sys.argv[2],collector=sys.argv[3]));p.write_text(json.dumps(rows,indent=2)+"\n")
PYSAVE
echo "Retry array $J; do not edit source/recipe/checkpoint metadata to suppress a failure."
