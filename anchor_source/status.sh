#!/usr/bin/env bash
set -euo pipefail
RUN="${1:-$(cat /users/nikkieh/reader_followup/session_B1VqC0/latest_fresh_study.txt)}"
[ -s "$RUN/jobs.json" ] || { echo 'Job submission has not finished; inspect logs/setup.out'; exit 1; }
python3 - "$RUN" <<'PY'
import json,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]);j=json.loads((r/'jobs.json').read_text())
print('Run:',r,flush=True)
ids=[j[k] for k in ('preflight','array','collector') if k in j]
retry=r/'retry_jobs.json'
if retry.exists():
 for pair in json.loads(retry.read_text()):ids.extend([pair['array'],pair['collector']])
subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,JobName%22,State,ExitCode,Elapsed'])
for p in [r/'smoke.json',r/'cohort_status.json']:
 if p.exists():
  d=json.loads(p.read_text());print(p.name, d.get('status'));print(d.get('error',''))
complete=list((r/'runs').glob('*/*/result.json'))
trained=list((r/'runs').glob('*/*/final.pt'))
failures=list((r/'failures').glob('*.json'))
print('Final checkpoints:',len(trained),'/420; measured models:',len(complete),'/420; task errors:',len(failures))
for p in failures[:5]:print(p, json.loads(p.read_text()).get('error'))
if (r/'return_outputs.zip').exists():print('RETURN_ARCHIVE='+str(r/'return_outputs.zip'))
PY
