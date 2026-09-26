"""Validate launch dependency/resource structure without an HPC submission."""
from pathlib import Path
import json,os,shutil,subprocess

def test_mock_submission_graph(tmp_path):
    root=Path(__file__).resolve().parents[1];source=tmp_path/'source';shutil.copytree(root,source,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    bindir=tmp_path/'bin';bindir.mkdir();actual=shutil.which('python3')
    py=bindir/'fakepython'
    py.write_text('#!/bin/bash\nif [ "$1" = "-m" ] && [ "$2" = "fresh.setup" ]; then echo MOCK_SETUP; exit 0; fi\nexec '+actual+' "$@"\n');py.chmod(0o755)
    sinfo=bindir/'sinfo';sinfo.write_text('#!/bin/bash\necho "batch*"\n');sinfo.chmod(0o755)
    sb=bindir/'sbatch';sb.write_text('''#!/usr/bin/env python3
import os,json
from pathlib import Path
p=Path(os.environ['MOCK_CALLS']);rows=json.loads(p.read_text()) if p.exists() else []
import sys
rows.append(sys.argv[1:]);p.write_text(json.dumps(rows));print(str(1000+len(rows))+';cluster')
''');sb.chmod(0o755)
    env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],MOCK_CALLS=str(tmp_path/'calls.json'),READER_PYTHON=str(py),READER_OUTPUT_PARENT=str(tmp_path/'output'),READER_STAGE=str(tmp_path/'stage'))
    c=subprocess.run(['bash',str(source/'launch.sh')],env=env,capture_output=True,text=True,timeout=10)
    assert c.returncode==0,c.stdout+c.stderr
    calls=json.loads((tmp_path/'calls.json').read_text());assert len(calls)==3
    assert '--gres=gpu:1' in calls[0] and '--gres=gpu:1' in calls[1]
    assert '--array=0-83%4' in calls[1] and '--dependency=afterok:1001' in calls[1]
    assert '--dependency=afterany:1002' in calls[2]
    assert not any('gpu' in x for x in calls[2])
    study=Path((tmp_path/'stage/latest_fresh_study.txt').read_text().strip())
    assert json.loads((study/'jobs.json').read_text())['array']=='1002'

def test_shell_syntax():
    root=Path(__file__).resolve().parents[1]
    for p in list(root.glob('*.sh'))+list((root/'scripts').glob('*.sbatch')):
        assert subprocess.run(['bash','-n',str(p)],capture_output=True).returncode==0
