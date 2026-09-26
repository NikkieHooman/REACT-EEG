from __future__ import annotations
import argparse, io, json, os, shutil, subprocess, tarfile, tempfile
from pathlib import Path
from .common import digest, file_sha, read_json, write_json
from .data import SPECS, DEFAULT_ROOTS, grid, paths
from .run import MODELS, RECIPE
from .bite import COMMIT, URL


def fetch_bite(study):
    study=Path(study);dest=study/'third_party/BiteEEG'
    if dest.exists():raise FileExistsError(dest)
    candidates=[Path('/users/nikkieh/RIFT_EEG/third_party/BiteEEG')]
    source=None
    for p in candidates:
        if p.exists() and subprocess.run(['git','-C',str(p),'cat-file','-e',COMMIT+'^{commit}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            source=p;break
    if source is None:
        source=study/'third_party/fetch_git';source.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','init',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'fetch','--depth=1',URL,COMMIT],check=True,timeout=240)
    raw=subprocess.check_output(['git','-C',str(source),'archive','--format=tar',COMMIT],timeout=120)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        for member in t.getmembers():
            name=Path(member.name)
            if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe archive path')
            if member.isdir():(dest/name).mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                if member.size>100_000_000:raise ValueError('Unexpectedly large baseline file')
                (dest/name).parent.mkdir(parents=True,exist_ok=True)
                with t.extractfile(member) as src,(dest/name).open('wb') as out:shutil.copyfileobj(src,out)
            else:raise ValueError('Upstream archive contains unsupported nonregular entry '+member.name)
    hashes={str(p.relative_to(dest)):file_sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    if not (dest/'model/BiTE.py').is_file():raise ValueError('Pinned source lacks model/BiTE.py')
    write_json(study/'bite_provenance.json',dict(commit=COMMIT,origin=URL,source_archive= str(source),
               file_hashes=hashes,source_hash=digest(hashes),scope='new study pinned source; not historical checkpoint identity'))

def freeze(study):
    study=Path(study);code=study/'code';planpath=study/'study.json'
    if planpath.exists():raise FileExistsError('This study is already frozen')
    hashes={str(p.relative_to(code)):file_sha(p) for p in sorted(code.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.sh','.sbatch','.tex','.bib','.md','.json') and '__pycache__' not in p.parts}
    write_json(study/'code_hashes.json',hashes)
    plan=dict(schema_version=1,evidence='real_fresh',description='New manuscript-based refit, never historical-result reproduction',
        models=MODELS,recipe=RECIPE,seeds=[2025,2026,2027],data_roots=DEFAULT_ROOTS,
        tasks=[dict(dataset=d,subject=s,seed=k) for d,c in SPECS.items() for s in range(1,c['subjects']+1) for k in [2025,2026,2027]],
        datasets={d:{**c,'grid':grid(d)} for d,c in SPECS.items()},
        prior_test_exposure='Original study results were previously inspected. This follow-up does not make the benchmark newly untouched.',
        resamples=10000,bootstrap_seed=170926,causality_atol=1e-4,
        excluded_legacy_analyses=['HGD','LOSO','deadline-specialists'],
        comparison_family='READER minus Compact, FF and Compact-Mean: within-dataset endpoint and normalized curve; descriptive percentile intervals')
    plan['plan_hash']=digest(plan);write_json(planpath,plan)
    missing=[]
    for d,c in SPECS.items():
        for s in range(1,c['subjects']+1):
            for role in ('train','test'):
                for p in paths(DEFAULT_ROOTS[d],d,s,role):
                    if not p.is_file():missing.append(str(p))
    if missing:
        write_json(study/'missing_files.json',missing);raise FileNotFoundError('Declared data files missing; see missing_files.json')
    fetch_bite(study)
    write_json(study/'submission_review.json',dict(author_tex=None,author_approval=False,
        prepared_data_provenance_reviewed=False,conference_format_checked=False,
        note='These attestations need actual author review; do not mark true merely to suppress a warning.'))
    print('FROZEN_STUDY',study,'tasks',len(plan['tasks']),'training runs',len(plan['tasks'])*len(plan['models']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();freeze(a.study)
