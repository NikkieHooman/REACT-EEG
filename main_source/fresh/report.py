"""Create tables and a review manuscript ONLY from a complete measured cohort."""
from __future__ import annotations
import argparse, json, shutil, subprocess, traceback, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .common import array_sha,digest,file_sha,read_json,write_json,verified_plan
from .data import SPECS,grid
from .statistics_base import normalized_area,analyze
from .run import code_identity

LABELS={'compact':'Compact','reader':'READER','ff':'Forward--Forward','compact_mean':'Compact--Mean','bite':'BiTE'}
DSETS={'2a':'BCICIV-2A','2b':'BCICIV-2B','ssvep':'SD-SSVEP'}

def complete_rows(study,plan):
    rows=[];results=[];missing=[];failures=[]
    for cell in plan['tasks']:
        ds,s,k=cell['dataset'],cell['subject'],cell['seed']
        group=Path(study)/'runs'/('%s_S%02d_seed%d'%(ds,s,k))
        for model in plan['models']:
            path=group/model/'result.json'
            if not path.is_file():missing.append(str(path));continue
            result=read_json(path);pred=path.parent/'predictions.npz'
            if result['cell']!=cell or result['model']!=model or result['evidence']!=plan['evidence'] or result['epochs']!=plan['recipe']['epochs'] or result['prefix_weight']!=0:
                raise ValueError('Result/protocol identity mismatch at '+str(path))
            if file_sha(path.parent/'final.pt')!=result['checkpoint_sha256']:
                raise ValueError('Final checkpoint checksum mismatch')
            if file_sha(pred)!=result['predictions_sha256']:raise ValueError('Prediction checksum mismatch')
            with np.load(pred,allow_pickle=False) as p:
                logits,y,ids,lengths=p['logits'],p['y'],p['ids'],p['lengths']
                if not np.isfinite(logits).all() or logits.shape!=(len(y),len(lengths),SPECS[ds]['classes']):raise ValueError('Invalid prediction array')
                correct=(logits.argmax(-1)==y[:,None]).sum(0)
                if lengths.tolist()!=result['lengths'] or correct.tolist()!=result['n_correct']:
                    raise ValueError('Stored metrics disagree with actual predictions')
                if array_sha(y)!=result['test_labels_hash'] or array_sha(ids)!=result['test_ids_hash']:raise ValueError('Prediction label/ID hashes disagree')
            if model!='bite':
                tests=read_json(path.parent/'causality.json')
                actual_pass=all(tests[tag]['status']=='PASSED' for tag in ('trained','nondegenerate'))
                if actual_pass!=result['causal_checks_passed']:
                    raise ValueError('Causality summary disagrees with actual test reports')
                if not actual_pass:failures.append(str(path.parent/'causality.json'))
            if s==1 and k==plan.get('seeds',[2025])[0]:
                timing=read_json(path.parent/'timing.json')
                if timing.get('batch_size')!=1:
                    raise ValueError('Missing declared batch-one timing measurement')
            result['path']=str(path);results.append(result)
            for j,m in enumerate(lengths):
                rows.append(dict(dataset=ds,subject=str(s),seed=k,model=model,n_samples=int(m),fs=SPECS[ds]['fs'],
                    accuracy=float(100*correct[j]/len(y)),n_correct=int(correct[j]),n_test=len(y),
                    test_ids_hash=result['test_ids_hash'],test_labels_hash=result['test_labels_hash'],
                    prepared_x_hash=result['test_x_hash'],normalizer_hash=result['normalizer_hash'],
                    checkpoint_sha256=result['checkpoint_sha256'],regime='endpoint',evidence_tag=plan['evidence']))
    return pd.DataFrame(rows),results,missing,failures

def latex_table(path,columns,rows):
    align='l'+'r'*(len(columns)-1)
    text=[r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(columns)+r' \\',r'\midrule']
    text+=[' & '.join(map(str,r))+r' \\' for r in rows]
    text+=[r'\bottomrule',r'\end{tabular}']
    Path(path).write_text('\n'.join(text)+'\n')

def generate_paper(study,plan,metrics,results,analysis,paper,layout_test=False):
    if plan['evidence']!='real_fresh' and not layout_test:
        raise ValueError('Synthetic results cannot populate a research manuscript')
    paper=Path(paper);paper.mkdir()
    template=Path(study)/'code/paper'
    for p in template.iterdir():
        if p.is_file():shutil.copy2(p,paper/p.name)
    subject=metrics[metrics.n_samples==metrics.dataset.map({d:c['samples'] for d,c in SPECS.items()})].groupby(['dataset','model','subject']).accuracy.mean()
    values={};rows=[]
    order=['bite','compact','compact_mean','ff','reader']
    for d in SPECS:
        row=[DSETS[d]]
        for m in order:
            a=subject.loc[(d,m)].to_numpy();values[d,m]=(float(a.mean()),float(a.std(ddof=1)))
            row.append(r'$%.2f\pm%.2f$'%values[d,m])
        rows.append(row)
    latex_table(paper/'endpoint_table.tex',['Dataset']+[LABELS[m] for m in order],rows)
    paramrows=[]
    for d in SPECS:
        row=[DSETS[d]]
        for m in order:
            nums={r['parameters'] for r in results if r['cell']['dataset']==d and r['model']==m}
            if len(nums)!=1:raise ValueError('Parameter count varies within model/dataset')
            row.append(str(next(iter(nums))))
        paramrows.append(row)
    latex_table(paper/'parameter_table.tex',['Dataset']+[LABELS[m] for m in order],paramrows)
    a=pd.read_csv(Path(analysis)/'curves.csv');paired=pd.read_csv(Path(analysis)/'paired_summary.csv')
    for d in SPECS:
        lengths=[250,500,750,1000] if d!='ssvep' else [64,128,192,256]
        rows=[]
        for n in lengths:
            row=['%.3f'%(n/SPECS[d]['fs'])]
            for m in ['compact','compact_mean','ff','reader']:
                r=a[(a.dataset==d)&(a.model==m)&(a.n_samples==n)].iloc[0]
                row.append('%.2f'%r.mean_accuracy)
            rows.append(row)
        latex_table(paper/(d+'_prefix_table.tex'),['Time (s)','Compact','C--Mean','FF','READER'],rows)
    rows=[]
    for d in SPECS:
        for base in ['compact','ff','compact_mean']:
            part=paired[(paired.dataset==d)&(paired.baseline==base)]
            e=part[part.measure=='endpoint'].iloc[0];q=part[part.measure=='normalized_accuracy_duration_area'].iloc[0]
            rows.append([DSETS[d],LABELS[base],r'$%+.2f\ [%.2f,%.2f]$'%(e.difference_pp,e.ci_low,e.ci_high),
                         r'$%+.2f\ [%.2f,%.2f]$'%(q.difference_pp,q.ci_low,q.ci_high)])
    # First two columns left aligned, overrides generic all-numeric form.
    latex_table(paper/'mechanism_table.tex',['Dataset','Comparator',r'$\Delta$ endpoint (pp)',r'$\Delta Q$ (pp)'],rows)
    p=paper/'mechanism_table.tex';p.write_text(p.read_text().replace('{lrrr}','{llrr}'))
    # Finite test scope is reported with coverage, not a bare universal claim.
    crows=[]
    for m in ['compact','reader','ff','compact_mean']:
        for tag in ['trained','nondegenerate']:
            allrows=[];nck=0
            for r in results:
                if r['model']!=m:continue
                c=read_json(Path(r['path']).parent/'causality.json')[tag]
                nck+=1;allrows+=c['rows']
            trunc=max(r['E_trunc'] for r in allrows);future=max(r['E_future'] for r in allrows if r['E_future'] is not None)
            crows.append([LABELS[m],tag,str(nck),'%.2e'%trunc,'%.2e'%future])
    latex_table(paper/'causality_table.tex',['Model','Weights','Checkpoints',r'$\max E_{trunc}$',r'$\max E_{future}$'],crows)
    p=paper/'causality_table.tex';p.write_text(p.read_text().replace('{lrrrr}','{llrrr}'))
    timings=[]
    for r in results:
        p=Path(r['path']).parent/'timing.json'
        if p.exists():
            t=read_json(p);d=r['cell']['dataset'];last=str(SPECS[d]['samples'])
            timings.append([DSETS[d],LABELS[r['model']],str(r['parameters']),'%.2f'%t['single_prefix'][last]['median_ms'],
                '%.2f'%t['shared_feature_full_curve']['median_ms'] if 'shared_feature_full_curve' in t else '--'])
    latex_table(paper/'timing_table.tex',['Dataset','Model','Parameters','Endpoint ms','Curve ms'],timings)
    p=paper/'timing_table.tex';p.write_text(p.read_text().replace('{lrrrr}','{llrrr}'))
    key=paired[(paired.dataset=='2a')&(paired.baseline=='compact')&(paired.measure=='normalized_accuracy_duration_area')].iloc[0]
    accuracy=', '.join('%.2f\\%%'%values[d,'reader'][0] for d in SPECS)
    abstract=(r'In a fresh within-subject study of 28 participants across BCICIV-2A, BCICIV-2B, and SD-SSVEP, '
       r'three-seed READER endpoint accuracies were '+accuracy+r', respectively. '
       r'Across the prespecified 1--4-s BCICIV-2A grid, the paired difference in normalized accuracy--duration area '
       r'from Compact was $%+.2f$ percentage points (95\%% subject-bootstrap interval $[%.2f,%.2f]$). '
       %(key.difference_pp,key.ci_low,key.ci_high)+
       r'Capacity and readout controls, finite trained-model causality checks, and scoped inference timings '
       r'characterize the complete configuration; they do not isolate temporal direction from every readout effect.')
    # Percent formatting above deliberately emits valid LaTeX percent escapes.
    abstract=abstract.replace(r'\\%',r'\%')
    (paper/'abstract_results.tex').write_text(abstract+'\n')
    endpoint_text=[]
    for d in SPECS:
        delta=values[d,'reader'][0]-values[d,'compact'][0]
        db=values[d,'reader'][0]-values[d,'bite'][0]
        endpoint_text.append('On %s, READER differed from Compact by $%+.2f$ percentage points and from BiTE by $%+.2f$ percentage points.'%(DSETS[d],delta,db))
    (paper/'endpoint_results.tex').write_text(' '.join(endpoint_text)+' These are measured mean differences, not claims of universal superiority.\n')
    txt=('For BCICIV-2A, the primary paired accuracy--duration difference was $%+.2f$ percentage points, with a 95\\%% subject-bootstrap interval $[%.2f,%.2f]$. '%(key.difference_pp,key.ci_low,key.ci_high))
    txt=txt.replace('\\\\%','\\%')
    if key.ci_low>0:
        txt+='The interval was above zero for this complete configuration comparison. '
    elif key.ci_high<0:
        txt+='The interval was below zero for this complete configuration comparison. '
    else:txt+='The interval included zero, leaving a positive average advantage unresolved. '
    txt+='These conditional intervals are descriptive and are not adjusted for multiple comparisons.'
    (paper/'prefix_results.tex').write_text(txt+'\n')
    interpretation=[]
    for base in ['ff','compact_mean']:
        row=paired[(paired.dataset=='2a')&(paired.baseline==base)&(paired.measure=='normalized_accuracy_duration_area')].iloc[0]
        result=('was above zero' if row.ci_low>0 else 'was below zero' if row.ci_high<0 else 'included zero')
        interpretation.append('The BCICIV-2A READER--%s accuracy--duration contrast was $%+.2f$ percentage points; its 95\\%% conditional subject-bootstrap interval $[%.2f,%.2f]$ %s.'%(LABELS[base],row.difference_pp,row.ci_low,row.ci_high,result))
    (paper/'mechanism_interpretation.tex').write_text(' '.join(interpretation)+' These comparisons constrain capacity and readout explanations but do not establish a direction-only effect.\n')
    conclusion=('In the primary BCICIV-2A comparison, READER differed from Compact in normalized accuracy--duration area by $%+.2f$ percentage points, with a 95\\%% conditional interval $[%.2f,%.2f]$. '%(key.difference_pp,key.ci_low,key.ci_high))
    if key.ci_low>0:conclusion+='This result supports a positive average difference for the complete configuration on this benchmark, not universal improvement across paradigms or observation lengths.'
    elif key.ci_high<0:conclusion+='The average difference favored the forward-only parent under the evaluated protocol.'
    else:conclusion+='The interval included zero; the present sample does not establish a positive average advantage over the forward-only parent.'
    (paper/'conclusion_results.tex').write_text(conclusion+'\n')
    for p in Path(analysis).glob('*.png'):shutil.copy2(p,paper/p.name)
    review=read_json(Path(study)/'submission_review.json')
    authors=review.get('author_tex') or r'\author{\IEEEauthorblockN{AUTHOR DETAILS REQUIRE APPROVAL}}'
    if layout_test:
        authors=r'\author{\IEEEauthorblockN{SYNTHETIC SOFTWARE TEST -- NOT RESEARCH RESULTS}}'
        main=paper/'main.tex'
        main.write_text(main.read_text().replace('READER: Observed-Prefix Reverse Reading for Causal EEG Decoding','SYNTHETIC LAYOUT TEST: NOT RESEARCH RESULTS'))
    (paper/'authors.tex').write_text(authors+'\n')
    (paper/'bib_version.tex').write_text(r'\newcommand{\BiteRevision}{\texttt{924eb32241ba}}'+'\n')
    compiled=False;errors=[]
    bibtex=shutil.which('bibtex') or shutil.which('bibtex.original')
    if shutil.which('pdflatex') and bibtex:
        commands=[['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','main.tex'],[bibtex,'main'],
                  ['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','main.tex'],['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','main.tex']]
        with (paper/'build.log').open('w') as log:
            for cmd in commands:
                c=subprocess.run(cmd,cwd=paper,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                if c.returncode:errors.append('Command failed: '+' '.join(cmd));break
        compiled=not errors
    else:errors.append('TeX tools not installed here; upload source directory to Overleaf or compile elsewhere.')
    return dict(status='SYNTHETIC_LAYOUT_ONLY' if layout_test else 'RESULTS_FILLED_REQUIRES_AUTHOR_REVIEW',compiled=compiled,errors=errors,
                author_review=review,requires=['verify prepared-data provenance and historical test exposure disclosure',
                'approve author block and final interpretation','verify venue page limit/anonymization','visually inspect compiled paper'])

def pack(study):
    study=Path(study);out=study/'return_outputs.zip';temp=study/'return_outputs.tmp'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.json','smoke.json','jobs.json','retry_jobs.json','cohort_status.json','submission_review.json','bite_provenance.json','code_hashes.json']:
            if (study/name).is_file():z.write(study/name,name)
        for p in study.glob('smoke_failure_*.json'):z.write(p,p.name)
        for folder in ['analysis','paper','failures','logs']:
            for p in (study/folder).rglob('*'):
                if p.is_file() and p.stat().st_size<8_000_000:z.write(p,str(p.relative_to(study)))
        for p in (study/'runs').rglob('*.json'):
            if p.is_file() and p.stat().st_size<4_000_000:z.write(p,str(p.relative_to(study)))
        # Own code snapshot and full protocol; upstream baseline is fetched separately.
        for p in (study/'code').rglob('*'):
            if p.is_file() and p.suffix in ('.py','.sh','.sbatch','.md','.tex','.bib','.json') and '__pycache__' not in p.parts:
                z.write(p,str(p.relative_to(study)))
    temp.replace(out);return out

def collect(study):
    study=Path(study);plan=verified_plan(study)
    try:
        code_identity(study)
        metrics,results,missing,failed=complete_rows(study,plan)
        if missing:
            status=dict(status='INCOMPLETE',expected_models=len(plan['tasks'])*len(plan['models']),found=len(results),missing=missing)
            write_json(study/'cohort_status.json',status);pack(study);return status
        # Model-independent held inputs and preprocessing must match across all seeds.
        for (ds,s),group in metrics.groupby(['dataset','subject']):
            for col in ['prepared_x_hash','test_labels_hash','test_ids_hash','normalizer_hash','n_test']:
                if group[col].nunique()!=1:raise ValueError('Cohort mismatch: '+str((ds,s,col)))
        if plan['evidence']!='real_fresh':
            status=dict(status='SYNTHETIC_SOFTWARE_VALIDATION_ONLY',found=len(results),causality_failures=failed)
            write_json(study/'cohort_status.json',status);pack(study);return status
        if plan['recipe']['epochs']!=600:raise ValueError('Do not publish an incomplete training horizon')
        a=study/'analysis'
        if a.exists():raise FileExistsError('Analysis already exists; preserve it, do not overwrite automatically')
        metrics.to_csv(study/'metrics.csv',index=False)
        manifest={'evidence_tag':'real_fresh','manifest_hash':plan['plan_hash'],'plan':{
            'datasets':{d:{**c,'subjects':list(range(1,c['subjects']+1)),'grid':grid(d)} for d,c in SPECS.items()},
            'seeds':plan['seeds'],'comparisons':[['reader',m] for m in ['compact','ff','compact_mean']],
            'bootstrap_resamples':plan['resamples'],'bootstrap_seed':plan['bootstrap_seed']}}
        analyze(study/'metrics.csv',manifest,a)
        shutil.copy2(study/'metrics.csv',a/'all_run_metrics.csv')
        if failed:
            status=dict(status='MEASURED_CAUSALITY_REVIEW_REQUIRED',failed_checks=failed,
                        note='Accuracy analyses saved; no passing-causality paper generated.')
        else:
            paperstatus=generate_paper(study,plan,metrics,results,a,study/'paper')
            status=dict(status='MEASURED_COHORT_COMPLETE',training_runs=len(results),paper=paperstatus,
                        note='Not certified submission-ready: author/provenance/venue/visual review still required.')
        write_json(study/'cohort_status.json',status);pack(study);return status
    except Exception as e:
        status=dict(status='REPORT_ERROR',error=repr(e),traceback=traceback.format_exc())
        write_json(study/'cohort_status.json',status);pack(study);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();print(json.dumps(collect(a.study),indent=2))
