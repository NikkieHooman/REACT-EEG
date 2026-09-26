"""Synthetic, nonresearch fixtures for report generation and rejection tests."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import pytest
from fresh.common import write_json,array_sha,file_sha,digest
from fresh.data import SPECS,grid
from fresh.run import RECIPE
from fresh.report import complete_rows,collect,generate_paper
from fresh.statistics_base import analyze

def fixture(study,full=False):
    code=study/'code';code.mkdir()
    shutil.copytree(Path(__file__).resolve().parents[1]/'paper',code/'paper')
    write_json(study/'code_hashes.json',{})
    write_json(study/'bite_provenance.json',{'file_hashes':{}})
    write_json(study/'submission_review.json',{'author_tex':None})
    models=['compact','reader','ff','compact_mean','bite']
    tasks=[{'dataset':d,'subject':s,'seed':k} for d in SPECS for s in [1,2] for k in [2025,2026]] if full else [{'dataset':'2a','subject':1,'seed':2025}]
    plan={'evidence':'synthetic','recipe':RECIPE,'tasks':tasks,'models':models,'resamples':100,'seeds':[2025,2026],'bootstrap_seed':170926}
    plan['plan_hash']=digest(plan);write_json(study/'study.json',plan)
    rng=np.random.default_rng(13)
    for task in tasks:
        d,s,k=task['dataset'],task['subject'],task['seed'];cfg=SPECS[d]
        y=np.arange(24)%cfg['classes'];ids=np.asarray(['trial_%d'%i for i in range(len(y))])
        for model in models:
            loc=study/'runs'/('%s_S%02d_seed%d'%(d,s,k))/model;loc.mkdir(parents=True)
            lengths=[cfg['samples']] if model=='bite' else grid(d)
            logits=rng.normal(size=(len(y),len(lengths),cfg['classes'])).astype('float32')
            np.savez_compressed(loc/'predictions.npz',logits=logits,y=y,ids=ids,lengths=np.asarray(lengths))
            (loc/'final.pt').write_bytes(b'SYNTHETIC UNIT TEST PLACEHOLDER NOT A MODEL')
            result={'status':'MEASURED','evidence':'synthetic','cell':task,'model':model,'epochs':600,'prefix_weight':0,
                'checkpoint_sha256':file_sha(loc/'final.pt'),'predictions_sha256':file_sha(loc/'predictions.npz'),
                'lengths':lengths,'n_correct':(logits.argmax(-1)==y[:,None]).sum(0).tolist(),
                'test_labels_hash':array_sha(y),'test_ids_hash':array_sha(ids),'test_x_hash':'synthetic_input',
                'normalizer_hash':'synthetic_training_scaler','parameters':100,'causal_checks_passed':True if model!='bite' else None}
            write_json(loc/'result.json',result)
            if s==1 and k==2025:write_json(loc/'timing.json',{'batch_size':1,'single_prefix':{str(cfg['samples']):{'median_ms':0.01}},'scope':'SYNTHETIC FIXTURE NOT REAL TIMING'})
            if model!='bite':write_json(loc/'causality.json',{x:{'status':'PASSED','rows':[{'E_trunc':0.,'E_future':0.}]} for x in ['trained','nondegenerate']})
    return plan

def test_synthetic_not_published(tmp_path):
    plan=fixture(tmp_path)
    out=collect(tmp_path)
    assert out['status']=='SYNTHETIC_SOFTWARE_VALIDATION_ONLY'
    assert not (tmp_path/'paper').exists()

def test_prediction_tamper_rejected(tmp_path):
    plan=fixture(tmp_path);p=next((tmp_path/'runs').rglob('predictions.npz'))
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):complete_rows(tmp_path,plan)

def test_checkpoint_tamper_rejected(tmp_path):
    plan=fixture(tmp_path);p=next((tmp_path/'runs').rglob('final.pt'));p.write_bytes(b'other checkpoint')
    with pytest.raises(ValueError,match='checkpoint checksum'):complete_rows(tmp_path,plan)

def test_false_600epoch_claim_rejected(tmp_path):
    plan=fixture(tmp_path);p=next((tmp_path/'runs').rglob('result.json'));r=json.loads(p.read_text());r['epochs']=2;write_json(p,r)
    with pytest.raises(ValueError,match='identity mismatch'):complete_rows(tmp_path,plan)

def test_full_reporting_layout(tmp_path):
    plan=fixture(tmp_path,full=True)
    metrics,results,missing,failed=complete_rows(tmp_path,plan)
    assert not missing and not failed and len(results)==60
    p=tmp_path/'metrics.csv';metrics.to_csv(p,index=False)
    manifest={'evidence_tag':'synthetic','manifest_hash':plan['plan_hash'],'plan':{'datasets':{d:{**c,'subjects':[1,2],'grid':grid(d)} for d,c in SPECS.items()},
        'seeds':[2025,2026],'comparisons':[['reader',m] for m in ['compact','ff','compact_mean']],'bootstrap_resamples':100,'bootstrap_seed':9}}
    a=analyze(p,manifest,tmp_path/'analysis')
    with pytest.raises(ValueError,match='Synthetic'):generate_paper(tmp_path,plan,metrics,results,a,tmp_path/'not_allowed')
    out=generate_paper(tmp_path,plan,metrics,results,a,tmp_path/'layout_only',layout_test=True)
    paper=tmp_path/'layout_only'
    assert 'SYNTHETIC' in (paper/'authors.tex').read_text()
    assert 'SYNTHETIC' in (paper/'main.tex').read_text()
    assert '95\\%' in (paper/'abstract_results.tex').read_text()
    assert len(pd.read_csv(a/'paired_summary.csv'))==18
    # Save only a NONNUMERIC local testing status, never the fixture predictions.
    write_json(Path(__file__).resolve().parents[1]/'validation/layout_status.json',out)
    dest=Path('/mnt/data/_fresh_build/layout_validation')
    if dest.exists():shutil.rmtree(dest)
    shutil.copytree(paper,dest)
