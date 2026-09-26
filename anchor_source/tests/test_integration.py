"""End-to-end new-core-model pipeline on synthetic arrays only; no BiTE imitation."""
from pathlib import Path
import numpy as np
import pytest
import torch
from fresh import run
from fresh.data import SPECS,paths
from fresh.common import write_json,digest,file_sha
from fresh.models import make_model
from fresh.report import collect

torch.set_num_threads(1)

def test_fresh_group_training_inference_causality_reporting(tmp_path,monkeypatch):
    (tmp_path/'code').mkdir();write_json(tmp_path/'code_hashes.json',{})
    write_json(tmp_path/'bite_provenance.json',{'source_hash':'NO_BASELINE_IN_THIS_SYNTHETIC_TEST','file_hashes':{}})
    roots={d:str(tmp_path/'data'/d) for d in SPECS};rng=np.random.default_rng(83)
    for d,c in SPECS.items():
        Path(roots[d]).mkdir(parents=True)
        for role in ['train','test']:
            for p in paths(roots[d],d,1,role):
                x=rng.normal(size=(12,c['channels'],c['samples'])).astype('float32');y=np.arange(12)%c['classes']
                np.savez_compressed(p,data=x,label=y)
    models=['compact','reader','ff','compact_mean']
    tasks=[{'dataset':d,'subject':1,'seed':2025} for d in SPECS]
    plan={'evidence':'synthetic','models':models,'tasks':tasks,'seeds':[2025], 'data_roots':roots,
          'recipe':{**run.RECIPE,'epochs':2,'batch_size':4,'save_every':1},'resamples':100,'bootstrap_seed':19}
    plan['plan_hash']=digest(plan);write_json(tmp_path/'study.json',plan)
    original=run.load_role;events=[]
    def audited_load(root,d,s,role):
        events.append((d,role))
        if role=='test':
            g=tmp_path/'runs'/('%s_S01_seed2025'%d)
            assert all((g/m/'final.pt').exists() for m in models)
        return original(root,d,s,role)
    monkeypatch.setattr(run,'load_role',audited_load)
    # Long performance benchmarking is tested separately, not conflated with training validation.
    monkeypatch.setattr(run,'measure_timing',lambda *args,**kwargs:{'batch_size':1,'scope':'TIMING SKIPPED IN SYNTHETIC INTEGRATION TEST'})
    for i in range(3):run.group_task(tmp_path,i,device='cpu')
    result=collect(tmp_path)
    assert result['status']=='SYNTHETIC_SOFTWARE_VALIDATION_ONLY' and result['found']==12
    assert not result['causality_failures'] and not (tmp_path/'paper').exists()
    assert events==[(d,role) for d in SPECS for role in ['train','test']]
    finals=list((tmp_path/'runs').rglob('final.pt'));before={str(p):file_sha(p) for p in finals}
    run.group_task(tmp_path,0,device='cpu')
    assert before=={str(p):file_sha(p) for p in finals}
    write_json(Path(__file__).resolve().parents[1]/'validation/integration_status.json',
        {'evidence':'synthetic_software_test','core_model_training_runs':12,'epochs_each':2,
         'new_core_models_tested':models,'datasets':'synthetic arrays at all three configured input shapes',
         'held_role_opened_after_all_models_trained':True,'resume_completed_group_no_weight_changes':True,
         'actual_upstream_BiTE_tested':False,'gpu_used':False,'paper_generated':False})

def test_timing_actual_cpu_function():
    m=make_model(3,2,1000,32,'compact').eval();x=np.random.default_rng(1).normal(size=(2,3,1000)).astype('float32')
    t=run.measure_timing(m,x,[250,1000],torch.device('cpu'),repetitions=2,warmup=1)
    assert t['batch_size']==1 and t['single_prefix']['250']['median_ms']>0
    assert 'shared_feature_full_curve' in t and 'naive_recomputed_full_curve' in t
