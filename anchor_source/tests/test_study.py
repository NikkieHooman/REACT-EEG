"""Local software tests. All arrays here are synthetic, not EEG outcomes."""
from pathlib import Path
import copy, json, shutil
import numpy as np
import pandas as pd
import pytest
import torch
from fresh.common import state_sha, digest, write_json, verified_plan
from fresh.data import SPECS, paths, grid, load_role, fit_scaler, normalize, check_role_overlap
from fresh.models import make_model, Block, Tokenizer
from fresh.run import constrain,shared_sha,train_one,RECIPE,check_causality,predict,weights_load
from fresh.statistics_base import normalized_area,exact_signflip,holm,paired_panel
from fresh.report import complete_rows,collect

torch.set_num_threads(1)

@pytest.mark.parametrize('d',['2a','2b','ssvep'])
def test_grid(d):
    g=grid(d)
    assert len(g)==(49 if d=='ssvep' else 28)
    assert g==sorted(set(g)) and g[-1]==SPECS[d]['samples']
    assert g[0]==(64 if d=='ssvep' else 250)

@pytest.mark.parametrize('d',['2a','2b','ssvep'])
def test_role_paths(d):
    tr=paths('/data',d,1,'train');te=paths('/data',d,1,'test')
    assert set(tr).isdisjoint(te)
    assert len(tr)==(3 if d=='2b' else 1)
    assert len(te)==(2 if d=='2b' else 1)
    if d=='2b':assert str(te[0]).endswith('B0104E.npz')

def test_coordinate_scaler_prefix_and_no_fit():
    rng=np.random.default_rng(7);x=rng.normal(size=(12,3,1000));sc=fit_scaler(x)
    x2=x[:2]+30;before=digest(sc)
    assert np.array_equal(normalize(x2,sc)[:,:,:250],normalize(x2[:,:,:250],sc))
    assert before==digest(sc)
    assert abs(normalize(x,sc).mean())<1e-6

def test_overlap_rejects():
    x=np.zeros((2,3,1000));y=np.ones((3,3,1000))
    assert check_role_overlap(x,y)['exact_duplicate_trials']==0
    with pytest.raises(ValueError):check_role_overlap(x,x[:1])

def test_no_label_guess_and_mi_crop(tmp_path):
    x=np.zeros((4,22,1001));y=np.arange(4)
    np.savez(tmp_path/'A01T.npz',data=x,label=y)
    out,label,ids,rec=load_role(tmp_path,'2a',1,'train')
    assert out.shape==(4,22,1000) and ids[0]=='A01T.npz#row000000'
    np.savez(tmp_path/'A01T.npz',data=x,label=y.astype(float))
    with pytest.raises(ValueError):load_role(tmp_path,'2a',1,'train')

def test_partial_mean_oracle():
    tok=Tokenizer(3,1000,32).eval()
    with torch.no_grad():
        tok.project.weight.zero_();tok.project.bias.zero_();tok.position.zero_()
        tok.project.weight[0,0]=1
    u=torch.arange(250,dtype=torch.float32).view(1,1,-1).repeat(1,96,1)
    z=tok.from_features(u)
    assert z.shape==(1,64,8)
    assert z[0,0,-1].item()==pytest.approx(np.arange(224,250).mean())
    ufull=torch.arange(256,dtype=torch.float32).view(1,1,-1).repeat(1,96,1)
    assert not torch.equal(z[:,:,-1],tok.from_features(ufull)[:,:,-1])

def test_identity_blocks():
    b=Block(64,4,.3).eval();x=torch.randn(3,64,32)
    assert torch.equal(x,b(x))

@pytest.mark.parametrize('d',['2a','2b','ssvep'])
def test_parameters_and_pairing(d):
    c=SPECS[d];models={k:make_model(c['channels'],c['classes'],c['samples'],c['pool'],k) for k in ['compact','reader','ff','compact_mean']}
    for m in models.values():constrain(m)
    n={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
    assert n['reader']-n['compact']==3136 and n['reader']==n['ff'] and n['compact']==n['compact_mean']
    assert len({shared_sha(m) for m in models.values()})==1
    assert torch.equal(models['reader'].gate_logits,torch.zeros(64))
    assert all(torch.equal(a,b) for a,b in zip(models['reader'].second_reader.parameters(),models['ff'].second_reader.parameters()))

def test_extra_rng_isolated():
    torch.manual_seed(90);a=torch.get_rng_state().clone();make_model(3,2,1000,32,'reader');assert torch.equal(a,torch.get_rng_state())

def test_training_and_resume_identical(tmp_path):
    rng=np.random.default_rng(2);x=rng.normal(size=(12,3,1000)).astype('float32');y=np.arange(12)%2
    recipe={**RECIPE,'epochs':3,'batch_size':4,'save_every':1}
    def new():
        m=make_model(3,2,1000,32,'reader');constrain(m);return m
    a=new();sh=shared_sha(a);ih=state_sha(a)
    full=train_one(a,x,y,tmp_path/'full','test',2025,recipe,torch.device('cpu'),ih,sh,evidence='synthetic')
    b=new();train_one(b,x,y,tmp_path/'res','test',2025,recipe,torch.device('cpu'),ih,sh,evidence='synthetic',interrupt_after=1)
    c=new();resume=train_one(c,x,y,tmp_path/'res','test',2025,recipe,torch.device('cpu'),ih,sh,evidence='synthetic')
    assert full['orders']==resume['orders']
    assert all(torch.equal(full['state_dict'][k],resume['state_dict'][k]) for k in full['state_dict'])
    assert full['prefix_weight']==0 and full['test_data_opened'] is False
    assert len(full['orders'])==3
    assert torch.equal(full['training_probe_logits'],resume['training_probe_logits'])
    with pytest.raises(ValueError):train_one(new(),x,y,tmp_path/'res','changed',2025,recipe,torch.device('cpu'),ih,sh,evidence='synthetic')

@pytest.mark.parametrize('kind',['compact','reader','ff','compact_mean'])
@pytest.mark.parametrize('d',['2a','2b','ssvep'])
def test_nontrivial_causal(kind,d):
    c=SPECS[d];m=make_model(c['channels'],c['classes'],c['samples'],c['pool'],kind).eval()
    x=np.random.default_rng(8).normal(size=(4,c['channels'],c['samples'])).astype('float32')
    ck=check_causality(m,x,d,torch.device('cpu'),randomize=True)
    assert ck['status']=='PASSED'
    assert all(r['E_future'] is None for r in ck['rows'] if r['m']==c['samples'])
    assert all(r['positive_leak_error']>1e-4 for r in ck['rows'] if r['m']<c['samples'])

def test_real_future_leak_is_detected():
    m=make_model(3,2,1000,32,'reader').eval();original=m.boundary_curve
    def leaky(x,lens):
        out=original(x,lens)
        for n in lens:
            if n<x.shape[-1]:out[n]=out[n]+x[:,0,n][:,None]
        return out
    # A class method is necessary because the suite copies the model.
    class Leak(type(m)):
        def boundary_curve(self,x,lens):
            out=super().boundary_curve(x,lens)
            for n in lens:
                if n<x.shape[-1]:out[n]=out[n]+x[:,0,n][:,None]
            return out
    m.__class__=Leak
    x=np.random.default_rng(8).normal(size=(4,3,1000)).astype('float32')
    ck=check_causality(m,x,'2b',torch.device('cpu'),randomize=True)
    assert ck['status']=='FAILED'

def test_statistics():
    assert normalized_area(np.array([[50.,60.,80.]]),np.array([1.,2.,4.]))[0]==pytest.approx(65.)
    assert exact_signflip(np.array([0.,0.,0.]))==1
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])

def test_plan_tamper(tmp_path):
    p={'evidence':'synthetic'};p['plan_hash']=digest(p);write_json(tmp_path/'study.json',p)
    assert verified_plan(tmp_path)['evidence']=='synthetic'
    p['evidence']='real_fresh';write_json(tmp_path/'study.json',p)
    with pytest.raises(ValueError):verified_plan(tmp_path)

def test_missing_panel_rejected():
    df=pd.DataFrame(columns=['subject','seed','model','n_samples'])
    with pytest.raises(ValueError):paired_panel(df,{'subjects':[1,2],'fs':250},[2025],['reader','compact'],[250,1000])

def test_missing_results_do_not_fill_paper(tmp_path):
    (tmp_path/'code').mkdir();write_json(tmp_path/'code_hashes.json',{})
    write_json(tmp_path/'bite_provenance.json',{'file_hashes':{}})
    plan={'evidence':'real_fresh','models':['reader'],'recipe':RECIPE,'tasks':[{'dataset':'2a','subject':1,'seed':2025}]}
    plan['plan_hash']=digest(plan);write_json(tmp_path/'study.json',plan)
    result=collect(tmp_path)
    assert result['status']=='INCOMPLETE' and not (tmp_path/'paper').exists() and (tmp_path/'return_outputs.zip').exists()
