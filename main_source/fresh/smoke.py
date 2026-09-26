"""Training-role-only readiness test. Never evaluates a held label or checkpoint."""
from __future__ import annotations
import argparse, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import environment,read_json,write_json,seed_all,file_sha,verified_plan
from .data import SPECS,load_role,fit_scaler,normalize,grid
from .run import model_for,constrain,shared_sha,check_causality,code_identity


def smoke(study,device='cuda'):
    study=Path(study);p=verified_plan(study);device=torch.device(device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Real HPC preflight requires an allocated CUDA GPU')
    # Check analysis dependencies before authorizing the expensive training array.
    from . import report as report_module
    import scipy, pandas, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure();plt.plot([0,1],[0,1]);plt.title('SOFTWARE BACKEND CHECK -- NOT EEG RESULTS')
    fig.savefig(study/'plot_backend_check.png');plt.close(fig)
    code_identity(study);seed_all(2025);torch.set_num_threads(2);rows=[]
    for d,c in SPECS.items():
        raw,y,ids,info=load_role(p['data_roots'][d],d,1,'train')
        sc=fit_scaler(raw);train=normalize(raw,sc)
        hashes=[];params={}
        for kind in p['models']:
            seed_all(2025);m=model_for(kind,d,2025,study).to(device);constrain(m)
            if kind!='bite':hashes.append(shared_sha(m))
            params[kind]=sum(q.numel() for q in m.parameters())
            opt=torch.optim.Adam(m.parameters(),lr=.002)
            for j in range(2):
                ix=np.arange(j*8,(j+1)*8)%len(y);m.train();opt.zero_grad(set_to_none=True)
                out=m(torch.from_numpy(train[ix]).to(device));loss=F.cross_entropy(out,torch.from_numpy(y[ix]).to(device))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training smoke loss')
                loss.backward();opt.step();constrain(m)
            m.eval()
            if kind!='bite':
                test=check_causality(m,train,d,device)
                if test['status']!='PASSED':
                    write_json(study/('smoke_failure_%s_%s.json'%(d,kind)),test);raise AssertionError('Training-only causality smoke failed')
            rows.append(dict(dataset=d,model=kind,parameters=params[kind],training_steps=2))
            del m,opt
            torch.cuda.empty_cache()
        if len(set(hashes))!=1:raise AssertionError('Initial common modules did not match')
        if params['reader']-params['compact']!=3136 or params['ff']!=params['reader'] or params['compact_mean']!=params['compact']:
            raise AssertionError('Parameter-control relationship differs')
    result=dict(status='SMOKE_PASSED',scope='training-role and synthetic checks only; not scientific results',analysis_dependencies_checked=True,rows=rows,environment=environment(device))
    write_json(study/'smoke.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    try:print(smoke(a.study))
    except Exception as e:
        write_json(Path(a.study)/'smoke.json',dict(status='ERROR',error=repr(e),traceback=traceback.format_exc()))
        raise
