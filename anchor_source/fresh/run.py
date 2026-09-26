"""Train/evaluate the NEW matched cohort. Existing runs are never read or overwritten."""
from __future__ import annotations
import argparse, copy, fcntl, json, os, random, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import (array_sha, checked_logits, digest, environment, file_sha,
                     read_json, seed_all, state_sha, sync, write_json, verified_plan)
from .data import SPECS, load_role, fit_scaler, normalize, check_role_overlap, grid
from .models import make_model
from .bite import BiTEWrapper

MODELS=['compact','reader','ff','compact_mean','bite']
RECIPE=dict(epochs=600,lr=.002,weight_decay=.002,batch_size=64,label_smoothing=.1,
            grad_clip=5.,save_every=25,betas=[.9,.999],eps=1e-8,prefix_weight=0.,
            maxnorm='after_optimizer_step_for_four_core_models; upstream_BiTE_unchanged',
            augmentation='none',early_stopping=False,selection='fixed epoch 600')

def model_for(kind,dataset,seed,study):
    if kind=='bite':return BiTEWrapper(Path(study)/'third_party/BiteEEG',dataset,seed)
    c=SPECS[dataset]
    return make_model(channels=c['channels'],classes=c['classes'],max_samples=c['samples'],pool=c['pool'],kind=kind,seed=seed)

def constrain(model):
    if hasattr(model,'project_constraints'):model.project_constraints()

def shared_sha(model):
    import hashlib
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        if name.startswith(('tokenizer.','forward_reader.','classifier.')):
            h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def rng_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(),numpy=[n[0],n[1].tolist(),*n[2:]])

def set_rng(r):
    torch.set_rng_state(r['torch'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])
    random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],np.asarray(n[1],np.uint32),*n[2:]))

def atomic_torch(path,payload):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)

def weights_load(path):
    return torch.load(path,map_location='cpu',weights_only=True)

def code_identity(study):
    root=Path(study)/'code'
    expected=read_json(Path(study)/'code_hashes.json')
    for rel,sha in expected.items():
        if file_sha(root/rel)!=sha:raise ValueError('Source changed after freezing: '+rel)
    provenance=read_json(Path(study)/'bite_provenance.json')
    baseline=Path(study)/'third_party/BiteEEG'
    for rel,sha in provenance.get('file_hashes',{}).items():
        if file_sha(baseline/rel)!=sha:
            raise ValueError('Pinned BiTE source changed: '+rel)
    return digest(expected)

def train_one(model,train,y,folder,signature,seed,recipe,device,initial_hash,init_shared,
              evidence='real_fresh',interrupt_after=None):
    """Only accepts training arrays. Held data cannot be accessed through this API."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    final=folder/'final.pt';last=folder/'last.pt'
    if final.exists():
        c=weights_load(final)
        if c['signature']!=signature or c['epoch']!=recipe['epochs']:
            raise ValueError('Existing final checkpoint belongs to a different run')
        model.load_state_dict(c['state_dict'],strict=True);model.eval();return c
    seed_all(seed+10000)
    opt=torch.optim.Adam(model.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay'],
                         betas=tuple(recipe['betas']),eps=recipe['eps'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=recipe['epochs'],eta_min=0.)
    start=0;history=[];orders=[];wall=0.
    env=environment(device)
    if last.exists():
        ck=weights_load(last)
        if ck['signature']!=signature:raise ValueError('Resume signature differs')
        for k in ('torch','numpy','cuda_runtime','device_name'):
            if ck['environment'].get(k)!=env.get(k):
                raise ValueError('Resume environment differs at '+k+'; use the original environment/device type')
        model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['epoch'];history=ck['history'];orders=ck['orders'];wall=ck['training_seconds'];set_rng(ck['rng'])
        initial_hash=ck['initial_hash'];init_shared=ck['initial_shared_hash']
    xcpu=torch.from_numpy(train);ycpu=torch.from_numpy(y);batch=recipe['batch_size'];t0=time.perf_counter()
    for epoch in range(start+1,recipe['epochs']+1):
        model.train();gen=torch.Generator().manual_seed(seed+20000+epoch)
        order=torch.randperm(len(y),generator=gen)
        orders.append(array_sha(order.numpy()))
        loss_total=0.;n=0;lr=float(opt.param_groups[0]['lr'])
        for off in range(0,len(order),batch):
            ix=order[off:off+batch];x=xcpu[ix].to(device);target=ycpu[ix].to(device)
            opt.zero_grad(set_to_none=True)
            out=checked_logits(model(x),len(x),int(y.max())+1)
            loss=F.cross_entropy(out,target,label_smoothing=recipe['label_smoothing'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['grad_clip'],error_if_nonfinite=True)
            opt.step();constrain(model)
            loss_total+=float(loss.detach())*len(ix);n+=len(ix)
        sched.step();history.append(dict(epoch=epoch,train_loss=loss_total/n,lr_used=lr,n_examples=n))
        if epoch==1 or epoch%25==0 or epoch==recipe['epochs']:
            print(folder.name,'epoch',epoch,'loss',round(loss_total/n,6),flush=True)
            write_json(folder/'progress.json',dict(status='TRAINING',epoch=epoch,epochs=recipe['epochs'],signature=signature))
        if epoch%recipe['save_every']==0 or epoch==recipe['epochs'] or epoch==interrupt_after:
            sync(device)
            ck=dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,
                    signature=signature,rng=rng_state(),initial_hash=initial_hash,initial_shared_hash=init_shared,
                    history=history,orders=orders,environment=env,training_seconds=wall+time.perf_counter()-t0,
                    evidence=evidence,prefix_weight=0.,test_data_opened=False)
            atomic_torch(last,ck);write_json(folder/'training_history.json',history)
        if epoch==interrupt_after:return ck
    model.eval();constrain(model)
    # Freeze any upstream forward-time max-norm projection before final checkpoint.
    with torch.inference_mode():
        probe=model(xcpu[:2].to(device)).detach().cpu()
    ck=dict(state_dict=model.state_dict(),epoch=recipe['epochs'],signature=signature,
            initial_hash=initial_hash,initial_shared_hash=init_shared,orders=orders,
            training_seconds=wall+time.perf_counter()-t0,environment=env,evidence=evidence,
            prefix_weight=0.,test_data_opened=False,training_probe_logits=probe)
    atomic_torch(final,ck)
    write_json(folder/'progress.json',dict(status='TRAINED',epoch=recipe['epochs'],signature=signature,
            checkpoint_sha256=file_sha(final),parameters=sum(p.numel() for p in model.parameters())))
    return ck

@torch.inference_mode()
def predict(model,x,lengths,device,batch=64):
    model.eval();cols=[]
    for m in lengths:
        values=[]
        for start in range(0,len(x),batch):
            inp=torch.from_numpy(np.ascontiguousarray(x[start:start+batch,:,:m])).to(device)
            logits=model(inp)
            if logits.ndim!=2 or not torch.isfinite(logits).all():raise ValueError('Invalid logits')
            values.append(logits.cpu().numpy())
        cols.append(np.concatenate(values))
    return np.stack(cols,axis=1)

@torch.inference_mode()
def diagnostics(model,x,y,device,batch=64):
    sums={};pred_f=[];pred_b=[];g=None
    for start in range(0,len(x),batch):
        part=torch.from_numpy(x[start:start+batch]).to(device)
        d=model.branch_diagnostics(part);g=d['gate'].cpu().numpy()
        for name in ('forward_norm_mean','reverse_norm_mean','weighted_forward_norm_mean','weighted_reverse_norm_mean','branch_cosine_mean'):
            sums[name]=sums.get(name,0.)+float(d[name])*len(part)
        sums['in_prefix_context_intervention_max']=max(sums.get('in_prefix_context_intervention_max',0.),float(d['in_prefix_context_intervention_max']))
        pred_f.extend(d['forward_only_posthoc_logits'].argmax(1).cpu().tolist());pred_b.extend(d['reverse_only_posthoc_logits'].argmax(1).cpu().tolist())
    for name in list(sums):
        if name.endswith('_mean'):sums[name]/=len(x)
    return {**sums,'gate':g.tolist(),'posthoc_forward_accuracy':float(np.mean(np.asarray(pred_f)==y))*100,
            'posthoc_reverse_accuracy':float(np.mean(np.asarray(pred_b)==y))*100,
            'interpretation':'shared-classifier interventions, not trained baselines'}

@torch.inference_mode()
def measure_timing(model,x,lengths,device,repetitions=50,warmup=10):
    model.eval();inp=torch.from_numpy(x[:1]).to(device)
    def measure(fn):
        for _ in range(warmup):fn()
        sync(device);times=[]
        for _ in range(repetitions):
            sync(device);t=time.perf_counter();fn();sync(device);times.append((time.perf_counter()-t)*1000)
        return dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)))
    points=[lengths[0],lengths[-1]]
    out=dict(single_prefix={str(m):measure(lambda m=m:model(inp[:,:,:m])) for m in points},
             repetitions=repetitions,warmup=warmup,batch_size=1,
             scope='device-resident normalized EEG -> logits, includes BiTE STFT, excludes file I/O, transfer, acquisition and normalization',
             environment=environment(device))
    if hasattr(model,'boundary_curve'):
        out['shared_feature_full_curve']=measure(lambda:model.boundary_curve(inp,lengths))
        out['naive_recomputed_full_curve']=measure(lambda:[model(inp[:,:,:m]) for m in lengths])
    return out

@torch.inference_mode()
def check_causality(model,x,dataset,device,seed=8173,tolerance=1e-4,randomize=False):
    """Independent cached-feature path vs direct cropped-input path on a model copy."""
    model=copy.deepcopy(model).eval();cfg=SPECS[dataset]
    lengths=([32,64,128,250,256,500,512,750,768,992,1000] if dataset in ('2a','2b') else [4,8,64,65,127,128,192,256])
    gen=torch.Generator().manual_seed(seed)
    if randomize:
        for reader in (model.forward_reader,model.second_reader):
            if reader is None:continue
            for mod in reader.modules():
                if isinstance(mod,torch.nn.Conv1d):
                    mod.weight.copy_(torch.randn(mod.weight.shape,generator=gen).to(device)*.05)
                if isinstance(mod,torch.nn.BatchNorm1d):
                    mod.reset_running_stats();mod.weight.fill_(1.);mod.bias.zero_()
    before=state_sha(model);rows=[];traces=[]
    real=torch.from_numpy(x[:4]).to(device)
    synthetic=torch.randn((4,cfg['channels'],cfg['samples']),generator=gen).to(device)
    for kind,inp in [('prepared_eeg',real),('synthetic',synthetic)]:
        trace=[];hook=None
        if model.second_reader is not None:
            hook=model.second_reader.register_forward_pre_hook(lambda module,args:trace.append(args[0].shape[-1]))
        full=model.boundary_curve(inp,lengths)
        if hook:hook.remove()
        expected=[(m+cfg['pool']-1)//cfg['pool'] for m in lengths]
        if model.second_reader is not None and trace!=expected:raise AssertionError('Reverse token boundary trace mismatch')
        traces.append(dict(input_kind=kind,token_counts=trace,expected=expected))
        for m in lengths:
            direct=model(inp[:,:,:m].clone());rep=model(inp[:,:,:m].clone())
            errors=[float((direct-full[m]).abs().max()),float((rep-direct).abs().max()),
                    float((model(inp[:1,:,:m])-direct[:1]).abs().max())]
            future=None;positive=None
            if m<inp.shape[-1]:
                changed=inp.clone();changed[:,:,m:]=torch.randn(changed[:,:,m:].shape,generator=gen).to(device)*1000
                updated=model.boundary_curve(changed,[m])[m]
                future=float((updated-full[m]).abs().max())
                # Deterministic injected leak uses the first unobserved value.
                a=full[m].clone();b=updated.clone();a[:,0]+=inp[:,0,m];b[:,0]+=changed[:,0,m]
                positive=float((a-b).abs().max())
            rows.append(dict(input_kind=kind,m=m,partial=bool(m%cfg['pool']),E_trunc=errors[0],E_repeat=errors[1],
                      E_batch=errors[2],E_future=future,positive_leak_error=positive,
                      passed=max(errors+[0. if future is None else future])<=tolerance and (positive is None or positive>tolerance)))
    passed=all(r['passed'] for r in rows) and before==state_sha(model)
    return dict(status='PASSED' if passed else 'FAILED',randomized_copy=randomize,threshold=tolerance,
                rows=rows,traces=traces,state_unchanged=before==state_sha(model),n_eeg=len(real),n_synthetic=4,
                scope='finite prepared-input tests, not verification of raw-signal acquisition/preprocessing')

def group_task(study,index,device='cuda'):
    study=Path(study);plan=verified_plan(study);cell=plan['tasks'][index]
    dataset,subject,seed=cell['dataset'],cell['subject'],cell['seed'];cfg=SPECS[dataset]
    directory=study/'runs'/('%s_S%02d_seed%d'%(dataset,subject,seed));directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    seed_all(seed);device=torch.device(device);torch.set_num_threads(2)
    recipe=plan['recipe'];evidence=plan['evidence']
    if evidence=='real_fresh' and (recipe['epochs']!=600 or recipe['prefix_weight']!=0.):raise ValueError('Frozen real study requires endpoint-only 600 epochs')
    codehash=code_identity(study)
    data_root=plan['data_roots'][dataset]
    train_raw,train_y,train_ids,train_records=load_role(data_root,dataset,subject,'train')
    if len(np.unique(train_y))!=cfg['classes']:raise ValueError('Training role does not include every declared class')
    normalizer=fit_scaler(train_raw);train=normalize(train_raw,normalizer)
    normalhash=digest(normalizer)
    np.savez(directory/'scaler.npz',**normalizer)
    write_json(directory/'training_data.json',dict(files=train_records,prepared_x_sha256=array_sha(train),
        normalizer_hash=normalhash,fit_role='train',id_scope='source filename + prepared row ordinal; not raw-trial provenance',
        prepared_stage='existing arrays, first1000 MI samples, no new filtering/alignment'))
    trained={};shared={};orders={};signatures={}
    for kind in plan['models']:
        model=model_for(kind,dataset,seed,study).to(device);constrain(model)
        sig=digest(dict(plan_hash=plan['plan_hash'],dataset=dataset,subject=subject,seed=seed,model=kind,codehash=codehash,
                       train=train_records,train_x=array_sha(train),normalizer=normalhash,bite=read_json(study/'bite_provenance.json')['source_hash']))
        ih=state_sha(model);sh=shared_sha(model) if kind!='bite' else None
        ck=train_one(model,train,train_y,directory/kind,sig,seed,recipe,device,ih,sh,evidence=evidence)
        # Round-trip check against a training-only reference, never against old results.
        model.load_state_dict(ck['state_dict'],strict=True);model.eval()
        with torch.inference_mode():
            current=model(torch.from_numpy(train[:2]).to(device)).cpu()
        if not torch.allclose(current,ck['training_probe_logits'],atol=1e-6,rtol=1e-6):raise AssertionError('Final checkpoint round-trip differs')
        shared[kind]=ck['initial_shared_hash'];orders[kind]=digest(ck['orders']);trained[kind]=ck;signatures[kind]=sig
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    if len({shared[k] for k in plan['models'] if k!='bite'})!=1 or len(set(orders.values()))!=1:
        raise AssertionError('Shared initialization/minibatch order mismatch')
    write_json(directory/'pairing.json',dict(initial_shared_hashes=shared,minibatch_order_hashes=orders,passed=True))
    # Only now open the held role. It never enters optimizer/model-selection code.
    test_raw,test_y,test_ids,test_records=load_role(data_root,dataset,subject,'test')
    overlap=check_role_overlap(train_raw,test_raw);test=normalize(test_raw,normalizer)
    testhash=array_sha(test);labelhash=array_sha(test_y);idsha=array_sha(test_ids)
    write_json(directory/'held_data.json',dict(files=test_records,normalizer_hash=normalhash,fit_role='train',
              overlap_check=overlap,opened_after_all_models_trained=True,x_hash=testhash,labels_hash=labelhash,ids_hash=idsha))
    for kind in plan['models']:
        folder=directory/kind;statuspath=folder/'result.json'
        if statuspath.exists():
            result=read_json(statuspath)
            if result['signature']!=signatures[kind] or result['test_x_hash']!=testhash:
                raise ValueError('Existing result signature differs')
            continue
        model=model_for(kind,dataset,seed,study).to(device);model.load_state_dict(trained[kind]['state_dict'],strict=True);model.eval()
        lengths=grid(dataset) if kind!='bite' else [cfg['samples']]
        logits=predict(model,test,lengths,device);accuracy=100*(logits.argmax(-1)==test_y[:,None]).mean(0)
        np.savez_compressed(folder/'predictions.npz',logits=logits,y=test_y,ids=test_ids,lengths=np.asarray(lengths))
        cause=None
        if kind!='bite':
            cause=dict(trained=check_causality(model,test,dataset,device),
                       nondegenerate=check_causality(model,test,dataset,device,randomize=True))
            write_json(folder/'causality.json',cause)
        if kind=='reader':write_json(folder/'diagnostics.json',diagnostics(model,test,test_y,device))
        if seed==plan['seeds'][0] and subject==1:
            write_json(folder/'timing.json',measure_timing(model,test,lengths,device))
        result=dict(status='MEASURED',evidence=evidence,cell=cell,model=kind,signature=signatures[kind],
             checkpoint_sha256=file_sha(folder/'final.pt'),predictions_sha256=file_sha(folder/'predictions.npz'),
             epochs=recipe['epochs'],prefix_weight=0.,parameters=sum(p.numel() for p in model.parameters()),
             lengths=lengths,accuracy_percent=accuracy.tolist(),n_test=len(test_y),fs=cfg['fs'],
             n_correct=(logits.argmax(-1)==test_y[:,None]).sum(0).tolist(),
             train_data_hash=digest(train_records),test_files=test_records,test_x_hash=testhash,
             test_labels_hash=labelhash,test_ids_hash=idsha,normalizer_hash=normalhash,
             causal_checks_passed=None if cause is None else all(v['status']=='PASSED' for v in cause.values()),
             environment=environment(device))
        write_json(statuspath,result);print('MEASURED',dataset,subject,seed,kind,accuracy[-1],flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    write_json(directory/'group_status.json',dict(status='MEASURED',evidence=evidence,cell=cell,models=plan['models']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    plan=read_json(Path(args.study)/'study.json');cell=plan['tasks'][args.index]
    errorpath=Path(args.study)/'failures'/('task_%03d.json'%args.index)
    try:
        group_task(args.study,args.index,args.device)
        if errorpath.exists():errorpath.unlink()
    except Exception as exc:
        write_json(errorpath,dict(status='ERROR',cell=cell,error=repr(exc),traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()
