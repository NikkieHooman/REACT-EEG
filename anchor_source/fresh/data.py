"""Explicit NEW-study reader for the recovered prepared NPZ role layout.

No original model imports, no historical-checkpoint claims, no label inference.
Matches the recovered audit/80 loader's bite_within training-only coordinate
StandardScaler. Raw-to-prepared provenance is recorded separately, not inferred.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from .common import array_sha, file_sha

SPECS = {
 '2a': dict(subjects=9, channels=22, classes=4, samples=1000, fs=250, pool=32),
 '2b': dict(subjects=9, channels=3, classes=2, samples=1000, fs=250, pool=32),
 'ssvep': dict(subjects=10, channels=8, classes=12, samples=256, fs=256, pool=4),
}
DEFAULT_ROOTS = {
 '2a': '/lustre/smuexa01/client/users/nikkieh/bibm_data/bci/2a/2a_pre',
 '2b': '/lustre/smuexa01/client/users/nikkieh/bibm_data/bci/2b/2b_pre',
 'ssvep': '/lustre/smuexa01/client/users/nikkieh/bibm_data/sd_ssvep/processed_kaggle_exact',
}

def grid(dataset):
    return (sorted(set(range(256,993,32)) | {250,500,750,1000})
            if dataset in ('2a','2b') else list(range(64,257,4)))

def paths(root, dataset, subject, role):
    if dataset not in SPECS or role not in ('train','test'):
        raise ValueError('Unknown dataset/role')
    if not 1 <= int(subject) <= SPECS[dataset]['subjects']:
        raise ValueError('Subject outside declared cohort')
    root=Path(root); subject=int(subject)
    if dataset=='2a':
        return [root / ('A0%d%s.npz' % (subject, 'T' if role=='train' else 'E'))]
    if dataset=='2b':
        return [root / ('B0%d0%d%s.npz' % (subject,s,'T' if role=='train' else 'E'))
                for s in ((1,2,3) if role=='train' else (4,5))]
    return [root / ('S%02d_%s.npz' % (subject,role))]

def load_role(root, dataset, subject, role):
    cfg=SPECS[dataset]; xs=[]; ys=[]; ids=[]; records=[]
    for path in paths(root,dataset,subject,role):
        with np.load(path, allow_pickle=False) as f:
            if not {'data','label'} <= set(f.files):
                raise ValueError(str(path)+': expected data and label')
            raw=f['data']; label=f['label']
            # Do not cast noninteger labels into apparently valid integers.
            if label.dtype.kind not in 'iu' or label.ndim!=1:
                raise ValueError(str(path)+': labels must be 1-D integers, zero based')
            if raw.ndim!=3 or raw.shape[0]!=len(label) or raw.shape[1]!=cfg['channels']:
                raise ValueError(str(path)+': unexpected data/label shape '+str(raw.shape))
            allowed=(1000,1001) if dataset in ('2a','2b') else (256,)
            if raw.shape[-1] not in allowed:
                raise ValueError(str(path)+': unexpected sample count '+str(raw.shape[-1]))
            # The original MI _npz loader keeps the first 1000 samples explicitly.
            x=np.asarray(raw[:,:,:cfg['samples']],dtype=np.float64)
            y=np.asarray(label,dtype=np.int64)
            if len(y)<2 or np.any(y<0) or np.any(y>=cfg['classes']) or not np.isfinite(x).all():
                raise ValueError(str(path)+': empty/nonfinite/invalid labels')
            record={'path':str(path.resolve()),'sha256':file_sha(path),'stored_shape':list(raw.shape),
                    'used_shape':list(x.shape),'array_sha256':array_sha(x),'labels_sha256':array_sha(y)}
        xs.append(x);ys.append(y);records.append(record)
        ids.extend(['%s#row%06d'%(path.name,i) for i in range(len(y))])
    x=np.concatenate(xs);y=np.concatenate(ys)
    return x,y,np.asarray(ids,dtype='U100'),records

def fit_scaler(x):
    scaler=StandardScaler().fit(x.reshape(len(x),-1))
    C,T=x.shape[1:]
    return {'mean':scaler.mean_.reshape(C,T),'scale':scaler.scale_.reshape(C,T),
            'var':scaler.var_.reshape(C,T),'n_train':int(scaler.n_samples_seen_)}

def normalize(x,state):
    m=x.shape[-1]
    mean=state['mean'][:,:m];scale=state['scale'][:,:m]
    if x.shape[1:]!=mean.shape or np.any(scale<=0):
        raise ValueError('Frozen scaler dimensions/scale do not match prefix')
    out=((x-mean[None])/scale[None]).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('Nonfinite normalized input')
    return out

def trial_fingerprints(x):
    return [array_sha(row) for row in x]

def check_role_overlap(train_x,test_x):
    # Exact prepared-array duplicates only; absence does not establish original
    # trial/block independence. Source-file ordinal IDs are NOT raw trial IDs.
    overlap=set(trial_fingerprints(train_x)) & set(trial_fingerprints(test_x))
    if overlap:raise ValueError('Exact prepared EEG trial duplicates across train/test roles: '+str(len(overlap)))
    return {'exact_duplicate_trials':0,'scope':'exact prepared-array duplication, not proof of block independence'}
