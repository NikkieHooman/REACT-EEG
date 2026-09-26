"""Pinned upstream BiTE, fetched to a separate directory, never relabelled historical.
The calling convention/STFT follows the supplied BIBM_V2 baseline adapter.
BiTE source is not redistributed. Fresh-study provenance records all source hashes.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
from torch import nn
from .data import SPECS
from .common import seed_all, checked_logits
COMMIT='924eb32241ba1a7c80dbc4ba097f8c979da17578'
URL='https://github.com/cindy-hong/BiteEEG.git'

class BiTEWrapper(nn.Module):
    def __init__(self, root, dataset, seed):
        super().__init__()
        self.dataset=dataset
        p=Path(root)/'model/BiTE.py'
        if not p.is_file():raise FileNotFoundError(p)
        # Preserve upstream imports if the pinned file refers to sibling modules.
        import sys
        sys.path.insert(0,str(Path(root).resolve()))
        spec=importlib.util.spec_from_file_location('reader_fresh_upstream_bite',p)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        c=SPECS[dataset];k,pool,lo,hi=(32,8,8,64) if dataset=='ssvep' else (64,32,4,40)
        self.kernel,self.rate,self.lo,self.hi=k,c['fs'],lo,hi
        self.classes=c['classes']
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.inner=mod.BiTE(n_channels=c['channels'],n_classes=c['classes'],sample_points=c['samples'],
                F1=16,kernel_size1=k,D=4,pk1=pool,tcn_k_size=6,BiTCN=True,dropout=.3,
                use_time=True,use_freq=True,use_att=True,sampling_rate=c['fs'],low_cut=lo,high_cut=hi)

    def spectrum(self,x):
        b,c,t=x.shape;k=self.kernel
        win=torch.hann_window(k,device=x.device,dtype=x.dtype)
        stft=torch.stft(x.reshape(b*c,t),n_fft=k,hop_length=1,win_length=k,window=win,
            center=True,pad_mode='constant',normalized=False,onesided=True,return_complex=True).abs()
        stft=stft.reshape(b,c,k//2+1,stft.shape[-1])
        begin=int(self.lo*k/self.rate);end=int(self.hi*k/self.rate)+1
        return stft[:,:,begin:end].permute(0,2,3,1).contiguous()

    def forward(self,x):
        # No full-trial spectral slice is substituted for a prefix.
        out=self.inner(x,self.spectrum(x))
        if not isinstance(out,(tuple,list)) or len(out)<1:
            raise TypeError('Pinned BiTE calling convention changed; expected logits,aux')
        return checked_logits(out[0],len(x),self.classes)
