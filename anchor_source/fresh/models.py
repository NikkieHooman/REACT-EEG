"""NEW endpoint-only manuscript implementation. Not a historical checkpoint loader.

Explicit fresh-study choices: bias-free convolutions, projection/classifier biases,
BN eps=1e-5/momentum=.1, learned positions N(0,.02) sized to the declared maximum,
zero left padding, stride one, Xavier first residual convolution, zero second
residual convolution, no post-addition activation, post-step max-norm projection.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .controls import ReadoutControl


class CausalDepthwise(nn.Module):
    def __init__(self, width: int, kernel: int, dilation: int):
        super().__init__()
        self.left = (kernel-1)*dilation
        self.conv = nn.Conv1d(width, width, kernel, dilation=dilation, groups=width, bias=False)

    def forward(self,x):
        return self.conv(F.pad(x,(self.left,0)))


class Block(nn.Module):
    def __init__(self,width:int,dilation:int,dropout:float):
        super().__init__()
        self.c1=CausalDepthwise(width,6,dilation)
        self.bn1=nn.BatchNorm1d(width)
        self.c2=CausalDepthwise(width,6,dilation)
        self.bn2=nn.BatchNorm1d(width)
        self.drop=nn.Dropout(dropout)
        self.act=nn.LeakyReLU(.2)
        nn.init.xavier_uniform_(self.c1.conv.weight)
        nn.init.zeros_(self.c2.conv.weight)

    def forward(self,x):
        y=self.drop(self.act(self.bn1(self.c1(x))))
        y=self.drop(self.act(self.bn2(self.c2(y))))
        return x+y


def tcn(width=64,dropout=.3):
    return nn.Sequential(*(Block(width,d,dropout) for d in (1,2,4)))


class Tokenizer(nn.Module):
    def __init__(self,channels:int,max_samples:int,pool:int,width:int=64,dropout:float=.3):
        super().__init__()
        self.channels,self.pool,self.max_samples=channels,pool,max_samples
        self.kernels=(16,32,64)
        self.temporal=nn.ModuleList(nn.Conv2d(1,16,(1,k),bias=False) for k in self.kernels)
        self.bn_temporal=nn.BatchNorm2d(48)
        self.spatial=nn.Conv2d(48,96,(channels,1),groups=48,bias=False)
        self.bn_spatial=nn.BatchNorm2d(96)
        self.act=nn.LeakyReLU(.2)
        self.drop=nn.Dropout(dropout)
        self.project=nn.Linear(96,width,bias=True)
        self.position=nn.Parameter(torch.empty(1,width,math.ceil(max_samples/pool)))
        nn.init.normal_(self.position,std=.02)

    def sample_features(self,x):
        if x.ndim!=3 or x.shape[1]!=self.channels or not 1<=x.shape[-1]<=self.max_samples:
            raise ValueError("Invalid input shape/length")
        x=x.unsqueeze(1)
        y=torch.cat([conv(F.pad(x,(k-1,0))) for k,conv in zip(self.kernels,self.temporal)],dim=1)
        return self.act(self.bn_spatial(self.spatial(self.bn_temporal(y)))).squeeze(2)

    def from_features(self,u):
        m=u.shape[-1]
        n=(m+self.pool-1)//self.pool
        pad=n*self.pool-m
        # Sum then divide by actual counts, never by right-padded width.
        sums=F.pad(u,(0,pad)).reshape(u.shape[0],96,n,self.pool).sum(-1)
        counts=torch.full((n,),self.pool,device=u.device,dtype=u.dtype)
        counts[-1]=m-(n-1)*self.pool
        avg=sums/counts
        z=self.project(self.drop(avg).transpose(1,2)).transpose(1,2)
        return z+self.position[:,:,:n]

    def forward(self,x):
        return self.from_features(self.sample_features(x))

    @torch.no_grad()
    def project_constraints(self):
        w=self.spatial.weight
        norm=w.flatten(1).norm(dim=1,keepdim=True)
        w.mul_((1./norm.clamp_min(1e-12)).clamp(max=1.).view(-1,1,1,1))


class Classifier(nn.Linear):
    @torch.no_grad()
    def project_constraints(self):
        n=self.weight.norm(dim=1,keepdim=True)
        self.weight.mul_((.25/n.clamp_min(1e-12)).clamp(max=1.))


class ReferenceControl(ReadoutControl):
    def boundary_curve(self,x,lengths):
        if self.training:
            raise ValueError("Shared-feature boundary_curve is evaluation-only")
        # Independent path from direct model(x[:,:,:m]): feature computation
        # receives full input, then prefix-specific pooling and reverse routing.
        u=self.tokenizer.sample_features(x)
        zfull=self.tokenizer.from_features(u)
        ffull=self.forward_reader(zfull)
        out={}
        for m in lengths:
            if not 1<=m<=x.shape[-1]:
                raise ValueError("Boundary outside supplied input")
            n=(m+self.tokenizer.pool-1)//self.tokenizer.pool
            complete=m%self.tokenizer.pool==0 or m==x.shape[-1]
            z=zfull[:,:,:n] if complete else self.tokenizer.from_features(u[:,:,:m])
            fs=ffull[:,:,:n] if complete else self.forward_reader(z)
            f=fs[:,:,-1]
            if self.kind=="compact_mean":
                h=fs.mean(-1)
            elif self.kind=="compact":
                h=f
            else:
                b=self.second_reader(z.flip(-1) if self.kind=="reader" else z)[:,:,-1]
                g=self.gate_logits.sigmoid()
                h=(1-g)*f+g*b
            out[m]=self.classifier(h)
        return out


def make_model(channels:int,classes:int,max_samples:int,pool:int,kind:str="reader",
               seed:int=2025,extra_seed:int=918273,dropout:float=.3) -> ReferenceControl:
    # Construct ALL shared modules before extra branch creation.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        tokenizer=Tokenizer(channels,max_samples,pool,64,dropout)
        forward=tcn(64,dropout)
        classifier=Classifier(64,classes)
    return ReferenceControl(tokenizer,forward,classifier,lambda:tcn(64,dropout),kind,64,extra_seed)
