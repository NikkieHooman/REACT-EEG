#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

p=argparse.ArgumentParser()
p.add_argument("--study",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
p.add_argument("--task-index",type=int,required=True)
p.add_argument("--device",default="cuda")
args=p.parse_args()

STUDY=args.study.resolve()
sys.path.insert(0,str(STUDY/"code"))

from fresh.data import SPECS,load_role,normalize,grid
from fresh.models import Tokenizer,tcn,Classifier,make_model
from fresh.run import RECIPE,train_one
from fresh.common import digest,state_sha,write_json

class ReverseOnly(nn.Module):
    """Separately trained start-anchored reverse reader.

    Initialization is copied exactly from a freshly constructed REACT model:
    tokenizer <- REACT tokenizer
    reverse_reader <- REACT second_reader
    classifier <- REACT classifier

    The forward reader and gate are not retained.
    """
    def __init__(self, reference):
        super().__init__()
        self.tokenizer = copy.deepcopy(reference.tokenizer)
        self.reverse_reader = copy.deepcopy(reference.second_reader)
        self.classifier = copy.deepcopy(reference.classifier)

    def forward(self, x):
        z = self.tokenizer(x)
        b = self.reverse_reader(z.flip(-1))[:, :, -1]
        return self.classifier(b)

    @torch.no_grad()
    def project_constraints(self):
        self.tokenizer.project_constraints()
        self.classifier.project_constraints()


def same_state(a,b,name):
    sa=a.state_dict();sb=b.state_dict()
    if sa.keys()!=sb.keys():
        raise AssertionError(name+" state keys differ")
    for k in sa:
        if not torch.equal(sa[k].cpu(),sb[k].cpu()):
            raise AssertionError(name+" initialization differs at "+k)

plan=json.loads((STUDY/"study.json").read_text())

tasks=[]
for ds in ("2a","2b","ssvep"):
    for subject in range(1,SPECS[ds]["subjects"]+1):
        for seed in plan["seeds"]:
            tasks.append((ds,subject,seed))

if not 0 <= args.task_index < len(tasks):
    raise ValueError(f"task index must be 0..{len(tasks)-1}")

ds,subject,seed=tasks[args.task_index]
cfg=SPECS[ds]
device=torch.device(args.device)

run_orig=STUDY/"runs"/f"{ds}_S{subject:02d}_seed{seed}"
folder=args.out/"runs"/f"{ds}_S{subject:02d}_seed{seed}"/"reverse_only"
folder.mkdir(parents=True,exist_ok=True)

# Use the exact scaler fitted in the final fresh run.
with np.load(run_orig/"scaler.npz",allow_pickle=False) as f:
    scaler={k:f[k] for k in f.files}

train_raw,train_y,_,train_records=load_role(
    plan["data_roots"][ds],ds,subject,"train"
)
train=normalize(train_raw,scaler)

# Construct the exact fresh REACT initialization first.
reference = make_model(
    cfg["channels"], cfg["classes"], cfg["samples"],
    cfg["pool"], "reader", seed
)
reference.project_constraints()

# Clone exactly the components belonging to the Reverse-only arm.
model = ReverseOnly(reference)

# Components were copied after the reference constraints were applied.
# Do not project again before exact initialization verification.
same_state(model.tokenizer, reference.tokenizer, "tokenizer")
same_state(model.classifier, reference.classifier, "classifier")
same_state(model.reverse_reader, reference.second_reader, "reverse_reader")

del reference
model = model.to(device)

recipe=dict(RECIPE)

signature=digest({
    "study":str(STUDY),
    "plan_hash":plan["plan_hash"],
    "dataset":ds,
    "subject":subject,
    "seed":seed,
    "arm":"trained_reverse_only",
    "epochs":recipe["epochs"],
    "objective":"endpoint_cross_entropy_only",
    "reverse_seed":918273,
})

initial=state_sha(model)

ck=train_one(
    model,
    train,
    train_y,
    folder,
    signature,
    seed,
    recipe,
    device,
    initial,
    initial,
    evidence="reviewer_trained_reverse_only",
)

model.load_state_dict(ck["state_dict"],strict=True)
model.eval()

# Held role is opened only after training.
test_raw,test_y,test_ids,test_records=load_role(
    plan["data_roots"][ds],ds,subject,"test"
)
test=normalize(test_raw,scaler)
lengths=list(grid(ds))

logits=[]
with torch.inference_mode():
    for m in lengths:
        vals=[]
        for start in range(0,len(test),64):
            xb=torch.from_numpy(
                np.ascontiguousarray(
                    test[start:start+64,:,:m]
                )
            ).to(device)
            vals.append(model(xb).cpu().numpy())
        logits.append(np.concatenate(vals))

logits=np.stack(logits,axis=1)
accuracy=100*(logits.argmax(-1)==test_y[:,None]).mean(0)

np.savez_compressed(
    folder/"predictions.npz",
    logits=logits,
    y=test_y,
    ids=test_ids,
    lengths=np.asarray(lengths)
)

write_json(folder/"result.json",{
    "status":"complete",
    "arm":"trained_reverse_only",
    "dataset":ds,
    "subject":subject,
    "seed":seed,
    "epochs":600,
    "prefix_weight":0.0,
    "lengths":lengths,
    "accuracy_percent":accuracy.tolist(),
    "parameters":sum(p.numel() for p in model.parameters()),
    "initialization_contract":
        "tokenizer, classifier, and reverse reader copied exactly from "
        "the frozen fresh REACT initialization; forward reader and gate omitted",
})

print(
    "COMPLETE",
    ds,subject,seed,
    "early",float(accuracy[0]),
    "endpoint",float(accuracy[-1]),
    "params",sum(p.numel() for p in model.parameters()),
    flush=True
)
