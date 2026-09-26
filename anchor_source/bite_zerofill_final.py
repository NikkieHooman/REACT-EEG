#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--study", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--device", default="cuda")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--bootstrap", type=int, default=10000)
p.add_argument("--bootstrap-seed", type=int, default=170926)
args = p.parse_args()

STUDY = args.study.resolve()
sys.path.insert(0, str(STUDY / "code"))

from fresh.data import SPECS, load_role, normalize, grid
from fresh.run import model_for

device = torch.device(args.device)
plan = json.loads((STUDY / "study.json").read_text())
args.out.mkdir(parents=True, exist_ok=True)

def load_ckpt(p):
    return torch.load(p, map_location="cpu", weights_only=True)

def load_scaler(p):
    with np.load(p, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

@torch.inference_mode()
def zerofill_curve(model, x, y, lengths):
    model.eval()
    acc = []
    for m in lengths:
        ncorrect = 0
        ntotal = 0
        for start in range(0, len(x), args.batch_size):
            xb = torch.from_numpy(
                np.ascontiguousarray(x[start:start+args.batch_size])
            ).to(device)
            xb = xb.clone()
            if m < xb.shape[-1]:
                xb[:,:,m:] = 0.0
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy()
            yy = y[start:start+len(pred)]
            ncorrect += int((pred == yy).sum())
            ntotal += len(pred)
        acc.append(100*ncorrect/ntotal)
    return np.asarray(acc,float)

def nauc(a,lengths,fs):
    t=np.asarray(lengths,float)/fs
    return float(np.trapezoid(a,t)/(t[-1]-t[0]))

rows=[]
for ds in ("2a","2b","ssvep"):
    cfg=SPECS[ds]
    lengths=list(grid(ds))
    for subject in range(1,cfg["subjects"]+1):
        for seed in plan["seeds"]:
            run=STUDY/"runs"/f"{ds}_S{subject:02d}_seed{seed}"
            scaler=load_scaler(run/"scaler.npz")
            raw,y,ids,_=load_role(
                plan["data_roots"][ds],ds,subject,"test"
            )
            x=normalize(raw,scaler)

            model=model_for("bite",ds,seed,STUDY).to(device)
            ck=load_ckpt(run/"bite"/"final.pt")
            model.load_state_dict(ck["state_dict"],strict=True)
            model.eval()

            curve=zerofill_curve(model,x,y,lengths)

            stored=json.loads((run/"bite"/"result.json").read_text())
            stored_endpoint=float(stored["accuracy_percent"][-1])

            if abs(curve[-1]-stored_endpoint)>1e-6:
                raise AssertionError(
                    f"{ds} S{subject} seed{seed}: endpoint mismatch "
                    f"{curve[-1]} vs {stored_endpoint}"
                )

            react=json.loads((run/"reader"/"result.json").read_text())
            rcurve=np.asarray(react["accuracy_percent"],float)

            rows.append({
                "dataset":ds,
                "subject":subject,
                "seed":seed,
                "bite_zerofill_early":float(curve[0]),
                "bite_zerofill_endpoint":float(curve[-1]),
                "bite_zerofill_nauc":nauc(curve,lengths,cfg["fs"]),
                "react_early":float(rcurve[0]),
                "react_endpoint":float(rcurve[-1]),
                "react_nauc":nauc(rcurve,lengths,cfg["fs"]),
            })

            print(
                ds,subject,seed,
                "BiTE zero early",round(curve[0],2),
                "nAUC",round(nauc(curve,lengths,cfg["fs"]),2)
            )

            del model
            if device.type=="cuda":
                torch.cuda.empty_cache()

rng=np.random.default_rng(args.bootstrap_seed)
summary={}

for ds in ("2a","2b","ssvep"):
    dsrows=[r for r in rows if r["dataset"]==ds]
    subjects=sorted({r["subject"] for r in dsrows})
    sub=[]

    for s in subjects:
        r=[q for q in dsrows if q["subject"]==s]
        sub.append({
            k:float(np.mean([q[k] for q in r]))
            for k in [
                "bite_zerofill_early",
                "bite_zerofill_endpoint",
                "bite_zerofill_nauc",
                "react_early",
                "react_endpoint",
                "react_nauc",
            ]
        })

    def report(stat):
        d=np.asarray([
            r[f"react_{stat}"]-r[f"bite_zerofill_{stat}"]
            for r in sub
        ])
        ix=rng.integers(0,len(d),(args.bootstrap,len(d)))
        boot=d[ix].mean(1)
        return {
            "mean":float(d.mean()),
            "ci95":[
                float(np.percentile(boot,2.5)),
                float(np.percentile(boot,97.5))
            ],
            "subject_differences":d.tolist()
        }

    summary[ds]={
        "react_early":float(np.mean([r["react_early"] for r in sub])),
        "bite_zerofill_early":float(np.mean([r["bite_zerofill_early"] for r in sub])),
        "react_nauc":float(np.mean([r["react_nauc"] for r in sub])),
        "bite_zerofill_nauc":float(np.mean([r["bite_zerofill_nauc"] for r in sub])),
        "react_minus_bite_zerofill_early":report("early"),
        "react_minus_bite_zerofill_nauc":report("nauc"),
    }

(args.out/"bite_zerofill_summary.json").write_text(
    json.dumps(summary,indent=2)+"\n"
)

print("\nFINAL")
print(json.dumps(summary,indent=2))
