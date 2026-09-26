#!/usr/bin/env python3
import itertools, json
from pathlib import Path
import numpy as np

ORIG=Path(
"/lustre/smuexa01/client/users/nikkieh/reader_fresh_runs/"
"study_20260918T005712Z_qDm6ij"
)

plan=json.loads((ORIG/"study.json").read_text())
models=["compact","ff","bite"]
datasets=["2a","2b","ssvep"]

B=10000
rng=np.random.default_rng(170926)

def endpoint(ds,s,seed,model):
    p=ORIG/"runs"/f"{ds}_S{s:02d}_seed{seed}"/model/"result.json"
    x=json.loads(p.read_text())
    return float(x["accuracy_percent"][-1])

def bootstrap(d):
    ix=rng.integers(0,len(d),(B,len(d)))
    q=d[ix].mean(1)
    return [
        float(np.percentile(q,2.5)),
        float(np.percentile(q,97.5))
    ]

def exact_signflip(d):
    d=np.asarray(d,float)
    obs=abs(d.mean())
    vals=[]
    for signs in itertools.product((-1,1),repeat=len(d)):
        vals.append(abs(np.mean(d*np.asarray(signs))))
    vals=np.asarray(vals)
    return float(np.mean(vals >= obs-1e-12))

tests=[]

for ds in datasets:
    n={"2a":9,"2b":9,"ssvep":10}[ds]

    reader={}
    for s in range(1,n+1):
        reader[s]=np.mean([
            endpoint(ds,s,seed,"reader")
            for seed in plan["seeds"]
        ])

    for model in models:
        control={}
        for s in range(1,n+1):
            control[s]=np.mean([
                endpoint(ds,s,seed,model)
                for seed in plan["seeds"]
            ])

        d=np.asarray([reader[s]-control[s] for s in range(1,n+1)])

        tests.append({
            "dataset":ds,
            "comparison":f"REACT-{model}",
            "react_mean":float(np.mean(list(reader.values()))),
            "control_mean":float(np.mean(list(control.values()))),
            "difference":float(d.mean()),
            "ci95":bootstrap(d),
            "permutation_p":exact_signflip(d),
            "subject_differences":d.tolist()
        })

# Holm correction across all 9 declared endpoint comparisons
order=np.argsort([x["permutation_p"] for x in tests])
m=len(tests)
running=0.0
adjusted=[None]*m

for rank,idx in enumerate(order):
    raw=tests[idx]["permutation_p"]
    val=min(1.0,(m-rank)*raw)
    running=max(running,val)
    adjusted[idx]=running

for x,p in zip(tests,adjusted):
    x["holm_p"]=float(min(1,p))

OUT=Path(
"/lustre/smuexa01/client/users/nikkieh/"
"reader_review_controls/final_fresh_20260922/"
"endpoint_statistics.json"
)
OUT.write_text(json.dumps(tests,indent=2)+"\n")

for x in tests:
    print(
        x["dataset"],
        f'{x["comparison"]:18s}',
        "delta",round(x["difference"],2),
        "CI",[round(z,2) for z in x["ci95"]],
        "p",round(x["permutation_p"],4),
        "Holm",round(x["holm_p"],4)
    )
print("\nWROTE",OUT)
