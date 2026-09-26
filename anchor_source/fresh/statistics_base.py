from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ensure_new_dir, file_sha, write_json


def normalized_area(a:np.ndarray,times:np.ndarray) -> np.ndarray:
    a=np.asarray(a,dtype=float); times=np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0) or a.shape[-1]!=len(times):
        raise ValueError("Area requires an ordered common grid with at least two points")
    trap=getattr(np,"trapezoid",None)
    if trap is None:
        trap=np.trapz
    return trap(a,x=times,axis=-1)/(times[-1]-times[0])


def exact_signflip(d:np.ndarray) -> float:
    """Two-sided mean sign-flip test. Requires justified symmetry/exchangeability."""
    d=np.asarray(d,dtype=float)
    if d.ndim!=1 or not np.isfinite(d).all() or not 1<=len(d)<=20:
        raise ValueError("Exact signflip supports 1..20 finite subject differences")
    signs=np.array(list(itertools.product([-1.,1.],repeat=len(d))))
    stats=(signs@d)/len(d)
    obs=abs(d.mean())
    return float(np.mean(np.abs(stats)>=obs-1e-12))


def holm(p:list[float]) -> list[float]:
    a=np.asarray(p,dtype=float)
    if np.any(~np.isfinite(a)) or np.any((a<0)|(a>1)):
        raise ValueError("Invalid p-values")
    order=np.argsort(a); result=np.empty(len(a)); running=0.
    for rank,idx in enumerate(order):
        running=max(running,(len(a)-rank)*a[idx])
        result[idx]=min(running,1.)
    return result.tolist()


def paired_panel(df:pd.DataFrame,cfg:dict,seeds:list[int],models:list[str],grid:list[int]) -> dict[str,np.ndarray]:
    subjects=[str(s) for s in cfg["subjects"]]
    keys=["subject","seed","model","n_samples"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate model/subject/seed/sample rows")
    expected=set(itertools.product(subjects,seeds,models,grid))
    found=set(map(tuple,df[keys].itertuples(index=False,name=None)))
    if found!=expected:
        missing=sorted(expected-found)[:6]
        extra=sorted(found-expected)[:6]
        raise ValueError(f"Incomplete or extra paired panel; missing={missing}, extra={extra}. No silent complete-case filtering.")
    if np.any(~np.isfinite(df.accuracy)) or np.any((df.accuracy<0)|(df.accuracy>100)):
        raise ValueError("Accuracy must be in percent [0,100]")
    if not np.allclose(df.accuracy,100.*df.n_correct/df.n_test,atol=1e-9,rtol=0):
        raise ValueError("Accuracy and integer trial counts disagree")
    # Every comparison must concern the same trials, labels, prepared input and
    # training-fitted normalization within subject. Seeds must not change test roles.
    for s,part in df.groupby("subject"):
        for col in ["n_test","test_ids_hash","test_labels_hash","prepared_x_hash","normalizer_hash"]:
            if part[col].nunique()!=1:
                raise ValueError(f"{s}: inconsistent {col} across compared runs")
    for _,part in df.groupby(["subject","seed","model"]):
        if part.checkpoint_sha256.nunique()!=1:
            raise ValueError("Multiple checkpoints used across the curve of one model run")
    if set(df.fs.astype(float))!={float(cfg["fs"])}:
        raise ValueError("Sampling frequency inconsistent with the plan")
    result={}
    for model in models:
        sub=df[df.model==model].set_index(["subject","seed","n_samples"])
        idx=pd.MultiIndex.from_product([subjects,seeds,grid],names=["subject","seed","n_samples"])
        raw=sub.reindex(idx).accuracy.to_numpy().reshape(len(subjects),len(seeds),len(grid))
        result[model]=raw.mean(axis=1)  # Not seed ensembling; accuracy then averaging.
    return result


def analyze(csv_path:str|Path,manifest:dict,out:str|Path,comparisons:list[list[str]]|None=None,
            formal_tests:bool=False,endpoint_only:bool=False) -> Path:
    df=pd.read_csv(csv_path,dtype={"subject":str})
    if set(df.evidence_tag)!={manifest["evidence_tag"]}:
        raise ValueError("Input evidence tag differs from manifest")
    comparisons=comparisons or manifest["plan"]["comparisons"]
    models=sorted({m for pair in comparisons for m in pair})
    if any(len(p)!=2 or p[0]==p[1] for p in comparisons):
        raise ValueError("Comparisons must have two distinct models")
    df=df[(df.regime=="endpoint") & df.model.isin(models)].copy()
    plan=manifest["plan"]
    reps=int(plan.get("bootstrap_resamples",10000)); seed=int(plan.get("bootstrap_seed",170926))
    if reps<100:
        raise ValueError("At least 100 bootstrap resamples required")
    panels={}
    for name,cfg in plan["datasets"].items():
        grid=[cfg["samples"]] if endpoint_only else cfg["grid"]
        part=df[(df.dataset==name)&df.n_samples.isin(grid)]
        if not endpoint_only:
            # Reject a hidden grid change, including selectively omitted points.
            extra=df[(df.dataset==name)&~df.n_samples.isin(grid)]
            if len(extra):
                raise ValueError("Rows contain a grid outside the declared primary curve")
        panels[name]=(grid,paired_panel(part,cfg,plan["seeds"],models,grid))
    directory=ensure_new_dir(out)
    curve_rows=[]; diff_rows=[]; subject_rows=[]; result_rows=[]
    rng=np.random.default_rng(seed)
    for name,(grid,panel) in panels.items():
        cfg=plan["datasets"][name]; subjects=[str(s) for s in cfg["subjects"]]; S=len(subjects)
        if S<2:
            raise ValueError("Subject uncertainty requires at least two subjects")
        ix=rng.integers(0,S,size=(reps,S))
        times=np.array(grid)/cfg["fs"]
        for model,a in panel.items():
            bs=a[ix].mean(1); low,high=np.quantile(bs,[.025,.975],axis=0)
            for j,m in enumerate(grid):
                curve_rows.append({"dataset":name,"model":model,"n_samples":m,"time_s":times[j],
                    "mean_accuracy":a[:,j].mean(),"subject_sd":a[:,j].std(ddof=1),"ci_low":low[j],"ci_high":high[j],"subjects":S})
                for s,val in zip(subjects,a[:,j]):
                    subject_rows.append({"dataset":name,"subject":s,"model":model,"n_samples":m,"accuracy":val})
        for target,base in comparisons:
            delta=panel[target]-panel[base]
            bs=delta[ix].mean(1); lo,hi=np.quantile(bs,[.025,.975],axis=0)
            for j,m in enumerate(grid):
                diff_rows.append({"dataset":name,"target":target,"baseline":base,"n_samples":m,"time_s":times[j],
                   "difference_pp":delta[:,j].mean(),"ci_low":lo[j],"ci_high":hi[j],"subjects":S})
            outcomes={"endpoint":delta[:,-1]}
            if not endpoint_only:
                outcomes["normalized_accuracy_duration_area"]=normalized_area(delta,times)
            for measure,d in outcomes.items():
                low,high=np.quantile(d[ix].mean(1),[.025,.975])
                row={"dataset":name,"target":target,"baseline":base,"measure":measure,
                     "difference_pp":d.mean(),"ci_low":low,"ci_high":high,"subjects":S,
                     "wins":int((d>1e-10).sum()),"ties":int((abs(d)<=1e-10).sum()),"losses":int((d< -1e-10).sum())}
                if formal_tests:
                    row["p_signflip"]=exact_signflip(d)
                result_rows.append(row)
    if formal_tests:
        adjusted=holm([r["p_signflip"] for r in result_rows])
        for row,p in zip(result_rows,adjusted):
            row["p_holm"]=p
    curves=pd.DataFrame(curve_rows); differences=pd.DataFrame(diff_rows); results=pd.DataFrame(result_rows)
    curves.to_csv(directory/"curves.csv",index=False)
    differences.to_csv(directory/"paired_curves.csv",index=False)
    pd.DataFrame(subject_rows).to_csv(directory/"subject_means.csv",index=False)
    results.to_csv(directory/"paired_summary.csv",index=False)
    # Simple ready-to-paste LaTeX, no invented placeholders or significance bolding.
    lines=[r"\begin{tabular}{lllrrr}",r"\toprule",r"Dataset & Comparison & Outcome & $\Delta$ (pp) & CI low & CI high \\",r"\midrule"]
    def esc(x): return str(x).replace("_",r"\_")
    for r in result_rows:
        outcome="Endpoint" if r["measure"]=="endpoint" else "Accuracy--duration"
        lines.append(f"{esc(r['dataset'])} & {esc(r['target'])}--{esc(r['baseline'])} & {outcome} & {r['difference_pp']:+.2f} & {r['ci_low']:.2f} & {r['ci_high']:.2f} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    if manifest["evidence_tag"]=="synthetic":
        lines.insert(0,"% SYNTHETIC SOFTWARE TEST OUTPUT -- NEVER PAPER RESULTS")
    (directory/"paired_summary.tex").write_text("\n".join(lines)+"\n")
    write_json(directory/"analysis.json",{"status":"complete","evidence_tag":manifest["evidence_tag"],
        "input_csv_sha256":file_sha(csv_path),"manifest_hash":manifest["manifest_hash"],"plan":plan,
        "comparisons":comparisons,"endpoint_only":endpoint_only,"formal_tests":formal_tests,
        "interval":"95% percentile paired subject bootstrap, conditional on observed training runs; pointwise, not simultaneous",
        "test_assumption":"sign exchangeability/symmetry of paired subject differences required",
        "holm_family":"ALL endpoint and accuracy-duration contrasts in this analysis" if formal_tests else None})
    make_plots(curves,differences,directory,manifest["evidence_tag"])
    return directory


def make_plots(curves:pd.DataFrame,diff:pd.DataFrame,directory:Path,tag:str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prefix="SYNTHETIC TEST — " if tag=="synthetic" else ""
    names={"reader":"READER","compact":"Compact","ff":"Forward–Forward","compact_mean":"Compact–Mean"}
    datasets={"2a":"BCICIV-2A","2b":"BCICIV-2B","ssvep":"SD-SSVEP"}
    for dataset,part in curves.groupby("dataset"):
        plt.figure(figsize=(6.4,4.0))
        for model,p in part.groupby("model"):
            plt.plot(p.time_s,p.mean_accuracy,marker=".",label=names.get(model,model))
        plt.xlabel("Observed EEG duration (s)"); plt.ylabel("Accuracy (%)")
        plt.title(prefix+datasets.get(dataset,dataset)); plt.legend(); plt.tight_layout()
        plt.savefig(directory/f"{dataset}_accuracy.png",dpi=180); plt.close()
    for (dataset,target,base),p in diff.groupby(["dataset","target","baseline"]):
        plt.figure(figsize=(6.4,4.0))
        plt.plot(p.time_s,p.difference_pp,marker=".")
        plt.fill_between(p.time_s,p.ci_low,p.ci_high,alpha=.2)
        plt.axhline(0,linestyle="--",linewidth=.8)
        plt.xlabel("Observed EEG duration (s)"); plt.ylabel("Paired accuracy difference (pp)")
        plt.title(prefix+f"{datasets.get(dataset,dataset)}: {names.get(target,target)} − {names.get(base,base)}\nPointwise 95% subject-bootstrap intervals")
        plt.tight_layout(); plt.savefig(directory/f"{dataset}_{target}_minus_{base}.png",dpi=180); plt.close()
