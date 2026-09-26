# REACT-EEG

## Reverse-Encoded Anytime Causal Temporal Modeling for EEG Decoding

Research code accompanying the manuscript **REACT-EEG: Reverse-Encoded Anytime Causal Temporal Modeling for EEG Decoding**.

REACT-EEG classifies EEG at different observation lengths using the same trained checkpoint. It combines a forward temporal reader, summarized at the newest observed token, with an independent reverse reader that processes the observed tokens from newest to oldest. A learned feature-wise gate fuses the two representations before classification.

The main experiment uses endpoint-only training. Separate control experiments investigate temporal direction, readout, input geometry, and matched random-duration training on **BCICIV-2A, BCICIV-2B, and SD-SSVEP**.

## Repository Structure

```text
REACT-EEG/
├── README.md
├── main_source/
│   ├── fresh/
│   │   ├── models.py                 
│   │   ├── controls.py              
│   │   ├── data.py                  
│   │   ├── bite.py                   
│   │   ├── setup.py                  
│   │   ├── run.py                   
│   │   ├── smoke.py                  
│   │   ├── report.py                 
│   │   └── statistics_base.py        
│   ├── tests/
│   ├── scripts/
│   ├── docs/
│   └── paper/                      
├── anchor_source/
│   ├── fresh/                      
│   ├── train_reverse_only_final.py
│   ├── train_paired_random_trunc_final.py
│   ├── reviewer_eval_controls.py
│   ├── bite_zerofill_final.py
│   ├── final_endpoint_stats.py
│   ├── summarize_reverse_only.py
│   ├── summarize_paired_random_trunc_final.py
│   ├── fresh_primary_nauc.py
│   ├── check_fresh_causality.py
│   └── make_fresh_anytime_figure.py
└── main_evidence/                    
```

The commands below assume this layout. Both source directories contain a package named `fresh`; use the working directories shown rather than adding both packages to `PYTHONPATH` together. The `main_evidence/` records are not required to start a new study.

## Installation

Use Python **3.11 or later** in an isolated environment. Full training and the GPU preflight require a CUDA-capable GPU; the model example and software tests can run on CPU.

Clone or download this repository, open a terminal at its root, and run:

```bash
export REACT_ROOT="$PWD"

python3 -m venv .venv
source .venv/bin/activate
```

Install a PyTorch build appropriate for your system using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). Then install:

```bash
python -m pip install \
    numpy==2.3.5 \
    scipy==1.17.0 \
    scikit-learn==1.8.0 \
    pandas==2.2.3 \
    matplotlib==3.10.8 \
    pytest==9.0.2
```

These dependency versions and PyTorch `2.10.0+cpu` were used for local core-model checks. They are not a recovered lockfile for the original HPC training environment. Git must be installed for baseline retrieval. BiTE's additional dependencies should be checked against the pinned baseline's own `requirements.txt` after study setup.

### Quick Model Check

This example uses random inputs to check output dimensions and parameter counts. It does not load trained weights or measure classification accuracy.

```bash
PYTHONPATH="$REACT_ROOT/main_source" python - <<'PY'
import torch
from fresh.data import SPECS, grid
from fresh.models import make_model

torch.set_num_threads(1)
expected = {"2a": 18916, "2b": 16962, "ssvep": 20140}

for dataset, cfg in SPECS.items():
    model = make_model(
        channels=cfg["channels"], classes=cfg["classes"],
        max_samples=cfg["samples"], pool=cfg["pool"],
        kind="reader", seed=2025,
    )
    model.project_constraints()
    model.eval()
    count = sum(p.numel() for p in model.parameters())
    assert count == expected[dataset]
    x = torch.randn(2, cfg["channels"], cfg["samples"])
    with torch.inference_mode():
        for m in (grid(dataset)[0], cfg["samples"]):
            assert model(x[:, :, :m]).shape == (2, cfg["classes"])
    print(f"{dataset}: {count:,} parameters; checks passed")
PY
```

## Datasets

Datasets and pretrained checkpoints are **not included**. The executable loader expects already-prepared `.npz` files.

| Dataset | Code key | Subjects | Channels | Classes | Sampling rate | Input | Pooling width |
|---|---|---:|---:|---:|---:|---:|---:|
| BCICIV-2A | `2a` | 9 | 22 | 4 | 250 Hz | 1,000 samples / 4 s | 32 |
| BCICIV-2B | `2b` | 9 | 3 | 2 | 250 Hz | 1,000 samples / 4 s | 32 |
| SD-SSVEP | `ssvep` | 10 | 8 | 12 | 256 Hz | 256 samples / 1 s | 4 |

Each NPZ file must contain:

```text
data   : finite numeric array [trials, channels, samples]
label  : one-dimensional integer array [trials], with zero-based labels
```

The required filenames are:

```text
2a_pre/
    A01T.npz, A01E.npz                    # Subject 1
    ...                                  # Subjects 1–9

2b_pre/
    B0101T.npz, B0102T.npz, B0103T.npz    # Subject 1, training sessions 1–3
    B0104E.npz, B0105E.npz                # Subject 1, test sessions 4–5
    ...                                  # Subjects 1–9

sdssvep_pre/
    S01_train.npz, S01_test.npz            # Subject 1
    ...                                  # Subjects 1–10
```

BCICIV-2A uses the training session for fitting and the evaluation session for testing. BCICIV-2B combines sessions 1–3 for training and 4–5 for testing. SD-SSVEP uses the supplied per-subject training and test files without repartitioning. Its observation times are measured from the beginning of the retained segment, not stimulus onset.

The loader fits `StandardScaler` on training trials at each channel–sample coordinate and applies the frozen transformation to held-out inputs. Motor-imagery arrays may contain 1,000 or 1,001 samples; only the first 1,000 are retained. SD-SSVEP arrays must contain 256 samples.

**Raw-to-NPZ conversion and the original SD-SSVEP split construction are not included in this source release.** The prepared files must be obtained or generated separately with documented preprocessing and partitions. See [`fresh/data.py`](main_source/fresh/data.py) for the input checks.

## Running the Code

Run computation on an allocated compute node when using HPC. The original shell and Slurm wrappers contain cluster-specific settings and must be reviewed before reuse.

### 1. Configure a New Study

Set the prepared-data locations and a **new output directory outside this repository**:

```bash
export REACT_DATA_2A="/absolute/path/to/2a_pre"
export REACT_DATA_2B="/absolute/path/to/2b_pre"
export REACT_DATA_SSVEP="/absolute/path/to/sdssvep_pre"
export REACT_STUDY="/absolute/path/to/new_react_study"
```

Replace the example paths with your own. The study directory must not already exist. Run the following setup block from the same environment:

```bash
python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

repo = Path(os.environ["REACT_ROOT"]).expanduser().resolve()
study = Path(os.environ["REACT_STUDY"]).expanduser().resolve()
roots = {
    "2a": str(Path(os.environ["REACT_DATA_2A"]).expanduser().resolve()),
    "2b": str(Path(os.environ["REACT_DATA_2B"]).expanduser().resolve()),
    "ssvep": str(Path(os.environ["REACT_DATA_SSVEP"]).expanduser().resolve()),
}
if not (repo / "main_source/fresh/models.py").is_file():
    raise SystemExit("REACT_ROOT must point to the repository root.")
if study.exists() or study == repo or repo in study.parents:
    raise SystemExit("Choose a new study directory outside the repository.")
if not all(Path(root).is_dir() for root in roots.values()):
    raise SystemExit("A prepared-data directory does not exist.")

study.mkdir(parents=True)
shutil.copytree(
    repo / "main_source", study / "code",
    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
)
sys.path.insert(0, str(study / "code"))
from fresh import setup
setup.DEFAULT_ROOTS.update(roots)
setup.freeze(study)
PY
```

This creates `study.json`, freezes a source copy in `code/`, records hashes, checks dataset filenames, and retrieves BiTE at commit `924eb32241ba1a7c80dbc4ba097f8c979da17578`. Network access is needed unless the pinned revision is available through the existing local cache.

The retrieved baseline is placed in `$REACT_STUDY/third_party/BiteEEG/`. Review its `requirements.txt` and install any additional dependencies before the next step. A failed setup can leave a partial study directory; preserve its diagnostics and use a new output directory for a corrected attempt.

Do not modify a study's frozen code or configuration after training begins. Set these environment variables again when opening a new shell or compute allocation.

### 2. Run the GPU Preflight

Inside an allocated GPU session:

```bash
(
    cd "$REACT_STUDY/code"
    python -m fresh.smoke --study "$REACT_STUDY"
)
```

Resolve any errors before starting full training. This preflight uses training-role data and writes `smoke.json`; it is not a held-out benchmark evaluation.

### 3. Train and Evaluate

Run one subject/seed task:

```bash
(
    cd "$REACT_STUDY/code"
    python -m fresh.run --study "$REACT_STUDY" --index 0 --device cuda
)
```

Task `0` is BCICIV-2A, subject 1, seed 2025. Each task trains **all five primary models**, not only REACT:

| Paper name | Code identifier |
|---|---|
| REACT-EEG / REACT | `reader` |
| No reverse | `compact` |
| Two forward | `ff` |
| Forward mean | `compact_mean` |
| BiTE | `bite` |

| Dataset | Task indices |
|---|---|
| BCICIV-2A | 0–26 |
| BCICIV-2B | 27–53 |
| SD-SSVEP | 54–83 |

Tasks are ordered by dataset, subject, and then seed: 2025, 2026, and 2027. The complete primary experiment contains **84 tasks and 420 model fits**. Run indices 0–83 in a reviewed scheduler array or sequentially within a sufficient compute allocation:

```bash
(
    set -e
    cd "$REACT_STUDY/code"
    for INDEX in $(seq 0 83); do
        python -m fresh.run --study "$REACT_STUDY" --index "$INDEX" --device cuda
    done
)
```

The main trainer supports identity-checked resume from `last.pt`. Held-out data are opened after all five models in a task have trained. The four core models are evaluated across observation lengths; the primary runner evaluates BiTE at its endpoint only.

**Training configuration:** 600 epochs, batch size 64, Adam with learning rate and weight decay 0.002, betas (0.9, 0.999), epsilon 1e-8, cosine decay to zero, label smoothing 0.1, and gradient clipping at 5. The evaluation checkpoint is the final epoch. Shared components use matched initialization; the second reader uses fixed constructor seed 918273.

**BiTE note:** the exported trainer clips gradients at 5 for BiTE as well. The manuscript describes disabling BiTE clipping. Resolve this difference against the final run provenance before claiming exact reproduction of that baseline.

### 4. Collect Results

After the full primary cohort completes:

```bash
(
    cd "$REACT_STUDY/code"
    python -m fresh.report --study "$REACT_STUDY"
)
```

The collector validates completeness and result identities. It refuses to overwrite an existing `analysis/` directory. Its generated paper uses a legacy template in `main_source/paper/`, not the final manuscript.

### Output Files

A completed task has the following structure:

```text
study/runs/2a_S01_seed2025/
├── scaler.npz
├── training_data.json
├── held_data.json
├── pairing.json
├── reader/
│   ├── final.pt
│   ├── last.pt
│   ├── result.json
│   ├── predictions.npz
│   ├── causality.json
│   └── training_history.json
├── compact/
├── ff/
├── compact_mean/
└── bite/
```

For a trained model, keep its `final.pt` together with the task's `scaler.npz`. Checkpoint weights are stored under `checkpoint["state_dict"]`. Prediction files store `logits`, `y`, `ids`, and `lengths`; core-model logits have shape `[test trials, observation boundaries, classes]`.

### Inference with a Trained Checkpoint

<details>
<summary>Example: BCICIV-2A subject 1, seed 2025, after one second of EEG</summary>

This requires the completed task and its prepared data. Only load checkpoints from a trusted source.

```bash
python - <<'PY'
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

study = Path(os.environ["REACT_STUDY"]).expanduser().resolve()
sys.path.insert(0, str(study / "code"))
from fresh.data import load_role, normalize
from fresh.run import model_for

plan = json.loads((study / "study.json").read_text())
task = study / "runs/2a_S01_seed2025"
with np.load(task / "scaler.npz", allow_pickle=False) as saved:
    scaler = {key: saved[key] for key in saved.files}
checkpoint = torch.load(
    task / "reader/final.pt", map_location="cpu", weights_only=True
)
model = model_for("reader", "2a", 2025, study)
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
raw, labels, _, _ = load_role(plan["data_roots"]["2a"], "2a", 1, "test")
observed = normalize(raw[:4, :, :250], scaler)
with torch.inference_mode():
    logits = model(torch.from_numpy(observed))
print("Predictions:", logits.argmax(-1).tolist())
print("Reference labels:", labels[:4].tolist())
PY
```

Change the sample boundary to query the same checkpoint at another duration. Do not refit the scaler on the test trial.

</details>

## Ablation and Control Experiments

These scripts require the primary study, its prepared data, frozen code, saved scalers, and relevant checkpoints. Set a separate output location:

```bash
export REACT_CONTROLS="/absolute/path/to/new_control_outputs"
```

### Reverse-Only Training

```bash
python "$REACT_ROOT/anchor_source/train_reverse_only_final.py" \
    --study "$REACT_STUDY" --out "$REACT_CONTROLS/reverse_only" \
    --task-index 0 --device cuda
```

This independently trains the tokenizer, reverse reader, and classifier from matched initialization. It is not a route removal from an already-trained REACT checkpoint.

### Paired Random-Duration Training

```bash
python "$REACT_ROOT/anchor_source/train_paired_random_trunc_final.py" \
    --study "$REACT_STUDY" --out "$REACT_CONTROLS/random_duration" \
    --task-index 0 --device cuda
```

The `react_rt` and `compact_rt` arms receive identical minibatch orders and duration schedules. Each minibatch samples one boundary uniformly from the evaluation grid and uses one label-smoothed cross-entropy loss, with no additional endpoint term. This script does not provide the main trainer's mid-training checkpoint-resume mechanism.

### Fixed-Anchor and Endpoint-Aligned-Position Controls

```bash
(
    cd "$REACT_ROOT/anchor_source"
    PYTHONPATH=. python -m fresh.anchor_diag \
        --orig-study "$REACT_STUDY" --out "$REACT_CONTROLS/anchor" \
        --index 0 --device cuda
)
```

For each of these three task-indexed workflows, run indices **0–83** for the full cohort. The anchor diagnostic also runs an earlier forward-only `random_truncation` control; it is not the final paired random-duration comparison.

Collect the completed anchor cohort with:

```bash
(
    cd "$REACT_ROOT/anchor_source"
    PYTHONPATH=. python -m fresh.anchor_report \
        --orig-study "$REACT_STUDY" --diag-study "$REACT_CONTROLS/anchor"
)
```

### Mean-Imputation Evaluation

```bash
python "$REACT_ROOT/anchor_source/reviewer_eval_controls.py" \
    --study "$REACT_STUDY" --out "$REACT_CONTROLS/mean_imputation" \
    --dataset all --device cuda

python "$REACT_ROOT/anchor_source/bite_zerofill_final.py" \
    --study "$REACT_STUDY" --out "$REACT_CONTROLS/bite_mean_imputation" \
    --device cuda
```

These evaluations preserve the original input length and replace unavailable samples with zero in standardized space, corresponding to training-set means. The reviewer-control script also evaluates interventions on trained REACT routes; those are distinct from independently trained reverse-only results.

## Analysis and Visualization

BCICIV-2A/2B are evaluated over **1–4 s at 28 boundaries**; SD-SSVEP is evaluated over **0.25–1 s of the retained segment at 49 boundaries**. Seed accuracies are averaged within subject before aggregation. The checkpoints are not combined into a prediction ensemble.

Normalized accuracy–time area (nAUC) is trapezoidal area divided by the evaluated time interval. It is not ROC AUC. Differences are reported in percentage points. The manuscript uses 10,000 paired subject-bootstrap resamples and separately defined nine-test Holm families.

```bash
python "$REACT_ROOT/anchor_source/fresh_primary_nauc.py" "$REACT_STUDY"
python "$REACT_ROOT/anchor_source/check_fresh_causality.py" "$REACT_STUDY"

mkdir -p "$REACT_CONTROLS/figures"
MPLBACKEND=Agg python "$REACT_ROOT/anchor_source/make_fresh_anytime_figure.py" \
    "$REACT_STUDY" "$REACT_CONTROLS/figures/fig2.pdf"
```

The causality-summary script reads existing `causality.json` records; it does not rerun model tests. Keep model-specific and pooled maxima distinct. The figure generator plots REACT, No reverse, Two forward, and Forward mean; shading is **±1 sample SD across subject-level seed averages**, with dotted chance lines.

### Additional Statistical Summaries

These scripts retain path constants rather than generic command-line path arguments. Configure them in a working copy before execution:

| Script in `anchor_source/` | Constants to configure |
|---|---|
| `final_endpoint_stats.py` | `ORIG`: primary study; `OUT`: output JSON file |
| `summarize_reverse_only.py` | `ORIG`: primary study; `REV`: reverse-only output |
| `summarize_paired_random_trunc_final.py` | `ORIG`, `RT`: random-duration output, `REV` |

Ensure the parent directory of `OUT` exists. The paired-duration summary also needs completed reverse-only results.

```bash
python "$REACT_ROOT/anchor_source/final_endpoint_stats.py"
python "$REACT_ROOT/anchor_source/summarize_reverse_only.py"
python "$REACT_ROOT/anchor_source/summarize_paired_random_trunc_final.py"
```

## Results

The following values are **reported in the manuscript**, not recomputed from the source archive.

| Dataset | Earliest accuracy (%) | nAUC | Endpoint accuracy (%) | REACT parameters |
|---|---:|---:|---:|---:|
| BCICIV-2A | 69.32 | 80.12 | 85.06 | 18,916 |
| BCICIV-2B | 74.31 | 83.59 | 86.20 | 16,962 |
| SD-SSVEP | 34.89 | 64.15 | 96.50 | 20,140 |

Earliest boundaries are 1 s for BCICIV-2A/2B and 0.25 s for SD-SSVEP. REACT adds 3,136 parameters relative to either single-reader control.

After matched random-duration training, the REACT-minus-No-reverse nAUC differences are:

| BCICIV-2A | BCICIV-2B | SD-SSVEP |
|---:|---:|---:|
| +10.42 [6.44, 15.21] | +0.22 [−1.17, 1.61] | +24.96 [19.17, 30.93] |

Values are percentage-point differences with 95% bootstrap intervals. These results use retrained random-duration models separately. Reverse-only can outperform fused REACT at earlier durations, especially on SD-SSVEP; the paper does not claim uniform superiority across models, datasets, or decision times.

## Testing

Run software tests in a working repository copy, not a frozen study directory, because some tests write validation records:

```bash
(
    cd "$REACT_ROOT/main_source"
    PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
        python -m pytest tests -q -k 'not full_reporting_layout'
)
```

The excluded legacy layout test writes to a fixed `/mnt/data` location. Software tests check implementation behavior; they do not establish reproduction of the reported benchmark scores.

