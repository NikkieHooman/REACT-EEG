# REACT-EEG

**Reverse-Encoded Anytime Causal Temporal Modeling for EEG Decoding**

REACT-EEG is an anytime EEG classifier that reuses the same trained checkpoint across different observation lengths. A forward temporal reader summarizes the newest observed token, while an independently parameterized reverse reader processes the available tokens from newest to oldest. A learned feature-wise gate combines both representations before classification.

The associated manuscript evaluates REACT-EEG on BCICIV-2A, BCICIV-2B, and SD-SSVEP. The primary experiment trains models at the complete-trial endpoint and evaluates their checkpoints at earlier observation boundaries. Additional experiments examine processing direction, readout, input geometry, and random-duration training.

## Repository structure

The commands below assume this directory structure:

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
│   │   ├── anchor_models.py
│   │   ├── anchor_diag.py
│   │   ├── anchor_report.py
│   │   └── ...
│   ├── train_reverse_only_final.py
│   ├── train_paired_random_trunc_final.py
│   ├── reviewer_eval_controls.py
│   ├── bite_zerofill_final.py
│   ├── summarize_reverse_only.py
│   ├── summarize_paired_random_trunc_final.py
│   ├── final_endpoint_stats.py
│   ├── fresh_primary_nauc.py
│   ├── check_fresh_causality.py
│   └── make_fresh_anytime_figure.py
└── main_evidence/
```

`main_source/` contains the primary experiment. `anchor_source/` contains the later control and analysis scripts, together with copies of the core modules.

Both directories contain a Python package named `fresh`. Do not add both directories to `PYTHONPATH` simultaneously. The commands below select the required package explicitly.

EEG datasets and trained checkpoints are not included.

## 1. Installation

Use a separate Python environment. Linux with a CUDA-capable GPU is the intended environment for the complete training workflow.

Open a terminal at the repository root—the directory containing `main_source/` and `anchor_source/`—and run:

```bash
export REACT_ROOT="$PWD"

python3 -m venv .venv
source .venv/bin/activate
```

Use Python 3.11 or later for the dependency versions below.

Install PyTorch using the [official installation instructions](https://pytorch.org/get-started/locally/). For GPU training, select a CUDA-enabled build compatible with your machine. The core code was locally checked with PyTorch `2.10.0+cpu`; this is a software-test configuration, not the original HPC training environment.

Install the remaining dependencies:

```bash
python -m pip install \
    numpy==2.3.5 \
    scipy==1.17.0 \
    scikit-learn==1.8.0 \
    pandas==2.2.3 \
    matplotlib==3.10.8 \
    pytest==9.0.2
```

Git must also be available because study setup retrieves the pinned BiTE baseline.

Check the environment:

```bash
git --version

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The GPU preflight requires `CUDA available: True` inside an allocated GPU session.

BiTE has additional upstream dependencies. After study setup retrieves its source, inspect its `requirements.txt` and install any missing dependencies before running the GPU preflight. Preserve the PyTorch build appropriate for your hardware.

## 2. Run a data-free model check

This example constructs REACT-EEG for all three datasets and evaluates randomly generated inputs at the earliest and final observation boundaries.

It checks the model interface and parameter counts; it does not measure EEG classification accuracy.

Run from the repository root:

```bash
PYTHONPATH="$REACT_ROOT/main_source" python - <<'PY'
import torch

from fresh.data import SPECS, grid
from fresh.models import make_model

torch.set_num_threads(1)

expected_parameters = {
    "2a": 18916,
    "2b": 16962,
    "ssvep": 20140,
}

for dataset, cfg in SPECS.items():
    model = make_model(
        channels=cfg["channels"],
        classes=cfg["classes"],
        max_samples=cfg["samples"],
        pool=cfg["pool"],
        kind="reader",
        seed=2025,
    )
    model.project_constraints()
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == expected_parameters[dataset]

    x = torch.randn(2, cfg["channels"], cfg["samples"])

    with torch.inference_mode():
        for m in (grid(dataset)[0], cfg["samples"]):
            logits = model(x[:, :, :m])
            assert logits.shape == (2, cfg["classes"])

    print(f"{dataset}: {parameter_count:,} parameters; checks passed")
PY
```

Expected output:

```text
2a: 18,916 parameters; checks passed
2b: 16,962 parameters; checks passed
ssvep: 20,140 parameters; checks passed
```

The core model implementation is in [`main_source/fresh/models.py`](main_source/fresh/models.py).

## 3. Prepare the datasets

The experiment loader reads prepared `.npz` files. It does not directly read raw GDF or MATLAB recordings.

### Dataset configuration

| Dataset | Code identifier | Subjects | Channels | Classes | Sampling rate | Input samples | Pooling width |
|---|---|---:|---:|---:|---:|---:|---:|
| BCICIV-2A | `2a` | 9 | 22 | 4 | 250 Hz | 1,000 | 32 |
| BCICIV-2B | `2b` | 9 | 3 | 2 | 250 Hz | 1,000 | 32 |
| SD-SSVEP | `ssvep` | 10 | 8 | 12 | 256 Hz | 256 | 4 |

Use `ssvep`, not `sdssvep`, when referring to the dataset in this experiment code.

### Required array format

Each `.npz` file must contain:

```text
data   : [number_of_trials, channels, samples]
label  : [number_of_trials]
```

`data` must contain finite numeric values. `label` must be a one-dimensional integer array with zero-based class indices.

The motor-imagery loader accepts 1,000 or 1,001 stored samples and retains the first 1,000. SD-SSVEP inputs must contain exactly 256 samples.

### Required filenames

```text
2a_pre/
├── A01T.npz
├── A01E.npz
├── ...
├── A09T.npz
└── A09E.npz

2b_pre/
├── B0101T.npz
├── B0102T.npz
├── B0103T.npz
├── B0104E.npz
├── B0105E.npz
└── ...                     # The same session pattern for subjects 1–9

sdssvep_pre/
├── S01_train.npz
├── S01_test.npz
├── ...
├── S10_train.npz
└── S10_test.npz
```

BCICIV-2A uses the training session for fitting and the evaluation session for testing. BCICIV-2B combines sessions 1–3 for training and sessions 4–5 for testing. SD-SSVEP uses the supplied per-subject training and test files.

Normalization is fitted on training trials only. Each channel–sample coordinate has its own training-set mean and scale. The same fitted transformation is applied to held-out trials and shorter observations.

The raw-to-NPZ conversion and the original SD-SSVEP split-construction workflow are not included in this export. Obtain or generate the prepared files using the appropriate dataset preparation pipeline; do not create a new split and describe it as the original experiment.

See [`main_source/fresh/data.py`](main_source/fresh/data.py) for the exact file and array checks.

## 4. Configure a new study

A study is an output directory containing a frozen source copy, the experiment plan, baseline provenance, checkpoints, and results.

Set the paths to your prepared datasets and choose a new study directory outside the repository:

```bash
export REACT_DATA_2A="/absolute/path/to/2a_pre"
export REACT_DATA_2B="/absolute/path/to/2b_pre"
export REACT_DATA_SSVEP="/absolute/path/to/sdssvep_pre"

export REACT_STUDY="/absolute/path/to/new_react_study"
```

Replace these example paths with actual locations. The new study directory must not already exist.

Run:

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

for dataset, root in roots.items():
    if not Path(root).is_dir():
        raise SystemExit(f"Prepared-data directory not found for {dataset}: {root}")

study.mkdir(parents=True)

shutil.copytree(
    repo / "main_source",
    study / "code",
    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
)

sys.path.insert(0, str(study / "code"))

from fresh import setup

setup.DEFAULT_ROOTS.update(roots)
setup.freeze(study)
PY
```

Setup checks the required dataset filenames, records the study configuration and source hashes, and retrieves the upstream BiTE implementation at:

```text
924eb32241ba1a7c80dbc4ba097f8c979da17578
```

The baseline is stored in:

```text
$REACT_STUDY/third_party/BiteEEG/
```

Its dependency file is:

```text
$REACT_STUDY/third_party/BiteEEG/requirements.txt
```

Network access is required unless the pinned baseline is available through the code's existing local cache.

Do not modify `study.json` or the frozen `code/` directory after training begins. Create a separate study for changes to the experimental configuration.

The exported shell and Slurm scripts contain original cluster-specific paths, accounts, and interpreter locations. Review and adapt them before use on another system.

## 5. Run the GPU preflight

Run training and GPU checks on an allocated compute node, not an HPC login node.

In the allocated GPU session, activate the environment and restore `REACT_STUDY` if necessary. Then run:

```bash
(
    cd "$REACT_STUDY/code"
    python -m fresh.smoke --study "$REACT_STUDY"
)
```

The preflight checks training-role data, model construction, short optimization runs, parameter relationships, analysis dependencies, and prepared-input causality.

It writes:

```text
$REACT_STUDY/smoke.json
```

Proceed with the cohort only after resolving any preflight errors.

## 6. Train and evaluate the primary models

### Code identifiers

| Manuscript name | Code identifier |
|---|---|
| REACT-EEG / REACT | `reader` |
| No reverse | `compact` |
| Two forward | `ff` |
| Forward mean | `compact_mean` |
| BiTE | `bite` |

The internal identifiers are preserved for import and checkpoint compatibility.

### Run one subject/seed task

```bash
(
    cd "$REACT_STUDY/code"
    python -m fresh.run \
        --study "$REACT_STUDY" \
        --index 0 \
        --device cuda
)
```

Task `0` corresponds to BCICIV-2A, subject 1, seed 2025.

Each task trains all five primary models sequentially:

```text
compact → reader → ff → compact_mean → bite
```

The runner evaluates held-out data after training all five models for the task.

### Task indices

| Dataset | Task indices |
|---|---|
| BCICIV-2A | 0–26 |
| BCICIV-2B | 27–53 |
| SD-SSVEP | 54–83 |

Within each dataset, tasks are ordered by subject and then by seed: 2025, 2026, and 2027.

The complete primary cohort contains 84 subject/seed tasks and 420 model fits.

For a sequential run inside a sufficiently provisioned GPU allocation:

```bash
(
    set -e
    cd "$REACT_STUDY/code"

    for INDEX in $(seq 0 83); do
        python -m fresh.run \
            --study "$REACT_STUDY" \
            --index "$INDEX" \
            --device cuda
    done
)
```

Alternatively, assign one index to each task in a reviewed scheduler array. Resource requests, concurrency limits, and environment activation must follow the local cluster configuration.

The main trainer supports identity-checked checkpoint resume. Reusing a task index resumes or verifies its existing outputs rather than defining a new experimental configuration.

### Training settings

| Setting | Value |
|---|---|
| Epochs | 600 |
| Evaluation checkpoint | Final epoch |
| Seeds | 2025, 2026, 2027 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.002 |
| Weight decay | 0.002 |
| Adam betas | 0.9, 0.999 |
| Adam epsilon | 1e-8 |
| Learning-rate schedule | Cosine decay to zero |
| Label smoothing | 0.1 |
| Gradient-norm clipping | 5 |
| Dropout | 0.3 |
| Primary objective | Complete-trial endpoint cross-entropy |

Shared components use matched initialization. REACT and Two forward also share the second-reader initialization, constructed using seed `918273`.

**BiTE reproduction note:** the exported primary trainer applies gradient clipping at 5 to all five models. The manuscript describes disabling clipping for BiTE. That difference must be reconciled with the final BiTE run provenance before treating this runner as an exact reproduction of the reported BiTE results.

See [`main_source/fresh/run.py`](main_source/fresh/run.py).

## 7. Locate checkpoints and predictions

Each task creates a directory such as:

```text
$REACT_STUDY/runs/2a_S01_seed2025/
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

`final.pt` contains the trained weights under `checkpoint["state_dict"]`.

For the four core models, `predictions.npz` contains:

```text
logits  : [test_trials, observation_boundaries, classes]
y       : test labels
ids     : prepared-trial identifiers
lengths : observation lengths in samples
```

The primary runner evaluates BiTE at the endpoint only. Variable-duration mean-imputed BiTE evaluation uses the separate control script below.

Keep the saved scaler with the corresponding trained checkpoint. Do not refit normalization on held-out data.

## 8. Run inference with a trained REACT checkpoint

The following example loads the BCICIV-2A subject-1 checkpoint trained with seed 2025 and predicts from the first 250 prepared samples per channel.

It requires the completed task and its prepared dataset files.

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

dataset = "2a"
subject = 1
seed = 2025
m = 250  # One second at 250 Hz.

plan = json.loads((study / "study.json").read_text())
task = study / "runs" / f"{dataset}_S{subject:02d}_seed{seed}"

with np.load(task / "scaler.npz", allow_pickle=False) as saved:
    scaler = {name: saved[name] for name in saved.files}

# Load only a checkpoint you created or otherwise trust.
checkpoint = torch.load(
    task / "reader/final.pt",
    map_location="cpu",
    weights_only=True,
)

model = model_for("reader", dataset, seed, study)
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()

raw, labels, _, _ = load_role(
    plan["data_roots"][dataset],
    dataset,
    subject,
    "test",
)

observed = normalize(raw[:4, :, :m], scaler)

with torch.inference_mode():
    logits = model(torch.from_numpy(observed))
    predictions = logits.argmax(dim=-1)

print("Logit shape:", tuple(logits.shape))
print("Predicted classes:", predictions.tolist())
print("Reference classes:", labels[:4].tolist())
PY
```

To query another observation boundary, change `m` while retaining the same checkpoint and saved scaler.

## 9. Collect the primary results

After all 84 primary tasks finish:

```bash
(
    cd "$REACT_STUDY/code"
    python -m fresh.report --study "$REACT_STUDY"
)
```

The collector checks cohort completeness and consistency between stored predictions, checkpoints, and result records before generating analyses.

It refuses to overwrite an existing `analysis/` directory. Preserve completed results rather than deleting them to rerun a collector.

The collector also uses an earlier manuscript template in `main_source/paper/`. Its generated draft is not the final submitted manuscript.

## 10. Run the additional control experiments

These commands require a primary study containing its plan, frozen code, prepared data, saved scalers, and relevant checkpoints.

Choose a separate output directory:

```bash
export REACT_CONTROLS="/absolute/path/to/new_react_controls"
```

Run the GPU commands below inside an allocated compute session.

### Independently trained reverse-only control

```bash
python "$REACT_ROOT/anchor_source/train_reverse_only_final.py" \
    --study "$REACT_STUDY" \
    --out "$REACT_CONTROLS/reverse_only" \
    --task-index 0 \
    --device cuda
```

Repeat `--task-index` from 0 through 83 for the complete cohort.

This control is trained independently. It copies the corresponding initialization from a newly constructed REACT model; it does not remove the forward route from an already-trained REACT checkpoint.

### Paired random-duration training

```bash
python "$REACT_ROOT/anchor_source/train_paired_random_trunc_final.py" \
    --study "$REACT_STUDY" \
    --out "$REACT_CONTROLS/random_duration" \
    --task-index 0 \
    --device cuda
```

Repeat `--task-index` from 0 through 83.

The two output arms are:

```text
react_rt
compact_rt
```

Both receive identical minibatch orders and sampled-duration schedules. One boundary is sampled uniformly from the evaluation grid for each minibatch. Each update uses one label-smoothed cross-entropy loss without an additional endpoint loss.

This script does not provide the primary trainer's mid-training `last.pt` resume mechanism.

### Fixed-anchor and endpoint-aligned-position controls

```bash
(
    cd "$REACT_ROOT/anchor_source"

    PYTHONPATH=. python -m fresh.anchor_diag \
        --orig-study "$REACT_STUDY" \
        --out "$REACT_CONTROLS/anchor" \
        --index 0 \
        --device cuda
)
```

Repeat `--index` from 0 through 83.

This script also trains an earlier forward-only `random_truncation` diagnostic. That diagnostic is not the final paired `react_rt` versus `compact_rt` experiment.

After all anchor tasks complete:

```bash
(
    cd "$REACT_ROOT/anchor_source"

    PYTHONPATH=. python -m fresh.anchor_report \
        --orig-study "$REACT_STUDY" \
        --diag-study "$REACT_CONTROLS/anchor"
)
```

### Mean-imputed forward-only evaluation and route interventions

```bash
python "$REACT_ROOT/anchor_source/reviewer_eval_controls.py" \
    --study "$REACT_STUDY" \
    --out "$REACT_CONTROLS/mean_imputation" \
    --dataset all \
    --device cuda
```

### Mean-imputed BiTE evaluation

```bash
python "$REACT_ROOT/anchor_source/bite_zerofill_final.py" \
    --study "$REACT_STUDY" \
    --out "$REACT_CONTROLS/bite_mean_imputation" \
    --device cuda
```

Mean imputation retains the full trained input length and replaces unavailable samples with zero in standardized space—the training-set mean. It does not insert future EEG.

Route interventions on a trained fused checkpoint and independently trained reverse-only models are different experiments and should be reported separately.

## 11. Generate statistics and the accuracy–time figure

### Primary accuracy–time summaries

```bash
python "$REACT_ROOT/anchor_source/fresh_primary_nauc.py" \
    "$REACT_STUDY"
```

### Summarize stored causality checks

```bash
python "$REACT_ROOT/anchor_source/check_fresh_causality.py" \
    "$REACT_STUDY"
```

This command summarizes existing `causality.json` records; it does not rerun the model tests. Keep the per-model maxima distinct from maxima pooled across models.

### Generate the four-model accuracy–time figure

```bash
mkdir -p "$REACT_CONTROLS/figures"

MPLBACKEND=Agg python \
    "$REACT_ROOT/anchor_source/make_fresh_anytime_figure.py" \
    "$REACT_STUDY" \
    "$REACT_CONTROLS/figures/fig2.pdf"
```

The figure compares REACT, No reverse, Two forward, and Forward mean. It averages seed accuracies within each subject and shades the across-subject mean ±1 sample standard deviation. Dotted lines indicate chance performance.

The later `make_final_figure2*.py` scripts use different input files and control combinations; they are not interchangeable with this generator.

### Later statistical summaries

The following scripts contain hardcoded paths and must be configured before execution:

| Script | Constants to update |
|---|---|
| `final_endpoint_stats.py` | `ORIG`, `OUT` |
| `summarize_reverse_only.py` | `ORIG`, `REV` |
| `summarize_paired_random_trunc_final.py` | `ORIG`, `RT`, `REV` |

Set:

```text
ORIG = the primary study directory
REV  = the reverse-only output directory
RT   = the paired random-duration output directory
OUT  = the desired endpoint-statistics JSON file
```

Edit these constants in a working source copy, not in an archived frozen study. Ensure the parent directory of `OUT` exists.

After updating the paths and completing the required experiments:

```bash
python "$REACT_ROOT/anchor_source/final_endpoint_stats.py"

python "$REACT_ROOT/anchor_source/summarize_reverse_only.py"

python "$REACT_ROOT/anchor_source/summarize_paired_random_trunc_final.py"
```

The paired random-duration summary also reads reverse-only results. Complete that experiment before running this combined summary.

## 12. Evaluation protocol

BCICIV-2A and BCICIV-2B are evaluated from 1 to 4 seconds at 28 observation boundaries. SD-SSVEP is evaluated from 0.25 to 1 second of its retained segment at 49 boundaries.

The same checkpoint is used at every boundary within a training regime. Accuracy is calculated for each seed, then averaged within subject; the checkpoints are not combined into a prediction ensemble.

Normalized accuracy–time area, or nAUC, is the trapezoidal area under the accuracy–time curve divided by the evaluated interval length. It is not ROC AUC. Because accuracy is expressed as a percentage, differences in nAUC are reported in percentage points.

The manuscript reports paired subject-level bootstrap confidence intervals using 10,000 resamples. Its specified sign-flip comparisons use Holm correction within separate nine-test families. The primary collector and the later statistical scripts cover different parts of this analysis.

## 13. Software tests

Run the tests in a working source copy, not in a frozen study's `code/` directory, because some tests write validation records.

```bash
(
    cd "$REACT_ROOT/main_source"

    PYTHONPATH=. \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    python -m pytest tests -q -k 'not full_reporting_layout'
)
```

The excluded legacy layout test writes to a fixed `/mnt/data` location.

These tests exercise software behavior, including model dimensions, paired initialization, pooling, normalization, checkpoint handling, and causality. Passing them does not establish reproduction of the manuscript's benchmark scores.

## Manuscript-reported results

The following values are reported in the associated manuscript. They are not recomputed from the source export.

### Endpoint-trained REACT

| Dataset | Earliest accuracy (%) | nAUC | Endpoint accuracy (%) | Parameters |
|---|---:|---:|---:|---:|
| BCICIV-2A | 69.32 | 80.12 | 85.06 | 18,916 |
| BCICIV-2B | 74.31 | 83.59 | 86.20 | 16,962 |
| SD-SSVEP | 34.89 | 64.15 | 96.50 | 20,140 |

“Earliest” means 1 second for BCICIV-2A/2B and 0.25 seconds for SD-SSVEP.

### Random-duration training: REACT minus No reverse

| Metric | BCICIV-2A | BCICIV-2B | SD-SSVEP |
|---|---:|---:|---:|
| Earliest accuracy | +15.34 | +3.47 | +29.83 |
| nAUC | +10.42 | +0.22 | +24.96 |
| Endpoint accuracy | +5.95 | −0.86 | +18.56 |

Differences are percentage points. These results use separately trained random-duration models, not the endpoint-trained checkpoints in the preceding table.

## Scope and limitations

Causality is defined from the prepared EEG input onward. It does not establish causal upstream acquisition or preprocessing. Both readers are recomputed for the supplied observation, so the implementation is not a constant-cost recurrent streaming model.

This export contains the core models, primary runner, and later control scripts. It does not include the executable raw-data preparation workflow, the LOSO runner, or the training workflow for the ten additional endpoint baselines reported in the manuscript. Complete benchmark reproduction also requires the dataset files and the corresponding run artifacts.

The exported BiTE clipping behavior differs from the manuscript description, as noted above. Preserve the original code and result provenance when resolving that discrepancy.

## Attribution

BiTE is an external baseline from [cindy-hong/BiteEEG](https://github.com/cindy-hong/BiteEEG). Its pinned revision is retrieved separately by the study setup code. Preserve its notices and cite the original work when using it.

Associated manuscript:

**REACT-EEG: Reverse-Encoded Anytime Causal Temporal Modeling for EEG Decoding**

Use the original dataset citations when reporting results on BCICIV-2A, BCICIV-2B, or SD-SSVEP.

No project-wide license is specified in this source export. Third-party code and datasets remain subject to their respective terms.
