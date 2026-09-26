import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
plan = json.loads((root / "study.json").read_text())

MODELS = {
    "REACT": "reader",
    "No reverse": "compact",
    "Two forward": "ff",
    "Fwd mean": "compact_mean",
}

SEED = 170926
NBOOT = 10000


def load_result(dataset, subject, seed, model):
    p = (
        root / "runs"
        / f"{dataset}_S{subject:02d}_seed{seed}"
        / model / "result.json"
    )
    return json.loads(p.read_text())


def nauc(acc, lengths, fs):
    a = np.asarray(acc, float)
    t = np.asarray(lengths, float) / float(fs)
    return float(np.trapz(a, t) / (t[-1] - t[0]))


def bootstrap(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(NBOOT, len(x)))
    b = x[idx].mean(axis=1)
    return (
        float(x.mean()),
        float(np.percentile(b, 2.5)),
        float(np.percentile(b, 97.5)),
    )


for dataset in ("2a", "2b", "ssvep"):
    cfg = plan["datasets"][dataset]
    subjects = range(1, cfg["subjects"] + 1)
    seeds = plan["seeds"]

    subject_curves = {m: [] for m in MODELS}

    lengths = None

    for subject in subjects:
        for label, dirname in MODELS.items():
            seed_curves = []

            for seed in seeds:
                r = load_result(dataset, subject, seed, dirname)

                if lengths is None:
                    lengths = r["lengths"]

                seed_curves.append(
                    np.asarray(r["accuracy_percent"], float)
                )

            subject_curves[label].append(
                np.mean(np.stack(seed_curves), axis=0)
            )

    print("\n========================================")
    print(dataset.upper())
    print("========================================")

    subject_nauc = {}

    for label in MODELS:
        curves = np.stack(subject_curves[label])

        subject_nauc[label] = np.asarray([
            nauc(row, lengths, cfg["fs"])
            for row in curves
        ])

        print(
            f"{label:12s} "
            f"early={curves[:,0].mean():7.3f} "
            f"endpoint={curves[:,-1].mean():7.3f} "
            f"nAUC={subject_nauc[label].mean():7.3f}"
        )

    print("\nPaired REACT minus control nAUC:")

    for label in ("No reverse", "Two forward", "Fwd mean"):
        d = subject_nauc["REACT"] - subject_nauc[label]
        mean, lo, hi = bootstrap(d)

        print(
            f"REACT - {label:11s}: "
            f"{mean:+.2f} [{lo:.2f}, {hi:.2f}]"
        )
