import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
out = Path(sys.argv[2])

plan = json.loads((root / "study.json").read_text())

MODELS = [
    ("reader", "REACT"),
    ("compact", "No reverse"),
    ("ff", "Two forward"),
    ("compact_mean", "Fwd mean"),
]

DATASETS = [
    ("2a", "BCICIV-2A"),
    ("2b", "BCICIV-2B"),
    ("ssvep", "SD-SSVEP"),
]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))

for ax, (dataset, title) in zip(axes, DATASETS):
    cfg = plan["datasets"][dataset]
    subjects = range(1, cfg["subjects"] + 1)

    for dirname, label in MODELS:
        subject_curves = []
        lengths = None

        for subject in subjects:
            seed_curves = []

            for seed in plan["seeds"]:
                p = (
                    root / "runs"
                    / f"{dataset}_S{subject:02d}_seed{seed}"
                    / dirname / "result.json"
                )

                r = json.loads(p.read_text())

                if lengths is None:
                    lengths = np.asarray(r["lengths"], float)

                seed_curves.append(
                    np.asarray(r["accuracy_percent"], float)
                )

            subject_curves.append(
                np.mean(np.stack(seed_curves), axis=0)
            )

        subject_curves = np.stack(subject_curves)

        mean = subject_curves.mean(axis=0)
        sd = subject_curves.std(axis=0, ddof=1)

        time = lengths / cfg["fs"]

        line = ax.plot(time, mean, label=label)[0]
        ax.fill_between(
            time,
            mean - sd,
            mean + sd,
            alpha=0.12,
            color=line.get_color(),
            linewidth=0,
        )

    chance = 100.0 / cfg["classes"]
    ax.axhline(chance, linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Observed duration (s)")
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Accuracy (%)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
)

fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(out, bbox_inches="tight")
print(out)
