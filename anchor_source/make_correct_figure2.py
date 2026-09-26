from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REV = Path(
    "/lustre/smuexa01/client/users/nikkieh/"
    "reader_reverse_only_final/curves.csv"
)

CTRL = Path(
    "/lustre/smuexa01/client/users/nikkieh/"
    "reader_review_controls/final_fresh_20260922/"
    "reviewer_boundary_results.csv"
)

OUTPDF = Path("fig_exact_duration_FINAL_CORRECT.pdf")
OUTPNG = Path("fig_exact_duration_FINAL_CORRECT.png")

# ------------------------------------------------------------
# Load final results
# ------------------------------------------------------------
rev = pd.read_csv(REV)
ctrl = pd.read_csv(CTRL)

print("REVERSE COLUMNS:", list(rev.columns))
print("CONTROL COLUMNS:", list(ctrl.columns))

def norm_ds(x):
    s = str(x).strip().lower().replace("-", "").replace("_", "")
    if "ssvep" in s:
        return "ssvep"
    if "2a" in s:
        return "2a"
    if "2b" in s:
        return "2b"
    return s

rev["dataset"] = rev["dataset"].map(norm_ds)
ctrl["dataset"] = ctrl["dataset"].map(norm_ds)

# ------------------------------------------------------------
# Reviewer-control file uses:
# compact_truncated = endpoint-trained No reverse on short input
# compact_zerofill  = same checkpoint with full input geometry
# ------------------------------------------------------------
needed = [
    "dataset", "subject", "seed", "m", "time_s",
    "compact_truncated", "compact_zerofill"
]

missing = [c for c in needed if c not in ctrl.columns]
if missing:
    raise RuntimeError(f"Missing control columns: {missing}")

ctrl = ctrl[needed].copy()

for c in ["m", "time_s", "compact_truncated", "compact_zerofill"]:
    ctrl[c] = pd.to_numeric(ctrl[c], errors="coerce")

# Convert to percent only if stored as fractions
for c in ["compact_truncated", "compact_zerofill"]:
    if ctrl[c].dropna().max() <= 1.5:
        ctrl[c] *= 100.0

# ------------------------------------------------------------
# SUBJECT-FIRST aggregation:
# average 3 seeds within each subject first,
# then average across subjects
# ------------------------------------------------------------
subject_ctrl = (
    ctrl.groupby(
        ["dataset", "subject", "m", "time_s"],
        as_index=False
    )[["compact_truncated", "compact_zerofill"]]
    .mean()
)

ctrl_mean = (
    subject_ctrl.groupby(
        ["dataset", "m", "time_s"],
        as_index=False
    )[["compact_truncated", "compact_zerofill"]]
    .mean()
)

# ------------------------------------------------------------
# Sanity checks against manuscript
# ------------------------------------------------------------
expected = {
    "2a": {
        "react_early": 69.32,
        "reverse_early": 71.73,
        "norev_early": 32.10,
        "impute_early": 50.09,
        "react_end": 85.06,
        "reverse_end": 83.20,
        "norev_end": 82.34,
    },
    "2b": {
        "react_early": 74.31,
        "reverse_early": 76.06,
        "norev_early": 53.97,
        "impute_early": 67.90,
        "react_end": 86.20,
        "reverse_end": 86.39,
        "norev_end": 86.04,
    },
    "ssvep": {
        "react_early": 34.89,
        "reverse_early": 61.33,
        "norev_early": 11.61,
        "impute_early": 12.17,
        "react_end": 96.50,
        "reverse_end": 90.06,
        "norev_end": 93.50,
    },
}

def chk(label, got, target, tol=0.35):
    print(f"{label:28s}: {got:7.2f}   expected {target:7.2f}")
    if abs(got - target) > tol:
        raise RuntimeError(
            f"{label} failed: got {got:.3f}, expected ~{target:.2f}"
        )

print("\n===== VALIDATING FINAL CURVES =====")

for ds in ["2a", "2b", "ssvep"]:
    r = rev[rev["dataset"] == ds].sort_values("time_s")
    c = ctrl_mean[ctrl_mean["dataset"] == ds].sort_values("time_s")

    if len(r) == 0:
        raise RuntimeError(f"No reverse-only data for {ds}")
    if len(c) == 0:
        raise RuntimeError(f"No reviewer-control data for {ds}")

    chk(
        f"{ds} REACT early",
        float(r.iloc[0]["react_accuracy"]),
        expected[ds]["react_early"]
    )
    chk(
        f"{ds} Reverse early",
        float(r.iloc[0]["trained_reverse_only_accuracy"]),
        expected[ds]["reverse_early"]
    )
    chk(
        f"{ds} No reverse early",
        float(c.iloc[0]["compact_truncated"]),
        expected[ds]["norev_early"]
    )
    chk(
        f"{ds} Mean-imputed early",
        float(c.iloc[0]["compact_zerofill"]),
        expected[ds]["impute_early"]
    )
    chk(
        f"{ds} REACT endpoint",
        float(r.iloc[-1]["react_accuracy"]),
        expected[ds]["react_end"]
    )
    chk(
        f"{ds} Reverse endpoint",
        float(r.iloc[-1]["trained_reverse_only_accuracy"]),
        expected[ds]["reverse_end"]
    )
    chk(
        f"{ds} No reverse endpoint",
        float(c.iloc[-1]["compact_truncated"]),
        expected[ds]["norev_end"]
    )

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
titles = {
    "2a": "BCICIV-2A",
    "2b": "BCICIV-2B",
    "ssvep": "SD-SSVEP",
}

chance = {
    "2a": 25.0,
    "2b": 50.0,
    "ssvep": 100.0 / 12.0,
}

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.15, 2.55),
    sharey=True
)

for ax, ds in zip(axes, ["2a", "2b", "ssvep"]):

    r = (
        rev[rev["dataset"] == ds]
        .sort_values("time_s")
    )

    c = (
        ctrl_mean[ctrl_mean["dataset"] == ds]
        .sort_values("time_s")
    )

    ax.plot(
        r["time_s"],
        r["react_accuracy"],
        linewidth=2.2,
        label="REACT"
    )

    ax.plot(
        r["time_s"],
        r["trained_reverse_only_accuracy"],
        linewidth=1.9,
        linestyle="--",
        label="Reverse only"
    )

    ax.plot(
        c["time_s"],
        c["compact_truncated"],
        linewidth=1.7,
        linestyle="-.",
        label="No reverse (truncated)"
    )

    ax.plot(
        c["time_s"],
        c["compact_zerofill"],
        linewidth=1.7,
        linestyle=":",
        label="Mean-imputed No reverse"
    )

    ax.axhline(
        chance[ds],
        linewidth=1.0,
        linestyle=(0, (2, 2)),
        label="Chance"
    )

    ax.set_title(titles[ds], fontsize=9)
    ax.set_xlabel("Observation duration (s)", fontsize=8)
    ax.set_ylim(0, 100)

    if ds in ["2a", "2b"]:
        ax.set_xlim(1.0, 4.0)
        ax.set_xticks([1, 2, 3, 4])
    else:
        ax.set_xlim(0.25, 1.0)
        ax.set_xticks([0.25, 0.50, 0.75, 1.00])

    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)

axes[0].set_ylabel("Accuracy (%)", fontsize=8)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=5,
    frameon=False,
    fontsize=6.5,
    handlelength=2.4,
    columnspacing=0.9
)

fig.subplots_adjust(
    left=0.075,
    right=0.995,
    bottom=0.20,
    top=0.79,
    wspace=0.12
)

fig.savefig(OUTPDF, bbox_inches="tight")
fig.savefig(OUTPNG, dpi=600, bbox_inches="tight")

print("\n===== CORRECT FIGURE SAVED =====")
print(OUTPDF.resolve())
print(OUTPNG.resolve())
