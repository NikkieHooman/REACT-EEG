from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Final result sources
# ============================================================

REV_PATH = Path(
    "/lustre/smuexa01/client/users/nikkieh/"
    "reader_reverse_only_final/curves.csv"
)

CTRL_PATH = Path(
    "/lustre/smuexa01/client/users/nikkieh/"
    "reader_review_controls/final_fresh_20260922/"
    "reviewer_boundary_results.csv"
)

OUT_PDF = Path("fig_exact_duration_FINAL.pdf")
OUT_PNG = Path("fig_exact_duration_FINAL.png")


# ============================================================
# Utilities
# ============================================================

def pick_column(df, candidates, description):
    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    raise RuntimeError(
        f"\nCould not identify {description} column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def normalize_dataset(x):
    s = str(x).strip().lower()
    s = s.replace("_", "").replace("-", "")

    if s in {"2a", "bciciv2a", "bciiv2a", "bcicivii2a"}:
        return "2a"
    if s in {"2b", "bciciv2b", "bciiv2b", "bcicivii2b"}:
        return "2b"
    if "ssvep" in s:
        return "ssvep"

    return str(x).strip().lower()


def to_percent(s):
    s = pd.to_numeric(s, errors="coerce")

    # If values look like proportions rather than percentages.
    if s.dropna().max() <= 1.5:
        s = 100.0 * s

    return s


# ============================================================
# Load trained reverse-only final curves
# These already contain the final subject-first mean REACT and
# independently trained reverse-only trajectories.
# ============================================================

if not REV_PATH.exists():
    raise FileNotFoundError(REV_PATH)

rev = pd.read_csv(REV_PATH)
rev["dataset"] = rev["dataset"].map(normalize_dataset)

required_rev = {
    "dataset",
    "n_samples",
    "time_s",
    "react_accuracy",
    "trained_reverse_only_accuracy",
}

missing = required_rev - set(rev.columns)
if missing:
    raise RuntimeError(
        f"Reverse-only curves missing columns: {sorted(missing)}"
    )

rev["react_accuracy"] = to_percent(rev["react_accuracy"])
rev["trained_reverse_only_accuracy"] = to_percent(
    rev["trained_reverse_only_accuracy"]
)


# ============================================================
# Load reviewer control runs
# ============================================================

if not CTRL_PATH.exists():
    raise FileNotFoundError(
        f"\nExpected reviewer control file not found:\n{CTRL_PATH}"
    )

ctrl = pd.read_csv(CTRL_PATH)

print("\nReviewer-control columns:")
print(list(ctrl.columns))


dataset_col = pick_column(
    ctrl,
    ["dataset", "data", "dataset_name"],
    "dataset",
)

subject_col = pick_column(
    ctrl,
    ["subject", "subject_id", "subj", "sid"],
    "subject",
)

seed_col = pick_column(
    ctrl,
    ["seed", "random_seed"],
    "seed",
)

sample_col = pick_column(
    ctrl,
    ["n_samples", "samples", "m", "boundary_samples"],
    "observation-sample",
)

# The original forward-only model was historically called Compact
# in these control files.
norev_col = pick_column(
    ctrl,
    [
        "compact",
        "no_reverse",
        "noreverse",
        "compact_accuracy",
        "no_reverse_accuracy",
    ],
    "directly truncated No-reverse accuracy",
)

zero_col = pick_column(
    ctrl,
    [
        "compact_zerofill",
        "compact_zero_fill",
        "no_reverse_zerofill",
        "no_reverse_zero_fill",
        "mean_imputed_no_reverse",
    ],
    "mean-imputed No-reverse accuracy",
)


print("\nUsing reviewer-control columns:")
print(" dataset      :", dataset_col)
print(" subject      :", subject_col)
print(" seed         :", seed_col)
print(" samples      :", sample_col)
print(" No reverse   :", norev_col)
print(" mean-imputed :", zero_col)


ctrl = ctrl[
    [
        dataset_col,
        subject_col,
        seed_col,
        sample_col,
        norev_col,
        zero_col,
    ]
].copy()

ctrl.columns = [
    "dataset",
    "subject",
    "seed",
    "n_samples",
    "no_reverse_accuracy",
    "mean_imputed_accuracy",
]

ctrl["dataset"] = ctrl["dataset"].map(normalize_dataset)
ctrl["n_samples"] = pd.to_numeric(ctrl["n_samples"], errors="coerce")
ctrl["no_reverse_accuracy"] = to_percent(ctrl["no_reverse_accuracy"])
ctrl["mean_imputed_accuracy"] = to_percent(
    ctrl["mean_imputed_accuracy"]
)

ctrl = ctrl.dropna(
    subset=[
        "dataset",
        "subject",
        "seed",
        "n_samples",
        "no_reverse_accuracy",
        "mean_imputed_accuracy",
    ]
)


# ============================================================
# IMPORTANT:
# manuscript aggregation is:
#
#   seed -> average WITHIN subject
#   subject -> average across subjects
#
# Seeds are NOT independent observations.
# ============================================================

subject_curves = (
    ctrl
    .groupby(
        ["dataset", "subject", "n_samples"],
        as_index=False
    )[
        [
            "no_reverse_accuracy",
            "mean_imputed_accuracy",
        ]
    ]
    .mean()
)

control_curves = (
    subject_curves
    .groupby(
        ["dataset", "n_samples"],
        as_index=False
    )[
        [
            "no_reverse_accuracy",
            "mean_imputed_accuracy",
        ]
    ]
    .mean()
)


# ============================================================
# Attach exact observation times using the final reverse-only grid
# ============================================================

time_grid = (
    rev[
        ["dataset", "n_samples", "time_s"]
    ]
    .drop_duplicates()
)

control_curves = control_curves.merge(
    time_grid,
    on=["dataset", "n_samples"],
    how="inner",
    validate="one_to_one",
)


# ============================================================
# Sanity checks against manuscript values
# ============================================================

expected = {
    "2a": {
        "react_early": 69.32,
        "reverse_early": 71.73,
        "no_reverse_early": 32.10,
        "imputed_early": 50.09,
        "react_endpoint": 85.06,
        "reverse_endpoint": 83.20,
        "no_reverse_endpoint": 82.34,
    },
    "2b": {
        "react_early": 74.31,
        "reverse_early": 76.06,
        "no_reverse_early": 53.97,
        "imputed_early": 67.90,
        "react_endpoint": 86.20,
        "reverse_endpoint": 86.39,
        "no_reverse_endpoint": 86.04,
    },
    "ssvep": {
        "react_early": 34.89,
        "reverse_early": 61.33,
        "no_reverse_early": 11.61,
        "imputed_early": 12.17,
        "react_endpoint": 96.50,
        "reverse_endpoint": 90.06,
        "no_reverse_endpoint": 93.50,
    },
}


def check_close(name, actual, target, tol=0.30):
    if abs(actual - target) > tol:
        raise RuntimeError(
            f"{name}: found {actual:.4f}, "
            f"expected approximately {target:.2f}"
        )


print("\n" + "=" * 70)
print("ANCHOR-VALUE VALIDATION")
print("=" * 70)

for ds in ["2a", "2b", "ssvep"]:

    r = (
        rev[rev["dataset"] == ds]
        .sort_values("n_samples")
        .reset_index(drop=True)
    )

    c = (
        control_curves[control_curves["dataset"] == ds]
        .sort_values("n_samples")
        .reset_index(drop=True)
    )

    if len(r) == 0 or len(c) == 0:
        raise RuntimeError(f"No curve data found for {ds}")

    vals = {
        "react_early": float(r.iloc[0]["react_accuracy"]),
        "reverse_early": float(
            r.iloc[0]["trained_reverse_only_accuracy"]
        ),
        "no_reverse_early": float(
            c.iloc[0]["no_reverse_accuracy"]
        ),
        "imputed_early": float(
            c.iloc[0]["mean_imputed_accuracy"]
        ),
        "react_endpoint": float(r.iloc[-1]["react_accuracy"]),
        "reverse_endpoint": float(
            r.iloc[-1]["trained_reverse_only_accuracy"]
        ),
        "no_reverse_endpoint": float(
            c.iloc[-1]["no_reverse_accuracy"]
        ),
    }

    print(f"\n{ds.upper()}")
    for key, value in vals.items():
        print(f"  {key:22s}: {value:7.2f}")
        check_close(
            f"{ds}/{key}",
            value,
            expected[ds][key],
        )


# ============================================================
# Figure
# ============================================================

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
    1,
    3,
    figsize=(7.15, 2.45),
    sharey=True,
)

handles = None
labels = None

for ax, ds in zip(axes, ["2a", "2b", "ssvep"]):

    r = (
        rev[rev["dataset"] == ds]
        .sort_values("n_samples")
        .copy()
    )

    c = (
        control_curves[control_curves["dataset"] == ds]
        .sort_values("n_samples")
        .copy()
    )

    # Final proposed model
    ax.plot(
        r["time_s"],
        r["react_accuracy"],
        linewidth=2.0,
        label="REACT",
    )

    # Independently trained reverse-only model
    ax.plot(
        r["time_s"],
        r["trained_reverse_only_accuracy"],
        linewidth=1.8,
        linestyle="--",
        label="Reverse only",
    )

    # Endpoint-trained forward-only, physically truncated
    ax.plot(
        c["time_s"],
        c["no_reverse_accuracy"],
        linewidth=1.6,
        linestyle="-.",
        label="No reverse (truncated)",
    )

    # Same forward-only checkpoint, original full input geometry
    ax.plot(
        c["time_s"],
        c["mean_imputed_accuracy"],
        linewidth=1.6,
        label="Mean-imputed No reverse",
    )

    # Dataset chance accuracy
    ax.axhline(
        chance[ds],
        linewidth=1.0,
        linestyle=":",
        label="Chance" if ds == "2a" else None,
    )

    ax.set_title(titles[ds], fontsize=9)
    ax.set_xlabel("Observation duration (s)", fontsize=8)

    ax.set_ylim(0, 100)

    if ds in {"2a", "2b"}:
        ax.set_xlim(1.0, 4.0)
        ax.set_xticks([1, 2, 3, 4])
    else:
        ax.set_xlim(0.25, 1.0)
        ax.set_xticks([0.25, 0.50, 0.75, 1.00])

    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", linewidth=0.4, alpha=0.25)

    if handles is None:
        handles, labels = ax.get_legend_handles_labels()


axes[0].set_ylabel("Accuracy (%)", fontsize=8)

# Use one shared legend rather than three duplicate legends.
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=5,
    frameon=False,
    fontsize=6.8,
    columnspacing=1.1,
    handlelength=2.5,
)

fig.subplots_adjust(
    left=0.075,
    right=0.995,
    bottom=0.20,
    top=0.78,
    wspace=0.12,
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
)

print("\n" + "=" * 70)
print("FINAL FIGURE WRITTEN")
print("=" * 70)
print(OUT_PDF.resolve())
print(OUT_PNG.resolve())
