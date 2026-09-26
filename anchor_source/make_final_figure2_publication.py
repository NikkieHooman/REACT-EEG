from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL DATA SOURCES
# ============================================================

REV = Path(
    "/lustre/smuexa01/client/users/nikkieh/"
    "reader_reverse_only_final/curves.csv"
)

CTRL = Path(
    "/lustre/smuexa01/client/users/nikkieh/"
    "reader_review_controls/final_fresh_20260922/"
    "reviewer_boundary_results.csv"
)

OUT_PDF = Path("fig_exact_duration_FINAL_PUBLICATION.pdf")
OUT_PNG = Path("fig_exact_duration_FINAL_PUBLICATION.png")


# ============================================================
# LOAD
# ============================================================

rev = pd.read_csv(REV)
ctrl = pd.read_csv(CTRL)


def norm_dataset(x):
    s = str(x).strip().lower().replace("-", "").replace("_", "")
    if "ssvep" in s:
        return "ssvep"
    if "2a" in s:
        return "2a"
    if "2b" in s:
        return "2b"
    return s


rev["dataset"] = rev["dataset"].map(norm_dataset)
ctrl["dataset"] = ctrl["dataset"].map(norm_dataset)


# ============================================================
# CONTROL CURVES
#
# compact_truncated:
# endpoint-trained No reverse evaluated on physically shortened input
#
# compact_zerofill:
# same endpoint-trained No reverse checkpoint, but unavailable
# standardized samples are replaced with zero (= training mean)
# while preserving full input length
# ============================================================

required_ctrl = [
    "dataset",
    "subject",
    "seed",
    "m",
    "time_s",
    "compact_truncated",
    "compact_zerofill",
]

missing = [c for c in required_ctrl if c not in ctrl.columns]
if missing:
    raise RuntimeError(f"Missing control columns: {missing}")

ctrl = ctrl[required_ctrl].copy()

for col in ["m", "time_s", "compact_truncated", "compact_zerofill"]:
    ctrl[col] = pd.to_numeric(ctrl[col], errors="coerce")

for col in ["compact_truncated", "compact_zerofill"]:
    if ctrl[col].dropna().max() <= 1.5:
        ctrl[col] *= 100.0


# ============================================================
# SUBJECT-FIRST AGGREGATION
#
# Exactly matches manuscript:
#   1. average 3 optimization seeds within each subject
#   2. average subject values
# ============================================================

subject_ctrl = (
    ctrl
    .groupby(
        ["dataset", "subject", "m", "time_s"],
        as_index=False
    )[["compact_truncated", "compact_zerofill"]]
    .mean()
)

ctrl_mean = (
    subject_ctrl
    .groupby(
        ["dataset", "m", "time_s"],
        as_index=False
    )[["compact_truncated", "compact_zerofill"]]
    .mean()
)


# ============================================================
# VALIDATE AGAINST FINAL MANUSCRIPT VALUES
# ============================================================

expected = {
    "2a": {
        "react_early": 69.32,
        "reverse_early": 71.73,
        "norev_early": 32.10,
        "imputed_early": 50.09,
        "react_endpoint": 85.06,
        "reverse_endpoint": 83.20,
        "norev_endpoint": 82.34,
    },
    "2b": {
        "react_early": 74.31,
        "reverse_early": 76.06,
        "norev_early": 53.97,
        "imputed_early": 67.90,
        "react_endpoint": 86.20,
        "reverse_endpoint": 86.39,
        "norev_endpoint": 86.04,
    },
    "ssvep": {
        "react_early": 34.89,
        "reverse_early": 61.33,
        "norev_early": 11.61,
        "imputed_early": 12.17,
        "react_endpoint": 96.50,
        "reverse_endpoint": 90.06,
        "norev_endpoint": 93.50,
    },
}


def check(label, actual, target, tol=0.35):
    print(f"{label:31s} {actual:7.2f}   expected {target:7.2f}")
    if abs(actual - target) > tol:
        raise RuntimeError(
            f"\nFAILED NUMERIC CHECK: {label}\n"
            f"actual={actual:.4f}, expected≈{target:.2f}"
        )


print("\n" + "=" * 78)
print("FINAL NUMERIC VALIDATION")
print("=" * 78)

for ds in ["2a", "2b", "ssvep"]:

    r = (
        rev[rev["dataset"] == ds]
        .sort_values("time_s")
        .reset_index(drop=True)
    )

    c = (
        ctrl_mean[ctrl_mean["dataset"] == ds]
        .sort_values("time_s")
        .reset_index(drop=True)
    )

    check(
        f"{ds} REACT early",
        float(r.iloc[0]["react_accuracy"]),
        expected[ds]["react_early"],
    )

    check(
        f"{ds} Reverse-only early",
        float(r.iloc[0]["trained_reverse_only_accuracy"]),
        expected[ds]["reverse_early"],
    )

    check(
        f"{ds} No reverse early",
        float(c.iloc[0]["compact_truncated"]),
        expected[ds]["norev_early"],
    )

    check(
        f"{ds} Mean-imputed early",
        float(c.iloc[0]["compact_zerofill"]),
        expected[ds]["imputed_early"],
    )

    check(
        f"{ds} REACT endpoint",
        float(r.iloc[-1]["react_accuracy"]),
        expected[ds]["react_endpoint"],
    )

    check(
        f"{ds} Reverse-only endpoint",
        float(r.iloc[-1]["trained_reverse_only_accuracy"]),
        expected[ds]["reverse_endpoint"],
    )

    check(
        f"{ds} No reverse endpoint",
        float(c.iloc[-1]["compact_truncated"]),
        expected[ds]["norev_endpoint"],
    )

    # This MUST hold at m = T:
    # there is nothing left to mean-impute.
    end_nr = float(c.iloc[-1]["compact_truncated"])
    end_imp = float(c.iloc[-1]["compact_zerofill"])

    if abs(end_nr - end_imp) > 1e-8:
        raise RuntimeError(
            f"{ds}: mean-imputed and ordinary No reverse "
            f"do not converge at endpoint: "
            f"{end_nr:.8f} vs {end_imp:.8f}"
        )

    print(
        f"{ds} endpoint convergence       "
        f"{end_imp:7.2f}   ✓"
    )


# ============================================================
# PUBLICATION FIGURE
# ============================================================

titles = {
    "2a": "BCICIV-2A   (4 classes)",
    "2b": "BCICIV-2B   (2 classes)",
    "ssvep": "SD-SSVEP   (12 classes)",
}

chance = {
    "2a": 25.0,
    "2b": 50.0,
    "ssvep": 100.0 / 12.0,
}

# Explicit plotting colors for consistency with your target format
C_REACT = "#1976D2"
C_REVERSE = "#F57C00"
C_NOREV = "#2E9D3D"
C_IMPUTE = "#D62728"
C_CHANCE = "#555555"

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.6, 4.25),
    sharey=True,
)

fig.patch.set_facecolor("white")


for ax, ds in zip(axes, ["2a", "2b", "ssvep"]):

    r = (
        rev[rev["dataset"] == ds]
        .sort_values("time_s")
        .reset_index(drop=True)
    )

    c = (
        ctrl_mean[ctrl_mean["dataset"] == ds]
        .sort_values("time_s")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Full exact trajectories
    # --------------------------------------------------------

    # Sparse visual markers only.
    # The line itself still contains ALL 28/49 actual points.
    if ds == "ssvep":
        marker_step = 4
    else:
        marker_step = 3

    ax.plot(
        r["time_s"],
        r["react_accuracy"],
        color=C_REACT,
        linewidth=2.6,
        marker="o",
        markersize=4.1,
        markevery=marker_step,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="REACT",
        zorder=5,
    )

    ax.plot(
        r["time_s"],
        r["trained_reverse_only_accuracy"],
        color=C_REVERSE,
        linewidth=2.3,
        linestyle="--",
        marker="o",
        markersize=4.0,
        markevery=marker_step,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="Reverse only",
        zorder=4,
    )

    ax.plot(
        c["time_s"],
        c["compact_truncated"],
        color=C_NOREV,
        linewidth=2.2,
        linestyle="-.",
        marker="D",
        markersize=3.6,
        markevery=marker_step,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="No reverse (truncated)",
        zorder=3,
    )

    ax.plot(
        c["time_s"],
        c["compact_zerofill"],
        color=C_IMPUTE,
        linewidth=2.2,
        linestyle=":",
        marker="s",
        markersize=3.5,
        markevery=marker_step,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="Mean-imputed No reverse",
        zorder=4,
    )

    # --------------------------------------------------------
    # Chance line
    # --------------------------------------------------------

    ax.axhline(
        chance[ds],
        color=C_CHANCE,
        linewidth=1.15,
        linestyle=(0, (1.5, 2.5)),
        label="Chance",
        zorder=1,
    )

    # Chance label
    if ds in ["2a", "2b"]:
        x_chance = 3.75
    else:
        x_chance = 0.96

    ax.text(
        x_chance,
        chance[ds] + 1.5,
        "chance",
        ha="right",
        va="bottom",
        fontsize=8,
        color=C_CHANCE,
    )

    # --------------------------------------------------------
    # Early-gap annotation:
    # REACT - directly truncated No reverse
    # --------------------------------------------------------

    y_react = float(r.iloc[0]["react_accuracy"])
    y_norev = float(c.iloc[0]["compact_truncated"])
    gap = y_react - y_norev

    x0 = float(r.iloc[0]["time_s"])

    # Double-headed vertical arrow
    ax.annotate(
        "",
        xy=(x0, y_react - 0.5),
        xytext=(x0, y_norev + 0.5),
        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.35,
            color="black",
        ),
        zorder=10,
    )

    if ds == "2a":
        txt_x = x0 + 0.14
        txt_y = (y_react + y_norev) / 2 + 1.5
        label = f"+{gap:.1f} pts\nat 1 s"

    elif ds == "2b":
        txt_x = x0 + 0.14
        txt_y = (y_react + y_norev) / 2 + 1.0
        label = f"+{gap:.1f} pts\nat 1 s"

    else:
        txt_x = x0 + 0.035
        txt_y = (y_react + y_norev) / 2 - 1.0
        label = f"+{gap:.1f} pts\nat 0.25 s"

    ax.text(
        txt_x,
        txt_y,
        label,
        fontsize=8.5,
        ha="left",
        va="center",
        color="black",
    )

    # --------------------------------------------------------
    # Axis styling
    # --------------------------------------------------------

    ax.set_title(
        titles[ds],
        fontsize=11,
        pad=10,
        fontweight="semibold",
    )

    ax.set_ylim(0, 100)

    if ds in ["2a", "2b"]:
        ax.set_xlim(0.88, 4.12)
        ax.set_xticks([1, 2, 3, 4])
    else:
        ax.set_xlim(0.225, 1.015)
        ax.set_xticks([0.25, 0.50, 0.75, 1.00])

    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.tick_params(
        axis="both",
        labelsize=8.5,
        width=0.8,
        length=3.5,
    )

    ax.grid(
        True,
        which="major",
        axis="both",
        linewidth=0.55,
        alpha=0.20,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


axes[0].set_ylabel(
    "Test accuracy (%)",
    fontsize=10,
)

fig.supxlabel(
    "EEG observed before the decision (s)",
    fontsize=10.5,
    y=0.035,
)


# ============================================================
# SHARED LEGEND
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

# Chance appears once.
unique = {}
for h, label in zip(handles, labels):
    if label not in unique:
        unique[label] = h

legend_order = [
    "REACT",
    "Reverse only",
    "No reverse (truncated)",
    "Mean-imputed No reverse",
    "Chance",
]

fig.legend(
    [unique[x] for x in legend_order],
    legend_order,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.015),
    ncol=5,
    frameon=False,
    fontsize=8.4,
    handlelength=2.7,
    columnspacing=1.25,
    handletextpad=0.55,
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.065,
    right=0.995,
    bottom=0.16,
    top=0.82,
    wspace=0.08,
)


# ============================================================
# SAVE VECTOR PDF + HIGH-RES PNG
# ============================================================

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    pad_inches=0.04,
)

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.04,
)

print("\n" + "=" * 78)
print("FINAL PUBLICATION FIGURE SAVED")
print("=" * 78)
print(OUT_PDF.resolve())
print(OUT_PNG.resolve())
