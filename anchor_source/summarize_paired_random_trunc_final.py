#!/usr/bin/env python3

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ORIG = Path(
    "/lustre/smuexa01/client/users/nikkieh/reader_fresh_runs/"
    "study_20260918T005712Z_qDm6ij"
)

RT = Path(
    "/lustre/smuexa01/client/users/nikkieh/"
    "reader_random_trunc_paired_final"
)

REV = Path(
    "/lustre/smuexa01/client/users/nikkieh/"
    "reader_reverse_only_final"
)

DATASETS = {
    "2a": {"subjects": 9, "fs": 250},
    "2b": {"subjects": 9, "fs": 250},
    "ssvep": {"subjects": 10, "fs": 256},
}

SEEDS = [2025, 2026, 2027]
B = 10000
BOOT_SEED = 230926

METRICS = ("early", "nauc", "endpoint")

COMPARISONS = {
    "react_rt_minus_compact_rt": (
        "react_rt",
        "compact_rt",
    ),
    "react_rt_minus_original_react": (
        "react_rt",
        "original_react",
    ),
    "compact_rt_minus_original_compact": (
        "compact_rt",
        "original_compact",
    ),
    "react_rt_minus_reverse_only": (
        "react_rt",
        "reverse_only",
    ),
}


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def nauc(curve, lengths, fs):
    curve = np.asarray(curve, dtype=float)
    t = np.asarray(lengths, dtype=float) / float(fs)
    return float(
        np.trapezoid(curve, t)
        / (t[-1] - t[0])
    )


def metrics(curve, lengths, fs):
    return {
        "early": float(curve[0]),
        "nauc": nauc(curve, lengths, fs),
        "endpoint": float(curve[-1]),
    }


def bootstrap_ci(diff, seed):
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)

    ix = rng.integers(
        0,
        len(diff),
        size=(B, len(diff)),
    )

    boot = diff[ix].mean(axis=1)

    return [
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]


def exact_signflip_p(diff):
    diff = np.asarray(diff, dtype=float)

    observed = abs(diff.mean())

    stats = []
    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=len(diff),
    ):
        signs = np.asarray(signs)
        stats.append(
            abs(np.mean(diff * signs))
        )

    stats = np.asarray(stats)

    return float(
        np.mean(stats >= observed - 1e-12)
    )


def holm_adjust(raw_ps):
    raw_ps = np.asarray(raw_ps, dtype=float)

    order = np.argsort(raw_ps)
    m = len(raw_ps)

    adjusted = np.empty(m, dtype=float)

    running = 0.0

    for rank, idx in enumerate(order):
        candidate = min(
            1.0,
            (m - rank) * raw_ps[idx],
        )

        running = max(
            running,
            candidate,
        )

        adjusted[idx] = running

    return adjusted


# --------------------------------------------------------------
# Verify full output inventory.
# --------------------------------------------------------------

for arm in ("compact_rt", "react_rt"):
    paths = list(
        RT.glob(
            f"runs/*/{arm}/result.json"
        )
    )

    print(
        f"{arm}: found {len(paths)} results"
    )

    if len(paths) != 84:
        raise RuntimeError(
            f"{arm}: expected 84, found {len(paths)}"
        )


subject_rows = []
curve_rows = []

# --------------------------------------------------------------
# Load, validate pairing, then seed-average within subject.
# --------------------------------------------------------------

for ds, cfg in DATASETS.items():

    print("\n" + "=" * 80)
    print("DATASET:", ds)

    for subject in range(
        1,
        cfg["subjects"] + 1,
    ):

        seed_curves = {
            "compact_rt": [],
            "react_rt": [],
            "original_compact": [],
            "original_react": [],
            "reverse_only": [],
        }

        reference_lengths = None

        for seed in SEEDS:

            name = (
                f"{ds}_S{subject:02d}_seed{seed}"
            )

            compact_rt = load_json(
                RT / "runs" / name
                / "compact_rt"
                / "result.json"
            )

            react_rt = load_json(
                RT / "runs" / name
                / "react_rt"
                / "result.json"
            )

            original_compact = load_json(
                ORIG / "runs" / name
                / "compact"
                / "result.json"
            )

            original_react = load_json(
                ORIG / "runs" / name
                / "reader"
                / "result.json"
            )

            reverse_only = load_json(
                REV / "runs" / name
                / "reverse_only"
                / "result.json"
            )

            # Pairing validation.
            if (
                compact_rt["length_schedule_hash"]
                != react_rt["length_schedule_hash"]
            ):
                raise AssertionError(
                    f"Duration schedule mismatch: {name}"
                )

            if (
                compact_rt["minibatch_order_hash"]
                != react_rt["minibatch_order_hash"]
            ):
                raise AssertionError(
                    f"Minibatch order mismatch: {name}"
                )

            if (
                compact_rt["length_counts"]
                != react_rt["length_counts"]
            ):
                raise AssertionError(
                    f"Length-count mismatch: {name}"
                )

            lengths = np.asarray(
                compact_rt["lengths"],
                dtype=int,
            )

            for other in (
                react_rt,
                original_compact,
                original_react,
                reverse_only,
            ):
                if list(lengths) != list(
                    other["lengths"]
                ):
                    raise AssertionError(
                        f"Grid mismatch: {name}"
                    )

            if reference_lengths is None:
                reference_lengths = lengths
            elif not np.array_equal(
                reference_lengths,
                lengths,
            ):
                raise AssertionError(
                    f"Grid differs across seeds: {name}"
                )

            seed_curves["compact_rt"].append(
                np.asarray(
                    compact_rt[
                        "accuracy_percent"
                    ],
                    dtype=float,
                )
            )

            seed_curves["react_rt"].append(
                np.asarray(
                    react_rt[
                        "accuracy_percent"
                    ],
                    dtype=float,
                )
            )

            seed_curves[
                "original_compact"
            ].append(
                np.asarray(
                    original_compact[
                        "accuracy_percent"
                    ],
                    dtype=float,
                )
            )

            seed_curves[
                "original_react"
            ].append(
                np.asarray(
                    original_react[
                        "accuracy_percent"
                    ],
                    dtype=float,
                )
            )

            seed_curves[
                "reverse_only"
            ].append(
                np.asarray(
                    reverse_only[
                        "accuracy_percent"
                    ],
                    dtype=float,
                )
            )

        subject_curves = {
            arm: np.stack(curves).mean(axis=0)
            for arm, curves
            in seed_curves.items()
        }

        subject_metrics = {
            arm: metrics(
                curve,
                reference_lengths,
                cfg["fs"],
            )
            for arm, curve
            in subject_curves.items()
        }

        row = {
            "dataset": ds,
            "subject": subject,
        }

        for arm, vals in subject_metrics.items():
            for metric, value in vals.items():
                row[
                    f"{arm}_{metric}"
                ] = value

        for comparison, (a, b) in (
            COMPARISONS.items()
        ):
            for metric in METRICS:
                row[
                    f"{comparison}_{metric}"
                ] = (
                    subject_metrics[a][metric]
                    - subject_metrics[b][metric]
                )

        subject_rows.append(row)

        # Save subject-level mean curves too.
        for j, m in enumerate(
            reference_lengths
        ):
            curve_row = {
                "dataset": ds,
                "subject": subject,
                "n_samples": int(m),
                "time_s": float(
                    m / cfg["fs"]
                ),
            }

            for arm, curve in (
                subject_curves.items()
            ):
                curve_row[arm] = float(
                    curve[j]
                )

            curve_rows.append(
                curve_row
            )


df = pd.DataFrame(subject_rows)

# --------------------------------------------------------------
# Statistics.
# Each comparison family gets its own Holm correction
# across 3 datasets x 3 metrics = 9 paired tests.
# --------------------------------------------------------------

summary = {}

for family_index, (
    comparison,
    (arm_a, arm_b),
) in enumerate(COMPARISONS.items()):

    family_results = []
    raw_ps = []

    for ds_index, ds in enumerate(
        DATASETS
    ):
        d = df[
            df["dataset"] == ds
        ].copy()

        for metric_index, metric in (
            enumerate(METRICS)
        ):

            a = d[
                f"{arm_a}_{metric}"
            ].to_numpy(float)

            b = d[
                f"{arm_b}_{metric}"
            ].to_numpy(float)

            diff = a - b

            seed = (
                BOOT_SEED
                + family_index * 1000
                + ds_index * 100
                + metric_index
            )

            raw_p = exact_signflip_p(
                diff
            )

            result = {
                "dataset": ds,
                "metric": metric,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "arm_a_mean":
                    float(a.mean()),
                "arm_b_mean":
                    float(b.mean()),
                "difference":
                    float(diff.mean()),
                "ci95":
                    bootstrap_ci(
                        diff,
                        seed,
                    ),
                "exact_signflip_p":
                    raw_p,
                "subject_differences":
                    diff.tolist(),
            }

            family_results.append(
                result
            )

            raw_ps.append(
                raw_p
            )

    adjusted = holm_adjust(
        raw_ps
    )

    for r, p_adj in zip(
        family_results,
        adjusted,
    ):
        r["holm_p"] = float(
            p_adj
        )

    summary[comparison] = (
        family_results
    )


# --------------------------------------------------------------
# Aggregate subject-first curves.
# --------------------------------------------------------------

curve_df = pd.DataFrame(
    curve_rows
)

mean_curve_rows = []

for ds in DATASETS:
    sub = curve_df[
        curve_df["dataset"] == ds
    ]

    for (
        n_samples,
        time_s,
    ), group in sub.groupby(
        ["n_samples", "time_s"],
        sort=True,
    ):
        row = {
            "dataset": ds,
            "n_samples":
                int(n_samples),
            "time_s":
                float(time_s),
        }

        for arm in (
            "compact_rt",
            "react_rt",
            "original_compact",
            "original_react",
            "reverse_only",
        ):
            row[arm] = float(
                group[arm].mean()
            )

        mean_curve_rows.append(row)


# --------------------------------------------------------------
# Save.
# --------------------------------------------------------------

summary_path = (
    RT / "paired_rt_summary.json"
)

subjects_path = (
    RT / "paired_rt_subject_summary.csv"
)

curves_path = (
    RT / "paired_rt_mean_curves.csv"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ) + "\n"
)

df.to_csv(
    subjects_path,
    index=False,
)

pd.DataFrame(
    mean_curve_rows
).to_csv(
    curves_path,
    index=False,
)


# --------------------------------------------------------------
# Compact terminal report.
# --------------------------------------------------------------

for comparison in COMPARISONS:

    print(
        "\n" + "=" * 80
    )

    print(
        comparison.upper()
    )

    print(
        "=" * 80
    )

    for r in summary[
        comparison
    ]:
        print(
            f"{r['dataset']:5s} "
            f"{r['metric']:8s} | "
            f"{r['arm_a']}="
            f"{r['arm_a_mean']:6.2f}  "
            f"{r['arm_b']}="
            f"{r['arm_b_mean']:6.2f}  "
            f"Delta="
            f"{r['difference']:+6.2f}  "
            f"CI=["
            f"{r['ci95'][0]:.2f}, "
            f"{r['ci95'][1]:.2f}]  "
            f"p="
            f"{r['exact_signflip_p']:.4f}  "
            f"Holm="
            f"{r['holm_p']:.4f}"
        )


print("\nWROTE:")
print(summary_path)
print(subjects_path)
print(curves_path)
