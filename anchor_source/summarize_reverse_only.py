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

REV = Path(
    "/lustre/smuexa01/client/users/nikkieh/reader_reverse_only_final"
)

DATASETS = {
    "2a": {"subjects": 9, "fs": 250},
    "2b": {"subjects": 9, "fs": 250},
    "ssvep": {"subjects": 10, "fs": 256},
}

B = 10000
BOOTSTRAP_SEED = 170926

plan = json.loads((ORIG / "study.json").read_text())
seeds = list(plan["seeds"])


def nauc(curve, lengths, fs):
    curve = np.asarray(curve, dtype=float)
    t = np.asarray(lengths, dtype=float) / float(fs)
    return float(np.trapezoid(curve, t) / (t[-1] - t[0]))


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(
        0,
        len(values),
        size=(B, len(values)),
    )
    means = values[ix].mean(axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def exact_signflip_p(values):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())

    stats = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        signs = np.asarray(signs)
        stats.append(abs(np.mean(values * signs)))

    stats = np.asarray(stats)
    return float(np.mean(stats >= observed - 1e-12))


# First verify that all 84 expected results exist.
expected = sum(v["subjects"] for v in DATASETS.values()) * len(seeds)

result_files = list(
    REV.glob("runs/*/reverse_only/result.json")
)

print("Expected Reverse-only results:", expected)
print("Found Reverse-only results   :", len(result_files))

if len(result_files) != expected:
    raise RuntimeError(
        f"Expected {expected} Reverse-only results, found {len(result_files)}"
    )


summary = {}
subject_rows = []
curve_rows = []
tests = []


for ds_index, (ds, cfg) in enumerate(DATASETS.items()):

    print("\n" + "=" * 80)
    print("DATASET:", ds)

    subject_react_curves = []
    subject_reverse_curves = []

    reference_lengths = None
    parameter_counts = set()

    for subject in range(1, cfg["subjects"] + 1):

        react_seed_curves = []
        reverse_seed_curves = []

        for seed in seeds:

            react_path = (
                ORIG
                / "runs"
                / f"{ds}_S{subject:02d}_seed{seed}"
                / "reader"
                / "result.json"
            )

            reverse_path = (
                REV
                / "runs"
                / f"{ds}_S{subject:02d}_seed{seed}"
                / "reverse_only"
                / "result.json"
            )

            if not react_path.is_file():
                raise FileNotFoundError(react_path)

            if not reverse_path.is_file():
                raise FileNotFoundError(reverse_path)

            react = json.loads(react_path.read_text())
            reverse = json.loads(reverse_path.read_text())

            if reverse.get("status") != "complete":
                raise RuntimeError(
                    f"Incomplete Reverse-only result: {reverse_path}"
                )

            if reverse.get("epochs") != 600:
                raise RuntimeError(
                    f"Wrong epoch count: {reverse_path}"
                )

            if float(reverse.get("prefix_weight", -1)) != 0.0:
                raise RuntimeError(
                    f"Reverse-only not endpoint-only: {reverse_path}"
                )

            if react["lengths"] != reverse["lengths"]:
                raise RuntimeError(
                    f"Boundary grid mismatch: {ds} S{subject} seed{seed}"
                )

            lengths = np.asarray(react["lengths"], dtype=int)

            if reference_lengths is None:
                reference_lengths = lengths
            elif not np.array_equal(reference_lengths, lengths):
                raise RuntimeError(
                    f"Inconsistent grid within {ds}"
                )

            react_curve = np.asarray(
                react["accuracy_percent"], dtype=float
            )

            reverse_curve = np.asarray(
                reverse["accuracy_percent"], dtype=float
            )

            react_seed_curves.append(react_curve)
            reverse_seed_curves.append(reverse_curve)

            parameter_counts.add(
                int(reverse["parameters"])
            )

        # Important: average seeds WITHIN subject first.
        react_subject = np.mean(
            np.stack(react_seed_curves), axis=0
        )

        reverse_subject = np.mean(
            np.stack(reverse_seed_curves), axis=0
        )

        subject_react_curves.append(react_subject)
        subject_reverse_curves.append(reverse_subject)

        react_early = float(react_subject[0])
        reverse_early = float(reverse_subject[0])

        react_endpoint = float(react_subject[-1])
        reverse_endpoint = float(reverse_subject[-1])

        react_nauc = nauc(
            react_subject,
            reference_lengths,
            cfg["fs"],
        )

        reverse_nauc = nauc(
            reverse_subject,
            reference_lengths,
            cfg["fs"],
        )

        subject_rows.append({
            "dataset": ds,
            "subject": subject,

            "react_early": react_early,
            "reverse_only_early": reverse_early,
            "delta_early": react_early - reverse_early,

            "react_nauc": react_nauc,
            "reverse_only_nauc": reverse_nauc,
            "delta_nauc": react_nauc - reverse_nauc,

            "react_endpoint": react_endpoint,
            "reverse_only_endpoint": reverse_endpoint,
            "delta_endpoint": react_endpoint - reverse_endpoint,
        })

    if len(parameter_counts) != 1:
        raise RuntimeError(
            f"Inconsistent Reverse-only parameter counts on {ds}: "
            f"{parameter_counts}"
        )

    react_subjects = np.stack(subject_react_curves)
    reverse_subjects = np.stack(subject_reverse_curves)

    react_mean_curve = react_subjects.mean(axis=0)
    reverse_mean_curve = reverse_subjects.mean(axis=0)

    for j, m in enumerate(reference_lengths):
        curve_rows.append({
            "dataset": ds,
            "n_samples": int(m),
            "time_s": float(m / cfg["fs"]),
            "react_accuracy": float(react_mean_curve[j]),
            "trained_reverse_only_accuracy": float(
                reverse_mean_curve[j]
            ),
            "react_minus_reverse_only": float(
                react_mean_curve[j]
                - reverse_mean_curve[j]
            ),
        })

    ds_subject_rows = [
        r for r in subject_rows
        if r["dataset"] == ds
    ]

    metric_results = {}

    for metric_index, metric in enumerate(
        ("early", "nauc", "endpoint")
    ):

        react_values = np.asarray([
            r[f"react_{metric}"]
            for r in ds_subject_rows
        ])

        reverse_values = np.asarray([
            r[f"reverse_only_{metric}"]
            for r in ds_subject_rows
        ])

        differences = react_values - reverse_values

        seed = (
            BOOTSTRAP_SEED
            + 100 * ds_index
            + metric_index
        )

        result = {
            "react_mean": float(react_values.mean()),
            "reverse_only_mean": float(
                reverse_values.mean()
            ),
            "react_minus_reverse_only": float(
                differences.mean()
            ),
            "ci95": bootstrap_ci(
                differences,
                seed,
            ),
            "exact_signflip_p": exact_signflip_p(
                differences
            ),
            "subject_differences":
                differences.tolist(),
        }

        metric_results[metric] = result

        tests.append({
            "dataset": ds,
            "metric": metric,
            "raw_p": result["exact_signflip_p"],
        })

    summary[ds] = {
        "subjects": cfg["subjects"],
        "seeds": seeds,
        "reverse_only_parameters":
            int(next(iter(parameter_counts))),
        "metrics": metric_results,
    }


# Holm correction across the 3 datasets x 3 metrics = 9 tests.
raw_ps = np.asarray([
    t["raw_p"] for t in tests
])

order = np.argsort(raw_ps)
m = len(raw_ps)

adjusted = np.empty(m, dtype=float)
running = 0.0

for rank, idx in enumerate(order):
    candidate = min(
        1.0,
        (m - rank) * raw_ps[idx]
    )
    running = max(running, candidate)
    adjusted[idx] = running

for test, adj in zip(tests, adjusted):
    test["holm_p"] = float(adj)

    summary[
        test["dataset"]
    ]["metrics"][
        test["metric"]
    ]["holm_p"] = float(adj)


# Write outputs.
summary_path = REV / "summary.json"
subjects_path = REV / "subject_summary.csv"
curves_path = REV / "curves.csv"

summary_path.write_text(
    json.dumps(summary, indent=2) + "\n"
)

pd.DataFrame(subject_rows).to_csv(
    subjects_path,
    index=False,
)

pd.DataFrame(curve_rows).to_csv(
    curves_path,
    index=False,
)


print("\n" + "=" * 80)
print("FINAL SUBJECT-FIRST SUMMARY")

for ds in DATASETS:

    print("\n", ds.upper())

    for metric in ("early", "nauc", "endpoint"):

        z = summary[ds]["metrics"][metric]

        print(
            f"{metric:9s} | "
            f"REACT={z['react_mean']:6.2f}  "
            f"Reverse-only={z['reverse_only_mean']:6.2f}  "
            f"Delta={z['react_minus_reverse_only']:+6.2f}  "
            f"CI=[{z['ci95'][0]:.2f}, {z['ci95'][1]:.2f}]  "
            f"p={z['exact_signflip_p']:.4f}  "
            f"Holm={z['holm_p']:.4f}"
        )

print("\nWROTE:")
print(summary_path)
print(subjects_path)
print(curves_path)
