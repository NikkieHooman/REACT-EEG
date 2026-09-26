from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = [
    "reader",
    "fixed_anchor",
    "relative_position",
    "random_truncation",
]


def read_json(p):
    return json.loads(Path(p).read_text())


def nauc(acc, lengths, fs):
    t = np.asarray(lengths, dtype=float) / float(fs)
    a = np.asarray(acc, dtype=float)
    return float(np.trapz(a, t) / (t[-1] - t[0]))


def bootstrap_mean_ci(values, seed=170926, n=10000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(
        0,
        len(x),
        size=(n, len(x)),
    )
    boots = x[ix].mean(1)
    return [
        float(x.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    ]


def cell_name(d, s, k):
    return f"{d}_S{s:02d}_seed{k}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-study", required=True)
    p.add_argument("--diag-study", required=True)
    args = p.parse_args()

    orig = Path(args.orig_study)
    diag = Path(args.diag_study)

    plan = read_json(orig / "study.json")
    seeds = plan["seeds"]

    summary = {}

    for dataset, cfg in plan["datasets"].items():
        subjects = range(1, cfg["subjects"] + 1)

        curves = {
            m: {} for m in MODELS
        }

        for subject in subjects:
            per_seed = {m: [] for m in MODELS}

            for seed in seeds:
                name = cell_name(dataset, subject, seed)

                r = read_json(
                    orig
                    / "runs"
                    / name
                    / "reader"
                    / "result.json"
                )
                per_seed["reader"].append(
                    np.asarray(r["accuracy_percent"], dtype=float)
                )

                for m in MODELS[1:]:
                    q = read_json(
                        diag
                        / "runs"
                        / name
                        / m
                        / "result.json"
                    )
                    per_seed[m].append(
                        np.asarray(q["accuracy_percent"], dtype=float)
                    )

            for m in MODELS:
                curves[m][subject] = np.mean(
                    np.stack(per_seed[m]),
                    axis=0,
                )

        lengths = read_json(
            orig
            / "runs"
            / cell_name(dataset, 1, seeds[0])
            / "reader"
            / "result.json"
        )["lengths"]

        out = {
            "lengths": lengths,
            "fs": cfg["fs"],
            "models": {},
            "reader_minus_control_nauc": {},
        }

        for m in MODELS:
            matrix = np.stack(
                [curves[m][s] for s in subjects]
            )

            subject_nauc = np.asarray([
                nauc(row, lengths, cfg["fs"])
                for row in matrix
            ])

            out["models"][m] = {
                "earliest_accuracy": float(matrix[:, 0].mean()),
                "endpoint_accuracy": float(matrix[:, -1].mean()),
                "nauc": float(subject_nauc.mean()),
                "subject_nauc": subject_nauc.tolist(),
            }

        reader_nauc = np.asarray(
            out["models"]["reader"]["subject_nauc"]
        )

        for m in MODELS[1:]:
            control_nauc = np.asarray(
                out["models"][m]["subject_nauc"]
            )
            out["reader_minus_control_nauc"][m] = bootstrap_mean_ci(
                reader_nauc - control_nauc
            )

        summary[dataset] = out

    # Aggregate reverse-route probes.
    reverse_summary = {}

    for dataset, cfg in plan["datasets"].items():
        subject_rows = {}

        for subject in range(1, cfg["subjects"] + 1):
            rows = []

            for seed in seeds:
                name = cell_name(dataset, subject, seed)
                pth = diag / "runs" / name / "reverse_probe.json"
                rows.append(read_json(pth))

            subject_rows[subject] = rows

        labels = sorted({
            label
            for rows in subject_rows.values()
            for row in rows
            for label in row["subset_probe"]["fused_accuracy"]
        })

        ds = {}

        for label in labels:
            vals = []

            for subject, rows in subject_rows.items():
                seed_vals = [
                    r["subset_probe"]["fused_accuracy"][label]
                    for r in rows
                    if label in r["subset_probe"]["fused_accuracy"]
                ]
                if seed_vals:
                    vals.append(float(np.mean(seed_vals)))

            ds[label] = float(np.mean(vals))

        profiles = []
        repro = []

        for rows in subject_rows.values():
            for r in rows:
                profiles.append(
                    np.asarray(
                        r["gradient_profile"]["profile_normalized"],
                        dtype=float,
                    )
                )
                repro.append(
                    r["prediction_reproduction_max_abs_error"]
                )

        min_n = min(map(len, profiles))
        mean_profile = np.mean(
            np.stack([x[:min_n] for x in profiles]),
            axis=0,
        )

        reverse_summary[dataset] = {
            "fused_accuracy_by_subset": ds,
            "mean_gradient_profile": mean_profile.tolist(),
            "gradient_onset_mass_first8": float(
                mean_profile[:min(8, min_n)].sum()
            ),
            "gradient_recent_mass_last8": float(
                mean_profile[-min(8, min_n):].sum()
            ),
            "max_original_prediction_reproduction_error": float(
                max(repro)
            ),
        }

    final = {
        "anchor_controls": summary,
        "reverse_route": reverse_summary,
    }

    (diag / "anchor_summary.json").write_text(
        json.dumps(final, indent=2)
    )

    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
