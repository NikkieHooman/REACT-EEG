#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--dataset",
        choices=["all", "2a", "2b", "ssvep"],
        default="all",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=170926)
    return p.parse_args()


ARGS = parse_args()
STUDY = ARGS.study.resolve()
CODE = STUDY / "code"

if not CODE.is_dir():
    raise SystemExit(f"Missing frozen code directory: {CODE}")

sys.path.insert(0, str(CODE))

from fresh.data import SPECS, load_role, normalize, grid
from fresh.run import model_for


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_scaler(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@torch.inference_mode()
def ordinary_accuracy(model, x, y, m, device, batch_size):
    correct = 0
    total = 0
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(
            np.ascontiguousarray(x[start:start + batch_size, :, :m])
        ).to(device)

        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        yy = y[start:start + len(pred)]
        correct += int((pred == yy).sum())
        total += len(pred)

    return 100.0 * correct / total


@torch.inference_mode()
def zerofill_accuracy(model, x, y, m, device, batch_size):
    """
    Full-length fixed-endpoint evaluation:
    samples <= m retain their normalized values;
    normalized samples > m are set to zero.

    Because StandardScaler was fit on the training role separately at every
    channel/sample coordinate, zero is the training mean in normalized space.
    """
    correct = 0
    total = 0

    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(
            np.ascontiguousarray(x[start:start + batch_size])
        ).to(device)

        xb = xb.clone()
        if m < xb.shape[-1]:
            xb[:, :, m:] = 0.0

        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        yy = y[start:start + len(pred)]

        correct += int((pred == yy).sum())
        total += len(pred)

    return 100.0 * correct / total


@torch.inference_mode()
def route_accuracies(model, x, y, m, device, batch_size):
    """
    Post-hoc interventions on the jointly trained REACT checkpoint.

    g=0   : forward route only
    g=.5  : equal mixture
    g=1   : reverse route only
    learned: trained static gate
    """
    counts = {
        "learned": 0,
        "forward_only": 0,
        "equal_gate": 0,
        "reverse_only": 0,
    }
    total = 0
    max_manual_vs_model = 0.0

    g = torch.sigmoid(model.gate_logits).detach()

    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(
            np.ascontiguousarray(x[start:start + batch_size, :, :m])
        ).to(device)

        z = model.tokenizer(xb)

        f = model.forward_reader(z)[:, :, -1]
        b = model.second_reader(z.flip(-1))[:, :, -1]

        learned_h = (1.0 - g) * f + g * b
        equal_h = 0.5 * f + 0.5 * b

        logits = {
            "learned": model.classifier(learned_h),
            "forward_only": model.classifier(f),
            "equal_gate": model.classifier(equal_h),
            "reverse_only": model.classifier(b),
        }

        direct = model(xb)
        err = float((direct - logits["learned"]).abs().max().cpu())
        max_manual_vs_model = max(max_manual_vs_model, err)

        yy = y[start:start + len(xb)]

        for key, value in logits.items():
            pred = value.argmax(1).cpu().numpy()
            counts[key] += int((pred == yy).sum())

        total += len(xb)

    return (
        {k: 100.0 * v / total for k, v in counts.items()},
        max_manual_vs_model,
    )


def nauc(values, lengths, fs):
    t = np.asarray(lengths, dtype=float) / float(fs)
    a = np.asarray(values, dtype=float)
    return float(np.trapz(a, t) / (t[-1] - t[0]))


def bootstrap_mean_ci(values, n_boot, seed):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    ids = rng.integers(
        0, len(x),
        size=(n_boot, len(x)),
    )
    means = x[ids].mean(axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def main():
    ARGS.out.mkdir(parents=True, exist_ok=True)

    plan = json.loads((STUDY / "study.json").read_text())

    datasets = (
        ["2a", "2b", "ssvep"]
        if ARGS.dataset == "all"
        else [ARGS.dataset]
    )

    if ARGS.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(ARGS.device)

    all_rows = []
    per_run = []

    for ds in datasets:
        cfg = SPECS[ds]
        lengths = list(grid(ds))
        data_root = plan["data_roots"][ds]

        print("\n" + "=" * 90)
        print("DATASET", ds)
        print("boundaries:", len(lengths), lengths[0], "->", lengths[-1])

        for subject in range(1, cfg["subjects"] + 1):
            for seed in plan["seeds"]:
                run_dir = (
                    STUDY / "runs"
                    / f"{ds}_S{subject:02d}_seed{seed}"
                )

                scaler_path = run_dir / "scaler.npz"
                compact_ckpt = run_dir / "compact" / "final.pt"
                reader_ckpt = run_dir / "reader" / "final.pt"

                for p in (scaler_path, compact_ckpt, reader_ckpt):
                    if not p.is_file():
                        raise FileNotFoundError(p)

                state = load_scaler(scaler_path)

                test_raw, test_y, test_ids, _ = load_role(
                    data_root, ds, subject, "test"
                )
                test = normalize(test_raw, state)

                compact = model_for(
                    "compact", ds, seed, STUDY
                ).to(device)
                compact.load_state_dict(
                    load_checkpoint(compact_ckpt)["state_dict"],
                    strict=True,
                )
                compact.eval()

                reader = model_for(
                    "reader", ds, seed, STUDY
                ).to(device)
                reader.load_state_dict(
                    load_checkpoint(reader_ckpt)["state_dict"],
                    strict=True,
                )
                reader.eval()

                gate = torch.sigmoid(
                    reader.gate_logits.detach()
                ).cpu().numpy()

                curves = defaultdict(list)
                max_route_error = 0.0

                for m in lengths:
                    c_trunc = ordinary_accuracy(
                        compact, test, test_y, m,
                        device, ARGS.batch_size
                    )

                    c_zero = zerofill_accuracy(
                        compact, test, test_y, m,
                        device, ARGS.batch_size
                    )

                    route, err = route_accuracies(
                        reader, test, test_y, m,
                        device, ARGS.batch_size
                    )
                    max_route_error = max(max_route_error, err)

                    curves["compact_truncated"].append(c_trunc)
                    curves["compact_zerofill"].append(c_zero)

                    for key, val in route.items():
                        curves[f"react_{key}"].append(val)

                    all_rows.append({
                        "dataset": ds,
                        "subject": subject,
                        "seed": seed,
                        "m": m,
                        "time_s": m / cfg["fs"],
                        "compact_truncated": c_trunc,
                        "compact_zerofill": c_zero,
                        "react_learned": route["learned"],
                        "react_forward_only": route["forward_only"],
                        "react_equal_gate": route["equal_gate"],
                        "react_reverse_only": route["reverse_only"],
                        "gate_mean": float(gate.mean()),
                        "gate_sd_features": float(gate.std(ddof=1)),
                    })

                # Validate against the stored final result curves.
                for kind, our_key in [
                    ("compact", "compact_truncated"),
                    ("reader", "react_learned"),
                ]:
                    stored = json.loads(
                        (run_dir / kind / "result.json").read_text()
                    )
                    if list(stored["lengths"]) != lengths:
                        raise AssertionError(
                            f"{ds} S{subject} seed{seed}: "
                            "stored grid mismatch"
                        )

                    ours = np.asarray(curves[our_key])
                    theirs = np.asarray(
                        stored["accuracy_percent"],
                        dtype=float,
                    )

                    if not np.allclose(
                        ours, theirs,
                        atol=1e-7, rtol=0
                    ):
                        raise AssertionError(
                            f"{ds} S{subject} seed{seed} "
                            f"{kind}: reproduced curve differs; "
                            f"max={np.max(np.abs(ours-theirs))}"
                        )

                if max_route_error > 1e-5:
                    raise AssertionError(
                        f"Manual route reconstruction mismatch: "
                        f"{max_route_error}"
                    )

                entry = {
                    "dataset": ds,
                    "subject": subject,
                    "seed": seed,
                    "gate_mean": float(gate.mean()),
                    "gate_sd_features": float(gate.std(ddof=1)),
                    "manual_route_max_logit_error": max_route_error,
                }

                for key, curve in curves.items():
                    entry[f"{key}_earliest"] = float(curve[0])
                    entry[f"{key}_endpoint"] = float(curve[-1])
                    entry[f"{key}_nauc"] = nauc(
                        curve, lengths, cfg["fs"]
                    )

                per_run.append(entry)

                print(
                    f"{ds} S{subject:02d} seed{seed}: "
                    f"gate={gate.mean():.3f} | "
                    f"early REACT={curves['react_learned'][0]:.2f} "
                    f"zero-fill={curves['compact_zerofill'][0]:.2f}"
                )

                del compact, reader
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # CSV output
    import csv

    rows_path = ARGS.out / "reviewer_boundary_results.csv"
    with rows_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(all_rows[0].keys())
        )
        w.writeheader()
        w.writerows(all_rows)

    runs_path = ARGS.out / "reviewer_run_summary.csv"
    with runs_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(per_run[0].keys())
        )
        w.writeheader()
        w.writerows(per_run)

    # Subject-first summary exactly matching manuscript logic.
    summary = {}

    metrics = [
        "compact_truncated",
        "compact_zerofill",
        "react_learned",
        "react_forward_only",
        "react_equal_gate",
        "react_reverse_only",
    ]

    for ds in datasets:
        dsruns = [r for r in per_run if r["dataset"] == ds]
        subjects = sorted({r["subject"] for r in dsruns})

        subject_values = {
            m: {
                "earliest": [],
                "endpoint": [],
                "nauc": [],
            }
            for m in metrics
        }

        subject_gate = []

        for subject in subjects:
            sruns = [
                r for r in dsruns
                if r["subject"] == subject
            ]

            subject_gate.append(
                np.mean([r["gate_mean"] for r in sruns])
            )

            for metric in metrics:
                for stat in ("earliest", "endpoint", "nauc"):
                    subject_values[metric][stat].append(
                        np.mean([
                            r[f"{metric}_{stat}"]
                            for r in sruns
                        ])
                    )

        zf_diff = (
            np.asarray(
                subject_values["react_learned"]["nauc"]
            )
            -
            np.asarray(
                subject_values["compact_zerofill"]["nauc"]
            )
        )

        reverse_diff = (
            np.asarray(
                subject_values["react_learned"]["nauc"]
            )
            -
            np.asarray(
                subject_values["react_reverse_only"]["nauc"]
            )
        )

        ds_summary = {
            "subjects": len(subjects),
            "gate_mean_subject_first": float(
                np.mean(subject_gate)
            ),
            "gate_sd_subject_first": float(
                np.std(subject_gate, ddof=1)
            ),
            "metrics": {},
            "react_minus_zerofill_nauc": {
                "mean": float(zf_diff.mean()),
                "ci95": bootstrap_mean_ci(
                    zf_diff,
                    ARGS.bootstrap,
                    ARGS.bootstrap_seed,
                ),
                "subject_differences": zf_diff.tolist(),
            },
            "react_minus_reverse_only_nauc": {
                "mean": float(reverse_diff.mean()),
                "ci95": bootstrap_mean_ci(
                    reverse_diff,
                    ARGS.bootstrap,
                    ARGS.bootstrap_seed,
                ),
                "subject_differences": reverse_diff.tolist(),
            },
        }

        for metric in metrics:
            ds_summary["metrics"][metric] = {
                stat: float(
                    np.mean(subject_values[metric][stat])
                )
                for stat in ("earliest", "endpoint", "nauc")
            }

        summary[ds] = ds_summary

    summary_path = ARGS.out / "reviewer_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\n" + "=" * 90)
    print("FINAL SUBJECT-FIRST SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nWROTE:")
    print(rows_path)
    print(runs_path)
    print(summary_path)


if __name__ == "__main__":
    main()
