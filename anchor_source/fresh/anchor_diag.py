"""Matched anchor-vs-reversal diagnostic using the frozen September-18 harness."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import fresh.run as core
from fresh.anchor_models import make_anchor_control


DIAG_MODELS = (
    "fixed_anchor",
    "relative_position",
    "random_truncation",
)


def source_hash():
    h = hashlib.sha256()
    for f in (Path(__file__), Path(__file__).with_name("anchor_models.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def cell_name(dataset, subject, seed):
    return f"{dataset}_S{subject:02d}_seed{seed}"


def load_saved_scaler(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def assert_same_scaler(a, b):
    if set(a) != set(b):
        raise AssertionError("Scaler keys differ from original study")

    for key in a:
        if not np.array_equal(a[key], b[key]):
            diff = float(np.max(np.abs(a[key] - b[key])))
            raise AssertionError(
                f"Refitted scaler differs from original at {key}: {diff}"
            )


def model_for(kind, dataset, seed):
    c = core.SPECS[dataset]
    return make_anchor_control(
        kind,
        channels=c["channels"],
        classes=c["classes"],
        max_samples=c["samples"],
        pool=c["pool"],
        seed=seed,
        train_lengths=core.grid(dataset),
    )


@torch.inference_mode()
def reverse_subset_probe(model, test, y, device, batch=64):
    """Compare onset-k vs recent-k inputs to the trained reverse route."""

    model.eval()
    cfg = {}

    full_correct = 0
    total = 0

    fused_correct = {}
    reverse_only_correct = {}

    for start in range(0, len(test), batch):
        xb = torch.from_numpy(
            np.ascontiguousarray(test[start:start + batch])
        ).to(device)
        yb = torch.from_numpy(y[start:start + batch]).to(device)

        z = model.tokenizer(xb)
        n = int(z.shape[-1])
        ks = [k for k in (4, 8, 16, 32) if k <= n]

        f = model.forward_reader(z)[:, :, -1]
        b_full = model.second_reader(z.flip(-1))[:, :, -1]
        g = model.gate_logits.sigmoid()

        h_full = (1 - g) * f + g * b_full
        full_logits = model.classifier(h_full)

        full_correct += int((full_logits.argmax(1) == yb).sum())
        total += len(yb)

        for k in ks:
            onset = z[:, :, :k]
            recent = z[:, :, -k:]

            b_onset = model.second_reader(onset.flip(-1))[:, :, -1]
            b_recent = model.second_reader(recent.flip(-1))[:, :, -1]

            for label, b in (
                (f"onset_{k}", b_onset),
                (f"recent_{k}", b_recent),
            ):
                h = (1 - g) * f + g * b

                pred_fused = model.classifier(h).argmax(1)
                pred_reverse = model.classifier(b).argmax(1)

                fused_correct[label] = fused_correct.get(label, 0) + int(
                    (pred_fused == yb).sum()
                )
                reverse_only_correct[label] = (
                    reverse_only_correct.get(label, 0)
                    + int((pred_reverse == yb).sum())
                )

    cfg["n_test"] = total
    cfg["full_fused_accuracy"] = 100.0 * full_correct / total
    cfg["fused_accuracy"] = {
        k: 100.0 * v / total for k, v in fused_correct.items()
    }
    cfg["reverse_only_accuracy"] = {
        k: 100.0 * v / total for k, v in reverse_only_correct.items()
    }
    return cfg


def gradient_profile(model, test, device, seed, n_examples=32, probes=8):
    """Hutchinson estimate of reverse-summary Jacobian energy by input token."""

    model.eval()

    x = torch.from_numpy(
        np.ascontiguousarray(test[:min(n_examples, len(test))])
    ).to(device)

    with torch.no_grad():
        z0 = model.tokenizer(x)

    z = z0.detach().requires_grad_(True)
    b = model.second_reader(z.flip(-1))[:, :, -1]

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 551231)

    score = torch.zeros(z.shape[-1], device=device)

    for i in range(probes):
        v = torch.randint(
            0,
            2,
            b.shape,
            device=device,
            generator=gen,
            dtype=torch.int64,
        ).to(b.dtype)
        v = v.mul(2).sub(1)

        scalar = (b * v).sum() / b.shape[0]
        grad = torch.autograd.grad(
            scalar,
            z,
            retain_graph=(i + 1 < probes),
            create_graph=False,
        )[0]

        score += grad.square().sum(dim=1).mean(dim=0)

    score = torch.sqrt(score / probes).detach().cpu().numpy()
    norm = score / max(float(score.sum()), 1e-12)

    index = np.arange(1, len(norm) + 1)
    correlation = float(np.corrcoef(index, norm)[0, 1])

    k = min(8, len(norm))

    return {
        "n_examples": int(x.shape[0]),
        "probes": probes,
        "tokens": len(norm),
        "profile_raw": score.tolist(),
        "profile_normalized": norm.tolist(),
        "onset_mass_first8": float(norm[:k].sum()),
        "recent_mass_last8": float(norm[-k:].sum()),
        "position_correlation": correlation,
    }


def original_reader_probe(
    *,
    orig_study,
    dataset,
    subject,
    seed,
    test,
    test_y,
    device,
):
    name = cell_name(dataset, subject, seed)
    folder = Path(orig_study) / "runs" / name / "reader"

    ck_path = folder / "final.pt"
    pred_path = folder / "predictions.npz"

    if not ck_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"Missing original reader artifacts: {folder}")

    model = core.model_for(
        "reader",
        dataset,
        seed,
        orig_study,
    ).to(device)

    ck = core.weights_load(ck_path)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    with np.load(pred_path, allow_pickle=False) as old:
        old_logits = old["logits"]
        lengths = old["lengths"].astype(int).tolist()

    reproduced = core.predict(
        model,
        test,
        lengths,
        device,
    )

    max_error = float(
        np.max(np.abs(reproduced - old_logits))
    )

    if max_error > 1e-4:
        raise AssertionError(
            f"Original READER prediction reproduction failed: {max_error}"
        )

    subset = reverse_subset_probe(
        model,
        test,
        test_y,
        device,
    )

    grad = gradient_profile(
        model,
        test,
        device,
        seed,
    )

    return {
        "checkpoint": str(ck_path),
        "prediction_reproduction_max_abs_error": max_error,
        "subset_probe": subset,
        "gradient_profile": grad,
    }


def run_task(orig_study, out, index, device):
    orig_study = Path(orig_study)
    out = Path(out)

    plan = core.read_json(orig_study / "study.json")
    cell = plan["tasks"][index]

    dataset = cell["dataset"]
    subject = int(cell["subject"])
    seed = int(cell["seed"])

    cfg = core.SPECS[dataset]
    name = cell_name(dataset, subject, seed)

    directory = out / "runs" / name
    directory.mkdir(parents=True, exist_ok=True)

    lock = (directory / "lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    core.seed_all(seed)
    torch.set_num_threads(2)

    data_root = plan["data_roots"][dataset]

    train_raw, train_y, train_ids, train_records = core.load_role(
        data_root,
        dataset,
        subject,
        "train",
    )

    normalizer = core.fit_scaler(train_raw)

    # Verify the diagnostic uses exactly the scaler that generated the
    # September-18 paper run.
    orig_cell = orig_study / "runs" / name
    original_scaler = load_saved_scaler(orig_cell / "scaler.npz")
    assert_same_scaler(normalizer, original_scaler)

    train = core.normalize(train_raw, normalizer)

    np.savez(directory / "scaler.npz", **normalizer)

    codehash = source_hash()
    trained = {}
    shared = {}
    orders = {}
    signatures = {}

    recipe = dict(plan["recipe"])

    for kind in DIAG_MODELS:
        model = model_for(kind, dataset, seed).to(device)
        core.constrain(model)

        sig = core.digest({
            "diagnostic": "anchor_vs_reversal",
            "source_hash": codehash,
            "original_plan_hash": plan["plan_hash"],
            "dataset": dataset,
            "subject": subject,
            "seed": seed,
            "model": kind,
            "train_records": train_records,
            "train_x": core.array_sha(train),
            "normalizer": core.digest(normalizer),
        })

        initial_hash = core.state_sha(model)
        initial_shared = core.shared_sha(model)

        folder = directory / kind

        ck = core.train_one(
            model,
            train,
            train_y,
            folder,
            sig,
            seed,
            recipe,
            device,
            initial_hash,
            initial_shared,
            evidence="anchor_diagnostic",
        )

        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval()

        with torch.inference_mode():
            current = model(
                torch.from_numpy(train[:2]).to(device)
            ).cpu()

        if not torch.allclose(
            current,
            ck["training_probe_logits"],
            atol=1e-6,
            rtol=1e-6,
        ):
            raise AssertionError(
                f"{kind}: checkpoint round-trip differs"
            )

        shared[kind] = ck["initial_shared_hash"]
        orders[kind] = core.digest(ck["orders"])
        trained[kind] = ck
        signatures[kind] = sig

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # All three controls begin from exactly the same Compact shared weights
    # and see exactly the same minibatch ordering.
    if len(set(shared.values())) != 1:
        raise AssertionError(
            f"Diagnostic shared initialization mismatch: {shared}"
        )

    if len(set(orders.values())) != 1:
        raise AssertionError(
            f"Diagnostic minibatch-order mismatch: {orders}"
        )

    core.write_json(
        directory / "pairing.json",
        {
            "initial_shared_hashes": shared,
            "minibatch_order_hashes": orders,
            "passed": True,
        },
    )

    # Held role is opened only after all diagnostic controls are trained.
    test_raw, test_y, test_ids, test_records = core.load_role(
        data_root,
        dataset,
        subject,
        "test",
    )

    overlap = core.check_role_overlap(train_raw, test_raw)
    test = core.normalize(test_raw, normalizer)

    testhash = core.array_sha(test)

    lengths = core.grid(dataset)

    for kind in DIAG_MODELS:
        folder = directory / kind

        model = model_for(kind, dataset, seed).to(device)
        model.load_state_dict(
            trained[kind]["state_dict"],
            strict=True,
        )
        model.eval()

        logits = core.predict(
            model,
            test,
            lengths,
            device,
        )

        accuracy = 100.0 * (
            logits.argmax(-1) == test_y[:, None]
        ).mean(0)

        np.savez_compressed(
            folder / "predictions.npz",
            logits=logits,
            y=test_y,
            ids=test_ids,
            lengths=np.asarray(lengths),
        )

        core.write_json(
            folder / "result.json",
            {
                "status": "MEASURED",
                "experiment": "anchor_vs_reversal",
                "model": kind,
                "cell": cell,
                "signature": signatures[kind],
                "epochs": recipe["epochs"],
                "prefix_weight": 0.0,
                "parameters": sum(
                    p.numel() for p in model.parameters()
                ),
                "lengths": lengths,
                "accuracy_percent": accuracy.tolist(),
                "n_test": len(test_y),
                "fs": cfg["fs"],
                "test_x_hash": testhash,
                "overlap_check": overlap,
                "source_hash": codehash,
            },
        )

        print(
            "MEASURED",
            name,
            kind,
            "endpoint",
            float(accuracy[-1]),
            flush=True,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reverse = original_reader_probe(
        orig_study=str(orig_study),
        dataset=dataset,
        subject=subject,
        seed=seed,
        test=test,
        test_y=test_y,
        device=device,
    )

    core.write_json(
        directory / "reverse_probe.json",
        reverse,
    )

    core.write_json(
        directory / "group_status.json",
        {
            "status": "MEASURED",
            "cell": cell,
            "models": list(DIAG_MODELS),
            "original_reader_probe": True,
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-study", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_task(
        args.orig_study,
        args.out,
        args.index,
        args.device,
    )


if __name__ == "__main__":
    main()
