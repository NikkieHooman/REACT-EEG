#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--study", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--task-index", type=int, required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

STUDY = args.study.resolve()
sys.path.insert(0, str(STUDY / "code"))

import fresh.run as core
from fresh.data import SPECS, load_role, normalize, grid
from fresh.common import (
    array_sha,
    checked_logits,
    digest,
    seed_all,
    state_sha,
    write_json,
)


ARMS = ("compact_rt", "react_rt")


def load_scaler(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def assert_shared_initialization(compact, reader):
    """
    Verify tokenizer, forward reader, and classifier start identically.
    """
    a = compact.state_dict()
    b = reader.state_dict()

    prefixes = (
        "tokenizer.",
        "forward_reader.",
        "classifier.",
    )

    keys = [
        k for k in a
        if k.startswith(prefixes)
    ]

    if not keys:
        raise AssertionError("No shared-state keys found")

    for k in keys:
        if k not in b:
            raise AssertionError(f"Missing REACT shared key: {k}")

        if not torch.equal(
            a[k].detach().cpu(),
            b[k].detach().cpu(),
        ):
            raise AssertionError(
                f"Shared initialization differs at {k}"
            )


def sha_schedule(items):
    h = hashlib.sha256()
    for x in items:
        h.update(str(x).encode())
        h.update(b"\n")
    return h.hexdigest()


def train_arm(
    *,
    model,
    train,
    y,
    lengths,
    folder,
    seed,
    arm,
    device,
):
    folder.mkdir(parents=True, exist_ok=True)

    final_path = folder / "final.pt"

    if final_path.is_file():
        ck = torch.load(
            final_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(
            ck["state_dict"],
            strict=True,
        )
        model.eval()
        return ck

    recipe = core.RECIPE

    # Same global training RNG reset for both arms.
    seed_all(seed + 10000)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=recipe["lr"],
        weight_decay=recipe["weight_decay"],
        betas=tuple(recipe["betas"]),
        eps=recipe["eps"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=recipe["epochs"],
        eta_min=0.0,
    )

    xcpu = torch.from_numpy(train)
    ycpu = torch.from_numpy(y)

    batch_size = recipe["batch_size"]

    sampled_lengths = []
    order_hashes = []

    initial_hash = state_sha(model)

    for epoch in range(1, recipe["epochs"] + 1):

        model.train()

        # Exactly the same minibatch order for both arms.
        order_gen = torch.Generator().manual_seed(
            seed + 20000 + epoch
        )
        order = torch.randperm(
            len(y),
            generator=order_gen,
        )

        order_hashes.append(
            array_sha(order.numpy())
        )

        n_batches = (
            len(order) + batch_size - 1
        ) // batch_size

        # Dedicated duration RNG:
        # independent of dropout/model architecture.
        length_gen = torch.Generator().manual_seed(
            seed + 30000 + epoch
        )

        length_indices = torch.randint(
            low=0,
            high=len(lengths),
            size=(n_batches,),
            generator=length_gen,
        )

        epoch_lengths = [
            int(lengths[int(i)])
            for i in length_indices
        ]

        sampled_lengths.extend(epoch_lengths)

        loss_total = 0.0
        n_seen = 0

        for batch_index, off in enumerate(
            range(0, len(order), batch_size)
        ):
            ix = order[off:off + batch_size]

            m = epoch_lengths[batch_index]

            xb = xcpu[ix, :, :m].to(device)
            target = ycpu[ix].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = checked_logits(
                model(xb),
                len(ix),
                int(y.max()) + 1,
            )

            loss = F.cross_entropy(
                logits,
                target,
                label_smoothing=recipe[
                    "label_smoothing"
                ],
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                recipe["grad_clip"],
                error_if_nonfinite=True,
            )

            optimizer.step()
            core.constrain(model)

            loss_total += (
                float(loss.detach()) * len(ix)
            )
            n_seen += len(ix)

        scheduler.step()

        if (
            epoch == 1
            or epoch % 25 == 0
            or epoch == recipe["epochs"]
        ):
            print(
                arm,
                "epoch",
                epoch,
                "loss",
                round(loss_total / n_seen, 6),
                flush=True,
            )

    model.eval()
    core.constrain(model)

    schedule_hash = sha_schedule(
        sampled_lengths
    )
    order_hash = sha_schedule(
        order_hashes
    )

    counts = {
        str(m): int(
            sum(x == m for x in sampled_lengths)
        )
        for m in lengths
    }

    ck = {
        "state_dict": model.state_dict(),
        "epoch": recipe["epochs"],
        "arm": arm,
        "seed": seed,
        "initial_hash": initial_hash,
        "length_schedule_hash": schedule_hash,
        "minibatch_order_hash": order_hash,
        "length_counts": counts,
        "sampling": (
            "one boundary per minibatch; "
            "uniform over core.grid(dataset); "
            "dedicated RNG seed=seed+30000+epoch"
        ),
        "prefix_weight": 0.0,
        "epochs": recipe["epochs"],
    }

    tmp = final_path.with_suffix(".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, final_path)

    return ck


# ------------------------------------------------------------------
# Task map: exactly the same 84 cells as the final fresh study.
# ------------------------------------------------------------------

plan = json.loads(
    (STUDY / "study.json").read_text()
)

tasks = []

for dataset in ("2a", "2b", "ssvep"):
    for subject in range(
        1,
        SPECS[dataset]["subjects"] + 1,
    ):
        for seed in plan["seeds"]:
            tasks.append(
                (dataset, subject, seed)
            )

if not 0 <= args.task_index < len(tasks):
    raise ValueError(
        f"task-index must be 0..{len(tasks)-1}"
    )

dataset, subject, seed = tasks[
    args.task_index
]

cfg = SPECS[dataset]
device = torch.device(args.device)

name = (
    f"{dataset}_S{subject:02d}_seed{seed}"
)

original_run = STUDY / "runs" / name

out_run = args.out / "runs" / name
out_run.mkdir(
    parents=True,
    exist_ok=True,
)

# If both final result files already exist, stop cleanly.
if all(
    (
        out_run
        / arm
        / "result.json"
    ).is_file()
    for arm in ARMS
):
    print(
        "ALREADY COMPLETE",
        name,
        flush=True,
    )
    raise SystemExit(0)


# ------------------------------------------------------------------
# Training-role data only.
# ------------------------------------------------------------------

scaler = load_scaler(
    original_run / "scaler.npz"
)

train_raw, train_y, _, train_records = (
    load_role(
        plan["data_roots"][dataset],
        dataset,
        subject,
        "train",
    )
)

train = normalize(
    train_raw,
    scaler,
)

lengths = list(
    grid(dataset)
)


# ------------------------------------------------------------------
# Construct both arms from frozen fresh code.
# ------------------------------------------------------------------

compact = core.model_for(
    "compact",
    dataset,
    seed,
    STUDY,
).to(device)

react = core.model_for(
    "reader",
    dataset,
    seed,
    STUDY,
).to(device)

core.constrain(compact)
core.constrain(react)

assert_shared_initialization(
    compact,
    react,
)

compact_initial_shared = core.shared_sha(
    compact
)

react_initial_shared = core.shared_sha(
    react
)

if (
    compact_initial_shared
    != react_initial_shared
):
    raise AssertionError(
        "Shared initialization hashes differ"
    )


# ------------------------------------------------------------------
# Train BOTH before opening held evaluation data.
# ------------------------------------------------------------------

compact_ck = train_arm(
    model=compact,
    train=train,
    y=train_y,
    lengths=lengths,
    folder=out_run / "compact_rt",
    seed=seed,
    arm="compact_rt",
    device=device,
)

react_ck = train_arm(
    model=react,
    train=train,
    y=train_y,
    lengths=lengths,
    folder=out_run / "react_rt",
    seed=seed,
    arm="react_rt",
    device=device,
)

if (
    compact_ck["length_schedule_hash"]
    != react_ck["length_schedule_hash"]
):
    raise AssertionError(
        "Random-duration schedules differ"
    )

if (
    compact_ck["minibatch_order_hash"]
    != react_ck["minibatch_order_hash"]
):
    raise AssertionError(
        "Minibatch orders differ"
    )


# ------------------------------------------------------------------
# Only now open held role.
# ------------------------------------------------------------------

test_raw, test_y, test_ids, test_records = (
    load_role(
        plan["data_roots"][dataset],
        dataset,
        subject,
        "test",
    )
)

test = normalize(
    test_raw,
    scaler,
)


def evaluate(model, arm, ck):

    folder = out_run / arm

    result_path = folder / "result.json"

    if result_path.is_file():
        return

    model.load_state_dict(
        ck["state_dict"],
        strict=True,
    )
    model.eval()

    logits = core.predict(
        model,
        test,
        lengths,
        device,
    )

    accuracy = (
        100.0
        * (
            logits.argmax(-1)
            == test_y[:, None]
        ).mean(0)
    )

    np.savez_compressed(
        folder / "predictions.npz",
        logits=logits,
        y=test_y,
        ids=test_ids,
        lengths=np.asarray(lengths),
    )

    write_json(
        result_path,
        {
            "status": "complete",
            "experiment":
                "paired_random_truncation",
            "arm": arm,
            "dataset": dataset,
            "subject": subject,
            "seed": seed,
            "epochs": 600,
            "prefix_weight": 0.0,
            "sampling": (
                "one uniformly sampled legal "
                "boundary per minibatch"
            ),
            "paired_duration_schedule": True,
            "length_schedule_hash":
                ck["length_schedule_hash"],
            "minibatch_order_hash":
                ck["minibatch_order_hash"],
            "length_counts":
                ck["length_counts"],
            "lengths": lengths,
            "accuracy_percent":
                accuracy.tolist(),
            "n_test": len(test_y),
            "fs": cfg["fs"],
            "parameters": sum(
                p.numel()
                for p in model.parameters()
            ),
        },
    )

    print(
        "MEASURED",
        name,
        arm,
        "early",
        float(accuracy[0]),
        "endpoint",
        float(accuracy[-1]),
        flush=True,
    )


evaluate(
    compact,
    "compact_rt",
    compact_ck,
)

evaluate(
    react,
    "react_rt",
    react_ck,
)

print(
    "COMPLETE",
    name,
    "paired random truncation",
    flush=True,
)
