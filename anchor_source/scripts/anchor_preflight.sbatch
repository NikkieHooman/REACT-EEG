#!/bin/bash
set -euo pipefail

RUN="$1"
PY="/users/nikkieh/bibm_env/bin/python"

cd "$RUN/code"
export PYTHONPATH="$PWD"

"$PY" - <<'PY'
import torch
from fresh.anchor_models import make_anchor_control

print("torch", torch.__version__)
print("CUDA build", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("GPU not visible inside allocation")

print("GPU", torch.cuda.get_device_name(0))

for kind in (
    "fixed_anchor",
    "relative_position",
    "random_truncation",
):
    m = make_anchor_control(
        kind,
        channels=22,
        classes=4,
        max_samples=1000,
        pool=32,
        seed=2025,
        train_lengths=[250, 500, 750, 1000],
    )
    print(kind, sum(p.numel() for p in m.parameters()))

print("ANCHOR PREFLIGHT PASS")
PY
