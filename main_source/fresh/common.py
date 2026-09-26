from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fail on NaN/Infinity rather than writing non-standard JSON silently.
    text = json.dumps(jsonable(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, val in sorted(model.state_dict().items()):
        h.update(key.encode())
        a = val.detach().cpu().contiguous()
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        # uint8 view also handles bfloat16, including scalar buffers.
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def resolve(spec: str):
    if ":" not in spec:
        raise ValueError(f"Use an explicit 'module:attribute' import, got {spec!r}")
    mod, name = spec.split(":", 1)
    out = importlib.import_module(mod)
    for part in name.split("."):
        out = getattr(out, part)
    return out


def seed_all(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # FP32 tensor dtype does not itself disable TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def environment(device: torch.device | str) -> dict:
    device = torch.device(device)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "git_commit": commit, "git_dirty": dirty,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    }


def ensure_new_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checked_logits(value: Any, batch: int, classes: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, classes):
        raise ValueError(f"Adapter must return logits [batch, classes] = {(batch, classes)}; got {getattr(value, 'shape', type(value))}")
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite logits")
    return value


def verified_plan(study):
    """Reject edited experiment decisions after the study was frozen."""
    p=read_json(Path(study)/"study.json")
    content={k:v for k,v in p.items() if k!="plan_hash"}
    if p.get("plan_hash")!=digest(content):
        raise ValueError("Study plan changed after freezing; create a new study, do not reuse checkpoints")
    return p
