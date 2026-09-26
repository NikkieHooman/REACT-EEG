import json
import sys
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])

models = ["reader", "compact", "ff", "compact_mean"]

values = defaultdict(
    lambda: defaultdict(
        lambda: {
            "E_trunc": [],
            "E_future": [],
            "positive_leak_error": [],
        }
    )
)


def collect(obj, target):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target and isinstance(v, (int, float)):
                target[k].append(abs(float(v)))
            collect(v, target)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, target)


for p in root.glob("runs/*/*/causality.json"):
    cell = p.parents[1].name
    dataset = cell.split("_")[0]
    model = p.parent.name

    if model not in models:
        continue

    obj = json.loads(p.read_text())

    # Use the trained-checkpoint branch, not randomized/nondegenerate controls.
    trained = obj.get("trained", obj)

    collect(trained, values[dataset][model])


for dataset in ("2a", "2b", "ssvep"):
    print("\n===", dataset.upper(), "===")

    all_trunc = []
    all_future = []
    all_leak = []

    for model in models:
        x = values[dataset][model]

        mt = max(x["E_trunc"]) if x["E_trunc"] else None
        mf = max(x["E_future"]) if x["E_future"] else None
        ml = max(x["positive_leak_error"]) if x["positive_leak_error"] else None

        print(
            model,
            "E_trunc=", mt,
            "E_future=", mf,
            "positive_leak=", ml,
        )

        all_trunc += x["E_trunc"]
        all_future += x["E_future"]
        all_leak += x["positive_leak_error"]

    print(
        "DATASET MAX:",
        "E_trunc=", max(all_trunc) if all_trunc else None,
        "E_future=", max(all_future) if all_future else None,
        "positive_leak=", max(all_leak) if all_leak else None,
    )
