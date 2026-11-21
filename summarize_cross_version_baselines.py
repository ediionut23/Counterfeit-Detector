"""Aggregate the fixed M→E baseline runs stored under benchmark/cross_version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/cross_version"))
    parser.add_argument("--prefix", default="baseline_seed")
    parser.add_argument("--name", default="cross_version_M_to_E")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    paths = sorted(args.root.glob(f"{args.prefix}*.json"))
    if not paths:
        raise ValueError(f"No {args.prefix}*.json files in {args.root}")
    runs = [json.loads(path.read_text())[args.name] for path in paths]
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("Duplicate seeds")
    for key in ("n_train", "n_test", "train_csv", "test_csv"):
        if len({run[key] for run in runs}) != 1:
            raise ValueError(f"Runs disagree on {key}")

    summary = {
        "seeds": [run["seed"] for run in runs],
        "n_train": runs[0]["n_train"],
        "n_test": runs[0]["n_test"],
        "train_csv": runs[0]["train_csv"],
        "test_csv": runs[0]["test_csv"],
        "models": {},
    }
    for model in ("property_only", "fingerprint_rf", "fingerprint_xgboost"):
        if not all(model in run for run in runs):
            continue
        summary["models"][model] = {}
        for metric in ("false_negative_rate", "precision", "recall", "f1",
                       "accuracy", "roc_auc"):
            values = [run[model][metric] for run in runs]
            summary["models"][model][metric] = {
                "mean": round(mean(values), 4),
                "std": round(stdev(values), 4) if len(values) > 1 else None,
            }
    output = args.out or args.root / "baseline_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
