"""Aggregate fixed-split GNN runs over seeds without claiming a tuned comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/gnn_comparison"))
    parser.add_argument("--prefix", default="Cl_to_F")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = {"dataset": None,
               "protocol": "fixed scaffold split; 40-epoch cap; raw validation AUC early stopping; two deterministic CPU threads",
               "aggregation_note": "Means and sample SDs use per-run test metrics rounded to three decimals",
               "models": {}}
    for model in ("gin", "gcn", "gat", "gin_ring"):
        paths = sorted(args.root.glob(f"{args.prefix}_{model}_seed*.json"))
        if not paths:
            continue
        runs = [json.loads(path.read_text()) for path in paths]
        if model == "gin_ring" and any(not run.get("ring_size_features") for run in runs):
            raise ValueError("Ring-size runs must enable ring-size atom features")
        datasets = {run["dataset"] for run in runs}
        test_sizes = {run["n_test"] for run in runs}
        if len(datasets) != 1 or len(test_sizes) != 1 or \
           len({run["n_train"] for run in runs}) != 1 or \
           len({run["n_val"] for run in runs}) != 1 or \
           len({run["seed"] for run in runs}) != len(runs):
            raise ValueError(f"Inconsistent dataset or test size for {model}")
        expected_architecture = "gin" if model == "gin_ring" else model
        if any(run["model"] != expected_architecture or run.get("threads") != 2
               or run["hidden"] != 64 or run["learning_rate"] != 0.001
               for run in runs):
            raise ValueError(f"Inconsistent training protocol for {model}")
        if any(bool(run.get("ring_size_features")) != (model == "gin_ring")
               for run in runs):
            raise ValueError(f"Inconsistent ring-size features for {model}")
        if summary["dataset"] is None:
            summary["dataset"] = datasets.pop()
        elif summary["dataset"] != next(iter(datasets)):
            raise ValueError("Different datasets among model runs")
        keys = ("fnr", "precision", "recall", "f1", "accuracy", "roc_auc")
        metrics = {}
        for key in keys:
            values = [run["test_metrics"][key] for run in runs]
            metrics[key] = {"mean": round(mean(values), 4),
                            "std": round(stdev(values), 4) if len(values) > 1 else None}
        summary["models"][model] = {
            "seeds": [run["seed"] for run in runs],
            "n_test": runs[0]["n_test"],
            "metrics": metrics,
        }

    fingerprint_paths = sorted(args.root.glob(f"{args.prefix}_fingerprint_seed*.json"))
    fingerprint_runs = []
    for path in fingerprint_paths:
        payload = json.loads(path.read_text())
        if len(payload) != 1:
            raise ValueError(f"Expected one dataset in {path}")
        dataset, run = next(iter(payload.items()))
        if summary["dataset"] is not None and dataset != summary["dataset"]:
            raise ValueError("Fingerprint and GNN runs use different datasets")
        summary["dataset"] = dataset
        fingerprint_runs.append(run)
    for model, key in (("rf", "fingerprint_rf"),
                       ("xgboost", "fingerprint_xgboost")):
        runs = [run for run in fingerprint_runs if key in run]
        if not runs:
            continue
        sizes = {run["n_test"] for run in runs}
        if len(sizes) != 1:
            raise ValueError("Fingerprint runs have different test sizes")
        metrics = {}
        for output_key, source_key in (("fnr", "false_negative_rate"),
                                       ("precision", "precision"),
                                       ("recall", "recall"),
                                       ("f1", "f1"),
                                       ("accuracy", "accuracy"),
                                       ("roc_auc", "roc_auc")):
            values = [run[key][source_key] for run in runs]
            metrics[output_key] = {
                "mean": round(mean(values), 4),
                "std": round(stdev(values), 4) if len(values) > 1 else None,
            }
        summary["models"][model] = {
            "seeds": [run["seed"] for run in runs],
            "n_test": runs[0]["n_test"],
            "metrics": metrics,
        }

    if not summary["models"]:
        raise ValueError(f"No runs found for {args.prefix}")
    output = args.out or args.root / f"{args.prefix}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
