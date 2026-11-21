"""Compare existing E models on original and stricter child-scaffold tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from train_gnn import metrics


def stats(values: list[float]) -> dict:
    return {"mean": round(mean(values), 4),
            "std": round(stdev(values), 4) if len(values) > 1 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    args = parser.parse_args()
    directory = args.root / "strict_child_scaffold"
    original_dir = args.root / "gnn_comparison"
    seeds = (42, 43, 44)
    runs = {model: {"original": [], "strict": []}
            for model in ("gin", "rf", "xgboost")}
    for seed in seeds:
        gnn = json.loads((original_dir / f"Evolved_gin_seed{seed}.json").read_text())
        fingerprint = json.loads((original_dir / f"Evolved_fingerprint_seed{seed}.json").read_text())["evolved"]
        strict_fp = json.loads((directory / f"fingerprint_seed{seed}.json").read_text())["evolved_strict_child_scaffold"]
        if (gnn["seed"] != seed or fingerprint["seed"] != seed or strict_fp["seed"] != seed or
                gnn["n_train"] != fingerprint["n_train"] or
                gnn["n_test"] != fingerprint["n_test"] or
                fingerprint["n_train"] != strict_fp["n_train"]):
            raise ValueError(f"Training protocol differs at seed {seed}")
        with (directory / f"GIN_seed{seed}_predictions.csv").open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != strict_fp["n_test"] or any(
                row["split"] != "test" for row in predictions):
            raise ValueError(f"Strict GIN predictions inconsistent at seed {seed}")
        strict_gin = metrics([int(row["label"]) for row in predictions],
                             [int(row["predicted_label"]) for row in predictions],
                             [float(row["counterfeit_score"]) for row in predictions])
        runs["gin"]["original"].append(gnn["test_metrics"])
        runs["gin"]["strict"].append(strict_gin)
        for model, key in (("rf", "fingerprint_rf"),
                           ("xgboost", "fingerprint_xgboost")):
            runs[model]["original"].append(fingerprint[key])
            runs[model]["strict"].append(strict_fp[key])

    summary = {
        "seeds": seeds, "original_test_rows": gnn["n_test"],
        "strict_test_rows": len(predictions),
        "note": "Same fitted-model settings; the strict test is a selected subset, so score changes combine overlap removal and composition shift",
        "models": {},
    }
    for model, data in runs.items():
        mapping = {"roc_auc": "roc_auc", "fnr": "fnr" if model == "gin" else "false_negative_rate",
                   "precision": "precision", "f1": "f1"}
        summary["models"][model] = {}
        for label in ("original", "strict"):
            summary["models"][model][label] = {
                metric: stats([run[source_key] for run in data[label]])
                for metric, source_key in mapping.items()
            }
        summary["models"][model]["paired_strict_minus_original"] = {
            metric: stats([strict[source_key] - original[source_key]
                           for original, strict in zip(data["original"], data["strict"])])
            for metric, source_key in mapping.items()
        }
    (directory / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
