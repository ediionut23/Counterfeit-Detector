"""Aggregate matched-seed RF/XGBoost scaffold-diversity stress results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def stats(values: list[float]) -> dict:
    return {"mean": round(mean(values), 4),
            "std": round(stdev(values), 4) if len(values) > 1 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path("benchmark/scaffold_diversity"))
    args = parser.parse_args()
    construction = json.loads((args.directory / "summary.json").read_text())
    expected_train = 2 * construction["train_examples_per_class"]
    expected_test = construction["variants"]["low"]["split_label_counts"]["test"]
    expected_test = sum(expected_test.values())
    seeds = (42, 43, 44)
    runs = {}
    for name in ("low", "high"):
        runs[name] = []
        for seed in seeds:
            path = args.directory / f"baseline_{name}_seed{seed}.json"
            payload = json.loads(path.read_text())[f"scaffold_{name}"]
            if (payload["seed"] != seed or payload["n_train"] != expected_train or
                    payload["n_test"] != expected_test or payload.get("target_fpr") != 0.01):
                raise ValueError(f"Unexpected protocol in {path}")
            runs[name].append(payload)

    report = {"dataset": "evolved", "seeds": seeds,
              "protocol": "same 3700 authentic train rows; 3700 counterfeit train rows; "
                          "matched heavy-atom bins; fixed val/test; RF/XGB train-only",
              "models": {}}
    for model in ("fingerprint_rf", "fingerprint_xgboost"):
        fields = {
            "auc": lambda r: r[model]["roc_auc"],
            "fixed_fnr": lambda r: r[model]["false_negative_rate"],
            "fixed_f1": lambda r: r[model]["f1"],
            "val_f1_tuned_fnr": lambda r: r["validation_tuned_test"][model]["false_negative_rate"],
            "val_f1_tuned_precision": lambda r: r["validation_tuned_test"][model]["precision"],
            "val_f1_tuned_f1": lambda r: r["validation_tuned_test"][model]["f1"],
            "val_fpr1_fnr": lambda r: r["validation_fpr_test"][model]["false_negative_rate"],
            "val_fpr1_test_fpr": lambda r: r["validation_fpr_test"][model]["false_positive_rate"],
        }
        report["models"][model] = {
            name: {field: stats([extract(run) for run in runs[name]])
                   for field, extract in fields.items()}
            for name in ("low", "high")
        }
        report["models"][model]["paired_high_minus_low"] = {
            field: stats([extract(high) - extract(low)
                          for low, high in zip(runs["low"], runs["high"])])
            for field, extract in fields.items()
        }
    output = args.directory / "baseline_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
