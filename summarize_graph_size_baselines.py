"""Compare fixed-count small/large counterfeit training subsets with random 1:10."""

from __future__ import annotations

import json
import csv
from pathlib import Path

from build_graph_size_variants import size_stats


def main() -> None:
    root = Path("benchmark/graph_size")
    variants = json.loads((root / "summary.json").read_text())
    random_payload = json.loads(Path("benchmark/imbalance/baseline_1_to_10.json").read_text())
    with Path("benchmark/imbalance/evolved_train_1_to_10.csv").open(newline="") as stream:
        random_positives = [row for row in csv.DictReader(stream)
                            if row["split"] == "train" and row["label"] == "1"]
    results = {"random": random_payload["evolved_train_1_to_10"]}
    for name in ("small", "large"):
        payload = json.loads((root / f"baseline_{name}.json").read_text())
        results[name] = payload[f"evolved_train_size_{name}"]
    expected_train = variants["train_authentic"] + variants["train_counterfeit"]
    for name, run in results.items():
        if run["n_train"] != expected_train or run["n_test"] != 9250 or \
           run["seed"] != variants["seed"] or not run["train_only"] or \
           not run["threshold_tuned_on_val"]:
            raise ValueError(f"Incompatible run: {name}")
    summary = {
        "source": variants["source"], "seed": variants["seed"],
        "protocol": "3,700 counterfeit + 37,001 authentic train; same balanced val/test; RF/XGB fixed settings; 0.5 and balanced-val F1 thresholds",
        "counterfeit_train_size": {
            "random": size_stats(random_positives),
            **{name: meta["counterfeit_size"] for name, meta in variants["variants"].items()},
        },
        "results": {
            name: {
                "n_train": run["n_train"], "n_test": run["n_test"],
                "fingerprint_rf": run["fingerprint_rf"],
                "fingerprint_xgboost": run["fingerprint_xgboost"],
                "validation_thresholds": {
                    key: value for key, value in run["validation_thresholds"].items()
                    if key.startswith("fingerprint_")
                },
                "validation_tuned_test": {
                    key: value for key, value in run["validation_tuned_test"].items()
                    if key.startswith("fingerprint_")
                },
            }
            for name, run in results.items()
        },
    }
    (root / "baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, run in summary["results"].items():
        print(name)
        for model in ("fingerprint_rf", "fingerprint_xgboost"):
            fixed = run[model]
            tuned = run["validation_tuned_test"][model]
            print(f"  {model}: AUC={fixed['roc_auc']}, "
                  f"FNR@0.5={fixed['false_negative_rate']}, "
                  f"FNR@val={tuned['false_negative_rate']}, "
                  f"precision@val={tuned['precision']}")


if __name__ == "__main__":
    main()
