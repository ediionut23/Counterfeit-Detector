"""Aggregate equal-split fixed-setting M/E architecture runs across seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = ("roc_auc", "fnr", "precision", "recall", "f1", "accuracy")


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": round(mean(values), 4),
            "sample_sd": round(stdev(values), 4) if len(values) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("training_runs/mixed_20k"))
    parser.add_argument("--epoch-cap", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    expected = set(args.seeds)
    if len(expected) != len(args.seeds):
        parser.error("Seeds must be unique")

    dataset_root = Path("final_dataset/examples/mixed_20k")
    data_summary = json.loads((dataset_root / "summary.json").read_text())
    audit = json.loads((dataset_root / "audit.json").read_text())
    manifest = json.loads(Path("final_dataset/manifest_sha256.json").read_text())
    if not audit["passed"]:
        raise ValueError("Dataset audit failed")
    dataset_csv = dataset_root / "dataset.csv"
    actual_digest = hashlib.sha256(dataset_csv.read_bytes()).hexdigest()
    if actual_digest != manifest["examples/mixed_20k/dataset.csv"]["sha256"]:
        raise ValueError("Training CSV no longer matches the final-bank manifest")
    expected_test = sum(data_summary["split_label_counts"]["test"].values())
    # JSON values in the constructor summary are integers, not strings.
    expected_train = sum(data_summary["split_label_counts"]["train"].values())
    expected_val = sum(data_summary["split_label_counts"]["val"].values())

    runs: dict[str, list[dict]] = {name: [] for name in
                                   ("properties", "rf", "xgboost", "gin", "gcn", "gat")}
    for seed in args.seeds:
        path = args.root / f"fingerprint_seed{seed}.json"
        payload = json.loads(path.read_text())
        row = payload["mixed_20k"]
        if (row["seed"] != seed or row["n_train"] != expected_train or
            row["n_test"] != expected_test or not row["train_only"] or
            row["train_csv"] != str(dataset_csv) or row["test_csv"] != str(dataset_csv)):
            raise ValueError(f"Inconsistent baseline run: {path}")
        for name, key in (("properties", "property_only"),
                          ("rf", "fingerprint_rf"),
                          ("xgboost", "fingerprint_xgboost")):
            metric = dict(row[key])
            metric["fnr"] = metric.pop("false_negative_rate")
            runs[name].append({"seed": seed, "metrics": metric, "path": str(path)})
        for name in ("gin", "gcn", "gat"):
            path = args.root / f"{name}_seed{seed}_{args.epoch_cap}.json"
            row = json.loads(path.read_text())
            if (row["seed"] != seed or row["model"] != name or
                row["n_train"] != expected_train or row["n_val"] != expected_val or
                row["n_test"] != expected_test or row["epochs_run"] > args.epoch_cap or
                row["hidden"] != 64 or row["learning_rate"] != 0.001 or
                row["ring_size_features"] or row["threads"] != 2):
                raise ValueError(f"Inconsistent GNN run: {path}")
            runs[name].append({"seed": seed, "metrics": row["test_metrics"],
                               "best_val_auc": row["best_val_auc"],
                               "epochs_run": row["epochs_run"], "path": str(path)})

    aggregate = {}
    for name, rows in runs.items():
        if {row["seed"] for row in rows} != expected:
            raise ValueError(f"Missing or duplicate seeds for {name}")
        aggregate[name] = {
            "n_runs": len(rows),
            "seeds": args.seeds,
            "metrics": {key: stats([float(row["metrics"][key]) for row in rows])
                        for key in METRICS},
            "runs": rows,
        }
    result = {
        "dataset": str(dataset_root / "dataset.csv"),
        "dataset_sha256": manifest["examples/mixed_20k/dataset.csv"]["sha256"],
        "n_train": expected_train, "n_val": expected_val,
        "n_test": expected_test, "seeds": args.seeds,
        "gnn_epoch_cap": args.epoch_cap,
        "protocol": "Train-only fit; GNN epoch selected by validation AUC; fixed hyperparameters; 0.5 test threshold",
        "limitations": "No equal hyperparameter tuning; selected chemistry modules; scaffold-disjoint split with strict child test filtering; generated labels",
        "models": aggregate,
    }
    output = args.root / f"architecture_summary_{args.epoch_cap}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    def cell(model: str, metric: str) -> str:
        values = aggregate[model]["metrics"][metric]
        return f"{values['mean']:.3f} ± {values['sample_sd']:.3f}"
    lines = [
        f"# Fixed-setting architecture comparison ({args.epoch_cap}-epoch cap)",
        "",
        f"Dataset SHA-256: `{result['dataset_sha256']}`. Same {expected_test:,} test rows for every model.",
        "Values are means ± sample standard deviations over seeds " +
        ", ".join(map(str, args.seeds)) + ".",
        "",
        "| Model | AUC | FNR at 0.5 | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in (("properties", "Three properties"), ("rf", "Morgan + RF"),
                        ("xgboost", "Morgan + XGBoost"), ("gin", "GIN"),
                        ("gcn", "GCN"), ("gat", "GAT")):
        lines.append(f"| {label} | {cell(name, 'roc_auc')} | {cell(name, 'fnr')} | "
                     f"{cell(name, 'precision')} | {cell(name, 'f1')} |")
    lines += [
        "",
        "The comparison uses fixed settings without equal hyperparameter optimization. "
        "Each label marks a generated analogue relative to a reference molecule; "
        "this is not a clinical counterfeit detection result.",
        "",
    ]
    (args.root / f"architecture_summary_{args.epoch_cap}.md").write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
