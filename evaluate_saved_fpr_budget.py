"""Evaluate saved mixed_20k models at a validation-selected false-alarm budget."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from train_baseline import morgan_array, prop_row, threshold_at_fpr


def read_predictions(path: Path, expected: list[dict]) -> np.ndarray:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected):
        raise ValueError(f"Wrong number of predictions: {path}")
    for pred, molecule in zip(rows, expected):
        if (pred["id"], pred["split"], pred["label"]) != (
            molecule["id"], molecule["split"], molecule["label"]
        ):
            raise ValueError(f"Prediction order or labels differ: {path}")
    scores = np.array([float(row["counterfeit_score"]) for row in rows])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"Invalid probability: {path}")
    return scores


def evaluate(val: list[dict], test: list[dict], val_score: np.ndarray,
             test_score: np.ndarray, target_fpr: float) -> dict:
    y_val = np.array([int(row["label"]) for row in val])
    y_test = np.array([int(row["label"]) for row in test])
    threshold = threshold_at_fpr(y_val, val_score, target_fpr)
    val_pred = val_score >= threshold
    test_pred = test_score >= threshold
    test_negative = y_test == 0
    test_positive = y_test == 1
    val_negative = y_val == 0
    tp = int((test_pred & test_positive).sum())
    fp = int((test_pred & test_negative).sum())
    fn = int((~test_pred & test_positive).sum())
    group = {}
    for method in ("E", "M"):
        selected = test_positive & np.array([row["generation_method"] == method
                                             for row in test])
        count = int(selected.sum())
        missed = int((~test_pred & selected).sum())
        group[method] = {"n_positive": count, "false_negatives": missed,
                         "fnr": round(missed / count, 4)}
    return {
        "threshold": float(threshold),
        "validation_fpr": round(float(val_pred[val_negative].mean()), 4),
        "validation_false_positives": int(val_pred[val_negative].sum()),
        "validation_negatives": int(val_negative.sum()),
        "test_fpr": round(float(fp / test_negative.sum()), 4),
        "test_fnr": round(float(fn / test_positive.sum()), 4),
        "test_precision": round(float(tp / (tp + fp)), 4) if tp + fp else 0.0,
        "test_auc": round(float(roc_auc_score(y_test, test_score)), 4),
        "test_tp": tp, "test_fp": fp, "test_fn": fn,
        "test_negatives": int(test_negative.sum()),
        "test_positives": int(test_positive.sum()),
        "positive_method": group,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=Path("final_dataset/examples/mixed_20k/dataset.csv"))
    parser.add_argument("--runs", type=Path,
                        default=Path("training_runs/mixed_20k"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-cap", type=int, default=80)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()
    if not 0 < args.target_fpr < 1:
        parser.error("--target-fpr must be in (0, 1)")
    manifest = json.loads(Path("final_dataset/manifest_sha256.json").read_text())
    digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    if digest != manifest["examples/mixed_20k/dataset.csv"]["sha256"]:
        raise ValueError("CSV differs from audited final-bank manifest")
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    val = [row for row in rows if row["split"] == "val"]
    test = [row for row in rows if row["split"] == "test"]
    if len(val) != 3968 or len(test) != 2948:
        raise ValueError("Unexpected validation or test split size")
    model_dir = args.runs / f"fingerprint_seed{args.seed}_models"
    metadata = json.loads((model_dir / "metadata.json").read_text())
    if metadata["source_csv"] != str(args.csv) or metadata["seed"] != args.seed \
            or not metadata["train_only"]:
        raise ValueError("Saved baseline models do not match dataset and seed")
    properties = joblib.load(model_dir / "properties.joblib")
    rf = joblib.load(model_dir / "rf.joblib")
    xgb = XGBClassifier()
    xgb.load_model(model_dir / "xgboost.json")
    val_properties = np.array([prop_row(row["smiles"]) for row in val], float)
    val_fingerprints = np.array([morgan_array(row["smiles"]) for row in val])
    val_scores = {
        "properties": properties["model"].predict_proba(
            properties["scaler"].transform(val_properties))[:, 1],
        "rf": rf.predict_proba(val_fingerprints)[:, 1],
        "xgboost": xgb.predict_proba(val_fingerprints)[:, 1],
    }
    result = {"dataset": str(args.csv), "dataset_sha256": digest,
              "seed": args.seed, "target_validation_fpr": args.target_fpr,
              "n_val": len(val), "n_test": len(test), "models": {}}
    for name in ("properties", "rf", "xgboost"):
        path = args.runs / f"fingerprint_seed{args.seed}_predictions/{name}.csv"
        test_scores = read_predictions(path, test)
        result["models"][name] = evaluate(val, test, val_scores[name],
                                           test_scores, args.target_fpr)
    for name in ("gin", "gcn", "gat"):
        stem = f"{name}_seed{args.seed}_{args.epoch_cap}"
        val_scores = read_predictions(args.runs / f"{stem}_val_predictions.csv", val)
        test_scores = read_predictions(args.runs / f"{stem}_test_predictions.csv", test)
        result["models"][name] = evaluate(val, test, val_scores,
                                           test_scores, args.target_fpr)
    stem = f"fpr_budget_seed{args.seed}_{args.epoch_cap}"
    json_path = args.runs / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# Validation FPR budget: {args.target_fpr:.0%} (seed {args.seed})", "",
             f"The threshold is selected only on {len(val):,} validation rows",
             f"({sum(int(r['label']) == 0 for r in val):,} authentic). The same",
             f"{len(test):,} scaffold-held-out test rows are scored for every model.",
             "A validation target does not guarantee the same FPR on test.", "",
             "| Model | Val FPR | Test FPR | Test FNR | Test precision | E FNR | M FNR |",
             "|---|---:|---:|---:|---:|---:|---:|",
             ]
    for name, label in (("properties", "Three properties"),
                        ("rf", "Morgan + RF"), ("xgboost", "Morgan + XGBoost"),
                        ("gin", "GIN"), ("gcn", "GCN"), ("gat", "GAT")):
        value = result["models"][name]
        lines.append(f"| {label} | {value['validation_fpr']:.3f} | "
                     f"{value['test_fpr']:.3f} | {value['test_fnr']:.3f} | "
                     f"{value['test_precision']:.3f} | "
                     f"{value['positive_method']['E']['fnr']:.3f} | "
                     f"{value['positive_method']['M']['fnr']:.3f} |")
    lines += ["", "E and M are unequal selected chemistry classes (1,164 and 310",
              "positive test rows); their difference does not isolate generator effects.",
              "Labels represent generated molecular analogues, not clinical samples.", ""]
    (args.runs / f"{stem}.md").write_text("\n".join(lines))
    print(json_path)


if __name__ == "__main__":
    main()
