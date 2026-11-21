"""Compare fixed E RF/XGBoost and saved GIN on parent-child score ordering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from audit_parent_child_counterfactual import summarize
from train_baseline import load_split, metrics, morgan_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("benchmark/verified_edits/manifest.csv"))
    parser.add_argument("--gin-pairs", type=Path,
                        default=Path("benchmark/explanations/evolved_gin_seed42/all_pairs/parent_child_counterfactual.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/pair_ordering"))
    args = parser.parse_args()
    split = load_split(args.dataset, train_only=True)
    train, test = split["train"], split["test"]
    positives = [row for row in test if row["label"] == "1"]
    with args.manifest.open(newline="") as stream:
        verified = {(row["parent_smiles"], row["smiles"])
                    for row in csv.DictReader(stream)}
    with args.gin_pairs.open(newline="") as stream:
        gin_pairs = {row["id"]: row for row in csv.DictReader(stream)}
    if set(gin_pairs) != {row["id"] for row in positives}:
        raise ValueError("Saved GIN pair scores do not cover the E test positives")
    print(f"Featurizing {len(train)} train rows and {len(test)} test rows", flush=True)
    x_train = np.array([morgan_array(row["smiles"]) for row in train])
    y_train = np.array([int(row["label"]) for row in train])
    x_test = np.array([morgan_array(row["smiles"]) for row in test])
    y_test = np.array([int(row["label"]) for row in test])
    pair_smiles = sorted({smiles for row in positives
                          for smiles in (row["parent_smiles"], row["smiles"])})
    x_pairs = np.array([morgan_array(smiles) for smiles in pair_smiles])
    expected = json.loads((Path("benchmark/gnn_comparison") /
                           f"Evolved_fingerprint_seed{args.seed}.json").read_text())["evolved"]
    models = {
        "rf": (RandomForestClassifier(n_estimators=300, n_jobs=args.jobs,
                                      random_state=args.seed), "fingerprint_rf"),
        "xgboost": (XGBClassifier(n_estimators=300, max_depth=6,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist",
                                  n_jobs=args.jobs, random_state=args.seed,
                                  eval_metric="logloss"), "fingerprint_xgboost"),
    }
    scores = {}
    test_checks = {}
    for name, (model, reference_key) in models.items():
        print(f"Fitting {name}", flush=True)
        model.fit(x_train, y_train)
        test_score = model.predict_proba(x_test)[:, 1]
        check = metrics(y_test, (test_score > 0.5).astype(int), test_score)
        if check != expected[reference_key]:
            raise ValueError(f"{name} test metrics differ from saved baseline: {check}")
        test_checks[name] = check
        pair_score = model.predict_proba(x_pairs)[:, 1]
        scores[name] = dict(zip(pair_smiles, pair_score))

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    summaries = {}
    for name in ("rf", "xgboost", "gin"):
        paired = []
        for row in positives:
            if name == "gin":
                parent = float(gin_pairs[row["id"]]["parent_score"])
                child = float(gin_pairs[row["id"]]["child_score"])
            else:
                parent = float(scores[name][row["parent_smiles"]])
                child = float(scores[name][row["smiles"]])
            record = {"id": row["id"], "model": name,
                      "verified_one_atom": int((row["parent_smiles"], row["smiles"]) in verified),
                      "parent_score": round(parent, 8),
                      "child_score": round(child, 8),
                      "score_change": round(child - parent, 8)}
            records.append(record)
            paired.append(record)
        summaries[name] = {
            "all_test_pairs": summarize(paired),
            "verified_one_atom_pairs": summarize([
                record for record in paired if record["verified_one_atom"]]),
        }
    output_csv = args.out / f"pair_scores_seed{args.seed}.csv"
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "seed": args.seed, "dataset": str(args.dataset),
        "train_rows": len(train), "test_counterfeit_pairs": len(positives),
        "protocol": "fixed E train-only fingerprint models; same seed and hyperparameters as saved test baselines",
        "test_metric_reproduction": test_checks,
        "models": summaries,
        "interpretation_limit": "A molecule-only classifier is not told which parent it imitates; ordering measures sensitivity to generated parent-child changes, not chemical causality",
        "per_pair": str(output_csv),
    }
    (args.out / f"summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({name: summaries[name] for name in summaries}, indent=2))


if __name__ == "__main__":
    main()
