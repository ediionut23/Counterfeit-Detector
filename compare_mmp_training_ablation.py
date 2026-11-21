"""Paired parent-scaffold bootstrap for the hard-negative training ablation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("research_runs/mmp_dedicated_1000"))
    parser.add_argument("--hard", type=Path, default=Path("research_runs/mmp_dedicated_hard_train"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    base_eval = json.loads((args.base / "evaluation.json").read_text())
    hard_eval = json.loads((args.hard / "evaluation.json").read_text())
    if base_eval["challenge_sha256"] != hard_eval["challenge_sha256"]:
        raise ValueError("Ablation must use the identical held-out challenge")
    challenge_path = args.base / "challenge.csv"
    if not challenge_path.exists():
        challenge_path = args.base.parent / "challenge.csv"
    challenge = read_csv(challenge_path)
    groups = {}
    for row in challenge:
        if row["label"] == "1":
            groups[row["pair_id"]] = row["parent_scaffold"]
    group_names = sorted(set(groups.values()))
    group_index = {name: index for index, name in enumerate(group_names)}
    pair_counts = np.bincount([group_index[groups[pair]] for pair in sorted(groups)],
                              minlength=len(group_names))
    rng = np.random.default_rng(20260927)
    draws = rng.integers(0, len(group_names), size=(args.bootstrap, len(group_names)))
    denominators = pair_counts[draws].sum(axis=1)
    base_preds = read_csv(args.base / "predictions.csv")
    hard_preds = read_csv(args.hard / "predictions.csv")
    base_by_key = {(r["model"], r["pair_id"], r["label"]): r for r in base_preds}
    hard_by_key = {(r["model"], r["pair_id"], r["label"]): r for r in hard_preds}
    if set(base_by_key) != set(hard_by_key):
        raise ValueError("Prediction sets do not match")
    result = {"n_pairs": len(groups), "n_parent_scaffolds": len(group_names),
              "bootstrap_resamples": args.bootstrap,
              "difference_definition": "hard-negative training minus original training",
              "models": {}}
    for model in sorted(base_eval["models"]):
        result["models"][model] = {}
        for label, metric in (("0", "fpr"), ("1", "tpr")):
            difference = np.zeros(len(group_names), dtype=float)
            for pair_id in groups:
                key = (model, pair_id, label)
                before = int(base_by_key[key]["predicted_at_validation_fpr"])
                after = int(hard_by_key[key]["predicted_at_validation_fpr"])
                difference[group_index[groups[pair_id]]] += after - before
            sample = difference[draws].sum(axis=1) / denominators
            result["models"][model][metric] = {
                "change": round(float(difference.sum() / len(groups)), 4),
                "95pct_parent_scaffold_bootstrap_ci": [round(float(x), 4)
                    for x in np.quantile(sample, [0.025, 0.975])],
            }
    destination = args.hard / "paired_ablation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
