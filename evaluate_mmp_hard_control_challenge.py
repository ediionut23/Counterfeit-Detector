"""Score a parent-matched MMP challenge with saved mixed_20k baselines.

The decision thresholds come only from the original validation split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from build_mmp_hard_control_challenge import Components
from train_baseline import morgan_array, prop_row, threshold_at_fpr


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def exact_interval(successes: int, trials: int) -> list[float]:
    if trials == 0:
        return [None, None]
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return [round(lower, 4), round(upper, 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("final_dataset/examples/mixed_20k/dataset.csv"))
    parser.add_argument("--challenge", type=Path,
                        default=Path("research_runs/mmp_hard_control_bank_pilot/challenge.csv"))
    parser.add_argument("--models", type=Path,
                        default=Path("training_runs/mixed_20k/fingerprint_seed42_models"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory; defaults to challenge directory")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Parent-scaffold cluster bootstrap resamples")
    parser.add_argument("--component-relations", type=Path,
                        default=Path("final_dataset/relations.csv"))
    args = parser.parse_args()
    if not 0 < args.target_fpr < 1:
        parser.error("--target-fpr must be in (0, 1)")
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    metadata = json.loads((args.models / "metadata.json").read_text())
    if metadata["source_csv"] != str(args.dataset) or not metadata["train_only"]:
        raise ValueError("Saved model metadata do not match the original training CSV")
    dataset_sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.dataset == Path("final_dataset/examples/mixed_20k/dataset.csv"):
        manifest = json.loads(Path("final_dataset/manifest_sha256.json").read_text())
        if dataset_sha != manifest["examples/mixed_20k/dataset.csv"]["sha256"]:
            raise ValueError("Training CSV differs from audited final-bank manifest")
    else:
        summary = json.loads((args.dataset.parent / "summary.json").read_text())
        if dataset_sha != summary["dataset_sha256"]:
            raise ValueError("Training CSV differs from dedicated split manifest")
        if hashlib.sha256(args.challenge.read_bytes()).hexdigest() != summary["challenge_sha256"]:
            raise ValueError("Challenge CSV differs from dedicated split manifest")

    source = read_csv(args.dataset)
    validation = [row for row in source if row["split"] == "val"]
    challenge = read_csv(args.challenge)
    test_by_id = {r["id"]: r for r in source if r["split"] == "test"}
    if args.dataset != Path("final_dataset/examples/mixed_20k/dataset.csv"):
        if len(test_by_id) != len(challenge) or any(
            row["id"] not in test_by_id or
            (row["smiles"], row["label"]) !=
            (test_by_id[row["id"]]["smiles"], test_by_id[row["id"]]["label"])
            for row in challenge
        ):
            raise ValueError("Challenge does not match the dedicated test partition")
    if not challenge or len(challenge) % 2:
        raise ValueError("Challenge needs complete positive/control pairs")
    pairs = {}
    for index, row in enumerate(challenge):
        pairs.setdefault(row["pair_id"], {})[row["label"]] = index
    if any(set(indices) != {"0", "1"} for indices in pairs.values()):
        raise ValueError("Each challenge pair must have one generated and one source molecule")
    group_names = sorted({challenge[indexes["1"]]["parent_scaffold"]
                          for indexes in pairs.values()})
    group_to_index = {name: index for index, name in enumerate(group_names)}
    pair_group_indices = np.array([group_to_index[challenge[indexes["1"]]["parent_scaffold"]]
                                   for indexes in pairs.values()], dtype=int)
    rng = np.random.default_rng(20260927)
    bootstrap_groups = rng.integers(0, len(group_names),
                                    size=(args.bootstrap, len(group_names)))
    group_pair_counts = np.bincount(pair_group_indices, minlength=len(group_names))
    sampled_denominators = group_pair_counts[bootstrap_groups].sum(axis=1)
    train_val_smiles = {r["smiles"] for r in source if r["split"] in ("train", "val")}
    train_val_scaffolds = {r["own_scaffold"] for r in source
                           if r["split"] in ("train", "val")}
    if any(r["smiles"] in train_val_smiles or r["own_scaffold"] in train_val_scaffolds
           for r in challenge):
        raise ValueError("Challenge overlaps the original train/validation molecules or own scaffolds")
    components = Components()
    with args.component_relations.open(newline="") as stream:
        for relation in csv.DictReader(stream):
            components.union(relation["parent_smiles"], relation["smiles"])
    training_components = {components.root(r["smiles"]) for r in source
                           if r["split"] in ("train", "val")}
    training_components.update(components.root(r["parent_smiles"]) for r in source
                               if r["split"] in ("train", "val") and r["label"] == "1")
    if any(components.root(r["smiles"]) in training_components
           or components.root(r["parent_smiles"]) in training_components
           for r in challenge):
        raise ValueError("Challenge has parent-child component overlap with training")

    val_prop = np.array([prop_row(r["smiles"]) for r in validation], float)
    val_fp = np.array([morgan_array(r["smiles"]) for r in validation])
    challenge_prop = np.array([prop_row(r["smiles"]) for r in challenge], float)
    challenge_fp = np.array([morgan_array(r["smiles"]) for r in challenge])
    properties = joblib.load(args.models / "properties.joblib")
    rf = joblib.load(args.models / "rf.joblib")
    xgboost = XGBClassifier()
    xgboost.load_model(args.models / "xgboost.json")
    scores = {
        "properties": (properties["model"].predict_proba(
            properties["scaler"].transform(val_prop))[:, 1],
            properties["model"].predict_proba(
                properties["scaler"].transform(challenge_prop))[:, 1]),
        "rf": (rf.predict_proba(val_fp)[:, 1], rf.predict_proba(challenge_fp)[:, 1]),
        "xgboost": (xgboost.predict_proba(val_fp)[:, 1],
                    xgboost.predict_proba(challenge_fp)[:, 1]),
    }
    y_val = np.array([int(r["label"]) for r in validation])
    y_challenge = np.array([int(r["label"]) for r in challenge])
    output = {
        "dataset": str(args.dataset), "dataset_sha256": dataset_sha,
        "challenge": str(args.challenge),
        "challenge_sha256": hashlib.sha256(args.challenge.read_bytes()).hexdigest(),
        "seed": metadata["seed"], "target_validation_fpr": args.target_fpr,
        "n_pairs": len(pairs), "models": {},
        "interpretation": "Internal source-MMP provenance challenge; no clinical counterfeit labels or external validation.",
    }
    prediction_rows = []
    for name, (val_score, challenge_score) in scores.items():
        cut = threshold_at_fpr(y_val, val_score, args.target_fpr)
        predictions = challenge_score >= cut
        negative, positive = y_challenge == 0, y_challenge == 1
        fp = int(predictions[negative].sum())
        tp = int(predictions[positive].sum())
        pair_margins = [float(challenge_score[indices["1"]] - challenge_score[indices["0"]])
                        for indices in pairs.values()]
        group_fp = np.bincount(pair_group_indices,
                               weights=[int(predictions[indices["0"]]) for indices in pairs.values()],
                               minlength=len(group_names))
        group_tp = np.bincount(pair_group_indices,
                               weights=[int(predictions[indices["1"]]) for indices in pairs.values()],
                               minlength=len(group_names))
        bootstrap_fpr = group_fp[bootstrap_groups].sum(axis=1) / sampled_denominators
        bootstrap_tpr = group_tp[bootstrap_groups].sum(axis=1) / sampled_denominators
        per_method = {}
        for method in ("M", "E"):
            selected = positive & np.array([r["generation_method"] == method for r in challenge])
            count = int(selected.sum())
            per_method[method] = {"n": count, "detected": int(predictions[selected].sum()),
                                  "tpr": round(float(predictions[selected].mean()), 4) if count else None}
        output["models"][name] = {
            "validation_threshold": float(cut),
            "validation_fpr": round(float((val_score[y_val == 0] >= cut).mean()), 4),
            "challenge_false_positives": fp, "challenge_negatives": int(negative.sum()),
            "challenge_fpr": round(fp / int(negative.sum()), 4),
            "challenge_fpr_95pct_exact_ci": exact_interval(fp, int(negative.sum())),
            "challenge_fpr_95pct_scaffold_bootstrap_ci": [round(float(x), 4)
                for x in np.quantile(bootstrap_fpr, [0.025, 0.975])],
            "challenge_true_positives": tp, "challenge_positives": int(positive.sum()),
            "challenge_tpr": round(tp / int(positive.sum()), 4),
            "challenge_tpr_95pct_exact_ci": exact_interval(tp, int(positive.sum())),
            "challenge_tpr_95pct_scaffold_bootstrap_ci": [round(float(x), 4)
                for x in np.quantile(bootstrap_tpr, [0.025, 0.975])],
            "challenge_auc": round(float(roc_auc_score(y_challenge, challenge_score)), 4),
            "mean_paired_score_margin": round(float(np.mean(pair_margins)), 4),
            "pairs_positive_scored_higher": sum(margin > 0 for margin in pair_margins),
            "positive_method": per_method,
        }
        prediction_rows.extend({"model": name, "pair_id": row["pair_id"],
                                "id": row["id"], "label": row["label"],
                                "origin": row["origin"], "score": float(score),
                                "predicted_at_validation_fpr": int(predicted)}
                               for row, score, predicted in zip(challenge, challenge_score, predictions))
    destination = args.out or args.challenge.parent
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    with (destination / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "pair_id", "id", "label",
                                                   "origin", "score", "predicted_at_validation_fpr"),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
