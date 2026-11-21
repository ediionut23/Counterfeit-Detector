"""Evaluate M-only training on matched held-out M and E exact-edit analogs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from build_mmp_hard_control_challenge import Components
from exact_edit_context import edit_site_and_shell
from train_baseline import morgan_array, prop_row, threshold_at_fpr


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def exact_interval(successes: int, trials: int) -> list[float]:
    if not trials:
        return [None, None]
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return [round(lower, 4), round(upper, 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("research_runs/generator_transfer_exact"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    args = parser.parse_args()
    if not 0 < args.target_fpr < 1 or args.bootstrap < 100:
        parser.error("Need target FPR in (0,1) and at least 100 bootstrap draws")
    dataset_path = args.root / "dataset.csv"
    pair_path = args.root / "matched_M_E_test.csv"
    summary = json.loads((args.root / "summary.json").read_text())
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != summary["dataset_sha256"]:
        raise ValueError("Dataset hash differs from split manifest")
    if hashlib.sha256(pair_path.read_bytes()).hexdigest() != summary["pair_manifest_sha256"]:
        raise ValueError("Pair manifest hash differs from split manifest")
    rows = read_csv(dataset_path)
    if len({r["smiles"] for r in rows}) != len(rows):
        raise ValueError("Dataset repeats a molecular structure")
    with (args.bank / "molecules.csv").open(newline="") as stream:
        bank_methods = {r["smiles"]: r["methods"] for r in csv.DictReader(stream)}
    if any(bank_methods.get(r["smiles"]) != r["generation_method"]
           for r in rows if r["label"] == "1"):
        raise ValueError("Positive generator provenance is not exclusive in bank")
    components = Components()
    with (args.bank / "relations.csv").open(newline="") as stream:
        for relation in csv.DictReader(stream):
            components.union(relation["parent_smiles"], relation["smiles"])
    split_roots = {split: {components.root(r["smiles"]) for r in rows
                           if r["split"] == split} for split in ("train", "val", "test")}
    split_scaffolds = {split: {s for r in rows if r["split"] == split
                                for s in (r["scaffold"], r["own_scaffold"])}
                       for split in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_roots[left] & split_roots[right] or split_scaffolds[left] & split_scaffolds[right]:
            raise ValueError(f"Component or scaffold leakage between {left} and {right}")
    validation = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    if any(r["generation_method"] != "M" for r in rows
           if r["split"] in ("train", "val") and r["label"] == "1"):
        raise ValueError("Non-M positive in train/validation")
    positives_by_pair = defaultdict(dict)
    for index, row in enumerate(test):
        if row["label"] == "1":
            positives_by_pair[row["comparison_pair_id"]][row["generation_method"]] = index
    if len(positives_by_pair) != summary["matched_test_pairs"] or any(
        set(pair) != {"M", "E"} for pair in positives_by_pair.values()
    ):
        raise ValueError("Test positives do not form matched M/E pairs")
    for pair in positives_by_pair.values():
        m_row, e_row = test[pair["M"]], test[pair["E"]]
        if m_row["signature"] != e_row["signature"]:
            raise ValueError("Matched M/E pair has different elemental substitutions")
        if summary.get("require_matched_local_shell") and (
            edit_site_and_shell(m_row)[1] != edit_site_and_shell(e_row)[1]
        ):
            raise ValueError("Matched M/E pair has different first-shell contexts")
    pair_ids = list(positives_by_pair)
    scaffold_components = Components()
    for row in test:
        if row["label"] != "1":
            continue
        pair_node = "pair:" + row["comparison_pair_id"]
        for scaffold in (row["scaffold"], row["own_scaffold"]):
            scaffold_components.union(pair_node, "scaffold:" + scaffold)
    grouped_pairs = defaultdict(list)
    for index, pair_id in enumerate(pair_ids):
        grouped_pairs[scaffold_components.root("pair:" + pair_id)].append(index)
    cluster_indices = [np.array(indices, dtype=int) for indices in grouped_pairs.values()]
    model_dir = args.root / f"seed{args.seed}/models"
    metadata = json.loads((model_dir / "metadata.json").read_text())
    if (metadata["source_csv"] != str(dataset_path)
            or metadata["seed"] != args.seed or not metadata["train_only"]):
        raise ValueError("Saved model provenance differs from experiment")

    val_prop = np.array([prop_row(r["smiles"]) for r in validation], float)
    test_prop = np.array([prop_row(r["smiles"]) for r in test], float)
    val_fp = np.array([morgan_array(r["smiles"]) for r in validation])
    test_fp = np.array([morgan_array(r["smiles"]) for r in test])
    properties = joblib.load(model_dir / "properties.joblib")
    rf = joblib.load(model_dir / "rf.joblib")
    xgb = XGBClassifier()
    xgb.load_model(model_dir / "xgboost.json")
    scores = {
        "properties": (properties["model"].predict_proba(
            properties["scaler"].transform(val_prop))[:, 1],
            properties["model"].predict_proba(
                properties["scaler"].transform(test_prop))[:, 1]),
        "rf": (rf.predict_proba(val_fp)[:, 1], rf.predict_proba(test_fp)[:, 1]),
        "xgboost": (xgb.predict_proba(val_fp)[:, 1], xgb.predict_proba(test_fp)[:, 1]),
    }
    y_val = np.array([int(r["label"]) for r in validation])
    mask = {
        "source": np.array([r["label"] == "0" for r in test]),
        "M": np.array([r["label"] == "1" and r["generation_method"] == "M"
                       for r in test]),
        "E": np.array([r["label"] == "1" and r["generation_method"] == "E"
                       for r in test]),
    }
    pair_indices = list(positives_by_pair.values())
    rng = np.random.default_rng(20260927)
    bootstrap_pairs = rng.integers(0, len(pair_indices),
                                   size=(args.bootstrap, len(pair_indices)))
    rng_clusters = np.random.default_rng(20260928)
    bootstrap_clusters = rng_clusters.integers(0, len(cluster_indices),
                                               size=(args.bootstrap, len(cluster_indices)))
    cluster_sizes = np.array([len(indices) for indices in cluster_indices])
    sampled_cluster_sizes = cluster_sizes[bootstrap_clusters].sum(axis=1)
    output = {
        "dataset": str(dataset_path), "dataset_sha256": summary["dataset_sha256"],
        "seed": args.seed, "validation_negatives": int((y_val == 0).sum()),
        "target_validation_fpr": args.target_fpr,
        "test_M": int(mask["M"].sum()), "test_E": int(mask["E"].sum()),
        "test_source": int(mask["source"].sum()), "models": {},
        "test_pair_scaffold_clusters": len(cluster_indices),
        "largest_test_pair_scaffold_cluster": int(cluster_sizes.max()),
        "interpretation": "M-only structure training; matched exact elemental substitutions in M/E test; source-library negatives and virtual analog labels.",
    }
    prediction_rows = []
    for name, (val_scores, test_scores) in scores.items():
        cut = threshold_at_fpr(y_val, val_scores, args.target_fpr)
        predicted = test_scores >= cut
        fp = int(predicted[mask["source"]].sum())
        positive_results = {}
        negative_scores = test_scores[mask["source"]]
        for method in ("M", "E"):
            selected = mask[method]
            tp = int(predicted[selected].sum())
            y = np.r_[np.zeros(len(negative_scores)), np.ones(int(selected.sum()))]
            s = np.r_[negative_scores, test_scores[selected]]
            positive_results[method] = {
                "detected": tp, "n": int(selected.sum()),
                "tpr": round(float(tp / selected.sum()), 4),
                "tpr_95pct_exact_ci": exact_interval(tp, int(selected.sum())),
                "auc_against_same_source_negatives": round(float(roc_auc_score(y, s)), 4),
                "mean_score": round(float(test_scores[selected].mean()), 4),
            }
        paired_detection_diff = np.array([
            int(predicted[pair["E"]]) - int(predicted[pair["M"]])
            for pair in pair_indices], dtype=float)
        paired_score_diff = np.array([
            float(test_scores[pair["E"]] - test_scores[pair["M"]])
            for pair in pair_indices], dtype=float)
        sampled_differences = paired_detection_diff[bootstrap_pairs].mean(axis=1)
        sorted_negative_scores = np.sort(negative_scores)

        def percentile_against_same_negatives(score: float) -> float:
            lower = np.searchsorted(sorted_negative_scores, score, side="left")
            upper = np.searchsorted(sorted_negative_scores, score, side="right")
            return float((lower + upper) / (2 * len(sorted_negative_scores)))

        paired_auc_diff = np.array([
            percentile_against_same_negatives(float(test_scores[pair["E"]]))
            - percentile_against_same_negatives(float(test_scores[pair["M"]]))
            for pair in pair_indices], dtype=float)
        sampled_auc_differences = paired_auc_diff[bootstrap_pairs].mean(axis=1)
        cluster_tpr_sums = np.array([paired_detection_diff[indices].sum()
                                     for indices in cluster_indices])
        cluster_auc_sums = np.array([paired_auc_diff[indices].sum()
                                     for indices in cluster_indices])
        sampled_cluster_tpr = (cluster_tpr_sums[bootstrap_clusters].sum(axis=1)
                               / sampled_cluster_sizes)
        sampled_cluster_auc = (cluster_auc_sums[bootstrap_clusters].sum(axis=1)
                               / sampled_cluster_sizes)
        by_signature = {}
        for signature in summary["signatures"]:
            by_signature[signature] = {}
            for method in ("M", "E"):
                selected = np.array([r["signature"] == signature for r in test]) & mask[method]
                by_signature[signature][method] = {
                    "n": int(selected.sum()), "detected": int(predicted[selected].sum()),
                    "tpr": round(float(predicted[selected].mean()), 4),
                }
        output["models"][name] = {
            "threshold": float(cut),
            "validation_fpr": round(float((val_scores[y_val == 0] >= cut).mean()), 4),
            "test_false_positives": fp,
            "test_fpr": round(float(fp / mask["source"].sum()), 4),
            "test_fpr_95pct_exact_ci": exact_interval(fp, int(mask["source"].sum())),
            "positive_method": positive_results,
            "paired_E_minus_M_tpr": round(float(paired_detection_diff.mean()), 4),
            "paired_E_minus_M_tpr_95pct_pair_bootstrap_ci": [round(float(x), 4)
                for x in np.quantile(sampled_differences, [0.025, 0.975])],
            "paired_E_minus_M_auc": round(float(paired_auc_diff.mean()), 4),
            "paired_E_minus_M_auc_95pct_pair_bootstrap_ci": [round(float(x), 4)
                for x in np.quantile(sampled_auc_differences, [0.025, 0.975])],
            "paired_E_minus_M_tpr_95pct_scaffold_cluster_bootstrap_ci": [
                round(float(x), 4) for x in np.quantile(sampled_cluster_tpr, [0.025, 0.975])],
            "paired_E_minus_M_auc_95pct_scaffold_cluster_bootstrap_ci": [
                round(float(x), 4) for x in np.quantile(sampled_cluster_auc, [0.025, 0.975])],
            "mean_paired_E_minus_M_score": round(float(paired_score_diff.mean()), 4),
            "by_signature": by_signature,
        }
        prediction_rows.extend({"model": name, "id": row["id"],
                                "smiles": row["smiles"], "label": row["label"],
                                "method": row["generation_method"],
                                "signature": row["signature"],
                                "comparison_pair_id": row["comparison_pair_id"],
                                "score": float(score), "predicted": int(prediction)}
                               for row, score, prediction in zip(test, test_scores, predicted))
    destination = args.root / f"seed{args.seed}"
    (destination / "evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    with (destination / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "id", "smiles", "label",
                                                   "method", "signature", "comparison_pair_id",
                                                   "score", "predicted"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
