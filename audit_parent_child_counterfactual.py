"""Score evolved parent/child pairs with a saved GIN checkpoint.

The pairwise score change uses real, valid molecules. It checks sensitivity to
the chemical edit but does not alone establish why the model made its decision.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from train_gnn import GIN, smiles_to_data


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "child_score_mean": round(float(np.mean([r["child_score"] for r in records])), 4),
        "parent_score_mean": round(float(np.mean([r["parent_score"] for r in records])), 4),
        "mean_child_minus_parent": round(float(np.mean([r["score_change"] for r in records])), 4),
        "child_scores_higher_fraction": round(float(np.mean([
            r["score_change"] > 0 for r in records])), 4),
        "pair_correct_at_half_fraction": round(float(np.mean([
            r["child_score"] > 0.5 and r["parent_score"] <= 0.5
            for r in records])), 4),
        "parent_scores_authentic_fraction": round(float(np.mean([
            r["parent_score"] <= 0.5 for r in records])), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("benchmark/models/Evolved_gin_seed42.pt"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("benchmark/verified_edits/manifest.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/explanations/evolved_gin_seed42"))
    parser.add_argument("--all-pairs", action="store_true",
                        help="Include all positive test pairs, marking unverified edits")
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint["model"] != "gin":
        raise ValueError("Expected GIN checkpoint")
    torch.set_num_threads(checkpoint["threads"])
    model = GIN(checkpoint["in_dim"], hidden=checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with args.manifest.open(newline="") as stream:
        verified = {(row["parent_smiles"], row["smiles"]): row
                    for row in csv.DictReader(stream)}
    with args.dataset.open(newline="") as stream:
        dataset_rows = list(csv.DictReader(stream))
    exposed_parent_labels = {}
    for row in dataset_rows:
        if row["split"] in ("train", "val"):
            exposed_parent_labels.setdefault(row["smiles"], set()).add(row["label"])
    positives = [row for row in dataset_rows
                 if row["split"] == "test" and row["label"] == "1"]
    rows = (positives if args.all_pairs else [row for row in positives
            if (row["parent_smiles"], row["smiles"]) in verified])
    if not rows:
        raise ValueError("No verified test counterfeits in dataset")
    graphs = []
    for row in rows:
        edit = verified.get((row["parent_smiles"], row["smiles"]))
        if edit is not None and row["edited_atoms"] != edit["verified_mask"]:
            raise ValueError(f"Verified mask not applied for {row['id']}")
        for smiles, label in ((row["parent_smiles"], 0), (row["smiles"], 1)):
            graph = smiles_to_data(smiles, label,
                                   ring_size_features=checkpoint["ring_size_features"])
            if graph is None:
                raise ValueError(f"Cannot featurize pair for {row['id']}")
            graphs.append(graph)
    scores = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=128):
            scores.extend(F.softmax(model(batch.x, batch.edge_index, batch.batch),
                                    dim=-1)[:, 1].tolist())
    records = []
    for i, row in enumerate(rows):
        parent, child = scores[2 * i:2 * i + 2]
        edit = verified.get((row["parent_smiles"], row["smiles"]))
        records.append({"id": row["id"],
                        "method": edit["method"] if edit else "unverified",
                        "parent_seen_in_train_val": int(
                            row["parent_smiles"] in exposed_parent_labels),
                        "parent_seen_as_counterfeit": int(
                            "1" in exposed_parent_labels.get(row["parent_smiles"], set())),
                        "parent_score": round(parent, 8),
                        "child_score": round(child, 8),
                        "score_change": round(child - parent, 8),
                        "child_predicted_counterfeit": int(child > 0.5),
                        "parent_predicted_authentic": int(parent <= 0.5)})
    args.out.mkdir(parents=True, exist_ok=True)
    per_pair = args.out / "parent_child_counterfactual.csv"
    with per_pair.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "checkpoint": str(args.weights),
        "method": "score valid parent and child molecules for each generated pair",
        "total_test_counterfeits": len(positives),
        "all_pairs": args.all_pairs,
        "scored_pairs": summarize(records),
        "parents_seen_as_train_val_counterfeits": sum(
            record["parent_seen_as_counterfeit"] for record in records),
        "parents_seen_as_train_val_authentic": sum(
            "0" in exposed_parent_labels.get(row["parent_smiles"], set())
            for row in rows),
        "unseen_parent_subset": summarize([
            record for record in records if not record["parent_seen_in_train_val"]]),
        "by_edit_type": {method: summarize([record for record in records
                                            if record["method"] == method])
                         for method in ("substitution", "addition", "unverified")},
        "interpretation_limit": "parent molecules are not necessarily authentic training rows; pairwise score change does not identify which internal features caused it",
        "per_pair": str(per_pair),
    }
    (args.out / "parent_child_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
