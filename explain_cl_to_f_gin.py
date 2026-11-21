"""Check GIN atom saliency against exactly verified Cl->F edit sites.

Only test counterfeits whose parent is recovered by one unique F->Cl reversal
enter the explanation score. This avoids treating ambiguous MCS masks as exact
ground truth. Saliency is absolute input-gradient times atom features, summed
over features for the counterfeit logit margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem

from repair_halogen_substitution_masks import exact_mask
from train_gnn import GIN, smiles_to_data


def rate(rows: list[dict], key: str) -> float | None:
    return round(sum(row[key] for row in rows) / len(rows), 4) if rows else None


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "top1_site_accuracy": rate(rows, "top1_site_hit"),
        "top2_site_recall": rate(rows, "top2_site_hit"),
        "top1_edit_region_accuracy": rate(rows, "top1_mask_hit"),
        "top2_edit_region_recall": rate(rows, "top2_mask_hit"),
        "mean_reciprocal_site_rank": round(statistics.mean(
            1 / row["site_rank"] for row in rows), 4),
        "mean_reciprocal_edit_region_rank": round(statistics.mean(
            1 / row["best_mask_rank"] for row in rows), 4),
        "random_top1_expectation": round(statistics.mean(
            1 / row["n_atoms"] for row in rows), 4),
        "random_top2_expectation": round(statistics.mean(
            min(2 / row["n_atoms"], 1) for row in rows), 4),
        "random_top1_edit_region_expectation": round(statistics.mean(
            row["n_mask_atoms"] / row["n_atoms"] for row in rows), 4),
        "random_top2_edit_region_expectation": round(statistics.mean(
            1 - ((row["n_atoms"] - row["n_mask_atoms"]) *
                 (row["n_atoms"] - row["n_mask_atoms"] - 1)) /
            (row["n_atoms"] * (row["n_atoms"] - 1)) for row in rows), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("benchmark/models/Cl_to_F_gin_seed42.pt"))
    parser.add_argument("--csv", type=Path,
                        default=Path("benchmark/subcategory_halogen-walk.Cl->F/dataset.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/explanations"))
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint["model"] != "gin" or checkpoint["ring_size_features"]:
        raise ValueError("This analysis expects the standard GIN checkpoint")
    torch.set_num_threads(checkpoint["threads"])
    model = GIN(checkpoint["in_dim"], hidden=checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    with args.csv.open(newline="") as stream:
        positives = [row for row in csv.DictReader(stream)
                     if row["split"] == "test" and row["label"] == "1"]
    records = []
    unresolved = 0
    for row in positives:
        mask = exact_mask(row["parent_smiles"], row["smiles"], "Cl", "F")
        if mask is None:
            unresolved += 1
            continue
        if row["edited_atoms"] != mask:
            raise ValueError(f"Exact mask not applied for {row['id']}")
        molecule = Chem.MolFromSmiles(row["smiles"])
        mask_indices = {int(piece) for piece in mask.split()}
        sites = [index for index in mask_indices
                 if molecule.GetAtomWithIdx(index).GetSymbol() == "F"]
        if len(sites) != 1:
            raise ValueError(f"Cannot identify unique F site for {row['id']}")
        graph = smiles_to_data(row["smiles"], 1)
        graph.x.requires_grad_(True)
        batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        model.zero_grad(set_to_none=True)
        logits = model(graph.x, graph.edge_index, batch)
        margin = logits[0, 1] - logits[0, 0]
        margin.backward()
        importance = (graph.x.grad * graph.x).abs().sum(dim=1)
        order = torch.argsort(importance, descending=True).tolist()
        score = float(F.softmax(logits, dim=-1)[0, 1].detach())
        site = sites[0]
        records.append({
            "id": row["id"], "n_atoms": graph.x.size(0),
            "counterfeit_score": round(score, 8),
            "predicted_counterfeit": int(score > 0.5),
            "edited_site": site, "site_rank": order.index(site) + 1,
            "n_mask_atoms": len(mask_indices),
            "best_mask_rank": min(order.index(index) + 1 for index in mask_indices),
            "top1_atom": order[0], "top2_atoms": " ".join(map(str, order[:2])),
            "top1_site_hit": int(site == order[0]),
            "top2_site_hit": int(site in order[:2]),
            "top1_mask_hit": int(order[0] in mask_indices),
            "top2_mask_hit": int(bool(mask_indices.intersection(order[:2]))),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "Cl_to_F_gin_seed42_test_saliency.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    true_positives = [row for row in records if row["predicted_counterfeit"]]
    report = {
        "dataset": str(args.csv), "checkpoint": str(args.weights),
        "method": "absolute input-gradient times atom features; counterfeit logit margin",
        "ground_truth": "unique exact F->Cl reversal; changed F atom only",
        "total_test_counterfeits": len(positives),
        "ambiguous_or_non_single_swap": unresolved,
        "verified_subset": summarize(records),
        "detected_verified_subset": summarize(true_positives),
        "per_molecule": str(output),
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
