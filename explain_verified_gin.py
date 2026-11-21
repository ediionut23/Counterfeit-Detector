"""Score GIN atom saliency on exactly reversible one-atom test edits.

Ground truth comes from benchmark/verified_edits/manifest.csv. The result is
localization of a chemical edit, not a perturbation-based fidelity estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from explain_cl_to_f_gin import summarize
from train_gnn import GIN, smiles_to_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("benchmark/models/Evolved_gin_seed42.pt"))
    parser.add_argument("--csv", type=Path,
                        default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("benchmark/verified_edits/manifest.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/explanations/evolved_gin_seed42"))
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint["model"] != "gin":
        raise ValueError("Expected a GIN checkpoint")
    torch.set_num_threads(checkpoint["threads"])
    model = GIN(checkpoint["in_dim"], hidden=checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with args.manifest.open(newline="") as stream:
        verified = {(row["parent_smiles"], row["smiles"]): row
                    for row in csv.DictReader(stream)}
    with args.csv.open(newline="") as stream:
        positives = [row for row in csv.DictReader(stream)
                     if row["split"] == "test" and row["label"] == "1"]
    records = []
    for row in positives:
        edit = verified.get((row["parent_smiles"], row["smiles"]))
        if edit is None:
            continue
        if row["edited_atoms"] != edit["verified_mask"]:
            raise ValueError(f"Verified mask not applied for {row['id']}")
        graph = smiles_to_data(row["smiles"], 1,
                               ring_size_features=checkpoint["ring_size_features"])
        if graph is None:
            raise ValueError(f"Invalid molecule in {row['id']}")
        graph.x.requires_grad_(True)
        batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        model.zero_grad(set_to_none=True)
        logits = model(graph.x, graph.edge_index, batch)
        (logits[0, 1] - logits[0, 0]).backward()
        importance = (graph.x.grad * graph.x).abs().sum(dim=1)
        order = torch.argsort(importance, descending=True).tolist()
        score = float(F.softmax(logits, dim=-1)[0, 1].detach())
        site = int(edit["site_atom"])
        mask_indices = {int(piece) for piece in edit["verified_mask"].split()}
        if site >= graph.x.size(0) or max(mask_indices) >= graph.x.size(0):
            raise ValueError(f"Manifest site or region out of range for {row['id']}")
        records.append({
            "id": row["id"], "method": edit["method"],
            "n_atoms": graph.x.size(0), "counterfeit_score": round(score, 8),
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
    if not records:
        raise ValueError("No verified test counterfeits in dataset")
    args.out.mkdir(parents=True, exist_ok=True)
    per_molecule = args.out / "test_saliency.csv"
    with per_molecule.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    report = {
        "dataset": str(args.csv), "checkpoint": str(args.weights),
        "manifest": str(args.manifest),
        "method": "absolute input-gradient times atom features; counterfeit logit margin",
        "ground_truth": "unique exact reverse single-atom substitution or addition",
        "total_test_counterfeits": len(positives),
        "unverified_test_counterfeits": len(positives) - len(records),
        "verified_subset": summarize(records),
        "detected_verified_subset": summarize(
            [row for row in records if row["predicted_counterfeit"]]),
        "by_edit_type": {method: summarize([row for row in records
                                            if row["method"] == method])
                         for method in ("substitution", "addition")},
        "per_molecule": str(per_molecule),
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
