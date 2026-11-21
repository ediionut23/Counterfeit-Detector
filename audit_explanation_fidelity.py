"""Measure feature-masking fidelity of top atom saliency on verified E edits.

Each atom is ablated by zeroing its input features while keeping graph edges.
This probes this GIN's behavior, but zero feature vectors are not valid chemical
atoms and the result is not a chemically causal explanation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train_gnn import GIN, smiles_to_data


def bootstrap_interval(values: list[float], seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(array), size=(2000, len(array)))
    means = array[indices].mean(axis=1)
    return [round(float(value), 6) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("benchmark/models/Evolved_gin_seed42.pt"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--saliency", type=Path,
                        default=Path("benchmark/explanations/evolved_gin_seed42/test_saliency.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/explanations/evolved_gin_seed42"))
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint["model"] != "gin":
        raise ValueError("Expected GIN checkpoint")
    torch.set_num_threads(checkpoint["threads"])
    model = GIN(checkpoint["in_dim"], hidden=checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with args.dataset.open(newline="") as stream:
        molecules = {row["id"]: row for row in csv.DictReader(stream)
                     if row["split"] == "test" and row["label"] == "1"}
    with args.saliency.open(newline="") as stream:
        explanations = list(csv.DictReader(stream))
    records = []
    with torch.no_grad():
        for explanation in explanations:
            row = molecules.get(explanation["id"])
            if row is None:
                raise ValueError(f"Saliency ID absent from positive test: {explanation['id']}")
            graph = smiles_to_data(row["smiles"], 1,
                                   ring_size_features=checkpoint["ring_size_features"])
            if graph is None:
                raise ValueError(f"Invalid molecule: {explanation['id']}")
            n_atoms = graph.x.size(0)
            if n_atoms != int(explanation["n_atoms"]):
                raise ValueError(f"Atom count changed: {explanation['id']}")
            base = model(graph.x, graph.edge_index,
                         torch.zeros(n_atoms, dtype=torch.long))
            base_prob = float(F.softmax(base, dim=-1)[0, 1])
            if abs(base_prob - float(explanation["counterfeit_score"])) > 1e-5:
                raise ValueError(f"Checkpoint score differs: {explanation['id']}")
            variants = []
            for atom_index in range(n_atoms):
                x = graph.x.clone()
                x[atom_index] = 0
                variants.append(Data(x=x, edge_index=graph.edge_index))
            probabilities = []
            for batch in DataLoader(variants, batch_size=64):
                logits = model(batch.x, batch.edge_index, batch.batch)
                probabilities.extend(F.softmax(logits, dim=-1)[:, 1].tolist())
            drops = np.asarray([base_prob - value for value in probabilities])
            impact = np.abs(drops)
            top1 = int(explanation["top1_atom"])
            site = int(explanation["edited_site"])
            if top1 >= n_atoms or site >= n_atoms:
                raise ValueError(f"Selected atom out of range: {explanation['id']}")
            rank = int((impact > impact[top1]).sum() + 1)
            records.append({
                "id": explanation["id"], "method": explanation["method"],
                "n_atoms": n_atoms, "sparsity_top1": round(1 / n_atoms, 8),
                "base_score": round(base_prob, 8),
                "top1_abs_probability_change": round(float(impact[top1]), 8),
                "mean_atom_abs_probability_change": round(float(impact.mean()), 8),
                "site_abs_probability_change": round(float(impact[site]), 8),
                "top1_signed_probability_drop": round(float(drops[top1]), 8),
                "mean_atom_signed_probability_drop": round(float(drops.mean()), 8),
                "top1_decreases_counterfeit_score": int(drops[top1] > 0),
                "top1_ablation_impact_rank": rank,
            })
    if not records:
        raise ValueError("No saliency records")
    args.out.mkdir(parents=True, exist_ok=True)
    per_molecule = args.out / "feature_mask_fidelity.csv"
    with per_molecule.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    difference = [row["top1_abs_probability_change"] -
                  row["mean_atom_abs_probability_change"] for row in records]
    summary = {
        "checkpoint": str(args.weights), "saliency": str(args.saliency),
        "method": "zero one atom's input features, keep molecular graph edges",
        "n_molecules": len(records),
        "mean_top1_abs_probability_change": round(float(np.mean([
            row["top1_abs_probability_change"] for row in records])), 6),
        "mean_random_atom_abs_probability_change": round(float(np.mean([
            row["mean_atom_abs_probability_change"] for row in records])), 6),
        "mean_site_abs_probability_change": round(float(np.mean([
            row["site_abs_probability_change"] for row in records])), 6),
        "top1_minus_random_mean_abs_change": round(float(np.mean(difference)), 6),
        "top1_minus_random_bootstrap_95": bootstrap_interval(difference),
        "top1_decreases_score_fraction": round(float(np.mean([
            row["top1_decreases_counterfeit_score"] for row in records])), 4),
        "mean_top1_ablation_impact_rank": round(float(np.mean([
            row["top1_ablation_impact_rank"] for row in records])), 3),
        "mean_top1_sparsity": round(float(np.mean([
            row["sparsity_top1"] for row in records])), 4),
        "warning": "Zero atom features are out of distribution; this is model feature-masking fidelity, not chemical causality",
        "per_molecule": str(per_molecule),
    }
    (args.out / "fidelity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
