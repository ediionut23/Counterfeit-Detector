"""Apply a saved train_gnn.py state_dict to benchmark molecules.

Example:
  python predict_gnn.py --weights benchmark/models/Cl_to_F_gin_seed42.pt \
      --csv 'benchmark/subcategory_halogen-walk.Cl->F/dataset.csv' \
      --out benchmark/models/Cl_to_F_gin_seed42_test_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from train_gnn import GIN, MessagePassingClassifier, metrics, smiles_to_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("layers") != 3:
        raise ValueError("Unsupported checkpoint format")
    torch.set_num_threads(checkpoint["threads"])
    architecture = checkpoint["model"]
    in_dim = checkpoint["in_dim"]
    hidden = checkpoint["hidden"]
    model = (GIN(in_dim, hidden=hidden) if architecture == "gin" else
             MessagePassingClassifier(architecture, in_dim, hidden=hidden))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    with args.csv.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if args.split == "all" or row["split"] == args.split]
    if not rows:
        raise ValueError("No rows in requested split")
    graphs = []
    for row in rows:
        graph = smiles_to_data(row["smiles"], int(row.get("label", 0)),
                               ring_size_features=checkpoint["ring_size_features"])
        if graph is None or graph.x.size(1) != in_dim:
            raise ValueError(f"Cannot featurize {row['id']} for this checkpoint")
        graphs.append(graph)

    scores = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=128):
            probabilities = F.softmax(model(batch.x, batch.edge_index, batch.batch), dim=-1)
            scores.extend(probabilities[:, 1].tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=("id", "split", "label", "counterfeit_score", "predicted_label"),
                                lineterminator="\n")
        writer.writeheader()
        for row, score in zip(rows, scores):
            writer.writerow({"id": row["id"], "split": row["split"],
                             "label": row.get("label", ""),
                             "counterfeit_score": round(score, 8),
                             "predicted_label": int(score > 0.5)})
    result = {"checkpoint": str(args.weights), "input": str(args.csv),
              "split": args.split, "rows": len(rows), "out": str(args.out)}
    if all(row.get("label") in ("0", "1") for row in rows):
        result["metrics"] = metrics([int(row["label"]) for row in rows],
                                    [int(score > 0.5) for score in scores], scores)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
