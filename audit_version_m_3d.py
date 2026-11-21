"""Audit Version M 3D tensors and list molecules missing from the graph banks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from audit_version_ga_3d import audit_one
from run_version_ga import CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("version_m"))
    args = parser.parse_args()
    results = {category: audit_one(args.root, category) for category in CATEGORIES}

    missing = []
    requested_total = 0
    embedded_total = 0
    for category in CATEGORIES:
        source = args.root / f"counterfeits_{category}.csv"
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        requested_total += len(rows)
        graph_path = args.root / "3d" / category / "graphs_3d.npz"
        with np.load(graph_path, allow_pickle=False) as graph:
            ids = set(graph["molecule_id"].tolist())
            embedded_total += len(ids)
        for row in rows:
            mol = Chem.MolFromSmiles(row["counterfeit_smiles"])
            if mol is None:
                mol_id = ""
            else:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                mol_id = hashlib.sha256(canonical.encode()).hexdigest()[:20]
            if mol_id not in ids:
                missing.append({"category": category, "molecule_id": mol_id,
                                "smiles": row["counterfeit_smiles"],
                                "parent_smiles": row.get("parent_smiles", ""),
                                "reason": "not present in graph archive; embedding or force-field setup failed"})

    failed_path = args.root / "3d" / "failed_embeddings.csv"
    with failed_path.open("w", newline="") as stream:
        fields = ["category", "molecule_id", "smiles", "parent_smiles", "reason"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(missing)
    report = {
        "categories": results,
        "requested": requested_total, "embedded": embedded_total,
        "failed": len(missing),
        "coverage": embedded_total / requested_total if requested_total else 0.0,
        "failed_embeddings_file": str(failed_path),
        "all_present_and_passed": all(r.get("passed", False) for r in results.values()),
    }
    serializable = lambda value: value.item() if isinstance(value, np.generic) else str(value)
    (args.root / "3d" / "audit_3d.json").write_text(
        json.dumps(report, indent=2, default=serializable) + "\n")
    print(json.dumps(report, indent=2, default=serializable))


if __name__ == "__main__":
    main()
