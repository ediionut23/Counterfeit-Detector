"""Assemble per-category GA CSVs into a labelled table and relation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
CATEGORIES = ["bioisostere", "halogen-walk", "scaffold-hop", "homologation", "other"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("version_ga"))
    args = parser.parse_args()
    all_rows, molecules, relations = [], {}, []
    by_category = {}
    seen = set()
    for category in CATEGORIES:
        path = args.root / f"counterfeits_evolved_{category}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_category[category] = len(rows)
        for row in rows:
            mol = Chem.MolFromSmiles(row["counterfeit_smiles"])
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol)
            if smiles in seen:
                raise ValueError(f"Duplicate child across category tables: {smiles}")
            seen.add(smiles)
            molecule_id = hashlib.sha256(smiles.encode()).hexdigest()[:20]
            row["counterfeit_smiles"] = smiles
            row["molecule_id"] = molecule_id
            row["label"] = 1
            all_rows.append(row)
            molecules[molecule_id] = {
                "molecule_id": molecule_id, "smiles": smiles, "label": 1,
                "category": row["category"], "subcategory": row["subcategory"],
                "qed": row["qed"], "sa_score": row["sa_score"], "mw": row["mw"],
                "logp": row["logp"], "heavy_atoms": row["heavy_atoms"],
                "child_scaffold": row["child_scaffold"],
            }
            relations.append({
                "molecule_id": molecule_id, "smiles": smiles,
                "parent_smiles": row["parent_smiles"], "label": 1,
                "method": "G", "category": row["category"],
                "subcategory": row["subcategory"],
                "generation_protocol": row["generation_protocol"],
                "similarity_to_parent": row["similarity_to_parent"],
                "edited_atoms": row["edited_atoms"],
                "edit_status": "mcs_approximate",
                "parent_scaffold": row["parent_scaffold"],
                "child_scaffold": row["child_scaffold"],
            })

    if not all_rows:
        raise ValueError(f"No category CSV rows found in {args.root}")
    columns = list(all_rows[0])
    columns += [col for col in ("molecule_id", "label") if col not in columns]
    with (args.root / "counterfeits_all.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    for name, rows in (("molecules.csv", list(molecules.values())),
                       ("relations.csv", relations)):
        with (args.root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "schema_version": 1,
        "n_unique_counterfeits": len(molecules),
        "category_counts": by_category,
        "label_semantics": "Generated analogue, label=1; not a confirmed counterfeit medicine.",
        "files": ["counterfeits_all.csv", "molecules.csv", "relations.csv"],
    }
    (args.root / "summary_dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
