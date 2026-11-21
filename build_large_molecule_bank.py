"""Combine the audited generated bank with the expanded public reference pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
FIELDS = ["molecule_id", "smiles", "label", "molecule_source", "generation_method",
          "generator_variant", "category", "subcategory", "parent_smiles",
          "source_id", "mw", "logp", "qed", "sa_score", "heavy_atoms",
          "smiles_length", "child_scaffold"]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(raw: str) -> str:
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        raise ValueError(f"Invalid SMILES in source bank: {raw}")
    return Chem.MolToSmiles(mol, canonical=True)


def molecule_id(smiles: str) -> str:
    return "mol_" + hashlib.sha256(smiles.encode()).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset_400k"))
    parser.add_argument("--ga-root", type=Path, default=Path("version_ga"))
    parser.add_argument("--m-root", type=Path, default=Path("version_m"))
    parser.add_argument("--controls", type=Path, default=Path("molecule_bank_500k/chembl_controls.csv"))
    parser.add_argument("--out", type=Path, default=Path("molecule_bank_500k"))
    parser.add_argument("--minimum", type=int, default=500000)
    args = parser.parse_args()

    rows_by_smiles: dict[str, dict] = {}
    source_counts = Counter()
    authentic_rows = read(args.bank / "authentic.csv")
    authentic_smiles = {canonical(row["smiles"]) for row in authentic_rows}
    for row in read(args.bank / "molecules.csv"):
        smiles = canonical(row["smiles"])
        method = row.get("methods", row.get("primary_method", ""))
        molecule = {
            "molecule_id": molecule_id(smiles), "smiles": smiles, "label": "1",
            "molecule_source": "final_dataset_400k", "generation_method": method,
            "generator_variant": f"Version {method}" if method else "mixed provenance",
            "category": row.get("primary_category", ""),
            "subcategory": row.get("primary_subcategory", ""), "parent_smiles": "",
            "source_id": row.get("molecule_id", ""), "mw": row.get("mw", ""),
            "logp": row.get("logp", ""), "qed": row.get("qed", ""),
            "sa_score": row.get("sa_score", ""), "heavy_atoms": row.get("heavy_atoms", ""),
            "smiles_length": row.get("smiles_length", ""),
            "child_scaffold": row.get("child_scaffold", ""),
        }
        rows_by_smiles.setdefault(smiles, molecule)
        source_counts["final_dataset_400k_generated"] += 1
    aggregate_generated = set(rows_by_smiles)

    ga_added = 0
    for row in read(args.ga_root / "counterfeits_all.csv"):
        smiles = canonical(row["counterfeit_smiles"])
        if smiles in rows_by_smiles or smiles in authentic_smiles:
            continue
        rows_by_smiles[smiles] = {
            "molecule_id": molecule_id(smiles), "smiles": smiles, "label": "1",
            "molecule_source": "version_ga", "generation_method": "E",
            "generator_variant": "Version E category-conditioned GA",
            "category": row.get("category", ""), "subcategory": row.get("subcategory", row.get("category", "")),
            "parent_smiles": row.get("parent_smiles", ""), "source_id": row.get("molecule_id", ""),
            "mw": row.get("mw", ""), "logp": row.get("logp", ""),
            "qed": row.get("qed", ""), "sa_score": row.get("sa_score", ""),
            "heavy_atoms": row.get("heavy_atoms", ""), "smiles_length": row.get("smiles_length", ""),
            "child_scaffold": row.get("child_scaffold", ""),
        }
        ga_added += 1
    source_counts["version_ga_unique_added"] = ga_added

    m_added = 0
    for row in read(args.m_root / "counterfeits_all.csv"):
        smiles = canonical(row["counterfeit_smiles"])
        if smiles in rows_by_smiles or smiles in authentic_smiles:
            continue
        rows_by_smiles[smiles] = {
            "molecule_id": molecule_id(smiles), "smiles": smiles, "label": "1",
            "molecule_source": "version_m", "generation_method": "M",
            "generator_variant": "Version M rule-based",
            "category": row.get("category", ""), "subcategory": row.get("rule_name", ""),
            "parent_smiles": row.get("parent_smiles", ""),
            "source_id": row.get("rule_name", ""), "mw": row.get("mw", ""),
            "logp": row.get("logp", ""), "qed": row.get("qed", ""),
            "sa_score": "", "heavy_atoms": row.get("heavy_atoms", ""),
            "smiles_length": len(smiles), "child_scaffold": "",
        }
        m_added += 1
    source_counts["version_m_unique_added"] = m_added

    for row in authentic_rows:
        smiles = canonical(row["smiles"])
        if smiles in rows_by_smiles:
            raise ValueError(f"Authentic reference overlaps a generated molecule: {smiles}")
        rows_by_smiles[smiles] = {
            "molecule_id": molecule_id(smiles), "smiles": smiles, "label": "0",
            "molecule_source": "project_public_library", "generation_method": "",
            "generator_variant": "", "category": "public_reference",
            "subcategory": "project_library", "parent_smiles": "", "source_id": row["molecule_id"],
            "mw": "", "logp": "", "qed": "", "sa_score": "", "heavy_atoms": "",
            "smiles_length": len(smiles), "child_scaffold": "",
        }
        source_counts["project_reference"] += 1

    for row in read(args.controls):
        smiles = canonical(row["smiles"])
        if smiles in rows_by_smiles:
            raise ValueError(f"ChEMBL reference overlaps another bank molecule: {smiles}")
        rows_by_smiles[smiles] = {
            "molecule_id": molecule_id(smiles), "smiles": smiles, "label": "0",
            "molecule_source": "ChEMBL", "generation_method": "", "generator_variant": "",
            "category": "public_reference", "subcategory": "ChEMBL", "parent_smiles": "",
            "source_id": row.get("source_id", ""), "mw": row.get("mw", ""),
            "logp": row.get("logp", ""), "qed": row.get("qed", ""),
            "sa_score": row.get("sa_score", ""), "heavy_atoms": row.get("heavy_atoms", ""),
            "smiles_length": row.get("smiles_length", ""),
            "child_scaffold": row.get("child_scaffold", ""),
        }
        source_counts["ChEMBL_reference"] += 1

    rows = sorted(rows_by_smiles.values(), key=lambda row: row["molecule_id"])
    if len(rows) < args.minimum:
        raise ValueError(f"Bank has only {len(rows):,} unique molecules; minimum is {args.minimum:,}")
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "molecules.csv"
    with out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "unique_molecules": len(rows), "minimum_requested": args.minimum,
        "generated_analogue_molecules": sum(r["label"] == "1" for r in rows),
        "public_reference_molecules": sum(r["label"] == "0" for r in rows),
        "source_rows_processed": dict(source_counts),
        "ga_exact_overlaps_with_aggregate_generated_bank": 50000 - ga_added,
        "labels": {"1": "generated analogue, not confirmed counterfeit",
                   "0": "public reference compound, not necessarily an approved medicine"},
        "csv": str(out_csv),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
