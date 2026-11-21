"""Independent chemical and provenance audit for generated GA category CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import evolutionary_generator as EG

RDLogger.DisableLog("rdApp.*")
CATEGORIES = ["bioisostere", "halogen-walk", "scaffold-hop", "homologation", "other"]


def canonical_reference(path: Path) -> set[str]:
    result = set()
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mol = Chem.MolFromSmiles(line.split()[0])
            if mol is not None:
                result.add(Chem.MolToSmiles(mol))
    return result


def canonical_csv(path: Path) -> set[str]:
    result = set()
    if not path.exists():
        return result
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            value = row.get("smiles") or row.get("counterfeit_smiles") or ""
            mol = Chem.MolFromSmiles(value)
            if mol is not None:
                result.add(Chem.MolToSmiles(mol))
    return result


def canonical_csv(path: Path) -> set[str]:
    result = set()
    if not path.exists():
        return result
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            value = row.get("smiles") or row.get("counterfeit_smiles") or ""
            mol = Chem.MolFromSmiles(value)
            if mol is not None:
                result.add(Chem.MolToSmiles(mol))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("version_ga"))
    parser.add_argument("--parents", type=Path, default=Path("public_molecules_100k.smi"))
    parser.add_argument("--authentic-extra", type=Path, default=Path("public_molecules.smi"))
    parser.add_argument("--existing-bank", type=Path, default=Path("final_dataset/molecules.csv"))
    parser.add_argument("--sim-lo", type=float, default=0.55)
    parser.add_argument("--sim-hi", type=float, default=0.87)
    parser.add_argument("--qed-min", type=float, default=0.45)
    parser.add_argument("--max-sa", type=float, default=4.20)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    args = parser.parse_args()
    authentic = canonical_reference(args.parents) | canonical_reference(args.authentic_extra)
    existing_bank = canonical_csv(args.existing_bank)
    global_children = {}
    report = {"filters": {
        "similarity_band": [args.sim_lo, args.sim_hi], "qed_min": args.qed_min,
        "sa_max": args.max_sa, "structural_alerts": 0,
        "parent_must_be_in_authentic_input": True,
        "child_must_not_be_in_authentic_input": True,
        "cross_category_child_duplicates": 0,
        "existing_bank_file": str(args.existing_bank),
    }, "categories": {}}

    for category in args.categories:
        path = args.root / f"counterfeits_evolved_{category}.csv"
        if not path.exists():
            report["categories"][category] = {"file_missing": True}
            continue
        rows = list(csv.DictReader(path.open(newline="")))
        counters = Counter()
        children, parents, child_scaffolds, parent_scaffolds = set(), set(), set(), set()
        sims, qeds, sas = [], [], []
        for row in rows:
            child_smi = row.get("counterfeit_smiles", "")
            parent_smi = row.get("parent_smiles", "")
            child = Chem.MolFromSmiles(child_smi)
            parent = Chem.MolFromSmiles(parent_smi)
            if child is None:
                counters["invalid_child_smiles"] += 1
                continue
            child_smi = Chem.MolToSmiles(child)
            if child_smi in children:
                counters["duplicate_child_within_category"] += 1
            children.add(child_smi)
            parents.add(parent_smi)
            if parent is None:
                counters["invalid_parent_smiles"] += 1
                continue
            if parent_smi not in authentic:
                counters["parent_not_in_authentic_library"] += 1
            if child_smi in authentic:
                counters["child_in_authentic_library"] += 1
            if child_smi in existing_bank:
                counters["child_in_existing_counterfeit_bank"] += 1
            if child_smi in existing_bank:
                counters["child_in_existing_counterfeit_bank"] += 1
            if row.get("category") != f"evolved_{category}":
                counters["category_label_mismatch"] += 1
            if row.get("generation_protocol") != "category_conditioned_genetic_search":
                counters["generation_protocol_mismatch"] += 1

            sim = DataStructs.TanimotoSimilarity(
                rdMolDescriptors.GetMorganFingerprintAsBitVect(parent, 3, 2048),
                rdMolDescriptors.GetMorganFingerprintAsBitVect(child, 3, 2048),
            )
            qed = float(Descriptors.qed(child))
            sa = float(EG.sascorer.calculateScore(child)) if EG.sascorer else None
            sims.append(sim)
            qeds.append(qed)
            if sa is not None:
                sas.append(sa)
            if not args.sim_lo <= sim <= args.sim_hi:
                counters["outside_similarity_band"] += 1
            if qed < args.qed_min:
                counters["below_qed_min"] += 1
            if sa is not None and sa > args.max_sa:
                counters["above_sa_max"] += 1
            if EG.structural_alerts(child) > 0:
                counters["structural_alert_hit"] += 1
            if child_smi == parent_smi:
                counters["unchanged_from_parent"] += 1
            if len(Chem.GetMolFrags(child)) != 1:
                counters["multicomponent_child"] += 1
            if not 10 <= child.GetNumHeavyAtoms() <= 70:
                counters["heavy_atoms_out_of_range"] += 1
            if not 150.0 <= Descriptors.MolWt(child) <= 800.0:
                counters["molecular_weight_out_of_range"] += 1

            try:
                edits = [int(x) for x in row.get("edited_atoms", "").split()]
                if any(x < 0 or x >= child.GetNumAtoms() for x in edits):
                    counters["edit_index_out_of_range"] += 1
                if len(edits) != int(row.get("n_edited_atoms", 0)):
                    counters["edit_mask_count_mismatch"] += 1
            except ValueError:
                counters["edit_mask_parse_error"] += 1

            child_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=child))
            parent_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=parent))
            if child_smi in global_children:
                if category not in global_children[child_smi]:
                    counters["cross_category_duplicate"] += 1
                    global_children[child_smi].append(category)
            else:
                global_children[child_smi] = [category]

        reasons = sum(v for k, v in counters.items() if k not in {"cross_category_duplicate"})
        report["categories"][category] = {
            "file": path.name, "rows": len(rows), "unique_children": len(children),
            "unique_parents": len(parents), "unique_parent_scaffolds": len(parent_scaffolds),
            "unique_child_scaffolds": len(child_scaffolds), "median_similarity": round(float(np.median(sims)), 4) if sims else None,
            "median_qed": round(float(np.median(qeds)), 4) if qeds else None,
            "median_sa": round(float(np.median(sas)), 4) if sas else None,
            "issues": dict(counters), "chemical_and_provenance_pass": reasons == 0,
        }

    report["n_cross_category_duplicate_children"] = sum(
        len(cats) > 1 for cats in global_children.values())
    report["n_unique_children_all_categories"] = len(global_children)
    (args.root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
