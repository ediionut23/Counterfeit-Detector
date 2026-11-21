"""Independently check identity, provenance, and modular slices in final_dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

from audit_verified_single_atom_edits import reverse_matches
from generate_100k_counterfeits import has_structural_alerts, sascorer


def read(path: Path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("final_dataset"))
    parser.add_argument("--min-counterfeits", type=int, default=200000)
    parser.add_argument("--sample-exact", type=int, default=300)
    parser.add_argument("--sample-quality", type=int, default=500,
                        help="Deterministic independent chemical QC sample")
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / "summary.json").read_text())
    slices = json.loads((root / "subcategories/index.json").read_text())
    errors = Counter()
    auth_rows = list(read(root / "authentic.csv"))
    auth = {r["smiles"] for r in auth_rows}
    if len(auth) != len(auth_rows):
        errors["duplicate_authentic"] += len(auth_rows) - len(auth)
    eligible_rows = list(read(root / "authentic_eligible.csv"))
    eligible = {r["smiles"] for r in eligible_rows}
    if len(eligible) != len(eligible_rows) or not eligible <= auth:
        errors["eligible_authentic_membership"] += 1
    if len(eligible) != summary["n_authentic_eligible_for_training"]:
        errors["eligible_authentic_summary_count"] += 1
    molecules = {}
    ids = set()
    for row in read(root / "molecules.csv"):
        smiles = row["smiles"]
        if smiles in molecules:
            errors["duplicate_counterfeit"] += 1
        if row["molecule_id"] in ids:
            errors["duplicate_molecule_id"] += 1
        if smiles in auth:
            errors["counterfeit_in_authentic"] += 1
        if row["label"] != "1":
            errors["wrong_label"] += 1
        molecules[smiles] = row
        ids.add(row["molecule_id"])
    if len(molecules) != summary["n_unique_counterfeits"]:
        errors["summary_molecule_count"] += 1
    if len(molecules) < args.min_counterfeits:
        errors["below_target"] += 1
    relation_ids = set()
    relation_lookup = {}
    relation_by_child = Counter()
    slice_membership = defaultdict(set)
    exact = mcs_approximate = source_approximate = approximate = missing = 0
    exact_relations = []
    for row in read(root / "relations.csv"):
        key = row["relation_id"]
        if key in relation_ids:
            errors["duplicate_relation_id"] += 1
        relation_ids.add(key)
        relation_lookup[key] = row
        child = row["smiles"]
        if child not in molecules or row["molecule_id"] != molecules[child]["molecule_id"]:
            errors["missing_or_wrong_child"] += 1
        if row["parent_smiles"] not in auth:
            errors["missing_parent"] += 1
        if not 0.4 <= float(row["similarity_to_parent"]) <= 0.98:
            errors["similarity_band"] += 1
        if row["method"] not in ("M", "E"):
            errors["unknown_method"] += 1
        if row["child_scaffold"] != molecules[child]["child_scaffold"]:
            errors["child_scaffold"] += 1
        relation_by_child[child] += 1
        slice_membership[f"{row['method']}/{row['category']}/{row['subcategory']}"] .add(child)
        status = row["edit_status"]
        if status == "exact_single_atom":
            exact += 1
            exact_relations.append(row)
        elif status == "mcs_approximate":
            mcs_approximate += 1
        elif status == "source_approximate":
            source_approximate += 1
        elif status == "approximate":
            approximate += 1
        elif status == "missing":
            missing += 1
        else:
            errors["unknown_edit_status"] += 1
        if status != "missing":
            mol = Chem.MolFromSmiles(child)
            indices = [int(x) for x in row["edited_atoms"].split()]
            if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= mol.GetNumAtoms():
                errors["invalid_edit_mask"] += 1
    if len(relation_ids) != summary["n_distinct_relations"]:
        errors["summary_relation_count"] += 1
    sample = sorted(exact_relations,
                    key=lambda row: hashlib.sha256(row["relation_id"].encode()).digest())[:args.sample_exact]
    for row in sample:
        child = Chem.MolFromSmiles(row["smiles"])
        mask = [int(x) for x in row["edited_atoms"].split()]
        valid = False
        for method in ("substitution", "addition"):
            sites = reverse_matches(row["parent_smiles"], child, method)
            if len(sites) != 1:
                continue
            site = sites[0]
            atom = child.GetAtomWithIdx(site)
            expected = sorted({site, *(neighbor.GetIdx() for neighbor in atom.GetNeighbors())})
            if mask == expected and atom.GetDegree() > 0:
                valid = True
                break
        if not valid:
            errors["exact_edit_independent_reversal"] += 1
    for smiles, row in molecules.items():
        if relation_by_child[smiles] != int(row["n_relations"]):
            errors["relation_count_per_child"] += 1
    quality_sample = sorted(molecules,
                            key=lambda smiles: hashlib.sha256(smiles.encode()).digest())[:args.sample_quality]
    for smiles in quality_sample:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or Chem.MolToSmiles(mol) != smiles or len(Chem.GetMolFrags(mol)) != 1:
            errors["sample_invalid_or_noncanonical_child"] += 1
            continue
        if not (10 <= mol.GetNumHeavyAtoms() <= 70 and 150 <= Descriptors.MolWt(mol) <= 800):
            errors["sample_child_size_or_mass"] += 1
        if Descriptors.qed(mol) < 0.35 or sascorer.calculateScore(mol) > 3.8 or has_structural_alerts(mol):
            errors["sample_child_quality"] += 1
    relation_sample = sorted(relation_lookup.values(),
                             key=lambda row: hashlib.sha256(row["relation_id"].encode()).digest())[:args.sample_quality]
    for row in relation_sample:
        parent = Chem.MolFromSmiles(row["parent_smiles"])
        child = Chem.MolFromSmiles(row["smiles"])
        if parent is None or child is None:
            errors["sample_invalid_relation_molecule"] += 1
            continue
        if Chem.MolToSmiles(parent) != row["parent_smiles"] or len(Chem.GetMolFrags(parent)) != 1:
            errors["sample_invalid_parent"] += 1
        fp_parent = rdMolDescriptors.GetMorganFingerprintAsBitVect(parent, 3, 2048)
        fp_child = rdMolDescriptors.GetMorganFingerprintAsBitVect(child, 3, 2048)
        actual = DataStructs.TanimotoSimilarity(fp_parent, fp_child)
        if not 0.4 <= actual <= 0.98 or abs(actual - float(row["similarity_to_parent"])) > 0.000051:
            errors["sample_similarity_mismatch"] += 1
    if set(slice_membership) != set(slices):
        errors["slice_keys"] += 1
    for name, meta in slices.items():
        path = root / meta["path"]
        if not path.exists():
            errors["missing_slice_file"] += 1
            continue
        rows = list(read(path))
        children = {r["smiles"] for r in rows}
        if len(rows) != len(children) or children != slice_membership[name] or len(rows) != meta["unique_molecules"]:
            errors["slice_membership_or_count"] += 1
        for row in rows:
            if row["relation_id"] not in relation_lookup or row != relation_lookup[row["relation_id"]]:
                errors["slice_relation_mismatch"] += 1
    report = {
        "unique_counterfeits": len(molecules), "authentic_reference": len(auth),
        "authentic_eligible_for_training": len(eligible),
        "relations": len(relation_ids), "subcategories": len(slices),
        "edit_status_relations": {"exact_single_atom": exact,
                                  "mcs_approximate": mcs_approximate,
                                  "source_approximate": source_approximate,
                                  "approximate": approximate, "missing": missing},
        "independent_exact_edit_reversals_sampled": len(sample),
        "independent_quality_molecules_sampled": len(quality_sample),
        "independent_similarity_relations_sampled": len(relation_sample),
        "errors": dict(errors), "passed": not errors,
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
