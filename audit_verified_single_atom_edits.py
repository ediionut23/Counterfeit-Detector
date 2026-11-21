"""Audit the optional exact single-atom edit manifest against the benchmark.

The full audit checks every manifest mask and all dataset memberships. A fixed
sample also repeats the reverse chemical edit, including uniqueness of its site.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def reverse_matches(parent: str, child: Chem.Mol, method: str) -> list[int]:
    parent_mol = Chem.MolFromSmiles(parent)
    if parent_mol is None:
        return []
    target = Chem.MolToSmiles(parent_mol, isomericSmiles=True)
    parent_counts = Counter(atom.GetSymbol() for atom in parent_mol.GetAtoms())
    child_counts = Counter(atom.GetSymbol() for atom in child.GetAtoms())
    lost = list((parent_counts - child_counts).elements())
    gained = list((child_counts - parent_counts).elements())
    if method == "substitution":
        if parent_mol.GetNumAtoms() != child.GetNumAtoms() or len(lost) != 1 or len(gained) != 1:
            return []
        old_number = Chem.GetPeriodicTable().GetAtomicNumber(lost[0])
        candidates = [atom.GetIdx() for atom in child.GetAtoms() if atom.GetSymbol() == gained[0]]
    elif method == "addition":
        if child.GetNumAtoms() != parent_mol.GetNumAtoms() + 1 or lost or len(gained) != 1:
            return []
        candidates = [atom.GetIdx() for atom in child.GetAtoms() if atom.GetSymbol() == gained[0]]
    else:
        return []
    matches = []
    for site in candidates:
        trial = Chem.RWMol(child)
        if method == "substitution":
            trial.GetAtomWithIdx(site).SetAtomicNum(old_number)
        else:
            trial.RemoveAtom(site)
        molecule = trial.GetMol()
        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            continue
        if Chem.MolToSmiles(molecule, isomericSmiles=True) == target:
            matches.append(site)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()
    directory = args.root / "verified_edits"
    manifest = directory / "manifest.csv"
    summary = json.loads((directory / "summary.json").read_text())
    records = {}
    issues = []
    with manifest.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["parent_smiles"], row["smiles"])
            if key in records:
                issues.append(f"duplicate pair: {key}")
            records[key] = row
            child = Chem.MolFromSmiles(key[1])
            if child is None:
                issues.append(f"invalid child: {key}")
                continue
            try:
                site = int(row["site_atom"])
                mask = [int(index) for index in row["verified_mask"].split()]
            except ValueError:
                issues.append(f"invalid site or mask: {key}")
                continue
            if site < 0 or site >= child.GetNumAtoms():
                issues.append(f"site out of range: {key}")
                continue
            atom = child.GetAtomWithIdx(site)
            expected = sorted({site, *(neighbor.GetIdx() for neighbor in atom.GetNeighbors())})
            if mask != expected or atom.GetDegree() == 0:
                issues.append(f"incorrect edit region: {key}")

    sample = sorted(records, key=lambda key: hashlib.sha256("\t".join(key).encode()).digest())[:args.sample]
    for key in sample:
        row = records[key]
        child = Chem.MolFromSmiles(key[1])
        if child is None or reverse_matches(key[0], child, row["method"]) != [int(row["site_atom"])]:
            issues.append(f"nonunique or failed reversal: {key}")

    counts = defaultdict(Counter)
    seen = set()
    for path in sorted(args.root.rglob("dataset.csv")):
        name = str(path.parent.relative_to(args.root))
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["label"] != "1":
                    continue
                counts[name]["counterfeits"] += 1
                key = (row["parent_smiles"], row["smiles"])
                if key not in records:
                    continue
                seen.add(key)
                record = records[key]
                counts[name]["verified_rows"] += 1
                counts[name][record["method"]] += 1
                if row["edited_atoms"] == record["verified_mask"]:
                    counts[name]["current_mask_agrees"] += 1

    if seen != records.keys():
        issues.append(f"manifest membership mismatch: {len(set(records)-seen)} absent pairs")
    if len(records) != summary["unique_verified_pairs"]:
        issues.append("manifest count differs from summary")
    actual = {name: dict(counter) for name, counter in sorted(counts.items())}
    if actual != summary["datasets"]:
        issues.append("per-dataset counts differ from summary")
    report = {
        "manifest_pairs": len(records),
        "chemical_reversals_sampled": len(sample),
        "verified_table_rows": sum(counter["verified_rows"] for counter in counts.values()),
        "issues": issues[:30],
        "n_issues": len(issues),
    }
    (directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
