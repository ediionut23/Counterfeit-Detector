"""Find parent/child edits provable by one exact atom substitution or addition.

This creates an optional high-confidence explanation subset without changing
the main benchmark CSVs. A candidate is accepted only if exactly one child
atom can be changed back or removed to reconstruct the canonical parent.
The returned mask contains the changed/new child atom and its bond neighbors.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def atom_counts(mol: Chem.Mol) -> Counter:
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def mask(child: Chem.Mol, site: int) -> str:
    atom = child.GetAtomWithIdx(site)
    indices = {site, *(neighbor.GetIdx() for neighbor in atom.GetNeighbors())}
    return " ".join(map(str, sorted(indices)))


@lru_cache(maxsize=100000)
def verify(parent_smiles: str, child_smiles: str) -> tuple[str, int, str] | None:
    parent = Chem.MolFromSmiles(parent_smiles)
    child = Chem.MolFromSmiles(child_smiles)
    if parent is None or child is None:
        return None
    target = Chem.MolToSmiles(parent, isomericSmiles=True)
    parent_counts, child_counts = atom_counts(parent), atom_counts(child)
    parent_only = list((parent_counts - child_counts).elements())
    child_only = list((child_counts - parent_counts).elements())

    if parent.GetNumAtoms() == child.GetNumAtoms() and \
       len(parent_only) == len(child_only) == 1:
        source, replacement = parent_only[0], child_only[0]
        source_atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(source)
        sites = []
        for atom in child.GetAtoms():
            if atom.GetSymbol() != replacement:
                continue
            index = atom.GetIdx()
            trial = Chem.RWMol(child)
            trial.GetAtomWithIdx(index).SetAtomicNum(source_atomic_number)
            molecule = trial.GetMol()
            try:
                Chem.SanitizeMol(molecule)
            except Exception:
                continue
            if Chem.MolToSmiles(molecule, isomericSmiles=True) == target:
                sites.append(index)
        if len(sites) == 1 and child.GetAtomWithIdx(sites[0]).GetDegree() > 0:
            return "substitution", sites[0], mask(child, sites[0])

    if child.GetNumAtoms() == parent.GetNumAtoms() + 1 and \
       not parent_only and len(child_only) == 1:
        added_element = child_only[0]
        sites = []
        for atom in child.GetAtoms():
            if atom.GetSymbol() != added_element:
                continue
            index = atom.GetIdx()
            trial = Chem.RWMol(child)
            trial.RemoveAtom(index)
            molecule = trial.GetMol()
            try:
                Chem.SanitizeMol(molecule)
            except Exception:
                continue
            if Chem.MolToSmiles(molecule, isomericSmiles=True) == target:
                sites.append(index)
        if len(sites) == 1 and child.GetAtomWithIdx(sites[0]).GetDegree() > 0:
            return "addition", sites[0], mask(child, sites[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--dataset", help="Process only one dataset directory")
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmark/verified_edits"))
    args = parser.parse_args()
    files = ([args.root / args.dataset / "dataset.csv"] if args.dataset else
             sorted(args.root.rglob("dataset.csv")))
    pairs = {}
    by_dataset = defaultdict(Counter)
    total_pairs_seen = set()
    for path in files:
        name = str(path.parent.relative_to(args.root))
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["label"] != "1":
                    continue
                by_dataset[name]["counterfeits"] += 1
                key = (row["parent_smiles"], row["smiles"])
                if args.limit_pairs is not None and key not in total_pairs_seen and \
                   len(total_pairs_seen) >= args.limit_pairs:
                    continue
                total_pairs_seen.add(key)
                result = verify(*key)
                if result is None:
                    continue
                method, site, verified_mask = result
                pairs[key] = result
                by_dataset[name]["verified_rows"] += 1
                by_dataset[name][method] += 1
                if row["edited_atoms"] == verified_mask:
                    by_dataset[name]["current_mask_agrees"] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("parent_smiles", "smiles", "method", "site_atom", "verified_mask"))
        for (parent, child), (method, site, verified_mask) in sorted(pairs.items()):
            writer.writerow((parent, child, method, site, verified_mask))
    report = {
        "method": "Unique exact reverse single-atom substitution or addition; mask includes bonded boundary",
        "partial": args.limit_pairs is not None or args.dataset is not None,
        "unique_pairs_examined": len(total_pairs_seen),
        "unique_verified_pairs": len(pairs),
        "datasets": {name: dict(counts) for name, counts in sorted(by_dataset.items())},
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "datasets"},
                     indent=2))
    for name in ("subcategory_homologation.grow-single",
                 "subcategory_bioisostere.F->O", "subcategory_halogen-walk.add-F"):
        if name in report["datasets"]:
            print(name, report["datasets"][name])


if __name__ == "__main__":
    main()
