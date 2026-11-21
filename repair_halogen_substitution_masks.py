"""Replace ambiguous MCS masks when an exact single halogen swap is provable.

For a row marked X->Y, substitute each Y atom in the child back to X. Only
when exactly one candidate reconstructs the canonical parent do we know the
edited child atom unambiguously. The mask contains that atom and its bonded
neighbors, matching the benchmark's edit-plus-boundary convention. Other rows
are left as they are. --apply also updates the versioned correction manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger

from edit_mask_corrections import FIELDS, read, write_table

RDLogger.DisableLog("rdApp.*")

HALOGENS = {"F": 9, "Cl": 17, "Br": 35, "I": 53}
PATTERN = re.compile(r"^(F|Cl|Br|I)->(F|Cl|Br|I)$")


@lru_cache(maxsize=50000)
def exact_mask(parent_smiles: str, child_smiles: str,
               source: str, target: str) -> str | None:
    parent = Chem.MolFromSmiles(parent_smiles)
    child = Chem.MolFromSmiles(child_smiles)
    if parent is None or child is None:
        return None
    parent_canonical = Chem.MolToSmiles(parent, isomericSmiles=True)
    candidates = []
    for atom in child.GetAtoms():
        if atom.GetSymbol() != target:
            continue
        index = atom.GetIdx()
        trial = Chem.RWMol(child)
        trial.GetAtomWithIdx(index).SetAtomicNum(HALOGENS[source])
        molecule = trial.GetMol()
        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            continue
        if Chem.MolToSmiles(molecule, isomericSmiles=True) == parent_canonical:
            candidates.append(index)
    if len(candidates) != 1:
        return None
    site = child.GetAtomWithIdx(candidates[0])
    mask = {site.GetIdx(), *(neighbor.GetIdx() for neighbor in site.GetNeighbors())}
    return " ".join(map(str, sorted(mask)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("benchmark/edit_mask_corrections.csv"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    _, existing_rows = read(args.manifest)
    manifest = {(row["parent_smiles"], row["smiles"]): row for row in existing_rows}
    files = sorted(args.root.rglob("dataset.csv"))
    updates: dict[Path, tuple[list[str], list[dict]]] = {}
    new_values: dict[tuple[str, str], str] = {}
    source_values: dict[tuple[str, str], set[str]] = {}
    counts = Counter()
    for path in files:
        fields, rows = read(path)
        changed = False
        for row in rows:
            if row["label"] != "1" or row["category"] != "halogen-walk":
                continue
            match = PATTERN.fullmatch(row["subcategory"])
            if not match:
                continue
            counts["eligible_rows"] += 1
            key = (row["parent_smiles"], row["smiles"])
            desired = exact_mask(*key, *match.groups())
            if desired is None:
                counts["unresolved_rows"] += 1
                continue
            counts["exact_rows"] += 1
            if key in new_values and new_values[key] != desired:
                raise ValueError(f"Conflicting exact masks for {key}")
            new_values[key] = desired
            source_values.setdefault(key, set()).add(row["edited_atoms"])
            if row["edited_atoms"] != desired:
                counts["changed_rows"] += 1
                changed = True
                row["edited_atoms"] = desired
        if changed:
            updates[path] = (fields, rows)

    for key, desired in new_values.items():
        entry = manifest.get(key)
        if entry is not None:
            entry["corrected_edited_atoms"] = desired
        elif source_values[key] != {desired}:
            if len(source_values[key]) != 1:
                raise ValueError(f"Inconsistent source masks for {key}")
            manifest[key] = dict(zip(FIELDS, (*key, next(iter(source_values[key])), desired)))
            counts["new_manifest_pairs"] += 1

    # The same parent/child pair may also occur in an evolved-category table
    # with a different source mask. Its structural edit is still the same.
    for path in files:
        fields, rows = updates[path] if path in updates else read(path)
        changed = path in updates
        for row in rows:
            key = (row["parent_smiles"], row["smiles"])
            desired = new_values.get(key)
            if desired is not None and row["edited_atoms"] != desired:
                if key not in manifest:
                    manifest[key] = dict(zip(
                        FIELDS, (*key, row["edited_atoms"], desired)))
                    counts["new_manifest_pairs"] += 1
                row["edited_atoms"] = desired
                counts["propagated_rows"] += 1
                changed = True
        if changed:
            updates[path] = (fields, rows)

    report = dict(counts)
    report["exact_parent_child_pairs"] = len(new_values)
    report["changed_files"] = len(updates)
    print(json.dumps(report, indent=2))
    if not args.apply:
        return
    for path, (fields, rows) in updates.items():
        write_table(path, fields, rows)
    manifest_rows = [manifest[key] for key in sorted(manifest)]
    write_table(args.manifest, FIELDS, manifest_rows)
    print(f"Updated {len(updates)} datasets and {len(manifest_rows)} manifest entries")


if __name__ == "__main__":
    main()
