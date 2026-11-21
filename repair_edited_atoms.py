"""Repair missing or out-of-range counterfeit atom indices from parent/child pairs.

The result is a child-atom mask: new or changed atoms plus child atoms bordering
deleted parent atoms. MCS alignments can be ambiguous in symmetric molecules,
so this repairs structurally invalid metadata, not a claim of unique atom mapping.

Usage: python repair_edited_atoms.py [benchmark_directory] [--apply]
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=32768)
def atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol is not None else -1


def invalid(row: dict) -> bool:
    if str(row["label"]) != "1":
        return False
    value = row["edited_atoms"].strip()
    if not value:
        return True
    try:
        indices = [int(piece) for piece in value.split()]
    except ValueError:
        return True
    if all(0 <= i < int(row["heavy_atoms"]) for i in indices):
        return False
    # Isotopic explicit hydrogens (e.g. [2H]) are atoms in parsed SMILES but
    # are excluded from the heavy-atom descriptor.
    n = atom_count(row["smiles"])
    return any(i < 0 or i >= n for i in indices)


def source_mask_needs_repair(row: dict) -> bool:
    """Detect source masks that used heavy-atom indexing before CSV assembly.

    The original generator sometimes emits indices past the heavy-atom count,
    even when an explicit isotopic hydrogen makes those indices parseable. We
    recompute those source masks once; the resulting child-atom mask may itself
    validly include parsed isotopic hydrogens.
    """
    if invalid(row):
        return True
    if str(row["label"]) != "1":
        return False
    return any(int(piece) >= int(row["heavy_atoms"])
               for piece in row["edited_atoms"].split())


@lru_cache(maxsize=50000)
def edited_child_atoms(parent_smiles: str, child_smiles: str,
                       max_timeout: int = 60) -> list[int]:
    parent = Chem.MolFromSmiles(parent_smiles)
    child = Chem.MolFromSmiles(child_smiles)
    if parent is None or child is None:
        return []
    options = dict(matchValences=False, ringMatchesRingOnly=True,
                   completeRingsOnly=False,
                   bondCompare=rdFMCS.BondCompare.CompareOrderExact)
    result = rdFMCS.FindMCS([parent, child], timeout=min(3, max_timeout), **options)
    if result.canceled:
        if max_timeout <= 3:
            return []
        result = rdFMCS.FindMCS([parent, child], timeout=min(15, max_timeout), **options)
        if result.canceled:
            if max_timeout <= 15:
                return []
            result = rdFMCS.FindMCS([parent, child], timeout=max_timeout, **options)
            if result.canceled:
                return []
    if not result.smartsString:
        return list(range(child.GetNumAtoms()))
    pattern = Chem.MolFromSmarts(result.smartsString)
    if pattern is None:
        return []
    parent_match = parent.GetSubstructMatch(pattern)
    child_match = child.GetSubstructMatch(pattern)
    if len(parent_match) != len(child_match):
        return []

    p_to_c = dict(zip(parent_match, child_match))
    matched_child = set(child_match)
    edited = {a.GetIdx() for a in child.GetAtoms() if a.GetIdx() not in matched_child}
    for p_idx, c_idx in p_to_c.items():
        p_atom, c_atom = parent.GetAtomWithIdx(p_idx), child.GetAtomWithIdx(c_idx)
        if (p_atom.GetFormalCharge(), p_atom.GetIsotope(), p_atom.GetChiralTag(),
            p_atom.GetTotalNumHs(), p_atom.GetIsAromatic()) != (
            c_atom.GetFormalCharge(), c_atom.GetIsotope(), c_atom.GetChiralTag(),
            c_atom.GetTotalNumHs(), c_atom.GetIsAromatic()
        ):
            edited.add(c_idx)
        if any(n.GetIdx() not in p_to_c for n in p_atom.GetNeighbors()):
            edited.add(c_idx)
        if any(n.GetIdx() not in matched_child for n in c_atom.GetNeighbors()):
            edited.add(c_idx)
        for p_neighbor in p_atom.GetNeighbors():
            mapped = p_to_c.get(p_neighbor.GetIdx())
            if mapped is None:
                continue
            old = parent.GetBondBetweenAtoms(p_idx, p_neighbor.GetIdx())
            new = child.GetBondBetweenAtoms(c_idx, mapped)
            if new is None or old.GetBondType() != new.GetBondType():
                edited.update((c_idx, mapped))
    return sorted(edited)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("benchmark"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--source-masks", action="store_true",
                        help="Recompute source masks using the original heavy-atom indexing rule")
    args = parser.parse_args()
    needs_repair = source_mask_needs_repair if args.source_masks else invalid

    files = sorted(args.root.rglob("dataset.csv"))
    pairs = set()
    affected = []
    for path in files:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if needs_repair(row):
                    pairs.add((row["parent_smiles"], row["smiles"]))
                    affected.append(path)
    print(f"Affected rows: {len(affected)}; unique parent/child pairs: {len(pairs)}", flush=True)

    repairs = {}
    for i, pair in enumerate(sorted(pairs), 1):
        indices = edited_child_atoms(*pair)
        if indices:
            repairs[pair] = " ".join(map(str, indices))
        if i % 500 == 0:
            print(f"Resolved {len(repairs)}/{i} pairs", flush=True)
    unresolved = len(pairs) - len(repairs)
    print(f"Resolved {len(repairs)} pairs; unresolved {unresolved}", flush=True)
    if unresolved:
        for parent, child in sorted(pairs - repairs.keys()):
            print(f"UNRESOLVED\t{parent}\t{child}", flush=True)
    if not args.apply:
        return 0 if unresolved == 0 else 1

    changed = 0
    for path in sorted(set(affected)):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = list(reader)
        n = 0
        for row in rows:
            if needs_repair(row):
                replacement = repairs.get((row["parent_smiles"], row["smiles"]))
                if replacement:
                    row["edited_atoms"] = replacement
                    n += 1
        if not n:
            continue
        with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent, delete=False) as temp:
            writer = csv.DictWriter(temp, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            temp_path = Path(temp.name)
        os.replace(temp_path, path)
        changed += n
    print(f"Updated {changed} CSV rows", flush=True)
    return 0 if unresolved == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
