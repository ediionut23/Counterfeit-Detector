"""Refresh tracked per-dataset SDF samples from the central conformer library.

Each existing ``conformers.sdf`` is rebuilt for the first 150 embeddable rows
of its current CSV. The central library must contain tagged ``smiles`` records.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from rdkit import Chem


def normalize_property_headers(path: Path) -> None:
    """Remove RDKit's trailing space on SD property header lines."""
    with path.open("rb") as stream:
        lines = stream.readlines()
    with path.open("wb") as stream:
        for line in lines:
            stream.write(line.rstrip(b" \t\r\n") + b"\n"
                         if line.startswith(b">  <") else line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--missing", type=Path, default=None,
                        help="SMILES without conformers; defaults to root/conformers_missing.smi")
    args = parser.parse_args()

    missing_path = args.missing or args.root / "conformers_missing.smi"
    missing = set(missing_path.read_text().splitlines()) if missing_path.exists() else set()

    targets = sorted(args.root.rglob("conformers.sdf"))
    wanted = defaultdict(list)
    expected = {}
    for path in targets:
        with (path.parent / "dataset.csv").open(newline="") as stream:
            rows = []
            for row in csv.DictReader(stream):
                if row["smiles"] in missing:
                    continue
                rows.append(row)
                if len(rows) == args.limit:
                    break
        expected[path] = len(rows)
        for row in rows:
            wanted[row["smiles"]].append((path, row))

    temp_paths = {path: path.with_name("conformers.sdf.tmp") for path in targets}
    writers = {path: Chem.SDWriter(str(temp_paths[path])) for path in targets}
    counts = defaultdict(int)
    try:
        for mol in Chem.SDMolSupplier(str(args.library), removeHs=False):
            if mol is None or not mol.HasProp("smiles"):
                raise ValueError("Central SDF contains an invalid or untagged record")
            smiles = mol.GetProp("smiles")
            for path, row in wanted.pop(smiles, []):
                mol.SetProp("_Name", row["id"])
                mol.SetProp("label", row["label"])
                mol.SetProp("split", row["split"])
                writers[path].write(mol)
                counts[path] += 1
    finally:
        for writer in writers.values():
            writer.close()

    if wanted:
        for temp in temp_paths.values():
            temp.unlink(missing_ok=True)
        raise ValueError(f"Central library lacks {len(wanted)} sampled SMILES")
    for path in targets:
        normalize_property_headers(temp_paths[path])
        os.replace(temp_paths[path], path)
        print(f"{path}: {counts[path]}/{expected[path]} records")


if __name__ == "__main__":
    main()
