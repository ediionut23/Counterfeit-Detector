"""Validate the central tagged SDF against all benchmark CSV molecules.

Usage: python audit_conformers.py --root benchmark --library PATH
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--missing-out", type=Path,
                        help="Write SMILES without a conformer, one per line")
    args = parser.parse_args()

    wanted = set()
    for path in args.root.rglob("dataset.csv"):
        with path.open(newline="") as stream:
            wanted.update(row["smiles"] for row in csv.DictReader(stream))

    seen = set()
    invalid = duplicate = extra = no_3d = 0
    for mol in Chem.SDMolSupplier(str(args.library), removeHs=False):
        if mol is None or not mol.HasProp("smiles"):
            invalid += 1
            continue
        smiles = mol.GetProp("smiles")
        if smiles in seen:
            duplicate += 1
        seen.add(smiles)
        if smiles not in wanted:
            extra += 1
        if not mol.GetNumConformers() or not mol.GetConformer().Is3D():
            no_3d += 1

    missing = wanted - seen
    if args.missing_out is not None:
        args.missing_out.write_text("".join(s + "\n" for s in sorted(missing)))
    coverage = len(seen & wanted) / len(wanted) if wanted else 0
    report = {
        "dataset_unique_smiles": len(wanted),
        "sdf_unique_smiles": len(seen),
        "missing": len(missing),
        "coverage": round(coverage, 6),
        "invalid_records": invalid,
        "duplicate_smiles": duplicate,
        "extra_smiles": extra,
        "records_without_3d": no_3d,
    }
    print(json.dumps(report, indent=2))
    return int(invalid > 0 or duplicate > 0 or extra > 0 or no_3d > 0
               or coverage < args.min_coverage)


if __name__ == "__main__":
    raise SystemExit(main())
