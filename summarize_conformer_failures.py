"""Summarize which benchmark molecules lack a 3D conformer."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    missing = set((args.root / "conformers_missing.smi").read_text().splitlines())
    unique = {}
    per_dataset = {}
    for path in sorted(args.root.rglob("dataset.csv")):
        name = str(path.parent.relative_to(args.root))
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        dataset_smiles = {row["smiles"] for row in rows}
        missing_here = dataset_smiles & missing
        if missing_here:
            per_dataset[name] = {
                "unique_smiles": len(dataset_smiles),
                "missing": len(missing_here),
                "coverage": round(1 - len(missing_here) / len(dataset_smiles), 6),
            }
        for row in rows:
            unique.setdefault(row["smiles"], row)
    if not missing <= unique.keys():
        raise ValueError("Missing list contains molecules absent from datasets")

    def describe(smis):
        entries = [unique[smi] for smi in smis]
        n = len(entries)
        return {
            "count": n,
            "mean_heavy_atoms": (round(sum(int(r["heavy_atoms"]) for r in entries) / n, 2)
                                 if n else None),
            "mean_mw": (round(sum(float(r["mw"]) for r in entries) / n, 2)
                        if n else None),
            "disconnected_smiles": sum("." in r["smiles"] for r in entries),
        }

    report = {
        "unique_molecules": len(unique),
        "missing": describe(missing),
        "embedded": describe(unique.keys() - missing),
        "datasets_with_missing": dict(sorted(per_dataset.items())),
        "missing_categories_from_first_occurrence": dict(sorted(
            Counter(unique[smi]["category"] for smi in missing).items())),
    }
    output = args.out or args.root / "conformers_missing_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "datasets_with_missing"}, indent=2))


if __name__ == "__main__":
    main()
