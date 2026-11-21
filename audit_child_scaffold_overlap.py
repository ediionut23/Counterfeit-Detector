"""Measure actual molecule-scaffold overlap across benchmark splits.

The benchmark's stored scaffold groups counterfeit rows by parent scaffold.
This audit separately computes the Bemis–Murcko scaffold of each row's own
SMILES and counts held-out rows whose scaffold occurs in train or validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

from assemble_36_benchmark import murcko


@lru_cache(maxsize=300000)
def own_scaffold(smiles: str) -> str:
    return murcko(smiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/child_scaffold_overlap_audit.json"))
    args = parser.parse_args()
    reports = {}
    for path in sorted(args.root.rglob("dataset.csv")):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        exposed = {own_scaffold(row["smiles"]) for row in rows
                   if row["split"] in ("train", "val")}
        test = [row for row in rows if row["split"] == "test"]
        overlap = {label: sum(own_scaffold(row["smiles"]) in exposed
                              for row in test if row["label"] == label)
                   for label in ("0", "1")}
        counts = {label: sum(row["label"] == label for row in test)
                  for label in ("0", "1")}
        parent_scaffolds_train_val = {row["scaffold"] for row in rows
                                      if row["split"] in ("train", "val")}
        reports[str(path.parent.relative_to(args.root))] = {
            "test_label_counts": counts,
            "test_own_scaffold_overlap_with_train_val": overlap,
            "parent_scaffold_overlap_test_rows": sum(
                row["scaffold"] in parent_scaffolds_train_val for row in test),
            "test_own_scaffold_overlap_fraction": round(
                sum(overlap.values()) / len(test), 4) if test else None,
        }
    output = {
        "meaning": "Actual row-SMILES scaffold seen in train/val; stored scaffold remains parent-based for counterfeits",
        "datasets": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: reports[name] for name in
                      ("evolved", "pooled", "category_scaffold-hop",
                       "subcategory_halogen-walk.Cl->F")}, indent=2))


if __name__ == "__main__":
    main()
