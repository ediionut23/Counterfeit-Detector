"""Summarize size and chemical quality of each selectable final-bank module."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FIELDS = (
    "module", "method", "category", "subcategory", "unique_molecules",
    "unique_child_scaffolds", "median_qed", "median_sa", "median_similarity",
    "exact_edit_fraction", "mcs_approximate_fraction",
    "source_approximate_fraction", "missing_edit_fraction",
)


def read(path: Path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    args = parser.parse_args()
    root = args.bank
    index = json.loads((root / "subcategories/index.json").read_text())
    molecules = {row["smiles"]: row for row in read(root / "molecules.csv")}
    results = []
    for module, metadata in sorted(index.items()):
        relations = list(read(root / metadata["path"]))
        method, category, subcategory = module.split("/", 2)
        n = len(relations)
        if n != metadata["unique_molecules"]:
            raise ValueError(f"Count mismatch for {module}")
        results.append({
            "module": module, "method": method, "category": category,
            "subcategory": subcategory, "unique_molecules": n,
            "unique_child_scaffolds": len({r["child_scaffold"] for r in relations}),
            "median_qed": round(statistics.median(float(molecules[r["smiles"]]["qed"])
                                                  for r in relations), 4),
            "median_sa": round(statistics.median(float(molecules[r["smiles"]]["sa_score"])
                                                 for r in relations), 3),
            "median_similarity": round(statistics.median(float(r["similarity_to_parent"])
                                                         for r in relations), 4),
            "exact_edit_fraction": round(sum(r["edit_status"] == "exact_single_atom"
                                             for r in relations) / n, 4),
            "mcs_approximate_fraction": round(sum(r["edit_status"] == "mcs_approximate"
                                                   for r in relations) / n, 4),
            "source_approximate_fraction": round(sum(r["edit_status"] == "source_approximate"
                                                      for r in relations) / n, 4),
            "missing_edit_fraction": round(sum(r["edit_status"] == "missing"
                                               for r in relations) / n, 4),
        })
    output = root / "subcategories/quality.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print(f"{len(results)} modules summarized in {output}")


if __name__ == "__main__":
    main()
