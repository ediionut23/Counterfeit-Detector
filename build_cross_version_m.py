"""Build one balanced, globally scaffold-disjoint four-category M source table.

Uses the train/val rows of the four category datasets. Identical SMILES are
deduplicated; molecules assigned conflicting labels are excluded. The larger
class is downsampled before a fresh 80/10/10 scaffold split. The E holdout
created by cross_version_holdout.py excludes all source train/val molecules and
scaffolds, including rows removed here, so it stays strictly unseen.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from assemble_36_benchmark import scaffold_split


DEFAULT_CATEGORIES = (
    "category_bioisostere", "category_halogen-walk",
    "category_homologation", "category_scaffold-hop",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/cross_version/M_four_categories.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_smiles: dict[str, list[dict]] = defaultdict(list)
    source_rows = 0
    fieldnames = None
    for category in args.categories:
        with (args.root / category / "dataset.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["split"] in ("train", "val"):
                    by_smiles[row["smiles"]].append(row)
                    source_rows += 1

    conflicting = {smiles for smiles, entries in by_smiles.items()
                   if len({row["label"] for row in entries}) > 1}
    unique = [entries[0] for smiles, entries in by_smiles.items()
              if smiles not in conflicting]
    labels = {label: [row for row in unique if row["label"] == label]
              for label in ("0", "1")}
    n = min(map(len, labels.values()))
    rng = random.Random(args.seed)
    rows = []
    for label in ("0", "1"):
        rows.extend(rng.sample(labels[label], n))
    scaffold_split(rows, seed=args.seed)
    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["label"]) for row in rows)
    report = {
        "source_datasets": args.categories,
        "source_splits": ["train", "val"],
        "source_rows": source_rows,
        "source_unique_smiles": len(by_smiles),
        "conflicting_smiles_removed": len(conflicting),
        "deduplicated_label_counts": {label: len(labels[label]) for label in labels},
        "seed": args.seed,
        "rows": len(rows),
        "split_label_counts": {
            split: {label: counts[split, label] for label in ("0", "1")}
            for split in ("train", "val", "test")
        },
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
