"""Filter evolved test rows to scaffolds absent from source train/validation.

This is a supplementary holdout for existing trained models. The benchmark's
original parent-scaffold split and its published rows are left intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

from assemble_36_benchmark import murcko


@lru_cache(maxsize=150000)
def own_scaffold(smiles: str) -> str:
    return murcko(smiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/strict_child_scaffold"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    exposed = {own_scaffold(row["smiles"]) for row in rows
               if row["split"] in ("train", "val")}
    test = [row for row in rows if row["split"] == "test"]
    eligible = [row for row in test if own_scaffold(row["smiles"]) not in exposed]
    before = Counter(row["label"] for row in test)
    after_filter = Counter(row["label"] for row in eligible)
    n = min(after_filter.values())
    rng = random.Random(args.seed)
    chosen = set()
    for label in ("0", "1"):
        candidates = [row["id"] for row in eligible if row["label"] == label]
        chosen.update(rng.sample(candidates, n))
    output = [row for row in eligible if row["id"] in chosen]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "evolved_test.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    report = {
        "source": str(args.source), "seed": args.seed,
        "exposed_train_val_own_scaffolds": len(exposed),
        "original_test_labels": dict(before),
        "test_labels_after_scaffold_filter": dict(after_filter),
        "removed_for_own_scaffold_overlap": {
            label: before[label] - after_filter[label] for label in ("0", "1")},
        "balanced_test_labels": dict(Counter(row["label"] for row in output)),
        "rows": len(output), "path": str(path),
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
