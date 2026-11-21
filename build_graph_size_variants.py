"""Create fixed-count evolved training subsets with opposite counterfeit sizes.

The variants hold the authentic class, counterfeit count, validation, and test
constant. They change which training counterfeits are sampled: smallest or
largest by heavy-atom count. Chemistry may co-vary with size, so this is a
selection stress test rather than an isolated causal effect of graph size.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path


def size_stats(rows: list[dict]) -> dict:
    values = [int(row["heavy_atoms"]) for row in rows]
    return {"min": min(values), "median": statistics.median(values),
            "mean": round(statistics.mean(values), 3), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/graph_size"))
    parser.add_argument("--authentic-per-counterfeit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.authentic_per_counterfeit < 2:
        parser.error("Expected at least 2 authentic molecules per counterfeit")
    with args.source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    authentic = [row for row in rows if row["split"] == "train" and row["label"] == "0"]
    counterfeits = [row for row in rows if row["split"] == "train" and row["label"] == "1"]
    target = round(len(authentic) / args.authentic_per_counterfeit)
    if target < 1 or target > len(counterfeits):
        raise ValueError("Requested class ratio is unavailable")

    shuffled = counterfeits.copy()
    random.Random(args.seed).shuffle(shuffled)
    by_size = sorted(shuffled, key=lambda row: int(row["heavy_atoms"]))
    selections = {
        "small": by_size[:target],
        "large": by_size[-target:],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.source), "seed": args.seed,
        "authentic_per_counterfeit": args.authentic_per_counterfeit,
        "train_authentic": len(authentic), "train_counterfeit": target,
        "source_counterfeit_size": size_stats(counterfeits),
        "variants": {},
    }
    for name, selected_rows in selections.items():
        selected = {row["id"] for row in selected_rows}
        variant = [row for row in rows if row["split"] != "train" or
                   row["label"] == "0" or row["id"] in selected]
        path = args.out / f"evolved_train_size_{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(variant)
        counts = Counter((row["split"], row["label"]) for row in variant)
        report["variants"][name] = {
            "path": str(path), "rows": len(variant),
            "counterfeit_size": size_stats(selected_rows),
            "split_label_counts": {split: {label: counts[split, label] for label in ("0", "1")}
                                   for split in ("train", "val", "test")},
        }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
