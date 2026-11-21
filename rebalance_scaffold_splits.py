"""Reassign whole scaffold families while balancing both labels per split.

Usage: python rebalance_scaffold_splits.py [benchmark_directory] [--apply]
Without --apply, prints the proposed counts and leaves files unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from assemble_36_benchmark import scaffold_split


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent, delete=False) as temp:
        writer = csv.DictWriter(temp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("benchmark"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    index_path = args.root / "index.json"
    index = json.loads(index_path.read_text())
    fractions = []
    for name in sorted(index):
        path = args.root / name / "dataset.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = list(reader)
        scaffold_split(rows)
        counts = Counter((r["split"],r["label"]) for r in rows)
        split_counts = {s: counts[s,"0"] + counts[s,"1"] for s in ("train","val","test")}
        split_label_counts = {s: {"0": counts[s,"0"], "1": counts[s,"1"]}
                              for s in ("train","val","test")}
        test_total = split_counts["test"]
        fraction = counts["test","1"] / test_total if test_total else 0
        fractions.append((abs(fraction - 0.5),name,fraction,test_total))
        if args.apply:
            write_csv(path, rows, fieldnames)
            summary_path = path.parent / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["split_counts"] = split_counts
            summary["split_label_counts"] = split_label_counts
            summary_path.write_text(json.dumps(summary, indent=2))
            index[name] = summary

    if args.apply:
        index_path.write_text(json.dumps(index, indent=2))
    print(f"Datasets: {len(index)}; apply={args.apply}")
    print("Worst test-class fractions:")
    for _,name,fraction,n in sorted(fractions,reverse=True)[:10]:
        print(f"  {name}: {fraction:.3f} counterfeit ({n} test rows)")


if __name__ == "__main__":
    main()
