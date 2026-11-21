"""Downsample evolved training positives while preserving fixed val/test rows.

The variants isolate the effect of class scarcity during training. Validation
and test stay at their official balanced, scaffold-disjoint sizes; deployment
prevalence is analyzed separately by prevalence_analysis.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/imbalance"))
    parser.add_argument("--ratios", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(ratio <= 1 for ratio in args.ratios):
        parser.error("All ratios must exceed 1")

    with args.source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    auth_train = [row for row in rows if row["split"] == "train" and row["label"] == "0"]
    fake_train = [row for row in rows if row["split"] == "train" and row["label"] == "1"]
    random.Random(args.seed).shuffle(fake_train)
    if not auth_train or not fake_train:
        raise ValueError("Source needs both classes in its training split")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.source), "seed": args.seed, "variants": {}}
    for ratio in sorted(set(args.ratios)):
        target = min(len(fake_train), max(1, round(len(auth_train) / ratio)))
        selected = {row["id"] for row in fake_train[:target]}
        variant = [row for row in rows if (row["split"] != "train"
                   or row["label"] == "0" or row["id"] in selected)]
        path = args.out / f"evolved_train_1_to_{ratio}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(variant)
        counts = Counter((row["split"], row["label"]) for row in variant)
        report["variants"][str(ratio)] = {
            "path": str(path),
            "rows": len(variant),
            "actual_train_authentic_to_counterfeit": round(len(auth_train) / target, 4),
            "split_label_counts": {
                split: {label: counts[split, label] for label in ("0", "1")}
                for split in ("train", "val", "test")
            },
        }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
