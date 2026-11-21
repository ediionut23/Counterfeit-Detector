"""Build a test slice with scaffolds and molecules unseen in source datasets.

Example:
  python cross_version_holdout.py --train category_bioisostere \
      category_halogen-walk category_homologation category_scaffold-hop \
      --test evolved --out benchmark/cross_version/evolved_unseen_from_M.csv

Rows come from the target dataset's fixed test split. Source exposure includes
both train and val because models may use val for selection or refitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--train", nargs="+", required=True,
                        help="Source dataset names under --root")
    parser.add_argument("--test", required=True, help="Target dataset name")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--balance", action="store_true",
                        help="Downsample the retained test rows to equal class counts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seen_scaffolds = set()
    seen_smiles = set()
    for name in args.train:
        _, rows = read_rows(args.root / name / "dataset.csv")
        for row in rows:
            if row["split"] in ("train", "val"):
                seen_scaffolds.add(row["scaffold"])
                seen_smiles.add(row["smiles"])

    fieldnames, target_rows = read_rows(args.root / args.test / "dataset.csv")
    test_rows = [r for r in target_rows if r["split"] == "test"]
    unseen = [r for r in test_rows
              if r["scaffold"] not in seen_scaffolds and r["smiles"] not in seen_smiles]
    before_balance = Counter(r["label"] for r in unseen)
    if args.balance:
        rng = random.Random(args.seed)
        n = min(before_balance["0"], before_balance["1"])
        chosen = set()
        for label in ("0", "1"):
            candidates = [i for i, r in enumerate(unseen) if r["label"] == label]
            chosen.update(rng.sample(candidates, n))
        unseen = [r for i, r in enumerate(unseen) if i in chosen]

    report = {
        "source_datasets": args.train,
        "source_splits": ["train", "val"],
        "target_dataset": args.test,
        "target_split": "test",
        "balanced": args.balance,
        "seed": args.seed if args.balance else None,
        "source_scaffolds": len(seen_scaffolds),
        "source_smiles": len(seen_smiles),
        "target_test_rows": len(test_rows),
        "unseen_before_balance": dict(before_balance),
        "retained_rows": len(unseen),
        "retained_labels": dict(Counter(r["label"] for r in unseen)),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(unseen)
        args.out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
