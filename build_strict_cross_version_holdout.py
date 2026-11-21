"""Filter the existing M→E holdout by actual row-SMILES scaffold novelty."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from build_strict_child_scaffold_holdout import own_scaffold


def load(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("benchmark/cross_version/M_four_categories.csv"))
    parser.add_argument("--candidate", type=Path,
                        default=Path("benchmark/cross_version/evolved_unseen_from_M.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/cross_version/evolved_unseen_from_M_own_scaffold.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, source = load(args.source)
    fields, candidates = load(args.candidate)
    exposed = {own_scaffold(row["smiles"]) for row in source
               if row["split"] in ("train", "val")}
    eligible = [row for row in candidates
                if own_scaffold(row["smiles"]) not in exposed]
    before = Counter(row["label"] for row in candidates)
    after_filter = Counter(row["label"] for row in eligible)
    count = min(after_filter.values())
    rng = random.Random(args.seed)
    chosen = set()
    for label in ("0", "1"):
        ids = [row["id"] for row in eligible if row["label"] == label]
        chosen.update(rng.sample(ids, count))
    output = [row for row in eligible if row["id"] in chosen]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "source": str(args.source), "candidate": str(args.candidate),
        "out": str(args.out), "seed": args.seed,
        "source_train_val_own_scaffolds": len(exposed),
        "candidate_labels": dict(before),
        "labels_after_own_scaffold_filter": dict(after_filter),
        "removed_for_own_scaffold_overlap": {
            label: before[label] - after_filter[label] for label in ("0", "1")},
        "balanced_labels": dict(Counter(row["label"] for row in output)),
        "rows": len(output),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
