"""Audit the supplementary M→E test with unseen row-SMILES scaffolds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_strict_child_scaffold_holdout import own_scaffold


def load(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/cross_version"))
    args = parser.parse_args()
    path = args.root / "evolved_unseen_from_M_own_scaffold.csv"
    meta = json.loads(path.with_suffix(".summary.json").read_text())
    source = load(Path(meta["source"]))
    candidates = {row["id"]: row for row in load(Path(meta["candidate"]))}
    target = load(path)
    exposed = {own_scaffold(row["smiles"]) for row in source
               if row["split"] in ("train", "val")}
    issues = []
    if len({row["id"] for row in target}) != len(target):
        issues.append("duplicate target IDs")
    if any(candidates.get(row["id"]) != row for row in target):
        issues.append("target row differs from candidate E holdout")
    if any(own_scaffold(row["smiles"]) in exposed for row in target):
        issues.append("target molecule scaffold seen in M train/val")
    counts = Counter(row["label"] for row in target)
    if counts["0"] != counts["1"] or dict(counts) != meta["balanced_labels"]:
        issues.append("target label counts are not balanced or differ from summary")
    if len(target) != meta["rows"]:
        issues.append("target row count differs from summary")
    report = {"rows": len(target), "label_counts": dict(counts),
              "source_train_val_own_scaffolds": len(exposed), "issues": issues}
    (args.root / "evolved_unseen_from_M_own_scaffold.audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
