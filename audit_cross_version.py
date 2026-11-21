"""Check the derived four-category M source and strict E holdout for leakage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    args = parser.parse_args()
    root = args.root
    source_path = root / "cross_version/M_four_categories.csv"
    target_path = root / "cross_version/evolved_unseen_from_M.csv"
    source = read(source_path)
    target = read(target_path)
    source_meta = json.loads(source_path.with_suffix(".summary.json").read_text())
    target_meta = json.loads(target_path.with_suffix(".summary.json").read_text())

    issues = []
    source_smiles = defaultdict(set)
    source_scaffolds = defaultdict(set)
    for row in source:
        source_smiles[row["smiles"]].add((row["split"], row["label"]))
        source_scaffolds[row["scaffold"]].add(row["split"])
    if len(source_smiles) != len(source):
        issues.append("duplicate SMILES in derived M")
    if any(len(splits) != 1 for splits in source_scaffolds.values()):
        issues.append("scaffold shared across derived M splits")
    actual_counts = Counter((row["split"], row["label"]) for row in source)
    if len(source) != source_meta["rows"]:
        issues.append("derived M row count differs from summary")
    original_m_rows = {}
    for name in source_meta["source_datasets"]:
        for row in read(root / name / "dataset.csv"):
            if row["split"] in source_meta["source_splits"]:
                original_m_rows[row["id"]] = row
    for row in source:
        original = original_m_rows.get(row["id"])
        if original is None or any(row[key] != original[key] for key in row
                                   if key != "split"):
            issues.append("derived M row differs from its current source")
            break
    for split in ("train", "val", "test"):
        if actual_counts[split, "0"] != actual_counts[split, "1"]:
            issues.append(f"derived M {split} is not class-balanced")
        for label in ("0", "1"):
            if actual_counts[split, label] != source_meta["split_label_counts"][split][label]:
                issues.append(f"derived M {split}/{label} differs from summary")

    original_seen_smiles = set()
    original_seen_scaffolds = set()
    for name in target_meta["source_datasets"]:
        for row in read(root / name / "dataset.csv"):
            if row["split"] in target_meta["source_splits"]:
                original_seen_smiles.add(row["smiles"])
                original_seen_scaffolds.add(row["scaffold"])
    evolved_test = {row["id"]: row for row in read(root / "evolved/dataset.csv")
                    if row["split"] == "test"}
    target_counts = Counter(row["label"] for row in target)
    if len({row["smiles"] for row in target}) != len(target):
        issues.append("duplicate SMILES in E holdout")
    if target_counts["0"] != target_counts["1"]:
        issues.append("E holdout is not class-balanced")
    if len(target) != target_meta["retained_rows"]:
        issues.append("E holdout row count differs from summary")
    for row in target:
        if row["id"] not in evolved_test or row != evolved_test[row["id"]]:
            issues.append("E holdout row differs from original evolved test")
            break
        if row["smiles"] in original_seen_smiles or row["scaffold"] in original_seen_scaffolds:
            issues.append("E holdout overlaps original M source exposure")
            break

    report = {
        "M_rows": len(source),
        "M_split_label_counts": {split: {label: actual_counts[split, label]
                                          for label in ("0", "1")}
                                 for split in ("train", "val", "test")},
        "E_test_rows": len(target),
        "E_label_counts": dict(target_counts),
        "issues": issues,
    }
    print(json.dumps(report, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
