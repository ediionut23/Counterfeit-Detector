"""Check a bank-derived balanced CSV and its scaffold separation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    summary = json.loads((root / "summary.json").read_text())
    with (root / "dataset.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    issues = Counter()
    if len(rows) != summary["rows"]:
        issues["row_count"] += 1
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows):
        issues["duplicate_id"] += 1
    labels = defaultdict(set)
    scaffolds = defaultdict(set)
    smiles_splits = defaultdict(set)
    counts = Counter()
    for row in rows:
        counts[row["split"], row["label"]] += 1
        labels[row["smiles"]].add(row["label"])
        scaffolds[row["scaffold"]].add(row["split"])
        smiles_splits[row["smiles"]].add(row["split"])
        if row["label"] == "1" and (not row["parent_smiles"] or not row["source_relation_id"]):
            issues["missing_counterfeit_provenance"] += 1
        if row["label"] == "0" and row["parent_smiles"]:
            issues["authentic_has_parent"] += 1
    if any(len(v) != 1 for v in labels.values()):
        issues["contradictory_label"] += 1
    if any(len(v) != 1 for v in scaffolds.values()):
        issues["scaffold_split_overlap"] += 1
    if any(len(v) != 1 for v in smiles_splits.values()):
        issues["molecule_split_overlap"] += 1
    for split in ("train", "val", "test"):
        if counts[split, "0"] != counts[split, "1"]:
            issues["split_imbalance"] += 1
        for label in ("0", "1"):
            if counts[split, label] != summary["split_label_counts"][split][label]:
                issues["summary_count_mismatch"] += 1
    if summary["strict_child_test"]:
        known = {r["own_scaffold"] for r in rows if r["split"] in ("train", "val")}
        if any(r["own_scaffold"] in known for r in rows if r["split"] == "test"):
            issues["strict_child_scaffold_overlap"] += 1
    report = {"rows": len(rows), "issues": dict(issues), "passed": not issues}
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
