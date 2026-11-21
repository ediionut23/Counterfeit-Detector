"""Check row counts, labels, and split isolation in every benchmark CSV.

Usage: python audit_benchmark_integrity.py [benchmark_directory]
Exits nonzero when the snapshot contains an inconsistency.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def audit(root: Path) -> dict:
    index = json.loads((root / "index.json").read_text())
    files = {str(p.parent.relative_to(root)): p for p in root.rglob("dataset.csv")}
    issues = []
    counts = Counter()
    for name in sorted(set(index) | set(files)):
        if name not in index or name not in files:
            issues.append(f"{name}: missing {'index entry' if name not in index else 'dataset.csv'}")
            continue

        labels = Counter()
        splits = Counter()
        split_labels = Counter()
        ids = set()
        smiles = {}
        scaffolds = {}
        with files[name].open(newline="") as stream:
            for row in csv.DictReader(stream):
                labels[row["label"]] += 1
                splits[row["split"]] += 1
                split_labels[row["split"], row["label"]] += 1
                if row["label"] == "1" and not row["edited_atoms"].strip():
                    counts["counterfeit_rows_without_edited_atoms"] += 1
                if row["label"] == "1" and not row["parent_smiles"].strip():
                    issues.append(f"{name}: counterfeit without parent: {row['id']}")
                if row["label"] == "0" and (row["parent_smiles"].strip() or row["edited_atoms"].strip()):
                    issues.append(f"{name}: authentic row has parent/edit metadata: {row['id']}")
                if row["id"] in ids:
                    issues.append(f"{name}: duplicate id {row['id']}")
                ids.add(row["id"])

                smi, label, split = row["smiles"], row["label"], row["split"]
                previous = smiles.setdefault(smi, (label, split))
                if previous[0] != label:
                    issues.append(f"{name}: conflicting labels for {smi}")
                if previous[1] != split:
                    issues.append(f"{name}: SMILES in multiple splits: {smi}")
                scaffold = row["scaffold"]
                if scaffold in scaffolds and scaffolds[scaffold] != split:
                    issues.append(f"{name}: scaffold in multiple splits: {scaffold}")
                scaffolds[scaffold] = split

        expected = index[name]
        n = sum(labels.values())
        counts["datasets"] += 1
        counts["rows"] += n
        for label in ("0", "1"):
            counts[f"label_{label}"] += labels[label]
        if set(labels) - {"0", "1"}:
            issues.append(f"{name}: invalid labels {sorted(labels)}")
        if set(splits) - {"train", "val", "test"}:
            issues.append(f"{name}: invalid splits {sorted(splits)}")
        if labels["0"] != labels["1"]:
            issues.append(f"{name}: class imbalance {dict(labels)}")
        if n != expected["n_total"]:
            issues.append(f"{name}: {n} rows, index says {expected['n_total']}")
        if labels["0"] != expected["n_authentic"] or labels["1"] != expected["n_counterfeit"]:
            issues.append(f"{name}: label counts disagree with index")
        if dict(splits) != expected["split_counts"]:
            issues.append(f"{name}: split counts disagree with index")
        if "split_label_counts" in expected:
            actual = {s: {label: split_labels[s, label] for label in ("0", "1")}
                      for s in ("train", "val", "test")}
            if actual != expected["split_label_counts"]:
                issues.append(f"{name}: split label counts disagree with index")

    return {"counts": dict(counts), "n_issues": len(issues), "examples": issues[:20]}


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmark")
    report = audit(root)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["n_issues"] else 0)
