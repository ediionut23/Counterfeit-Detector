"""Audit the supplementary evolved test with unseen row-SMILES scaffolds."""

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
    parser.add_argument("--directory", type=Path,
                        default=Path("benchmark/strict_child_scaffold"))
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    source = load(Path(summary["source"]))
    output = load(args.directory / "evolved_test.csv")
    by_id = {row["id"]: row for row in source}
    exposed = {own_scaffold(row["smiles"]) for row in source
               if row["split"] in ("train", "val")}
    source_parent_scaffolds = {row["scaffold"] for row in source
                               if row["split"] in ("train", "val")}
    issues = []
    if len({row["id"] for row in output}) != len(output):
        issues.append("duplicate IDs")
    if any(by_id.get(row["id"]) != row for row in output):
        issues.append("output row differs from source")
    if any(row["split"] != "test" for row in output):
        issues.append("non-test row in holdout")
    if any(own_scaffold(row["smiles"]) in exposed for row in output):
        issues.append("row-SMILES scaffold exposed in source train/val")
    if any(row["scaffold"] in source_parent_scaffolds for row in output):
        issues.append("parent scaffold exposed in source train/val")
    counts = Counter(row["label"] for row in output)
    if counts["0"] != counts["1"] or dict(counts) != summary["balanced_test_labels"]:
        issues.append("label counts unbalanced or differ from summary")
    if len(output) != summary["rows"]:
        issues.append("row count differs from summary")
    report = {"rows": len(output), "label_counts": dict(counts),
              "exposed_row_scaffolds": len(exposed), "issues": issues}
    (args.directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
