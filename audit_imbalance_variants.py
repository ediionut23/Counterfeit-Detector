"""Validate the evolved train-only class-imbalance variants."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    root = Path("benchmark/imbalance")
    summary = json.loads((root / "summary.json").read_text())
    source = read(Path(summary["source"]))
    source_rows = {row["id"]: row for row in source}
    fixed = {row["id"]: row for row in source if row["split"] in ("val", "test")}
    authentic_train = {row["id"] for row in source
                       if row["split"] == "train" and row["label"] == "0"}
    issues = []
    positive_sets = {}
    for ratio, meta in summary["variants"].items():
        rows = read(Path(meta["path"]))
        actual = {row["id"]: row for row in rows}
        if len(actual) != len(rows) or not all(source_rows.get(key) == row
                                              for key, row in actual.items()):
            issues.append(f"1:{ratio}: duplicate or changed source rows")
        if {key: row for key, row in actual.items()
            if row["split"] in ("val", "test")} != fixed:
            issues.append(f"1:{ratio}: validation/test differs from source")
        if {row["id"] for row in rows if row["split"] == "train"
            and row["label"] == "0"} != authentic_train:
            issues.append(f"1:{ratio}: authentic training class changed")
        positive_sets[int(ratio)] = {row["id"] for row in rows
                                     if row["split"] == "train" and row["label"] == "1"}
        counts = Counter((row["split"], row["label"]) for row in rows)
        for split in ("train", "val", "test"):
            for label in ("0", "1"):
                if counts[split, label] != meta["split_label_counts"][split][label]:
                    issues.append(f"1:{ratio}: summary count mismatch for {split}/{label}")
        scaffolds = defaultdict(set)
        for row in rows:
            scaffolds[row["scaffold"]].add(row["split"])
        if any(len(splits) > 1 for splits in scaffolds.values()):
            issues.append(f"1:{ratio}: scaffold shared across splits")
    ratios = sorted(positive_sets)
    for smaller, larger in zip(ratios, ratios[1:]):
        if not positive_sets[larger] <= positive_sets[smaller]:
            issues.append(f"1:{larger} positives are not nested in 1:{smaller}")
    report = {"variants": len(ratios), "issues": issues}
    print(json.dumps(report, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
