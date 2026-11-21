"""Verify fixed splits, counts, and size selection in graph-size stress variants."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    root = Path("benchmark/graph_size")
    report = json.loads((root / "summary.json").read_text())
    source = read(Path(report["source"]))
    source_by_id = {row["id"]: row for row in source}
    fixed = {row["id"]: row for row in source if row["split"] in ("val", "test")}
    authentic = {row["id"] for row in source
                 if row["split"] == "train" and row["label"] == "0"}
    positives = [row for row in source if row["split"] == "train" and row["label"] == "1"]
    random.Random(report["seed"]).shuffle(positives)
    by_size = sorted(positives, key=lambda row: int(row["heavy_atoms"]))
    target = report["train_counterfeit"]
    expected_ids = {"small": {row["id"] for row in by_size[:target]},
                    "large": {row["id"] for row in by_size[-target:]}}
    issues = []
    for name, meta in report["variants"].items():
        rows = read(Path(meta["path"]))
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or len(rows) != meta["rows"]:
            issues.append(f"{name}: duplicate IDs or row count changed")
        if any(source_by_id.get(key) != row for key, row in by_id.items()):
            issues.append(f"{name}: source rows changed")
        if {key: row for key, row in by_id.items()
            if row["split"] in ("val", "test")} != fixed:
            issues.append(f"{name}: validation/test changed")
        if {row["id"] for row in rows if row["split"] == "train" and
            row["label"] == "0"} != authentic:
            issues.append(f"{name}: authentic training rows changed")
        selected = {row["id"] for row in rows if row["split"] == "train" and
                    row["label"] == "1"}
        if selected != expected_ids[name]:
            issues.append(f"{name}: incorrect size-based selection")
        counts = Counter((row["split"], row["label"]) for row in rows)
        for split in ("train", "val", "test"):
            for label in ("0", "1"):
                if counts[split, label] != meta["split_label_counts"][split][label]:
                    issues.append(f"{name}: class count mismatch")
        scaffolds = defaultdict(set)
        for row in rows:
            scaffolds[row["scaffold"]].add(row["split"])
        if any(len(splits) > 1 for splits in scaffolds.values()):
            issues.append(f"{name}: scaffold shared across splits")
    result = {"variants": len(report["variants"]), "issues": issues}
    print(json.dumps(result, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
