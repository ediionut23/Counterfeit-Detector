"""Compare regenerated benchmark tables, summaries and index to a snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def table(path: Path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, default=Path("benchmark"))
    parser.add_argument("--actual", type=Path, default=Path("benchmark_regen"))
    args = parser.parse_args()

    expected_index = json.loads((args.expected / "index.json").read_text())
    actual_index = json.loads((args.actual / "index.json").read_text())
    names = sorted(set(expected_index) | set(actual_index))
    mismatches = {}
    for name in names:
        expected_csv = args.expected / name / "dataset.csv"
        actual_csv = args.actual / name / "dataset.csv"
        if not expected_csv.exists() or not actual_csv.exists():
            mismatches[name] = {"missing_csv": str(expected_csv if not expected_csv.exists()
                                                   else actual_csv)}
            continue
        expected_fields, expected_rows = table(expected_csv)
        actual_fields, actual_rows = table(actual_csv)
        columns = Counter()
        for left, right in zip(expected_rows, actual_rows):
            for field in expected_fields:
                if left[field] != right.get(field):
                    columns[field] += 1
        expected_summary = json.loads((args.expected / name / "summary.json").read_text())
        actual_summary = json.loads((args.actual / name / "summary.json").read_text())
        if (expected_fields != actual_fields or len(expected_rows) != len(actual_rows)
                or columns or expected_summary != actual_summary):
            mismatches[name] = {
                "expected_rows": len(expected_rows),
                "actual_rows": len(actual_rows),
                "column_differences": dict(columns),
                "summary_equal": expected_summary == actual_summary,
            }
    report = {
        "expected_datasets": len(expected_index),
        "actual_datasets": len(actual_index),
        "index_equal": expected_index == actual_index,
        "dataset_mismatches": mismatches,
    }
    print(json.dumps(report, indent=2))
    return int(bool(mismatches) or not report["index_equal"])


if __name__ == "__main__":
    raise SystemExit(main())
