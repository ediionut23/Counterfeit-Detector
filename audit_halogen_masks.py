"""Audit exact single-swap halogen edit masks across all benchmark tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from repair_halogen_substitution_masks import PATTERN, exact_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/halogen_mask_audit.json"))
    args = parser.parse_args()
    counts = Counter()
    exact_pairs = set()
    issues = []
    by_subcategory = Counter()
    for path in sorted(args.root.rglob("dataset.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["label"] != "1" or row["category"] != "halogen-walk":
                    continue
                match = PATTERN.fullmatch(row["subcategory"])
                if match is None:
                    continue
                counts["eligible_rows"] += 1
                desired = exact_mask(row["parent_smiles"], row["smiles"],
                                     *match.groups())
                if desired is None:
                    counts["unresolved_rows"] += 1
                    continue
                counts["exact_rows"] += 1
                by_subcategory[row["subcategory"]] += 1
                exact_pairs.add((row["parent_smiles"], row["smiles"]))
                if row["edited_atoms"] != desired:
                    issues.append(f"{path}: {row['id']} has {row['edited_atoms']!r}, expected {desired!r}")
    report = {**dict(counts), "unique_exact_pairs": len(exact_pairs),
              "exact_rows_by_subcategory": dict(sorted(by_subcategory.items())),
              "n_issues": len(issues), "issue_examples": issues[:10]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
