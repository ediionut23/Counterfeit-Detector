"""Check matched size bins and fixed rows in scaffold-diversity variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_scaffold_diversity_variants import size_bin


def load(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path("benchmark/scaffold_diversity"))
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    source = load(Path(summary["source"]))
    by_id = {row["id"]: row for row in source}
    if len(by_id) != len(source):
        raise ValueError("Source IDs are not unique")
    original_eval = [row for row in source if row["split"] != "train"]
    variants = {}
    issues = []
    for name in ("low", "high"):
        path = args.directory / f"evolved_train_scaffolds_{name}.csv"
        rows = load(path)
        if len({row["id"] for row in rows}) != len(rows):
            issues.append(f"{name}: duplicate IDs")
        if any(by_id.get(row["id"]) != row for row in rows):
            issues.append(f"{name}: row changed from source")
        if [row for row in rows if row["split"] != "train"] != original_eval:
            issues.append(f"{name}: validation/test rows changed")
        authentic = {row["id"] for row in rows
                     if row["split"] == "train" and row["label"] == "0"}
        fake = [row for row in rows
                if row["split"] == "train" and row["label"] == "1"]
        bins = {str(i): sum(size_bin(row) == i for row in fake) for i in range(3)}
        scaffolds = len({row["scaffold"] for row in fake})
        expected = summary["variants"][name]
        if (len(rows) != expected["rows"] or
                len(authentic) != summary["train_examples_per_class"] or
                len(fake) != summary["train_examples_per_class"] or
                bins != expected["counterfeit_train_size_bins"] or
                scaffolds != expected["counterfeit_train_scaffolds"]):
            issues.append(f"{name}: counts or scaffold diversity differ from summary")
        variants[name] = {"authentic_train_ids": authentic, "size_bins": bins,
                          "counterfeit_scaffolds": scaffolds}
    if variants["low"]["authentic_train_ids"] != variants["high"]["authentic_train_ids"]:
        issues.append("authentic training rows differ")
    if variants["low"]["size_bins"] != variants["high"]["size_bins"]:
        issues.append("counterfeit heavy-atom size bins differ")
    if variants["low"]["counterfeit_scaffolds"] >= variants["high"]["counterfeit_scaffolds"]:
        issues.append("scaffold diversity contrast absent")
    report = {"variants": 2,
              "counterfeit_train_scaffolds": {
                  name: variants[name]["counterfeit_scaffolds"] for name in variants},
              "matched_size_bins": variants["low"]["size_bins"],
              "issues": issues}
    (args.directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
