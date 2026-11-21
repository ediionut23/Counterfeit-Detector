"""Merge a small GA top-up into a category table and rebalance scaffold caps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("version_ga"))
    parser.add_argument("--category", required=True,
                        choices=["bioisostere", "halogen-walk", "scaffold-hop", "homologation", "other"])
    parser.add_argument("--supplement", type=Path, action="append", required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[],
                        help="Remove exact structures present in these CSVs before selection.")
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--max-parent-scaffold", type=int, default=50)
    parser.add_argument("--max-child-scaffold", type=int, default=200)
    args = parser.parse_args()
    output = args.root / f"counterfeits_evolved_{args.category}.csv"
    rows = read_rows(output)
    n_before = len(rows)
    rows.extend(row for source in args.supplement for row in read_rows(source))

    excluded = set()
    for path in args.exclude_csv:
        for row in read_rows(path):
            mol = Chem.MolFromSmiles(row.get("counterfeit_smiles", row.get("smiles", "")))
            if mol is not None:
                excluded.add(Chem.MolToSmiles(mol))
    unique = {}
    for row in rows:
        mol = Chem.MolFromSmiles(row.get("counterfeit_smiles", ""))
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol)
        if smiles in excluded:
            continue
        row["counterfeit_smiles"] = smiles
        old = unique.get(smiles)
        if old is None or float(row.get("objective_score", 9e9)) < float(old.get("objective_score", 9e9)):
            unique[smiles] = row

    by_parent_scaffold: dict[str, list[dict]] = defaultdict(list)
    for row in unique.values():
        by_parent_scaffold[row.get("parent_scaffold", "")].append(row)
    for candidates in by_parent_scaffold.values():
        candidates.sort(key=lambda r: (float(r.get("objective_score", 9e9)), r["counterfeit_smiles"]))

    scaffold_order = sorted(by_parent_scaffold)
    selected, child_counts, parent_counts = [], {}, {}
    while len(selected) < args.target:
        added = False
        for scaffold in scaffold_order:
            candidates = by_parent_scaffold[scaffold]
            if not candidates or parent_counts.get(scaffold, 0) >= args.max_parent_scaffold:
                continue
            row = candidates.pop(0)
            child_scaffold = row.get("child_scaffold", "")
            if child_counts.get(child_scaffold, 0) >= args.max_child_scaffold:
                continue
            selected.append(row)
            child_counts[child_scaffold] = child_counts.get(child_scaffold, 0) + 1
            parent_counts[scaffold] = parent_counts.get(scaffold, 0) + 1
            added = True
            if len(selected) == args.target:
                break
        if not added:
            break
    if len(selected) < args.target:
        raise RuntimeError(f"Only {len(selected)} selected; target {args.target} not reached. "
                           "Increase parent/scaffold caps or generate another supplement.")

    columns = list(selected[0])
    temp = output.with_suffix(".csv.partial")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)
    temp.replace(output)

    summary_path = args.root / f"summary_{args.category}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({
        "n_unique_counterfeits": len(selected),
        "target": args.target,
        "target_reached": len(selected) == args.target,
        "pre_supplement_rows": n_before,
        "supplement_rows_read": len(rows) - n_before,
        "unique_candidates_before_balanced_selection": len(unique),
        "final_selection_caps": {
            "max_per_parent_scaffold": args.max_parent_scaffold,
            "max_per_child_scaffold": args.max_child_scaffold,
        },
        "quality": {
            "mean_similarity": round(float(np.mean([float(r["similarity_to_parent"]) for r in selected])), 4),
            "median_similarity": round(float(np.median([float(r["similarity_to_parent"]) for r in selected])), 4),
            "mean_qed": round(float(np.mean([float(r["qed"]) for r in selected])), 4),
            "median_qed": round(float(np.median([float(r["qed"]) for r in selected])), 4),
            "mean_sa_score": round(float(np.mean([float(r["sa_score"]) for r in selected if r["sa_score"]])), 4),
            "median_sa_score": round(float(np.median([float(r["sa_score"]) for r in selected if r["sa_score"]])), 4),
            "n_parent_scaffolds": len({r["parent_scaffold"] for r in selected}),
            "n_child_scaffolds": len({r["child_scaffold"] for r in selected}),
        },
    })
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"category": args.category, "rows": len(selected),
                      "unique_before_selection": len(unique), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
