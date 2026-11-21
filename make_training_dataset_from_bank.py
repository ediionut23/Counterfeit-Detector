"""Select final-bank subcategories and make a balanced training-ready CSV.

Example:
  python make_training_dataset_from_bank.py --select 'M/halogen-walk/*' \
      --select 'E/unconditioned/*' --max-counterfeits 20000 \
      --out derived_training_dataset
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import random
from collections import Counter
from pathlib import Path

import assemble_36_benchmark as asm


FIELDS = [
    "id", "smiles", "label", "generation_method", "category", "subcategory",
    "parent_smiles", "edited_atoms", "edit_status", "source_relation_id",
    "scaffold", "own_scaffold", "split", "mw", "logp", "qed",
    "heavy_atoms", "smiles_length",
]


def read(path: Path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    parser.add_argument("--select", action="append", default=[],
                        help="Glob over method/category/subcategory, e.g. M/bioisostere/*")
    parser.add_argument("--max-counterfeits", type=int, default=None)
    parser.add_argument("--strict-child-test", action="store_true",
                        help="Remove test rows with own scaffold in train/validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("derived_training_dataset"))
    args = parser.parse_args()
    if args.max_counterfeits is not None and args.max_counterfeits <= 0:
        parser.error("--max-counterfeits must be positive")
    selectors = args.select or ["M/*/*", "E/*/*"]
    index = json.loads((args.bank / "subcategories/index.json").read_text())
    keys = [key for key in sorted(index) if any(fnmatch.fnmatchcase(key, pattern)
                                                for pattern in selectors)]
    if not keys:
        raise ValueError(f"No subcategories match {selectors}")
    molecules = {row["smiles"]: row for row in read(args.bank / "molecules.csv")}
    chosen = {}
    for key in keys:
        for rel in read(args.bank / index[key]["path"]):
            chosen.setdefault(rel["smiles"], rel)
    selected_smiles = sorted(chosen)
    if args.max_counterfeits and len(selected_smiles) > args.max_counterfeits:
        selected_smiles = sorted(random.Random(args.seed).sample(
            selected_smiles, args.max_counterfeits))
    fake = []
    for smiles in selected_smiles:
        rel, mol = chosen[smiles], molecules[smiles]
        fake.append({
            "smiles": smiles, "label": "1", "generation_method": rel["method"],
            "category": rel["category"], "subcategory": rel["subcategory"],
            "parent_smiles": rel["parent_smiles"],
            "edited_atoms": rel["edited_atoms"], "edit_status": rel["edit_status"],
            "source_relation_id": rel["relation_id"],
            "scaffold": rel["parent_scaffold"],
            "own_scaffold": rel["child_scaffold"],
            "mw": float(mol["mw"]), "logp": float(mol["logp"]),
            "qed": float(mol["qed"]),
            "heavy_atoms": int(mol["heavy_atoms"]),
            "smiles_length": int(mol["smiles_length"]),
        })
    authentic = []
    for row in read(args.bank / "authentic_eligible.csv"):
        authentic.append({
            "smiles": row["smiles"], "label": "0", "generation_method": "",
            "category": "authentic", "subcategory": "authentic",
            "parent_smiles": "", "edited_atoms": "", "edit_status": "",
            "source_relation_id": "", "scaffold": row["child_scaffold"],
            "own_scaffold": row["child_scaffold"],
            "mw": float(row["mw"]), "logp": float(row["logp"]),
            "qed": float(row["qed"]),
            "heavy_atoms": int(row["heavy_atoms"]),
            "smiles_length": int(row["smiles_length"]),
        })
    auth_matched, fake_matched = asm.property_match(authentic, fake, seed=args.seed)
    rows = auth_matched + fake_matched
    asm.scaffold_split(rows, seed=args.seed)
    overlap_removed = 0
    if args.strict_child_test:
        seen = {row["own_scaffold"] for row in rows if row["split"] in ("train", "val")}
        retained = [row for row in rows if row["split"] != "test" or row["own_scaffold"] not in seen]
        overlap_removed = len(rows) - len(retained)
        rows = retained
    # Balance labels independently within each split without moving scaffolds.
    rng = random.Random(args.seed)
    balanced = []
    for split in ("train", "val", "test"):
        a = [row for row in rows if row["split"] == split and row["label"] == "0"]
        f = [row for row in rows if row["split"] == split and row["label"] == "1"]
        rng.shuffle(a)
        rng.shuffle(f)
        n = min(len(a), len(f))
        balanced.extend(a[:n] + f[:n])
    rng.shuffle(balanced)
    for i, row in enumerate(balanced):
        row["id"] = f"row_{i:07d}"
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(balanced)
    counts = Counter((row["split"], row["label"]) for row in balanced)
    own_train_val = {row["own_scaffold"] for row in balanced if row["split"] in ("train", "val")}
    child_overlap_test = sum(row["own_scaffold"] in own_train_val
                             for row in balanced if row["split"] == "test")
    summary = {
        "selectors": selectors, "selected_subcategories": keys,
        "seed": args.seed, "selected_unique_counterfeits": len(selected_smiles),
        "property_matched_per_class": len(fake_matched),
        "rows": len(balanced),
        "split_label_counts": {split: {label: counts[split, label] for label in ("0", "1")}
                               for split in ("train", "val", "test")},
        "strict_child_test": args.strict_child_test,
        "rows_removed_for_child_scaffold_overlap": overlap_removed,
        "test_own_scaffold_overlap_with_train_val": child_overlap_test,
        "split_definition": "parent Bemis-Murcko scaffold globally disjoint; strict option filters own-scaffold-overlap from test",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
