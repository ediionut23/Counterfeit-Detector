"""Build a scaffold-split binary dataset with equal counterfeit subcategory counts.

The source bank is kept intact. Optional targeted MMP rows are merged as
additional relations; duplicate child molecules are never repeated in the
output. Classes below ``--min-class-count`` are listed in the report and
excluded from this balanced training view because they lack enough unique
examples for a useful split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import assemble_36_benchmark as asm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def own_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold.GetNumAtoms() else "<acyclic>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    parser.add_argument("--extension", type=Path,
                        default=Path("training_subcategory_expansion/counterfeits_mmp_by_subcategory.csv"))
    parser.add_argument("--authentic-extra", type=Path,
                        default=Path("molecule_bank_500k/chembl_controls.csv"),
                        help="Optional additional label-0 reference molecules (for example, filtered ChEMBL controls)")
    parser.add_argument("--max-per-subcategory", type=int, default=200)
    parser.add_argument("--min-class-count", type=int, default=50)
    parser.add_argument("--property-bins", type=int, default=6,
                        help="Quantile bins used for MW/heavy-atom/SMILES-length matching")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("training_subcategory_balanced"))
    args = parser.parse_args()
    if args.max_per_subcategory < args.min_class_count or args.min_class_count < 2:
        parser.error("max-per-subcategory must be >= min-class-count >= 2")
    if args.property_bins < 1:
        parser.error("property-bins must be positive")

    bank = args.bank
    index = json.loads((bank / "subcategories/index.json").read_text())
    molecules = {r["smiles"]: r for r in read(bank / "molecules.csv")}
    fakes_by_class = defaultdict(dict)
    for class_name, meta in sorted(index.items()):
        for rel in read(bank / meta["path"]):
            mol = molecules[rel["smiles"]]
            fakes_by_class[class_name].setdefault(rel["smiles"], {
                "smiles": rel["smiles"], "label": "1", "generation_method": rel["method"],
                "category": rel["category"], "subcategory": rel["subcategory"],
                "parent_smiles": rel["parent_smiles"], "edited_atoms": rel["edited_atoms"],
                "edit_status": rel["edit_status"], "source_relation_id": rel["relation_id"],
                "scaffold": rel["parent_scaffold"], "own_scaffold": rel["child_scaffold"],
                "mw": float(mol["mw"]), "logp": float(mol["logp"]),
                "qed": float(mol["qed"]), "heavy_atoms": int(mol["heavy_atoms"]),
                "smiles_length": int(mol["smiles_length"]),
            })

    # A child may belong to several source relations. Give it to its rarest
    # class so one physical molecule can never appear with multiple subclass
    # labels or leak across splits.
    owners = defaultdict(list)
    for class_name, candidates in fakes_by_class.items():
        for smi in candidates:
            owners[smi].append(class_name)
    duplicate_child_count = sum(len(classes) > 1 for classes in owners.values())
    assigned = defaultdict(dict)
    for smi, classes in owners.items():
        owner = min(classes, key=lambda name: (len(fakes_by_class[name]), name))
        assigned[owner][smi] = fakes_by_class[owner][smi]
    fakes_by_class = assigned

    existing_child_smiles = set(molecules)
    if args.extension.exists():
        for r in read(args.extension):
            smi = r["counterfeit_smiles"]
            if smi in existing_child_smiles:
                continue
            class_name = f"M/{r['category']}/{r['subcategory']}"
            fakes_by_class[class_name].setdefault(smi, {
                "smiles": smi, "label": "1", "generation_method": "M",
                "category": r["category"], "subcategory": r["subcategory"],
                "parent_smiles": r["parent_smiles"], "edited_atoms": r.get("edited_atoms", ""),
                "edit_status": "source_approximate",
                "source_relation_id": "gen_" + hashlib.sha256(
                    f"{r['parent_smiles']}\x1f{smi}\x1f{r['category']}\x1f{r['subcategory']}".encode()
                ).hexdigest()[:20],
                "scaffold": asm.murcko(r["parent_smiles"]), "own_scaffold": own_scaffold(smi),
                "mw": float(r["mw"]), "logp": float(r["logp"]),
                "qed": float(r["qed"]), "heavy_atoms": int(r["heavy_atoms"]),
                "smiles_length": len(smi),
            })

    support = {key: len(rows) for key, rows in fakes_by_class.items()}
    eligible = {key: count for key, count in support.items()
                if count >= args.min_class_count}
    excluded = {key: count for key, count in support.items()
                if count < args.min_class_count}
    if not eligible:
        raise ValueError("No subcategories meet the minimum class count")
    target = min(args.max_per_subcategory, min(eligible.values()))
    rng = random.Random(args.seed)
    fake_rows = []
    selected_counts = {}
    for key in sorted(eligible):
        options = list(fakes_by_class[key].values())
        rng.shuffle(options)
        chosen = options[:target]
        selected_counts[key] = len(chosen)
        fake_rows.extend(chosen)

    auth_rows = []
    auth_by_smiles = {}
    authentic_paths = [bank / "authentic_eligible.csv"]
    if args.authentic_extra.exists():
        authentic_paths.append(args.authentic_extra)
    for path in authentic_paths:
        for r in read(path):
            smi = r["smiles"]
            if smi in existing_child_smiles or smi in auth_by_smiles:
                continue
            auth_rows.append({
                "smiles": smi, "label": "0", "generation_method": "",
                "category": "authentic", "subcategory": "authentic",
                "parent_smiles": "", "edited_atoms": "", "edit_status": "",
                "source_relation_id": "", "scaffold": r["child_scaffold"],
                "own_scaffold": r["child_scaffold"], "mw": float(r["mw"]),
                "logp": float(r["logp"]), "qed": float(r["qed"]),
                "heavy_atoms": int(r["heavy_atoms"]), "smiles_length": int(r["smiles_length"]),
            })
            auth_by_smiles[smi] = True
    matched_auth, matched_fake = asm.property_match(
        auth_rows, fake_rows, n_bins=args.property_bins, seed=args.seed
    )
    matched_counts = Counter(f"{r['generation_method']}/{r['category']}/{r['subcategory']}"
                             for r in matched_fake)
    dropped_after_property_match = sorted(set(eligible) - set(matched_counts))
    final_target = min(matched_counts.values(), default=0)
    if final_target < args.min_class_count:
        raise ValueError(
            f"Property matching leaves only {final_target} examples in the smallest class; "
            "lower --min-class-count or add more valid class-specific molecules"
        )
    fake_by_class = defaultdict(list)
    for row in matched_fake:
        fake_by_class[f"{row['generation_method']}/{row['category']}/{row['subcategory']}"].append(row)
    final_fake = []
    for key in sorted(fake_by_class):
        group = fake_by_class[key]
        rng.shuffle(group)
        final_fake.extend(group[:final_target])
    rng.shuffle(matched_auth)
    final_auth = matched_auth[:len(final_fake)]
    rows = final_auth + final_fake
    for i, row in enumerate(rows):
        row["id"] = f"row_{i:07d}"
    asm.scaffold_split(rows, seed=args.seed)
    rng.shuffle(rows)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fields = ["id", "smiles", "label", "generation_method", "category", "subcategory",
              "parent_smiles", "edited_atoms", "edit_status", "source_relation_id",
              "scaffold", "own_scaffold", "split", "mw", "logp", "qed",
              "heavy_atoms", "smiles_length"]
    with (out / "dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    split_counts = Counter((row["split"], str(row["label"])) for row in rows)
    split_subcategory_counts = {
        split: dict(sorted(Counter(
            f"{r['generation_method']}/{r['category']}/{r['subcategory']}"
            for r in rows if r["split"] == split and str(r["label"]) == "1"
        ).items()))
        for split in ("train", "val", "test")
    }
    summary = {
        "bank": str(bank), "extension": str(args.extension),
        "authentic_sources": [str(path) for path in authentic_paths], "seed": args.seed,
        "target_per_subcategory": args.max_per_subcategory,
        "minimum_supported_class_count": args.min_class_count,
        "property_matching_quantile_bins": args.property_bins,
        "balanced_count_per_counterfeit_subcategory": final_target,
        "included_counterfeit_subcategories": len(fake_by_class),
        "excluded_subcategories_below_minimum": excluded,
        "excluded_after_property_matching": dropped_after_property_match,
        "source_unique_counts": support,
        "matched_counts_before_final_balance": dict(sorted(matched_counts.items())),
        "cross_subcategory_duplicate_children_assigned_once": duplicate_child_count,
        "selected_counts_after_balancing": {key: final_target for key in sorted(fake_by_class)},
        "property_match_shortcut_metrics": asm.shortcut_metrics(final_auth, final_fake),
        "rows": len(rows), "rows_per_class": {"0": len(final_auth), "1": len(final_fake)},
        "split_label_counts": {split: {label: split_counts[split, label]
                                       for label in ("0", "1")}
                               for split in ("train", "val", "test")},
        "split_counterfeit_subcategory_counts": split_subcategory_counts,
        "split_definition": "parent Bemis-Murcko scaffold groups assigned to one split",
        "strict_child_test": False,
        "authentic_label_note": "label 0 contains quality-filtered public reference molecules, not a verified genuine/counterfeit adjudication",
        "generated_label_note": "label 1 denotes generated analogues, not confirmed counterfeit products",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "rows", "rows_per_class", "balanced_count_per_counterfeit_subcategory",
        "included_counterfeit_subcategories", "excluded_subcategories_below_minimum",
        "split_label_counts")}, indent=2))


if __name__ == "__main__":
    main()
