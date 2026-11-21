"""Generate unique MMP analogues for underrepresented bank subcategories.

The original large-scale generator allocates mutations by broad category and
samples mined rules in proportion to their support. That is efficient for
volume, but can leave rare mapped subcategories almost empty. This extension
allocates a target per category/subcategory pair and excludes every molecule
already present in the audited bank.

Example:
  python generate_subcategory_expansion.py --target-per-subcategory 250 \
      --out training_subcategory_expansion
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

import generate_100k_counterfeits as base
from mined_transformations import MMP_TRANSFORMATIONS, SUBCATEGORY

RDLogger.DisableLog("rdApp.*")

FIELDS = [
    "counterfeit_smiles", "parent_smiles", "category", "subcategory",
    "rule_name", "difficulty", "similarity_to_parent", "edited_atoms",
    "n_edited_atoms", "label", "mw", "logp", "qed", "sa_score",
    "heavy_atoms", "source",
]
ALLOWED_ATOMS = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "H"}


def is_plausible(mol: Chem.Mol) -> bool:
    if mol is None:
        return False
    try:
        heavy = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        return (10 <= heavy <= 70 and 150 <= mw <= 800
                and all(atom.GetSymbol() in ALLOWED_ATOMS for atom in mol.GetAtoms())
                and rdMolDescriptors.CalcNumRings(mol) > 0)
    except Exception:
        return False


def existing_counts(bank: Path) -> tuple[Counter, set[str]]:
    index = json.loads((bank / "subcategories/index.json").read_text())
    counts = Counter()
    for label, metadata in index.items():
        if not label.startswith("M/"):
            continue
        with (bank / metadata["path"]).open(newline="") as stream:
            unique = {row["smiles"] for row in csv.DictReader(stream)}
        counts[label.removeprefix("M/")] = len(unique)
    with (bank / "molecules.csv").open(newline="") as stream:
        seen = {row["smiles"] for row in csv.DictReader(stream)}
    return counts, seen


def generate_group(category, subcategory, rules, parents, target, initial, seen,
                   seed, sim_lo, sim_hi, attempts_per_parent):
    """Try each parent with this subcategory's rules until its unique target."""
    rows = []
    if initial >= target:
        return rows, "already_at_target"
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    weights = np.array([max(1.0, rule["weight"]) for rule in rules], dtype=float)
    probs = weights / weights.sum()
    order = list(range(len(parents)))
    rng.shuffle(order)

    for parent_i in order:
        if initial + len(rows) >= target:
            break
        parent_smi = parents[parent_i]
        parent = Chem.MolFromSmiles(parent_smi)
        if parent is None:
            continue
        parent_fp = base.morgan(parent)
        parent_mw = Descriptors.MolWt(parent)
        for _ in range(attempts_per_parent):
            if initial + len(rows) >= target:
                break
            rule = rules[int(np_rng.choice(len(rules), p=probs))]
            # Reaction execution, product sanitization and quality filters are
            # much more expensive than a SMARTS match. Skip impossible
            # parent/rule pairs first, which matters for rare transformations.
            if not parent.HasSubstructMatch(rule["reaction"].GetReactantTemplate(0)):
                continue
            child = base.apply_rule(parent, rule)
            if child is None:
                continue
            try:
                child_smi = Chem.MolToSmiles(child)
            except Exception:
                continue
            if child_smi == parent_smi or child_smi in seen:
                continue
            if not is_plausible(child):
                continue
            child_mw = Descriptors.MolWt(child)
            if abs(child_mw - parent_mw) > 80.0:
                continue
            similarity = DataStructs.TanimotoSimilarity(parent_fp, base.morgan(child))
            if not sim_lo <= similarity <= sim_hi:
                continue
            qed = float(Descriptors.qed(child))
            if qed < 0.35 or base.has_structural_alerts(child):
                continue
            sa = base.sascorer.calculateScore(child) if base.sascorer else None
            if sa is None or sa > 3.8:
                continue
            # Reuse the source generator's canonical-SMILES-indexed edit mask.
            edits = base.reaction_edited_atoms(child)
            if not edits:
                edits = base.edited_atoms(parent, Chem.MolFromSmiles(child_smi))
            seen.add(child_smi)
            rows.append({
                "counterfeit_smiles": child_smi,
                "parent_smiles": parent_smi,
                "category": category,
                "subcategory": subcategory,
                "rule_name": rule["name"],
                "difficulty": rule["difficulty"],
                "similarity_to_parent": round(float(similarity), 4),
                "edited_atoms": " ".join(map(str, edits)),
                "n_edited_atoms": len(edits),
                "label": 1,
                "mw": round(float(child_mw), 2),
                "logp": round(float(Descriptors.MolLogP(child)), 2),
                "qed": round(qed, 4),
                "sa_score": round(float(sa), 3),
                "heavy_atoms": child.GetNumHeavyAtoms(),
                "source": "targeted_mmp_subcategory",
            })
    return rows, "generated" if rows else "no_valid_candidates"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    parser.add_argument("--parents", type=Path, action="append", default=None,
                        help="Authentic parent SMILES file; repeat to combine independent sources")
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[],
                        help="Existing generated CSV to exclude and count; repeatable")
    parser.add_argument("--target-per-subcategory", type=int, default=250)
    parser.add_argument("--attempts-per-parent", type=int, default=6)
    parser.add_argument("--sim-lo", type=float, default=0.4)
    parser.add_argument("--sim-hi", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42019)
    parser.add_argument("--include-new-subcategories", action="store_true",
                        help="Also explore rule classes not yet present in the source bank")
    parser.add_argument("--only-group", action="append", default=[],
                        help="Restrict generation to category/subcategory; repeatable")
    parser.add_argument("--out", type=Path, default=Path("training_subcategory_expansion"))
    args = parser.parse_args()
    if args.target_per_subcategory < 1 or args.attempts_per_parent < 1:
        parser.error("target and attempts per parent must be positive")
    if not 0.0 <= args.sim_lo <= args.sim_hi <= 1.0:
        parser.error("similarity bounds must satisfy 0 <= low <= high <= 1")

    current, seen = existing_counts(args.bank)
    parent_files = args.parents or [Path("public_molecules_100k.smi")]
    parents = []
    parent_seen = set()
    for source_i, source_path in enumerate(parent_files):
        for parent in base.load_parents(str(source_path), limit=None,
                                        seed=args.seed + source_i):
            if parent not in parent_seen:
                parent_seen.add(parent)
                parents.append(parent)
    auth = set(parents)
    seen.update(auth)
    prior_counts = Counter()
    excluded_rows = []
    for path in args.exclude_csv:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        excluded_rows.extend(rows)
        for row in rows:
            seen.add(row["counterfeit_smiles"])
            prior_counts[f"{row['category']}/{row['subcategory']}"] += 1

    by_group = defaultdict(list)
    for name, difficulty, category, reactant, product, citation in MMP_TRANSFORMATIONS:
        reaction = base.AllChem.ReactionFromSmarts(f"{reactant}>>{product}")
        if reaction is None:
            continue
        by_group[(category, SUBCATEGORY.get(name, "other" if category == "other" else "unknown"))].append({
            "name": name, "difficulty": difficulty, "reaction": reaction,
            "weight": float(getattr(__import__("mined_transformations"), "SUPPORT", {}).get(name, 1)),
            "citation": citation,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "counterfeits_mmp_by_subcategory.csv"
    summary_path = args.out / "summary.json"
    prior_rows = []
    report = {}
    if out_csv.exists() and summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous.get("target_per_subcategory") != args.target_per_subcategory:
            parser.error("Existing output uses a different target; choose a new --out directory")
        report = previous
        with out_csv.open(newline="") as stream:
            prior_rows = list(csv.DictReader(stream))
        seen.update(r["counterfeit_smiles"] for r in prior_rows)
    else:
        with out_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
        report = {
            "bank": str(args.bank), "parents": [str(path) for path in parent_files],
            "excluded_csvs": [str(path) for path in args.exclude_csv], "seed": args.seed,
            "target_per_subcategory": args.target_per_subcategory,
            "source_parent_count": len(parents), "groups": {},
        }
    all_rows = list(prior_rows)
    ordered = sorted(by_group.items(), key=lambda item: (
        current.get(f"{item[0][0]}/{item[0][1]}", 0), item[0][0], item[0][1]))
    for i, ((category, subcategory), rules) in enumerate(ordered):
        key = f"{category}/{subcategory}"
        if args.only_group and key not in args.only_group:
            continue
        if key in report["groups"]:
            continue
        if not args.include_new_subcategories and key not in current:
            continue
        existing = current.get(key, 0) + prior_counts[key]
        existing += sum(1 for row in prior_rows
                        if row["category"] == category and row["subcategory"] == subcategory)
        rows, status = generate_group(
            category, subcategory, rules, parents, args.target_per_subcategory,
            existing, seen, args.seed + i, args.sim_lo, args.sim_hi,
            args.attempts_per_parent,
        )
        all_rows.extend(rows)
        if rows:
            with out_csv.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
                writer.writerows(rows)
        report["groups"][key] = {
            "existing_unique": existing, "new_unique": len(rows),
            "final_unique": existing + len(rows), "target": args.target_per_subcategory,
            "n_rules": len(rules), "status": status,
        }
        print(f"{key}: {existing}+{len(rows)} / {args.target_per_subcategory} ({status})", flush=True)
        report["generated_unique"] = len(all_rows)
        report["output_csv"] = str(out_csv)
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    report["generated_unique"] = len(all_rows)
    report["quality_filters"] = {
        "heavy_atoms": [10, 70], "molecular_weight": [150, 800],
        "qed_min": 0.35, "sa_max": 3.8, "structural_alerts": "excluded",
        "morgan_radius3_similarity": [args.sim_lo, args.sim_hi],
        "global_duplicate_exclusion": "all unique children already in selected bank plus new rows",
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Generated {len(all_rows):,} unique molecules -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
