"""Build balanced, provenance-aware M, E, and combined benchmarks.

Positive rows are generated analogues (not confirmed counterfeit medicines);
negative rows are quality-filtered public reference molecules. Version GA is
included as an E subtype when its bank is supplied. Each split is balanced by
label, scaffold grouped, and the combined positive class is balanced M/E.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import assemble_36_benchmark as benchmark


OUTPUT_FIELDS = [
    "id", "molecule_id", "smiles", "label", "generation_method", "generator_variant",
    "category", "subcategory", "parent_smiles", "source_relation_id",
    "reference_source", "source_id",
    "scaffold", "own_scaffold", "split", "mw", "logp", "qed",
    "sa_score", "heavy_atoms", "smiles_length",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(smiles: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def generated_candidates(bank: Path, method: str, authentic_smiles: set[str],
                         ga_root: Path | None) -> list[dict]:
    molecules = read_csv(bank / "molecules.csv")
    bank_smiles = {canonical(row["smiles"]) for row in molecules}
    relations = read_csv(bank / "relations.csv")
    relation_by_id: dict[str, list[dict]] = defaultdict(list)
    for relation in relations:
        if relation["method"] == method:
            relation_by_id[relation["molecule_id"]].append(relation)

    candidates = []
    for molecule in molecules:
        # Keep only molecules attributable to exactly one generator. Shared
        # M/E products are deliberately held out of the single-generator sets.
        if molecule.get("methods") != method or molecule.get("label") != "1":
            continue
        rels = sorted(relation_by_id.get(molecule["molecule_id"], []),
                      key=lambda row: row["relation_id"])
        if not rels:
            continue
        rel = rels[0]
        smiles = canonical(molecule["smiles"])
        if smiles in authentic_smiles:
            continue
        parent = canonical(rel["parent_smiles"])
        child_scaffold = molecule.get("child_scaffold") or rel.get("child_scaffold") or ""
        parent_scaffold = rel.get("parent_scaffold") or ""
        # Preserve parent-family grouping even for acyclic molecules whose
        # Bemis-Murcko scaffold is empty.
        group_scaffold = parent_scaffold or f"ACYCLIC_PARENT:{parent}"
        own_scaffold = child_scaffold or f"ACYCLIC_CHILD:{smiles}"
        candidates.append({
            "molecule_id": molecule["molecule_id"], "smiles": smiles,
            "label": "1", "generation_method": method,
            "generator_variant": f"Version {method}",
            "category": rel.get("category", ""),
            "subcategory": rel.get("subcategory", ""),
            "parent_smiles": parent, "source_relation_id": rel["relation_id"],
            "reference_source": "", "source_id": rel["relation_id"],
            "scaffold": group_scaffold, "own_scaffold": own_scaffold,
            "mw": molecule.get("mw", ""), "logp": molecule.get("logp", ""),
            "qed": molecule.get("qed", ""), "sa_score": molecule.get("sa_score", ""),
            "heavy_atoms": molecule.get("heavy_atoms", ""),
            "smiles_length": molecule.get("smiles_length", ""),
        })
    if method == "E" and ga_root is not None:
        base_smiles = set(bank_smiles)
        # Exclude any GA output already present in the aggregate bank or in
        # the negative reference pool before assigning it to the E family.
        for row in read_csv(ga_root / "counterfeits_all.csv"):
            smiles = canonical(row["counterfeit_smiles"])
            if smiles in base_smiles or smiles in authentic_smiles:
                continue
            parent = canonical(row["parent_smiles"])
            mid = row.get("molecule_id") or "ga_" + __import__("hashlib").sha256(smiles.encode()).hexdigest()[:20]
            candidates.append({
                "molecule_id": mid, "smiles": smiles, "label": "1",
                "generation_method": "E", "generator_variant": "Version E category-conditioned GA",
                "category": row.get("category", ""),
                "subcategory": row.get("subcategory", row.get("category", "")),
                "parent_smiles": parent,
                "source_relation_id": f"GA_{mid}",
                "reference_source": "", "source_id": mid,
                "scaffold": row.get("parent_scaffold") or f"ACYCLIC_PARENT:{parent}",
                "own_scaffold": row.get("child_scaffold") or f"ACYCLIC_CHILD:{smiles}",
                "mw": row.get("mw", ""), "logp": row.get("logp", ""),
                "qed": row.get("qed", ""), "sa_score": row.get("sa_score", ""),
                "heavy_atoms": row.get("heavy_atoms", ""),
                "smiles_length": row.get("smiles_length", ""),
            })
            base_smiles.add(smiles)
    return candidates


def authentic_candidates(bank: Path, extra_path: Path | None) -> list[dict]:
    rows = []
    for row in read_csv(bank / "authentic_eligible.csv"):
        smiles = canonical(row["smiles"])
        scaffold = row.get("child_scaffold") or f"ACYCLIC_CHILD:{smiles}"
        rows.append({
            "molecule_id": row["molecule_id"], "smiles": smiles,
            "label": "0", "generation_method": "", "generator_variant": "",
            "category": "public_reference",
            "subcategory": "authentic", "parent_smiles": "",
            "source_relation_id": "", "scaffold": scaffold,
            "reference_source": "project_public_library", "source_id": row["molecule_id"],
            "own_scaffold": scaffold, "mw": row.get("mw", ""),
            "logp": row.get("logp", ""), "qed": row.get("qed", ""),
            "sa_score": row.get("sa_score", ""),
            "heavy_atoms": row.get("heavy_atoms", ""),
            "smiles_length": row.get("smiles_length", ""),
        })
    existing_smiles = {r["smiles"] for r in rows}
    if extra_path is not None:
        for row in read_csv(extra_path):
            smiles = canonical(row["smiles"])
            if smiles in existing_smiles:
                continue
            scaffold = row.get("child_scaffold") or f"ACYCLIC_CHILD:{smiles}"
            rows.append({
                "molecule_id": row["molecule_id"], "smiles": smiles, "label": "0",
                "generation_method": "", "generator_variant": "",
                "category": "public_reference", "subcategory": "ChEMBL",
                "parent_smiles": "", "source_relation_id": "",
                "scaffold": scaffold, "own_scaffold": scaffold,
                "reference_source": row.get("reference_source", "ChEMBL"),
                "source_id": row.get("source_id", row.get("molecule_id", "")),
                "mw": row.get("mw", ""), "logp": row.get("logp", ""),
                "qed": row.get("qed", ""), "sa_score": row.get("sa_score", ""),
                "heavy_atoms": row.get("heavy_atoms", ""),
                "smiles_length": row.get("smiles_length", ""),
            })
            existing_smiles.add(smiles)
    # Retain one row per exact standardized structure.
    unique = {}
    for row in rows:
        unique.setdefault(row["smiles"], row)
    return list(unique.values())


def remove_test_child_scaffold_overlap(rows: list[dict]) -> int:
    train_val = {r["own_scaffold"] for r in rows if r["split"] in {"train", "val"}}
    before = len(rows)
    rows[:] = [r for r in rows if r["split"] != "test" or r["own_scaffold"] not in train_val]
    return before - len(rows)


def balance_splits(rows: list[dict], kind: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for split in ("train", "val", "test"):
        auth = [r for r in rows if r["split"] == split and r["label"] == "0"]
        positives = [r for r in rows if r["split"] == split and r["label"] == "1"]
        rng.shuffle(auth)
        if kind == "combined":
            m_rows = [r for r in positives if r["generation_method"] == "M"]
            e_rows = [r for r in positives if r["generation_method"] == "E"]
            n_each = min(len(m_rows), len(e_rows), len(auth) // 2)
            if n_each == 0:
                raise ValueError(f"Combined dataset has no balanced M/E pool in {split}")
            rng.shuffle(m_rows); rng.shuffle(e_rows)
            chosen_pos = m_rows[:n_each] + e_rows[:n_each]
            chosen_auth = auth[:2 * n_each]
        else:
            n = min(len(auth), len(positives))
            if n == 0:
                raise ValueError(f"No balanced class pool in {kind}/{split}")
            rng.shuffle(positives)
            chosen_pos, chosen_auth = positives[:n], auth[:n]
        selected.extend(chosen_pos)
        selected.extend(chosen_auth)
    rng.shuffle(selected)
    return selected


def build_one(kind: str, positives: list[dict], authentic: list[dict],
              out_dir: Path, seed: int, strict_child_test: bool) -> dict:
    rows = [dict(r) for r in positives + authentic]
    # The source-family scaffold is grouped before balancing so no source
    # family crosses train/validation/test.
    benchmark.scaffold_split(rows, seed=seed)
    removed_test = remove_test_child_scaffold_overlap(rows) if strict_child_test else 0
    balanced = balance_splits(rows, kind, seed)
    for i, row in enumerate(balanced):
        row["id"] = f"row_{i:07d}"

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(balanced)

    counts = Counter((r["split"], r["label"]) for r in balanced)
    method_counts = Counter((r["split"], r["generation_method"] or "authentic")
                            for r in balanced)
    category_counts = Counter((r["split"], r["category"]) for r in balanced)
    scaffold_splits = defaultdict(set)
    for row in balanced:
        scaffold_splits[row["scaffold"]].add(row["split"])
    overlap = sum(len(splits) > 1 for splits in scaffold_splits.values())
    train_val_child_scaffolds = {r["own_scaffold"] for r in balanced
                                 if r["split"] in {"train", "val"}}
    test_rows = [r for r in balanced if r["split"] == "test"]
    test_child_overlap = sum(r["own_scaffold"] in train_val_child_scaffolds for r in test_rows)
    summary = {
        "dataset_version": kind,
        "label_semantics": {"0": "quality-filtered public reference molecule; not necessarily an approved medicine", "1": "generated analogue; not confirmed counterfeit"},
        "generator_policy": "pure M or pure E provenance; shared M/E products excluded; category-conditioned Version GA is an E subtype",
        "combined_positive_policy": "equal M and E counts within each split",
        "seed": seed, "rows": len(balanced),
        "class_counts": dict(Counter(r["label"] for r in balanced)),
        "split_label_counts": {
            split: {label: counts[split, label] for label in ("0", "1")}
            for split in ("train", "val", "test")
        },
        "split_method_counts": {
            split: {method: method_counts[split, method]
                    for method in ("authentic", "M", "E") if method_counts[split, method]}
            for split in ("train", "val", "test")
        },
        "split_category_counts": {
            split: {category: category_counts[split, category]
                    for category in sorted({c for s, c in category_counts if s == split})}
            for split in ("train", "val", "test")
        },
        "test_rows_removed_for_child_scaffold_overlap": removed_test,
        "test_rows_with_child_scaffold_overlap_train_val": test_child_overlap,
        "test_child_scaffold_overlap_fraction": test_child_overlap / len(test_rows) if test_rows else 0.0,
        "source_scaffold_overlap_across_splits": overlap,
        "split_definition": (
            "80/10/10 source-parent Bemis-Murcko scaffold groups; child-scaffold-overlapping test rows removed before per-split balancing"
            if strict_child_test else
            "80/10/10 source-parent Bemis-Murcko scaffold groups; child scaffolds may repeat between test and train/validation"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset_400k"))
    parser.add_argument("--ga-root", type=Path, default=Path("version_ga"))
    parser.add_argument("--authentic-extra", type=Path,
                        default=Path("molecule_bank_500k/chembl_controls.csv"))
    parser.add_argument("--min-per-class", type=int, default=100000,
                        help="Require at least this many rows in each class for every version")
    parser.add_argument("--out", type=Path,
                        default=Path("balanced_datasets/generator_versions_200k"))
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--keep-child-scaffold-overlap-in-test", action="store_true")
    args = parser.parse_args()

    authentic = authentic_candidates(args.bank, args.authentic_extra)
    authentic_smiles = {r["smiles"] for r in authentic}
    m_pool = generated_candidates(args.bank, "M", authentic_smiles, None)
    e_pool = generated_candidates(args.bank, "E", authentic_smiles, args.ga_root)
    if len(m_pool) < len(authentic) or len(e_pool) < len(authentic):
        raise ValueError("A generator pool is smaller than the authentic eligible pool")
    combined_pool = m_pool + e_pool

    pools = {"generator_m": m_pool, "generator_e": e_pool, "combined": combined_pool}
    summaries = {}
    for i, (name, pool) in enumerate(pools.items()):
        summaries[name] = build_one(
            name, pool, authentic, args.out / name, args.seed + i,
            strict_child_test=not args.keep_child_scaffold_overlap_in_test)
        class_counts = summaries[name]["class_counts"]
        if min(int(class_counts.get(label, 0)) for label in ("0", "1")) < args.min_per_class:
            raise ValueError(f"{name} produced fewer than {args.min_per_class:,} rows in a class: {class_counts}")
    run_summary = {
        "bank": str(args.bank), "output": str(args.out), "seed": args.seed,
        "authentic_eligible_unique": len(authentic),
        "pure_M_generated_candidates": len(m_pool),
        "pure_E_generated_candidates": len(e_pool),
        "ga_E_candidates_included": sum(
            r["generator_variant"] == "Version E category-conditioned GA" for r in e_pool),
        "public_reference_candidates": len(authentic),
        "shared_method_rows_excluded": sum(1 for r in read_csv(args.bank / "molecules.csv")
                                            if r.get("methods") == "EM"),
        "versions": summaries,
    }
    (args.out / "summary_all.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
