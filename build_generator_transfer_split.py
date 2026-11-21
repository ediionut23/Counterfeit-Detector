"""Construct an exclusive-M training / matched-M-versus-E test benchmark.

Only verified one-atom substitutions with the same elemental signature enter
the positive pools. The two test groups are matched within each signature on
basic descriptors and parent similarity. Labels denote generator provenance,
not clinical counterfeits or measured biological activity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neighbors import NearestNeighbors

from build_mmp_hard_control_challenge import Components
from exact_edit_context import edit_site_and_shell


FIELDS = ["id", "smiles", "label", "split", "generation_method", "signature",
          "category", "parent_smiles", "scaffold", "own_scaffold", "mw",
          "logp", "heavy_atoms", "smiles_length", "relation_id", "edited_atoms",
          "comparison_pair_id", "source_origin"]
PARENT_FINGERPRINTER = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def features(row: dict) -> np.ndarray:
    return np.array([float(row["mw"]) / 50, float(row["logp"]),
                     float(row["heavy_atoms"]) / 5,
                     float(row["smiles_length"]) / 10], dtype=float)


@lru_cache(maxsize=None)
def parent_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid parent SMILES: {smiles}")
    return PARENT_FINGERPRINTER.GetFingerprint(mol)


def parent_similarity(first: dict, second: dict) -> float:
    return float(DataStructs.TanimotoSimilarity(
        parent_fingerprint(first["parent_smiles"]),
        parent_fingerprint(second["parent_smiles"])))


def pair_cost(first: dict, second: dict, parent_fingerprint_weight: float = 0.0) -> float:
    descriptor_cost = float(np.abs(features(first) - features(second)).sum())
    similarity_cost = abs(float(first["similarity_to_parent"])
                          - float(second["similarity_to_parent"])) / 0.15
    parent_cost = (parent_fingerprint_weight * (1 - parent_similarity(first, second))
                   if parent_fingerprint_weight else 0.0)
    return descriptor_cost + similarity_cost + parent_cost


def row_scaffolds(row: dict) -> set[str]:
    return {s for s in (row.get("scaffold"), row.get("own_scaffold")) if s is not None}


def positive_record(row: dict, split: str, pair_id: str = "") -> dict:
    return {"id": row["relation_id"], "smiles": row["smiles"], "label": "1",
            "split": split, "generation_method": row["method"],
            "signature": row["signature"], "category": row["category"],
            "parent_smiles": row["parent_smiles"],
            "scaffold": row["parent_scaffold"],
            "own_scaffold": row["child_scaffold"],
            "mw": row["mw"], "logp": row["logp"],
            "heavy_atoms": row["heavy_atoms"],
            "smiles_length": row["smiles_length"],
            "relation_id": row["relation_id"],
            "edited_atoms": row["edited_atoms"],
            "comparison_pair_id": pair_id,
            "source_origin": "generated_exclusive_" + row["method"]}


def negative_record(row: dict, split: str) -> dict:
    return {"id": row["molecule_id"], "smiles": row["smiles"],
            "label": "0", "split": split, "generation_method": "",
            "signature": "", "category": "source", "parent_smiles": "",
            "scaffold": row["child_scaffold"],
            "own_scaffold": row["child_scaffold"],
            "mw": row["mw"], "logp": row["logp"],
            "heavy_atoms": row["heavy_atoms"],
            "smiles_length": row["smiles_length"],
            "relation_id": "", "edited_atoms": "", "comparison_pair_id": "",
            "source_origin": "eligible_source"}


def choose_negatives(split: str, targets: list[dict], source: list[dict],
                     n_desired: int, components: Components,
                     positive_roots: dict[str, set[str]],
                     positive_scaffolds: dict[str, set[str]],
                     used_molecules: set[str], blocked_roots: set[str],
                     blocked_scaffolds: set[str]) -> tuple[list[dict], list[float]]:
    other = {name for name in ("train", "val", "test") if name != split}
    excluded_roots = set().union(*(positive_roots[name] for name in other), blocked_roots)
    excluded_scaffolds = set().union(*(positive_scaffolds[name] for name in other),
                                     blocked_scaffolds)
    available = [row for row in source if row["smiles"] not in used_molecules
                 and components.root(row["smiles"]) not in excluded_roots
                 and row["child_scaffold"] not in excluded_scaffolds]
    if len(available) < n_desired:
        raise ValueError(f"Only {len(available)} candidate negatives for {split}")
    matrix = np.stack([features(row) for row in available])
    target_matrix = np.stack([features(targets[i % len(targets)])
                              for i in range(n_desired)])
    neighbors = NearestNeighbors(n_neighbors=min(500, len(available)), algorithm="ball_tree")
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(target_matrix)
    chosen, costs = [], []
    used_here = set()
    for target_index in range(n_desired):
        for distance, index in zip(distances[target_index], indices[target_index]):
            row = available[int(index)]
            if row["smiles"] in used_here:
                continue
            used_here.add(row["smiles"])
            chosen.append(negative_record(row, split))
            costs.append(float(distance))
            break
    if len(chosen) < n_desired:
        raise ValueError(f"Only {len(chosen)}/{n_desired} unique negatives found for {split}")
    return chosen, costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("research_runs/shared_exact_edit_relations.csv"))
    parser.add_argument("--signature-audit", type=Path,
                        default=Path("research_runs/shared_exact_edit_signatures.json"))
    parser.add_argument("--test-per-signature", type=int, default=40)
    parser.add_argument("--val-per-signature", type=int, default=20)
    parser.add_argument("--min-m", type=int, default=300)
    parser.add_argument("--min-e", type=int, default=100)
    parser.add_argument("--parent-fingerprint-weight", type=float, default=0.0,
                        help="Penalty for chemically dissimilar parents in M/E test matching")
    parser.add_argument("--require-matched-local-shell", action="store_true",
                        help="Require identical edited-atom first-shell contexts in M/E test pairs")
    parser.add_argument("--test-negatives", type=int, default=1000)
    parser.add_argument("--val-negatives", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--out", type=Path,
                        default=Path("research_runs/generator_transfer_exact"))
    args = parser.parse_args()
    if args.parent_fingerprint_weight < 0:
        parser.error("--parent-fingerprint-weight must be nonnegative")

    audit = json.loads(args.signature_audit.read_text())
    signatures = sorted(row["signature"] for row in audit["signatures"]
                        if "∅" not in row["signature"]
                        and row["M_children"] >= args.min_m
                        and row["E_children"] >= args.min_e)
    if not signatures:
        raise ValueError("No shared substitution signatures satisfy minimum counts")
    generated_props = {row["smiles"]: row for row in read_csv(args.bank / "molecules.csv")}
    source = read_csv(args.bank / "authentic_eligible.csv")
    manifest = sorted((r for r in read_csv(args.manifest) if r["signature"] in signatures),
                      key=lambda r: r["relation_id"])
    by_key = defaultdict(list)
    seen_children = set()
    for row in manifest:
        if row["smiles"] in seen_children:
            continue
        seen_children.add(row["smiles"])
        props = generated_props[row["smiles"]]
        row.update({field: props[field] for field in
                    ("mw", "logp", "heavy_atoms", "smiles_length")})
        if args.require_matched_local_shell:
            row["_local_shell"] = edit_site_and_shell(row)[1]
        by_key[row["signature"], row["method"]].append(row)

    components = Components()
    with (args.bank / "relations.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            components.union(row["parent_smiles"], row["smiles"])
    rng = random.Random(args.seed)
    used_test_roots = set()
    test_m, test_e, comparison = [], [], []
    for sig in signatures:
        e_pool = list(by_key[sig, "E"])
        m_pool = list(by_key[sig, "M"])
        rng.shuffle(e_pool)
        for e in e_pool:
            if sum(row["signature"] == sig for row in test_e) >= args.test_per_signature:
                break
            e_root = components.root(e["smiles"])
            if e_root in used_test_roots:
                continue
            options = [m for m in m_pool if components.root(m["smiles"]) not in used_test_roots
                       and components.root(m["smiles"]) != e_root
                       and (not args.require_matched_local_shell
                            or m["_local_shell"] == e["_local_shell"])]
            if not options:
                continue
            m = min(options, key=lambda row: (
                pair_cost(e, row, args.parent_fingerprint_weight), row["relation_id"]))
            used_test_roots.update((e_root, components.root(m["smiles"])))
            pair_id = hashlib.sha256((e["relation_id"] + "|" + m["relation_id"]).encode()).hexdigest()[:16]
            test_e.append(positive_record(e, "test", pair_id))
            test_m.append(positive_record(m, "test", pair_id))
            comparison.append({"pair_id": pair_id, "signature": sig,
                               "E_relation_id": e["relation_id"],
                               "M_relation_id": m["relation_id"],
                               "descriptor_and_similarity_cost": pair_cost(e, m)})
        if sum(row["signature"] == sig for row in test_e) != args.test_per_signature:
            raise ValueError(f"Could not fill E/M test quota for {sig}")

    test_pos = test_e + test_m
    test_roots = {components.root(r["smiles"]) for r in test_pos}
    test_scaffolds = set().union(*(row_scaffolds(r) for r in test_pos))
    remaining_m = [row for sig in signatures for row in by_key[sig, "M"]
                   if components.root(row["smiles"]) not in test_roots
                   and row["parent_scaffold"] not in test_scaffolds
                   and row["child_scaffold"] not in test_scaffolds]
    rng.shuffle(remaining_m)
    val_selected = []
    val_roots = set()
    for sig in signatures:
        choices = [r for r in remaining_m if r["signature"] == sig]
        for row in choices:
            root = components.root(row["smiles"])
            if root in val_roots:
                continue
            val_selected.append(row)
            val_roots.add(root)
            if sum(r["signature"] == sig for r in val_selected) >= args.val_per_signature:
                break
        if sum(r["signature"] == sig for r in val_selected) != args.val_per_signature:
            raise ValueError(f"Could not fill M validation quota for {sig}")
    val_pos = [positive_record(row, "val") for row in val_selected]
    val_scaffolds = set().union(*(row_scaffolds(r) for r in val_pos))
    train_candidates = [row for row in remaining_m
                        if components.root(row["smiles"]) not in val_roots
                        and row["parent_scaffold"] not in val_scaffolds
                        and row["child_scaffold"] not in val_scaffolds]
    train_pos = [positive_record(row, "train") for row in train_candidates]
    if len(train_pos) < 1000:
        raise ValueError(f"Only {len(train_pos)} training positives after disjoint filters")

    pos = {"train": train_pos, "val": val_pos, "test": test_pos}
    positive_roots = {split: {components.root(row["smiles"]) for row in rows}
                      for split, rows in pos.items()}
    positive_scaffolds = {split: set().union(*(row_scaffolds(row) for row in rows))
                          for split, rows in pos.items()}
    negatives = {}
    match_costs = {}
    used_molecules = {row["smiles"] for rows in pos.values() for row in rows}
    blocked_roots, blocked_scaffolds = set(), set()
    for split, targets, count in (("test", test_pos, args.test_negatives),
                                  ("val", val_pos, args.val_negatives),
                                  ("train", train_pos, len(train_pos))):
        rows, costs = choose_negatives(split, targets, source, count, components,
                                       positive_roots, positive_scaffolds,
                                       used_molecules, blocked_roots, blocked_scaffolds)
        negatives[split] = rows
        match_costs[split] = costs
        used_molecules.update(row["smiles"] for row in rows)
        blocked_roots.update(components.root(row["smiles"]) for row in rows)
        blocked_scaffolds.update(row["own_scaffold"] for row in rows)

    dataset = []
    for split in ("train", "val", "test"):
        dataset.extend(pos[split] + negatives[split])
    if len({row["smiles"] for row in dataset}) != len(dataset):
        raise ValueError("Duplicate molecule across dataset rows")
    roots = {split: {components.root(r["smiles"]) for r in dataset if r["split"] == split}
             for split in ("train", "val", "test")}
    scaffolds = {split: set().union(*(row_scaffolds(r) for r in dataset if r["split"] == split))
                 for split in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if roots[left] & roots[right] or scaffolds[left] & scaffolds[right]:
            raise ValueError(f"Component or scaffold leakage between {left} and {right}")
    assert all(row["generation_method"] == "M" for row in train_pos + val_pos)
    assert all(row["generation_method"] == "E" for row in test_e)
    assert len(test_m) == len(test_e) == len(comparison)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(dataset)
    with (args.out / "matched_M_E_test.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)
    counts = Counter((r["split"], r["label"], r["generation_method"]) for r in dataset)
    summary = {
        "seed": args.seed, "signatures": signatures,
        "test_per_signature_per_method": args.test_per_signature,
        "val_per_signature": args.val_per_signature,
        "counts": {split: {"source_negative": counts[split, "0", ""],
                           "M_positive": counts[split, "1", "M"],
                           "E_positive": counts[split, "1", "E"]}
                   for split in ("train", "val", "test")},
        "matched_test_pairs": len(comparison),
        "parent_fingerprint_weight": args.parent_fingerprint_weight,
        "require_matched_local_shell": args.require_matched_local_shell,
        "mean_M_E_match_cost": round(float(np.mean([r["descriptor_and_similarity_cost"]
                                                    for r in comparison])), 4),
        "mean_negative_descriptor_distance": {split: round(float(np.mean(match_costs[split])), 4)
                                               for split in ("train", "val", "test")},
        "dataset_sha256": hashlib.sha256((args.out / "dataset.csv").read_bytes()).hexdigest(),
        "pair_manifest_sha256": hashlib.sha256((args.out / "matched_M_E_test.csv").read_bytes()).hexdigest(),
        "audit": "Exact molecules, parent-child components and parent/child scaffolds disjoint across splits; train/val positives M only; test M/E signatures paired."
                 + (" Test M/E first-shell contexts identical." if args.require_matched_local_shell else ""),
        "limitation": "Elemental substitution signature and descriptor matching do not fully equate edit site or chemical context; source negatives are not clinical controls.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
