"""Build parent-matched source-MMP controls for an existing held-out test split.

Controls have catalogue-source provenance, not verified biological inactivity or
medicine authenticity. The local MMP database helped mine M rules, so this is
an internal stress test rather than an external validation set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")
FINGERPRINTER = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)


class Components:
    def __init__(self):
        self.parent = {}

    def root(self, item):
        if item not in self.parent:
            self.parent[item] = item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        first, second = self.root(first), self.root(second)
        if first != second:
            self.parent[second] = first


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return FINGERPRINTER.GetFingerprint(mol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("final_dataset/examples/mixed_20k/dataset.csv"))
    parser.add_argument("--positive-relations", type=Path, default=None,
                        help="Use held-out relations from the full bank instead of only dataset test positives")
    parser.add_argument("--molecules", type=Path, default=Path("final_dataset/molecules.csv"))
    parser.add_argument("--component-relations", type=Path,
                        default=Path("final_dataset/relations.csv"),
                        help="All parent-child edges used to prevent connected-component leakage")
    parser.add_argument("--unassigned-cohort", action="store_true",
                        help="Select pairs before constructing a dedicated train/validation split")
    parser.add_argument("--eligible", type=Path,
                        default=Path("final_dataset/authentic_eligible.csv"))
    parser.add_argument("--mmpdb", type=Path,
                        default=Path("mmp_mining_big/mined.mmpdb"))
    parser.add_argument("--out", type=Path,
                        default=Path("research_runs/mmp_hard_control_pilot"))
    parser.add_argument("--max-sim-gap", type=float, default=0.15)
    parser.add_argument("--max-mw-gap", type=float, default=60.0)
    parser.add_argument("--max-logp-gap", type=float, default=1.25)
    parser.add_argument("--max-heavy-gap", type=int, default=6)
    args = parser.parse_args()
    if args.unassigned_cohort and not args.positive_relations:
        parser.error("--unassigned-cohort requires --positive-relations")

    rows = read_csv(args.dataset)
    source_splits = (() if args.unassigned_cohort else
                     ("train", "val") if args.positive_relations else ("train", "val", "test"))
    seen_molecules = {canonical(r["smiles"]) for r in rows if r["split"] in source_splits}
    train_val_scaffolds = {r["own_scaffold"] for r in rows
                           if not args.unassigned_cohort and r["split"] in ("train", "val")}
    train_val_parent_scaffolds = {r["scaffold"] for r in rows
                                  if not args.unassigned_cohort and r["split"] in ("train", "val")}
    components = Components()
    with args.component_relations.open(newline="") as stream:
        for relation in csv.DictReader(stream):
            components.union(relation["parent_smiles"], relation["smiles"])
    training_components = {components.root(r["smiles"]) for r in rows
                           if not args.unassigned_cohort and r["split"] in ("train", "val")}
    training_components.update(components.root(r["parent_smiles"]) for r in rows
                               if not args.unassigned_cohort and r["split"] in ("train", "val")
                               and r["label"] == "1")
    if args.positive_relations:
        positive_molecules = {r["smiles"]: r for r in read_csv(args.molecules)}
        test_pos = []
        for relation in read_csv(args.positive_relations):
            if (relation["smiles"] in seen_molecules
                    or relation["child_scaffold"] in train_val_scaffolds
                    or relation["parent_scaffold"] in train_val_parent_scaffolds
                    or components.root(relation["parent_smiles"]) in training_components):
                continue
            test_pos.append({"id": relation["relation_id"], "smiles": relation["smiles"],
                             "parent_smiles": relation["parent_smiles"],
                             "generation_method": relation["method"],
                             "category": relation["category"],
                             "parent_scaffold": relation["parent_scaffold"],
                             "own_scaffold": relation["child_scaffold"],
                             "mw": positive_molecules[relation["smiles"]]["mw"],
                             "logp": positive_molecules[relation["smiles"]]["logp"],
                             "heavy_atoms": positive_molecules[relation["smiles"]]["heavy_atoms"]})
    else:
        test_pos = [r for r in rows if r["split"] == "test" and r["label"] == "1"
                    and components.root(r["parent_smiles"]) not in training_components]
        for row in test_pos:
            row["parent_scaffold"] = row["scaffold"]
    eligible = {canonical(r["smiles"]): r for r in read_csv(args.eligible)}
    eligible.pop(None, None)
    parent_rows = defaultdict(list)
    for row in test_pos:
        parent_rows[canonical(row["parent_smiles"])].append(row)

    db = sqlite3.connect(f"file:{args.mmpdb.resolve()}?mode=ro", uri=True)
    compounds = {identifier: canonical(smiles) for identifier, smiles in
                 db.execute("SELECT id, clean_smiles FROM compound")}
    parents = set(parent_rows)
    candidates = defaultdict(set)
    for first_id, second_id in db.execute("SELECT compound1_id, compound2_id FROM pair"):
        first, second = compounds[first_id], compounds[second_id]
        if first in parents and second in eligible:
            candidates[first].add(second)
        if second in parents and first in eligible:
            candidates[second].add(first)
    db.close()

    fp_cache = {}

    def fp(smiles):
        if smiles not in fp_cache:
            fp_cache[smiles] = fingerprint(smiles)
        return fp_cache[smiles]

    edges = []
    rejection = Counter()
    for parent in sorted(candidates):
        parent_fp = fp(parent)
        for control in sorted(candidates[parent]):
            reference = eligible[control]
            if components.root(control) in training_components:
                rejection["component_overlap"] += 1
                continue
            if control in seen_molecules or reference["child_scaffold"] in train_val_scaffolds:
                rejection["seen_or_scaffold_overlap"] += 1
                continue
            control_similarity = DataStructs.TanimotoSimilarity(parent_fp, fp(control))
            for positive in parent_rows[parent]:
                positive_similarity = DataStructs.TanimotoSimilarity(parent_fp, fp(positive["smiles"]))
                gaps = {
                    "sim": abs(control_similarity - positive_similarity),
                    "mw": abs(float(reference["mw"]) - float(positive["mw"])),
                    "logp": abs(float(reference["logp"]) - float(positive["logp"])),
                    "heavy": abs(int(reference["heavy_atoms"]) - int(positive["heavy_atoms"])),
                }
                if (gaps["sim"] > args.max_sim_gap or gaps["mw"] > args.max_mw_gap
                        or gaps["logp"] > args.max_logp_gap
                        or gaps["heavy"] > args.max_heavy_gap):
                    rejection["outside_match_tolerance"] += 1
                    continue
                cost = (gaps["sim"] / args.max_sim_gap
                        + gaps["mw"] / args.max_mw_gap
                        + gaps["logp"] / args.max_logp_gap
                        + gaps["heavy"] / args.max_heavy_gap)
                edges.append((cost, parent, positive["id"], control,
                              positive, reference, positive_similarity,
                              control_similarity, gaps))

    # One positive and one control per parent; each source molecule is used
    # once. This makes the parent the sampling unit in the pilot.
    used_parents, used_controls, used_positives, used_components, selected = set(), set(), set(), set(), []
    for edge in sorted(edges, key=lambda x: (x[0], x[1], x[2], x[3])):
        _, parent, _, control = edge[:4]
        if (parent in used_parents or control in used_controls
                or components.root(parent) in used_components
                or components.root(control) in used_components
                or edge[4]["smiles"] in used_positives):
            continue
        used_parents.add(parent)
        used_controls.add(control)
        used_positives.add(edge[4]["smiles"])
        used_components.update((components.root(parent), components.root(control)))
        selected.append(edge)

    output = []
    for cost, parent, _, control, positive, reference, pos_sim, neg_sim, gaps in selected:
        pair_id = hashlib.sha256((positive["id"] + "|" + control).encode()).hexdigest()[:16]
        shared = {"pair_id": pair_id, "parent_smiles": parent,
                  "generation_method": positive["generation_method"],
                  "category": positive["category"],
                  "parent_scaffold": positive["parent_scaffold"],
                  "positive_similarity": round(pos_sim, 6),
                  "control_similarity": round(neg_sim, 6),
                  "match_cost": round(cost, 6),
                  "sim_gap": round(gaps["sim"], 6),
                  "mw_gap": round(gaps["mw"], 4),
                  "logp_gap": round(gaps["logp"], 4),
                  "heavy_gap": gaps["heavy"]}
        output.append({**shared, "id": positive["id"], "smiles": positive["smiles"],
                       "label": 1, "origin": "generated", "own_scaffold": positive["own_scaffold"]})
        output.append({**shared, "id": "mmp_" + pair_id, "smiles": control,
                       "label": 0, "origin": "source_mmp", "own_scaffold": reference["child_scaffold"]})

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id", "id", "smiles", "label", "origin", "parent_smiles",
              "generation_method", "category", "parent_scaffold", "own_scaffold", "positive_similarity",
              "control_similarity", "match_cost", "sim_gap", "mw_gap", "logp_gap", "heavy_gap"]
    with (args.out / "challenge.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "dataset": str(args.dataset), "positive_relations": str(args.positive_relations) if args.positive_relations else None,
        "eligible": str(args.eligible),
        "mmpdb": str(args.mmpdb), "component_relations": str(args.component_relations),
        "status": ("unassigned_cohort_requires_new_split" if args.unassigned_cohort
                   else "internal_pilot_not_external_validation"),
        "test_positives": len(test_pos), "test_positive_parents": len(parent_rows),
        "parents_with_eligible_mmp_control": len(candidates),
        "candidate_edges_within_tolerances": len(edges),
        "matched_parent_pairs": len(selected),
        "matched_pairs_by_method": dict(Counter(edge[4]["generation_method"] for edge in selected)),
        "rejections": dict(rejection),
        "tolerances": {"similarity": args.max_sim_gap, "mw": args.max_mw_gap,
                       "logp": args.max_logp_gap, "heavy_atoms": args.max_heavy_gap},
        "mean_absolute_gaps": {
            key: round(sum(edge[8][key] for edge in selected) / len(selected), 4)
            if selected else None for key in ("sim", "mw", "logp", "heavy")},
        "checks": {
            "unique_parents": len(used_parents) == len(selected),
            "unique_controls": len(used_controls) == len(selected),
            "unique_positive_molecules": len(used_positives) == len(selected),
            "no_training_molecule_overlap": all(r["smiles"] not in seen_molecules
                                                for r in output if r["label"] == 0),
            "no_training_scaffold_overlap": all(r["own_scaffold"] not in train_val_scaffolds
                                                for r in output),
            "no_training_parent_child_component_overlap": all(
                components.root(r["smiles"]) not in training_components
                and components.root(r["parent_smiles"]) not in training_components
                for r in output),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
