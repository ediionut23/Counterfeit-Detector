"""Replace ordinary training negatives with source-MMP near-analogue controls.

The validation and held-out test rows remain fixed. This is an exploratory
ablation of negative sampling, not proof of external generalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

from build_mmp_hard_control_challenge import Components

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("research_runs/mmp_dedicated_1000/dataset.csv"))
    parser.add_argument("--eligible", type=Path,
                        default=Path("final_dataset/authentic_eligible.csv"))
    parser.add_argument("--mmpdb", type=Path,
                        default=Path("mmp_mining_big/mined.mmpdb"))
    parser.add_argument("--relations", type=Path,
                        default=Path("final_dataset/relations.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("research_runs/mmp_dedicated_hard_train"))
    parser.add_argument("--max-hard", type=int, default=500)
    args = parser.parse_args()

    rows = read_csv(args.dataset)
    components = Components()
    with args.relations.open(newline="") as stream:
        for relation in csv.DictReader(stream):
            components.union(relation["parent_smiles"], relation["smiles"])
    val_test = [r for r in rows if r["split"] in ("val", "test")]
    blocked_roots = {components.root(r["smiles"]) for r in val_test}
    blocked_roots.update(components.root(r["parent_smiles"]) for r in val_test
                         if r["label"] == "1")
    blocked_scaffolds = {s for r in val_test for s in
                         (r.get("scaffold"), r.get("own_scaffold")) if s}
    existing_smiles = {r["smiles"] for r in rows}
    train_parents = {canonical(r["parent_smiles"]) for r in rows
                     if r["split"] == "train" and r["label"] == "1"}
    eligible = {canonical(r["smiles"]): r for r in read_csv(args.eligible)}
    eligible.pop(None, None)
    db = sqlite3.connect(f"file:{args.mmpdb.resolve()}?mode=ro", uri=True)
    compounds = {identifier: canonical(smiles) for identifier, smiles in
                 db.execute("SELECT id, clean_smiles FROM compound")}
    candidates = set()
    for first_id, second_id in db.execute("SELECT compound1_id, compound2_id FROM pair"):
        first, second = compounds[first_id], compounds[second_id]
        if first in train_parents and second in eligible:
            candidates.add(second)
        if second in train_parents and first in eligible:
            candidates.add(first)
    db.close()
    valid = sorted(smiles for smiles in candidates if
                   smiles not in existing_smiles
                   and eligible[smiles]["child_scaffold"] not in blocked_scaffolds
                   and components.root(smiles) not in blocked_roots)
    valid = valid[:args.max_hard]
    train_negatives = [r for r in rows if r["split"] == "train" and r["label"] == "0"]
    unused = {r["id"]: r for r in train_negatives}
    substitutions = []
    for smiles in valid:
        candidate = eligible[smiles]
        if not unused:
            break

        def distance(row):
            return (abs(float(row["mw"]) - float(candidate["mw"])) / 50
                    + abs(float(row["logp"]) - float(candidate["logp"]))
                    + abs(int(row["heavy_atoms"]) - int(candidate["heavy_atoms"])) / 5)

        replaced = min(unused.values(), key=lambda r: (distance(r), r["id"]))
        unused.pop(replaced["id"])
        new_row = dict(replaced)
        new_row.update({"id": "hard_mmp_" + hashlib.sha256(smiles.encode()).hexdigest()[:16],
                        "smiles": smiles, "label": "0", "split": "train",
                        "generation_method": "", "category": "source_mmp",
                        "subcategory": "near_training_parent", "parent_smiles": "",
                        "source_relation_id": "", "edited_atoms": "", "edit_status": "",
                        "scaffold": candidate["child_scaffold"],
                        "own_scaffold": candidate["child_scaffold"],
                        "mw": candidate["mw"], "logp": candidate["logp"],
                        "qed": candidate["qed"], "heavy_atoms": candidate["heavy_atoms"],
                        "smiles_length": candidate["smiles_length"],
                        "pair_id": "", "origin": "hard_mmp_train"})
        substitutions.append((replaced, new_row, distance(replaced)))

    removed_ids = {old["id"] for old, _, _ in substitutions}
    new_rows = [r for r in rows if r["id"] not in removed_ids]
    new_rows.extend(new for _, new, _ in substitutions)
    if len(new_rows) != len(rows):
        raise ValueError("Replacement changed dataset size")
    for split in ("train", "val", "test"):
        before = Counter(r["label"] for r in rows if r["split"] == split)
        after = Counter(r["label"] for r in new_rows if r["split"] == split)
        if before != after:
            raise ValueError(f"Label counts changed in {split}")
    if [r for r in rows if r["split"] != "train"] != [r for r in new_rows if r["split"] != "train"]:
        raise ValueError("Validation or test rows changed")
    for row in new_rows:
        if row["split"] == "train" and (components.root(row["smiles"]) in blocked_roots
                                        or row["own_scaffold"] in blocked_scaffolds
                                        or row["scaffold"] in blocked_scaffolds):
            raise ValueError("Hard-negative training row leaks into validation or test")

    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.out / "dataset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(new_rows)
    challenge_source = args.dataset.parent / "challenge.csv"
    challenge_dest = args.out / "challenge.csv"
    challenge_dest.write_bytes(challenge_source.read_bytes())
    summary = {
        "base_dataset": str(args.dataset),
        "base_dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((args.out / "dataset.csv").read_bytes()).hexdigest(),
        "challenge_sha256": hashlib.sha256(challenge_dest.read_bytes()).hexdigest(),
        "candidate_controls_near_training_parents": len(candidates),
        "eligible_after_val_test_leakage_filters": len(valid),
        "replaced_training_negatives": len(substitutions),
        "mean_replacement_descriptor_distance": round(
            sum(distance for _, _, distance in substitutions) / len(substitutions), 4)
            if substitutions else None,
        "validation_and_test_unchanged": True,
        "interpretation": "Exploratory hard-negative replacement; source-MMP controls are not clinically verified negatives.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
