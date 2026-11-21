"""Reserve matched MMP pairs before fitting a new, leakage-audited baseline.

The source-MMP label is catalogue provenance, not clinical authenticity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_mmp_hard_control_challenge import Components


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose_pairs(rows: list[dict], n_pairs: int, seed: int) -> list[dict]:
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    if any(len(pair) != 2 or {r["label"] for r in pair} != {"0", "1"}
           for pair in by_pair.values()):
        raise ValueError("Unassigned cohort must have complete 1:1 pairs")
    by_method = defaultdict(list)
    for pair in by_pair.values():
        positive = next(r for r in pair if r["label"] == "1")
        by_method[positive["generation_method"]].append(pair)
    if set(by_method) != {"M", "E"}:
        raise ValueError("Expected both M and E analogs")
    rng = random.Random(seed)
    for pairs in by_method.values():
        rng.shuffle(pairs)
    per_method = min(n_pairs // 2, *(len(by_method[m]) for m in ("M", "E")))
    selected = by_method["M"][:per_method] + by_method["E"][:per_method]
    if n_pairs % 2:
        extra = (by_method["M"][per_method:] + by_method["E"][per_method:])
        if extra:
            selected.append(extra[0])
    rng.shuffle(selected)
    return [row for pair in selected for row in pair]


def row_components(row: dict, components: Components) -> set[str]:
    result = {components.root(row["smiles"])}
    if row["label"] == "1" and row.get("parent_smiles"):
        result.add(components.root(row["parent_smiles"]))
    return result


def row_scaffolds(row: dict) -> set[str]:
    return {s for s in (row.get("scaffold"), row.get("parent_scaffold"),
                        row.get("own_scaffold")) if s}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("final_dataset/examples/mixed_20k/dataset.csv"))
    parser.add_argument("--cohort", type=Path,
                        default=Path("research_runs/mmp_unassigned_cohort/challenge.csv"))
    parser.add_argument("--relations", type=Path,
                        default=Path("final_dataset/relations.csv"))
    parser.add_argument("--out", type=Path, default=Path("research_runs/mmp_dedicated_1000"))
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("--pairs must be at least 2")
    source = read_csv(args.source)
    cohort = read_csv(args.cohort)
    test = choose_pairs(cohort, args.pairs, args.seed)
    test_components = Components()
    with args.relations.open(newline="") as stream:
        for relation in csv.DictReader(stream):
            test_components.union(relation["parent_smiles"], relation["smiles"])
    components = test_components
    test_roots = set().union(*(row_components(row, components) for row in test))
    test_scaffolds = set().union(*(row_scaffolds(row) for row in test))
    test_smiles = {row["smiles"] for row in test}
    if len(test_smiles) != len(test):
        raise ValueError("Test cohort has duplicate molecules")

    rejection = Counter()
    pool = {"train": [], "val": []}
    for row in source:
        if row["split"] not in pool:
            continue
        reason = None
        if row["smiles"] in test_smiles or row_components(row, components) & test_roots:
            reason = "test_component_or_molecule"
        elif row_scaffolds(row) & test_scaffolds:
            reason = "test_scaffold"
        if reason:
            rejection[row["split"], reason] += 1
        else:
            pool[row["split"]].append(row)
    train_roots = set().union(*(row_components(r, components) for r in pool["train"]))
    train_scaffolds = set().union(*(row_scaffolds(r) for r in pool["train"]))
    validation = []
    for row in pool["val"]:
        if row_components(row, components) & train_roots:
            rejection["val", "train_component"] += 1
        elif row_scaffolds(row) & train_scaffolds:
            rejection["val", "train_scaffold"] += 1
        else:
            validation.append(row)

    rng = random.Random(args.seed)
    balanced = {}
    for split, available in (("train", pool["train"]), ("val", validation)):
        zero = [r for r in available if r["label"] == "0"]
        one = [r for r in available if r["label"] == "1"]
        rng.shuffle(zero)
        rng.shuffle(one)
        count = min(len(zero), len(one))
        balanced[split] = zero[:count] + one[:count]
        rng.shuffle(balanced[split])
    if len(balanced["val"]) < 200 or len(balanced["train"]) < 2000:
        raise ValueError("Insufficient train/validation rows after leakage filters")

    dataset = balanced["train"] + balanced["val"]
    for row in test:
        dataset.append({"id": row["id"], "smiles": row["smiles"],
                        "label": row["label"], "split": "test",
                        "generation_method": row["generation_method"] if row["label"] == "1" else "",
                        "category": row["category"] if row["label"] == "1" else "source_mmp",
                        "parent_smiles": row["parent_smiles"] if row["label"] == "1" else "",
                        "scaffold": row["parent_scaffold"],
                        "own_scaffold": row["own_scaffold"],
                        "pair_id": row["pair_id"], "origin": row["origin"]})
    # Independent final audit on retained rows, not on the larger input pools.
    split_roots = {split: set().union(*(row_components(r, components) for r in dataset
                                       if r["split"] == split)) for split in ("train", "val", "test")}
    split_scaffolds = {split: set().union(*(row_scaffolds(r) for r in dataset
                                           if r["split"] == split)) for split in ("train", "val", "test")}
    split_smiles = {split: {r["smiles"] for r in dataset if r["split"] == split}
                    for split in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if (split_roots[left] & split_roots[right]
                or split_scaffolds[left] & split_scaffolds[right]
                or split_smiles[left] & split_smiles[right]):
            raise ValueError(f"Leakage between {left} and {right}")

    args.out.mkdir(parents=True, exist_ok=True)
    fieldnames = list(source[0]) + ["pair_id", "origin"]
    write_csv(args.out / "dataset.csv", dataset, fieldnames)
    write_csv(args.out / "challenge.csv", test, list(cohort[0]))
    counts = Counter((r["split"], r["label"]) for r in dataset)
    summary = {
        "source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "cohort": str(args.cohort), "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((args.out / "dataset.csv").read_bytes()).hexdigest(),
        "challenge_sha256": hashlib.sha256((args.out / "challenge.csv").read_bytes()).hexdigest(),
        "relations": str(args.relations), "seed": args.seed,
        "requested_pairs": args.pairs, "selected_pairs": len(test) // 2,
        "selected_by_method": dict(Counter(r["generation_method"] for r in test if r["label"] == "1")),
        "split_label_counts": {split: {label: counts[split, label] for label in ("0", "1")}
                               for split in ("train", "val", "test")},
        "removed_rows": {f"{split}:{reason}": number
                         for (split, reason), number in sorted(rejection.items())},
        "audit": "Disjoint exact molecules, parent-child connected components, parent and own scaffolds across train/val/test.",
        "control_interpretation": "MMP source provenance; no external or clinical authenticity claim.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
