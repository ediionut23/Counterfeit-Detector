#!/usr/bin/env python3
"""Create exact hierarchical M/E and subcategory balanced 2D and 3D views.

The input bank is the existing scaffold-split combined Version M + Version E
dataset. E category-conditioned GA rows are merged with their matching broad E
chemistry group; unconditioned and size-stratified rows are merged together.
Rare M fine-grained groups below the configured support threshold are reported
and omitted. The source dataset and 3D shards are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def group_key(row: dict[str, str]) -> str:
    method, category, subcategory = row["generation_method"], row["category"], row["subcategory"]
    if method == "M":
        return f"M/{category}/{subcategory}"
    if method == "E":
        if category == "unconditioned" or subcategory in {"unconditioned", "size_stratified"}:
            return "E/unconditioned"
        # Both native Version E and category-conditioned GA use the same broad
        # chemistry categories; GA stores these with an `evolved_` prefix.
        normalized = category.removeprefix("evolved_")
        return f"E/{normalized}"
    raise ValueError(f"Unexpected generation method: {method!r}")


def balanced_sample(rows: list[dict[str, str]], support_min: int, seed: int):
    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    authentic_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["label"] == "0":
            authentic_by_split[row["split"]].append(row)
        else:
            by_group[group_key(row)].append(row)

    method_groups = {
        method: sorted(k for k, v in by_group.items() if k.startswith(method + "/") and len(v) >= support_min)
        for method in ("M", "E")
    }
    excluded = {k: len(v) for k, v in sorted(by_group.items()) if len(v) < support_min}
    n_m, n_e = len(method_groups["M"]), len(method_groups["E"])
    if not n_m or not n_e:
        raise ValueError(f"Need supported M and E groups; found {n_m} M and {n_e} E")
    divisor = math.gcd(n_m, n_e)
    m_step, e_step = n_e // divisor, n_m // divisor
    max_t = min(min(len(by_group[k]) for k in method_groups["M"]) // m_step,
                min(len(by_group[k]) for k in method_groups["E"]) // e_step)
    if max_t < 1:
        raise ValueError("Insufficient support for exact M/E and per-subcategory balance")
    per_group = {"M": m_step * max_t, "E": e_step * max_t}

    selected_fake = []
    selected_counts = {}
    for method in ("M", "E"):
        for key in method_groups[method]:
            options = list(by_group[key])
            rng.shuffle(options)
            chosen = options[:per_group[method]]
            selected_fake.extend(chosen)
            selected_counts[key] = len(chosen)

    fake_by_split = Counter(row["split"] for row in selected_fake)
    selected_auth = []
    for split, n in sorted(fake_by_split.items()):
        candidates = list(authentic_by_split[split])
        if len(candidates) < n:
            raise ValueError(f"Not enough authentic references in {split}: need {n}, have {len(candidates)}")
        rng.shuffle(candidates)
        selected_auth.extend(candidates[:n])
    selected = selected_fake + selected_auth
    rng.shuffle(selected)
    return selected, {
        "supported_m_groups": n_m,
        "supported_e_groups": n_e,
        "minimum_support_threshold": support_min,
        "selected_per_m_subcategory": per_group["M"],
        "selected_per_e_broad_category": per_group["E"],
        "generated_m_total": n_m * per_group["M"],
        "generated_e_total": n_e * per_group["E"],
        "authentic_total": len(selected_auth),
        "excluded_generated_groups_below_threshold": excluded,
        "generated_count_per_group": dict(sorted(selected_counts.items())),
        "split_counts": {
            split: {
                "authentic": sum(r["split"] == split and r["label"] == "0" for r in selected),
                "generated": sum(r["split"] == split and r["label"] == "1" for r in selected),
                "M": sum(r["split"] == split and r["label"] == "1" and r["generation_method"] == "M" for r in selected),
                "E": sum(r["split"] == split and r["label"] == "1" and r["generation_method"] == "E" for r in selected),
            }
            for split in ("train", "val", "test")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("balanced_datasets/generator_versions_200k/combined/dataset.csv"))
    parser.add_argument("--graph-index", type=Path, default=Path("balanced_datasets/training_200k/3d/balanced_graph_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("balanced_datasets/training_200k/subcategory_exact_balanced"))
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()

    source = read_csv(args.input)
    selected_2d, summary_2d = balanced_sample(source, args.min_support, args.seed)
    fields_2d = list(source[0])
    write_csv(args.out / "dataset.csv", selected_2d, fields_2d)

    # The 3D view is balanced independently over molecules that have a stored
    # graph/conformer entry. It points into the existing immutable shard files.
    row_by_molecule = {r["molecule_id"]: r for r in source}
    graph_rows = read_csv(args.graph_index)
    graph_source_ids = {r["source_id"] for r in graph_rows}
    available_3d = [r for r in source if r["molecule_id"] in graph_source_ids]
    selected_3d, summary_3d = balanced_sample(available_3d, args.min_support, args.seed + 1)
    selected_ids = {r["molecule_id"] for r in selected_3d}
    selected_graph_rows = [r for r in graph_rows if r["source_id"] in selected_ids]
    if len(selected_graph_rows) != len(selected_3d):
        raise ValueError("3D graph index has missing or duplicate source IDs for selected rows")
    write_csv(args.out / "3d" / "dataset.csv", selected_3d, list(source[0]))
    write_csv(args.out / "3d" / "graph_index.csv", selected_graph_rows, list(graph_rows[0]))

    summary = {
        "source_dataset": str(args.input),
        "source_graph_index": str(args.graph_index),
        "seed": args.seed,
        "label_semantics": {
            "0": "quality-filtered public reference molecule; not necessarily an approved medicine",
            "1": "generated analogue; not confirmed counterfeit",
        },
        "balance_definition": "equal physical rows per eligible M fine-grained subcategory; equal physical rows per harmonized E broad category; equal total generated M and E; authentic references equal total generated rows",
        "E_taxonomy": "evolved_<category> is merged with the matching E category; unconditioned and size_stratified are merged into E/unconditioned; original provenance columns remain unchanged",
        "2d": summary_2d,
        "3d": {
            **summary_3d,
            "available_graph_rows_in_source": len(available_3d),
            "selected_graph_rows": len(selected_graph_rows),
            "graph_files_are_references": "graph_index.csv points to the existing 3D shard files under balanced_datasets/training_200k/3d; coordinates were not regenerated or copied",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# Exact subcategory-balanced training set

This view is sampled from `{args.input}` with seed `{args.seed}`. It balances actual row counts: each eligible M fine-grained subcategory has **{summary_2d['selected_per_m_subcategory']}** molecules, each harmonized E broad chemistry group has **{summary_2d['selected_per_e_broad_category']}**, total M and E counts are equal, and public references match the total generated count.

The 2D set contains {summary_2d['generated_m_total']:,} M + {summary_2d['generated_e_total']:,} E generated molecules and {summary_2d['authentic_total']:,} reference molecules ({summary_2d['generated_m_total'] + summary_2d['generated_e_total'] + summary_2d['authentic_total']:,} total rows). Groups below {args.min_support} source examples are excluded and listed in `summary.json`.

The 3D subset is independently balanced among source molecules with stored 3D graph entries. Its graph index points to existing shard files; it does not regenerate coordinates. See `3d/graph_index.csv` and `summary.json` for counts. The existing scaffold-based train/validation/test assignment is preserved; exact balance is global, so individual split counts by subcategory can differ.

Labels are dataset-generation labels: label 0 means a quality-filtered public reference and label 1 means a generated analogue. These labels do not establish legal authenticity or counterfeiting.
"""
    (args.out / "README.md").write_text(readme)
    print(json.dumps({
        "output": str(args.out),
        "2d_rows": len(selected_2d),
        "2d_method_groups": [summary_2d["supported_m_groups"], summary_2d["supported_e_groups"]],
        "2d_per_group": [summary_2d["selected_per_m_subcategory"], summary_2d["selected_per_e_broad_category"]],
        "2d_method_totals": [summary_2d["generated_m_total"], summary_2d["generated_e_total"]],
        "3d_rows": len(selected_3d),
        "3d_method_groups": [summary_3d["supported_m_groups"], summary_3d["supported_e_groups"]],
        "3d_per_group": [summary_3d["selected_per_m_subcategory"], summary_3d["selected_per_e_broad_category"]],
        "3d_method_totals": [summary_3d["generated_m_total"], summary_3d["generated_e_total"]],
    }, indent=2))


if __name__ == "__main__":
    main()
