"""Add macro-subcategory loss weights while retaining all training molecules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


_E_CATEGORIES = {"bioisostere", "halogen-walk", "homologation", "scaffold-hop", "other"}


def balancing_key(row: dict[str, str]) -> tuple[str, str, str]:
    if row["label"] == "0":
        return "reference", "public_reference", "authentic"
    method = row["generation_method"]
    category = row["category"]
    subcategory = row["subcategory"]
    if method == "E":
        # Collapse generation-variant labels into their shared chemistry
        # category. Provenance remains in the original category/subcategory
        # and generator_variant columns.
        if category.startswith("evolved_"):
            category = category.removeprefix("evolved_")
        if category in _E_CATEGORIES:
            subcategory = category
        elif category == "unconditioned":
            subcategory = "unconditioned"
    return method, category, subcategory


def weight_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    negative_count = sum(row["label"] == "0" for row in rows)
    positive_by_group: Counter[tuple[str, str, str]] = Counter()
    groups_by_method: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    method_count: Counter[str] = Counter()
    for row in rows:
        if row["label"] != "1":
            continue
        method = row["generation_method"]
        key = balancing_key(row)
        positive_by_group[key] += 1
        groups_by_method[method].add(key)
        method_count[method] += 1

    # Give M and E half of the positive-class total weight, matching their
    # balanced counts in this combined benchmark. Within each family, make
    # every observed generator/category/subcategory group contribute equally.
    family_target = negative_count / 2.0
    output = []
    weight_sums = Counter()
    group_weight_sums = Counter()
    for row in rows:
        item = dict(row)
        item["balancing_category"] = balancing_key(row)[1]
        item["balancing_subcategory"] = balancing_key(row)[2]
        if row["label"] == "0":
            weight = 1.0
            group = "public_reference"
        else:
            method = row["generation_method"]
            key = balancing_key(row)
            n_groups = len(groups_by_method[method])
            weight = family_target / (n_groups * positive_by_group[key])
            group = "/".join(key)
        item["sample_weight"] = f"{weight:.12g}"
        output.append(item)
        weight_sums[(row["label"], row["generation_method"] or "reference")] += weight
        group_weight_sums[group] += weight

    summary = {
        "rows": len(output),
        "label_counts": dict(Counter(row["label"] for row in output)),
        "positive_groups_by_method": {
            method: {
                "subcategories": len(groups_by_method[method]),
                "molecules": method_count[method],
                "target_weight_total": family_target,
                "observed_weight_total": sum(
                    group_weight_sums["/".join(key)] for key in groups_by_method[method]
                ),
                "weight_per_subcategory": family_target / len(groups_by_method[method]),
            }
            for method in sorted(groups_by_method)
        },
        "weighted_totals_by_label_and_method": {
            f"label_{label}/{method}": round(value, 8)
            for (label, method), value in sorted(weight_sums.items())
        },
        "positive_subcategory_weight_range": {
            "min": min(group_weight_sums[k] for k in group_weight_sums if k != "public_reference"),
            "max": max(group_weight_sums[k] for k in group_weight_sums if k != "public_reference"),
        },
        "sample_weight_range": {
            "min": min(float(row["sample_weight"]) for row in output),
            "max": max(float(row["sample_weight"]) for row in output),
        },
        "formula": "label 0 weight=1; each positive M/E family receives half the total reference weight, divided equally across its harmonized chemistry groups and then across molecules in each group",
        "E_taxonomy_harmonization": "Version E category-conditioned and Version E category-conditioned GA labels are merged into their shared broad chemistry category; unconditioned and size-stratified E rows are one unconditioned group. Original provenance fields are retained.",
        "caveat": "Rare groups receive larger per-molecule weights. Weighting balances the loss contribution, not the number of unique molecules, and cannot create information absent from small groups.",
    }
    return output, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-splits", type=Path,
                        default=Path("balanced_datasets/generator_versions_200k/combined/dataset.csv"))
    parser.add_argument("--graph-index", type=Path,
                        default=Path("balanced_datasets/training_200k/3d/balanced_graph_index.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("balanced_datasets/training_200k/subcategory_weighted/dataset.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("balanced_datasets/training_200k/subcategory_weighting_summary.json"))
    args = parser.parse_args()

    with args.all_splits.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    train_rows = [row for row in all_rows if row["split"] == "train"]
    weighted_train, train_summary = weight_rows(train_rows)
    weight_by_id = {row["molecule_id"]: row["sample_weight"] for row in weighted_train}
    weighted_all = []
    for row in all_rows:
        item = dict(row)
        _, item["balancing_category"], item["balancing_subcategory"] = balancing_key(row)
        item["sample_weight"] = weight_by_id.get(row["molecule_id"], "1")
        weighted_all.append(item)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, weighted_all)

    graph_rows = []
    with args.graph_index.open(newline="") as stream:
        graph_rows = list(csv.DictReader(stream))
    graph_train = [row for row in graph_rows if row["split"] == "train"]
    weighted_graph_train, graph_summary = weight_rows(graph_train)
    graph_weight_by_id = {row["source_id"]: row["sample_weight"] for row in weighted_graph_train}
    graph_fields = list(graph_rows[0])
    for field in ("balancing_category", "balancing_subcategory", "sample_weight"):
        if field not in graph_fields:
            graph_fields.append(field)
    for row in graph_rows:
        _, row["balancing_category"], row["balancing_subcategory"] = balancing_key(row)
        row["sample_weight"] = graph_weight_by_id.get(row["source_id"], "1")
    with args.graph_index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=graph_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(graph_rows)

    summary = {
        "method": "macro-subcategory sample weighting",
        "2d_dataset": {
            "input": str(args.all_splits), "output": str(args.out),
            "rows_all_splits": len(weighted_all),
            "rows_with_nonunit_weights": sum(float(row["sample_weight"]) != 1.0 for row in weighted_all),
            "validation_test_weight": 1.0,
            **train_summary,
        },
        "3d_train": {"graph_index": str(args.graph_index), **graph_summary,
                     "validation_test_weight": 1.0},
        "evaluation_policy": "Validation and test rows are not reweighted; report overall and per-subcategory metrics.",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    local_summary = args.out.parent / "summary.json"
    local_summary.write_text(json.dumps(summary, indent=2) + "\n")
    weighted_manifest = {
        "schema_version": 1,
        "root": str(args.out.parent),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (args.out, local_summary)
        },
    }
    (args.out.parent / "manifest_sha256.json").write_text(
        json.dumps(weighted_manifest, indent=2) + "\n"
    )

    # Refresh the graph manifest after adding per-row sample weights.
    graph_root = args.graph_index.parent
    manifest_path = graph_root / "manifest_sha256.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for path in (args.graph_index, graph_root / "balanced_index_summary.json", graph_root / "summary_3d.json", graph_root / "failed_embeddings.csv"):
            if path.exists():
                manifest["files"][str(path.relative_to(graph_root))] = {
                    "bytes": path.stat().st_size, "sha256": sha256(path)
                }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "dataset_csv_rows": len(weighted_all),
        "train_rows": train_summary["rows"],
        "train_subcategories": sum(v["subcategories"] for v in train_summary["positive_groups_by_method"].values()),
        "graph_index_rows": len(graph_rows),
        "weighted_totals_by_label_and_method": train_summary["weighted_totals_by_label_and_method"],
    }, indent=2))


if __name__ == "__main__":
    main()
