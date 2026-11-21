"""Create an exactly label-balanced index over available packed 3D graphs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("balanced_datasets/training_200k/3d"))
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    root = args.root
    main_path = root / "summary_3d.json"
    report = json.loads(main_path.read_text())

    archives = [(Path(item["graph_file"]), item["split"]) for item in report["shards"]]
    retry_path = root / "retries" / "graphs_3d.npz"
    if retry_path.exists():
        archives.append((retry_path, "retry"))

    counts: Counter[tuple[str, int]] = Counter()
    candidate_ids: dict[tuple[str, int], list[str]] = defaultdict(list)
    seen: set[str] = set()
    for path, default_split in archives:
        with np.load(path, allow_pickle=False) as data:
            ids = data["source_id"]
            labels = data["label"]
            splits = data["split"] if "split" in data else np.full(len(labels), default_split)
            for source_id, label, split in zip(ids, labels, splits):
                source_id = str(source_id)
                split, label = str(split), int(label)
                if source_id in seen:
                    raise ValueError(f"duplicate molecule graph: {source_id}")
                seen.add(source_id)
                counts[(split, label)] += 1
                candidate_ids[(split, label)].append(source_id)

    excluded: set[str] = set()
    rng = random.Random(args.seed)
    for split in ("train", "val", "test"):
        target = min(counts[(split, 0)], counts[(split, 1)])
        for label in (0, 1):
            excess = counts[(split, label)] - target
            if excess:
                excluded.update(rng.sample(candidate_ids[(split, label)], excess))

    fields = ["split", "graph_file", "graph_index", "source_id", "molecule_id", "label",
              "generation_method", "category", "subcategory"]
    index_path = root / "balanced_graph_index.csv"
    final_counts: Counter[tuple[str, int]] = Counter()
    kept_ids: set[str] = set()
    with index_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for path, default_split in archives:
            with np.load(path, allow_pickle=False) as data:
                molecule_ids = data["molecule_id"]
                source_ids = data["source_id"]
                labels = data["label"]
                splits = data["split"] if "split" in data else np.full(len(labels), default_split)
                methods = data["generation_method"] if "generation_method" in data else np.full(len(labels), "")
                categories = data["category"] if "category" in data else np.full(len(labels), "")
                subcategories = data["subcategory"] if "subcategory" in data else np.full(len(labels), "")
                for i, label in enumerate(labels):
                    source_id = str(source_ids[i])
                    if source_id in excluded:
                        continue
                    if source_id in kept_ids:
                        raise ValueError(f"duplicate indexed molecule graph: {source_id}")
                    kept_ids.add(source_id)
                    split = str(splits[i])
                    label = int(label)
                    final_counts[(split, label)] += 1
                    writer.writerow({
                        "split": split, "graph_file": str(path.relative_to(root)),
                        "graph_index": i, "source_id": source_id,
                        "molecule_id": str(molecule_ids[i]), "label": label,
                        "generation_method": str(methods[i]), "category": str(categories[i]),
                        "subcategory": str(subcategories[i]),
                    })

    summary = {
        "available_graphs": len(seen),
        "graphs_after_exact_per_split_balance": len(kept_ids),
        "geometry_failures_from_source": report["requested"] - len(seen),
        "graphs_removed_to_balance": len(excluded),
        "counts_before_balance_by_split": {
            split: {str(label): counts[(split, label)] for label in (0, 1)}
            for split in ("train", "val", "test")
        },
        "balanced_counts_by_split": {
            split: {str(label): final_counts[(split, label)] for label in (0, 1)}
            for split in ("train", "val", "test")
        },
        "excluded_source_ids": sorted(excluded),
        "seed": args.seed,
        "note": "Index contains only unique graphs and has equal label counts within train, validation, and test.",
    }
    index_summary_path = root / "balanced_index_summary.json"
    index_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report["balanced_graph_index"] = index_path.name
    report["balanced_graph_index_rows"] = len(kept_ids)
    report["balanced_label_counts_by_split"] = summary["balanced_counts_by_split"]
    main_path.write_text(json.dumps(report, indent=2) + "\n")

    files = {}
    for path, _ in archives:
        shard_report = path.parent / "conformers_3d_report.json"
        for item in (path, shard_report):
            files[str(item.relative_to(root))] = {"bytes": item.stat().st_size, "sha256": sha256(item)}
    for item in (index_path, index_summary_path, main_path, root / "failed_embeddings.csv"):
        if item.exists():
            files[str(item.relative_to(root))] = {"bytes": item.stat().st_size, "sha256": sha256(item)}
    manifest = {
        "schema_version": 1, "root": str(root), "files": files,
        "note": "SDFs and logs are excluded; graph archives, reports, balanced index, summary, and failure list are checksummed.",
    }
    (root / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
