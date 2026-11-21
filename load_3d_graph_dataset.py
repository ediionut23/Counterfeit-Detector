"""Iterate balanced 3D molecular graphs from the packed NPZ shards."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np


def iter_graphs(root: Path, split: str) -> Iterator[dict]:
    """Yield one graph at a time, with edge indices localized to its atoms.

    ``root`` is ``balanced_datasets/training_200k/3d``. Each yielded graph
    contains NumPy arrays suitable for conversion to a tensor library such as
    PyTorch Geometric. The balanced index excludes geometry failures and
    removes a small number of majority-class graphs per split.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    index_path = root / "balanced_graph_index.csv"
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    with index_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == split:
                by_file[row["graph_file"]].append(row)

    for relative_path in sorted(by_file):
        graph_path = root / relative_path
        with np.load(graph_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
            for row in by_file[relative_path]:
                graph_id = int(row["graph_index"])
                atom_start, atom_end = map(int, arrays["atom_ptr"][graph_id:graph_id + 2])
                edge_start, edge_end = map(int, arrays["edge_ptr"][graph_id:graph_id + 2])
                radius_start, radius_end = map(int, arrays["radius_ptr"][graph_id:graph_id + 2])
                angle_start, angle_end = map(int, arrays["angle_ptr"][graph_id:graph_id + 2])
                yield {
                    "source_id": row["source_id"],
                    "molecule_id": row["molecule_id"],
                    "label": int(row["label"]),
                    "sample_weight": float(row.get("sample_weight", 1.0)),
                    "split": split,
                    "generation_method": row["generation_method"],
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                    "balancing_category": row.get("balancing_category", row["category"]),
                    "balancing_subcategory": row.get("balancing_subcategory", row["subcategory"]),
                    "atom_features": arrays["atom_features"][atom_start:atom_end],
                    "positions_angstrom": arrays["positions_angstrom"][atom_start:atom_end],
                    "edge_index": arrays["edge_index"][:, edge_start:edge_end] - atom_start,
                    "edge_features": arrays["edge_features"][edge_start:edge_end],
                    "edge_distance_angstrom": arrays["edge_distance_angstrom"][edge_start:edge_end],
                    "radius_edge_index": arrays["radius_edge_index"][:, radius_start:radius_end] - atom_start,
                    "radius_edge_distance_angstrom": arrays["radius_edge_distance_angstrom"][radius_start:radius_end],
                    "angle_index": arrays["angle_index"][:, angle_start:angle_end] - atom_start,
                    "angle_radians": arrays["angle_radians"][angle_start:angle_end],
                }


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("balanced_datasets/training_200k/3d"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--limit", type=int, default=5, help="Number of graphs to summarize")
    args = parser.parse_args()
    counts = Counter()
    for i, graph in enumerate(iter_graphs(args.root, args.split)):
        counts[graph["label"]] += 1
        if i + 1 >= args.limit:
            break
    print({"split": args.split, "loaded_graphs": sum(counts.values()), "labels": dict(counts)})
