"""Build balanced evolved training sets with low/high counterfeit scaffold diversity.

Both variants use the same 3,700 authentic train rows and unchanged validation
and test rows. They match counterfeit counts in three heavy-atom size bins.
Only the selected counterfeit train molecules differ between variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx


def size_bin(row: dict) -> int:
    n = int(row["heavy_atoms"])
    return 0 if n < 20 else 1 if n < 30 else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/scaffold_diversity"))
    parser.add_argument("--count", type=int, default=3700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rng = random.Random(args.seed)
    authentic = [r for r in rows if r["split"] == "train" and r["label"] == "0"]
    fake = [r for r in rows if r["split"] == "train" and r["label"] == "1"]
    if len(authentic) < args.count or len(fake) < args.count:
        raise ValueError("Fewer training rows than requested")
    by_scaffold = defaultdict(list)
    for row in fake:
        by_scaffold[row["scaffold"]].append(row)
    if len(by_scaffold) < args.count:
        raise ValueError("Too few unique counterfeit scaffolds for high diversity")

    selected_auth = {row["id"] for row in rng.sample(authentic, args.count)}
    low = []
    for scaffold, group in sorted(by_scaffold.items(),
                                  key=lambda item: (-len(item[1]), item[0])):
        shuffled = group.copy()
        rng.shuffle(shuffled)
        low.extend(shuffled[:args.count - len(low)])
        if len(low) == args.count:
            break
    required = Counter(size_bin(row) for row in low)

    # Maximum flow assigns at most one molecule from each scaffold to a size
    # bin while exactly matching the low-diversity sample's coarse size mix.
    graph = nx.DiGraph()
    sink = "sink"
    source = "source"
    by_scaffold_bin = {}
    for index, (scaffold, group) in enumerate(sorted(by_scaffold.items())):
        node = f"scaffold_{index}"
        graph.add_edge(source, node, capacity=1)
        bins = defaultdict(list)
        for row in group:
            bins[size_bin(row)].append(row)
        by_scaffold_bin[node] = bins
        for bin_id in sorted(bins):
            graph.add_edge(node, f"bin_{bin_id}", capacity=1)
    for bin_id in range(3):
        graph.add_edge(f"bin_{bin_id}", sink, capacity=required[bin_id])
    flow_value, flow = nx.maximum_flow(graph, source, sink,
                                       flow_func=nx.algorithms.flow.preflow_push)
    if flow_value != args.count:
        raise ValueError(f"Could match only {flow_value}/{args.count} scaffolds to size bins")
    high = []
    for node, bins in by_scaffold_bin.items():
        for bin_id in bins:
            if flow[node].get(f"bin_{bin_id}", 0):
                high.append(rng.choice(bins[bin_id]))
    if len(high) != args.count:
        raise AssertionError("Flow assignment did not select requested count")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.source), "seed": args.seed,
              "train_examples_per_class": args.count, "variants": {}}
    for name, fake_sample in (("low", low), ("high", high)):
        selected = selected_auth | {row["id"] for row in fake_sample}
        output = [row for row in rows
                  if row["split"] != "train" or row["id"] in selected]
        path = args.out / f"evolved_train_scaffolds_{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(output)
        report["variants"][name] = {
            "path": str(path),
            "rows": len(output),
            "counterfeit_train_scaffolds": len({row["scaffold"] for row in fake_sample}),
            "counterfeit_train_size_bins": {str(i): sum(size_bin(row) == i for row in fake_sample)
                                            for i in range(3)},
            "counterfeit_mean_heavy_atoms": round(
                sum(int(row["heavy_atoms"]) for row in fake_sample) / args.count, 3),
            "split_label_counts": {
                split: {label: sum(row["split"] == split and row["label"] == label
                                   for row in output) for label in ("0", "1")}
                for split in ("train", "val", "test")
            },
        }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
