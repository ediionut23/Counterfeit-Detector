"""Plot paired GIN changes from adding atom ring-size bits on three tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = (
    ("Cl_to_F", "Cl→F"),
    ("Ring_resize", "Ring resize"),
    ("Ring_hetero_swap", "Ring heteroatom swap"),
)
SEEDS = (42, 43, 44)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/gnn_comparison"))
    args = parser.parse_args()
    values = {"roc_auc": [], "fnr": []}
    for prefix, label in TASKS:
        for key in values:
            paired = []
            for seed in SEEDS:
                standard = json.loads((args.root / f"{prefix}_gin_seed{seed}.json").read_text())
                ring = json.loads((args.root / f"{prefix}_gin_ring_seed{seed}.json").read_text())
                if standard["dataset"] != ring["dataset"]:
                    raise ValueError(f"Different datasets for {prefix}, seed {seed}")
                paired.append(ring["test_metrics"][key] - standard["test_metrics"][key])
            values[key].append((label, paired))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout="constrained")
    for ax, key, title, color in zip(
        axes,
        ("roc_auc", "fnr"),
        ("Change in test AUC", "Change in false-negative rate"),
        ("#27609c", "#ba5c22"),
    ):
        for i, (label, paired) in enumerate(values[key]):
            avg, sd = mean(paired), stdev(paired)
            ax.errorbar(avg, i, xerr=sd, fmt="o", color=color,
                        capsize=4, markersize=8, zorder=3)
            for j, point in enumerate(paired):
                ax.scatter(point, i + (j - 1) * 0.11, color=color, alpha=0.5,
                           s=25, zorder=4)
        ax.axvline(0, color="#454545", linewidth=1, linestyle="--")
        ax.set_yticks(range(len(TASKS)), [label for _, label in TASKS])
        ax.invert_yaxis()
        ax.set_xlabel("Ring-feature GIN minus standard GIN")
        ax.set_title(title)
        ax.grid(axis="x", color="#dedede", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Paired representation ablation · same split, seeds 42–44")
    fig.savefig(args.root / "ring_feature_ablation.png", dpi=180)
    fig.savefig(args.root / "ring_feature_ablation.svg")
    print(args.root / "ring_feature_ablation.png")


if __name__ == "__main__":
    main()
