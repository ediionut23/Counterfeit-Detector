"""Plot site and edit-region localization for verified Cl->F test molecules."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path("benchmark/explanations")
    report = json.loads((root / "summary.json").read_text())
    measured = report["verified_subset"]
    labels = ["F site\ntop 1", "F site\ntop 2",
              "Edit region\ntop 1", "Edit region\ntop 2"]
    observed = [measured["top1_site_accuracy"], measured["top2_site_recall"],
                measured["top1_edit_region_accuracy"],
                measured["top2_edit_region_recall"]]
    random = [measured["random_top1_expectation"],
              measured["random_top2_expectation"],
              measured["random_top1_edit_region_expectation"],
              measured["random_top2_edit_region_expectation"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.18, observed, width=0.36, color="#1f77b4", label="GIN saliency")
    ax.bar(x + 0.18, random, width=0.36, color="#b0b0b0", label="Uniform random")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Fraction localized")
    ax.set_title(f"Cl→F GIN atom localization · {measured['n']} exactly verified test edits")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.text(0.5, 0.01,
             "Edit region = changed F atom + bonded neighbor; saliency = |input × gradient|.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(root / "localization.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "localization.svg", bbox_inches="tight")
    print(root / "localization.png")


if __name__ == "__main__":
    main()
