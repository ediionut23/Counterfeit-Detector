"""Summarize edited-atom mask size as a diagnostic, without asserting correctness."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    root = Path("benchmark")
    datasets = {}
    for path in sorted(root.rglob("dataset.csv")):
        sizes = []
        fractions = []
        blank = 0
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["label"] != "1":
                    continue
                size = len(row["edited_atoms"].split())
                if not size:
                    blank += 1
                sizes.append(size)
                fractions.append(size / max(int(row["heavy_atoms"]), 1))
        if not sizes:
            continue
        name = str(path.parent.relative_to(root))
        datasets[name] = {
            "counterfeits": len(sizes), "blank_masks": blank,
            "median_mask_atoms": statistics.median(sizes),
            "mean_mask_atoms": round(statistics.mean(sizes), 3),
            "masks_over_half_heavy_atoms": sum(value > 0.5 for value in fractions),
            "masks_over_quarter_heavy_atoms": sum(value > 0.25 for value in fractions),
        }
    output = {
        "note": "Large masks are a warning for atom-attribution evaluation, not proof of incorrect labels; datasets overlap",
        "datasets": datasets,
    }
    destination = root / "edit_mask_size_audit.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    for name in ("pooled", "evolved", "category_halogen-walk",
                 "subcategory_halogen-walk.Cl->F"):
        print(name, datasets.get(name))
    print(destination)


if __name__ == "__main__":
    main()
