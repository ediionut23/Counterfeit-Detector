"""Assemble the MMP sources for the isolated 400k counterfeit bank build."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True,
                        help="CSV produced by generate_100k_counterfeits.py")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "molecule_expansion_400k/counterfeits_m_combined.csv")
    args = parser.parse_args()

    sources = [
        ROOT / "counterfeits_extension_m/counterfeits_100k.csv",
        ROOT / "training_subcategory_expansion/counterfeits_mmp_combined.csv",
        args.generated,
    ]
    fields: list[str] = []
    for path in sources:
        with path.open(newline="") as stream:
            for name in csv.DictReader(stream).fieldnames or ():
                if name not in fields:
                    fields.append(name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with args.out.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for path in sources:
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    writer.writerow(row)
                    n_rows += 1
    print(f"Wrote {n_rows:,} MMP source relations from {len(sources)} files to {args.out}")


if __name__ == "__main__":
    main()
