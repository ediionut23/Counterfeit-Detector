"""Apply the exact edit regions in verified_edits/manifest.csv to benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("benchmark/verified_edits/manifest.csv"))
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        verified = {
            (row["parent_smiles"], row["smiles"]): row["verified_mask"]
            for row in csv.DictReader(stream)
        }
    seen = set()
    changed = matched = 0
    for path in sorted(args.root.rglob("dataset.csv")):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        local = 0
        for row in rows:
            key = (row["parent_smiles"], row["smiles"])
            mask = verified.get(key)
            if mask is None:
                continue
            if row["label"] != "1":
                raise ValueError(f"Manifest pair appears with authentic label: {path}, {key}")
            seen.add(key)
            matched += 1
            if row["edited_atoms"] != mask:
                row["edited_atoms"] = mask
                local += 1
        if local:
            with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent,
                                             delete=False) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                temporary = Path(stream.name)
            os.replace(temporary, path)
            changed += local
    missing = set(verified) - seen
    if missing:
        raise ValueError(f"{len(missing)} verified pairs absent from {args.root}")
    print(f"Verified {len(seen)} pairs across {matched} table rows; "
          f"updated {changed} masks")


if __name__ == "__main__":
    main()
