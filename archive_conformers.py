"""Split a central SDF into independently readable gzip shards under 100 MB.

Usage: python archive_conformers.py benchmark/conformers_full.sdf --out benchmark
Each shard ends at an SDF record boundary and contains at most 50,000 records.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmark"))
    parser.add_argument("--records-per-shard", type=int, default=50000)
    args = parser.parse_args()
    if args.records_per_shard <= 0:
        parser.error("--records-per-shard must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    files = []
    writer = None
    records = 0
    last_content = b""
    try:
        with args.source.open("rb") as source:
            for line in source:
                if line.strip():
                    last_content = line.strip()
                if writer is None:
                    number = len(files)
                    final = args.out / f"conformers_part_{number:03d}.sdf.gz"
                    temp = final.with_name(final.name + ".tmp")
                    files.append((temp, final))
                    writer = gzip.open(temp, "wb", compresslevel=6)
                writer.write(line)
                if line.strip() == b"$$$$":
                    records += 1
                    if records % args.records_per_shard == 0:
                        writer.close()
                        writer = None
        if writer is not None:
            writer.close()
            writer = None
    finally:
        if writer is not None:
            writer.close()

    if records == 0 or last_content != b"$$$$":
        raise ValueError("Source SDF is empty or ends with an incomplete record")

    for temp, final in files:
        if temp.stat().st_size >= 100_000_000:
            raise ValueError(f"Shard exceeds GitHub's 100 MB limit: {temp}")
    for temp, final in files:
        os.replace(temp, final)
        print(f"{final}: {final.stat().st_size:,} bytes")
    current = {final for _, final in files}
    for stale in args.out.glob("conformers_part_*.sdf.gz"):
        if stale not in current:
            stale.unlink()
            print(f"Removed obsolete shard: {stale}")
    print(f"Archived {records:,} SDF records in {len(files)} shards")


if __name__ == "__main__":
    main()
