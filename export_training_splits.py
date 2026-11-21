"""Export split-specific CSVs from a dataset that has a `split` column."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("balanced_datasets/generator_versions_200k/combined/dataset.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("balanced_datasets/training_200k"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter[str]] = {name: Counter() for name in ("train", "val", "test")}
    row_counts = Counter()
    with args.source.open(newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "split" not in reader.fieldnames or "label" not in reader.fieldnames:
            parser.error("source CSV must have split and label columns")
        fields = reader.fieldnames
        with args.source.open(newline="") as input_stream:
            reader = csv.DictReader(input_stream)
            streams = {
                split: (args.out / f"{split}.csv").open("w", newline="")
                for split in counts
            }
            try:
                writers = {}
                for split, stream in streams.items():
                    writers[split] = csv.DictWriter(
                        stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
                    )
                    writers[split].writeheader()
                for row in reader:
                    split = row["split"]
                    if split not in writers:
                        raise ValueError(f"unexpected split value: {split!r}")
                    writers[split].writerow(row)
                    row_counts[split] += 1
                    counts[split][row["label"]] += 1
            finally:
                for stream in streams.values():
                    stream.close()

    summary = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output_directory": str(args.out),
        "rows_per_split": dict(row_counts),
        "label_counts_per_split": {
            split: {label: counts[split][label] for label in ("0", "1")}
            for split in counts
        },
        "columns": fields,
        "note": "split files are exported without changing source rows or labels",
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "root": str(args.out),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.out.glob("*.csv"))
        },
    }
    (args.out / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary["rows_per_split"], indent=2))


if __name__ == "__main__":
    main()
