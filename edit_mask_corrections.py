"""Version and apply the edited-atom corrections used by this snapshot.

The source generators sometimes emit missing or misindexed edit masks. MCS
repair can be ambiguous for symmetric structures and expensive for explicit
isotopic hydrogen. This manifest records the selected child-atom mask for each
affected parent/child pair, so a benchmark rebuild reproduces those annotations
without repeating the MCS search.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


FIELDS = ("parent_smiles", "smiles", "source_edited_atoms", "corrected_edited_atoms")


def read(path: Path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_table(path: Path, fields, rows) -> None:
    with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent,
                                     delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build(snapshot: Path, source: Path, output: Path) -> None:
    corrections = {}
    n_rows = 0
    for expected_path in sorted(snapshot.rglob("dataset.csv")):
        relative = expected_path.relative_to(snapshot)
        _, expected = read(expected_path)
        _, raw = read(source / relative)
        if len(expected) != len(raw):
            raise ValueError(f"Different row count: {relative}")
        for left, right in zip(expected, raw):
            if any(left[key] != right[key] for key in left
                   if key not in ("split", "edited_atoms")):
                raise ValueError(f"Molecule or metadata mismatch: {relative}")
            if left["edited_atoms"] == right["edited_atoms"]:
                continue
            key = (left["parent_smiles"], left["smiles"])
            value = (right["edited_atoms"], left["edited_atoms"])
            if key in corrections and corrections[key] != value:
                raise ValueError(f"Conflicting correction for {key}")
            corrections[key] = value
            n_rows += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        for (parent, child), (source_mask, corrected_mask) in sorted(corrections.items()):
            writer.writerow((parent, child, source_mask, corrected_mask))
    print(f"Wrote {len(corrections)} parent/child corrections covering {n_rows} rows")


def merge(snapshot: Path, source: Path, manifest_path: Path) -> None:
    """Add mask differences from a staged rebuild to an existing manifest."""
    _, entries = read(manifest_path)
    corrections = {(row["parent_smiles"], row["smiles"]): row for row in entries}
    added = set()
    differences = 0
    for expected_path in sorted(snapshot.rglob("dataset.csv")):
        relative = expected_path.relative_to(snapshot)
        expected_fields, expected = read(expected_path)
        actual_fields, actual = read(source / relative)
        if expected_fields != actual_fields or len(expected) != len(actual):
            raise ValueError(f"Different table shape: {relative}")
        for left, right in zip(expected, actual):
            if any(left[key] != right[key] for key in left
                   if key != "edited_atoms"):
                raise ValueError(f"Non-mask difference: {relative}")
            if left["edited_atoms"] == right["edited_atoms"]:
                continue
            differences += 1
            key = (left["parent_smiles"], left["smiles"])
            previous = corrections.get(key)
            if previous is not None:
                if previous["corrected_edited_atoms"] != left["edited_atoms"]:
                    raise ValueError(f"Conflicting target mask for {key}")
                continue
            corrections[key] = dict(zip(FIELDS, (*key, right["edited_atoms"],
                                                 left["edited_atoms"])))
            added.add(key)
    write_table(manifest_path, FIELDS, [corrections[key] for key in sorted(corrections)])
    print(f"Merged {len(added)} new parent/child corrections from {differences} "
          f"differing rows; manifest has {len(corrections)} entries")


def apply(root: Path, manifest: Path) -> None:
    _, entries = read(manifest)
    corrections = {(row["parent_smiles"], row["smiles"]): row for row in entries}
    seen = set()
    changed = unexpected = 0
    for path in sorted(root.rglob("dataset.csv")):
        fields, rows = read(path)
        local = 0
        for row in rows:
            key = (row["parent_smiles"], row["smiles"])
            entry = corrections.get(key)
            if entry is None:
                continue
            seen.add(key)
            current = row["edited_atoms"]
            expected_source = entry["source_edited_atoms"]
            corrected = entry["corrected_edited_atoms"]
            if current not in (expected_source, corrected):
                unexpected += 1
            if current != corrected:
                row["edited_atoms"] = corrected
                local += 1
        if local:
            write_table(path, fields, rows)
            changed += local
    missing = len(corrections.keys() - seen)
    if missing:
        raise ValueError(f"{missing} manifest pairs absent from {root}")
    print(f"Applied {changed} row corrections; {unexpected} masks differed from "
          "both the recorded source and correction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--snapshot", type=Path, default=Path("benchmark"))
    builder.add_argument("--source", type=Path, default=Path("benchmark_regen"))
    builder.add_argument("--out", type=Path,
                         default=Path("benchmark/edit_mask_corrections.csv"))
    merger = sub.add_parser("merge")
    merger.add_argument("--snapshot", type=Path, default=Path("benchmark"))
    merger.add_argument("--source", type=Path, default=Path("benchmark_regen"))
    merger.add_argument("--manifest", type=Path,
                        default=Path("benchmark/edit_mask_corrections.csv"))
    applier = sub.add_parser("apply")
    applier.add_argument("--root", type=Path, default=Path("benchmark"))
    applier.add_argument("--manifest", type=Path,
                         default=Path("benchmark/edit_mask_corrections.csv"))
    args = parser.parse_args()
    if args.command == "build":
        build(args.snapshot, args.source, args.out)
    elif args.command == "merge":
        merge(args.snapshot, args.source, args.manifest)
    else:
        apply(args.root, args.manifest)


if __name__ == "__main__":
    main()
