"""Create sharded 3D GNN graphs for a split-aware balanced CSV dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("balanced_datasets/generator_versions_200k/combined/dataset.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("balanced_datasets/training_200k/3d")
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--conformers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=15000)
    parser.add_argument("--radius-cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--retry-failures", action="store_true",
                        help="Retry failed conformers with at least eight conformers (can take much longer)")
    args = parser.parse_args()
    if args.workers < 1 or args.conformers < 1 or args.shard_size < 1 or args.radius_cutoff <= 0:
        parser.error("workers, conformers, shard-size, and radius-cutoff must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    work_dir = args.out / ".work"
    logs_dir = args.out / "logs"
    work_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    shard_reports = []
    class_counts = {split: Counter() for split in ("train", "val", "test")}
    failed_rows = []
    source_counts = Counter()
    source_sha256 = digest(args.input)

    with args.input.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"smiles", "label", "split"}.issubset(reader.fieldnames):
            parser.error("input CSV must include smiles, label, and split columns")
        fields = reader.fieldnames
        buffers = {split: [] for split in ("train", "val", "test")}

        def process_rows(split: str, rows: list[dict[str, str]], shard_number: int) -> None:
            shard_name = f"shard_{shard_number:03d}"
            source_path = work_dir / f"{split}_{shard_name}.csv"
            shard_dir = args.out / split / shard_name
            shard_dir.mkdir(parents=True, exist_ok=True)
            with source_path.open("w", newline="") as shard_stream:
                writer = csv.DictWriter(shard_stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            for row in rows:
                class_counts[split][row["label"]] += 1
                source_counts[row.get("generation_method", "") or "reference"] += 1
            command = [
                sys.executable, "prepare_3d_gnn.py", "--input", str(source_path),
                "--out", str(shard_dir), "--workers", str(args.workers),
                "--conformers", str(args.conformers), "--seed",
                str(args.seed + shard_number), "--radius-cutoff", str(args.radius_cutoff),
            ]
            log_path = logs_dir / f"{split}_{shard_name}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    if "processed;" in line:
                        try:
                            completed = int(line.split("/", 1)[0].strip().replace(",", ""))
                        except ValueError:
                            completed = 0
                        if completed and completed % 5000 == 0:
                            print(f"{split}/{shard_name}: {line.strip()}", flush=True)
                status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
            report_path = shard_dir / "conformers_3d_report.json"
            report = json.loads(report_path.read_text())
            report["split"] = split
            report["shard"] = shard_name
            report["graph_file"] = str(shard_dir / "graphs_3d.npz")
            report["sdf_file"] = str(shard_dir / "conformers.sdf.gz")
            shard_reports.append(report)
            if report.get("failed_rows_csv"):
                failed_rows.append(Path(report["failed_rows_csv"]))
            source_path.unlink(missing_ok=True)
            print(f"Completed {split}/{shard_name}: {report['embedded']}/{report['requested']} graphs", flush=True)

        shard_number = 0
        for row in reader:
            split = row["split"]
            if split not in buffers:
                raise ValueError(f"unexpected split value: {split!r}")
            buffers[split].append(row)
            if len(buffers[split]) >= args.shard_size:
                process_rows(split, buffers[split], shard_number)
                buffers[split] = []
                shard_number += 1
        for split in ("train", "val", "test"):
            if buffers[split]:
                process_rows(split, buffers[split], shard_number)
                shard_number += 1

    if failed_rows:
        destination = args.out / "failed_embeddings.csv"
        with destination.open("w", newline="") as target:
            writer = None
            for path in failed_rows:
                with path.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    if writer is None:
                        writer = csv.DictWriter(target, fieldnames=reader.fieldnames,
                                                extrasaction="ignore", lineterminator="\n")
                        writer.writeheader()
                    writer.writerows(reader)

    requested = sum(report["requested"] for report in shard_reports)
    initial_embedded = sum(report["embedded"] for report in shard_reports)
    retry_report = None
    retry_success = 0
    if failed_rows and args.retry_failures:
        candidates_path = args.out / "failed_embeddings.csv"
        retry_dir = args.out / "retries"
        retry_dir.mkdir(exist_ok=True)
        retry_command = [
            sys.executable, "prepare_3d_gnn.py", "--input", str(candidates_path),
            "--out", str(retry_dir), "--workers", str(args.workers),
            "--conformers", str(max(8, args.conformers)), "--seed", str(args.seed + 99991),
            "--radius-cutoff", str(args.radius_cutoff),
        ]
        retry_log = logs_dir / "retries.log"
        with retry_log.open("w") as log:
            process = subprocess.run(retry_command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, retry_command)
        retry_report_path = retry_dir / "conformers_3d_report.json"
        retry_report = json.loads(retry_report_path.read_text())
        retry_success = retry_report["embedded"]
        candidates_path.replace(args.out / "retry_candidates.csv")
        if retry_report.get("failed_rows_csv"):
            Path(retry_report["failed_rows_csv"]).replace(candidates_path)

    embedded = initial_embedded + retry_success
    report = {
        "input_csv": str(args.input),
        "input_sha256": source_sha256,
        "requested": requested,
        "embedded": embedded,
        "failed": requested - embedded,
        "coverage": round(embedded / max(requested, 1), 6),
        "label_counts_requested_by_split": {
            split: {label: class_counts[split][label] for label in ("0", "1")}
            for split in class_counts
        },
        "generation_source_counts_requested": dict(source_counts),
        "split_reports": {
            split: {
                "requested": sum(r["requested"] for r in shard_reports if r["split"] == split),
                "embedded": (
                    sum(r["embedded"] for r in shard_reports if r["split"] == split)
                    + (retry_report or {}).get("embedded_split_counts", {}).get(split, 0)
                ),
                "shards": [r["shard"] for r in shard_reports if r["split"] == split],
            }
            for split in ("train", "val", "test")
        },
        "retry": ({
            "requested": retry_report["requested"],
            "recovered": retry_success,
            "remaining_failures": retry_report["failed"],
            "conformers_attempted": max(8, args.conformers),
            "graph_file": str(args.out / "retries" / "graphs_3d.npz"),
        } if retry_report else {"attempted": False, "candidate_file": str(args.out / "failed_embeddings.csv") if failed_rows else None}),
        "conformer_settings": {
            "method": f"ETKDGv3 up to {args.conformers} conformers, lowest-energy minimized conformer",
            "forcefield": "MMFF94s with UFF fallback",
            "hydrogens_retained": True,
            "radius_cutoff_angstrom": args.radius_cutoff,
            "shard_size": args.shard_size,
        },
        "geometry_note": "predicted force-field-minimized coordinates, not experimental structures",
        "graph_arrays": [
            "atom_features", "positions_angstrom", "edge_index", "edge_features",
            "edge_distance_angstrom", "radius_edge_index", "radius_edge_distance_angstrom",
            "angle_index", "angle_radians", "label", "split", "generation_method", "subcategory",
        ],
        "shards": shard_reports,
    }
    report_path = args.out / "summary_3d.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest_files = {}
    for item in shard_reports:
        graph_path = Path(item["graph_file"])
        shard_report_path = graph_path.parent / "conformers_3d_report.json"
        for path in (graph_path, shard_report_path):
            manifest_files[str(path.relative_to(args.out))] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    if retry_report:
        for path in (
            args.out / "retries" / "graphs_3d.npz",
            args.out / "retries" / "conformers_3d_report.json",
        ):
            manifest_files[str(path.relative_to(args.out))] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    manifest_path = args.out / "manifest_sha256.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "root": str(args.out),
        "files": manifest_files,
        "note": "SDFs are not included in this manifest to keep verification fast; graph archives and shard reports are checksummed.",
    }, indent=2) + "\n")
    for path in work_dir.glob("*"):
        path.unlink(missing_ok=True)
    work_dir.rmdir()
    print(json.dumps({key: report[key] for key in ("requested", "embedded", "failed", "coverage", "split_reports")}, indent=2))


if __name__ == "__main__":
    main()
