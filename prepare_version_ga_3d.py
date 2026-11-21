"""Build the 3D SDF and graph tensors for every completed GA category."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_version_ga import CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("version_ga"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--conformers", type=int, default=8)
    parser.add_argument("--radius-cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.workers < 1 or args.conformers < 1 or args.radius_cutoff <= 0:
        parser.error("workers, conformers, and radius cutoff must be positive")
    reports = {}
    logs = args.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    for index, category in enumerate(CATEGORIES):
        source = args.root / f"counterfeits_evolved_{category}.csv"
        if not source.exists():
            print(f"Skipping {category}: category CSV is not present", flush=True)
            continue
        out_dir = args.root / "3d" / category
        report_path = out_dir / "conformers_3d_report.json"
        if report_path.exists():
            old = json.loads(report_path.read_text())
            if old.get("requested") == old.get("embedded") and old.get("coverage") == 1.0:
                reports[category] = old
                print(f"Skipping {category}: complete conformer report already exists", flush=True)
                continue

        command = [
            sys.executable, "prepare_3d_gnn.py", "--input", str(source),
            "--out", str(out_dir), "--workers", str(args.workers),
            "--conformers", str(args.conformers), "--seed", str(args.seed + index),
            "--radius-cutoff", str(args.radius_cutoff),
        ]
        log_path = logs / f"3d_{category}.log"
        with log_path.open("w") as log:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if "processed;" in line or '"coverage"' in line:
                    print(f"{category}: {line.rstrip()}", flush=True)
            status = proc.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
        report = json.loads(report_path.read_text())
        reports[category] = report
        (args.root / "3d" / "summary_3d.json").write_text(
            json.dumps(reports, indent=2) + "\n")

    print(json.dumps({category: {
        "requested": report["requested"], "embedded": report["embedded"],
        "coverage": report["coverage"],
    } for category, report in reports.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
