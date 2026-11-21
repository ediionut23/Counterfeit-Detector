"""Build the five category-conditioned genetic analogue tables in sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CATEGORIES = ["bioisostere", "halogen-walk", "scaffold-hop", "homologation", "other"]


def run_one(args, category: str, max_parents: int, seed: int,
            exclude_categories: list[str]) -> dict:
    log_path = args.out / "logs" / f"{category}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "evolve_category_pilot.py", "--category", category,
        "--rules", str(args.rules), "--parents", str(args.parents),
        "--max-parents", str(max_parents), "--target", str(args.target),
        "--pop", str(args.pop), "--generations", str(args.generations),
        "--per-parent-keep", str(args.per_parent_keep), "--workers", str(args.workers),
        "--seed", str(seed), "--out", str(args.out),
    ]
    exclusion_files = [args.out / f"counterfeits_evolved_{name}.csv"
                       for name in exclude_categories
                       if (args.out / f"counterfeits_evolved_{name}.csv").exists()]
    if exclusion_files:
        command.extend(["--exclude", *map(str, exclusion_files)])
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if "Progress:" in line or "Completed '" in line or "Loaded " in line:
                print(line.rstrip(), flush=True)
        status = proc.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)
    return json.loads((args.out / f"summary_{category}.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, default=Path("public_molecules_100k.smi"))
    parser.add_argument("--rules", type=Path, default=Path("mined_transformations.py"))
    parser.add_argument("--out", type=Path, default=Path("version_ga"))
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--max-parents", type=int, default=7000)
    parser.add_argument("--retry-max-parents", type=int, default=12000)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--per-parent-keep", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--prior-categories", nargs="+", choices=CATEGORIES, default=[],
                        help="Previously generated categories to exclude from every new run.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for category in args.prior_categories:
        summary_path = args.out / f"summary_{category}.json"
        if summary_path.exists():
            summaries[category] = json.loads(summary_path.read_text())

    completed_categories = list(args.prior_categories)
    for offset, category in enumerate(args.categories):
        seed = args.seed + offset * 10000
        summary = run_one(args, category, args.max_parents, seed, completed_categories)
        if not summary["target_reached"] and args.retry_max_parents > args.max_parents:
            print(f"{category}: {summary['n_unique_counterfeits']:,}/{args.target:,}; "
                  f"retrying with {args.retry_max_parents:,} parents", flush=True)
            summary = run_one(args, category, args.retry_max_parents, seed + 1,
                              completed_categories)
        summaries[category] = summary
        completed_categories.append(category)
        (args.out / "summary_all_categories.json").write_text(
            json.dumps(summaries, indent=2) + "\n")

    print(json.dumps({cat: {
        "n_unique_counterfeits": s["n_unique_counterfeits"],
        "target_reached": s["target_reached"],
    } for cat, s in summaries.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
