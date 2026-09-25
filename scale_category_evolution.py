"""
Scale Category-Conditioned Multi-Objective Evolutionary Generation
===================================================================
Runs NSGA-II constrained evolution across all 5 chemical categories
(bioisostere, halogen-walk, scaffold-hop, homologation, other) using
large parent batches (e.g. 600 parents per category).

Saves to `version_e_categories/counterfeits_evolved_{category}.csv`
and automatically runs the comprehensive quality audit at the end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CATEGORIES = [
    ("bioisostere", "scaffold"),
    ("halogen-walk", "scaffold"),
    ("scaffold-hop", "scaffold"),
    ("homologation", "mass-balanced"),
    ("other", "scaffold"),
]


def main():
    parser = argparse.ArgumentParser(description="Scale Category-Conditioned Evolution")
    parser.add_argument("--parents", default="public_molecules.smi", help="Authentic parents pool")
    parser.add_argument("--max-parents", type=int, default=600, help="Parents per category")
    parser.add_argument("--pop", type=int, default=20, help="NSGA-II population size")
    parser.add_argument("--generations", type=int, default=8, help="Generations")
    parser.add_argument("--per-parent-keep", type=int, default=6, help="Kept counterfeits per parent")
    parser.add_argument("--out", default="version_e_categories", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print(f"  SCALING CATEGORY-CONDITIONED EVOLUTION: {len(CATEGORIES)} CATEGORIES")
    print(f"  Parents per category: {args.max_parents} | Pop: {args.pop} | Gen: {args.generations}")
    print(f"  Output directory: {out_dir}")
    print("=" * 80)

    for cat_idx, (cat, cross_op) in enumerate(CATEGORIES, 1):
        print(f"\n>>> [{cat_idx}/{len(CATEGORIES)}] Starting Evolution for '{cat.upper()}' (crossover: {cross_op}) <<<")
        cmd = [
            sys.executable, "evolve_category_pilot.py",
            "--category", cat,
            "--parents", args.parents,
            "--max-parents", str(args.max_parents),
            "--pop", str(args.pop),
            "--generations", str(args.generations),
            "--crossover", cross_op,
            "--per-parent-keep", str(args.per_parent_keep),
            "--out", str(out_dir),
            "--seed", str(42 + cat_idx * 100),
        ]
        t0 = time.time()
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"ERROR: Generation for {cat} failed with code {res.returncode}")
        else:
            print(f">>> Completed '{cat}' in {time.time() - t0:.1f}s <<<\n")

    total_dur = time.time() - t_start
    print("=" * 80)
    print(f"  ALL CATEGORIES COMPLETED IN {total_dur / 60:.1f} MINUTES")
    print("=" * 80)

    # Run complete audit
    print("\n>>> Launching Comprehensive Quality Audit on All Evolved Categories <<<")
    audit_cmd = [
        sys.executable, "audit_evolved_categories.py",
        "--input-dir", str(out_dir),
        "--parents", args.parents,
    ]
    subprocess.run(audit_cmd)


if __name__ == "__main__":
    main()
