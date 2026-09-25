"""
Category-Conditioned Multi-Objective Evolutionary Generator (Parallel High-Throughput)
=======================================================================================
Evolves authentic pharmaceutical drugs using NSGA-II constrained strictly to mutation
rules belonging to a specific chemical category (e.g. bioisostere, halogen-walk, scaffold-hop).

Produces Pareto-optimal, multi-step "hard counterfeits" that preserve drug-likeness,
synthesizability, and avoid macroscopic property shortcuts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

import evolutionary_generator as EG

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evolve_category_pilot")

# Worker state
_WORKER_RULE_LIB = None
_WORKER_CATEGORY = None


def _init_worker(rules_path: str, category: str):
    global _WORKER_RULE_LIB, _WORKER_CATEGORY
    RDLogger.DisableLog("rdApp.*")
    _WORKER_CATEGORY = category
    _WORKER_RULE_LIB = EG.RuleLibrary(rules_path, category=category)


def _worker_process_parent(args_tuple) -> List[Dict]:
    global _WORKER_RULE_LIB, _WORKER_CATEGORY
    (parent, sim_lo, sim_hi, pop, generations, crossover_op,
     qed_min, max_sa, per_parent_keep, seed, objective_profile) = args_tuple

    try:
        objectives = EG.Objectives(parent, sim_lo, sim_hi, adversarial=None, profile=objective_profile)
        front = EG.build_and_run(
            parent, _WORKER_RULE_LIB, objectives,
            pop_size=pop, generations=generations, seed=seed,
            crossover_op=crossover_op, compute_shape=False
        )
    except Exception:
        return []

    picked = []
    seen = set()
    for d in front:
        smi = d["smiles"]
        if smi in seen or smi == parent:
            continue
        seen.add(smi)
        sim = d["similarity_to_parent"]
        if not (sim_lo - 0.05 <= sim <= sim_hi + 0.03):
            continue
        if d["qed"] < qed_min:
            continue
        if d.get("sa_score") is not None and d["sa_score"] > max_sa:
            continue
        if d.get("structural_alerts", 0) > 0:
            continue
        picked.append(d)

    # Rank by aggregated Pareto objectives
    picked.sort(key=lambda d: sum(d["objectives"].values()))

    out_rows = []
    for d in picked[:per_parent_keep]:
        m = Chem.MolFromSmiles(d["smiles"])
        if m is None:
            continue
        out_rows.append({
            "counterfeit_smiles": d["smiles"],
            "parent_smiles": parent,
            "category": f"evolved_{_WORKER_CATEGORY}",
            "subcategory": f"evolved_{_WORKER_CATEGORY}",
            "similarity_to_parent": d["similarity_to_parent"],
            "qed": d["qed"],
            "sa_score": d["sa_score"],
            "n_edited_atoms": len(d["edited_atoms"]),
            "edited_atoms": " ".join(map(str, d["edited_atoms"])),
            "mw": round(float(Descriptors.MolWt(m)), 2),
            "logp": round(float(Descriptors.MolLogP(m)), 2),
            "heavy_atoms": m.GetNumHeavyAtoms(),
            "smiles_length": len(d["smiles"]),
        })
    return out_rows


def load_parents(path: str, max_parents: int, seed: int = 42) -> List[str]:
    seen, out = set(), []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            # Keep typical drug-like parents (MW 180 - 600, heavy atoms 12 - 45)
            mw = Descriptors.MolWt(m)
            ha = m.GetNumHeavyAtoms()
            if not (180.0 <= mw <= 600.0 and 12 <= ha <= 45):
                continue
            c = Chem.MolToSmiles(m)
            if c not in seen:
                seen.add(c)
                out.append(c)
    random.Random(seed).shuffle(out)
    return out[:max_parents]


def main():
    parser = argparse.ArgumentParser(description="Category-Conditioned Multi-Objective Evolutionary Generator")
    parser.add_argument("--category", required=True, choices=["bioisostere", "halogen-walk", "scaffold-hop", "homologation", "other"],
                        help="Chemical category to constrain mutations to")
    parser.add_argument("--rules", default="mined_transformations.py", help="Path to rules script")
    parser.add_argument("--parents", default="public_molecules.smi", help="Path to parent SMILES file")
    parser.add_argument("--max-parents", type=int, default=200, help="Number of parent drugs to evolve")
    parser.add_argument("--pop", type=int, default=20, help="Population size in NSGA-II")
    parser.add_argument("--generations", type=int, default=8, help="Generations per parent evolution")
    parser.add_argument("--sim-lo", type=float, default=0.60, help="Lower Tanimoto similarity bound")
    parser.add_argument("--sim-hi", type=float, default=0.85, help="Upper Tanimoto similarity bound")
    parser.add_argument("--qed-min", type=float, default=0.45, help="Minimum QED drug-likeness")
    parser.add_argument("--max-sa", type=float, default=4.20, help="Maximum Ertl SA score")
    parser.add_argument("--per-parent-keep", type=int, default=6, help="Maximum counterfeits to keep per parent")
    parser.add_argument("--crossover", default="scaffold", choices=["scaffold", "mass-balanced", "hybrid"],
                        help="Crossover operator")
    parser.add_argument("--objective-profile", default="anti-shortcut", help="Objectives profile")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: cpu_count)")
    parser.add_argument("--out", default="version_e_categories", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    workers = args.workers or max(1, os.cpu_count() or 4)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Category-Conditioned Evolution: '{args.category.upper()}' ===")
    parents = load_parents(args.parents, args.max_parents, seed=args.seed)
    logger.info(f"Loaded {len(parents)} authentic parents from {args.parents}")

    tasks = [
        (parent, args.sim_lo, args.sim_hi, args.pop, args.generations,
         args.crossover, args.qed_min, args.max_sa, args.per_parent_keep,
         args.seed + idx, args.objective_profile)
        for idx, parent in enumerate(parents)
    ]

    t0 = time.time()
    all_results: List[Dict] = []
    seen_fakes = set()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(args.rules, args.category)
    ) as pool:
        futures = {pool.submit(_worker_process_parent, t): t[0] for t in tasks}
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            completed += 1
            for row in res:
                smi = row["counterfeit_smiles"]
                if smi not in seen_fakes:
                    seen_fakes.add(smi)
                    all_results.append(row)
            if completed % 25 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.1)
                logger.info(f"Progress: {completed}/{len(tasks)} parents [{completed/len(tasks)*100:.1f}%] "
                            f"({rate:.1f} parents/s) -> {len(all_results)} valid unique counterfeits")

    out_csv = out_dir / f"counterfeits_evolved_{args.category}.csv"
    if all_results:
        cols = list(all_results[0].keys())
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_results)

    dur = time.time() - t0
    logger.info(f"=== Completed '{args.category}': {len(all_results):,} counterfeits from {len(parents)} parents "
                f"in {dur:.1f}s ({len(all_results)/max(dur, 0.1):.1f} fake/s) -> Saved to {out_csv} ===")

    # Print summary statistics
    if all_results:
        sims = [r["similarity_to_parent"] for r in all_results]
        qeds = [r["qed"] for r in all_results]
        sas = [r["sa_score"] for r in all_results if r["sa_score"] is not None]
        edits = [r["n_edited_atoms"] for r in all_results]
        print("\n" + "=" * 60)
        print(f"  SUMMARY FOR EVOLVED {args.category.upper()}")
        print("=" * 60)
        print(f"  Total Valid Counterfeits : {len(all_results):,}")
        print(f"  Mean Tanimoto to Parent  : {np.mean(sims):.3f} (Range: {np.min(sims):.3f} - {np.max(sims):.3f})")
        print(f"  Mean QED Drug-Likeness   : {np.mean(qeds):.3f} (Target >= 0.45)")
        print(f"  Mean Synthesizability SA : {np.mean(sas):.2f} (Target <= 4.20)")
        print(f"  Mean Atoms Edited (MCS)  : {np.mean(edits):.1f} atoms")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
