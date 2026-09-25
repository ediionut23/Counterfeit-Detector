"""
Build "Version E" — the evolved, adversarial counterfeit dataset (High-Throughput Parallel)
=============================================================================================

Scales the multi-objective genetic search of `evolutionary_generator.py` across
thousands of authentic drug parents from `public_molecules_100k.smi` using multi-core
parallel processing to generate 10,000 - 20,000+ high-quality counterfeits.
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

# Use fork on POSIX / macOS for fast zero-overhead worker sharing
try:
    multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

import evolutionary_generator as EG

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Global worker variables for process pooling
_WORKER_RULE_LIB = None
_WORKER_SCORER = None


def _init_worker(rules_path: str, adversarial: bool):
    global _WORKER_RULE_LIB, _WORKER_SCORER
    RDLogger.DisableLog("rdApp.*")
    _WORKER_RULE_LIB = EG.RuleLibrary(rules_path)
    if adversarial:
        _WORKER_SCORER = EG.load_detector_scorer()
    else:
        _WORKER_SCORER = None


def _worker_process_parent(args_tuple) -> List[Dict]:
    global _WORKER_RULE_LIB, _WORKER_SCORER
    (parent, sim_lo, sim_hi, pop, generations, crossover_op, adversarial,
     p_max, qed_min, max_sa, per_parent_keep, seed, objective_profile) = args_tuple

    try:
        objectives = EG.Objectives(parent, sim_lo, sim_hi, adversarial=_WORKER_SCORER,
                                  profile=objective_profile)
        front = EG.build_and_run(parent, _WORKER_RULE_LIB, objectives,
                                 pop_size=pop, generations=generations, seed=seed,
                                 crossover_op=crossover_op, compute_shape=False)
    except Exception:
        return []

    # Filter & rank front
    picked = []
    for d in front:
        if not (sim_lo <= d["similarity_to_parent"] <= sim_hi):
            continue
        if d["qed"] < qed_min:
            continue
        if d.get("sa_score") is not None and d["sa_score"] > max_sa:
            continue
        if d.get("structural_alerts", 0) > 0:
            continue
        if adversarial and d["objectives"].get("p_counterfeit", 1.0) > p_max:
            continue
        picked.append(d)

    if adversarial:
        picked.sort(key=lambda d: d["objectives"].get("p_counterfeit", 1.0))
    else:
        picked.sort(key=lambda d: sum(d["objectives"].values()))

    out_rows = []
    for d in picked[:per_parent_keep]:
        pc = d["objectives"].get("p_counterfeit")
        out_rows.append({
            "counterfeit_smiles": d["smiles"],
            "parent_smiles": parent,
            "category": "evolved-adversarial" if adversarial else "evolved",
            "similarity_to_parent": d["similarity_to_parent"],
            "qed": d["qed"],
            "sa_score": d["sa_score"],
            "p_counterfeit": round(pc, 4) if pc is not None else None,
            "n_edited_atoms": len(d["edited_atoms"]),
            "edited_atoms": " ".join(map(str, d["edited_atoms"])),
        })
    return out_rows


def load_parents(parents_file: Optional[str], from_pt: Optional[str],
                 max_parents: Optional[int], seed: int) -> List[str]:
    smis: List[str] = []
    if from_pt:
        import torch
        graphs, labels, _ = torch.load(from_pt, weights_only=False)
        for g, lab in zip(graphs, labels):
            if int(lab) == 0 and getattr(g, "smiles", None):
                smis.append(g.smiles)
    else:
        path = parents_file or "public_molecules_100k.smi"
        if not Path(path).exists() and Path("mmp_mining/input.smi").exists():
            path = "mmp_mining/input.smi"
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    smis.append(line.split()[0])
    seen, out = set(), []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if c not in seen:
            seen.add(c)
            out.append(c)
    random.Random(seed).shuffle(out)
    if max_parents:
        out = out[:max_parents]
    logger.info(f"Loaded {len(out)} unique authentic parents from {path}")
    return out


def save_checkpoint(outdir: Path, rows: List[Dict], summary: Dict):
    cols = ["counterfeit_smiles", "parent_smiles", "category",
            "similarity_to_parent", "qed", "sa_score", "p_counterfeit",
            "n_edited_atoms", "edited_atoms"]
    csv_file = outdir / "counterfeits_evolved.csv"
    with open(csv_file, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rules", default="mined_transformations.py")
    p.add_argument("--parents", default="public_molecules_100k.smi",
                   help="SMILES file of authentic parents (default public_molecules_100k.smi)")
    p.add_argument("--from-pt", default=None)
    p.add_argument("--max-parents", type=int, default=8000,
                   help="Number of authentic parents to evolve (default 8000)")
    p.add_argument("--target-count", type=int, default=15000,
                   help="Target number of unique evolved counterfeits to generate (default 15000)")
    p.add_argument("--per-parent-keep", type=int, default=10,
                   help="Max counterfeits kept per parent (default 10)")
    p.add_argument("--pop", type=int, default=20, help="GA population per parent (default 20)")
    p.add_argument("--generations", type=int, default=6, help="GA generations per parent (default 6)")
    p.add_argument("--crossover-op", choices=["hybrid", "mass-balanced", "scaffold", "brics", "jensen"],
                   default="mass-balanced", help="Graph crossover operator (default: mass-balanced)")
    p.add_argument("--objective-profile", choices=["anti-shortcut", "adversarial", "baseline"],
                   default="anti-shortcut", help="Multi-objective function formulation profile (default: anti-shortcut)")
    p.add_argument("--sim-lo", type=float, default=0.4, help="Lower Tanimoto band")
    p.add_argument("--sim-hi", type=float, default=0.9, help="Upper Tanimoto band")
    p.add_argument("--no-adversarial", action="store_true",
                   help="Disable the detector objective (structural E only)")
    p.add_argument("--p-max", type=float, default=1.0)
    p.add_argument("--qed-min", type=float, default=0.35, help="Min QED drug-likeness (default 0.35)")
    p.add_argument("--max-sa", type=float, default=3.8, help="Max synthetic accessibility (default 3.8)")
    p.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    p.add_argument("--out", default="version_e")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    parents = load_parents(args.parents, args.from_pt, args.max_parents, args.seed)
    adversarial = not args.no_adversarial

    logger.info("=" * 80)
    logger.info(f"🚀 Launching Large-Scale Genetic Generation: Target ~{args.target_count} counterfeits")
    logger.info(f"   Pool: {len(parents)} parents | Workers: {args.workers} cores | Operator: {args.crossover_op}")
    logger.info(f"   Objective Profile: {args.objective_profile} | Adversarial: {adversarial}")
    logger.info(f"   Filters: Tanimoto=[{args.sim_lo}, {args.sim_hi}] | Min QED={args.qed_min} | Max SA={args.max_sa}")
    logger.info("=" * 80)

    tasks = [
        (parent, args.sim_lo, args.sim_hi, args.pop, args.generations,
         args.crossover_op, adversarial, args.p_max, args.qed_min, args.max_sa,
         args.per_parent_keep, args.seed + i, args.objective_profile)
        for i, parent in enumerate(parents)
    ]

    t0 = time.time()
    global_seen: set = set()
    rows: List[Dict] = []
    n_fooling = 0

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(args.rules, adversarial)) as executor:
        futures = [executor.submit(_worker_process_parent, t) for t in tasks]
        
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            res = fut.result()
            for r in res:
                smi = r["counterfeit_smiles"]
                if smi in global_seen:
                    continue
                global_seen.add(smi)
                pc = r.get("p_counterfeit")
                if pc is not None and pc < 0.5:
                    n_fooling += 1
                rows.append(r)

            # Periodic checkpoint and logging
            if done_count % 100 == 0 or done_count == len(parents) or len(rows) >= args.target_count:
                elapsed = time.time() - t0
                speed = done_count / max(elapsed, 0.001)
                eta_sec = (len(parents) - done_count) / max(speed, 0.001) if len(rows) < args.target_count else 0
                logger.info(f"  ⚡ [{done_count}/{len(parents)} parents ({speed:.1f} parents/sec)] -> {len(rows)}/{args.target_count} counterfeits (ETA: {eta_sec/60:.1f} min)")

                # Save checkpoint
                sims = [r["similarity_to_parent"] for r in rows] or [0]
                qeds = [r["qed"] for r in rows if r["qed"] is not None] or [0]
                sas = [r["sa_score"] for r in rows if r["sa_score"] is not None] or [0]
                summary = {
                    "n_parents_processed": done_count,
                    "n_counterfeits": len(rows),
                    "target_count": args.target_count,
                    "crossover_operator": args.crossover_op,
                    "objective_profile": args.objective_profile,
                    "adversarial": adversarial,
                    "mean_similarity": round(float(np.mean(sims)), 3),
                    "mean_qed": round(float(np.mean(qeds)), 3),
                    "mean_sa_score": round(float(np.mean(sas)), 3),
                    "similarity_band": [args.sim_lo, args.sim_hi],
                    "elapsed_seconds": round(elapsed, 1),
                }
                save_checkpoint(outdir, rows, summary)

            if len(rows) >= args.target_count:
                logger.info(f"🎯 Target count of {args.target_count} counterfeits reached! Concluding generation.")
                break

    # Final summary save
    sims = [r["similarity_to_parent"] for r in rows] or [0]
    qeds = [r["qed"] for r in rows if r["qed"] is not None] or [0]
    sas = [r["sa_score"] for r in rows if r["sa_score"] is not None] or [0]
    pcs = [r["p_counterfeit"] for r in rows if r["p_counterfeit"] is not None]
    
    summary = {
        "n_parents_processed": done_count,
        "n_counterfeits": len(rows),
        "target_count": args.target_count,
        "crossover_operator": args.crossover_op,
        "objective_profile": args.objective_profile,
        "adversarial": adversarial,
        "n_fool_detector": n_fooling,
        "frac_fool_detector": round(n_fooling / max(len(rows), 1), 3),
        "mean_similarity": round(float(np.mean(sims)), 3),
        "mean_qed": round(float(np.mean(qeds)), 3),
        "mean_sa_score": round(float(np.mean(sas)), 3),
        "mean_p_counterfeit": round(float(np.mean(pcs)), 3) if pcs else None,
        "similarity_band": [args.sim_lo, args.sim_hi],
        "search": {"pop": args.pop, "generations": args.generations, "workers": args.workers},
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    save_checkpoint(outdir, rows, summary)

    logger.info("=" * 80)
    logger.info(f"🏆 Version E Large-Scale Build Completed: {len(rows)} evolved counterfeits in {outdir}/")
    logger.info(f"   Mean QED: {summary['mean_qed']} | Mean SA Score: {summary['mean_sa_score']} | Mean Similarity: {summary['mean_similarity']}")
    logger.info(f"   Total Time: {summary['elapsed_seconds']/60:.1f} minutes ({done_count/summary['elapsed_seconds']:.1f} parents/sec)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
