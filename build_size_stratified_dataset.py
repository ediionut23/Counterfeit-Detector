"""
Size-Stratified Multi-Scale Evolutionary Generator for Small, Medium, and Large Molecules
==========================================================================================

Adapts graph genetic generation, crossover operators, and multi-objective fitness
specifically for three molecular weight and complexity tiers:

1. SMALL MOLECULES (< 350 Da, < 25 heavy atoms):
   - Examples: Aspirin (180 Da), Paracetamol (151 Da), Ibuprofen (206 Da), Metformin (129 Da)
   - Tanimoto Target: [0.35, 0.85]
   - QED Target: >= 0.45
   - SA Target: <= 3.2

2. MEDIUM MOLECULES (350 - 550 Da, 25 - 40 heavy atoms):
   - Standard FDA Oral Drug space (Rule of 5 lead-like)
   - Examples: Atorvastatin (558 Da), Imatinib (493 Da), Sildenafil (474 Da), Dexamethasone (392 Da)
   - Tanimoto Target: [0.45, 0.90]
   - QED Target: >= 0.35
   - SA Target: <= 3.8

3. LARGE MOLECULES (550 - 900+ Da, 40 - 75+ heavy atoms):
   - Complex macrocycles, natural products, proteolysis targeting chimeras (PROTACs), macrolides
   - Examples: Paclitaxel (853 Da), Rifampicin (822 Da), Rapamycin (914 Da), Telmisartan (514 Da)
   - Tanimoto Target: [0.55, 0.95] (accounts for high bit density in fingerprints)
   - QED Target: Adaptive (relative to parent QED, allowing beyond-Ro5 therapeutics)
   - SA Target: <= 4.5
   - Crossover: Scaffold-Preserving & Mass-Balanced prioritized to preserve complex core scaffolds.
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import evolutionary_generator as EG

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Size Tier Definitions
TIER_PARAMS = {
    "small": {
        "mw_min": 150.0,
        "mw_max": 350.0,
        "sim_lo": 0.35,
        "sim_hi": 0.85,
        "qed_min": 0.45,
        "sa_max": 3.2,
        "crossover_op": "mass-balanced"
    },
    "medium": {
        "mw_min": 350.0,
        "mw_max": 550.0,
        "sim_lo": 0.45,
        "sim_hi": 0.90,
        "qed_min": 0.35,
        "sa_max": 3.8,
        "crossover_op": "mass-balanced"
    },
    "large": {
        "mw_min": 550.0,
        "mw_max": 1000.0,
        "sim_lo": 0.55,
        "sim_hi": 0.95,
        "qed_min": 0.15, # Beyond Rule of 5 natural products / macrocycles have lower baseline QED
        "sa_max": 4.5,
        "crossover_op": "scaffold" # Lock complex Murcko core, mutate periphery
    }
}

_WORKER_RULE_LIB = None

def _init_worker(rules_path: str):
    global _WORKER_RULE_LIB
    RDLogger.DisableLog("rdApp.*")
    _WORKER_RULE_LIB = EG.RuleLibrary(rules_path)


def classify_size_tier(mw: float) -> str:
    if mw < 350.0:
        return "small"
    elif mw <= 550.0:
        return "medium"
    else:
        return "large"


def _worker_process_parent_tiered(args_tuple) -> List[Dict]:
    global _WORKER_RULE_LIB
    parent, tier, pop, generations, per_parent_keep, seed = args_tuple
    params = TIER_PARAMS[tier]

    try:
        parent_mol = Chem.MolFromSmiles(parent)
        parent_qed = float(Descriptors.qed(parent_mol))
        
        # Adaptive QED for large molecules (don't drop more than 0.25 below parent)
        effective_qed_min = min(params["qed_min"], max(0.12, parent_qed - 0.25))

        objectives = EG.Objectives(parent, params["sim_lo"], params["sim_hi"], adversarial=None)
        front = EG.build_and_run(parent, _WORKER_RULE_LIB, objectives,
                                 pop_size=pop, generations=generations, seed=seed,
                                 crossover_op=params["crossover_op"], compute_shape=False)
    except Exception:
        return []

    picked = []
    for d in front:
        sim = d["similarity_to_parent"]
        if not (params["sim_lo"] <= sim <= params["sim_hi"]):
            continue
        if d["qed"] < effective_qed_min:
            continue
        if d.get("sa_score") is not None and d["sa_score"] > params["sa_max"]:
            continue
        if d.get("structural_alerts", 0) > 0:
            continue
        picked.append(d)

    picked.sort(key=lambda d: sum(d["objectives"].values()))

    out_rows = []
    for d in picked[:per_parent_keep]:
        cmol = Chem.MolFromSmiles(d["smiles"])
        out_rows.append({
            "counterfeit_smiles": d["smiles"],
            "parent_smiles": parent,
            "size_tier": tier,
            "parent_mw": round(Descriptors.MolWt(parent_mol), 1),
            "counterfeit_mw": round(Descriptors.MolWt(cmol), 1) if cmol else None,
            "category": f"evolved-{tier}",
            "similarity_to_parent": d["similarity_to_parent"],
            "qed": d["qed"],
            "sa_score": d["sa_score"],
            "n_edited_atoms": len(d["edited_atoms"]),
            "edited_atoms": " ".join(map(str, d["edited_atoms"])),
        })
    return out_rows


def load_stratified_parents(path: str, tier: str, max_parents: int, seed: int) -> List[Tuple[str, str]]:
    """Loads authentic parents and stratifies them by MW size tier."""
    buckets: Dict[str, List[str]] = {"small": [], "medium": [], "large": []}
    seen = set()

    with open(path) as fh:
        for line in fh:
            s = line.strip().split()
            if not s or s[0].startswith("#"):
                continue
            smi = s[0]
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            c = Chem.MolToSmiles(m)
            if c in seen:
                continue
            seen.add(c)
            mw = Descriptors.MolWt(m)
            t = classify_size_tier(mw)
            buckets[t].append(c)

    rng = random.Random(seed)
    for t in buckets:
        rng.shuffle(buckets[t])

    logger.info(f"Loaded Pool from {path}: Small={len(buckets['small'])} | Medium={len(buckets['medium'])} | Large={len(buckets['large'])}")

    selected = []
    if tier in ["small", "medium", "large"]:
        for smi in buckets[tier][:max_parents]:
            selected.append((smi, tier))
    elif tier == "balanced":
        per_tier = max_parents // 3
        for t in ["small", "medium", "large"]:
            for smi in buckets[t][:per_tier]:
                selected.append((smi, t))
    else: # all
        all_mols = []
        for t, smis in buckets.items():
            for smi in smis:
                all_mols.append((smi, t))
        rng.shuffle(all_mols)
        selected = all_mols[:max_parents]

    logger.info(f"Selected {len(selected)} parents for size tier mode: '{tier}'")
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rules", default="mined_transformations.py")
    p.add_argument("--parents", default="public_molecules_100k.smi")
    p.add_argument("--size-tier", choices=["small", "medium", "large", "balanced", "all"],
                   default="balanced", help="Size tier to generate (default: balanced)")
    p.add_argument("--max-parents", type=int, default=300,
                   help="Total number of authentic parents (default: 300)")
    p.add_argument("--per-parent-keep", type=int, default=8)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    p.add_argument("--out", default="version_e_tiered")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    parents_list = load_stratified_parents(args.parents, args.size_tier, args.max_parents, args.seed)

    tasks = [
        (parent, tier, args.pop, args.generations, args.per_parent_keep, args.seed + i)
        for i, (parent, tier) in enumerate(parents_list)
    ]

    t0 = time.time()
    rows = []
    global_seen = set()

    logger.info(f"Starting Multi-Scale Size-Stratified Generation across {args.workers} cores...")

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(args.rules,)) as executor:
        futures = [executor.submit(_worker_process_parent_tiered, t) for t in tasks]
        
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            for r in res:
                smi = r["counterfeit_smiles"]
                if smi in global_seen:
                    continue
                global_seen.add(smi)
                rows.append(r)

            if done % 25 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                speed = done / max(elapsed, 0.001)
                logger.info(f"  [{done}/{len(tasks)} parents done ({speed:.1f} parents/sec)] -> {len(rows)} counterfeits")

    # Group metrics per size tier
    tier_stats = {}
    for t in ["small", "medium", "large"]:
        t_rows = [r for r in rows if r["size_tier"] == t]
        if t_rows:
            tier_stats[t] = {
                "count": len(t_rows),
                "mean_parent_mw": round(float(np.mean([r["parent_mw"] for r in t_rows])), 1),
                "mean_counterfeit_mw": round(float(np.mean([r["counterfeit_mw"] for r in t_rows if r["counterfeit_mw"]])), 1),
                "mean_similarity": round(float(np.mean([r["similarity_to_parent"] for r in t_rows])), 3),
                "mean_qed": round(float(np.mean([r["qed"] for r in t_rows])), 3),
                "mean_sa_score": round(float(np.mean([r["sa_score"] for r in t_rows])), 3),
                "mean_edited_atoms": round(float(np.mean([r["n_edited_atoms"] for r in t_rows])), 1),
            }

    # Write unified CSV
    cols = ["counterfeit_smiles", "parent_smiles", "size_tier", "parent_mw",
            "counterfeit_mw", "category", "similarity_to_parent", "qed",
            "sa_score", "n_edited_atoms", "edited_atoms"]
    csv_file = outdir / "counterfeits_size_stratified.csv"
    with open(csv_file, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write individual CSVs per tier
    for t in ["small", "medium", "large"]:
        t_rows = [r for r in rows if r["size_tier"] == t]
        if t_rows:
            with open(outdir / f"counterfeits_{t}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for r in t_rows:
                    w.writerow(r)

    summary = {
        "total_parents": len(parents_list),
        "total_counterfeits": len(rows),
        "tier_mode": args.size_tier,
        "tier_stats": tier_stats,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("=" * 80)
    logger.info(f"Size-Stratified Dataset Generation Complete: {len(rows)} counterfeits saved to {outdir}/")
    for t, s in tier_stats.items():
        logger.info(f"  [{t.upper()}] Count={s['count']} | Parent MW={s['mean_parent_mw']} Da | Counterfeit MW={s['mean_counterfeit_mw']} Da | QED={s['mean_qed']} | SA={s['mean_sa_score']} | Sim={s['mean_similarity']}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
