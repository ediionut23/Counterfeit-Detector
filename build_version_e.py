"""
Build "Version E" — the evolved, adversarial counterfeit dataset
================================================================

Version R uses hand-written rules; Version M uses data-mined rules; Version E
does not stay inside any rule list at all. For each genuine parent drug it runs
the multi-objective genetic search of `evolutionary_generator.py` with the
trained detector in the loop, and keeps the Pareto-front counterfeits that are
simultaneously (a) in the similarity band, (b) drug-like, and (c) hard for the
detector (low P(counterfeit)) — i.e. plausible modifications the current model
misclassifies as authentic.

This is the dataset the benchmark's central question needs: train a detector on
the modifications we *enumerated* (R / M) and test on modifications *discovered*
adversarially (E) to measure the generalization gap.

The detector is loaded ONCE and shared across all parents. Parents are read
offline from a cached SMILES file of authentic, publicly-sourced drugs
(ChEMBL / PubChem, via mine_mmp_rules.py) or from a .pt dataset.

Usage
-----
    python build_version_e.py --parents mmp_mining/input.smi \
        --max-parents 150 --per-parent-keep 10 --pop 24 --generations 8 \
        --out version_e
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, RDLogger

import evolutionary_generator as EG

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


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
        path = parents_file or "mmp_mining/input.smi"
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
    logger.info(f"Loaded {len(out)} unique authentic parents")
    return out


def keep_from_front(front: List[Dict], sim_lo: float, sim_hi: float,
                    adversarial: bool, p_max: float, qed_min: float,
                    keep: int) -> List[Dict]:
    """Select the 'promising' counterfeits from one parent's Pareto front."""
    picked = []
    for d in front:
        if not (sim_lo <= d["similarity_to_parent"] <= sim_hi):
            continue
        if d["qed"] < qed_min:
            continue
        if adversarial and d["objectives"].get("p_counterfeit", 1.0) > p_max:
            continue
        picked.append(d)
    # most adversarial first (lowest P), else best summed objectives
    if adversarial:
        picked.sort(key=lambda d: d["objectives"].get("p_counterfeit", 1.0))
    else:
        picked.sort(key=lambda d: sum(d["objectives"].values()))
    return picked[:keep]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rules", default="mined_transformations.py")
    p.add_argument("--parents", default=None,
                   help="SMILES file of authentic parents (default mmp_mining/input.smi)")
    p.add_argument("--from-pt", default=None)
    p.add_argument("--max-parents", type=int, default=150)
    p.add_argument("--per-parent-keep", type=int, default=10)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--sim-lo", type=float, default=0.4)
    p.add_argument("--sim-hi", type=float, default=0.9)
    p.add_argument("--no-adversarial", action="store_true",
                   help="Disable the detector objective (structural E only)")
    p.add_argument("--p-max", type=float, default=1.0,
                   help="Hard cap on P(counterfeit) to keep (default 1.0 = no "
                        "cap; kept counterfeits are always ranked most-adversarial "
                        "first, so tighten this only to force fooling-only sets)")
    p.add_argument("--qed-min", type=float, default=0.3)
    p.add_argument("--out", default="version_e")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    rule_lib = EG.RuleLibrary(args.rules)
    parents = load_parents(args.parents, args.from_pt, args.max_parents, args.seed)

    adversarial = not args.no_adversarial
    scorer = EG.load_detector_scorer() if adversarial else None
    if adversarial and scorer is None:
        logger.warning("Detector unavailable -> Version E will be structural only.")
        adversarial = False

    global_seen: set = set()
    rows: List[Dict] = []
    n_fooling = 0

    for pi, parent in enumerate(parents):
        try:
            objectives = EG.Objectives(parent, args.sim_lo, args.sim_hi,
                                       adversarial=scorer)
            front = EG.build_and_run(parent, rule_lib, objectives,
                                     pop_size=args.pop, generations=args.generations,
                                     seed=args.seed)
        except Exception as e:
            logger.warning(f"[parent {pi}] search failed: {e}")
            continue

        picked = keep_from_front(front, args.sim_lo, args.sim_hi, adversarial,
                                 args.p_max, args.qed_min, args.per_parent_keep)
        for d in picked:
            smi = d["smiles"]
            if smi in global_seen:
                continue
            global_seen.add(smi)
            pc = d["objectives"].get("p_counterfeit")
            if pc is not None and pc < 0.5:
                n_fooling += 1
            rows.append({
                "counterfeit_smiles": smi,
                "parent_smiles": parent,
                "category": "evolved-adversarial" if adversarial else "evolved",
                "similarity_to_parent": d["similarity_to_parent"],
                "qed": d["qed"],
                "sa_score": d["sa_score"],
                "p_counterfeit": round(pc, 4) if pc is not None else None,
                "n_edited_atoms": len(d["edited_atoms"]),
                "edited_atoms": " ".join(map(str, d["edited_atoms"])),
            })

        if (pi + 1) % 10 == 0:
            logger.info(f"  ...{pi+1}/{len(parents)} parents, "
                        f"{len(rows)} counterfeits so far "
                        f"({n_fooling} fool the detector)")

    # write outputs
    cols = ["counterfeit_smiles", "parent_smiles", "category",
            "similarity_to_parent", "qed", "sa_score", "p_counterfeit",
            "n_edited_atoms", "edited_atoms"]
    with open(outdir / "counterfeits_evolved.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    sims = [r["similarity_to_parent"] for r in rows] or [0]
    pcs = [r["p_counterfeit"] for r in rows if r["p_counterfeit"] is not None]
    summary = {
        "n_parents": len(parents),
        "n_counterfeits": len(rows),
        "adversarial": adversarial,
        "n_fool_detector": n_fooling,
        "frac_fool_detector": round(n_fooling / max(len(rows), 1), 3),
        "mean_similarity": round(float(np.mean(sims)), 3),
        "mean_p_counterfeit": round(float(np.mean(pcs)), 3) if pcs else None,
        "similarity_band": [args.sim_lo, args.sim_hi],
        "search": {"pop": args.pop, "generations": args.generations},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("=" * 60)
    logger.info(f"Version E built: {len(rows)} evolved counterfeits from "
                f"{len(parents)} parents -> {outdir}/")
    logger.info(f"  fooling the detector (P<0.5): {n_fooling} "
                f"({summary['frac_fool_detector']*100:.0f}%)")
    if pcs:
        logger.info(f"  mean P(counterfeit): {summary['mean_p_counterfeit']}  "
                    f"mean similarity: {summary['mean_similarity']}")


if __name__ == "__main__":
    main()
