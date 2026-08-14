"""
Generalization experiment: does the detector catch modifications it never saw?
=============================================================================

The benchmark's central question (proposal C1): a detector trained on the
modifications we *enumerated* — does it still flag modifications *discovered*
some other way? We answer it with the trained HGT detector by measuring its
COUNTERFEIT DETECTION RATE (fraction of counterfeits it correctly scores as
counterfeit, P(counterfeit) > 0.5) on three increasingly out-of-distribution
counterfeit sets:

    R  : hand-written MMP rules   (the detector's own training distribution)
    M  : data-mined MMP rules     (mild distribution shift), broken out per
                                    transformation category
    E  : NSGA-II evolved, adversarial (large shift)

A detection rate that falls R -> M -> E is the generalization gap, quantified.
Per-category rates on M show what a single pooled F1 hides.

Honesty notes printed with the results:
  * R is in-distribution (upper bound; these molecules shaped the model).
  * E was optimized white-box against THIS detector, so its low rate is partly
    by construction — it is an upper bound on adversarial vulnerability.
  * M was NOT optimized against the detector, so M is the cleanest
    generalization signal.

Usage
-----
    python evaluate_generalization.py --sample 300
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rdkit import Chem, RDLogger

import evolutionary_generator as EG

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_R_counterfeits(pt_path: str, n: int) -> List[str]:
    """Counterfeit SMILES (label==1) from the original dataset = Version R."""
    import torch
    graphs, labels, _ = torch.load(pt_path, weights_only=False)
    smis = [g.smiles for g, l in zip(graphs, labels)
            if int(l) == 1 and getattr(g, "smiles", None)]
    random.shuffle(smis)
    return smis[:n]


def load_M_by_category(n_per_cat: int) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for f in glob.glob("version_m/counterfeits_*.csv"):
        cat = Path(f).stem.replace("counterfeits_", "")
        if cat == "all":
            continue
        rows = list(csv.DictReader(open(f)))
        random.shuffle(rows)
        out[cat] = [r["counterfeit_smiles"] for r in rows[:n_per_cat]]
    return out


def load_E_counterfeits(n: int) -> List[str]:
    rows = list(csv.DictReader(open("version_e/counterfeits_evolved.csv")))
    random.shuffle(rows)
    return [r["counterfeit_smiles"] for r in rows[:n]]


def detection_rate(scorer, smiles_list: List[str]) -> Dict:
    """Fraction correctly flagged as counterfeit (P>0.5), plus mean P."""
    ps = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        ps.append(scorer(smi))
    if not ps:
        return {"n": 0, "detection_rate": None, "mean_p_counterfeit": None}
    detected = sum(1 for p in ps if p > 0.5)
    return {
        "n": len(ps),
        "detection_rate": round(detected / len(ps), 3),
        "mean_p_counterfeit": round(sum(ps) / len(ps), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="intelligent_pharma_50k_v2.pt")
    ap.add_argument("--sample", type=int, default=300,
                    help="Molecules sampled per set / per category")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="generalization_results.json")
    args = ap.parse_args()
    random.seed(args.seed)

    scorer = EG.load_detector_scorer()
    if scorer is None:
        logger.error("Detector unavailable; cannot run the experiment.")
        return

    logger.info("Scoring Version R (in-distribution) ...")
    R = detection_rate(scorer, load_R_counterfeits(args.pt, args.sample))

    logger.info("Scoring Version M per category ...")
    M_by_cat = {cat: detection_rate(scorer, smis)
                for cat, smis in load_M_by_category(args.sample).items()}

    logger.info("Scoring Version E (evolved adversarial) ...")
    E = detection_rate(scorer, load_E_counterfeits(args.sample))

    results = {"R": R, "M_by_category": M_by_cat, "E": E, "sample": args.sample}
    Path(args.out).write_text(json.dumps(results, indent=2))

    # ---- report ----
    logger.info("=" * 62)
    logger.info("COUNTERFEIT DETECTION RATE  (higher = detector catches them)")
    logger.info("-" * 62)
    logger.info(f"  {'set':<28}{'n':>5}{'detect%':>10}{'mean P':>10}")
    logger.info(f"  {'R  hand-written (in-dist)':<28}{R['n']:>5}"
                f"{R['detection_rate']*100:>9.1f}%{R['mean_p_counterfeit']:>10}")
    m_all = []
    for cat, s in sorted(M_by_cat.items(), key=lambda kv: kv[1]['detection_rate'] or 0):
        if s["detection_rate"] is None:
            continue
        m_all.append(s["detection_rate"])
        logger.info(f"  {'M  ' + cat:<28}{s['n']:>5}"
                    f"{s['detection_rate']*100:>9.1f}%{s['mean_p_counterfeit']:>10}")
    logger.info(f"  {'E  evolved adversarial':<28}{E['n']:>5}"
                f"{E['detection_rate']*100:>9.1f}%{E['mean_p_counterfeit']:>10}")
    logger.info("-" * 62)
    if m_all:
        gap = R["detection_rate"] - min(m_all)
        logger.info(f"  Worst-category gap vs in-distribution R: "
                    f"{gap*100:.1f} percentage points")
    logger.info(f"  E vs R gap: {(R['detection_rate']-E['detection_rate'])*100:.1f} pts "
                f"(white-box; upper bound on vulnerability)")
    logger.info("=" * 62)
    logger.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
