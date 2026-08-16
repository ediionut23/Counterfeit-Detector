"""
Does a "distance-to-known-drug" signal help detect evolved counterfeits?
========================================================================

A counterfeit is a *perturbation of a real drug*: it should sit close to some
known authentic molecule without being it. A plain fingerprint classifier does
not use that relational signal. This adds it and measures whether it helps —
especially on Version E, where the plain fingerprint baseline was near chance.

For each molecule we add nearest-neighbour features against a reference set of
authentic molecules (the dataset's OWN training-split authentics, so there is no
test leakage; a molecule never matches itself):

  nn_max      — highest Tanimoto to any reference drug
  nn_top5_mean, nn_top5_std
  nn_gap      — nn_max minus 2nd-highest (isolated near-copy vs dense region)

We then compare, on the scaffold-disjoint test split:
  A) fingerprint only            (the previous baseline)
  B) nearest-neighbour features only
  C) fingerprint + nearest-neighbour

Usage
-----
    python learnability_nn.py --datasets evolved pooled category_bioisostere
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def fp(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048)


def fp_bits(smiles: str) -> np.ndarray:
    f = fp(smiles)
    arr = np.zeros((2048,), dtype=np.int8)
    if f is not None:
        DataStructs.ConvertToNumpyArray(f, arr)
    return arr


def nn_features(query_fp, ref_fps, self_smiles=None, ref_smiles=None) -> List[float]:
    """Nearest-neighbour Tanimoto stats of query against reference fingerprints,
    excluding an exact self-match by SMILES."""
    if query_fp is None or not ref_fps:
        return [0.0, 0.0, 0.0, 0.0]
    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps))
    if self_smiles is not None and ref_smiles is not None:
        for i, s in enumerate(ref_smiles):
            if s == self_smiles:
                sims[i] = -1.0  # drop self
    order = np.sort(sims)[::-1]
    top5 = order[:5]
    nn_max = float(order[0])
    nn_gap = float(order[0] - order[1]) if order.size > 1 else 0.0
    return [nn_max, float(top5.mean()), float(top5.std()), nn_gap]


def evaluate(path: Path) -> Dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, f1_score

    rows = list(csv.DictReader(open(path)))
    train = [r for r in rows if r["split"] in ("train", "val")]
    test = [r for r in rows if r["split"] == "test"]
    if len(test) < 20 or len(train) < 40:
        return {}

    # reference = authentic training molecules
    ref = [r["smiles"] for r in train if r["label"] == "0"]
    ref_fps = [fp(s) for s in ref]
    keep = [i for i, f in enumerate(ref_fps) if f is not None]
    ref_fps = [ref_fps[i] for i in keep]
    ref_smiles = [ref[i] for i in keep]

    def features(split_rows):
        Xfp, Xnn, y = [], [], []
        for r in split_rows:
            f = fp(r["smiles"])
            if f is None:
                continue
            Xfp.append(fp_bits(r["smiles"]))
            Xnn.append(nn_features(f, ref_fps, r["smiles"], ref_smiles))
            y.append(int(r["label"]))
        return np.array(Xfp), np.array(Xnn), np.array(y)

    Xtr_fp, Xtr_nn, ytr = features(train)
    Xte_fp, Xte_nn, yte = features(test)

    def run(Xtr, Xte, tag):
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42).fit(Xtr, ytr)
        s = rf.predict_proba(Xte)[:, 1]
        pred = (s > 0.5).astype(int)
        try:
            auc = round(float(roc_auc_score(yte, s)), 3)
        except Exception:
            auc = None
        fn = int(((pred == 0) & (yte == 1)).sum())
        npos = int((yte == 1).sum())
        return {"auc": auc, "f1": round(float(f1_score(yte, pred, zero_division=0)), 3),
                "fnr": round(fn / npos, 3) if npos else None}

    return {
        "n_test": len(yte),
        "A_fingerprint": run(Xtr_fp, Xte_fp, "fp"),
        "B_nn_only": run(Xtr_nn, Xte_nn, "nn"),
        "C_fingerprint_plus_nn": run(np.hstack([Xtr_fp, Xtr_nn]),
                                     np.hstack([Xte_fp, Xte_nn]), "fp+nn"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="benchmark")
    ap.add_argument("--datasets", nargs="*",
                    default=["evolved", "pooled", "category_bioisostere",
                             "category_scaffold-hop"])
    ap.add_argument("--out", default="learnability_nn_results.json")
    args = ap.parse_args()

    results = {}
    for d in args.datasets:
        p = Path(args.root) / d / "dataset.csv"
        if not p.exists():
            continue
        logger.info(f"Evaluating {d} ...")
        r = evaluate(p)
        if r:
            results[d] = r
    Path(args.out).write_text(json.dumps(results, indent=2))

    logger.info("=" * 82)
    logger.info("ROC-AUC on scaffold test  (does distance-to-known-drug help?)")
    logger.info(f"{'dataset':<28}{'ntest':>6}{'A: fp':>9}{'B: nn':>9}{'C: fp+nn':>10}{'Δ(C-A)':>9}")
    logger.info("-" * 82)
    for name, r in results.items():
        a = r["A_fingerprint"]["auc"] or 0
        b = r["B_nn_only"]["auc"] or 0
        c = r["C_fingerprint_plus_nn"]["auc"] or 0
        logger.info(f"{name:<28}{r['n_test']:>6}{a:>9}{b:>9}{c:>10}{round(c-a,3):>9}")
    logger.info("=" * 82)
    logger.info("B (nn only) captures the 'near a known drug' signal alone; "
                "C-A > 0 means adding it helps over plain fingerprints.")
    logger.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
