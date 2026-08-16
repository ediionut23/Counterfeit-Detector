"""
Learnability check: can a model tell counterfeit from authentic at all?
=======================================================================

Before any GNN, the first question is whether the task is learnable — i.e.
whether authentic and counterfeit molecules actually differ in a way a model can
pick up, or whether (after property matching) they are effectively
indistinguishable. This trains the C2 "lower-tier" baselines and answers that.

For each dataset it trains, on the dataset's OWN scaffold split (train on
train+val, test on the scaffold-disjoint test set):

  * property-only baseline  — logistic regression on MW / heavy atoms / logP.
    This SHOULD be near chance (property matching removed those shortcuts); if it
    is, good — it means any real signal must come from structure.
  * fingerprint baseline     — Random Forest on 2048-bit Morgan fingerprints.
    If this beats chance on unseen scaffolds, the local edit is a learnable
    structural signal — the task is real. This is also the C2 fingerprint
    baseline a GNN must beat to justify itself.

Metrics (proposal C4 order): False Negative Rate first (a missed counterfeit is
the costly error), then Precision, Recall, F1, plus accuracy and ROC-AUC.

Usage
-----
    python train_baseline.py --dataset pooled
    python train_baseline.py --all
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def morgan_array(smiles: str) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros((2048,), dtype=np.int8)
    if m is not None:
        DataStructs.ConvertToNumpyArray(
            rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048), arr)
    return arr


def prop_row(smiles: str) -> List[float]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return [0.0, 0.0, 0.0]
    return [Descriptors.MolWt(m), float(m.GetNumHeavyAtoms()), Descriptors.MolLogP(m)]


def load_split(path: Path) -> Dict[str, List[dict]]:
    rows = list(csv.DictReader(open(path)))
    train = [r for r in rows if r["split"] in ("train", "val")]
    test = [r for r in rows if r["split"] == "test"]
    return {"train": train, "test": test}


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    # counterfeit = positive class (label 1)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n_pos = int((y_true == 1).sum())
    fnr = fn / n_pos if n_pos else None  # missed counterfeits
    out = {
        "false_negative_rate": round(fnr, 3) if fnr is not None else None,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "accuracy": round(float((y_pred == y_true).mean()), 3),
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 3)
    except Exception:
        out["roc_auc"] = None
    return out


def train_eval(path: Path) -> Dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sp = load_split(path)
    if len(sp["test"]) < 20 or len(sp["train"]) < 40:
        return {}

    ytr = np.array([int(r["label"]) for r in sp["train"]])
    yte = np.array([int(r["label"]) for r in sp["test"]])

    # --- property-only baseline (should be ~chance after matching) ---
    Xtr_p = np.array([prop_row(r["smiles"]) for r in sp["train"]], float)
    Xte_p = np.array([prop_row(r["smiles"]) for r in sp["test"]], float)
    sc = StandardScaler().fit(Xtr_p)
    lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr_p), ytr)
    p_score = lr.predict_proba(sc.transform(Xte_p))[:, 1]
    prop_m = metrics(yte, (p_score > 0.5).astype(int), p_score)

    # --- fingerprint baseline (the real learnability test) ---
    Xtr_f = np.array([morgan_array(r["smiles"]) for r in sp["train"]])
    Xte_f = np.array([morgan_array(r["smiles"]) for r in sp["test"]])
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42).fit(Xtr_f, ytr)
    f_score = rf.predict_proba(Xte_f)[:, 1]
    fp_m = metrics(yte, (f_score > 0.5).astype(int), f_score)

    return {
        "n_train": len(sp["train"]), "n_test": len(sp["test"]),
        "property_only": prop_m,
        "fingerprint_rf": fp_m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="benchmark")
    ap.add_argument("--dataset", default=None, help="single dataset name")
    ap.add_argument("--all", action="store_true", help="run every dataset")
    ap.add_argument("--out", default="baseline_results.json")
    args = ap.parse_args()

    if args.dataset:
        paths = [Path(args.root) / args.dataset / "dataset.csv"]
    elif args.all:
        paths = [Path(p) for p in sorted(glob.glob(f"{args.root}/*/dataset.csv"))]
    else:
        paths = [Path(args.root) / d / "dataset.csv"
                 for d in ("pooled", "evolved", "category_bioisostere",
                           "category_scaffold-hop", "category_halogen-walk",
                           "category_homologation")]

    results = {}
    for p in paths:
        if not p.exists():
            continue
        name = p.parent.name
        logger.info(f"Training baselines on {name} ...")
        r = train_eval(p)
        if r:
            results[name] = r
    Path(args.out).write_text(json.dumps(results, indent=2))

    # report
    logger.info("=" * 96)
    logger.info("LEARNABILITY on scaffold-disjoint test set  (FNR = missed counterfeits, lower better)")
    logger.info(f"{'dataset':<30}{'ntest':>6} | {'PROPERTY-ONLY':^22} | {'FINGERPRINT RF':^28}")
    logger.info(f"{'':<30}{'':>6} | {'F1':>6}{'AUC':>7}{'acc':>7} | {'FNR':>6}{'F1':>7}{'AUC':>7}{'acc':>7}")
    logger.info("-" * 96)
    for name, r in results.items():
        p, f = r["property_only"], r["fingerprint_rf"]
        logger.info(f"{name:<30}{r['n_test']:>6} | "
                    f"{p['f1']:>6}{str(p['roc_auc']):>7}{p['accuracy']:>7} | "
                    f"{str(f['false_negative_rate']):>6}{f['f1']:>7}"
                    f"{str(f['roc_auc']):>7}{f['accuracy']:>7}")
    logger.info("=" * 96)
    logger.info("Read: property-only AUC ~0.5 = matching worked (no gross-property shortcut). "
                "Fingerprint AUC >> 0.5 on UNSEEN scaffolds = the edit is a real, learnable "
                "structural signal => the task is learnable, molecules DO differ.")
    logger.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
