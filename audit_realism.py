"""
Realism & bias audit of the generated datasets
===============================================

Rather than *assume* the counterfeits are realistic, this measures it, using
the metrics established for de-novo generation (GuacaMol / MOSES) and for
benchmark-decoy bias (DUD-E / DEKOIS, DeepCoy). For each dataset it reports:

Generation quality (are the counterfeits well-formed and diverse?)
  - validity        : fraction parseable by RDKit (should be ~1.0)
  - uniqueness      : fraction of distinct counterfeit SMILES
  - novelty         : fraction of counterfeits absent from the authentic pool
  - scaffold_div    : distinct Bemis-Murcko scaffolds / #counterfeits

Realism (are they plausible counterfeits of a real drug?)
  - sim_to_parent   : Tanimoto(counterfeit, parent) — realistic counterfeits sit
                      close to the drug they imitate; we report the mean and the
                      share in [0.4,0.7), [0.7,0.9), [0.9,1.0)
  - qed / sa        : drug-likeness and synthetic accessibility vs the authentic
                      class (a realistic counterfeit is as drug-like/makeable)

Bias (is the task trivially separable = artificial enrichment / analogue bias?)
  - prop_shortcut   : accuracy of a logistic regression on MW/heavy/length only
                      (target ~0.5 = classes not separable on gross properties)
  - fp_separability : accuracy + ROC-AUC of a Random-Forest on Morgan
                      fingerprints (this IS the fingerprint baseline of C2; very
                      high = the edit is a fingerprint giveaway, moderate =
                      genuine but non-trivial signal)

Usage
-----
    python audit_realism.py --datasets pooled evolved category_bioisostere
    python audit_realism.py                       # all datasets in benchmark/
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
except Exception:
    sascorer = None


def fp(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048)


def fp_array(smiles_list: List[str]) -> np.ndarray:
    out = []
    for s in smiles_list:
        f = fp(s)
        arr = np.zeros((2048,), dtype=np.int8)
        if f is not None:
            DataStructs.ConvertToNumpyArray(f, arr)
        out.append(arr)
    return np.array(out)


def scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles)
    except Exception:
        return smiles


def audit_dataset(path: Path, sample: int, seed: int) -> Dict:
    rows = list(csv.DictReader(open(path)))
    auth = [r for r in rows if r["label"] == "0"]
    fake = [r for r in rows if r["label"] == "1"]
    if not fake or not auth:
        return {}
    rng = random.Random(seed)

    fake_smi = [r["smiles"] for r in fake]
    auth_smi = set(r["smiles"] for r in auth)

    # --- generation quality ---
    valid = [s for s in fake_smi if Chem.MolFromSmiles(s) is not None]
    uniq = set(fake_smi)
    novelty = sum(1 for s in uniq if s not in auth_smi) / max(len(uniq), 1)
    scaffs = set(scaffold(s) for s in uniq)

    # --- realism: similarity to parent ---
    sims = []
    for r in fake:
        p = r.get("parent_smiles", "")
        if not p:
            continue
        fpm, fpp = fp(r["smiles"]), fp(p)
        if fpm is not None and fpp is not None:
            sims.append(DataStructs.TanimotoSimilarity(fpm, fpp))
    sims = np.array(sims) if sims else np.array([np.nan])

    def band(lo, hi):
        return float(np.mean((sims >= lo) & (sims < hi))) if sims.size else None

    # --- realism: druglikeness ---
    def mean_prop(smis, fn):
        vals = []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                try:
                    vals.append(fn(m))
                except Exception:
                    pass
        return round(float(np.mean(vals)), 3) if vals else None

    qed_fake = mean_prop(fake_smi, Descriptors.qed)
    qed_auth = mean_prop([r["smiles"] for r in auth], Descriptors.qed)
    sa_fake = mean_prop(fake_smi, sascorer.calculateScore) if sascorer else None
    sa_auth = mean_prop([r["smiles"] for r in auth], sascorer.calculateScore) if sascorer else None

    # --- bias: property shortcut + fingerprint separability ---
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # balance + subsample for speed
    k = min(len(auth), len(fake), sample)
    a = rng.sample(auth, k)
    f = rng.sample(fake, k)

    def prop_row(r):
        m = Chem.MolFromSmiles(r["smiles"])
        return [Descriptors.MolWt(m), m.GetNumHeavyAtoms(), len(r["smiles"])]

    Xp = np.array([prop_row(r) for r in a + f], float)
    y = np.array([0] * k + [1] * k)
    Xtr, Xte, ytr, yte = train_test_split(Xp, y, test_size=0.3, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)
    prop_acc = round(float(lr.score(sc.transform(Xte), yte)), 3)

    Xf = fp_array([r["smiles"] for r in a + f])
    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.3, stratify=y, random_state=seed)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    fp_acc = round(float(rf.score(Xte, yte)), 3)
    try:
        fp_auc = round(float(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])), 3)
    except Exception:
        fp_auc = None

    return {
        "n_counterfeit": len(fake),
        "validity": round(len(valid) / len(fake_smi), 3),
        "uniqueness": round(len(uniq) / len(fake_smi), 3),
        "novelty": round(novelty, 3),
        "scaffold_diversity": round(len(scaffs) / max(len(uniq), 1), 3),
        "sim_to_parent_mean": round(float(np.nanmean(sims)), 3),
        "sim_band_0.4_0.7": band(0.4, 0.7),
        "sim_band_0.7_0.9": band(0.7, 0.9),
        "sim_band_0.9_1.0": band(0.9, 1.001),
        "qed_fake": qed_fake, "qed_auth": qed_auth,
        "sa_fake": sa_fake, "sa_auth": sa_auth,
        "prop_shortcut_acc": prop_acc,
        "fp_separability_acc": fp_acc,
        "fp_separability_auc": fp_auc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="benchmark")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Dataset names to audit (default: all)")
    ap.add_argument("--sample", type=int, default=800, help="Max per class for classifiers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmark/realism_audit.json")
    args = ap.parse_args()

    paths = []
    if args.datasets:
        paths = [Path(args.root) / d / "dataset.csv" for d in args.datasets]
    else:
        paths = [Path(p) for p in sorted(glob.glob(f"{args.root}/*/dataset.csv"))]

    results = {}
    for p in paths:
        name = p.parent.name
        logger.info(f"Auditing {name} ...")
        r = audit_dataset(p, args.sample, args.seed)
        if r:
            results[name] = r

    Path(args.out).write_text(json.dumps(results, indent=2))

    # report
    logger.info("=" * 100)
    logger.info(f"{'dataset':<34}{'valid':>6}{'uniq':>6}{'novel':>6}{'scafDiv':>8}"
                f"{'simP':>6}{'QEDf/a':>10}{'propLR':>7}{'fpAUC':>7}")
    logger.info("-" * 100)
    for name, r in results.items():
        logger.info(f"{name:<34}{r['validity']:>6}{r['uniqueness']:>6}{r['novelty']:>6}"
                    f"{r['scaffold_diversity']:>8}{r['sim_to_parent_mean']:>6}"
                    f"{str(r['qed_fake'])+'/'+str(r['qed_auth']):>10}"
                    f"{r['prop_shortcut_acc']:>7}{str(r['fp_separability_auc']):>7}")
    logger.info("=" * 100)
    logger.info("Reading guide: validity/uniqueness/novelty ~1 good; simP (sim-to-parent) "
                "~0.5-0.8 realistic; QEDf≈QEDa = counterfeits as drug-like as reals; "
                "propLR ~0.5 = no property shortcut; fpAUC high = edit is a fingerprint "
                "giveaway (expected — that's the local signal), very high may mean too-easy.")
    logger.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
