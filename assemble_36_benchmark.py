"""
Assemble Complete 36-Category Benchmark Datasets
================================================
Combines all counterfeit libraries:
  - 100K catalog (counterfeits_100k/counterfeits_100k.csv)
  - Boosted subcategories (counterfeits_boosted.csv)
  - 100K Evolved set (version_e_100k/counterfeits_evolved.csv)
  - Prior Version M sets (version_m/counterfeits_*.csv)
  - Authentic negatives from public_molecules_100k.smi (58.6k drugs)

Assembles:
  - 36 fine-grained subcategory datasets (subcategory_<cat>.<subcat>)
  - 5 macro-category datasets (category_<cat>)
  - 1 pure evolutionary dataset (evolved)
  - 1 pooled dataset (pooled)

Enforces:
  - DUD-E style property matching (MW, heavy atoms, SMILES length) -> Cohen's d ~ 0
  - Bemis-Murcko scaffold disjoint train/val/test splits (80/10/10)
  - Preserved modified atom indices for XAI evaluation
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("assemble_36_benchmark")


def descriptors(smiles: str) -> Optional[Dict]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return {
            "mw": float(Descriptors.MolWt(m)),
            "logp": float(Descriptors.MolLogP(m)),
            "qed": float(Descriptors.qed(m)),
            "heavy_atoms": m.GetNumHeavyAtoms(),
            "smiles_length": len(smiles),
        }
    except Exception:
        return None


def murcko(smiles: str) -> str:
    try:
        m = Chem.MolFromSmiles(smiles)
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s.GetNumAtoms() else smiles
    except Exception:
        return smiles


def property_match(auth: List[Dict], fake: List[Dict],
                   cols=("mw", "heavy_atoms", "smiles_length"),
                   n_bins: int = 6, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Keep equal number of authentic and counterfeit molecules per property bin."""
    rng = random.Random(seed)
    combined = np.array([[r[c] for c in cols] for r in auth + fake], dtype=float)
    edges = {}
    for j, c in enumerate(cols):
        q = np.quantile(combined[:, j], np.linspace(0, 1, n_bins + 1))
        q[0] -= 1e-6
        q[-1] += 1e-6
        edges[c] = q

    def bin_of(r):
        return tuple(int(np.digitize(r[c], edges[c]) - 1) for c in cols)

    a_by, f_by = defaultdict(list), defaultdict(list)
    for r in auth:
        a_by[bin_of(r)].append(r)
    for r in fake:
        f_by[bin_of(r)].append(r)

    keep_a, keep_f = [], []
    for b in set(a_by) | set(f_by):
        a, f = a_by.get(b, []), f_by.get(b, [])
        k = min(len(a), len(f))
        if k == 0:
            continue
        rng.shuffle(a)
        rng.shuffle(f)
        keep_a.extend(a[:k])
        keep_f.extend(f[:k])
    return keep_a, keep_f


def shortcut_metrics(auth: List[Dict], fake: List[Dict]) -> Dict:
    def cohens_d(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        return float(abs(a.mean() - b.mean()) / max(pooled, 1e-9))

    cols = ["mw", "heavy_atoms", "smiles_length"]
    out = {f"cohens_d_{c}": round(cohens_d([r[c] for r in auth], [r[c] for r in fake]), 3) for c in cols}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = np.array([[r[c] for c in cols] for r in auth + fake], float)
        y = np.array([0] * len(auth) + [1] * len(fake))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)
        out["trivial_lr_accuracy"] = round(float(lr.score(sc.transform(Xte), yte)), 3)
    except Exception:
        out["trivial_lr_accuracy"] = None
    return out


def scaffold_split(rows: List[Dict], frac=(0.8, 0.1, 0.1), seed: int = 42) -> None:
    groups = defaultdict(list)
    for r in rows:
        ref = r.get("parent_smiles") or r["smiles"]
        groups[murcko(ref)].append(r)
    scaffs = list(groups.values())
    random.Random(seed).shuffle(scaffs)
    n = len(rows)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    counts = {"train": 0, "val": 0, "test": 0}
    for grp in sorted(scaffs, key=len, reverse=True):
        if counts["train"] < n_train:
            split = "train"
        elif counts["val"] < n_val:
            split = "val"
        else:
            split = "test"
        for r in grp:
            r["split"] = split
        counts[split] += len(grp)


def load_authentic_library(path: str) -> List[Dict]:
    logger.info(f"Loading authentic pool from {path}...")
    rows = []
    seen = set()
    with open(path) as fh:
        for line in fh:
            s = line.strip().split()
            if not s:
                continue
            smi = s[0]
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            c = Chem.MolToSmiles(m)
            if c in seen:
                continue
            seen.add(c)
            d = descriptors(c)
            if d is None:
                continue
            rows.append({
                "smiles": c,
                "label": 0,
                "category": "authentic",
                "subcategory": "authentic",
                "parent_smiles": "",
                "edited_atoms": "",
                **d
            })
    random.Random(0).shuffle(rows)
    logger.info(f"Authentic pool: {len(rows):,} valid molecules.")
    return rows


COLS = [
    "id", "smiles", "label", "category", "subcategory", "parent_smiles",
    "edited_atoms", "scaffold", "split", "mw", "logp", "qed",
    "heavy_atoms", "smiles_length"
]


def assemble_slice(
    name: str,
    counterfeits: List[Dict],
    authentic_pool: List[Dict],
    outdir: Path,
    seed: int = 42
) -> Optional[Dict]:
    # Deduplicate counterfeits by SMILES
    seen, fake = set(), []
    for r in counterfeits:
        smi = r["smiles"]
        if smi in seen:
            continue
        seen.add(smi)
        fake.append(r)

    if len(fake) < 10:
        logger.warning(f"  [{name}] Too few counterfeits ({len(fake)}), skipping.")
        return None

    auth_matched, fake_matched = property_match(authentic_pool, fake, seed=seed)
    k = min(len(auth_matched), len(fake_matched))
    if k < 10:
        logger.warning(f"  [{name}] Too small after matching (k={k}), skipping.")
        return None

    rng = random.Random(seed)
    rng.shuffle(auth_matched)
    rng.shuffle(fake_matched)
    auth_matched, fake_matched = auth_matched[:k], fake_matched[:k]

    metrics = shortcut_metrics(auth_matched, fake_matched)
    rows = auth_matched + fake_matched
    for i, r in enumerate(rows):
        r["id"] = f"{name}_{i}"
        r["scaffold"] = murcko(r.get("parent_smiles") or r["smiles"])

    scaffold_split(rows, seed=seed)
    rng.shuffle(rows)

    ds_dir = outdir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    with open(ds_dir / "dataset.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "name": name,
        "n_total": len(rows),
        "n_authentic": k,
        "n_counterfeit": k,
        "balance": 1.0,
        "shortcut_metrics": metrics,
        "split_counts": {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")},
    }
    (ds_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"  ✓ [{name}]: {len(rows):,} mols (1:1), LR acc={metrics.get('trivial_lr_accuracy')}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Assemble 36-Category Benchmark Datasets.")
    parser.add_argument("--out", type=str, default="benchmark")
    parser.add_argument("--authentic", type=str, default="public_molecules_100k.smi")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import mined_transformations as mt

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    authentic_pool = load_authentic_library(args.authentic)

    # Collect all counterfeits across all sources
    logger.info("Collecting counterfeits across sources...")
    by_category = defaultdict(list)
    by_subcat = defaultdict(list)
    evolved = []
    pooled = []
    seen_fakes = set()

    def add_counterfeit(smi, cat, subcat, parent_smi, ed_atoms, d=None):
        if smi in seen_fakes:
            return
        if d is None:
            d = descriptors(smi)
        if d is None:
            return
        seen_fakes.add(smi)
        row = {
            "smiles": smi,
            "label": 1,
            "category": cat,
            "subcategory": subcat,
            "parent_smiles": parent_smi,
            "edited_atoms": ed_atoms,
            **d
        }
        by_category[cat].append(row)
        by_subcat[f"{cat}.{subcat}"].append(row)
        if cat == "evolved":
            evolved.append(row)
        pooled.append(row)

    # 1. Boosted counterfeits
    if os.path.exists("counterfeits_boosted.csv"):
        logger.info("Loading counterfeits_boosted.csv...")
        with open("counterfeits_boosted.csv") as f:
            for r in csv.DictReader(f):
                add_counterfeit(
                    r["counterfeit_smiles"], r["category"], r["subcategory"],
                    r.get("parent_smiles", ""), r.get("edited_atoms", "")
                )

    # 2. 100K Counterfeits
    if os.path.exists("counterfeits_100k/counterfeits_100k.csv"):
        logger.info("Loading counterfeits_100k/counterfeits_100k.csv...")
        with open("counterfeits_100k/counterfeits_100k.csv") as f:
            for r in csv.DictReader(f):
                cat = r["category"]
                rule = r.get("rule_name", "")
                subcat = mt.SUBCATEGORY.get(rule, "other" if cat == "other" else "unknown")
                add_counterfeit(
                    r["counterfeit_smiles"], cat, subcat,
                    r.get("parent_smiles", ""), r.get("edited_atoms", "")
                )

    # 3. 100K Evolved set
    if os.path.exists("version_e_100k/counterfeits_evolved.csv"):
        logger.info("Loading version_e_100k/counterfeits_evolved.csv...")
        with open("version_e_100k/counterfeits_evolved.csv") as f:
            for r in csv.DictReader(f):
                add_counterfeit(
                    r["counterfeit_smiles"], "evolved", "evolved",
                    r.get("parent_smiles", ""), r.get("edited_atoms", "")
                )

    # 4. Version M
    for f in glob.glob("version_m/counterfeits_*.csv"):
        if f.endswith("counterfeits_all.csv"):
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rule = r.get("rule_name", "")
                subcat = mt.SUBCATEGORY.get(rule, "unknown")
                add_counterfeit(
                    r["counterfeit_smiles"], r["category"], subcat,
                    r.get("parent_smiles", ""), r.get("edited_atoms", "")
                )

    # 5. Version E
    if os.path.exists("version_e/counterfeits_evolved.csv"):
        with open("version_e/counterfeits_evolved.csv") as f:
            for r in csv.DictReader(f):
                add_counterfeit(
                    r["counterfeit_smiles"], "evolved", "evolved",
                    r.get("parent_smiles", ""), r.get("edited_atoms", "")
                )

    logger.info(f"Total unique counterfeits loaded: {len(seen_fakes):,}")

    # Build slices
    slices = {}
    # Macro categories
    for cat, rows in by_category.items():
        if cat != "evolved":
            slices[f"category_{cat}"] = rows

    # Subcategories (the 36 subcategories)
    for sub, rows in by_subcat.items():
        if not sub.startswith("evolved"):
            slices[f"subcategory_{sub}"] = rows

    if evolved:
        slices["evolved"] = evolved
    slices["pooled"] = pooled

    logger.info(f"Assembling {len(slices)} benchmark datasets into {outdir}...")

    index = {}
    for name, fakes in slices.items():
        summary = assemble_slice(name, fakes, authentic_pool, outdir, seed=args.seed)
        if summary:
            index[name] = summary

    (outdir / "index.json").write_text(json.dumps(index, indent=2))
    logger.info("=" * 70)
    logger.info(f"Successfully assembled {len(index)} benchmark datasets in {outdir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
