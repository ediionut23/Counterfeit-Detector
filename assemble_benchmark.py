"""
Assemble complete benchmark datasets from counterfeit sets (proposal C1)
========================================================================

Versions M and E are, so far, lists of *counterfeits only*. A usable benchmark
dataset needs the other half and a protocol. This script turns each counterfeit
slice into a complete, robust dataset:

  1. AUTHENTIC NEGATIVES     — real drugs (label 0) drawn from the same pool the
                               counterfeits were derived from.
  2. PROPERTY MATCHING       — DUD-E style (Mysinger et al., 2012): authentic and
                               counterfeit distributions are matched on molecular
                               weight, heavy-atom count and SMILES length, so no
                               classifier can separate the classes on a gross
                               property alone. Reported with Cohen's d and a
                               trivial logistic-regression accuracy (target ~0.5).
  3. SCAFFOLD SPLIT          — fixed train/val/test where whole Bemis-Murcko
                               scaffold families land on one side only
                               (MoleculeNet / OGB style), so a model is tested on
                               cores it never saw. Counterfeits inherit their
                               parent's scaffold, so a counterfeit and its parent
                               family never straddle the split.
  4. GROUND TRUTH PRESERVED  — every counterfeit keeps category, subcategory
                               (joined by rule name) and its modified-atom indices.
  5. 3D CONFORMERS (optional) — ETKDGv3 + MMFF per molecule, written to SDF, for
                               the geometry-based experiments in C1/C3.

It emits one dataset per transformation category, one per well-populated
subcategory, one for the evolved set (E), and one pooled dataset — each a CSV
table plus a shortcut-metrics JSON, under benchmark/.

Usage
-----
    python assemble_benchmark.py --authentic mmp_mining/input.smi \
        --min-subcat 4 --conformers
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
#  Molecule helpers
# ------------------------------------------------------------------
def descriptors(smiles: str) -> Optional[Dict]:
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


def murcko(smiles: str) -> str:
    try:
        m = Chem.MolFromSmiles(smiles)
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s.GetNumAtoms() else smiles
    except Exception:
        return smiles


# ------------------------------------------------------------------
#  DUD-E style property matching
# ------------------------------------------------------------------
def property_match(auth: List[Dict], fake: List[Dict],
                   cols=("mw", "heavy_atoms", "smiles_length"),
                   n_bins: int = 6, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Keep, per multi-dimensional property bin, an equal number of authentic and
    counterfeit molecules, so the two classes share the same marginal
    distributions on `cols`."""
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
    """Cohen's d on the matched properties + a trivial logistic-regression
    accuracy. Good matching => d ~ 0 and accuracy ~ 0.5."""
    def cohens_d(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        return float(abs(a.mean() - b.mean()) / max(pooled, 1e-9))

    cols = ["mw", "heavy_atoms", "smiles_length"]
    out = {f"cohens_d_{c}": round(cohens_d([r[c] for r in auth],
                                           [r[c] for r in fake]), 3) for c in cols}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        X = np.array([[r[c] for c in cols] for r in auth + fake], float)
        y = np.array([0] * len(auth) + [1] * len(fake))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              stratify=y, random_state=0)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)
        out["trivial_lr_accuracy"] = round(float(lr.score(sc.transform(Xte), yte)), 3)
    except Exception as e:
        out["trivial_lr_accuracy"] = None
    return out


# ------------------------------------------------------------------
#  Scaffold split (families disjoint across splits)
# ------------------------------------------------------------------
def scaffold_split(rows: List[Dict], frac=(0.8, 0.1, 0.1), seed: int = 42) -> None:
    """Assign a 'split' field. Grouping key = scaffold of the reference authentic
    molecule (parent for a counterfeit, self for an authentic), so entire
    scaffold families stay on one side."""
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
        # greedily place each scaffold family in the split that is most under quota
        if counts["train"] < n_train:
            split = "train"
        elif counts["val"] < n_val:
            split = "val"
        else:
            split = "test"
        for r in grp:
            r["split"] = split
        counts[split] += len(grp)
    logger.info(f"    split sizes: {counts}")


# ------------------------------------------------------------------
#  3D conformers (ETKDGv3 + MMFF) -> SDF
# ------------------------------------------------------------------
def write_conformers(rows: List[Dict], sdf_path: Path, max_mols: Optional[int]) -> int:
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    writer = Chem.SDWriter(str(sdf_path))
    n_ok = 0
    subset = rows if max_mols is None else rows[:max_mols]
    for r in subset:
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, params) != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(m)
        except Exception:
            pass
        m.SetProp("_Name", r["id"])
        m.SetProp("label", str(r["label"]))
        m.SetProp("split", r["split"])
        writer.write(m)
        n_ok += 1
    writer.close()
    return n_ok


# ------------------------------------------------------------------
#  Loading counterfeit slices + authentic pool
# ------------------------------------------------------------------
def load_authentic(path: str, limit: Optional[int]) -> List[Dict]:
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
            rows.append({"smiles": c, "label": 0, "category": "authentic",
                         "subcategory": "authentic", "parent_smiles": "",
                         "edited_atoms": "", **d})
    random.Random(0).shuffle(rows)
    if limit:
        rows = rows[:limit]
    logger.info(f"Authentic pool: {len(rows)} molecules")
    return rows


def _subcat_map() -> Dict[str, str]:
    try:
        from mined_transformations import SUBCATEGORY
        return dict(SUBCATEGORY)
    except Exception:
        return {}


def load_counterfeit_slices(min_subcat: int) -> Dict[str, List[Dict]]:
    """Return {slice_name: [counterfeit rows]} for every category, every
    well-populated subcategory, the evolved set, and the pooled set."""
    submap = _subcat_map()
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    by_subcat: Dict[str, List[Dict]] = defaultdict(list)
    evolved: List[Dict] = []
    pooled: List[Dict] = []

    for f in glob.glob("version_m/counterfeits_*.csv"):
        if f.endswith("counterfeits_all.csv"):
            continue
        for r in csv.DictReader(open(f)):
            smi = r["counterfeit_smiles"]
            d = descriptors(smi)
            if d is None:
                continue
            subcat = submap.get(r.get("rule_name", ""), "unknown")
            row = {"smiles": smi, "label": 1, "category": r["category"],
                   "subcategory": subcat, "parent_smiles": r["parent_smiles"],
                   "edited_atoms": r.get("edited_atoms", ""), **d}
            by_category[r["category"]].append(row)
            by_subcat[f"{r['category']}.{subcat}"].append(row)
            pooled.append(row)

    ef = Path("version_e/counterfeits_evolved.csv")
    if ef.exists():
        for r in csv.DictReader(open(ef)):
            smi = r["counterfeit_smiles"]
            d = descriptors(smi)
            if d is None:
                continue
            row = {"smiles": smi, "label": 1, "category": "evolved",
                   "subcategory": "evolved", "parent_smiles": r["parent_smiles"],
                   "edited_atoms": r.get("edited_atoms", ""), **d}
            evolved.append(row)
            pooled.append(row)

    slices: Dict[str, List[Dict]] = {}
    for cat, rows in by_category.items():
        slices[f"category_{cat}"] = rows
    for sub, rows in by_subcat.items():
        if len(rows) >= min_subcat * 20:  # enough counterfeits to matter
            slices[f"subcategory_{sub}"] = rows
    if evolved:
        slices["evolved"] = evolved
    slices["pooled"] = pooled
    return slices


# ------------------------------------------------------------------
#  Assemble one dataset
# ------------------------------------------------------------------
COLS = ["id", "smiles", "label", "category", "subcategory", "parent_smiles",
        "edited_atoms", "scaffold", "split", "mw", "logp", "qed",
        "heavy_atoms", "smiles_length"]


def assemble_one(name: str, counterfeits: List[Dict], authentic_pool: List[Dict],
                 outdir: Path, conformers: bool, conf_cap: Optional[int],
                 seed: int) -> Dict:
    # dedup counterfeits by SMILES
    seen, fake = set(), []
    for r in counterfeits:
        if r["smiles"] in seen:
            continue
        seen.add(r["smiles"])
        fake.append(r)

    auth_matched, fake_matched = property_match(authentic_pool, fake, seed=seed)
    k = min(len(auth_matched), len(fake_matched))
    if k < 20:
        logger.warning(f"  [{name}] too small after matching (k={k}); skipped")
        return {}
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
        for r in rows:
            w.writerow(r)

    n_conf = 0
    if conformers:
        n_conf = write_conformers(rows, ds_dir / "conformers.sdf", conf_cap)

    summary = {
        "name": name, "n_total": len(rows),
        "n_authentic": k, "n_counterfeit": k, "balance": 1.0,
        "shortcut_metrics": metrics,
        "split_counts": {s: sum(1 for r in rows if r["split"] == s)
                         for s in ("train", "val", "test")},
        "conformers_written": n_conf,
    }
    (ds_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"  [{name}] {len(rows)} mols (1:1), "
                f"LR acc={metrics.get('trivial_lr_accuracy')}, "
                f"conformers={n_conf}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authentic", default="mmp_mining/input.smi")
    ap.add_argument("--authentic-limit", type=int, default=None)
    ap.add_argument("--min-subcat", type=int, default=4,
                    help="Only build a subcategory dataset if its rule count>=this")
    ap.add_argument("--conformers", action="store_true",
                    help="Generate 3D conformers (ETKDGv3+MMFF) per molecule -> SDF")
    ap.add_argument("--conf-cap", type=int, default=None,
                    help="Cap conformers per dataset (speed)")
    ap.add_argument("--out", default="benchmark")
    ap.add_argument("--slices", nargs="*", default=None,
                    help="Only (re)build these slice names (e.g. evolved pooled)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    authentic_pool = load_authentic(args.authentic, args.authentic_limit)
    slices = load_counterfeit_slices(args.min_subcat)
    if args.slices:
        slices = {k: v for k, v in slices.items() if k in set(args.slices)}
    logger.info(f"Assembling {len(slices)} datasets: {sorted(slices)}")

    # keep prior index if only rebuilding a subset
    index_path = outdir / "index.json"
    index = json.loads(index_path.read_text()) if (args.slices and index_path.exists()) else {}
    for name, cfs in slices.items():
        s = assemble_one(name, cfs, authentic_pool, outdir,
                         args.conformers, args.conf_cap, args.seed)
        if s:
            index[name] = s
    (outdir / "index.json").write_text(json.dumps(index, indent=2))
    logger.info("=" * 60)
    logger.info(f"Built {len(index)} benchmark datasets -> {outdir}/")
    worst = max((v["shortcut_metrics"].get("trivial_lr_accuracy") or 0
                 for v in index.values()), default=0)
    logger.info(f"Worst trivial-LR accuracy across datasets: {worst} "
                f"(target ~0.5 = no shortcut)")


if __name__ == "__main__":
    main()
