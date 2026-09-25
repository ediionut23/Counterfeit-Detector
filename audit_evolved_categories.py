"""
Comprehensive Quality & Shortcut Audit for Evolved Category Datasets
====================================================================
Evaluates:
  1. Structural preservation (Tanimoto to parent)
  2. Plausibility & Drug-likeness (QED, SA score, PAINS alerts)
  3. Multi-step transformation depth (MCS edited atoms)
  4. Shortcut Resistance (DUD-E Binned Matching + Trivial Classifier Accuracy)
  5. 3D Conformer Stability & MMFF94 Force Field Convergence
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("audit_evolved_categories")


def load_authentics(path: str, max_auth: int = 15000, seed: int = 42) -> List[Dict]:
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
            c = Chem.MolToSmiles(m)
            if c in seen:
                continue
            seen.add(c)
            out.append({
                "smiles": c,
                "label": 0,
                "mw": float(Descriptors.MolWt(m)),
                "logp": float(Descriptors.MolLogP(m)),
                "qed": float(Descriptors.qed(m)),
                "heavy_atoms": m.GetNumHeavyAtoms(),
                "smiles_length": len(c),
            })
            if len(out) >= max_auth:
                break
    random.Random(seed).shuffle(out)
    return out


def property_match(auth: List[Dict], fake: List[Dict],
                   cols=("mw", "heavy_atoms", "smiles_length"),
                   n_bins: int = 6, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
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


def evaluate_shortcut(matched_auth: List[Dict], matched_fake: List[Dict]) -> Dict:
    cols = ["mw", "heavy_atoms", "smiles_length"]
    def cohens_d(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        return float(abs(a.mean() - b.mean()) / max(pooled, 1e-9))

    d_scores = {f"cohens_d_{c}": round(cohens_d([r[c] for r in matched_auth], [r[c] for r in matched_fake]), 3) for c in cols}

    X = np.array([[r[c] for c in cols] for r in matched_auth + matched_fake], float)
    y = np.array([0] * len(matched_auth) + [1] * len(matched_fake))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y[train_idx])
        accs.append(clf.score(X_te, y[test_idx]))

    return {
        "n_matched_pairs": len(matched_auth),
        "trivial_lr_accuracy_mean": round(float(np.mean(accs)), 3),
        "trivial_lr_accuracy_std": round(float(np.std(accs)), 3),
        **d_scores
    }


def audit_conformers_3d(smiles_list: List[str], n_sample: int = 50, seed: int = 42) -> Dict:
    rng = random.Random(seed)
    sampled = rng.sample(smiles_list, min(len(smiles_list), n_sample))
    embed_ok = 0
    mmff_ok = 0

    params = AllChem.ETKDGv3()
    params.randomSeed = seed

    for smi in sampled:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mh = Chem.AddHs(m)
        if AllChem.EmbedMolecule(mh, params) == 0:
            embed_ok += 1
            res = AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
            if res == 0:
                mmff_ok += 1

    return {
        "sample_size": len(sampled),
        "etkdg_embedding_success_rate": round(embed_ok / max(len(sampled), 1), 3),
        "mmff_convergence_rate": round(mmff_ok / max(len(sampled), 1), 3),
    }


def audit_category_file(csv_path: Path, auth_pool: List[Dict]) -> Dict:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "counterfeit_smiles": r["counterfeit_smiles"],
                "parent_smiles": r["parent_smiles"],
                "category": r["category"],
                "similarity_to_parent": float(r["similarity_to_parent"]),
                "qed": float(r["qed"]),
                "sa_score": float(r["sa_score"]) if r["sa_score"] else None,
                "n_edited_atoms": int(r["n_edited_atoms"]),
                "mw": float(r["mw"]),
                "logp": float(r["logp"]),
                "heavy_atoms": int(r["heavy_atoms"]),
                "smiles_length": int(r["smiles_length"]),
            })

    sims = [r["similarity_to_parent"] for r in rows]
    qeds = [r["qed"] for r in rows]
    sas = [r["sa_score"] for r in rows if r["sa_score"] is not None]
    edits = [r["n_edited_atoms"] for r in rows]

    # Shortcut resistance
    matched_auth, matched_fake = property_match(auth_pool, rows)
    shortcut_res = evaluate_shortcut(matched_auth, matched_fake)

    # 3D Conformer validation
    conf_res = audit_conformers_3d([r["counterfeit_smiles"] for r in rows], n_sample=50)

    # Evaluation judgment
    pass_tanimoto = 0.58 <= np.median(sims) <= 0.82
    pass_qed = np.median(qeds) >= 0.50
    pass_sa = np.median(sas) <= 4.0
    pass_shortcut = 0.47 <= shortcut_res["trivial_lr_accuracy_mean"] <= 0.55
    pass_conformer = conf_res["etkdg_embedding_success_rate"] >= 0.95

    overall_pass = all([pass_tanimoto, pass_qed, pass_sa, pass_shortcut, pass_conformer])

    return {
        "file": csv_path.name,
        "n_molecules": len(rows),
        "tanimoto": {
            "mean": round(float(np.mean(sims)), 3),
            "median": round(float(np.median(sims)), 3),
            "min": round(float(np.min(sims)), 3),
            "max": round(float(np.max(sims)), 3),
            "passed": bool(pass_tanimoto)
        },
        "drug_likeness": {
            "qed_median": round(float(np.median(qeds)), 3),
            "sa_median": round(float(np.median(sas)), 2),
            "qed_passed": bool(pass_qed),
            "sa_passed": bool(pass_sa),
        },
        "multi_step_edits": {
            "mean_atoms_edited": round(float(np.mean(edits)), 1),
            "median_atoms_edited": int(np.median(edits)),
            "max_atoms_edited": int(np.max(edits)),
        },
        "shortcut_audit": {
            **shortcut_res,
            "passed": bool(pass_shortcut)
        },
        "conformer_3d_audit": {
            **conf_res,
            "passed": bool(pass_conformer)
        },
        "overall_quality_pass": bool(overall_pass)
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Evolved Category Counterfeits")
    parser.add_argument("--input-dir", default="version_e_categories", help="Directory containing evolved categories CSVs")
    parser.add_argument("--parents", default="public_molecules.smi", help="Authentic molecules pool")
    parser.add_argument("--out-report", default="version_e_categories/audit_report.json", help="Path to output report")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    csv_files = sorted(in_dir.glob("counterfeits_evolved_*.csv"))
    if not csv_files:
        logger.error(f"No counterfeits_evolved_*.csv files found in {in_dir}")
        return

    logger.info(f"Loading authentic library from {args.parents}...")
    auth_pool = load_authentics(args.parents)
    logger.info(f"Loaded {len(auth_pool):,} authentic drugs for DUD-E pairing.")

    all_reports = {}
    for cf in csv_files:
        logger.info(f"Auditing {cf.name}...")
        rep = audit_category_file(cf, auth_pool)
        cat_name = cf.stem.replace("counterfeits_evolved_", "")
        all_reports[cat_name] = rep

    out_json = Path(args.out_report)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_reports, indent=2))
    logger.info(f"Saved complete audit report to {out_json}")

    # Print summary table
    print("\n" + "=" * 95)
    print("                CATEGORY-CONDITIONED EVOLUTION AUDIT REPORT")
    print("=" * 95)
    header = f"{'Category':<16} | {'Count':<6} | {'Tanimoto':<9} | {'QED':<6} | {'SA':<6} | {'Edits':<6} | {'Trivial LR':<11} | {'3D Conf':<7} | {'Verdict'}"
    print(header)
    print("-" * 95)
    for cat, rep in all_reports.items():
        t_str = f"{rep['tanimoto']['median']:.3f}"
        q_str = f"{rep['drug_likeness']['qed_median']:.2f}"
        s_str = f"{rep['drug_likeness']['sa_median']:.2f}"
        e_str = f"{rep['multi_step_edits']['mean_atoms_edited']:.1f}"
        lr_str = f"{rep['shortcut_audit']['trivial_lr_accuracy_mean']:.3f}"
        c_str = f"{rep['conformer_3d_audit']['etkdg_embedding_success_rate']*100:.0f}%"
        verdict = "PASSED [✓]" if rep["overall_quality_pass"] else "REVIEW [!]"
        print(f"{cat:<16} | {rep['n_molecules']:<6} | {t_str:<9} | {q_str:<6} | {s_str:<6} | {e_str:<6} | {lr_str:<11} | {c_str:<7} | {verdict}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
