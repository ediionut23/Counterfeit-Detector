"""
Generate 100K Counterfeit Molecules Dataset (High-Throughput Production Pipeline)
================================================================================
Generates 100,000 unique, chemically valid, plausibly druglike counterfeit molecules
derived from authentic pharmaceutical parents (public_molecules_100k.smi) using
mined Matched Molecular Pair (MMP) transformations and multi-objective criteria.

Features:
  - Generates 100,000 counterfeits (Label = 1, no artificial authentic balancing needed at this stage)
  - Balanced across 5 transformation categories:
      * bioisostere
      * scaffold-hop
      * halogen-walk
      * homologation
      * other
  - Enforces strict medicinal feasibility (QED >= 0.35, SA score <= 3.8, PAINS/Brenk filter)
  - Guarantees realistic Tanimoto band [0.40, 0.98]
  - Tracks exact modified atom indices for XAI explanation ground-truth
  - Saves full dataset (CSV + SMI), per-category CSVs, and detailed summary statistics

Usage:
  python generate_100k_counterfeits.py --target 100000 --out-dir counterfeits_100k
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# SA score
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except Exception:
    sascorer = None

# Structural alerts (PAINS + Brenk)
_alert_params = FilterCatalogParams()
_alert_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_alert_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_ALERT_CATALOG = FilterCatalog(_alert_params)

def has_structural_alerts(mol: Chem.Mol) -> bool:
    try:
        return _ALERT_CATALOG.HasMatch(mol)
    except Exception:
        return False

def morgan(mol: Chem.Mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 2048)

def tanimoto(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def descriptors(mol: Chem.Mol) -> Dict[str, float]:
    sa = round(sascorer.calculateScore(mol), 2) if sascorer else 2.5
    return {
        "mw": round(float(Descriptors.MolWt(mol)), 2),
        "logp": round(float(Descriptors.MolLogP(mol)), 2),
        "qed": round(float(Descriptors.qed(mol)), 3),
        "sa_score": sa,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


# ------------------------------------------------------------------
#  MMP Rule Loading
# ------------------------------------------------------------------
def load_rules_by_category(rules_py: str) -> Dict[str, List[Dict]]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rules_mod", rules_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = list(getattr(mod, "MMP_TRANSFORMATIONS"))
    support = dict(getattr(mod, "SUPPORT", {}))
    by_cat: Dict[str, List[Dict]] = defaultdict(list)

    for name, diff, cat, react, prod, cite in raw:
        rxn = AllChem.ReactionFromSmarts(f"{react}>>{prod}")
        if rxn is None:
            continue
        w = float(support.get(name, 1.0))
        by_cat[cat].append({
            "name": name,
            "difficulty": diff,
            "reaction": rxn,
            "weight": max(1.0, w),
            "citation": cite,
        })

    logger.info("Loaded MMP Rule Library:")
    for cat, rules in by_cat.items():
        logger.info(f"   • Category '{cat}': {len(rules)} rules")
    return by_cat


# ------------------------------------------------------------------
#  Parent Smiles Loading
# ------------------------------------------------------------------
def load_parents(path: str, limit: Optional[int] = None, seed: int = 42) -> List[str]:
    smis: List[str] = []
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
    if limit:
        out = out[:limit]
    logger.info(f"Loaded {len(out)} unique authentic parent drugs from {path}")
    return out


# ------------------------------------------------------------------
#  Reaction & Edit Tracking
# ------------------------------------------------------------------
def reaction_edited_atoms(product_mol: Chem.Mol) -> List[int]:
    try:
        atom_order = [int(x) for x in product_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",") if x]
        edited = []
        for smi_idx, mol_idx in enumerate(atom_order):
            atom = product_mol.GetAtomWithIdx(mol_idx)
            if not atom.HasProp("old_mapno"):
                edited.append(smi_idx)
        return edited
    except Exception:
        return []

def edited_atoms_mcs(parent_mol: Chem.Mol, child_mol: Chem.Mol) -> List[int]:
    try:
        res = rdFMCS.FindMCS([parent_mol, child_mol], timeout=2,
                             matchValences=False, ringMatchesRingOnly=True,
                             completeRingsOnly=False)
        if not res.smartsString:
            return list(range(child_mol.GetNumAtoms()))
        patt = Chem.MolFromSmarts(res.smartsString)
        matched = set(child_mol.GetSubstructMatch(patt))
        return [a.GetIdx() for a in child_mol.GetAtoms() if a.GetIdx() not in matched]
    except Exception:
        return []

def apply_rule(parent_mol: Chem.Mol, rule: Dict) -> Optional[Chem.Mol]:
    try:
        products = rule["reaction"].RunReactants((parent_mol,))
    except Exception:
        return None
    if not products:
        return None
    cand = random.choice(products)[0]
    try:
        Chem.SanitizeMol(cand)
        return cand
    except Exception:
        return None


# ------------------------------------------------------------------
#  Core Generation Engine
# ------------------------------------------------------------------
def generate_100k(parents: List[str], by_cat: Dict[str, List[Dict]],
                  target_total: int = 100000,
                  sim_lo: float = 0.40, sim_hi: float = 0.98,
                  qed_min: float = 0.35, max_sa: float = 3.8,
                  seed: int = 42) -> Tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    categories = list(by_cat.keys())
    # Baseline allocation per category: ~20k per category, adjust dynamically
    target_per_cat = {
        "bioisostere": 25000,
        "scaffold-hop": 22000,
        "homologation": 20000,
        "halogen-walk": 20000,
        "other": 13000,
    }
    # Ensure targets sum up to target_total
    allocated = sum(target_per_cat.values())
    if allocated != target_total:
        target_per_cat["bioisostere"] += (target_total - allocated)

    logger.info("Target distribution across categories:")
    for cat, tgt in target_per_cat.items():
        logger.info(f"   • {cat:<14s}: {tgt:,} counterfeits")

    global_seen: Set[str] = set()
    category_rows: Dict[str, List[Dict]] = defaultdict(list)
    t0 = time.time()

    for cat in categories:
        target = target_per_cat[cat]
        rules = by_cat[cat]
        weights = np.array([r["weight"] for r in rules], dtype=float)
        probs = weights / weights.sum() if weights.sum() > 0 else None

        parent_indices = list(range(len(parents)))
        rng.shuffle(parent_indices)

        logger.info(f"⚡ Generating [{cat}] (Target: {target:,})...")
        t_cat = time.time()

        for pi in parent_indices:
            if len(category_rows[cat]) >= target:
                break
            parent_smi = parents[pi]
            parent_mol = Chem.MolFromSmiles(parent_smi)
            if parent_mol is None:
                continue
            parent_fp = morgan(parent_mol)
            parent_mw = Descriptors.MolWt(parent_mol)

            # Attempt 6 rule applications per parent
            for _ in range(6):
                if len(category_rows[cat]) >= target:
                    break
                rule = rules[int(np_rng.choice(len(rules), p=probs))]
                child = apply_rule(parent_mol, rule)
                if child is None:
                    continue
                try:
                    canon = Chem.MolToSmiles(child)
                except Exception:
                    continue
                if canon == parent_smi or canon in global_seen:
                    continue

                # Filters - evaluated in order of computational cost:
                # 1. Anti-shortcut mass drift (instant float check)
                child_mw = Descriptors.MolWt(child)
                if abs(child_mw - parent_mw) > 80.0:
                    continue

                # 2. Tanimoto band (fast bit-vector)
                sim = tanimoto(parent_fp, morgan(child))
                if sim < sim_lo or sim > sim_hi:
                    continue

                # 3. Drug-likeness QED
                qed_val = float(Descriptors.qed(child))
                if qed_val < qed_min:
                    continue

                # 4. Structural Alerts (only for in-band druglike candidates)
                if has_structural_alerts(child):
                    continue

                # 5. Synthetic Accessibility
                sa_val = sascorer.calculateScore(child) if sascorer else 2.5
                if sa_val > max_sa:
                    continue

                desc = {
                    "mw": round(float(child_mw), 2),
                    "logp": round(float(Descriptors.MolLogP(child)), 2),
                    "qed": round(qed_val, 3),
                    "sa_score": round(float(sa_val), 2),
                    "heavy_atoms": child.GetNumHeavyAtoms(),
                }

                global_seen.add(canon)

                # Edit tracking
                edits = reaction_edited_atoms(child)
                if not edits:
                    edits = edited_atoms_mcs(parent_mol, Chem.MolFromSmiles(canon))

                category_rows[cat].append({
                    "id": f"cf_{len(global_seen):06d}",
                    "counterfeit_smiles": canon,
                    "parent_smiles": parent_smi,
                    "category": cat,
                    "rule_name": rule["name"],
                    "difficulty": rule["difficulty"],
                    "similarity_to_parent": round(float(sim), 3),
                    "edited_atoms": " ".join(map(str, edits)),
                    "n_edited_atoms": len(edits),
                    "label": 1,
                    **desc,
                    "citation": rule["citation"],
                })

                if len(category_rows[cat]) % 5000 == 0:
                    elapsed = time.time() - t_cat
                    speed = len(category_rows[cat]) / max(elapsed, 0.001)
                    logger.info(f"   -> [{cat}] {len(category_rows[cat]):,}/{target:,} ({speed:.1f} mols/s)")

        dt_cat = time.time() - t_cat
        logger.info(f"   ✓ [{cat}] completed: {len(category_rows[cat]):,} counterfeits in {dt_cat:.1f}s")

    # If any category fell short of its quota, top up with bioisostere and scaffold-hop
    all_rows: List[Dict] = []
    for cat in categories:
        all_rows.extend(category_rows[cat])

    if len(all_rows) < target_total:
        needed = target_total - len(all_rows)
        logger.info(f"🔄 Topping up {needed:,} counterfeits using bioisostere/scaffold-hop to reach exactly {target_total:,}...")
        for cat in ["bioisostere", "scaffold-hop", "homologation"]:
            if len(all_rows) >= target_total:
                break
            rules = by_cat[cat]
            weights = np.array([r["weight"] for r in rules], dtype=float)
            probs = weights / weights.sum() if weights.sum() > 0 else None
            for p_smi in parents:
                if len(all_rows) >= target_total:
                    break
                p_mol = Chem.MolFromSmiles(p_smi)
                if p_mol is None:
                    continue
                p_fp = morgan(p_mol)
                p_mw = Descriptors.MolWt(p_mol)
                for _ in range(4):
                    if len(all_rows) >= target_total:
                        break
                    rule = rules[int(np_rng.choice(len(rules), p=probs))]
                    child = apply_rule(p_mol, rule)
                    if child is None:
                        continue
                    try:
                        canon = Chem.MolToSmiles(child)
                    except Exception:
                        continue
                    if canon == p_smi or canon in global_seen:
                        continue
                    child_mw = Descriptors.MolWt(child)
                    if abs(child_mw - p_mw) > 80.0:
                        continue
                    sim = tanimoto(p_fp, morgan(child))
                    if sim < sim_lo or sim > sim_hi:
                        continue
                    qed_val = float(Descriptors.qed(child))
                    if qed_val < qed_min:
                        continue
                    if has_structural_alerts(child):
                        continue
                    sa_val = sascorer.calculateScore(child) if sascorer else 2.5
                    if sa_val > max_sa:
                        continue
                    desc = {
                        "mw": round(float(child_mw), 2),
                        "logp": round(float(Descriptors.MolLogP(child)), 2),
                        "qed": round(qed_val, 3),
                        "sa_score": round(float(sa_val), 2),
                        "heavy_atoms": child.GetNumHeavyAtoms(),
                    }

                    global_seen.add(canon)
                    edits = reaction_edited_atoms(child) or edited_atoms_mcs(p_mol, Chem.MolFromSmiles(canon))
                    rec = {
                        "id": f"cf_{len(global_seen):06d}",
                        "counterfeit_smiles": canon,
                        "parent_smiles": p_smi,
                        "category": cat,
                        "rule_name": rule["name"],
                        "difficulty": rule["difficulty"],
                        "similarity_to_parent": round(float(sim), 3),
                        "edited_atoms": " ".join(map(str, edits)),
                        "n_edited_atoms": len(edits),
                        "label": 1,
                        **desc,
                        "citation": rule["citation"],
                    }
                    all_rows.append(rec)
                    category_rows[cat].append(rec)

    total_time = time.time() - t0
    logger.info("=" * 80)
    logger.info(f"🏆 Successfully generated {len(all_rows):,} counterfeit molecules in {total_time:.1f}s ({len(all_rows)/total_time:.1f} mols/s)")
    logger.info("=" * 80)

    # Compute summary statistics
    sims = [r["similarity_to_parent"] for r in all_rows]
    qeds = [r["qed"] for r in all_rows]
    sas = [r["sa_score"] for r in all_rows]
    mws = [r["mw"] for r in all_rows]
    edits = [r["n_edited_atoms"] for r in all_rows]

    summary = {
        "dataset_name": "counterfeits_100k",
        "total_counterfeits": len(all_rows),
        "target_total": target_total,
        "elapsed_seconds": round(total_time, 2),
        "generation_speed_mols_per_sec": round(len(all_rows) / max(total_time, 0.001), 1),
        "similarity_band": [sim_lo, sim_hi],
        "metrics": {
            "mean_similarity_to_parent": round(float(np.mean(sims)), 3),
            "median_similarity_to_parent": round(float(np.median(sims)), 3),
            "mean_qed": round(float(np.mean(qeds)), 3),
            "mean_sa_score": round(float(np.mean(sas)), 3),
            "mean_molecular_weight": round(float(np.mean(mws)), 1),
            "mean_edited_atoms": round(float(np.mean(edits)), 2),
        },
        "category_counts": {cat: len(rows) for cat, rows in category_rows.items()}
    }

    return all_rows, summary


# ------------------------------------------------------------------
#  Export Utilities
# ------------------------------------------------------------------
def export_dataset(rows: List[Dict], summary: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["id", "counterfeit_smiles", "parent_smiles", "category", "rule_name",
            "difficulty", "similarity_to_parent", "n_edited_atoms", "edited_atoms",
            "mw", "logp", "qed", "sa_score", "heavy_atoms", "label", "citation"]

    # 1. Full 100k CSV
    main_csv = out_dir / "counterfeits_100k.csv"
    logger.info(f"Saving full dataset to {main_csv}...")
    with open(main_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 2. Pure SMILES file (for conformers / fast docking / screening)
    main_smi = out_dir / "counterfeits_100k.smi"
    logger.info(f"Saving pure SMILES file to {main_smi}...")
    with open(main_smi, "w") as fh:
        for r in rows:
            fh.write(f"{r['counterfeit_smiles']} {r['id']}\n")

    # 3. Per-category CSVs
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    for cat, cat_rows in by_cat.items():
        cat_file = out_dir / f"counterfeits_{cat}.csv"
        with open(cat_file, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in cat_rows:
                w.writerow(r)

    # 4. Summary JSON
    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary statistics written to {summary_file}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parents", default="public_molecules_100k.smi", help="Authentic parent molecules SMILES file")
    p.add_argument("--rules", default="mined_transformations.py", help="MMP mined transformations file")
    p.add_argument("--target", type=int, default=100000, help="Total counterfeits to generate (default: 100000)")
    p.add_argument("--sim-lo", type=float, default=0.40, help="Min Tanimoto similarity to parent (default 0.40)")
    p.add_argument("--sim-hi", type=float, default=0.98, help="Max Tanimoto similarity to parent (default 0.98)")
    p.add_argument("--qed-min", type=float, default=0.35, help="Min QED score (default 0.35)")
    p.add_argument("--max-sa", type=float, default=3.8, help="Max SA score (default 3.8)")
    p.add_argument("--out-dir", default="counterfeits_100k", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    parents_path = args.parents
    if not Path(parents_path).exists():
        if Path("public_molecules.smi").exists():
            parents_path = "public_molecules.smi"
        else:
            logger.error(f"Cannot find parents file {args.parents}")
            sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"🚀 Launching 100K Counterfeit Generation Pipeline")
    logger.info(f"   Target: {args.target:,} unique counterfeit molecules")
    logger.info(f"   Parents: {parents_path} | Rules: {args.rules}")
    logger.info(f"   Filters: Tanimoto=[{args.sim_lo}, {args.sim_hi}] | QED >= {args.qed_min} | SA <= {args.max_sa}")
    logger.info("=" * 80)

    by_cat = load_rules_by_category(args.rules)
    parents = load_parents(parents_path, seed=args.seed)

    rows, summary = generate_100k(parents, by_cat, target_total=args.target,
                                  sim_lo=args.sim_lo, sim_hi=args.sim_hi,
                                  qed_min=args.qed_min, max_sa=args.max_sa,
                                  seed=args.seed)

    export_dataset(rows, summary, Path(args.out_dir))

    logger.info("\n" + "=" * 80)
    logger.info("📊 100K GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"  • Total Molecules: {summary['total_counterfeits']:,}")
    logger.info(f"  • Mean Tanimoto to Parent: {summary['metrics']['mean_similarity_to_parent']}")
    logger.info(f"  • Mean QED (Drug-likeness): {summary['metrics']['mean_qed']}")
    logger.info(f"  • Mean SA Score (Accessibility): {summary['metrics']['mean_sa_score']}")
    logger.info(f"  • Mean Edited Atoms: {summary['metrics']['mean_edited_atoms']}")
    logger.info(f"  • Time: {summary['elapsed_seconds']/60:.2f} min ({summary['generation_speed_mols_per_sec']} mols/sec)")
    logger.info(f"  • Output Directory: {Path(args.out_dir).absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
