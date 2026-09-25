"""
Boost Subcategories — High-Efficiency Targeted Counterfeit Generation
=====================================================================
Ensures that all 36 benchmark subcategories have a solid, reasonable number of
high-quality, chemically-valid, druglike counterfeit molecules.
"""

from __future__ import annotations

import argparse
import csv
import glob
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

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("boost_subcategories")

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

def find_edited_atoms(parent_mol: Chem.Mol, child_mol: Chem.Mol) -> List[int]:
    try:
        res = rdFMCS.FindMCS([parent_mol, child_mol], timeout=1,
                             matchValences=False, ringMatchesRingOnly=True,
                             completeRingsOnly=False)
        if not res.smartsString:
            return list(range(child_mol.GetNumAtoms()))
        patt = Chem.MolFromSmarts(res.smartsString)
        matched = set(child_mol.GetSubstructMatch(patt))
        return [a.GetIdx() for a in child_mol.GetAtoms() if a.GetIdx() not in matched]
    except Exception:
        return []

def load_parents(path: str) -> List[str]:
    logger.info(f"Loading parent library from {path}...")
    parents = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smi = line.split()[0]
            if smi not in seen:
                seen.add(smi)
                parents.append(smi)
    logger.info(f"Loaded {len(parents):,} authentic parents.")
    return parents

def main():
    parser = argparse.ArgumentParser(description="Boost underrepresented subcategories.")
    parser.add_argument("--target-min", type=int, default=800, help="Minimum counterfeits per subcategory")
    parser.add_argument("--parents", type=str, default="public_molecules_100k.smi")
    parser.add_argument("--out", type=str, default="counterfeits_boosted.csv")
    parser.add_argument("--max-checks-per-sub", type=int, default=8000, help="Max parents tested per subcategory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    import mined_transformations as mt

    logger.info("Scanning existing counterfeits...")
    existing_by_subcat = defaultdict(list)
    seen_smiles = set()

    # From 100k
    if os.path.exists("counterfeits_100k/counterfeits_100k.csv"):
        with open("counterfeits_100k/counterfeits_100k.csv") as f:
            for row in csv.DictReader(f):
                cat = row["category"]
                rule = row.get("rule_name", "")
                subcat = mt.SUBCATEGORY.get(rule, "other" if cat == "other" else "unknown")
                key = f"{cat}.{subcat}"
                smi = row.get("counterfeit_smiles", "")
                if smi and smi not in seen_smiles:
                    seen_smiles.add(smi)
                    existing_by_subcat[key].append(row)

    # From benchmark
    for p in glob.glob("benchmark/subcategory_*/dataset.csv"):
        sub = os.path.basename(os.path.dirname(p)).replace("subcategory_", "")
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("label") == "1":
                    smi = row.get("smiles", "")
                    if smi and smi not in seen_smiles:
                        seen_smiles.add(smi)
                        existing_by_subcat[sub].append({
                            "counterfeit_smiles": smi,
                            "parent_smiles": row.get("parent_smiles", ""),
                            "category": row.get("category", ""),
                            "rule_name": row.get("subcategory", ""),
                            "edited_atoms": row.get("edited_atoms", ""),
                            "mw": row.get("mw", 0),
                            "logp": row.get("logp", 0),
                            "qed": row.get("qed", 0),
                            "heavy_atoms": row.get("heavy_atoms", 0),
                        })

    bench_dirs = sorted([os.path.basename(d).replace("subcategory_", "") 
                         for d in glob.glob("benchmark/subcategory_*")])
    logger.info(f"Total benchmark subcategories tracked: {len(bench_dirs)}")

    deficits = {}
    for sub in bench_dirs:
        curr = len(existing_by_subcat[sub])
        if curr < args.target_min:
            deficits[sub] = args.target_min - curr

    logger.info(f"Subcategories requiring boost: {len(deficits)}")
    for sub, deficit in sorted(deficits.items()):
        logger.info(f"  • {sub:<36s}: current={len(existing_by_subcat[sub]):>4d}, deficit=+{deficit}")

    if not deficits:
        logger.info("All subcategories already exceed target minimum!")
        return

    # Load rules and group by subcategory key
    logger.info("Organizing rules by subcategory...")
    rules_by_subcat = defaultdict(list)
    for name, diff, cat, react, prod, cite in mt.MMP_TRANSFORMATIONS:
        subcat = mt.SUBCATEGORY.get(name, "other")
        key = f"{cat}.{subcat}"
        if key in deficits:
            try:
                rxn = AllChem.ReactionFromSmarts(f"{react}>>{prod}")
                patt = Chem.MolFromSmarts(react.replace("[*:1]", ""))
                rules_by_subcat[key].append({
                    "name": name,
                    "diff": diff,
                    "cat": cat,
                    "subcat": subcat,
                    "rxn": rxn,
                    "patt": patt,
                    "support": mt.SUPPORT.get(name, 1),
                })
            except Exception:
                continue

    parents = load_parents(args.parents)
    random.shuffle(parents)

    newly_generated = []
    t0 = time.time()

    for sub, deficit in sorted(deficits.items()):
        rules = rules_by_subcat.get(sub, [])
        if not rules:
            continue

        weights = np.array([r["support"] for r in rules], dtype=float)
        probs = weights / weights.sum()

        logger.info(f"⚡ Boosting [{sub}] (Target deficit: +{deficit})...")
        generated_for_sub = 0
        p_idx = 0
        checks = 0

        # Adjust minimum Tanimoto similarity for bulky fragments (e.g. Phosphate / Iodine)
        min_sim = 0.28 if ("add-P" in sub or "add-I" in sub or "F->I" in sub or "Cl->I" in sub) else 0.38

        while generated_for_sub < deficit and p_idx < len(parents) and checks < args.max_checks_per_sub:
            parent_smi = parents[p_idx]
            p_idx += 1
            checks += 1
            parent_mol = Chem.MolFromSmiles(parent_smi)
            if parent_mol is None:
                continue

            matched_rules = [r for r in rules if r["patt"] is None or parent_mol.HasSubstructMatch(r["patt"])]
            if not matched_rules:
                continue

            parent_fp = None

            for rule in matched_rules[:3]:
                if generated_for_sub >= deficit:
                    break
                try:
                    products = rule["rxn"].RunReactants((parent_mol,))
                except Exception:
                    continue
                if not products:
                    continue

                for ptuple in products[:4]:
                    if generated_for_sub >= deficit:
                        break
                    cand = ptuple[0]
                    try:
                        Chem.SanitizeMol(cand)
                    except Exception:
                        continue

                    cand_smi = Chem.MolToSmiles(cand)
                    if not cand_smi or cand_smi in seen_smiles or cand_smi == parent_smi:
                        continue

                    if parent_fp is None:
                        parent_fp = morgan(parent_mol)

                    cand_fp = morgan(cand)
                    sim = tanimoto(parent_fp, cand_fp)
                    if not (min_sim <= sim <= 0.98):
                        continue

                    desc = descriptors(cand)
                    if desc["qed"] < 0.25 or desc["sa_score"] > 4.5:
                        continue
                    if has_structural_alerts(cand):
                        continue

                    ed_atoms = find_edited_atoms(parent_mol, cand)
                    ed_str = " ".join(map(str, ed_atoms))

                    seen_smiles.add(cand_smi)
                    row = {
                        "id": f"boost_{sub}_{len(newly_generated)}",
                        "counterfeit_smiles": cand_smi,
                        "parent_smiles": parent_smi,
                        "category": rule["cat"],
                        "subcategory": rule["subcat"],
                        "rule_name": rule["name"],
                        "similarity_to_parent": round(sim, 3),
                        "edited_atoms": ed_str,
                        "mw": desc["mw"],
                        "logp": desc["logp"],
                        "qed": desc["qed"],
                        "sa_score": desc["sa_score"],
                        "heavy_atoms": desc["heavy_atoms"],
                        "label": 1
                    }
                    newly_generated.append(row)
                    generated_for_sub += 1

        logger.info(f"   ✓ [{sub}]: generated +{generated_for_sub} (checked {checks} parents).")

    logger.info(f"Boosting finished in {time.time() - t0:.1f}s. Total new counterfeits: {len(newly_generated):,}")

    if newly_generated:
        fields = list(newly_generated[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(newly_generated)
        logger.info(f"Saved boosted counterfeits to {args.out}")

if __name__ == "__main__":
    main()
