"""
Build "Version M" — the mined-rule counterfeit dataset, split per category
==========================================================================

Closes the Phase-A loop: the transformation rules learned from real drugs by
`mine_mmp_rules.py` are now USED to synthesize a counterfeit dataset. Following
the benchmark design (proposal C1), counterfeits are emitted in a SEPARATE
file per transformation category (bioisostere / scaffold-hop / halogen-walk /
homologation / other), so a detector can be scored per category instead of on
a single pooled F1.

Contrast with "Version R": Version R uses the 21 hand-written transformations
in `mmp_transformations.py`; Version M uses the ~131 data-mined, categorized
rules in `mined_transformations.py`. Training on one and testing on the other
measures how well a detector generalizes beyond the modifications it saw.

Each counterfeit row carries: parent, counterfeit SMILES, rule + category,
Tanimoto to parent, the atom indices the edit changed (via MCS, for
explanation ground-truth), and basic descriptors.

Input parents are read offline from a cached SMILES file (default
`mmp_mining/input.smi`, produced by mine_mmp_rules.py) or from --parents, so
no network or torch is required.

Usage
-----
    python build_version_m.py --rules mined_transformations.py \
        --parents mmp_mining/input.smi --per-category 2000 --out version_m
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_ALLOWED = frozenset({"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "H"})


# ------------------------------------------------------------------
#  Lightweight, class-symmetric quality filter
# ------------------------------------------------------------------
def is_plausible(mol: Chem.Mol) -> bool:
    if mol is None:
        return False
    try:
        n = mol.GetNumHeavyAtoms()
        if n < 10 or n > 70:
            return False
        if any(a.GetSymbol() not in _ALLOWED for a in mol.GetAtoms()):
            return False
        mw = Descriptors.MolWt(mol)
        if mw < 150 or mw > 800:
            return False
        if rdMolDescriptors.CalcNumRings(mol) == 0:
            return False
        return True
    except Exception:
        return False


def morgan(mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 2048)


def reaction_edited_atoms(child: Chem.Mol) -> List[int]:
    """Edited atoms via reaction provenance (fast, no MCS).

    After RunReactants, product atoms carried over from the reactant get a
    'react_atom_idx' property; atoms the reaction introduced do not. The edit =
    those new atoms plus their immediate neighbours (the boundary). Indices are
    mapped to canonical-SMILES atom order, so they index the stored `smiles`.
    Requires Chem.MolToSmiles(child) to have been called first (it sets
    '_smilesAtomOutputOrder')."""
    try:
        new_atoms = {a.GetIdx() for a in child.GetAtoms()
                     if not a.HasProp("react_atom_idx")}
        if not new_atoms or len(new_atoms) == child.GetNumAtoms():
            return []  # provenance unavailable/degenerate -> let caller fall back
        edited = set(new_atoms)
        for idx in list(new_atoms):
            for nb in child.GetAtomWithIdx(idx).GetNeighbors():
                edited.add(nb.GetIdx())
        order = child.GetProp("_smilesAtomOutputOrder")
        order = [int(x) for x in order.strip("[]").split(",") if x.strip() != ""]
        prod_to_canon = {p: c for c, p in enumerate(order)}
        return sorted(prod_to_canon[p] for p in edited if p in prod_to_canon)
    except Exception:
        return []


def edited_atoms(parent: Chem.Mol, child: Chem.Mol) -> List[int]:
    """Child atom indices outside the parent<->child MCS = what changed.
    Fallback for when reaction provenance is unavailable."""
    try:
        res = rdFMCS.FindMCS([parent, child], timeout=5,
                             ringMatchesRingOnly=True, completeRingsOnly=False)
        if not res.smartsString:
            return list(range(child.GetNumAtoms()))
        patt = Chem.MolFromSmarts(res.smartsString)
        matched = set(child.GetSubstructMatch(patt))
        return [a.GetIdx() for a in child.GetAtoms() if a.GetIdx() not in matched]
    except Exception:
        return []


def descriptors(mol) -> Dict:
    return {
        "mw": round(float(Descriptors.MolWt(mol)), 2),
        "logp": round(float(Descriptors.MolLogP(mol)), 2),
        "qed": round(float(Descriptors.qed(mol)), 3),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


# ------------------------------------------------------------------
#  Rule catalogue grouped by category
# ------------------------------------------------------------------
def load_rules_by_category(rules_py: str):
    spec = importlib.util.spec_from_file_location("_rules_mod", rules_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    support = dict(getattr(mod, "SUPPORT", {}))
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for name, diff, cat, react, prod, cite in mod.MMP_TRANSFORMATIONS:
        rxn = AllChem.ReactionFromSmarts(f"{react}>>{prod}")
        if rxn is None:
            continue
        by_cat[cat].append({
            "name": name, "difficulty": diff, "reaction": rxn,
            "weight": float(support.get(name, 1)), "citation": cite,
        })
    for cat, rules in by_cat.items():
        logger.info(f"  category '{cat}': {len(rules)} rules")
    return by_cat


def load_parents(parents_file: Optional[str], from_pt: Optional[str],
                 max_parents: Optional[int]) -> List[str]:
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
    # canonicalize + dedup
    seen, out = set(), []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if c not in seen:
            seen.add(c)
            out.append(c)
    random.shuffle(out)
    if max_parents:
        out = out[:max_parents]
    logger.info(f"Loaded {len(out)} unique authentic parents")
    return out


# ------------------------------------------------------------------
#  Generation, per category
# ------------------------------------------------------------------
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


def generate_category(category: str, rules: List[Dict], parents: List[str],
                      target: int, sim_lo: float, sim_hi: float,
                      global_seen: set) -> List[Dict]:
    weights = np.array([r["weight"] for r in rules], dtype=float)
    probs = weights / weights.sum() if weights.sum() > 0 else None
    rows: List[Dict] = []
    order = list(range(len(parents)))
    random.shuffle(order)

    for pi in order:
        if len(rows) >= target:
            break
        parent_smi = parents[pi]
        parent_mol = Chem.MolFromSmiles(parent_smi)
        if parent_mol is None:
            continue
        parent_fp = morgan(parent_mol)
        # try a few rules of this category per parent
        for _ in range(6):
            if len(rows) >= target:
                break
            rule = rules[int(np.random.choice(len(rules), p=probs))]
            child = apply_rule(parent_mol, rule)
            if child is None:
                continue
            try:
                canon = Chem.MolToSmiles(child)  # also sets _smilesAtomOutputOrder
            except Exception:
                continue
            if canon == parent_smi or canon in global_seen:
                continue
            if not is_plausible(child):
                continue
            sim = DataStructs.TanimotoSimilarity(parent_fp, morgan(child))
            if sim < sim_lo or sim > sim_hi:
                continue
            global_seen.add(canon)
            # fast provenance-based edit tracking, MCS only as fallback
            edits = reaction_edited_atoms(child)
            if not edits:
                edits = edited_atoms(parent_mol, Chem.MolFromSmiles(canon))
            rows.append({
                "counterfeit_smiles": canon,
                "parent_smiles": parent_smi,
                "rule_name": rule["name"],
                "category": category,
                "difficulty": rule["difficulty"],
                "citation": rule["citation"],
                "similarity_to_parent": round(float(sim), 3),
                "edited_atoms": edits,
                **descriptors(child),
            })
    logger.info(f"[{category}] generated {len(rows)}/{target}")
    return rows


def write_category_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    cols = ["counterfeit_smiles", "parent_smiles", "rule_name", "category",
            "difficulty", "similarity_to_parent", "edited_atoms",
            "mw", "logp", "qed", "heavy_atoms", "citation"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["edited_atoms"] = " ".join(map(str, r["edited_atoms"]))
            w.writerow({k: row.get(k) for k in cols})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rules", default="mined_transformations.py")
    p.add_argument("--parents", default=None,
                   help="SMILES file of authentic parents (default mmp_mining/input.smi)")
    p.add_argument("--from-pt", default=None,
                   help="Instead extract authentic parents from a .pt dataset")
    p.add_argument("--max-parents", type=int, default=None)
    p.add_argument("--per-category", type=int, default=2000,
                   help="Target counterfeits per category")
    p.add_argument("--sim-lo", type=float, default=0.4)
    p.add_argument("--sim-hi", type=float, default=0.98)
    p.add_argument("--out", default="version_m", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    logger.info("Loading mined, categorized rules ...")
    by_cat = load_rules_by_category(args.rules)
    parents = load_parents(args.parents, args.from_pt, args.max_parents)

    global_seen: set = set()
    summary = {"categories": {}, "per_category_target": args.per_category,
               "n_parents": len(parents),
               "similarity_band": [args.sim_lo, args.sim_hi]}
    all_rows: List[Dict] = []

    for category, rules in by_cat.items():
        rows = generate_category(category, rules, parents, args.per_category,
                                 args.sim_lo, args.sim_hi, global_seen)
        write_category_csv(rows, outdir / f"counterfeits_{category}.csv")
        all_rows.extend(rows)
        if rows:
            sims = [r["similarity_to_parent"] for r in rows]
            edits = [len(r["edited_atoms"]) for r in rows]
            summary["categories"][category] = {
                "n_counterfeits": len(rows),
                "n_rules_available": len(rules),
                "mean_similarity": round(float(np.mean(sims)), 3),
                "mean_edited_atoms": round(float(np.mean(edits)), 2),
            }

    # combined file
    write_category_csv(all_rows, outdir / "counterfeits_all.csv")
    summary["total_counterfeits"] = len(all_rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("=" * 60)
    logger.info(f"Version M built: {len(all_rows)} counterfeits across "
                f"{len(summary['categories'])} categories -> {outdir}/")
    for cat, s in summary["categories"].items():
        logger.info(f"  {cat:14s}: {s['n_counterfeits']:>5}  "
                    f"sim={s['mean_similarity']}  edits={s['mean_edited_atoms']}")


if __name__ == "__main__":
    main()
