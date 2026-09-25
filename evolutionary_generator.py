"""
Evolutionary counterfeit generator (Phase B of the benchmark plan)
==================================================================

Where `mine_mmp_rules.py` (Phase A) LEARNS a catalogue of modifications from
data, this module DISCOVERS and COMBINES modifications under multi-objective
selection pressure — it does not stay inside any fixed rule list.

It answers the two open questions:

  * "Can the modifications be combined?"  -> yes: a genetic algorithm applies
    mined MMP rules as mutation operators and recombines whole substructures
    via graph crossover, so a single counterfeit can carry several stacked
    edits that no single hand-written rule produces.

  * "How do we keep the most promising changes?" (Laura's question) -> with a
    MULTI-OBJECTIVE fitness handled by NSGA-II. A counterfeit is 'promising'
    only if it is simultaneously (a) similar-enough-but-not-identical to the
    genuine drug, (b) realistic / synthesizable, (c) [optionally] hard for the
    current detector, and (d) novel. These objectives conflict, so instead of
    an arbitrary weighted sum we keep the whole Pareto front (non-dominated
    sorting + crowding distance). The front IS the set of "most promising"
    counterfeits, spanning the trade-offs.

Genetic representation & operators (Graph-GA, after Jensen 2019)
    - individual        : a canonical SMILES string
    - crossover         : cut both parents at a random acyclic single bond and
                          zip a fragment of one onto a fragment of the other
    - mutation          : apply one mined MMP rule (weighted by data support)

Objectives (all minimized; maximization terms are negated)
    f1  band penalty    : 0 if Tanimoto(parent, child) in [sim_lo, sim_hi],
                          else distance outside the band
    f2  1 - QED         : drug-likeness (maximize QED)
    f3  SA / 10         : synthetic accessibility (minimize; easy to make)
    f4  P(counterfeit)  : OPTIONAL, model-in-the-loop. Low value = the detector
                          is fooled -> an adversarially hard counterfeit.

Each surviving counterfeit is annotated with the atom indices it changed
relative to the parent (via maximum common substructure), giving the
explanation ground-truth the benchmark needs for explanation-accuracy /
Fidelity / Sparsity scoring.

Usage
-----
    # 3-objective run on a seed drug (SMILES):
    python evolutionary_generator.py --parent "CC(=O)Oc1ccccc1C(=O)O" \
        --rules mined_transformations.py --generations 30 --pop 60

    # add the detector as a 4th, adversarial objective:
    python evolutionary_generator.py --parent "..." --adversarial
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Ertl synthetic-accessibility score (ships with RDKit Contrib) ---
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
except Exception:  # pragma: no cover
    sascorer = None
    logger.warning("SA_Score not importable; f3 will fall back to 0.")


# ==================================================================
#  Mutation-rule catalogue (from Phase A)
# ==================================================================
class RuleLibrary:
    """Loads mined MMP rules and exposes them as weighted mutation operators."""

    def __init__(self, rules_py: str, category: Optional[str] = None):
        mod = self._load_module(rules_py)
        raw = list(getattr(mod, "MMP_TRANSFORMATIONS"))
        support = dict(getattr(mod, "SUPPORT", {}))
        self.category = category
        self.names: List[str] = []
        self.reactions: List[AllChem.ChemicalReaction] = []
        self.weights: List[float] = []
        for name, _diff, _cat, react, prod, _cite in raw:
            if category and _cat != category:
                continue
            rxn = AllChem.ReactionFromSmarts(f"{react}>>{prod}")
            if rxn is None:
                continue
            self.names.append(name)
            self.reactions.append(rxn)
            self.weights.append(float(support.get(name, 1)))
        w = np.asarray(self.weights, dtype=float)
        self.probs = (w / w.sum()) if w.sum() > 0 else None
        cat_str = f" for category '{category}'" if category else ""
        logger.info(f"Loaded {len(self.names)} mutation rules{cat_str} from {rules_py}")

    @staticmethod
    def _load_module(path: str):
        spec = importlib.util.spec_from_file_location("_rules_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    def mutate(self, smiles: str, max_tries: int = 12) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        idxs = np.arange(len(self.reactions))
        for _ in range(max_tries):
            i = int(np.random.choice(idxs, p=self.probs))
            try:
                products = self.reactions[i].RunReactants((mol,))
            except Exception:
                continue
            if not products:
                continue
            cand = random.choice(products)[0]
            try:
                Chem.SanitizeMol(cand)
                return Chem.MolToSmiles(cand)
            except Exception:
                continue
        return smiles  # no applicable rule -> unchanged (crossover may still act)


# ==================================================================
#  Advanced Graph Crossover Operators
# ==================================================================
def _cut_to_fragments(mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, Chem.Mol]]:
    """Break one random acyclic single bond -> two fragments, each carrying a
    map-numbered dummy atom so molzip can later re-join fragments."""
    cut_bonds = [b.GetIdx() for b in mol.GetBonds()
                 if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
    if not cut_bonds:
        return None
    frag_mol = Chem.FragmentOnBonds(mol, [random.choice(cut_bonds)], addDummies=True)
    try:
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return None
    if len(frags) != 2:
        return None
    tagged = []
    for f in frags:
        rw = Chem.RWMol(f)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:  # the break-point dummy
                atom.SetAtomMapNum(1)
                atom.SetIsotope(0)
        tagged.append(rw.GetMol())
    return tagged[0], tagged[1]


def crossover_jensen(smiles_a: str, smiles_b: str, **kwargs) -> Optional[str]:
    """Baseline Jensen 2019: random acyclic cut and random fragment join."""
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    fa, fb = _cut_to_fragments(ma), _cut_to_fragments(mb)
    if fa is None or fb is None:
        return None
    piece_a = fa[random.randint(0, 1)]
    piece_b = fb[random.randint(0, 1)]
    try:
        combined = Chem.CombineMols(piece_a, piece_b)
        child = Chem.molzip(combined)
        Chem.SanitizeMol(child)
        return Chem.MolToSmiles(child)
    except Exception:
        return None


def crossover_mass_balanced(smiles_a: str, smiles_b: str, target_mw: Optional[float] = None, **kwargs) -> Optional[str]:
    """Mass-Balanced: selects fragment pairing that minimizes drift from parent MW."""
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    if target_mw is None:
        target_mw = (Descriptors.MolWt(ma) + Descriptors.MolWt(mb)) / 2.0

    fa, fb = _cut_to_fragments(ma), _cut_to_fragments(mb)
    if fa is None or fb is None:
        return None

    candidates = []
    for pa in fa:
        for pb in fb:
            try:
                comb = Chem.CombineMols(pa, pb)
                child = Chem.molzip(comb)
                Chem.SanitizeMol(child)
                mw = Descriptors.MolWt(child)
                diff = abs(mw - target_mw)
                candidates.append((diff, Chem.MolToSmiles(child)))
            except Exception:
                continue

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def crossover_brics(smiles_a: str, smiles_b: str, target_mw: Optional[float] = None, **kwargs) -> Optional[str]:
    """BRICS Retrosynthetic: recombines only on synthetically accessible bonds."""
    from rdkit.Chem import BRICS
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    if target_mw is None:
        target_mw = (Descriptors.MolWt(ma) + Descriptors.MolWt(mb)) / 2.0

    try:
        frags_a = list(BRICS.BRICSDecompose(ma, returnMols=True))
        frags_b = list(BRICS.BRICSDecompose(mb, returnMols=True))
    except Exception:
        return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

    if len(frags_a) < 2 or len(frags_b) < 2:
        return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

    random.shuffle(frags_a)
    random.shuffle(frags_b)
    
    candidates = []
    for fa in frags_a[:3]:
        for fb in frags_b[:3]:
            try:
                builder = BRICS.BRICSBuild([fa, fb])
                for count, child in enumerate(builder):
                    if count >= 3:
                        break
                    try:
                        Chem.SanitizeMol(child)
                        smi = Chem.MolToSmiles(child)
                        if smi and Chem.MolFromSmiles(smi) is not None:
                            diff = abs(Descriptors.MolWt(child) - target_mw)
                            candidates.append((diff, smi))
                    except Exception:
                        continue
            except Exception:
                continue

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)


def crossover_scaffold_preserving(smiles_a: str, smiles_b: str, target_mw: Optional[float] = None, **kwargs) -> Optional[str]:
    """Scaffold-Preserving: locks Murcko core and swaps peripheral R-groups."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    if target_mw is None:
        target_mw = (Descriptors.MolWt(ma) + Descriptors.MolWt(mb)) / 2.0

    try:
        scaf_a = MurckoScaffold.GetScaffoldForMol(ma)
        if scaf_a.GetNumAtoms() == 0 or scaf_a.GetNumAtoms() == ma.GetNumAtoms():
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        scaf_match = set(ma.GetSubstructMatch(scaf_a))
        exo_bonds = []
        for b in ma.GetBonds():
            b_idx = b.GetIdx()
            begin, end = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (begin in scaf_match and end not in scaf_match) or (end in scaf_match and begin not in scaf_match):
                if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing():
                    exo_bonds.append(b_idx)

        if not exo_bonds:
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        frag_mol_a = Chem.FragmentOnBonds(ma, [random.choice(exo_bonds)], addDummies=True)
        frags_a = Chem.GetMolFrags(frag_mol_a, asMols=True, sanitizeFrags=True)
        if len(frags_a) != 2:
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        core_frag = None
        for f in frags_a:
            if f.GetNumAtoms() >= scaf_a.GetNumAtoms() * 0.7:
                core_frag = f
                break
        if core_frag is None:
            core_frag = frags_a[0]

        rw_core = Chem.RWMol(core_frag)
        for atom in rw_core.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(1)
                atom.SetIsotope(0)
        core_tagged = rw_core.GetMol()

        cut_bonds_b = [b.GetIdx() for b in mb.GetBonds()
                       if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
        if not cut_bonds_b:
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        frag_mol_b = Chem.FragmentOnBonds(mb, [random.choice(cut_bonds_b)], addDummies=True)
        frags_b = Chem.GetMolFrags(frag_mol_b, asMols=True, sanitizeFrags=True)
        if len(frags_b) != 2:
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        r_frag = min(frags_b, key=lambda m: m.GetNumHeavyAtoms())
        rw_r = Chem.RWMol(r_frag)
        for atom in rw_r.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(1)
                atom.SetIsotope(0)
        r_tagged = rw_r.GetMol()

        comb = Chem.CombineMols(core_tagged, r_tagged)
        child = Chem.molzip(comb)
        Chem.SanitizeMol(child)
        return Chem.MolToSmiles(child)
    except Exception:
        return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)


def crossover_hybrid_adaptive(smiles_a: str, smiles_b: str, target_mw: Optional[float] = None, **kwargs) -> Optional[str]:
    """Hybrid-Adaptive: 45% scaffold preservation, 35% BRICS, 20% mass-balanced."""
    roll = random.random()
    if roll < 0.45:
        res = crossover_scaffold_preserving(smiles_a, smiles_b, target_mw=target_mw)
        if res: return res
    if roll < 0.80:
        res = crossover_brics(smiles_a, smiles_b, target_mw=target_mw)
        if res: return res
    return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)


CROSSOVER_OPS = {
    "hybrid": crossover_hybrid_adaptive,
    "mass-balanced": crossover_mass_balanced,
    "scaffold": crossover_scaffold_preserving,
    "brics": crossover_brics,
    "jensen": crossover_jensen,
}


def crossover(smiles_a: str, smiles_b: str, op: str = "hybrid", target_mw: Optional[float] = None) -> Optional[str]:
    fn = CROSSOVER_OPS.get(op, crossover_hybrid_adaptive)
    return fn(smiles_a, smiles_b, target_mw=target_mw)


# ==================================================================
#  Objective helpers
# ==================================================================
def morgan(mol: Chem.Mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 2048)


def tanimoto(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# --- structural alerts (PAINS + Brenk): a plausible compound avoids them ---
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

_alert_params = FilterCatalogParams()
_alert_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_alert_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_ALERT_CATALOG = FilterCatalog(_alert_params)


def structural_alerts(mol: Chem.Mol) -> int:
    """Number of PAINS/Brenk unwanted-substructure hits (0 = clean)."""
    try:
        return len(_ALERT_CATALOG.GetMatches(mol))
    except Exception:
        return 0


def edit_metrics(parent_smiles: str, child_smiles: str, edited: List[int]) -> Dict:
    """Evaluate the CHANGE itself (not just the molecule): its size, where it
    sits (scaffold vs periphery), and the property shift it causes. These are
    the difficulty knobs — small, peripheral, low-shift edits are subtler."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    out: Dict = {"edit_size": len(edited)}
    parent, child = Chem.MolFromSmiles(parent_smiles), Chem.MolFromSmiles(child_smiles)
    if parent is None or child is None:
        return out
    out["delta_logp"] = round(Descriptors.MolLogP(child) - Descriptors.MolLogP(parent), 2)
    out["delta_mw"] = round(Descriptors.MolWt(child) - Descriptors.MolWt(parent), 1)
    out["delta_tpsa"] = round(Descriptors.TPSA(child) - Descriptors.TPSA(parent), 1)
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(child)
        scaf_atoms = set(child.GetSubstructMatch(scaf)) if scaf.GetNumAtoms() else set()
        if edited:
            frac = sum(1 for a in edited if a in scaf_atoms) / len(edited)
            out["scaffold_edit_fraction"] = round(frac, 2)
            out["edit_location"] = "scaffold" if frac > 0.5 else "periphery"
    except Exception:
        pass
    return out


def shape_similarity_3d(parent_mol: Chem.Mol, child_smiles: str) -> Optional[float]:
    """3D shape Tanimoto (O3A-aligned) between parent and child. Uses the
    conformers we can build; run post-hoc on the Pareto front, not in the inner
    loop (ETKDG per candidate would slow the GA ~10-20x)."""
    try:
        from rdkit.Chem import rdMolAlign, rdShapeHelpers
        ref = Chem.AddHs(Chem.Mol(parent_mol))
        prb = Chem.AddHs(Chem.MolFromSmiles(child_smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(ref, params) != 0 or AllChem.EmbedMolecule(prb, params) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(ref)
        AllChem.MMFFOptimizeMolecule(prb)
        rdMolAlign.GetO3A(prb, ref).Align()
        return round(1.0 - rdShapeHelpers.ShapeTanimotoDist(prb, ref), 3)
    except Exception:
        return None


class Objectives:
    """Vector of minimization objectives for one candidate vs. the parent.
    Supports 3 profiles:
      - 'anti-shortcut': Property-invariance, target-centered proximity & scaffold preservation (Formulation 2 - Recommended)
      - 'adversarial': Chemical feasibility manifold + GNN detector deception probability (Formulation 3)
      - 'baseline': Jensen 2019 flat band + QED + SA (Formulation 1)
    """

    def __init__(self, parent_smiles: str, sim_lo: float = 0.4, sim_hi: float = 0.9,
                 adversarial: Optional[Callable[[str], float]] = None,
                 profile: str = "anti-shortcut"):
        self.parent_smiles = parent_smiles
        self.parent_mol = Chem.MolFromSmiles(parent_smiles)
        self.parent_fp = morgan(self.parent_mol)
        self.parent_mw = float(Descriptors.MolWt(self.parent_mol))
        self.parent_logp = float(Descriptors.MolLogP(self.parent_mol))
        self.parent_atoms = self.parent_mol.GetNumHeavyAtoms()
        
        scaf = MurckoScaffold.GetScaffoldForMol(self.parent_mol)
        self.parent_scaffold = scaf if scaf.GetNumAtoms() > 0 else None
        
        self.sim_lo, self.sim_hi = sim_lo, sim_hi
        self.adversarial = adversarial
        self.profile = profile

        if profile == "adversarial" and adversarial:
            self.obj_names = ["proximity_mass", "feasibility", "p_counterfeit"]
            self.n_obj = 3
        elif profile == "anti-shortcut":
            self.obj_names = ["proximity_scaffold", "anti_shortcut_drift", "plausibility_sparsity"]
            if adversarial:
                self.obj_names.append("p_counterfeit")
            self.n_obj = len(self.obj_names)
        else: # baseline
            self.obj_names = ["band_penalty", "implausibility", "sa_over_10"]
            if adversarial:
                self.obj_names.append("p_counterfeit")
            self.n_obj = len(self.obj_names)

    def evaluate(self, smiles: str) -> List[float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < 5:
            return [1.0] * self.n_obj
        sim = tanimoto(self.parent_fp, morgan(mol))
        if sim >= 0.999:
            return [1.0] * self.n_obj

        # -----------------------------------------------------------
        # Profile 1: Anti-Shortcut & Bioisosteric Sparsity (Recommended)
        # -----------------------------------------------------------
        if self.profile == "anti-shortcut":
            # f1: Target proximity (sweet-spot around 0.70) + Scaffold Retention
            target_pen = 3.5 * ((sim - 0.70) ** 2)
            scaf_pen = 0.0
            if self.parent_scaffold is not None and not mol.HasSubstructMatch(self.parent_scaffold):
                scaf_pen = 0.35
            f1 = min(1.0, target_pen + scaf_pen)

            # f2: Anti-Shortcut Property Invariance (|ΔMW| & |ΔLogP|)
            delta_mw = abs(Descriptors.MolWt(mol) - self.parent_mw)
            delta_logp = abs(Descriptors.MolLogP(mol) - self.parent_logp)
            f2 = min(1.0, (delta_mw / 60.0) + (delta_logp / 1.5))

            # f3: Composite Plausibility, SA, Alerts & Local Edit Sparsity
            plaus = 1.0 - float(Descriptors.qed(mol))
            sa = ((sascorer.calculateScore(mol) - 1.0) / 9.0) if sascorer else 0.0
            sa = min(max(sa, 0.0), 1.0)
            alerts = structural_alerts(mol)
            atom_diff = abs(mol.GetNumHeavyAtoms() - self.parent_atoms)
            sparsity_pen = min(1.0, atom_diff / 5.0)

            f3 = min(1.0, 0.45 * plaus + 0.35 * sa + 0.15 * min(alerts, 2) + 0.05 * sparsity_pen)
            objs = [float(f1), float(f2), float(f3)]
            if self.adversarial:
                objs.append(float(self.adversarial(smiles)))
            return objs

        # -----------------------------------------------------------
        # Profile 2: Adversarial Hardness
        # -----------------------------------------------------------
        elif self.profile == "adversarial" and self.adversarial:
            band = max(0.0, self.sim_lo - sim) + max(0.0, sim - self.sim_hi)
            delta_mw = abs(Descriptors.MolWt(mol) - self.parent_mw)
            f1 = min(1.0, band + 0.3 * (delta_mw / 100.0))

            plaus = 1.0 - float(Descriptors.qed(mol))
            sa = ((sascorer.calculateScore(mol) - 1.0) / 9.0) if sascorer else 0.0
            alerts = structural_alerts(mol)
            f2 = min(1.0, 0.5 * plaus + 0.35 * sa + 0.15 * min(alerts, 3))

            f3 = float(self.adversarial(smiles))
            return [float(f1), float(f2), float(f3)]

        # -----------------------------------------------------------
        # Profile 3: Baseline Jensen 2019
        # -----------------------------------------------------------
        else:
            band = max(0.0, self.sim_lo - sim) + max(0.0, sim - self.sim_hi)
            plausibility_obj = 1.0 - float(Descriptors.qed(mol))
            n_alerts = structural_alerts(mol)
            if n_alerts:
                plausibility_obj = min(1.0, plausibility_obj + 0.3 * min(n_alerts, 3))
            sa_obj = ((sascorer.calculateScore(mol) - 1.0) / 9.0) if sascorer else 0.0
            sa_obj = min(max(sa_obj, 0.0), 1.0)
            objs = [float(band), float(plausibility_obj), float(sa_obj)]
            if self.adversarial:
                objs.append(float(self.adversarial(smiles)))
            return objs


def edited_atoms(parent_smiles: str, child_smiles: str) -> List[int]:
    """Atom indices in the CHILD that fall outside the maximum common
    substructure with the parent — i.e. what the transformation changed.
    This is the explanation ground-truth for the benchmark."""
    parent, child = Chem.MolFromSmiles(parent_smiles), Chem.MolFromSmiles(child_smiles)
    if parent is None or child is None:
        return []
    try:
        res = rdFMCS.FindMCS([parent, child], timeout=5,
                             matchValences=False, ringMatchesRingOnly=True,
                             completeRingsOnly=False)
        if not res.smartsString:
            return list(range(child.GetNumAtoms()))
        patt = Chem.MolFromSmarts(res.smartsString)
        matched = set(child.GetSubstructMatch(patt))
        return [a.GetIdx() for a in child.GetAtoms() if a.GetIdx() not in matched]
    except Exception:
        return []


# ==================================================================
#  Optional adversarial objective: the trained detector, model-in-the-loop
# ==================================================================
def _select_checkpoint(results_dir: str) -> Optional[Path]:
    """Highest-F1 HGT checkpoint (by filename) in results_dir, or None."""
    import re
    best = None
    for f in Path(results_dir).glob("best_model_f1_*.pt"):
        m = re.search(r"f1_([0-9]+\.[0-9]+)", f.name)
        f1 = float(m.group(1)) if m else 0.0
        if best is None or f1 > best[0]:
            best = (f1, f)
    return best[1] if best else None


def load_detector_scorer(results_dir: str = "HGT_Enhanced_Results"
                         ) -> Optional[Callable[[str], float]]:
    """Load the trained HGT detector as a scorer  smiles -> P(counterfeit).

    Robust to the architecture drift in the saved checkpoints: it picks the
    best-F1 checkpoint and rebuilds the model with the exact hidden width and
    node-type set stored in that checkpoint (inferred from the state dict),
    rather than the code defaults. Returns None (and logs) on any failure so
    the GA still runs with the 3 structural objectives.
    """
    try:
        import torch
        from hyp import RobustEnhancedHGTDetector
        from explicablity import GraphConverter

        ckpt_path = _select_checkpoint(results_dir)
        if ckpt_path is None:
            logger.warning(f"No checkpoint in {results_dir}; running 3-objective.")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ck["model_state_dict"]
        feature_dims = ck["feature_dims"]
        # hidden width == output dim of any node-embedding Linear
        emb_key = next(k for k in sd
                       if k.startswith("node_embeddings") and k.endswith("weight"))
        hidden = sd[emb_key].shape[0]

        model = RobustEnhancedHGTDetector(feature_dims=feature_dims,
                                          hidden_channels=hidden).to(device)
        model.load_state_dict(sd)  # strict: verifies the architecture matches
        model.eval()
        logger.info(f"Detector loaded ({ckpt_path.name}, hidden={hidden}, "
                    f"{len(feature_dims)} node types): adversarial objective ENABLED.")

        def scorer(smiles: str) -> float:
            g = GraphConverter.smiles_to_heterograph(smiles)
            if g is None:
                return 1.0  # can't build graph -> treat as easy (not adversarial)
            g = g.to(device)
            with torch.no_grad():
                out = model(g.x_dict, g.edge_index_dict)
                prob = torch.softmax(out, dim=-1).squeeze()
                return float(prob[1].item())  # class 1 == counterfeit
        return scorer
    except Exception as e:
        logger.warning(f"Could not load detector ({e}); running 3-objective.")
        return None


# ==================================================================
#  pymoo wiring — custom operators over SMILES objects
# ==================================================================
def build_and_run_probabilistic_crowding(parent_smiles: str, rule_lib: RuleLibrary, objectives: Objectives,
                                        pop_size: int, generations: int, seed: int,
                                        crossover_op: str = "mass-balanced",
                                        compute_shape: bool = False) -> List[Dict]:
    """Mengshoel & Goldberg (1999) Probabilistic Crowding for Molecular Niching & Diversity."""
    random.seed(seed)
    np.random.seed(seed)
    parent_mw = Descriptors.MolWt(objectives.parent_mol) if objectives.parent_mol else 250.0

    def get_fitness(smi: str) -> float:
        objs = objectives.evaluate(smi)
        cost = sum(objs)
        return 1.0 / (cost + 1e-4)

    # Initial Population seeded with mutations
    pop = []
    for _ in range(pop_size):
        s = parent_smiles
        for _ in range(random.randint(1, 3)):
            s = rule_lib.mutate(s)
        pop.append(s)

    all_history = set(pop)
    logger.info(f"Running Probabilistic Crowding (Mengshoel & Goldberg 1999) ({crossover_op}): pop={pop_size}, generations={generations}")

    for gen in range(generations):
        random.shuffle(pop)
        new_pop = []
        for i in range(0, len(pop) - 1, 2):
            p1, p2 = pop[i], pop[i + 1]
            
            # Crossover & Mutation
            c1 = crossover(p1, p2, op=crossover_op, target_mw=parent_mw) or p1
            c2 = crossover(p2, p1, op=crossover_op, target_mw=parent_mw) or p2
            
            if random.random() < 0.6:
                c1 = rule_lib.mutate(c1)
            if random.random() < 0.6:
                c2 = rule_lib.mutate(c2)

            all_history.add(c1)
            all_history.add(c2)

            m1, m2 = Chem.MolFromSmiles(p1), Chem.MolFromSmiles(p2)
            mc1, mc2 = Chem.MolFromSmiles(c1), Chem.MolFromSmiles(c2)

            if None in (m1, m2, mc1, mc2):
                new_pop.extend([p1, p2])
                continue

            fp1, fp2 = morgan(m1), morgan(m2)
            fpc1, fpc2 = morgan(mc1), morgan(mc2)

            # Structural distance matching on Tanimoto: d(x, y) = 1 - Tanimoto(x, y)
            d_direct = (1.0 - tanimoto(fp1, fpc1)) + (1.0 - tanimoto(fp2, fpc2))
            d_cross  = (1.0 - tanimoto(fp1, fpc2)) + (1.0 - tanimoto(fp2, fpc1))

            pairs = [(p1, c1), (p2, c2)] if d_direct <= d_cross else [(p1, c2), (p2, c1)]

            for p_cand, c_cand in pairs:
                f_p = get_fitness(p_cand)
                f_c = get_fitness(c_cand)
                # Mengshoel & Goldberg Probabilistic Tournament: P(child wins) = f(c) / (f(c) + f(p))
                p_win = f_c / (f_c + f_p + 1e-9)
                winner = c_cand if random.random() < p_win else p_cand
                new_pop.append(winner)

        pop = new_pop

    # Format candidates
    front = []
    obj_names = getattr(objectives, "obj_names", ["band_penalty", "implausibility", "sa_over_10"])

    seen = set()
    for smi in all_history:
        if smi in seen or Chem.MolFromSmiles(smi) is None:
            continue
        seen.add(smi)
        mol = Chem.MolFromSmiles(smi)
        sim = tanimoto(objectives.parent_fp, morgan(mol))
        if sim >= 0.999:
            continue
        fi = objectives.evaluate(smi)
        edits = edited_atoms(parent_smiles, smi)
        rec = {
            "smiles": smi,
            "similarity_to_parent": round(sim, 3),
            "qed": round(float(Descriptors.qed(mol)), 3),
            "sa_score": round(sascorer.calculateScore(mol), 2) if sascorer else None,
            "structural_alerts": structural_alerts(mol),
            "objectives": {n: round(float(v), 4) for n, v in zip(obj_names, fi)},
            "edited_atoms": edits,
            **edit_metrics(parent_smiles, smi, edits),
        }
        if compute_shape:
            rec["shape_similarity_3d"] = shape_similarity_3d(objectives.parent_mol, smi)
        front.append(rec)

    front.sort(key=lambda d: sum(d["objectives"].values()))
    return front


def build_and_run(parent_smiles: str, rule_lib: RuleLibrary, objectives: Objectives,
                  pop_size: int, generations: int, seed: int,
                  crossover_op: str = "hybrid",
                  algorithm: str = "nsga2",
                  compute_shape: bool = False) -> List[Dict]:
    if algorithm == "probabilistic-crowding":
        return build_and_run_probabilistic_crowding(parent_smiles, rule_lib, objectives,
                                                   pop_size=pop_size, generations=generations,
                                                   seed=seed, crossover_op=crossover_op,
                                                   compute_shape=compute_shape)

    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.duplicate import ElementwiseDuplicateElimination
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    parent_mw = Descriptors.MolWt(objectives.parent_mol) if objectives.parent_mol else 250.0

    class MolProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=1, n_obj=objectives.n_obj, n_constr=0)

        def _evaluate(self, x, out, *a, **k):
            out["F"] = np.array(objectives.evaluate(x[0]), dtype=float)

    class MolSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            # seed the population with mutated variants of the parent
            X = np.empty((n_samples, 1), dtype=object)
            for i in range(n_samples):
                s = parent_smiles
                for _ in range(random.randint(1, 3)):  # 1-3 stacked edits
                    s = rule_lib.mutate(s)
                X[i, 0] = s
            return X

    class MolCrossover(Crossover):
        def __init__(self):
            super().__init__(n_parents=2, n_offsprings=2)

        def _do(self, problem, X, **kwargs):
            _, n_matings, _ = X.shape
            Y = np.empty((self.n_offsprings, n_matings, 1), dtype=object)
            for m in range(n_matings):
                a, b = X[0, m, 0], X[1, m, 0]
                for o in range(self.n_offsprings):
                    child = crossover(a, b, op=crossover_op, target_mw=parent_mw)
                    Y[o, m, 0] = child if child is not None else (a if o == 0 else b)
            return Y

    class MolMutation(Mutation):
        def __init__(self, prob=0.6):
            super().__init__()
            self.prob = prob

        def _do(self, problem, X, **kwargs):
            for i in range(len(X)):
                if random.random() < self.prob:
                    X[i, 0] = rule_lib.mutate(X[i, 0])
            return X

    class SmilesDuplicate(ElementwiseDuplicateElimination):
        def is_equal(self, a, b):
            return a.X[0] == b.X[0]

    ga_algo = NSGA2(
        pop_size=pop_size,
        sampling=MolSampling(),
        crossover=MolCrossover(),
        mutation=MolMutation(),
        eliminate_duplicates=SmilesDuplicate(),
    )

    logger.info(f"Running NSGA-II ({crossover_op} crossover): pop={pop_size}, generations={generations}, "
                f"objectives={objectives.n_obj}")
    res = minimize(MolProblem(), ga_algo, ("n_gen", generations),
                   seed=seed, verbose=False)

    # Collect the Pareto front
    front = []
    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    obj_names = getattr(objectives, "obj_names", ["band_penalty", "implausibility", "sa_over_10"])
    seen = set()
    for xi, fi in zip(X, F):
        smi = xi[0]
        if smi in seen or Chem.MolFromSmiles(smi) is None:
            continue
        seen.add(smi)
        mol = Chem.MolFromSmiles(smi)
        sim = tanimoto(objectives.parent_fp, morgan(mol))
        edits = edited_atoms(parent_smiles, smi)
        rec = {
            "smiles": smi,
            "similarity_to_parent": round(sim, 3),
            "qed": round(float(Descriptors.qed(mol)), 3),
            "sa_score": round(sascorer.calculateScore(mol), 2) if sascorer else None,
            "structural_alerts": structural_alerts(mol),
            "objectives": {n: round(float(v), 4) for n, v in zip(obj_names, fi)},
            "edited_atoms": edits,
            **edit_metrics(parent_smiles, smi, edits),
        }
        if compute_shape:
            rec["shape_similarity_3d"] = shape_similarity_3d(objectives.parent_mol, smi)
        front.append(rec)
    front.sort(key=lambda d: sum(d["objectives"].values()))
    return front


# ==================================================================
#  Main
# ==================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parent", required=True, help="Genuine drug SMILES to counterfeit")
    p.add_argument("--rules", default="mined_transformations.py",
                   help="Phase-A rule module (with MMP_TRANSFORMATIONS + SUPPORT)")
    p.add_argument("--pop", type=int, default=60, help="Population size")
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--sim-lo", type=float, default=0.4, help="Lower Tanimoto band")
    p.add_argument("--sim-hi", type=float, default=0.9, help="Upper Tanimoto band")
    p.add_argument("--adversarial", action="store_true",
                   help="Add the trained detector as a 4th, adversarial objective")
    p.add_argument("--objective-profile", choices=["anti-shortcut", "adversarial", "baseline"],
                   default="anti-shortcut", help="Multi-objective function formulation profile (default: anti-shortcut)")
    p.add_argument("--no-shape", action="store_true",
                   help="Skip 3D shape similarity on the Pareto front (faster)")
    p.add_argument("--algorithm", choices=["nsga2", "probabilistic-crowding"],
                   default="nsga2", help="GA search algorithm (default: nsga2)")
    p.add_argument("--crossover-op", choices=["hybrid", "mass-balanced", "scaffold", "brics", "jensen"],
                   default="hybrid", help="Graph crossover operator (default: hybrid)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="evolved_counterfeits.json")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    parent = Chem.MolFromSmiles(args.parent)
    if parent is None:
        logger.error(f"Unparseable parent SMILES: {args.parent}")
        sys.exit(1)
    canon_parent = Chem.MolToSmiles(parent)
    logger.info(f"Parent: {canon_parent}  "
                f"(QED={Descriptors.qed(parent):.2f}, "
                f"SA={sascorer.calculateScore(parent):.2f})" if sascorer else "")

    rule_lib = RuleLibrary(args.rules)
    adv = load_detector_scorer() if args.adversarial else None
    objectives = Objectives(canon_parent, args.sim_lo, args.sim_hi, adversarial=adv,
                            profile=args.objective_profile)

    front = build_and_run(canon_parent, rule_lib, objectives,
                          pop_size=args.pop, generations=args.generations, seed=args.seed,
                          crossover_op=args.crossover_op,
                          algorithm=args.algorithm,
                          compute_shape=not args.no_shape)

    out = {
        "parent_smiles": canon_parent,
        "similarity_band": [args.sim_lo, args.sim_hi],
        "n_objectives": objectives.n_obj,
        "pareto_front_size": len(front),
        "front": front,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info(f"Pareto front: {len(front)} counterfeits -> {args.out}")

    has_adv = objectives.n_obj == 4
    pc_hdr = f"{'P(cf)':>6}" if has_adv else ""
    logger.info(f"\nTop {min(15, len(front))} counterfeits (by summed objectives):")
    logger.info(f"  {'sim':>5} {'QED':>5} {'SA':>5}{pc_hdr}  #edit  SMILES")
    for d in front[:15]:
        pc = f"{d['objectives']['p_counterfeit']:>6.3f}" if has_adv else ""
        logger.info(f"  {d['similarity_to_parent']:>5} {d['qed']:>5} "
                    f"{str(d['sa_score']):>5}{pc}  {len(d['edited_atoms']):>5}  {d['smiles']}")
    if has_adv:
        logger.info("  (P(cf) = detector's counterfeit probability; low = fools the detector)")


if __name__ == "__main__":
    main()
