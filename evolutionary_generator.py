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

    def __init__(self, rules_py: str):
        mod = self._load_module(rules_py)
        raw = list(getattr(mod, "MMP_TRANSFORMATIONS"))
        support = dict(getattr(mod, "SUPPORT", {}))
        self.names: List[str] = []
        self.reactions: List[AllChem.ChemicalReaction] = []
        self.weights: List[float] = []
        for name, _diff, _cat, react, prod, _cite in raw:
            rxn = AllChem.ReactionFromSmarts(f"{react}>>{prod}")
            if rxn is None:
                continue
            self.names.append(name)
            self.reactions.append(rxn)
            self.weights.append(float(support.get(name, 1)))
        w = np.asarray(self.weights, dtype=float)
        self.probs = (w / w.sum()) if w.sum() > 0 else None
        logger.info(f"Loaded {len(self.names)} mutation rules from {rules_py}")

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
#  Graph crossover (Jensen 2019 style, via RDKit molzip)
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


def crossover(smiles_a: str, smiles_b: str) -> Optional[str]:
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
        child = Chem.molzip(combined)  # joins the two map-1 dummies
        Chem.SanitizeMol(child)
        return Chem.MolToSmiles(child)
    except Exception:
        return None


# ==================================================================
#  Objective helpers
# ==================================================================
def morgan(mol: Chem.Mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 2048)


def tanimoto(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


class Objectives:
    """Vector of minimization objectives for one candidate vs. the parent."""

    def __init__(self, parent_smiles: str, sim_lo: float, sim_hi: float,
                 adversarial: Optional[Callable[[str], float]] = None):
        self.parent_smiles = parent_smiles
        self.parent_mol = Chem.MolFromSmiles(parent_smiles)
        self.parent_fp = morgan(self.parent_mol)
        self.sim_lo, self.sim_hi = sim_lo, sim_hi
        self.adversarial = adversarial
        self.n_obj = 4 if adversarial else 3

    def evaluate(self, smiles: str) -> List[float]:
        mol = Chem.MolFromSmiles(smiles)
        # Invalid / degenerate candidates get the worst score on every axis.
        if mol is None or mol.GetNumHeavyAtoms() < 5:
            return [1.0] * self.n_obj
        sim = tanimoto(self.parent_fp, morgan(mol))
        # identical molecule is useless as a "counterfeit"
        if sim >= 0.999:
            return [1.0] * self.n_obj

        band = max(0.0, self.sim_lo - sim) + max(0.0, sim - self.sim_hi)
        qed_obj = 1.0 - float(Descriptors.qed(mol))
        sa_obj = ((sascorer.calculateScore(mol) - 1.0) / 9.0) if sascorer else 0.0
        sa_obj = min(max(sa_obj, 0.0), 1.0)
        objs = [band, qed_obj, sa_obj]
        if self.adversarial:
            objs.append(float(self.adversarial(smiles)))  # P(counterfeit): low = fools detector
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
def load_detector_scorer() -> Optional[Callable[[str], float]]:
    """Best-effort loader for the HGT detector as a scorer smiles -> P(counterfeit).

    Returns None (and logs) if the model/checkpoint/converter cannot be loaded,
    so the GA still runs with the 3 structural objectives.
    """
    try:
        import torch
        from hyp import load_model_and_dataset  # reuses the project's loader
        from explicablity import GraphConverter
        model, device, *_ = _unpack(load_model_and_dataset())

        model.eval()

        def scorer(smiles: str) -> float:
            g = GraphConverter.smiles_to_heterograph(smiles)
            if g is None:
                return 1.0  # can't build graph -> treat as easy (not adversarial)
            g = g.to(device)
            with torch.no_grad():
                out = model(g.x_dict, g.edge_index_dict)
                prob = torch.softmax(out, dim=-1).squeeze()
                # class index 1 == counterfeit in this project
                return float(prob[1].item())
        logger.info("Detector loaded: adversarial objective ENABLED.")
        return scorer
    except Exception as e:
        logger.warning(f"Could not load detector ({e}); running 3-objective.")
        return None


def _unpack(ret):
    """load_model_and_dataset may return a tuple of varying length."""
    if isinstance(ret, (tuple, list)):
        return list(ret) + [None] * (2 - len(ret)) if len(ret) < 2 else list(ret)
    return [ret, "cpu"]


# ==================================================================
#  pymoo wiring — custom operators over SMILES objects
# ==================================================================
def build_and_run(parent_smiles: str, rule_lib: RuleLibrary, objectives: Objectives,
                  pop_size: int, generations: int, seed: int) -> List[Dict]:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.duplicate import ElementwiseDuplicateElimination
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

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
                    child = crossover(a, b)
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

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MolSampling(),
        crossover=MolCrossover(),
        mutation=MolMutation(),
        eliminate_duplicates=SmilesDuplicate(),
    )

    logger.info(f"Running NSGA-II: pop={pop_size}, generations={generations}, "
                f"objectives={objectives.n_obj}")
    res = minimize(MolProblem(), algorithm, ("n_gen", generations),
                   seed=seed, verbose=False)

    # Collect the Pareto front
    front = []
    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    obj_names = ["band_penalty", "one_minus_qed", "sa_over_10"]
    if objectives.n_obj == 4:
        obj_names.append("p_counterfeit")
    seen = set()
    for xi, fi in zip(X, F):
        smi = xi[0]
        if smi in seen or Chem.MolFromSmiles(smi) is None:
            continue
        seen.add(smi)
        mol = Chem.MolFromSmiles(smi)
        sim = tanimoto(objectives.parent_fp, morgan(mol))
        front.append({
            "smiles": smi,
            "similarity_to_parent": round(sim, 3),
            "qed": round(float(Descriptors.qed(mol)), 3),
            "sa_score": round(sascorer.calculateScore(mol), 2) if sascorer else None,
            "objectives": {n: round(float(v), 4) for n, v in zip(obj_names, fi)},
            "edited_atoms": edited_atoms(parent_smiles, smi),
        })
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
    objectives = Objectives(canon_parent, args.sim_lo, args.sim_hi, adversarial=adv)

    front = build_and_run(canon_parent, rule_lib, objectives,
                          pop_size=args.pop, generations=args.generations, seed=args.seed)

    out = {
        "parent_smiles": canon_parent,
        "similarity_band": [args.sim_lo, args.sim_hi],
        "n_objectives": objectives.n_obj,
        "pareto_front_size": len(front),
        "front": front,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info(f"Pareto front: {len(front)} counterfeits -> {args.out}")

    logger.info(f"\nTop {min(15, len(front))} counterfeits (by summed objectives):")
    logger.info(f"  {'sim':>5} {'QED':>5} {'SA':>5}  #edit  SMILES")
    for d in front[:15]:
        logger.info(f"  {d['similarity_to_parent']:>5} {d['qed']:>5} "
                    f"{str(d['sa_score']):>5}  {len(d['edited_atoms']):>5}  {d['smiles']}")


if __name__ == "__main__":
    main()
