"""
Benchmark and Evaluation of Graph Crossover Operators for Counterfeit / Counterfactual Generation
==================================================================================================
Tests 5 graph crossover operators:
  1. Baseline Jensen (2019) Graph-GA (Cut & MolZip)
  2. Mass-Balanced Jensen (Target-preserving fragment matching)
  3. BRICS Retrosynthetic Crossover (Reaction-aware valid bonding)
  4. Scaffold-Preserving (Murcko / MCS R-group exchange)
  5. Hybrid-Adaptive (Multi-tier hierarchical crossover)

Evaluates:
  - Micro-benchmarks (1,000 iterations per operator on multiple drugs)
  - Macro-benchmarks (Full NSGA-II evolutionary runs)
  - Produces detailed metrics (Validity, SA, QED, Tanimoto band, Scaffold preservation, Speed)
  - Generates comprehensive charts (PNG) and JSON logs.
"""

import sys
import os
import time
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# SA score
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except Exception:
    sascorer = None

# Structural alerts filter (PAINS & Brenk)
_alert_params = FilterCatalogParams()
_alert_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_alert_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_ALERT_CATALOG = FilterCatalog(_alert_params)

def get_structural_alerts(mol: Chem.Mol) -> int:
    try:
        return len(_ALERT_CATALOG.GetMatches(mol))
    except Exception:
        return 0

def morgan_fp(mol: Chem.Mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, 2048)

def tanimoto(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# =====================================================================
#  CROSSOVER OPERATORS IMPLEMENTATION
# =====================================================================

# 1. Base helper: cut on acyclic single bonds
def _cut_to_fragments(mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, Chem.Mol]]:
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
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(1)
                atom.SetIsotope(0)
        tagged.append(rw.GetMol())
    return tagged[0], tagged[1]

# 1. Operator: Jensen Baseline
def crossover_jensen(smiles_a: str, smiles_b: str, **kwargs) -> Optional[str]:
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

# 2. Operator: Mass-Balanced Jensen
def crossover_mass_balanced(smiles_a: str, smiles_b: str, target_mw: Optional[float] = None, **kwargs) -> Optional[str]:
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

# 3. Operator: BRICS Retrosynthetic Crossover
def crossover_brics(smiles_a: str, smiles_b: str, target_mw: Optional[float] = None, **kwargs) -> Optional[str]:
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

    # Recombine via BRICS build
    random.shuffle(frags_a)
    random.shuffle(frags_b)
    
    candidates = []
    # Test top complementary fragment pairs
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

    # Fallback to mass-balanced
    return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

# 4. Operator: Scaffold-Preserving (Murcko / MCS R-group exchange)
def crossover_scaffold_preserving(smiles_a: str, smiles_b: str, target_mw: Optional[float] = None, **kwargs) -> Optional[str]:
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    if target_mw is None:
        target_mw = (Descriptors.MolWt(ma) + Descriptors.MolWt(mb)) / 2.0

    try:
        scaf_a = MurckoScaffold.GetScaffoldForMol(ma)
        if scaf_a.GetNumAtoms() == 0 or scaf_a.GetNumAtoms() == ma.GetNumAtoms():
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        # Cut bonds between scaffold and substituents
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

        # Break one exocyclic bond on A
        frag_mol_a = Chem.FragmentOnBonds(ma, [random.choice(exo_bonds)], addDummies=True)
        frags_a = Chem.GetMolFrags(frag_mol_a, asMols=True, sanitizeFrags=True)
        if len(frags_a) != 2:
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        # Identify which fragment is scaffold core
        core_frag = None
        for f in frags_a:
            if f.GetNumAtoms() >= scaf_a.GetNumAtoms() * 0.7:
                core_frag = f
                break
        if core_frag is None:
            core_frag = frags_a[0]

        # Tag dummy on core_frag
        rw_core = Chem.RWMol(core_frag)
        for atom in rw_core.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(1)
                atom.SetIsotope(0)
        core_tagged = rw_core.GetMol()

        # Extract an R-group fragment from B
        cut_bonds_b = [b.GetIdx() for b in mb.GetBonds()
                       if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
        if not cut_bonds_b:
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        frag_mol_b = Chem.FragmentOnBonds(mb, [random.choice(cut_bonds_b)], addDummies=True)
        frags_b = Chem.GetMolFrags(frag_mol_b, asMols=True, sanitizeFrags=True)
        if len(frags_b) != 2:
            return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)

        # Choose the smaller fragment as substituent
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

# 5. Operator: Hybrid Adaptive Crossover
def crossover_hybrid_adaptive(smiles_a: str, smiles_b: str, target_mw: Optional[float] = None, **kwargs) -> Optional[str]:
    roll = random.random()
    if roll < 0.45:
        res = crossover_scaffold_preserving(smiles_a, smiles_b, target_mw=target_mw)
        if res: return res
    if roll < 0.80:
        res = crossover_brics(smiles_a, smiles_b, target_mw=target_mw)
        if res: return res
    return crossover_mass_balanced(smiles_a, smiles_b, target_mw=target_mw)


# Dictionary of operators
OPERATORS = {
    "Jensen (Baseline)": crossover_jensen,
    "Mass-Balanced": crossover_mass_balanced,
    "BRICS Retrosynthetic": crossover_brics,
    "Scaffold-Preserving": crossover_scaffold_preserving,
    "Hybrid-Adaptive": crossover_hybrid_adaptive,
}

# Test drugs across different classes and sizes
TEST_DRUGS = [
    {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "mw": 180.16},
    {"name": "Paracetamol", "smiles": "CC(=O)Nc1ccc(O)cc1", "mw": 151.16},
    {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "mw": 206.28},
    {"name": "Sildenafil", "smiles": "CCCC1=NN(C)C2=C1N=C(NC2=O)C1=C(OCC)C=CC(=C1)S(=O)(=O)N1CCN(C)CC1", "mw": 474.58},
]


# =====================================================================
#  MICRO-BENCHMARK: DIRECT OPERATOR COMPARISON
# =====================================================================

def run_micro_benchmark(n_trials_per_drug: int = 200) -> Dict[str, Any]:
    print(f"--- Running Micro-Benchmark ({n_trials_per_drug} pairs per drug per operator) ---")
    import mined_transformations
    from evolutionary_generator import RuleLibrary

    rule_lib = RuleLibrary("mined_transformations.py")

    results = {op_name: {
        "valid_count": 0,
        "total_count": 0,
        "validity_rate": 0.0,
        "tanimoto_similarities": [],
        "in_band_count": 0,
        "qed_scores": [],
        "sa_scores": [],
        "delta_mw": [],
        "scaffold_preserved_count": 0,
        "structural_alerts": [],
        "time_ms_per_call": 0.0
    } for op_name in OPERATORS}

    for drug in TEST_DRUGS:
        parent_smi = drug["smiles"]
        parent_mol = Chem.MolFromSmiles(parent_smi)
        parent_fp = morgan_fp(parent_mol)
        parent_mw = Descriptors.MolWt(parent_mol)
        parent_scaf = MurckoScaffold.GetScaffoldForMol(parent_mol)
        parent_scaf_smi = Chem.MolToSmiles(parent_scaf) if parent_scaf.GetNumAtoms() > 0 else ""

        # Create diverse pool of mutated parents
        pool = []
        for _ in range(50):
            s = parent_smi
            for _ in range(random.randint(1, 3)):
                s = rule_lib.mutate(s)
            if s and Chem.MolFromSmiles(s) is not None:
                pool.append(s)
        if len(pool) < 10:
            pool = [parent_smi] * 20

        for op_name, op_func in OPERATORS.items():
            t0 = time.perf_counter()
            for _ in range(n_trials_per_drug):
                sa = random.choice(pool)
                sb = random.choice(pool)
                results[op_name]["total_count"] += 1
                
                child_smi = op_func(sa, sb, target_mw=parent_mw)
                if child_smi:
                    child_mol = Chem.MolFromSmiles(child_smi)
                    if child_mol is not None:
                        results[op_name]["valid_count"] += 1
                        
                        # Metrics
                        sim = tanimoto(parent_fp, morgan_fp(child_mol))
                        results[op_name]["tanimoto_similarities"].append(sim)
                        if 0.4 <= sim <= 0.9:
                            results[op_name]["in_band_count"] += 1
                            
                        results[op_name]["qed_scores"].append(Descriptors.qed(child_mol))
                        if sascorer:
                            results[op_name]["sa_scores"].append(sascorer.calculateScore(child_mol))
                        results[op_name]["delta_mw"].append(abs(Descriptors.MolWt(child_mol) - parent_mw))
                        results[op_name]["structural_alerts"].append(get_structural_alerts(child_mol))

                        # Scaffold preservation check
                        c_scaf = MurckoScaffold.GetScaffoldForMol(child_mol)
                        c_scaf_smi = Chem.MolToSmiles(c_scaf) if c_scaf.GetNumAtoms() > 0 else ""
                        if parent_scaf_smi and (parent_scaf_smi == c_scaf_smi or child_mol.HasSubstructMatch(parent_scaf)):
                            results[op_name]["scaffold_preserved_count"] += 1

            dt = time.perf_counter() - t0
            results[op_name]["time_ms_per_call"] += (dt / n_trials_per_drug) * 1000.0 / len(TEST_DRUGS)

    # Compute aggregate summary statistics
    summary = {}
    for op_name, data in results.items():
        v_count = data["valid_count"]
        tot = data["total_count"]
        summary[op_name] = {
            "validity_rate_pct": round((v_count / tot) * 100.0, 2) if tot > 0 else 0.0,
            "in_band_pct": round((data["in_band_count"] / v_count) * 100.0, 2) if v_count > 0 else 0.0,
            "mean_tanimoto": round(float(np.mean(data["tanimoto_similarities"])), 3) if data["tanimoto_similarities"] else 0.0,
            "mean_qed": round(float(np.mean(data["qed_scores"])), 3) if data["qed_scores"] else 0.0,
            "mean_sa": round(float(np.mean(data["sa_scores"])), 3) if data["sa_scores"] else 0.0,
            "mean_delta_mw": round(float(np.mean(data["delta_mw"])), 2) if data["delta_mw"] else 0.0,
            "scaffold_preservation_pct": round((data["scaffold_preserved_count"] / v_count) * 100.0, 2) if v_count > 0 else 0.0,
            "mean_alerts": round(float(np.mean(data["structural_alerts"])), 3) if data["structural_alerts"] else 0.0,
            "time_ms_per_op": round(data["time_ms_per_call"], 3),
        }
    return {"raw": results, "summary": summary}


# =====================================================================
#  MACRO-BENCHMARK: FULL NSGA-II EVOLUTIONARY SEARCH COMPARISON
# =====================================================================

def run_macro_ga_benchmark(generations: int = 15, pop_size: int = 40) -> Dict[str, Any]:
    print(f"\n--- Running Macro-Benchmark (NSGA-II Evolution: {generations} gens, pop {pop_size}) ---")
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.duplicate import ElementwiseDuplicateElimination
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    from evolutionary_generator import RuleLibrary, Objectives, edited_atoms, edit_metrics

    rule_lib = RuleLibrary("mined_transformations.py")
    test_drug = TEST_DRUGS[0] # Aspirin
    parent_smi = test_drug["smiles"]
    parent_mol = Chem.MolFromSmiles(parent_smi)
    parent_mw = Descriptors.MolWt(parent_mol)
    objectives = Objectives(parent_smi, sim_lo=0.4, sim_hi=0.9, adversarial=None)

    ga_results = {}

    for op_name, op_func in OPERATORS.items():
        print(f"  > Testing GA with operator: {op_name}...")

        class CustomMolCrossover(Crossover):
            def __init__(self):
                super().__init__(n_parents=2, n_offsprings=2)

            def _do(self, problem, X, **kwargs):
                _, n_matings, _ = X.shape
                Y = np.empty((self.n_offsprings, n_matings, 1), dtype=object)
                for m in range(n_matings):
                    a, b = X[0, m, 0], X[1, m, 0]
                    for o in range(self.n_offsprings):
                        child = op_func(a, b, target_mw=parent_mw)
                        Y[o, m, 0] = child if child is not None else (a if o == 0 else b)
                return Y

        class MolProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=1, n_obj=objectives.n_obj, n_constr=0)
            def _evaluate(self, x, out, *a, **k):
                out["F"] = np.array(objectives.evaluate(x[0]), dtype=float)

        class MolSampling(Sampling):
            def _do(self, problem, n_samples, **kwargs):
                X = np.empty((n_samples, 1), dtype=object)
                for i in range(n_samples):
                    s = parent_smi
                    for _ in range(random.randint(1, 3)):
                        s = rule_lib.mutate(s)
                    X[i, 0] = s
                return X

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
            crossover=CustomMolCrossover(),
            mutation=MolMutation(),
            eliminate_duplicates=SmilesDuplicate(),
        )

        t0 = time.perf_counter()
        res = minimize(MolProblem(), algorithm, ("n_gen", generations), seed=42, verbose=False)
        runtime = round(time.perf_counter() - t0, 2)

        # Extract Pareto Front
        X = np.atleast_2d(res.X)
        F = np.atleast_2d(res.F)
        front = []
        seen = set()
        for xi, fi in zip(X, F):
            smi = xi[0]
            if smi in seen or Chem.MolFromSmiles(smi) is None:
                continue
            seen.add(smi)
            mol = Chem.MolFromSmiles(smi)
            sim = tanimoto(objectives.parent_fp, morgan_fp(mol))
            edits = edited_atoms(parent_smi, smi)
            front.append({
                "smiles": smi,
                "similarity": round(sim, 3),
                "qed": round(float(Descriptors.qed(mol)), 3),
                "sa_score": round(sascorer.calculateScore(mol), 2) if sascorer else None,
                "alerts": get_structural_alerts(mol),
                "edit_size": len(edits),
                "f_band": round(float(fi[0]), 4),
                "f_plaus": round(float(fi[1]), 4),
                "f_sa": round(float(fi[2]), 4),
                "sum_fitness": round(float(np.sum(fi)), 4)
            })

        front.sort(key=lambda d: d["sum_fitness"])
        ga_results[op_name] = {
            "runtime_s": runtime,
            "pareto_front_size": len(front),
            "mean_pareto_fitness": round(float(np.mean([d["sum_fitness"] for d in front])), 4) if front else 1.0,
            "best_pareto_fitness": front[0]["sum_fitness"] if front else 1.0,
            "mean_qed": round(float(np.mean([d["qed"] for d in front])), 3) if front else 0.0,
            "mean_sa": round(float(np.mean([d["sa_score"] for d in front])), 2) if front else 0.0,
            "mean_similarity": round(float(np.mean([d["similarity"] for d in front])), 3) if front else 0.0,
            "mean_edit_size": round(float(np.mean([d["edit_size"] for d in front])), 1) if front else 0.0,
            "top_candidates": front[:5]
        }

    return ga_results


# =====================================================================
#  VISUALIZATION & PLOTTING
# =====================================================================

def generate_benchmark_plots(micro_res: Dict, ga_res: Dict, output_dir: str = "."):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary = micro_res["summary"]
    ops = list(summary.keys())

    # Style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Comprehensive Benchmark of Graph Crossover Operators for Molecular Counterfeits", fontsize=16, fontweight='bold', y=0.98)

    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']

    # 1. Validity Rate & In-Band Percentage
    validity = [summary[op]["validity_rate_pct"] for op in ops]
    in_band = [summary[op]["in_band_pct"] for op in ops]
    x = np.arange(len(ops))
    width = 0.35
    axs[0, 0].bar(x - width/2, validity, width, label='Validity Rate (%)', color='#4e79a7')
    axs[0, 0].bar(x + width/2, in_band, width, label='Tanimoto In-Band (%)', color='#59a14f')
    axs[0, 0].set_title('Chemical Validity & Tanimoto Band [0.4, 0.9]', fontweight='bold')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(ops, rotation=25, ha='right', fontsize=9)
    axs[0, 0].set_ylim(0, 105)
    axs[0, 0].legend(loc='lower right')

    # 2. Scaffold Preservation Rate
    scaf_pres = [summary[op]["scaffold_preservation_pct"] for op in ops]
    axs[0, 1].bar(ops, scaf_pres, color='#e15759', width=0.55)
    axs[0, 1].set_title('Parent Scaffold Preservation Rate (%)', fontweight='bold')
    axs[0, 1].set_xticklabels(ops, rotation=25, ha='right', fontsize=9)
    axs[0, 1].set_ylim(0, 105)
    for i, v in enumerate(scaf_pres):
        axs[0, 1].text(i, v + 2, f"{v}%", ha='center', fontweight='bold', fontsize=9)

    # 3. Mass Drift |Delta MW|
    delta_mw = [summary[op]["mean_delta_mw"] for op in ops]
    axs[0, 2].bar(ops, delta_mw, color='#f28e2b', width=0.55)
    axs[0, 2].set_title('Mass Drift |Δ MW| from Parent (Lower is Better)', fontweight='bold')
    axs[0, 2].set_xticklabels(ops, rotation=25, ha='right', fontsize=9)
    for i, v in enumerate(delta_mw):
        axs[0, 2].text(i, v + 1.5, f"{v} Da", ha='center', fontweight='bold', fontsize=9)

    # 4. Drug-Likeness (QED) vs Synthetic Accessibility (SA)
    qeds = [summary[op]["mean_qed"] for op in ops]
    sas = [summary[op]["mean_sa"] for op in ops]
    axs[1, 0].scatter(sas, qeds, s=220, c=colors, edgecolors='black', linewidth=1.5)
    for i, op in enumerate(ops):
        axs[1, 0].annotate(op, (sas[i], qeds[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    axs[1, 0].set_title('QED vs SA Score (Top-Left = Ideal)', fontweight='bold')
    axs[1, 0].set_xlabel('Synthetic Accessibility Score (Lower = Easier)', fontweight='bold')
    axs[1, 0].set_ylabel('Drug-Likeness QED (Higher = Better)', fontweight='bold')

    # 5. GA Pareto Front Quality (Hypervolume proxy / Front size)
    front_sizes = [ga_res[op]["pareto_front_size"] for op in ops]
    best_fits = [ga_res[op]["best_pareto_fitness"] for op in ops]
    axs[1, 1].bar(x - width/2, front_sizes, width, label='Pareto Front Size', color='#76b7b2')
    axs[1, 1].bar(x + width/2, [1.0 - f for f in best_fits], width, label='Best Fitness Score (1 - Loss)', color='#edc948')
    axs[1, 1].set_title('NSGA-II Search: Front Size & Best Fitness', fontweight='bold')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(ops, rotation=25, ha='right', fontsize=9)
    axs[1, 1].legend(loc='upper right')

    # 6. Execution Time per Crossover Call (Speed)
    speed = [summary[op]["time_ms_per_op"] for op in ops]
    axs[1, 2].bar(ops, speed, color='#b07aa1', width=0.55)
    axs[1, 2].set_title('Crossover Runtime (ms / call)', fontweight='bold')
    axs[1, 2].set_xticklabels(ops, rotation=25, ha='right', fontsize=9)
    for i, v in enumerate(speed):
        axs[1, 2].text(i, v + 0.1, f"{v} ms", ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, "crossover_comparison_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plots successfully saved to: {plot_path}")
    return plot_path


# =====================================================================
#  MAIN BENCHMARK RUNNER
# =====================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    micro_res = run_micro_benchmark(n_trials_per_drug=200)
    ga_res = run_macro_ga_benchmark(generations=15, pop_size=40)

    all_data = {
        "micro_summary": micro_res["summary"],
        "ga_summary": ga_res
    }

    out_json = "crossover_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Results JSON saved to: {out_json}")

    # Generate plots
    plot_file = generate_benchmark_plots(micro_res, ga_res, output_dir=".")

    # Also copy plot to brain artifact directory if exists
    artifact_dir = "/Users/ediionut23/.gemini/antigravity-ide/brain/2b278d2f-cf0c-4258-be50-a816e986b6cb"
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(plot_file, os.path.join(artifact_dir, "crossover_comparison_metrics.png"))
        print(f"Copied plot to artifact directory: {artifact_dir}")

    # Print summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY COMPARISON OF GRAPH CROSSOVER OPERATORS")
    print("="*80)
    headers = ["Operator", "Validity%", "InBand%", "Scaffold%", "ΔMW (Da)", "QED", "SA", "ParetoSize", "Time(ms)"]
    print(f"{headers[0]:<22} {headers[1]:<10} {headers[2]:<9} {headers[3]:<10} {headers[4]:<10} {headers[5]:<6} {headers[6]:<6} {headers[7]:<12} {headers[8]:<8}")
    print("-" * 95)
    for op in OPERATORS:
        m = micro_res["summary"][op]
        g = ga_res[op]
        print(f"{op:<22} {m['validity_rate_pct']:<10.1f} {m['in_band_pct']:<9.1f} {m['scaffold_preservation_pct']:<10.1f} {m['mean_delta_mw']:<10.1f} {m['mean_qed']:<6.2f} {m['mean_sa']:<6.2f} {g['pareto_front_size']:<12} {m['time_ms_per_op']:<8.2f}")
    print("="*80)


if __name__ == "__main__":
    main()
