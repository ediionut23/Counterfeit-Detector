"""
Benchmark and Evaluation of Multi-Objective Optimization Formulations for Molecular Counterfeits
=================================================================================================
Compares 3 distinct multi-objective function paradigms for generating counterfeit molecules:

  1. Formulation 1: "Baseline Medicinal" (Jensen 2019 Graph-GA)
     - f1: Piecewise-flat Tanimoto band [0.4, 0.9]
     - f2: Drug-likeness (1 - QED) + alerts
     - f3: Synthetic Accessibility (SA / 10)

  2. Formulation 2: "Anti-Shortcut Bioisosteric" (Thesis DUD-E Grounded & Scaffold-Preserving)
     - f1: Smooth target-centered proximity (target ~0.70) + Murcko core retention penalty
     - f2: Anti-shortcut property invariance (|ΔMW| / 60 + |ΔLogP| / 1.5)
     - f3: Composite Plausibility & Edit Sparsity (1 - QED + SA/10 + Alerts + |ΔV|/6)

  3. Formulation 3: "Adversarial Hardness" (Model-in-the-Loop HGT Detector)
     - f1: Bounded proximity + soft mass matching
     - f2: Chemical feasibility manifold (QED + SA + Alerts)
     - f3: Detection probability P(counterfeit) from trained GNN (low = deceives detector)

Evaluates:
  - Pareto front size and yield
  - Tanimoto similarity distribution and sweet-spot compliance
  - Medicinal plausibility (QED, PAINS/Brenk alerts)
  - Synthesizability (SA score)
  - Property drift (ΔMW, ΔLogP - crucial for DUD-E matching)
  - Scaffold preservation rate (%)
  - Edit sparsity (number of edited atoms)
  - Model deception rate (% fooling the trained HGT detector)
  - Runtime / computational throughput
"""

import os
import sys
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
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

import evolutionary_generator as EG

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

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
#  3 OBJECTIVE FUNCTION FORMULATIONS
# =====================================================================

class BaseObjectives:
    def __init__(self, parent_smiles: str, detector_scorer=None):
        self.parent_smiles = parent_smiles
        self.parent_mol = Chem.MolFromSmiles(parent_smiles)
        self.parent_fp = morgan_fp(self.parent_mol)
        self.parent_mw = float(Descriptors.MolWt(self.parent_mol))
        self.parent_logp = float(Descriptors.MolLogP(self.parent_mol))
        self.parent_atoms = self.parent_mol.GetNumHeavyAtoms()
        
        scaf = MurckoScaffold.GetScaffoldForMol(self.parent_mol)
        self.parent_scaffold = scaf if scaf.GetNumAtoms() > 0 else None
        self.detector_scorer = detector_scorer


class BaselineMedicinalObjectives(BaseObjectives):
    """Formulation 1: Standard Jensen 2019 Graph-GA (Band + QED + SA)"""
    name = "Formulation 1: Baseline Medicinal (Jensen)"
    short_name = "Baseline (Jensen)"
    n_obj = 3

    def evaluate(self, smiles: str) -> List[float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < 5:
            return [1.0, 1.0, 1.0]
        sim = tanimoto(self.parent_fp, morgan_fp(mol))
        if sim >= 0.999:
            return [1.0, 1.0, 1.0]

        # f1: Discontinuous band penalty [0.4, 0.9]
        band = max(0.0, 0.4 - sim) + max(0.0, sim - 0.9)

        # f2: Implausibility (1 - QED + alerts)
        plaus = 1.0 - float(Descriptors.qed(mol))
        alerts = get_structural_alerts(mol)
        if alerts:
            plaus = min(1.0, plaus + 0.3 * min(alerts, 3))

        # f3: Synthetic Accessibility (SA / 10)
        sa = ((sascorer.calculateScore(mol) - 1.0) / 9.0) if sascorer else 0.0
        sa = min(max(sa, 0.0), 1.0)

        return [float(band), float(plaus), float(sa)]


class AntiShortcutBioisostericObjectives(BaseObjectives):
    """Formulation 2: Anti-Shortcut & Bioisosteric Sparsity (Thesis Tailored)"""
    name = "Formulation 2: Anti-Shortcut Bioisosteric"
    short_name = "Anti-Shortcut (Thesis)"
    n_obj = 3

    def evaluate(self, smiles: str) -> List[float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < 5:
            return [1.0, 1.0, 1.0]
        sim = tanimoto(self.parent_fp, morgan_fp(mol))
        if sim >= 0.999:
            return [1.0, 1.0, 1.0]

        # f1: Target-centered smooth proximity (target=0.70) + Scaffold Retention
        # Quad penalty: (sim - 0.70)^2 * 3.5. Inside [0.6, 0.8] error is < 0.035.
        target_penalty = 3.5 * ((sim - 0.70) ** 2)
        
        # Scaffold check
        scaf_penalty = 0.0
        if self.parent_scaffold is not None:
            if not mol.HasSubstructMatch(self.parent_scaffold):
                scaf_penalty = 0.35
        f1 = min(1.0, target_penalty + scaf_penalty)

        # f2: Anti-Shortcut Property Invariance (|ΔMW| and |ΔLogP|)
        delta_mw = abs(Descriptors.MolWt(mol) - self.parent_mw)
        delta_logp = abs(Descriptors.MolLogP(mol) - self.parent_logp)
        f2 = min(1.0, (delta_mw / 60.0) + (delta_logp / 1.5))

        # f3: Plausibility + SA + Local Edit Sparsity
        plaus = 1.0 - float(Descriptors.qed(mol))
        sa = ((sascorer.calculateScore(mol) - 1.0) / 9.0) if sascorer else 0.0
        sa = min(max(sa, 0.0), 1.0)
        alerts = get_structural_alerts(mol)
        
        # Fast edit size estimation (atom count difference + non-common atoms)
        atom_diff = abs(mol.GetNumHeavyAtoms() - self.parent_atoms)
        sparsity_pen = min(1.0, atom_diff / 5.0)

        f3 = min(1.0, 0.45 * plaus + 0.35 * sa + 0.15 * min(alerts, 2) + 0.05 * sparsity_pen)

        return [float(f1), float(f2), float(f3)]


class AdversarialHardnessObjectives(BaseObjectives):
    """Formulation 3: Adversarial Hardness & Subtle Evasion (Model-in-the-Loop)"""
    name = "Formulation 3: Adversarial Hardness"
    short_name = "Adversarial (GNN)"
    n_obj = 3

    def evaluate(self, smiles: str) -> List[float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < 5:
            return [1.0, 1.0, 1.0]
        sim = tanimoto(self.parent_fp, morgan_fp(mol))
        if sim >= 0.999:
            return [1.0, 1.0, 1.0]

        # f1: Bounded proximity [0.45, 0.85] + soft MW match
        band = max(0.0, 0.45 - sim) + max(0.0, sim - 0.85)
        delta_mw = abs(Descriptors.MolWt(mol) - self.parent_mw)
        f1 = min(1.0, band + 0.3 * (delta_mw / 100.0))

        # f2: Chemical Feasibility Manifold (QED + SA + Alerts)
        plaus = 1.0 - float(Descriptors.qed(mol))
        sa = ((sascorer.calculateScore(mol) - 1.0) / 9.0) if sascorer else 0.0
        alerts = get_structural_alerts(mol)
        f2 = min(1.0, 0.5 * plaus + 0.35 * sa + 0.15 * min(alerts, 3))

        # f3: Adversarial Deception: P(counterfeit) predicted by detector
        # Lower P(counterfeit) means model thinks it is Authentic (fooled!)
        if self.detector_scorer:
            try:
                p_cf = self.detector_scorer(smiles)
                f3 = float(p_cf)
            except Exception:
                f3 = 0.5
        else:
            # Model-free subtlety proxy: high similarity with minimal heavy-atom delta
            atom_diff = abs(mol.GetNumHeavyAtoms() - self.parent_atoms)
            f3 = min(1.0, (1.0 - sim) + 0.1 * atom_diff)

        return [float(f1), float(f2), float(f3)]


OBJECTIVE_CLASSES = [
    BaselineMedicinalObjectives,
    AntiShortcutBioisostericObjectives,
    AdversarialHardnessObjectives,
]

TEST_DRUGS = [
    {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "mw": 180.16},
    {"name": "Paracetamol", "smiles": "CC(=O)Nc1ccc(O)cc1", "mw": 151.16},
    {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "mw": 206.28},
    {"name": "Sildenafil", "smiles": "CCCC1=NN(C)C2=C1N=C(NC2=O)C1=C(OCC)C=CC(=C1)S(=O)(=O)N1CCN(C)CC1", "mw": 474.58},
]


# =====================================================================
#  GA BENCHMARK ENGINE
# =====================================================================

def run_objective_benchmark(generations: int = 15, pop_size: int = 36, seed: int = 42) -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f"🚀 RUNNING MACRO NSGA-II BENCHMARK: 3 OBJECTIVE FORMULATIONS ({len(TEST_DRUGS)} DRUGS)")
    print(f"   Generations: {generations} | Population: {pop_size} | Crossover: Hybrid-Adaptive")
    print(f"================================================================================")

    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.duplicate import ElementwiseDuplicateElimination
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    rule_lib = EG.RuleLibrary("mined_transformations.py")
    detector = EG.load_detector_scorer("HGT_Enhanced_Results")
    if detector is not None:
        print("✅ Trained HGT detector loaded successfully for adversarial scoring and auditing.")
    else:
        print("⚠️ HGT detector not found; using structural auditor.")

    results = {}

    for obj_cls in OBJECTIVE_CLASSES:
        obj_name = obj_cls.name
        short_name = obj_cls.short_name
        print(f"\n--- Testing {obj_name} ---")

        agg_metrics = {
            "runtimes": [],
            "pareto_sizes": [],
            "similarities": [],
            "in_band_pcts": [],
            "sweet_spot_pcts": [], # 0.55 - 0.80
            "qeds": [],
            "sas": [],
            "alerts": [],
            "delta_mws": [],
            "delta_logps": [],
            "scaffold_preserved_pcts": [],
            "edit_sizes": [],
            "p_counterfeits": [],
            "fooling_pcts": [],
            "top_candidates": []
        }

        for drug in TEST_DRUGS:
            d_name = drug["name"]
            parent_smi = drug["smiles"]
            parent_mol = Chem.MolFromSmiles(parent_smi)
            parent_fp = morgan_fp(parent_mol)
            parent_mw = Descriptors.MolWt(parent_mol)
            parent_logp = Descriptors.MolLogP(parent_mol)
            parent_scaf = MurckoScaffold.GetScaffoldForMol(parent_mol)
            has_scaf = parent_scaf.GetNumAtoms() > 0

            obj_inst = obj_cls(parent_smi, detector_scorer=detector)

            class MolProblem(ElementwiseProblem):
                def __init__(self):
                    super().__init__(n_var=1, n_obj=obj_inst.n_obj, n_constr=0)
                def _evaluate(self, x, out, *a, **k):
                    out["F"] = np.array(obj_inst.evaluate(x[0]), dtype=float)

            class MolSampling(Sampling):
                def _do(self, problem, n_samples, **kwargs):
                    X = np.empty((n_samples, 1), dtype=object)
                    for i in range(n_samples):
                        s = parent_smi
                        for _ in range(random.randint(1, 3)):
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
                            child = EG.crossover(a, b, op="hybrid", target_mw=parent_mw)
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

            algo = NSGA2(
                pop_size=pop_size,
                sampling=MolSampling(),
                crossover=MolCrossover(),
                mutation=MolMutation(),
                eliminate_duplicates=SmilesDuplicate(),
            )

            t0 = time.perf_counter()
            res = minimize(MolProblem(), algo, ("n_gen", generations), seed=seed, verbose=False)
            dt = time.perf_counter() - t0
            agg_metrics["runtimes"].append(dt)

            X = np.atleast_2d(res.X)
            F = np.atleast_2d(res.F)

            drug_candidates = []
            seen = set()

            for xi, fi in zip(X, F):
                smi = xi[0]
                if smi in seen or Chem.MolFromSmiles(smi) is None:
                    continue
                seen.add(smi)
                mol = Chem.MolFromSmiles(smi)
                sim = tanimoto(parent_fp, morgan_fp(mol))
                if sim >= 0.999:
                    continue

                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                delta_mw = abs(mw - parent_mw)
                delta_logp = abs(logp - parent_logp)
                qed = Descriptors.qed(mol)
                sa = sascorer.calculateScore(mol) if sascorer else 2.5
                alerts = get_structural_alerts(mol)
                
                scaf_preserved = False
                if has_scaf and mol.HasSubstructMatch(parent_scaf):
                    scaf_preserved = True

                edits = EG.edited_atoms(parent_smi, smi)
                
                # Model evaluation (independent audit for all formulations)
                p_cf = 0.5
                if detector:
                    try:
                        p_cf = detector(smi)
                    except Exception:
                        p_cf = 0.5

                rec = {
                    "drug": d_name,
                    "smiles": smi,
                    "similarity": round(sim, 3),
                    "qed": round(float(qed), 3),
                    "sa_score": round(float(sa), 2),
                    "alerts": alerts,
                    "delta_mw": round(float(delta_mw), 1),
                    "delta_logp": round(float(delta_logp), 2),
                    "scaffold_preserved": scaf_preserved,
                    "edit_size": len(edits),
                    "p_counterfeit": round(float(p_cf), 3),
                    "fitness_vector": [round(float(v), 4) for v in fi],
                    "sum_fitness": round(float(np.sum(fi)), 4),
                }
                drug_candidates.append(rec)

            agg_metrics["pareto_sizes"].append(len(drug_candidates))
            if drug_candidates:
                sims = [d["similarity"] for d in drug_candidates]
                agg_metrics["similarities"].extend(sims)
                agg_metrics["in_band_pcts"].append(sum(1 for s in sims if 0.4 <= s <= 0.9) / len(sims) * 100.0)
                agg_metrics["sweet_spot_pcts"].append(sum(1 for s in sims if 0.55 <= s <= 0.80) / len(sims) * 100.0)
                agg_metrics["qeds"].extend([d["qed"] for d in drug_candidates])
                agg_metrics["sas"].extend([d["sa_score"] for d in drug_candidates])
                agg_metrics["alerts"].extend([d["alerts"] for d in drug_candidates])
                agg_metrics["delta_mws"].extend([d["delta_mw"] for d in drug_candidates])
                agg_metrics["delta_logps"].extend([d["delta_logp"] for d in drug_candidates])
                agg_metrics["scaffold_preserved_pcts"].append(sum(1 for d in drug_candidates if d["scaffold_preserved"]) / len(drug_candidates) * 100.0)
                agg_metrics["edit_sizes"].extend([d["edit_size"] for d in drug_candidates])
                agg_metrics["p_counterfeits"].extend([d["p_counterfeit"] for d in drug_candidates])
                agg_metrics["fooling_pcts"].append(sum(1 for d in drug_candidates if d["p_counterfeit"] < 0.5) / len(drug_candidates) * 100.0)
                
                drug_candidates.sort(key=lambda d: d["sum_fitness"])
                agg_metrics["top_candidates"].append(drug_candidates[0])

            print(f"  ✓ {d_name:<12} -> Pareto size: {len(drug_candidates):<3} | Time: {dt:.2f}s | Mean Sim: {np.mean([d['similarity'] for d in drug_candidates]):.3f}")

        # Compute formulation summary
        summary = {
            "name": obj_name,
            "short_name": short_name,
            "mean_runtime_s": round(float(np.mean(agg_metrics["runtimes"])), 2),
            "mean_pareto_size": round(float(np.mean(agg_metrics["pareto_sizes"])), 1),
            "mean_tanimoto": round(float(np.mean(agg_metrics["similarities"])), 3),
            "in_band_pct": round(float(np.mean(agg_metrics["in_band_pcts"])), 1),
            "sweet_spot_pct": round(float(np.mean(agg_metrics["sweet_spot_pcts"])), 1),
            "mean_qed": round(float(np.mean(agg_metrics["qeds"])), 3),
            "mean_sa": round(float(np.mean(agg_metrics["sas"])), 2),
            "mean_alerts": round(float(np.mean(agg_metrics["alerts"])), 3),
            "mean_delta_mw": round(float(np.mean(agg_metrics["delta_mws"])), 1),
            "mean_delta_logp": round(float(np.mean(agg_metrics["delta_logps"])), 2),
            "scaffold_preservation_pct": round(float(np.mean(agg_metrics["scaffold_preserved_pcts"])), 1),
            "mean_edit_size": round(float(np.mean(agg_metrics["edit_sizes"])), 1),
            "mean_p_counterfeit": round(float(np.mean(agg_metrics["p_counterfeits"])), 3),
            "fooling_rate_pct": round(float(np.mean(agg_metrics["fooling_pcts"])), 1),
            "top_candidates": agg_metrics["top_candidates"]
        }
        results[short_name] = summary

    return results


# =====================================================================
#  PLOTTING
# =====================================================================

def generate_comparison_plots(results: Dict[str, Any], output_path: str = "objective_comparison_metrics.png"):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Comparative Evaluation of Multi-Objective Optimization Formulations", fontsize=16, fontweight='bold', y=0.98)

    names = list(results.keys())
    colors = ['#4e79a7', '#59a14f', '#e15759']

    # 1. Similarity Sweet-Spot [0.55, 0.80] & General In-Band [0.4, 0.9]
    sweet = [results[n]["sweet_spot_pct"] for n in names]
    in_band = [results[n]["in_band_pct"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    axs[0, 0].bar(x - width/2, in_band, width, label='In Band [0.4, 0.9] (%)', color='#76b7b2')
    axs[0, 0].bar(x + width/2, sweet, width, label='Sweet-Spot [0.55, 0.80] (%)', color='#59a14f')
    axs[0, 0].set_title('Tanimoto Similarity Band Compliance (%)', fontweight='bold')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(names, fontsize=10, fontweight='bold')
    axs[0, 0].set_ylim(0, 115)
    axs[0, 0].legend(loc='lower right')
    for i in range(len(names)):
        axs[0, 0].text(i + width/2, sweet[i] + 2, f"{sweet[i]}%", ha='center', fontweight='bold', fontsize=9)

    # 2. Anti-Shortcut Property Drift (ΔMW & ΔLogP)
    mws = [results[n]["mean_delta_mw"] for n in names]
    axs[0, 1].bar(names, mws, color=colors, width=0.55)
    axs[0, 1].set_title('Property Drift: Mean |Δ MW| (Da) — Lower is Better', fontweight='bold')
    axs[0, 1].set_xticklabels(names, fontsize=10, fontweight='bold')
    for i, v in enumerate(mws):
        axs[0, 1].text(i, v + 1.5, f"{v} Da", ha='center', fontweight='bold', fontsize=10)

    # 3. Scaffold Preservation & Edit Sparsity
    scaf = [results[n]["scaffold_preservation_pct"] for n in names]
    edits = [results[n]["mean_edit_size"] for n in names]
    ax3 = axs[0, 2]
    ax3_twin = ax3.twinx()
    b1 = ax3.bar(x - width/2, scaf, width, label='Scaffold Preserved (%)', color='#e15759')
    b2 = ax3_twin.bar(x + width/2, edits, width, label='Edit Size (# Atoms)', color='#f28e2b')
    ax3.set_title('Scaffold Preservation (%) & Edit Size (# Atoms)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax3.set_ylabel('Scaffold Preserved (%)', color='#e15759', fontweight='bold')
    ax3_twin.set_ylabel('Mean Edit Size (# Atoms)', color='#f28e2b', fontweight='bold')
    ax3.set_ylim(0, 115)
    for i, v in enumerate(scaf):
        ax3.text(i - width/2, v + 2, f"{v}%", ha='center', fontweight='bold', fontsize=9)

    # 4. Drug Quality: QED vs SA Score
    qeds = [results[n]["mean_qed"] for n in names]
    sas = [results[n]["mean_sa"] for n in names]
    for i, n in enumerate(names):
        axs[1, 0].scatter(sas[i], qeds[i], s=300, color=colors[i], edgecolors='black', linewidth=1.5, label=n)
        axs[1, 0].annotate(f"{n}\n(QED={qeds[i]:.2f}, SA={sas[i]:.2f})",
                           (sas[i], qeds[i]), textcoords="offset points", xytext=(0, 12),
                           ha='center', fontsize=9, fontweight='bold')
    axs[1, 0].set_title('Drug-Likeness (QED) vs Synthesizability (SA)', fontweight='bold')
    axs[1, 0].set_xlabel('Synthetic Accessibility (Lower = Easier to Synthesize)', fontweight='bold')
    axs[1, 0].set_ylabel('QED Score (Higher = More Druglike)', fontweight='bold')
    axs[1, 0].set_xlim(min(sas) - 0.2, max(sas) + 0.3)
    axs[1, 0].set_ylim(min(qeds) - 0.05, max(qeds) + 0.08)

    # 5. GNN Detector Deception / Hardness (% Fooling Detector)
    fooling = [results[n]["fooling_rate_pct"] for n in names]
    p_cf = [results[n]["mean_p_counterfeit"] for n in names]
    axs[1, 1].bar(names, fooling, color=['#4e79a7', '#59a14f', '#9c755f'], width=0.55)
    axs[1, 1].set_title('Detector Evasion Rate (% P(cf) < 0.5) — Hardness', fontweight='bold')
    axs[1, 1].set_xticklabels(names, fontsize=10, fontweight='bold')
    axs[1, 1].set_ylim(0, 105)
    for i, v in enumerate(fooling):
        axs[1, 1].text(i, v + 2, f"{v}%\n(avg P={p_cf[i]:.2f})", ha='center', fontweight='bold', fontsize=9)

    # 6. Runtime & Pareto Front Size
    sizes = [results[n]["mean_pareto_size"] for n in names]
    runtimes = [results[n]["mean_runtime_s"] for n in names]
    axs[1, 2].bar(x - width/2, sizes, width, label='Pareto Front Size', color='#b07aa1')
    axs[1, 2].bar(x + width/2, runtimes, width, label='Runtime (s / drug)', color='#bab0ac')
    axs[1, 2].set_title('Pareto Yield & Computational Throughput', fontweight='bold')
    axs[1, 2].set_xticks(x)
    axs[1, 2].set_xticklabels(names, fontsize=10, fontweight='bold')
    axs[1, 2].legend(loc='upper right')
    for i, v in enumerate(sizes):
        axs[1, 2].text(i - width/2, v + 0.8, f"{v}", ha='center', fontweight='bold', fontsize=9)
    for i, v in enumerate(runtimes):
        axs[1, 2].text(i + width/2, v + 0.8, f"{v}s", ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Benchmark plots successfully saved to: {output_path}")


# =====================================================================
#  MAIN ENTRY
# =====================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    results = run_objective_benchmark(generations=15, pop_size=36, seed=42)

    out_json = "objective_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {out_json}")

    plot_path = "objective_comparison_metrics.png"
    generate_comparison_plots(results, output_path=plot_path)

    # Copy to brain artifact directory if it exists
    artifact_dir = "/Users/ediionut23/.gemini/antigravity-ide/brain/3f90209c-0d41-4677-a0c8-e463d6d2d3a1"
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(plot_path, os.path.join(artifact_dir, "objective_comparison_metrics.png"))
        shutil.copy(out_json, os.path.join(artifact_dir, "objective_benchmark_results.json"))
        print(f"✅ Copied artifacts to: {artifact_dir}")

    # Print markdown-style table
    print("\n" + "="*105)
    print("FINAL SUMMARY: COMPARATIVE EVALUATION OF 3 OBJECTIVE FUNCTION PARADIGMS")
    print("="*105)
    headers = ["Formulation", "Tanimoto", "SweetSpot%", "QED", "SA", "|ΔMW|", "Scaffold%", "#Edits", "FoolGNN%", "ParetoSz", "Time(s)"]
    print(f"{headers[0]:<24} {headers[1]:<9} {headers[2]:<11} {headers[3]:<6} {headers[4]:<5} {headers[5]:<8} {headers[6]:<10} {headers[7]:<7} {headers[8]:<9} {headers[9]:<9} {headers[10]:<7}")
    print("-" * 115)
    for k, v in results.items():
        print(f"{k:<24} {v['mean_tanimoto']:<9.3f} {v['sweet_spot_pct']:<11.1f} {v['mean_qed']:<6.2f} {v['mean_sa']:<5.2f} {v['mean_delta_mw']:<8.1f} {v['scaffold_preservation_pct']:<10.1f} {v['mean_edit_size']:<7.1f} {v['fooling_rate_pct']:<9.1f} {v['mean_pareto_size']:<9.1f} {v['mean_runtime_s']:<7.2f}")
    print("="*105)


if __name__ == "__main__":
    main()
