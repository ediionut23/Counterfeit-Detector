"""
Category-Conditioned Multi-Objective Evolutionary Generator (Parallel High-Throughput)
=======================================================================================
Evolves named reference molecules or legacy public-library molecules using NSGA-II.
Mutation operators are restricted to a requested category; crossover effects still
require separate verification of the final chemical transformation.

Outputs are synthetic analogue candidates with computational chemical filters,
not confirmed falsified medicines or experimentally demonstrated syntheses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import multiprocessing
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import evolutionary_generator as EG

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evolve_category_pilot")

# Worker state
_WORKER_RULE_LIB = None
_WORKER_CATEGORY = None
_RULE_LIBRARY_CACHE = {}


def _init_worker(rules_path: str, category: str, applicable_rule_sampling: bool = False):
    global _WORKER_RULE_LIB, _WORKER_CATEGORY
    RDLogger.DisableLog("rdApp.*")
    _WORKER_CATEGORY = category
    path = Path(rules_path).resolve()
    stat = path.stat()
    key = (str(path), category, stat.st_mtime_ns, stat.st_size)
    if key not in _RULE_LIBRARY_CACHE:
        _RULE_LIBRARY_CACHE[key] = EG.RuleLibrary(str(path), category=category)
    _WORKER_RULE_LIB = _RULE_LIBRARY_CACHE[key]
    _WORKER_RULE_LIB.applicable_rule_sampling = applicable_rule_sampling


def _worker_process_parent(args_tuple) -> List[Dict]:
    global _WORKER_RULE_LIB, _WORKER_CATEGORY
    (parent_record, sim_lo, sim_hi, pop, generations, crossover_op,
     qed_min, max_sa, per_parent_keep, seed, objective_profile) = args_tuple[:11]
    settings = args_tuple[11] if len(args_tuple) > 11 else {}
    reference = parent_record if isinstance(parent_record, dict) else {"smiles": parent_record}
    parent = reference["smiles"]

    try:
        objectives = EG.Objectives(parent, sim_lo, sim_hi, adversarial=None, profile=objective_profile,
            similarity_target=settings.get("similarity_target", 0.70),
            scaffold_penalty=settings.get("scaffold_penalty", 0.35),
            fitness_weights=settings.get("fitness_weights"))
        front = EG.build_and_run(
            parent, _WORKER_RULE_LIB, objectives,
            pop_size=pop, generations=generations, seed=seed,
            crossover_op=crossover_op, compute_shape=False,
            archive_history=True,
            record_operations=True,
        )
    except Exception:
        if settings.get("raise_errors", False):
            raise
        return []

    picked = []
    seen = set()
    for d in front:
        smi = d["smiles"]
        if smi in seen or smi == parent:
            continue
        seen.add(smi)
        sim = d["similarity_to_parent"]
        if not (sim_lo <= sim <= sim_hi):
            continue
        if d["qed"] < qed_min:
            continue
        if d.get("sa_score") is not None and d["sa_score"] > max_sa:
            continue
        if d.get("structural_alerts", 0) > 0:
            continue
        picked.append(d)

    # Rank by aggregated Pareto objectives
    picked.sort(key=lambda d: sum(d["objectives"].values()))

    out_rows = []
    chosen_fps = []
    for d in picked:
        m = Chem.MolFromSmiles(d["smiles"])
        if m is None:
            continue
        exact_sim = EG.tanimoto(objectives.parent_fp, EG.morgan(m))
        exact_qed = float(Descriptors.qed(m))
        exact_sa = float(EG.sascorer.calculateScore(m)) if EG.sascorer else None
        if not (sim_lo <= exact_sim <= sim_hi) or exact_qed < qed_min:
            continue
        if exact_sa is not None and exact_sa > max_sa:
            continue
        if len(Chem.GetMolFrags(m)) != 1 or not (
                settings.get("child_heavy_min", 10) <= m.GetNumHeavyAtoms() <= settings.get("child_heavy_max", 70)):
            continue
        mw = float(Descriptors.MolWt(m))
        if not (settings.get("child_mw_min", 150.0) <= mw <= settings.get("child_mw_max", 800.0)):
            continue
        fp = EG.morgan(m)
        if any(EG.tanimoto(fp, previous) > settings.get("max_pairwise_similarity", 0.90) for previous in chosen_fps):
            continue
        chosen_fps.append(fp)
        p = Chem.MolFromSmiles(parent)
        out_rows.append({
            "counterfeit_smiles": d["smiles"],
            "parent_smiles": parent,
            "category": f"evolved_{_WORKER_CATEGORY}",
            "subcategory": f"evolved_{_WORKER_CATEGORY}",
            "generation_protocol": "category_conditioned_genetic_search",
            "generation_method": "E",
            "scenario": settings.get("scenario", "synthetic_near_analogue_substitution"),
            "difficulty": settings.get("difficulty", "legacy_similarity_band"),
            "reference_id": reference.get("reference_id", ""),
            "reference_name": reference.get("name", ""),
            "reference_source": reference.get("source", ""),
            "reference_source_id": reference.get("source_id", ""),
            "reference_status": reference.get("status", "unclassified_public_reference"),
            "reference_aliases_json": reference.get("reference_aliases_json", "[]"),
            "mutation_rule_category": _WORKER_CATEGORY,
            "verified_chemical_category": "",
            "chemical_category_verification": "operator_conditioned_not_verified_on_final_pair",
            "evidence_tier": "synthetic_generator_relation",
            "confirmed_real_case": 0,
            "operator_history_json": json.dumps(d.get("operator_history", []), sort_keys=True),
            "operator_history_status": d.get("operator_history_status", "not_recorded"),
            "search_operation_event_count": d.get("search_operation_event_count", 0),
            "parent_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=p) if p else "",
            "child_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=m),
            "objective_score": round(float(sum(d["objectives"].values())), 6),
            "similarity_to_parent": round(exact_sim, 8),
            "qed": round(exact_qed, 8),
            "sa_score": round(exact_sa, 6) if exact_sa is not None else None,
            "n_edited_atoms": "",
            "edited_atoms": "",
            "edit_annotation_status": "not_computed",
            "mw": round(mw, 2),
            "logp": round(float(Descriptors.MolLogP(m)), 2),
            "heavy_atoms": m.GetNumHeavyAtoms(),
            "smiles_length": len(d["smiles"]),
        })
        if len(out_rows) >= per_parent_keep:
            break
    return out_rows


def generate_for_reference_e(reference, category, seed, max_candidates, *,
                             sim_lo, sim_hi, similarity_target=0.70,
                             scaffold_penalty=0.35, pop=32, generations=8,
                             qed_min=0.35, max_sa=4.2,
                             rules_path="mined_transformations.py",
                             difficulty="custom", scenario="synthetic_structural_analogue",
                             fitness_weights=None, crossover_op="scaffold",
                             child_mw_min=150.0, child_mw_max=800.0,
                             child_heavy_min=10, child_heavy_max=70,
                             max_pairwise_similarity=0.90,
                             applicable_rule_sampling=True):
    """Generate one named-reference/cell job with process-local cached rules.

    Annotation is explicitly deferred. The returned mutation category is the
    requested search category; final chemical verification remains separate.
    Exceptions are propagated for scheduler accounting rather than converted
    into silently empty batches. Use one call at a time in each worker process.
    """
    if category not in {"bioisostere", "halogen-walk", "scaffold-hop", "homologation", "other"}:
        raise ValueError(f"Unknown generation category: {category}")
    if min(pop, generations, max_candidates) < 1:
        raise ValueError("Population, generations and candidate limit must be positive")
    if not isinstance(reference, dict) or not reference.get("smiles"):
        raise ValueError("reference must include a nonempty smiles field")
    if not 0 <= qed_min <= 1 or not math.isfinite(max_sa) or max_sa < 0:
        raise ValueError("QED must be in [0,1]; maximum SA finite/nonnegative")
    if not (math.isfinite(child_mw_min) and math.isfinite(child_mw_max) and 0 <= child_mw_min <= child_mw_max):
        raise ValueError("Child MW bounds must be finite and satisfy 0 <= minimum <= maximum")
    if not 1 <= child_heavy_min <= child_heavy_max:
        raise ValueError("Child heavy-atom bounds must satisfy 1 <= minimum <= maximum")
    if not math.isfinite(max_pairwise_similarity) or not 0 <= max_pairwise_similarity <= 1:
        raise ValueError("Pairwise child similarity must be finite and in [0,1]")
    if not scenario.strip() or not difficulty.strip():
        raise ValueError("Scenario and difficulty must be nonempty")
    if crossover_op not in {"scaffold", "mass-balanced", "hybrid", "brics", "jensen"}:
        raise ValueError(f"Unknown crossover operator: {crossover_op}")
    # Validate before compatibility short-circuiting so an infeasible cell
    # cannot hide a malformed configuration from the scheduler.
    EG.Objectives(reference["smiles"], sim_lo, sim_hi, profile="anti-shortcut",
        similarity_target=similarity_target, scaffold_penalty=scaffold_penalty,
        fitness_weights=fitness_weights)
    _init_worker(rules_path, category, applicable_rule_sampling)
    if applicable_rule_sampling and not _WORKER_RULE_LIB.applicable_indices(reference["smiles"]):
        return []
    settings = {"similarity_target": similarity_target, "scaffold_penalty": scaffold_penalty,
        "fitness_weights": fitness_weights, "scenario": scenario, "difficulty": difficulty,
        "child_mw_min": child_mw_min, "child_mw_max": child_mw_max,
        "child_heavy_min": child_heavy_min, "child_heavy_max": child_heavy_max,
        "max_pairwise_similarity": max_pairwise_similarity, "raise_errors": True}
    return _worker_process_parent((reference, sim_lo, sim_hi, pop, generations, crossover_op,
        qed_min, max_sa, max_candidates, seed, "anti-shortcut", settings))


def _compute_edit_mask(payload: Tuple[str, str]) -> List[int]:
    parent, child = payload
    return EG.edited_atoms(parent, child)


def load_parents(path: str, max_parents: int, seed: int = 42) -> List[str]:
    """Backward-compatible SMILES-only view of the named-reference loader."""
    return [row["smiles"] for row in load_parent_records(path, max_parents, seed)]


def load_parent_records(path: str, max_parents: int, seed: int = 42, *,
                        mw_min: float = 180.0, mw_max: float = 600.0,
                        heavy_min: int = 12, heavy_max: int = 45) -> List[Dict]:
    """Load CSV identity metadata or legacy SMI, preserving duplicate aliases."""
    if Path(path).suffix.lower() == ".csv":
        with open(path, newline="") as stream:
            reader = csv.DictReader(stream)
            if not ({"smiles", "normalized_smiles"} & set(reader.fieldnames or [])):
                raise ValueError("Parent CSV requires smiles or normalized_smiles")
            records = list(reader)
    else:
        records = []
        with open(path) as stream:
            for line in stream:
                tokens = line.strip().split()
                if not tokens or tokens[0].startswith("#"):
                    continue
                records.append({"smiles": tokens[0], "reference_id": "",
                    "name": " ".join(tokens[1:]), "source": "legacy_smi",
                    "status": "unclassified_public_reference"})
    seen: Dict[str, Dict] = {}
    aliases: Dict[str, List[Dict]] = {}
    for row in records:
        if str(row.get("generation_eligible", "1")).strip().lower() not in {"1", "1.0", "true", "yes"}:
            continue
        mol = Chem.MolFromSmiles(row.get("smiles") or row.get("normalized_smiles") or "")
        if mol is None:
            continue
        if not (mw_min <= Descriptors.MolWt(mol) <= mw_max and heavy_min <= mol.GetNumHeavyAtoms() <= heavy_max):
            continue
        canonical = Chem.MolToSmiles(mol)
        record = {"smiles": canonical, "reference_id": row.get("reference_id", ""),
            "name": row.get("name", row.get("pref_name", "")), "source": row.get("source", ""),
            "source_id": row.get("source_id", ""),
            "status": row.get("status", "unclassified_public_reference")}
        aliases.setdefault(canonical, []).append({k: record[k] for k in record if k != "smiles"})
        prior = seen.get(canonical)
        if prior is None or tuple(record[k] for k in ("reference_id", "source_id", "name")) < tuple(prior[k] for k in ("reference_id", "source_id", "name")):
            seen[canonical] = record
    out = list(seen.values())
    for record in out:
        record["reference_aliases_json"] = json.dumps(sorted(aliases[record["smiles"]], key=lambda r: (r["reference_id"], r["source_id"], r["name"])), sort_keys=True)
    random.Random(seed).shuffle(out)
    return out[:max_parents]


def load_canonical_smiles(path: str) -> set:
    """Load a full canonical reference set for generated-child exclusion."""
    seen = set()
    if Path(path).suffix.lower() == ".csv":
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                mol = Chem.MolFromSmiles(row.get("smiles") or row.get("normalized_smiles") or "")
                if mol is not None:
                    seen.add(Chem.MolToSmiles(mol))
        return seen
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mol = Chem.MolFromSmiles(line.split()[0])
            if mol is not None:
                seen.add(Chem.MolToSmiles(mol))
    return seen


def load_canonical_csv(path: str) -> set:
    seen = set()
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            value = row.get("counterfeit_smiles") or row.get("smiles") or ""
            mol = Chem.MolFromSmiles(value)
            if mol is not None:
                seen.add(Chem.MolToSmiles(mol))
    return seen


def candidate_provenance_rank(row: Dict) -> Tuple:
    """Choose the same parent relation regardless of worker completion order."""
    return (row["objective_score"], row.get("parent_smiles", ""),
            row.get("category", ""), row.get("subcategory", ""),
            row["counterfeit_smiles"])


def select_scaffold_balanced_candidates(
    rows: List[Dict], target: int, *,
    max_child_scaffold: Optional[int] = None,
    max_parent_scaffold: Optional[int] = None,
) -> List[Dict]:
    """Take one feasible candidate per parent scaffold on each round.

    Candidates blocked by a full child-scaffold quota cannot become feasible
    later. Drain those heads before deciding whether a scaffold has anything
    left to contribute; stopping after one blocked head can discard a useful
    tail. Input rows are read without mutation.
    """
    if target <= 0:
        return []
    child_cap = max(200, target // 50) if max_child_scaffold is None else max_child_scaffold
    parent_cap = max(50, target // 200) if max_parent_scaffold is None else max_parent_scaffold
    if child_cap < 1 or parent_cap < 1:
        raise ValueError("Scaffold caps must be positive")
    by_parent_scaffold: Dict[str, List[Dict]] = {}
    for row in rows:
        by_parent_scaffold.setdefault(row["parent_scaffold"], []).append(row)
    queues = {
        scaffold: deque(sorted(candidates, key=lambda r: (
            r["objective_score"], r["counterfeit_smiles"],
            r.get("parent_smiles", ""), r.get("category", ""), r.get("subcategory", ""))))
        for scaffold, candidates in by_parent_scaffold.items()
    }
    scaffold_order = sorted(queues)
    selected: List[Dict] = []
    child_counts: Dict[str, int] = {}
    parent_counts: Dict[str, int] = {}
    while len(selected) < target:
        added = False
        for scaffold in scaffold_order:
            candidates = queues[scaffold]
            if parent_counts.get(scaffold, 0) >= parent_cap:
                continue
            while candidates:
                row = candidates.popleft()
                child = row["child_scaffold"]
                if child_counts.get(child, 0) >= child_cap:
                    continue
                selected.append(row)
                child_counts[child] = child_counts.get(child, 0) + 1
                parent_counts[scaffold] = parent_counts.get(scaffold, 0) + 1
                added = True
                break
            if len(selected) >= target:
                break
        if not added:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description="Category-Conditioned Multi-Objective Evolutionary Generator")
    parser.add_argument("--category", required=True, choices=["bioisostere", "halogen-walk", "scaffold-hop", "homologation", "other"],
                        help="Chemical category to constrain mutations to")
    parser.add_argument("--rules", default="mined_transformations.py", help="Path to rules script")
    parser.add_argument("--parents", default="public_molecules.smi", help="Path to parent SMILES file")
    parser.add_argument("--max-parents", type=int, default=3000, help="Number of parent drugs to evolve")
    parser.add_argument("--pop", type=int, default=20, help="Population size in NSGA-II")
    parser.add_argument("--generations", type=int, default=8, help="Generations per parent evolution")
    parser.add_argument("--sim-lo", type=float, default=0.55, help="Lower Tanimoto similarity bound")
    parser.add_argument("--sim-hi", type=float, default=0.87, help="Upper Tanimoto similarity bound")
    parser.add_argument("--similarity-target", type=float, default=0.70,
                        help="Preferred Tanimoto in anti-shortcut fitness; independent of acceptance band")
    parser.add_argument("--scaffold-penalty", type=float, default=0.35,
                        help="Anti-shortcut scaffold-loss penalty in [0,1]; use 0 for unconstrained scaffold-hop")
    parser.add_argument("--fitness-weights", type=json.loads, default=None,
                        help='JSON anti-shortcut weights: proximity,mw,logp,qed,sa,alerts,sparsity')
    parser.add_argument("--scenario", default="synthetic_near_analogue_substitution")
    parser.add_argument("--difficulty", default="legacy_similarity_band")
    parser.add_argument("--parent-mw-min", type=float, default=180.0)
    parser.add_argument("--parent-mw-max", type=float, default=600.0)
    parser.add_argument("--parent-heavy-min", type=int, default=12)
    parser.add_argument("--parent-heavy-max", type=int, default=45)
    parser.add_argument("--child-mw-min", type=float, default=150.0)
    parser.add_argument("--child-mw-max", type=float, default=800.0)
    parser.add_argument("--child-heavy-min", type=int, default=10)
    parser.add_argument("--child-heavy-max", type=int, default=70)
    parser.add_argument("--max-pairwise-similarity", type=float, default=0.90,
                        help="Maximum fingerprint similarity among one parent's chosen children")
    parser.add_argument("--defer-edit-annotations", action="store_true",
                        help="Store explicit deferred status instead of computing MCS masks for every output")
    parser.add_argument("--applicable-rule-sampling", action="store_true",
                        help="Renormalize mutation-rule weights among exact matching reactant patterns; cache compatibility")
    parser.add_argument("--qed-min", type=float, default=0.45, help="Minimum QED drug-likeness")
    parser.add_argument("--max-sa", type=float, default=4.20, help="Maximum Ertl SA score")
    parser.add_argument("--per-parent-keep", type=int, default=6, help="Maximum counterfeits to keep per parent")
    parser.add_argument("--target", type=int, default=10000,
                        help="Target unique evolved analogues for this category")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="CSV files whose counterfeit SMILES should be excluded")
    parser.add_argument("--authentic-exclude", nargs="*",
                        default=["public_molecules_100k.smi", "public_molecules.smi"],
                        help="Authentic SMILES libraries used to exclude known molecules")
    parser.add_argument("--existing-bank", type=Path, default=Path("final_dataset/molecules.csv"),
                        help="Optional prior counterfeit molecule bank to exclude exact matches")
    parser.add_argument("--crossover", default="scaffold", choices=["scaffold", "mass-balanced", "hybrid"],
                        help="Crossover operator")
    parser.add_argument("--objective-profile", default="anti-shortcut", help="Objectives profile")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: cpu_count)")
    parser.add_argument("--out", default="version_ga", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if args.fitness_weights is not None and not isinstance(args.fitness_weights, dict):
        parser.error("--fitness-weights must be a JSON object")
    try:
        EG.Objectives("c1ccccc1", args.sim_lo, args.sim_hi, profile=args.objective_profile,
                      similarity_target=args.similarity_target, scaffold_penalty=args.scaffold_penalty,
                      fitness_weights=args.fitness_weights)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if any(not math.isfinite(value) or value < 0 for value in (
            args.parent_mw_min, args.parent_mw_max, args.child_mw_min, args.child_mw_max)):
        parser.error("Molecular-weight bounds must be finite and nonnegative")
    if args.parent_mw_min > args.parent_mw_max or args.child_mw_min > args.child_mw_max:
        parser.error("Minimum molecular-weight bounds must not exceed maxima")
    if not (1 <= args.parent_heavy_min <= args.parent_heavy_max and 1 <= args.child_heavy_min <= args.child_heavy_max):
        parser.error("Heavy-atom bounds must satisfy 1 <= minimum <= maximum")
    if not 0 <= args.qed_min <= 1 or not math.isfinite(args.max_sa) or args.max_sa < 0:
        parser.error("QED must be in [0,1] and maximum SA finite/nonnegative")
    if not math.isfinite(args.max_pairwise_similarity) or not 0 <= args.max_pairwise_similarity <= 1:
        parser.error("--max-pairwise-similarity must be finite and in [0,1]")
    if min(args.pop, args.generations, args.max_parents, args.target, args.per_parent_keep) < 1:
        parser.error("Population, generations, parents, target and per-parent keep must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if not args.scenario.strip() or not args.difficulty.strip():
        parser.error("--scenario and --difficulty must be nonempty")

    workers = args.workers or max(1, os.cpu_count() or 4)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Category-Conditioned Evolution: '{args.category.upper()}' ===")
    parents = load_parent_records(args.parents, args.max_parents, seed=args.seed,
        mw_min=args.parent_mw_min, mw_max=args.parent_mw_max,
        heavy_min=args.parent_heavy_min, heavy_max=args.parent_heavy_max)
    logger.info(f"Loaded {len(parents)} authentic parents from {args.parents}")
    excluded = set()
    for reference_path in set([args.parents, *args.authentic_exclude]):
        if Path(reference_path).exists():
            excluded.update(load_canonical_smiles(reference_path))
    for exclude_path in args.exclude:
        excluded.update(load_canonical_csv(exclude_path))
    n_before_existing_bank = len(excluded)
    if args.existing_bank.exists():
        excluded.update(load_canonical_csv(str(args.existing_bank)))
    logger.info(f"Excluding {len(excluded):,} authentic/previous-category structures")

    settings = {name: getattr(args, name) for name in (
        "similarity_target", "scaffold_penalty", "fitness_weights", "scenario", "difficulty",
        "child_mw_min", "child_mw_max", "child_heavy_min", "child_heavy_max", "max_pairwise_similarity")}
    tasks = [
        (parent, args.sim_lo, args.sim_hi, args.pop, args.generations,
         args.crossover, args.qed_min, args.max_sa, args.per_parent_keep,
         args.seed + idx, args.objective_profile, settings)
        for idx, parent in enumerate(parents)
    ]

    t0 = time.time()
    all_results: List[Dict] = []
    seen_fakes: Dict[str, Dict] = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(args.rules, args.category, args.applicable_rule_sampling)
    ) as pool:
        futures = {pool.submit(_worker_process_parent, t): t[0] for t in tasks}
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            completed += 1
            for row in res:
                smi = row["counterfeit_smiles"]
                if smi in excluded:
                    continue
                old = seen_fakes.get(smi)
                if old is None or candidate_provenance_rank(row) < candidate_provenance_rank(old):
                    seen_fakes[smi] = row
            if completed % 25 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.1)
                logger.info(f"Progress: {completed}/{len(tasks)} parents [{completed/len(tasks)*100:.1f}%] "
                            f"({rate:.1f} parents/s) -> {len(seen_fakes)} valid unique counterfeits")

    all_results = list(seen_fakes.values())

    # Select a quality-ranked, scaffold-balanced archive. Each round takes the
    # best remaining candidate from each parent scaffold before taking a
    # second candidate from any one scaffold.
    all_results = select_scaffold_balanced_candidates(all_results, args.target)

    # MCS is relatively expensive and only needed for delivered rows, not for
    # per-parent candidate filtering. Compute masks after category-level
    # selection and parallelize this final annotation pass.
    if all_results and not args.defer_edit_annotations:
        payloads = [(r["parent_smiles"], r["counterfeit_smiles"]) for r in all_results]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row, edits in zip(all_results, pool.map(_compute_edit_mask, payloads, chunksize=16)):
                row["n_edited_atoms"] = len(edits)
                row["edited_atoms"] = " ".join(map(str, edits))
                row["edit_annotation_status"] = "mcs_approximation_not_rule_verified"
    elif args.defer_edit_annotations:
        for row in all_results:
            row["edit_annotation_status"] = "deferred"

    out_csv = out_dir / f"counterfeits_evolved_{args.category}.csv"
    if all_results:
        cols = list(all_results[0].keys())
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_results)

    summary = {
        "schema_version": 2,
        "category": args.category,
        "generation_protocol": "category_conditioned_genetic_search",
        "n_unique_counterfeits": len(all_results),
        "target": args.target,
        "target_reached": len(all_results) >= args.target,
        "n_parents_loaded": len(parents),
        "n_authentic_and_cross_category_exclusions": len(excluded),
        "n_existing_bank_exclusions": len(excluded) - n_before_existing_bank,
        "existing_bank_file": str(args.existing_bank),
        "population_size": args.pop,
        "generations": args.generations,
        "similarity_band": [args.sim_lo, args.sim_hi],
        "fitness_configuration": {"profile": args.objective_profile, **settings},
        "parent_bounds": {"mw": [args.parent_mw_min, args.parent_mw_max],
                          "heavy_atoms": [args.parent_heavy_min, args.parent_heavy_max]},
        "edit_annotation_status": "deferred" if args.defer_edit_annotations else "mcs_approximation_not_rule_verified",
        "reference_metadata_policy": "CSV identifiers/names/status retained; legacy SMI identifiers are unclassified public references",
        "operator_history_policy": "Recorded successful immediate producer events, not complete candidate ancestry",
        "mutation_sampling_policy": "applicable_patterns_reweighted" if args.applicable_rule_sampling else "legacy_weighted_random_tries",
        "chemical_category_policy": "Mutation search category is not a verified final chemical transformation category",
        "minimum_qed": args.qed_min,
        "maximum_sa_score": args.max_sa,
        "per_parent_keep": args.per_parent_keep,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "input_parents_file": args.parents,
        "rules_file": args.rules,
        "input_sha256": hashlib.sha256(Path(args.parents).read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(Path(args.rules).read_bytes()).hexdigest(),
        "authentic_exclusion_files": list(args.authentic_exclude),
        "output_csv": str(out_csv),
        "quality": {},
    }
    if all_results:
        summary["quality"] = {
            "mean_similarity": round(float(np.mean([r["similarity_to_parent"] for r in all_results])), 4),
            "median_similarity": round(float(np.median([r["similarity_to_parent"] for r in all_results])), 4),
            "mean_qed": round(float(np.mean([r["qed"] for r in all_results])), 4),
            "median_qed": round(float(np.median([r["qed"] for r in all_results])), 4),
            "mean_sa_score": round(float(np.mean([r["sa_score"] for r in all_results if r["sa_score"] is not None])), 4),
            "median_sa_score": round(float(np.median([r["sa_score"] for r in all_results if r["sa_score"] is not None])), 4),
            "n_parent_scaffolds": len({r["parent_scaffold"] for r in all_results}),
            "n_child_scaffolds": len({r["child_scaffold"] for r in all_results}),
        }
    (out_dir / f"summary_{args.category}.json").write_text(json.dumps(summary, indent=2) + "\n")

    dur = time.time() - t0
    logger.info(f"=== Completed '{args.category}': {len(all_results):,}/{args.target:,} target counterfeits from {len(parents)} parents "
                f"in {dur:.1f}s ({len(all_results)/max(dur, 0.1):.1f} fake/s) -> Saved to {out_csv} ===")

    # Print summary statistics
    if all_results:
        sims = [r["similarity_to_parent"] for r in all_results]
        qeds = [r["qed"] for r in all_results]
        sas = [r["sa_score"] for r in all_results if r["sa_score"] is not None]
        edits = [r["n_edited_atoms"] for r in all_results if r["n_edited_atoms"] != ""]
        print("\n" + "=" * 60)
        print(f"  SUMMARY FOR EVOLVED {args.category.upper()}")
        print("=" * 60)
        print(f"  Total Valid Counterfeits : {len(all_results):,}")
        print(f"  Mean Tanimoto to Parent  : {np.mean(sims):.3f} (Range: {np.min(sims):.3f} - {np.max(sims):.3f})")
        print(f"  Mean QED Drug-Likeness   : {np.mean(qeds):.3f} (Target >= 0.45)")
        print(f"  Mean Synthesizability SA : {np.mean(sas):.2f} (Target <= 4.20)")
        print(f"  Mean Atoms Edited (MCS)  : {np.mean(edits):.1f} atoms" if edits else "  Atom edit annotation    : deferred")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
