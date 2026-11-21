#!/usr/bin/env python3
"""Deterministic, reference-conditioned MMP proposals for the large bank.

``generate_for_reference`` executes only applicable mined reaction templates,
samples rules without replacement with tempered support weights, and records
the actual intermediate structures/rules for every returned product. Records
denote synthetic structural analogues, never experimentally confirmed product
falsification. The molecule has no intrinsic authentic/counterfeit label.

The default gates are computational size/validity bounds, not clinical claims.
Callers may specify similarity, QED, SA and structural-alert gates explicitly.
Use a standardized single-component active-moiety seed; this module does not
silently remove salts. Compilation/descriptors are cached within each worker.
Global normalized deduplication, persistent descriptor storage, holdouts and
reference/category budget allocation belong to the orchestrator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator


DEFAULT_RULES = Path(__file__).with_name("mined_transformations.py")
_FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
FINGERPRINT_DESCRIPTION = "Morgan radius=3, 2048 bits, chirality-independent"


@dataclass(frozen=True)
class Rule:
    name: str
    difficulty: str
    category: str
    subcategory: str
    citation: str
    support: float
    reaction: object
    query: object


@dataclass(frozen=True)
class Node:
    smiles: str
    history: tuple


@lru_cache(maxsize=4)
def _compiled_rules(path: str, mtime_ns: int, size: int):
    # mtime/size participate in the cache key so edited rule files are not
    # silently interpreted through a stale compiled library.
    spec = importlib.util.spec_from_file_location("_large_bank_mmp_rules", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import local rule library: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    support = getattr(module, "SUPPORT", {})
    subcategories = getattr(module, "SUBCATEGORY", {})
    by_category = defaultdict(list)
    for name, difficulty, category, reactant, product, citation in module.MMP_TRANSFORMATIONS:
        try:
            with rdBase.BlockLogs():
                reaction = AllChem.ReactionFromSmarts(f"{reactant}>>{product}")
                if reaction is None or reaction.GetNumReactantTemplates() != 1:
                    continue
                reaction.Initialize()
            rule = Rule(name, difficulty, category, subcategories.get(name, difficulty),
                        citation, max(1.0, float(support.get(name, 1))),
                        reaction, reaction.GetReactantTemplate(0))
            by_category[category].append(rule)
        except Exception:
            continue
    if not by_category:
        raise ValueError(f"No usable single-reactant rules in {path}")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {key: tuple(value) for key, value in by_category.items()}, digest


def _rule_library(path: str | Path):
    path = Path(path).resolve()
    stat = path.stat()
    return _compiled_rules(str(path), stat.st_mtime_ns, stat.st_size)


def rule_summary(rules_path: str | Path = DEFAULT_RULES) -> dict:
    categories, digest = _rule_library(rules_path)
    return {"rules_path": str(Path(rules_path).resolve()), "rules_sha256": digest,
            "categories": {name: len(rules) for name, rules in sorted(categories.items())},
            "subcategories": {name: dict(Counter(r.subcategory for r in rules))
                              for name, rules in sorted(categories.items())}}


def applicable_subcategories(reference: dict, category: str,
                             rules_path: str | Path = DEFAULT_RULES) -> list[str]:
    """Return source rule subcategories with a matching site; no reactions run.

    This is a scheduling feasibility screen, not proof that any sanitized
    product will pass downstream gates. Duplicate reactant queries are matched
    only once for this parent, and the result order is deterministic.
    """
    if str(reference.get("generation_eligible", "true")).strip().lower() in {"0", "false", "no"}:
        return []
    molecule = Chem.MolFromSmiles(reference.get("smiles") or reference.get("normalized_smiles") or "")
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError("Invalid reference for applicability screen")
    molecule = Chem.MolFromSmiles(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    categories, _ = _rule_library(rules_path)
    if category == "all":
        rules = tuple(rule for key in sorted(categories) for rule in categories[key])
    elif category in categories:
        rules = categories[category]
    else:
        raise ValueError(f"Unknown category {category!r}")
    matched_queries, applicable = {}, set()
    for rule in rules:
        if rule.subcategory in applicable:
            continue
        query = Chem.MolToSmarts(rule.query)
        if query not in matched_queries:
            matched_queries[query] = bool(molecule.HasSubstructMatch(rule.query))
        if matched_queries[query]:
            applicable.add(rule.subcategory)
    return sorted(applicable)


@lru_cache(maxsize=20000)
def _fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical molecule: {smiles}")
    return _FINGERPRINT.GetFingerprint(molecule)


@lru_cache(maxsize=50000)
def _qed(smiles: str) -> float:
    return float(Descriptors.qed(Chem.MolFromSmiles(smiles)))


@lru_cache(maxsize=50000)
def _sa(smiles: str) -> float:
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError as error:
        raise RuntimeError("Requested an SA gate but RDKit SA scorer is unavailable") from error
    return float(sascorer.calculateScore(Chem.MolFromSmiles(smiles)))


@lru_cache(maxsize=1)
def _alert_catalogue():
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


@lru_cache(maxsize=50000)
def _alerts(smiles: str) -> int:
    return len(_alert_catalogue().GetMatches(Chem.MolFromSmiles(smiles)))


def _canonical_product(product):
    with rdBase.BlockLogs():
        Chem.SanitizeMol(product)
        for atom in product.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
        # A fresh parse verifies the serialized graph; it also discards stale
        # reaction properties while preserving explicit stereochemistry.
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError("Reaction produced an empty or unparseable structure")
    return smiles, molecule


def _applicable_rules(molecule, rules, rng, support_power, stats, site_limit):
    applicable = []
    for rule in rules:
        stats["rule_template_checks"] += 1
        try:
            # The capped count is explicitly a lower bound when the cap is
            # reached, not a claimed exact count of all reaction sites.
            matches = molecule.GetSubstructMatches(rule.query, uniquify=True, maxMatches=256)
        except Exception:
            stats["template_match_errors"] += 1
            continue
        if not matches:
            continue
        stats["applicable_rule_events"] += 1
        stats["template_match_events_lower_bound"] += len(matches)
        weight = rule.support ** support_power
        key = -math.log(max(rng.random(), 1e-15)) / weight
        applicable.append((key, rule.name, rule, {
            "reactant_template_match_count_lower_bound": len(matches),
            "template_match_count_capped": len(matches) == 256,
            "reactant_template_matches": [list(match) for match in matches[:site_limit]],
            "site_catalogue_truncated": len(matches) > site_limit,
        }))
    applicable.sort(key=lambda item: (item[0], item[1]))
    return applicable


def _diverse_frontier(nodes: list[Node], limit: int) -> list[Node]:
    if len(nodes) <= limit:
        return nodes
    # Greedy max-min selection over a bounded per-level pool. Deterministic
    # rule order breaks equal distances; it does not assert chemical efficacy.
    selected = [nodes[0]]
    pool = list(nodes[1:])
    minima = [1.0 - DataStructs.TanimotoSimilarity(_fingerprint(n.smiles), _fingerprint(nodes[0].smiles))
              for n in pool]
    while pool and len(selected) < limit:
        index = max(range(len(pool)), key=lambda i: (minima[i], -i))
        chosen = pool.pop(index)
        minima.pop(index)
        selected.append(chosen)
        chosen_fp = _fingerprint(chosen.smiles)
        for i, candidate in enumerate(pool):
            minima[i] = min(minima[i], 1.0 - DataStructs.TanimotoSimilarity(_fingerprint(candidate.smiles), chosen_fp))
    return selected


def generate_for_reference(
    reference: dict, category: str, seed: int, max_candidates: int, depth: int = 1,
    *, sim_lo: float = 0.35, sim_hi: float = 0.98, min_depth: int = 1,
    subcategory: str | None = None, difficulty: str | None = None,
    rules_path: str | Path = DEFAULT_RULES, beam_width: int = 64,
    max_products_per_rule: int = 256, max_rule_applications_per_node: int | None = None,
    support_power: float = 0.5, history_site_limit: int = 8,
    min_heavy_atoms: int = 5, max_heavy_atoms: int = 90,
    min_mw: float = 50.0, max_mw: float = 1000.0,
    qed_min: float | None = None, max_sa: float | None = None,
    max_alerts: int | None = None, allow_radicals: bool = False,
    report: dict | None = None,
) -> Iterator[dict]:
    """Yield unique synthetic M products and their actual reaction histories.

    ``depth`` is the maximum number of recorded transformations. With more than
    one permitted depth, the per-depth quotas reserve output capacity for deeper
    products. ``min_depth=depth`` forces exactly that many transformations.
    Source rule subcategories can be targeted directly by the deficit scheduler.
    Optional ``report`` receives counters and an exhaustion reason on completion.
    """
    if max_candidates < 0 or not 1 <= min_depth <= depth:
        raise ValueError("Need max_candidates>=0 and 1<=min_depth<=depth")
    if not 0 <= sim_lo <= sim_hi <= 1 or not 0 <= support_power <= 1:
        raise ValueError("Similarity bounds and support_power must lie in [0,1]")
    if min(beam_width, max_products_per_rule, history_site_limit) < 1:
        raise ValueError("beam_width, max_products_per_rule and history_site_limit must be positive")
    if max_rule_applications_per_node is not None and max_rule_applications_per_node < 1:
        raise ValueError("max_rule_applications_per_node must be positive when supplied")
    if not 1 <= min_heavy_atoms <= max_heavy_atoms or not 0 <= min_mw <= max_mw:
        raise ValueError("Invalid molecular size bounds")
    if qed_min is not None and not 0 <= qed_min <= 1:
        raise ValueError("qed_min must be in [0,1]")
    if max_sa is not None and not 1 <= max_sa <= 10:
        raise ValueError("max_sa must be in [1,10]")
    if max_alerts is not None and max_alerts < 0:
        raise ValueError("max_alerts cannot be negative")
    reference_id = str(reference.get("reference_id") or reference.get("source_id") or "").strip()
    reference_name = str(reference.get("name") or reference.get("reference_name") or "").strip()
    if not reference_id or not reference_name:
        raise ValueError("Named references require nonempty reference_id/source_id and name/reference_name")
    source = reference.get("smiles") or reference.get("normalized_smiles")
    parent = Chem.MolFromSmiles(source or "")
    if parent is None or not parent.GetNumAtoms():
        raise ValueError(f"Invalid reference structure: {reference_id}")
    if len(Chem.GetMolFrags(parent)) != 1:
        raise ValueError("Pass an explicitly standardized single-component reference, retaining its source salt/form separately")
    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    parent_fp = _fingerprint(parent_smiles)
    categories, rules_digest = _rule_library(rules_path)
    if category == "all":
        rules = tuple(rule for name in sorted(categories) for rule in categories[name])
    elif category in categories:
        rules = categories[category]
    else:
        raise ValueError(f"Unknown category {category!r}; choose {sorted(categories)} or all")
    if subcategory is not None:
        rules = tuple(rule for rule in rules if rule.subcategory == subcategory)
    rng = random.Random(seed)
    counters = Counter()
    accepted_depths, accepted_categories = Counter(), Counter()
    stats = report if report is not None else {}
    stats.update({"reference_id": reference_id, "category": category, "subcategory_filter": subcategory,
                  "seed": seed, "max_candidates": max_candidates, "depth": depth, "min_depth": min_depth,
                  "compiled_rules_in_cell": len(rules), "rules_sha256": rules_digest,
                  "similarity_fingerprint": FINGERPRINT_DESCRIPTION})
    start = time.monotonic()
    accepted = 0
    reason = "exhausted_applicable_search_budget"
    try:
        eligible = str(reference.get("generation_eligible", "true")).strip().lower()
        if eligible in {"0", "false", "no"}:
            reason = "reference_marked_ineligible"
            return
        if not max_candidates:
            reason = "zero_requested"
            return
        if not rules:
            reason = "no_rules_in_requested_cell"
            return
        seen = {parent_smiles}
        frontier = [Node(parent_smiles, ())]
        for step in range(1, depth + 1):
            next_pool: list[Node] = []
            step_accepted = 0
            per_step_quota = math.ceil((max_candidates - accepted) / (depth - step + 1)) if step >= min_depth else 0
            for node in frontier:
                molecule = Chem.MolFromSmiles(node.smiles)
                applicable = _applicable_rules(molecule, rules, rng, support_power, counters, history_site_limit)
                if max_rule_applications_per_node is not None:
                    applicable = applicable[:max_rule_applications_per_node]
                for _, _, rule, site_info in applicable:
                    counters["reaction_calls"] += 1
                    try:
                        with rdBase.BlockLogs():
                            products = rule.reaction.RunReactants((molecule,), maxProducts=max_products_per_rule)
                    except Exception:
                        counters["reaction_errors"] += 1
                        continue
                    indexes = list(range(len(products)))
                    rng.shuffle(indexes)
                    if len(products) >= max_products_per_rule:
                        counters["product_enumeration_capped_events"] += 1
                    for product_index in indexes:
                        counters["reaction_products"] += 1
                        if len(products[product_index]) != 1:
                            counters["multi_product_rejections"] += 1
                            continue
                        try:
                            smiles, child = _canonical_product(products[product_index][0])
                        except Exception:
                            counters["invalid_products"] += 1
                            continue
                        if smiles in seen:
                            counters["duplicate_or_unchanged_products"] += 1
                            continue
                        if len(Chem.GetMolFrags(child)) != 1:
                            counters["fragment_rejections"] += 1
                            continue
                        if any(atom.GetAtomicNum() == 0 or (not allow_radicals and atom.GetNumRadicalElectrons()) for atom in child.GetAtoms()):
                            counters["dummy_or_radical_rejections"] += 1
                            continue
                        heavy = child.GetNumHeavyAtoms()
                        mw = float(Descriptors.MolWt(child))
                        if not min_heavy_atoms <= heavy <= max_heavy_atoms or not min_mw <= mw <= max_mw:
                            counters["size_rejections"] += 1
                            continue
                        seen.add(smiles)
                        optional_quality = {}
                        if qed_min is not None:
                            optional_quality["qed"] = _qed(smiles)
                            if optional_quality["qed"] < qed_min:
                                counters["qed_rejections"] += 1
                                continue
                        if max_sa is not None:
                            optional_quality["sa_score"] = _sa(smiles)
                            if optional_quality["sa_score"] > max_sa:
                                counters["sa_rejections"] += 1
                                continue
                        if max_alerts is not None:
                            optional_quality["structural_alerts"] = _alerts(smiles)
                            if optional_quality["structural_alerts"] > max_alerts:
                                counters["structural_alert_rejections"] += 1
                                continue
                        history_entry = {"step": step, "rule_name": rule.name,
                            "source_rule_category": rule.category, "source_rule_subcategory": rule.subcategory,
                            "source_rule_difficulty": rule.difficulty, "mined_support": rule.support,
                            "before_smiles": node.smiles, "after_smiles": smiles,
                            "product_enumeration_index": product_index,
                            "site_indices_scope": "intermediate_before_smiles; matching sites are not final edited-atom masks",
                            **site_info}
                        new_node = Node(smiles, node.history + (history_entry,))
                        if step < depth and len(next_pool) < beam_width * 2:
                            next_pool.append(new_node)
                        similarity = float(DataStructs.TanimotoSimilarity(parent_fp, _fingerprint(smiles)))
                        if step < min_depth or step_accepted >= per_step_quota:
                            continue
                        if not sim_lo <= similarity <= sim_hi:
                            counters["similarity_rejections"] += 1
                            continue
                        actual_categories = {op["source_rule_category"] for op in new_node.history}
                        assigned_category = next(iter(actual_categories)) if len(actual_categories) == 1 else "mixed"
                        measured_band = "near" if similarity >= 0.8 else "intermediate" if similarity >= 0.55 else "distant"
                        accepted += 1
                        step_accepted += 1
                        accepted_depths[step] += 1
                        accepted_categories[assigned_category] += 1
                        counters["accepted"] = accepted
                        yield {
                            "smiles": smiles, "parent_smiles": parent_smiles,
                            "reference_id": reference_id, "reference_name": reference_name,
                            "reference_source": reference.get("source", ""),
                            "reference_source_id": reference.get("source_id", ""),
                            "reference_status": reference.get("status", "unspecified"),
                            "generation_method": "M", "generation_protocol": "applicable_mined_mmp_enumeration",
                            "category": assigned_category, "subcategory": rule.subcategory,
                            "rule_name": rule.name, "rule_difficulty": rule.difficulty,
                            "scenario": "synthetic_structural_analogue",
                            "evidence_tier": "synthetic_generator_relation", "confirmed_real_case": 0,
                            "difficulty": difficulty or measured_band, "measured_similarity_band": measured_band,
                            "similarity_to_parent": similarity, "similarity_fingerprint": FINGERPRINT_DESCRIPTION,
                            "operator_history": json.dumps(new_node.history, separators=(",", ":"), sort_keys=True),
                            "operator_history_steps": step, "lineage_schema_version": 1,
                            "verification_status": "sanitized_recorded_reaction_lineage;net_edit_category_not_reclassified",
                            "category_verification_status": "source_rule_annotations;net_parent_child_edit_unverified",
                            "stereochemistry_policy": "RDKit reaction propagation and isomeric SMILES; no stereoisomer enumeration",
                            "rules_sha256": rules_digest, "seed": seed,
                            "mw": mw, "heavy_atoms": heavy, **optional_quality,
                        }
                        if accepted >= max_candidates:
                            reason = "max_candidates_reached"
                            return
                    if step_accepted >= per_step_quota and (step == depth or len(next_pool) >= beam_width * 2):
                        break
                if step_accepted >= per_step_quota and (step == depth or len(next_pool) >= beam_width * 2):
                    break
            frontier = _diverse_frontier(next_pool, beam_width)
            if step < depth and not frontier:
                reason = "no_valid_frontier_for_next_depth"
                break
    finally:
        stats.update(dict(counters))
        stats.update({"accepted": accepted, "accepted_by_depth": dict(accepted_depths),
                      "accepted_by_category": dict(accepted_categories), "stop_reason": reason,
                      "elapsed_seconds": round(time.monotonic() - start, 6)})


def generate_with_report(*args, **kwargs):
    report = {}
    rows = list(generate_for_reference(*args, **kwargs, report=report))
    return rows, report


def self_check():
    reference = {"reference_id": "fixture_aspirin", "name": "aspirin fixture",
                 "smiles": "CC(=O)Oc1ccccc1C(=O)O", "status": "test_fixture"}
    options = dict(category="homologation", seed=712, max_candidates=12, sim_lo=0.1, sim_hi=0.99)
    first, report = generate_with_report(reference, **options)
    second = list(generate_for_reference(reference, **options))
    assert first and first == second, "Seeded M enumeration is not deterministic"
    assert len({r["smiles"] for r in first}) == len(first)
    assert all("label" not in row and row["confirmed_real_case"] == 0 for row in first)
    deeper, deeper_report = generate_with_report(reference, **{**options, "depth": 2, "min_depth": 2})
    assert deeper and all(row["operator_history_steps"] == 2 for row in deeper)
    for row in first + deeper:
        history = json.loads(row["operator_history"])
        assert history[0]["before_smiles"] == row["parent_smiles"]
        assert history[-1]["after_smiles"] == row["smiles"]
        assert all(a["after_smiles"] == b["before_smiles"] for a, b in zip(history, history[1:]))
        molecule = Chem.MolFromSmiles(row["smiles"])
        assert molecule is not None and len(Chem.GetMolFrags(molecule)) == 1
        assert not any(atom.GetAtomMapNum() or atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms())
    forbidden = list(generate_for_reference({**reference, "generation_eligible": "0"}, **options))
    assert not forbidden, "Ineligible holdout reference generated children"
    # A remote ether extension must preserve the untouched stereocentre.
    stereo = Chem.MolFromSmiles("COC[C@H](O)C(=O)O")
    reaction = AllChem.ReactionFromSmarts("[*:1][O:2][CH3:3]>>[*:1][O:2][CH2:3]C")
    stereo_products = [_canonical_product(products[0])[0] for products in reaction.RunReactants((stereo,))]
    assert stereo_products and all("@" in smi for smi in stereo_products)
    return {"checks_passed": True, "single_step_candidates": len(first),
            "two_step_candidates": len(deeper), "single_step_reaction_calls": report["reaction_calls"],
            "two_step_reaction_calls": deeper_report["reaction_calls"],
            "checks": ["determinism", "canonical uniqueness", "lineage chain", "chemical serialization",
                       "no counterfeit labels", "ineligible reference exclusion", "remote stereochemistry preservation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--category", default="all")
    parser.add_argument("--subcategory")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-lo", type=float, default=0.35)
    parser.add_argument("--sim-hi", type=float, default=0.98)
    parser.add_argument("--output", type=Path, help="Explicit JSONL proposal output; no bank is modified")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if args.reference_json is None:
        print(json.dumps(rule_summary(), indent=2))
        return
    reference = json.loads(args.reference_json.read_text())
    rows, report = generate_with_report(reference, args.category, args.seed, args.max_candidates,
        depth=args.depth, min_depth=args.min_depth, sim_lo=args.sim_lo, sim_hi=args.sim_hi,
        subcategory=args.subcategory)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
