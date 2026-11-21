#!/usr/bin/env python3
"""Build an evidence-grounded case registry and a declared/observed challenge.

Real cases are never assigned to training. Positive-only case reports cannot
establish specificity, ROC-AUC, F1, or population-level counterfeit prevalence.
Known exact chemical graphs support an identity-comparison baseline; a GNN
must eventually add value under uncertain measured observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from real_case_curation import CASES, COMPOUNDS, CURATION_DATE, SOURCES


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=150000)
def standardized(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid chemical structure: {smiles}")
    molecule = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(molecule))
    molecule = rdMolStandardize.Uncharger().uncharge(molecule)
    Chem.SanitizeMol(molecule)  # standardization can invalidate cached ring info
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(molecule, isomericSmiles=False)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False)
    scaffold_key = scaffold or f"ACYCLIC:{connectivity}"
    return canonical, connectivity, scaffold_key


def load_molecules(cache):
    rows = []
    for compound_id, spec in COMPOUNDS.items():
        path = cache / f"{compound_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; first run fetch_real_case_structures.py")
        raw = json.loads(path.read_text())
        if raw["query"] != spec["query"]:
            raise ValueError(f"Cached name lookup differs from curation for {compound_id}")
        entries = raw["response"]["PropertyTable"]["Properties"]
        if len(entries) != 1:
            raise ValueError(f"Ambiguous chemical identity for {compound_id}")
        entry = entries[0]
        smiles = entry.get("SMILES") or entry.get("IsomericSMILES")
        normalized, connectivity, scaffold = standardized(smiles)
        rows.append({
            "compound_id": compound_id, "name": compound_id.replace("_", " "),
            "pubchem_title": entry["Title"], "pubchem_query": spec["query"],
            "pubchem_cid": entry["CID"], "pubchem_smiles": smiles,
            "pubchem_inchikey": entry["InChIKey"], "normalized_smiles": normalized,
            "inchikey": Chem.MolToInchiKey(Chem.MolFromSmiles(normalized)),
            "connectivity_smiles": connectivity, "scaffold_key": scaffold,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{entry['CID']}",
            "request_url": raw["request_url"], "retrieved_utc": raw["retrieved_utc"],
            "identity_scope": "normalized_active_moiety_representative_not_whole_product",
            "mapping_note": spec.get("mapping_note", "Named API/chemical identity; report does not provide a measured molecular structure file."),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("real_world_falsification/pubchem_cache"))
    parser.add_argument("--out", type=Path, default=Path("real_world_falsification/v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    molecules = load_molecules(args.cache)
    by_id = {r["compound_id"]: r for r in molecules}
    if len(by_id) != len(molecules):
        raise ValueError("Duplicate compound identifiers")
    if len({case["case_id"] for case in CASES}) != len(CASES):
        raise ValueError("Duplicate case identifiers")

    components, case_rows, graph_pairs, composition_sets = [], [], [], []
    for case in CASES:
        declared = case["declared_compound_ids"]
        detected = case["detected_compound_ids"]
        absent = case["not_detected_declared_compound_ids"]
        source = SOURCES[case["source_id"]]
        if not set(absent).issubset(declared):
            raise ValueError(f"Absence annotation not in declared set: {case['case_id']}")
        if set(absent) & set(detected):
            raise ValueError(f"Conflicting presence evidence: {case['case_id']}")
        for compound_id in declared + detected + absent:
            if compound_id not in by_id:
                raise ValueError(f"Unresolved compound {compound_id}")

        eligible = (case["scenario"] == "identified_api_substitution"
                    and len(declared) == len(detected) == 1)
        if eligible:
            reason = "single_declared_api_and_identified_substitute"
        elif case["scenario"] == "identified_substitute_mixture":
            reason = "mixture_requires_component_set_or_analytical_observation"
        elif case["scenario"] == "identified_undeclared_contaminant":
            reason = "contaminant_is_not_full_sample_requires_composition_and_assay"
        elif absent:
            reason = "missing_api_without_identified_substitute_requires_assay_status"
        else:
            reason = "chemical_composition_not_reported_requires_provenance_or_packaging"

        case_rows.append({
            "case_id": case["case_id"], "evidence_group_id": case["evidence_group_id"],
            "source_id": case["source_id"], "source_url": source["url"],
            "source_date": source["date"], "authority": source["authority"],
            "product_name": case["product_name"], "batch_lot": case["batch_lot"] or "",
            "reported_region": case["reported_region"], "scenario": case["scenario"],
            "declared_compound_ids": json.dumps(declared), "detected_compound_ids": json.dumps(detected),
            "not_detected_declared_compound_ids": json.dumps(absent),
            "declared_form_as_reported": case["declared_form_as_reported"],
            "declared_strength_as_reported": case["declared_strength_as_reported"],
            "reported_detected_quantities": json.dumps(case["reported_detected_quantities"]),
            "analysis_status": case["analysis_status"], "falsification_status": case["falsification_status"],
            "evidence_tier": case["evidence_tier"], "partition": case["partition"],
            "single_graph_pair_eligible": int(eligible), "eligibility_reason": reason,
            "complete_sample_composition_known": 0, "raw_measurements_available": 0,
            "curation_notes": case["curation_notes"],
        })
        for role, ids in (("declared", declared), ("detected", detected), ("not_detected", absent)):
            for compound_id in ids:
                components.append({"case_id": case["case_id"], "role": role,
                    "compound_id": compound_id, "normalized_smiles": by_id[compound_id]["normalized_smiles"],
                    "pubchem_cid": by_id[compound_id]["pubchem_cid"], "source_id": case["source_id"],
                    "reported_detected_quantity": json.dumps([q for q in case["reported_detected_quantities"]
                        if q["compound_id"] == compound_id]) if role == "detected" else "[]"})
        composition_sets.append({**case,
            "declared_structures": [by_id[k]["normalized_smiles"] for k in declared],
            "detected_structures": [by_id[k]["normalized_smiles"] for k in detected],
            "single_graph_pair_eligible": eligible, "eligibility_reason": reason,
            "source_url": source["url"], "source_date": source["date"],
        })
        if eligible:
            claimed, observed = by_id[declared[0]], by_id[detected[0]]
            relation = claimed["normalized_smiles"] + ">>" + observed["normalized_smiles"]
            relation_id = "real_relation_" + hashlib.sha256(relation.encode()).hexdigest()[:16]
            graph_pairs.append({"case_id": case["case_id"], "evidence_group_id": case["evidence_group_id"],
                "relation_id": relation_id, "declared_compound_id": declared[0],
                "observed_compound_id": detected[0], "declared_smiles": claimed["normalized_smiles"],
                "observed_smiles": observed["normalized_smiles"], "composition_mismatch": 1,
                "observed_structure_scope": "identified_substitute_component_not_complete_sample",
                "evidence_tier": "official_case_report", "partition": "external_case_challenge_only",
                "source_url": source["url"]})

    # All source clusters stay in a single external partition. Keep repeated
    # batch/presentation records but report their unique relation count.
    write_csv(args.out / "molecules.csv", molecules)
    write_csv(args.out / "cases.csv", case_rows)
    write_csv(args.out / "components.csv", components)
    write_csv(args.out / "structure_pairs.csv", graph_pairs)
    with (args.out / "composition_sets.jsonl").open("w") as stream:
        for record in composition_sets:
            stream.write(json.dumps(record) + "\n")
    (args.out / "sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    (args.out / "curation_snapshot.json").write_text(json.dumps({"date": CURATION_DATE,
        "cases": CASES, "compound_queries": COMPOUNDS}, indent=2) + "\n")
    (args.out / "external_holdout.json").write_text(json.dumps({
        "partition": "external_case_challenge_only",
        "case_ids": [case["case_id"] for case in CASES],
        "source_report_groups": sorted(SOURCES),
        "blocked_connectivity_smiles": sorted({row["connectivity_smiles"] for row in molecules}),
        "blocked_scaffold_keys": sorted({row["scaffold_key"] for row in molecules}),
        "policy": "Exclude these identities/scaffolds from both declared and detected sides of future molecular training. Preserve report groups; audit any measurement dataset by product/API/sample family separately.",
        "old_model_retroactively_clean": False,
    }, indent=2) + "\n")
    identity_results = [{"relation_id": r["relation_id"], "case_id": r["case_id"],
        "different_normalized_structures": r["declared_smiles"] != r["observed_smiles"]} for r in graph_pairs]
    summary = {
        "curation_date": CURATION_DATE, "source_count": len(SOURCES), "reported_case_records": len(CASES),
        "source_report_clusters": len({r["evidence_group_id"] for r in CASES}),
        "cluster_policy": "One group per alert/source report; statistical independence between sources is not established.",
        "unique_compound_records": len(molecules), "scenario_counts": dict(Counter(r["scenario"] for r in CASES)),
        "single_graph_pair_records": len(graph_pairs),
        "unique_single_substitution_relations": len({r["relation_id"] for r in graph_pairs}),
        "single_graph_pair_evidence_groups": len({r["evidence_group_id"] for r in graph_pairs}),
        "not_eligible_for_single_graph_pair": len(CASES) - len(graph_pairs),
        "verified_authentic_control_samples": 0, "available_raw_spectra": 0,
        "case_partition": "external_case_challenge_only",
        "identity_normalization": "RDKit Cleanup + FragmentParent + Uncharger, preserving reported database stereochemistry; no automatic tautomer merging. Salts/forms remain separately documented.",
        "metric_policy": "Positive-only reports do not permit clinical ROC-AUC/F1/specificity. Aggregate related batches by relation and evidence group.",
        "exact_identity_baseline": {
            "different_structures_records": sum(r["different_normalized_structures"] for r in identity_results),
            "records": len(identity_results),
            "interpretation": "Deterministic identity check with already identified exact structures, not a measured counterfeit-detection performance estimate.",
        },
        "limitations": [
            "Public reports summarize selected tested samples; no claim of complete surveillance or sample-level raw analytical data.",
            "A detected ingredient is not a complete sample composition; unreported and absent are distinct.",
            "PubChem structures are named chemical reference mappings, not experimental conformations from the seized products.",
            "M/E generator outputs remain synthetic analogues and are not relabeled as real cases.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
