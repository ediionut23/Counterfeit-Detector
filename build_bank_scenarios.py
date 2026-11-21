#!/usr/bin/env python3
"""Create resumable synthetic declared-reference/composition observations.

Only the BankStore ``samples`` and ``metadata`` tables are written. Chemical
catalogue size never increases because of a sample, mixture or quantity factor.
Amounts are dimensionless simulation units, not clinical doses or measured
assays. Source drug status does not authenticate a product, and no intrinsic
fraud label is assigned to any molecule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, deque
from pathlib import Path

from scalable_bank_store import BankStore

PROTOCOL = "claim_conditioned_scenarios_v1"
EVIDENCE_TIER = "synthetic_scenario"
SAMPLE_FIELDS = ["sample_id", "declared_reference_id", "observed_components_json",
                 "quantities_json", "scenario", "evidence_tier",
                 "composition_mismatch", "metadata_json"]


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def semantic_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_eligible_references(store, holdout):
    """Read eligible named references already registered in the bank.

    Apply the external holdout again to guard against an incorrectly imported
    registry. Multiple source IDs for one molecule cannot inflate partner pools.
    """
    blocked_identity = set(holdout.get("blocked_connectivity_smiles", []))
    blocked_scaffold = set(holdout.get("blocked_scaffold_keys", []))
    query = """SELECT r.reference_id,r.name,r.source,r.source_id,r.status,r.smiles AS reference_smiles,
        r.molecule_id,m.smiles,m.connectivity_smiles,m.scaffold_key
        FROM references_catalogue r JOIN molecules m ON r.molecule_id=m.molecule_id
        WHERE r.generation_eligible=1 AND m.training_eligible=1
          AND r.name IS NOT NULL AND TRIM(r.name)<>''
        ORDER BY r.reference_id"""
    output, seen, excluded = [], set(), Counter()
    for record in store.connection.execute(query):
        row = dict(record)
        if row["connectivity_smiles"] in blocked_identity:
            excluded["external_identity_holdout"] += 1
            continue
        if row["scaffold_key"] in blocked_scaffold:
            excluded["external_scaffold_holdout"] += 1
            continue
        if row["reference_smiles"] != row["smiles"]:
            raise ValueError(f"Registered reference and chemical catalogue structures disagree: {row['reference_id']}")
        if row["molecule_id"] in seen:
            excluded["duplicate_reference_identity"] += 1
            continue
        seen.add(row["molecule_id"])
        output.append(row)
    if not output:
        raise ValueError("No eligible named references registered in BankStore; import the named registry first")
    return output, dict(excluded)


def component(reference):
    return {key: reference[key] for key in ("molecule_id", "reference_id", "name", "smiles")}


def build_sample(declared, observed, amounts, scenario, *, identity_mismatch,
                 quantity_mismatch, metadata=None):
    if len(observed) != len(amounts):
        raise ValueError("Each simulated detected component needs one relative amount")
    if len({r["molecule_id"] for r in observed}) != len(observed):
        raise ValueError("Component sets must not repeat a chemical identity")
    if any(not math.isfinite(value) or value <= 0 for value in amounts):
        raise ValueError("Present components require positive finite dimensionless amounts")
    zipped = sorted(zip(observed, amounts), key=lambda pair: pair[0]["molecule_id"])
    observations = [component(reference) for reference, _ in zipped]
    quantities = {
        "unit": "dimensionless_relative_component_amount",
        "declared": [{"molecule_id": declared["molecule_id"], "relative_amount": 1.0}],
        "observed": [{"molecule_id": reference["molecule_id"], "relative_amount": float(amount)}
                     for reference, amount in zipped],
        "basis": "synthetic_nominal_reference_unit_not_a_clinical_dose_or_assay",
        "measured": False,
    }
    detail = {
        "protocol": PROTOCOL, "simulation_status": "explicitly_synthetic",
        "confirmed_real_case": False, "measured_observation_available": False,
        "identity_mismatch": identity_mismatch, "quantity_mismatch": quantity_mismatch,
        "clinical_authenticity_or_fraud_label": None,
        "component_scope": "simulated_active_components_not_complete_product",
        "inactive_matrix_status": "unknown_unspecified",
        "declared_source_status": declared["status"],
        "observed_source_statuses": sorted({r["status"] for r in observed}),
        "declared_reference_source": declared["source"],
        "declared_reference_source_id": declared["source_id"],
        "split_group_id": declared["reference_id"],
        "split_policy": "declared reference is a minimum grouping key; chemical-family and batch grouping depend on the evaluation task",
    }
    if metadata:
        detail.update(metadata)
    if not observations:
        if scenario != "simulated_declared_api_absence" or detail.get("declared_api_presence") != "simulated_absent":
            raise ValueError("An empty component list requires an explicit simulated-absence state")
        detail["empty_component_list_semantics"] = "no_detected_component_identity_supplied_unknown_matrix_not_empty_physical_sample_or_molecule_graph"
        quantities["observed_quantity_status"] = "not_available_unknown_matrix"
    record = {
        "declared_reference_id": declared["reference_id"],
        "observed_components_json": canonical_json(observations),
        "quantities_json": canonical_json(quantities), "scenario": scenario,
        "evidence_tier": EVIDENCE_TIER,
        "composition_mismatch": int(bool(identity_mismatch or quantity_mismatch)),
        "metadata_json": canonical_json(detail),
    }
    # All stored semantic fields are hashed. Operational run seed/checkpoints
    # live in the metadata run ledger, so an identical self-control is not
    # duplicated merely because it was requested in a second random run.
    return {"sample_id": "sample_" + semantic_hash(record), **record}


def partners_for(reference, pool, seed):
    candidates = [row for row in pool if row["molecule_id"] != reference["molecule_id"]]
    return sorted(candidates, key=lambda row: semantic_hash({
        "seed": seed, "declared_reference_id": reference["reference_id"],
        "observed_molecule_id": row["molecule_id"], "purpose": "known_reference_partner_order"}))


def samples_for_reference(reference, pool, *, seed, substitutions, mixtures,
                          controls, absences, quantity_factors):
    partners = partners_for(reference, pool, seed)
    samples, deficits = [], Counter()
    for substitute in partners[:substitutions]:
        samples.append(build_sample(reference, [substitute], [1.0], "simulated_known_reference_substitution",
            identity_mismatch=True, quantity_mismatch=None,
            metadata={"mismatch_basis": "declared_identity_replaced", "declared_api_presence": "simulated_absent",
                      "relative_amounts_do_not_imply_pharmacological_equivalence": True}))
    deficits["substitutions"] = max(0, substitutions - len(partners))

    # Choose diverse component sets and relative fractions. A mixture stays one
    # observation even when it contains multiple known molecular structures.
    mixture_candidates = []
    for index, first in enumerate(partners):
        for retained_fraction in (0.9, 0.5, 0.1):
            mixture_candidates.append(build_sample(reference, [reference, first],
                [retained_fraction, round(1.0 - retained_fraction, 12)], "simulated_two_component_adulteration",
                identity_mismatch=True, quantity_mismatch=True,
                metadata={"mismatch_basis": "additional_identity_and_changed_declared_component_amount",
                          "declared_api_presence": "simulated_present", "mixture_type": "declared_api_plus_other_known_compound"}))
        if len(partners) > 1:
            second = partners[(index + 1) % len(partners)]
            mixture_candidates.append(build_sample(reference, [first, second], [0.5, 0.5],
                "simulated_two_component_replacement", identity_mismatch=True, quantity_mismatch=None,
                metadata={"mismatch_basis": "declared_identity_replaced_by_component_set",
                          "declared_api_presence": "simulated_absent", "mixture_type": "two_other_known_compounds",
                          "relative_amounts_do_not_imply_pharmacological_equivalence": True}))
        if len(mixture_candidates) >= max(12, mixtures * 4):
            break
    # Fixed archetype ordering ensures default quotas cover both retained-API
    # adulteration and replacement sets when enough partner identities exist.
    mixture_candidates.sort(key=lambda r: semantic_hash({
        "seed": seed, "sample_id": r["sample_id"], "purpose": "mixture_selection"}))
    unique_mixtures = {row["sample_id"]: row for row in mixture_candidates}
    replacement = deque(r for r in unique_mixtures.values() if r["scenario"] == "simulated_two_component_replacement")
    adulteration = deque(r for r in unique_mixtures.values() if r["scenario"] == "simulated_two_component_adulteration")
    selected_mixtures = []
    cycle = (replacement, adulteration, adulteration)
    index = 0
    while len(selected_mixtures) < mixtures and (replacement or adulteration):
        bucket = cycle[index % len(cycle)]
        if bucket:
            selected_mixtures.append(bucket.popleft())
        index += 1
    samples.extend(selected_mixtures)
    deficits["mixtures"] = max(0, mixtures - len(selected_mixtures))

    if controls:
        samples.append(build_sample(reference, [reference], [1.0], "synthetic_identity_consistent_control",
            identity_mismatch=False, quantity_mismatch=False,
            metadata={"mismatch_basis": "none_in_modeled_identity_or_amount",
                      "control_semantics": "identity_consistent_simulation_not_verified_authentic_product"}))
    if absences:
        samples.append(build_sample(reference, [], [], "simulated_declared_api_absence",
            identity_mismatch=True, quantity_mismatch=None,
            metadata={"mismatch_basis": "simulated_presence_status_without_observed_component_graph",
                      "declared_api_presence": "simulated_absent", "observation_modality": "simulated_presence_status"}))
    for factor in quantity_factors:
        samples.append(build_sample(reference, [reference], [factor], "simulated_relative_quantity_drift",
            identity_mismatch=False, quantity_mismatch=True,
            metadata={"mismatch_basis": "relative_quantity_only", "declared_api_presence": "simulated_present",
                      "relative_quantity_factor": factor, "molecular_graph_alone_is_insufficient": True}))
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Scenario configuration generated duplicate semantic observations")
    return samples, {key: value for key, value in deficits.items() if value}


def export_samples(store, out):
    """Stream the complete sample table, preserving other evidence tiers."""
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "samples.csv.partial"
    rows, own_rows = 0, 0
    scenarios, tiers, mismatches, own_scenarios = Counter(), Counter(), Counter(), Counter()
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        cursor = store.connection.execute("SELECT " + ",".join(SAMPLE_FIELDS) + " FROM samples ORDER BY sample_id")
        while batch := cursor.fetchmany(1000):
            for record in batch:
                row = dict(record)
                writer.writerow(row)
                rows += 1
                scenarios[row["scenario"]] += 1
                tiers[row["evidence_tier"]] += 1
                mismatches[str(row["composition_mismatch"])] += 1
                detail = json.loads(row["metadata_json"] or "{}")
                if detail.get("protocol") == PROTOCOL:
                    own_rows += 1
                    own_scenarios[row["scenario"]] += 1
    temporary.replace(out / "samples.csv")
    return {"sample_records_in_store": rows, "sample_records_from_this_protocol": own_rows,
            "scenario_counts_in_store": dict(scenarios), "evidence_tier_counts_in_store": dict(tiers),
            "composition_mismatch_counts_in_store": dict(mismatches),
            "scenario_counts_from_this_protocol": dict(own_scenarios)}


def run(store, out, holdout, *, seed=20261031, substitutions=5, mixtures=3,
        controls=1, absences=1, quantity_factors=(0.5, 1.5), batch_size=100,
        max_references=None):
    if substitutions < 0 or mixtures < 0 or controls not in (0, 1) or absences not in (0, 1) or batch_size < 1:
        raise ValueError("Invalid quotas; identity controls and absence statuses are at most one per reference")
    if max_references is not None and max_references < 1:
        raise ValueError("Declared-reference limit must be positive")
    factors = sorted(set(float(value) for value in quantity_factors))
    if any(not math.isfinite(v) or v <= 0 or v == 1 for v in factors):
        raise ValueError("Relative quantity factors must be positive finite values different from 1")
    references, excluded = load_eligible_references(store, holdout)
    target = references[:max_references] if max_references is not None else references
    pool_fingerprint = semantic_hash(references)
    config = {"protocol": PROTOCOL, "seed": seed, "substitutions_per_reference": substitutions,
              "mixtures_per_reference": mixtures, "controls_per_reference": controls,
              "absences_per_reference": absences, "quantity_factors": factors,
              "reference_pool_sha256": pool_fingerprint, "declared_reference_limit": max_references,
              "external_holdout_sha256": semantic_hash(holdout)}
    run_id = "scenario_run_" + semantic_hash(config)
    metadata_key = "scenario_generation:" + run_id
    previous = store.connection.execute("SELECT value FROM metadata WHERE key=?", (metadata_key,)).fetchone()
    checkpoint = json.loads(previous[0]) if previous else {}
    # Completed invocations can be replayed to repair a removed sample while
    # INSERT OR IGNORE keeps the usual second invocation strictly idempotent.
    start = int(checkpoint.get("references_completed", 0)) if checkpoint.get("state") == "running" else 0
    if not 0 <= start <= len(target):
        raise ValueError("Stored scenario checkpoint is inconsistent with the frozen reference pool")
    attempted = int(checkpoint.get("attempted_semantic_records", 0)) if start else 0
    deficits = Counter(checkpoint.get("unmet_quotas", {})) if start else Counter()
    added = 0
    catalogue_before = store.count()
    for base in range(start, len(target), batch_size):
        block = target[base:base + batch_size]
        with store.connection:
            for reference in block:
                records, missing = samples_for_reference(reference, references, seed=seed,
                    substitutions=substitutions, mixtures=mixtures, controls=controls,
                    absences=absences, quantity_factors=factors)
                deficits.update(missing)
                attempted += len(records)
                for record in records:
                    cursor = store.connection.execute(
                        "INSERT OR IGNORE INTO samples(" + ",".join(SAMPLE_FIELDS) + ") VALUES(" + ",".join("?" for _ in SAMPLE_FIELDS) + ")",
                        tuple(record[field] for field in SAMPLE_FIELDS))
                    added += int(cursor.rowcount == 1)
                    if cursor.rowcount == 0:
                        existing = store.connection.execute(
                            "SELECT " + ",".join(SAMPLE_FIELDS) + " FROM samples WHERE sample_id=?",
                            (record["sample_id"],)).fetchone()
                        if existing is None or any(existing[field] != record[field] for field in SAMPLE_FIELDS):
                            raise ValueError("Sample semantic hash collision or altered existing sample payload")
            checkpoint = {"config": config, "state": "running", "references_completed": base + len(block),
                          "attempted_semantic_records": attempted, "unmet_quotas": dict(deficits)}
            store.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                                     (metadata_key, canonical_json(checkpoint)))
        print(f"Scenario references {base + len(block):,}/{len(target):,}; new sample records this invocation {added:,}", flush=True)
    checkpoint = {"config": config, "state": "complete", "references_completed": len(target),
                  "attempted_semantic_records": attempted, "unmet_quotas": dict(deficits)}
    with store.connection:
        store.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (metadata_key, canonical_json(checkpoint)))
    exported = export_samples(store, out)
    summary = {
        "protocol": PROTOCOL, "run_id": run_id, "config": config,
        "eligible_unique_reference_pool": len(references), "declared_references_processed": len(target),
        "excluded_registered_references": excluded, "newly_inserted_samples": added,
        "attempted_semantic_records_for_this_run": attempted, "unmet_quotas": dict(deficits),
        "resumed_after_reference_index": start, "sample_export": str(out / "samples.csv"),
        "chemical_catalogue_count_before": catalogue_before, "chemical_catalogue_count_after": store.count(),
        "molecule_insert_operations_by_this_script": 0, "sample_rows_count_as_new_molecules": False,
        "evidence_policy": "All records created by this protocol are synthetic scenarios, without intrinsic molecule fraud/authenticity labels.",
        "quantity_policy": "Dimensionless normalized component amounts only; no clinical doses or measured assays are invented.",
        "empty_component_policy": "Only explicitly simulated API absence; matrix remains unknown and no empty molecule graph is constructed.",
        "id_policy": "SHA-256 of all stored semantic fields; operational seed/run checkpoint is kept separately so duplicate self-controls are not inflated across seeds.",
        **exported,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text("""# Synthetic claim-conditioned composition observations

Each row is a simulated declared-reference/observed-component record. Samples
are not new chemical catalogue molecules and are not confirmed real cases.
Ordinary reference APIs can substitute for another declared reference without
becoming intrinsically counterfeit molecules.

`samples.csv` streams the complete bank sample table, including its evidence
tier and metadata. This script creates known-reference substitutions,
two-component adulteration/replacement mixtures, identity-consistent controls,
explicit API-absence statuses and relative-quantity drift observations.
Controls are synthetic identity consistency, not authenticated genuine products.
Amounts are dimensionless simulation units; they do not represent clinical
doses, assay measurements or pharmacological equivalence.

Mixtures preserve component sets. An empty component list is permitted only for
explicit simulated API absence with an unknown matrix. It never means an empty
physical product or a graph representing a nonexistent molecule.

Metadata separates identity mismatch from quantity-only mismatch. A molecular
graph alone cannot reveal a changed quantity, missing API, packaging or fraud.
All source-reference eligibility and external identity/scaffold holdouts are
checked again before selection. Future evaluation must group relevant declared
identities, chemical families, lots and replicates according to its task.

The metadata table stores the deterministic run configuration and reference
checkpoint. INSERT OR IGNORE plus full semantic hashes makes retries idempotent.
Completed runs can be replayed without duplicate observations. No molecules
are inserted by this script.
""")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--holdout", type=Path, default=Path("real_world_falsification/v1/external_holdout.json"))
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--substitutions-per-reference", type=int, default=5)
    parser.add_argument("--mixtures-per-reference", type=int, default=3)
    parser.add_argument("--controls-per-reference", type=int, choices=(0, 1), default=1)
    parser.add_argument("--absence-per-reference", type=int, choices=(0, 1), default=1)
    parser.add_argument("--quantity-factors", type=float, nargs="*", default=[0.5, 1.5])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-references", type=int, default=None)
    args = parser.parse_args()
    if args.max_references is not None and args.max_references < 1:
        parser.error("--max-references must be positive")
    out = args.out or args.db.parent / "scenarios"
    store = BankStore(args.db)
    try:
        summary = run(store, out, json.loads(args.holdout.read_text()), seed=args.seed,
            substitutions=args.substitutions_per_reference, mixtures=args.mixtures_per_reference,
            controls=args.controls_per_reference, absences=args.absence_per_reference,
            quantity_factors=args.quantity_factors, batch_size=args.batch_size, max_references=args.max_references)
    finally:
        store.connection.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
