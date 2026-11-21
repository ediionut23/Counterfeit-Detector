#!/usr/bin/env python3
"""Audit a frozen normalized molecular bank and publish streaming CSV exports.

The minimum is checked against distinct catalogue molecules, never against
relations, simulated samples, pharmaceutical forms or conformers. Origin flags
are not clinical authenticity/counterfeit labels. Full identity parsing is
separate from sampled descriptor recomputation and job-specific stored gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import platform
import shutil
import sqlite3
import tempfile
import urllib.parse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

TABLES = ["molecules", "molecule_sources", "molecule_methods", "references_catalogue",
          "relations", "samples", "jobs", "import_sources", "import_chunks", "metadata", "descriptor_cache"]
EXPORTS = {
    "references_catalogue": "references.csv", "relations": "relations.csv",
    "samples": "samples.csv", "molecule_sources": "sources.csv",
    "molecule_methods": "molecule_methods.csv", "jobs": "jobs.csv",
    "import_sources": "import_sources.csv", "metadata": "metadata.csv",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_readonly(path):
    uri = "file:" + urllib.parse.quote(str(Path(path).resolve()), safe="/") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-65536")
    return connection


def table_counts(connection):
    return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES}


def database_fingerprint(path):
    result = {}
    for candidate in (Path(path), Path(str(path) + "-wal")):
        if not candidate.exists() or candidate.name.endswith("-wal") and candidate.stat().st_size == 0:
            continue
        before = candidate.stat()
        digest = checksum(candidate)
        after = candidate.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError(f"Source database changed while fingerprinting: {candidate}")
        result[str(candidate.resolve())] = {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                                            "inode": after.st_ino, "sha256": digest}
    return result


class SmallestHashSample:
    """Bounded deterministic identity sample, unaffected by repeated relations."""
    def __init__(self, size):
        self.size, self.heap, self.rows = size, [], {}

    def add(self, row):
        if self.size < 1 or row["molecule_id"] in self.rows:
            return
        key = int(hashlib.sha256(row["molecule_id"].encode()).hexdigest(), 16)
        entry = (-key, row["molecule_id"])
        if len(self.rows) < self.size:
            heapq.heappush(self.heap, entry)
            self.rows[row["molecule_id"]] = dict(row)
        elif key < -self.heap[0][0]:
            _, old = heapq.heapreplace(self.heap, entry)
            del self.rows[old]
            self.rows[row["molecule_id"]] = dict(row)


def init_chemistry():
    RDLogger.DisableLog("rdApp.*")


def check_chemistry_chunk(rows):
    counts, examples = Counter(), []
    for row in rows:
        counts["molecules_checked"] += 1
        problems = []
        try:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None or not mol.GetNumAtoms():
                problems.append("invalid_or_empty_smiles")
            else:
                if Chem.MolToSmiles(mol, isomericSmiles=True) != row["smiles"]:
                    problems.append("noncanonical_isomeric_smiles")
                if Chem.MolToSmiles(mol, isomericSmiles=False) != row["connectivity_smiles"]:
                    problems.append("connectivity_metadata_mismatch")
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                key = Chem.MolToSmiles(scaffold, isomericSmiles=False) or "ACYCLIC:" + row["connectivity_smiles"]
                if key != row["scaffold_key"]:
                    problems.append("scaffold_metadata_mismatch")
                if len(Chem.GetMolFrags(mol)) != 1:
                    problems.append("multiple_components_in_active_moiety_catalogue")
                if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                    problems.append("unspecified_dummy_atom")
                if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
                    counts["radical_containing_structures_reported_not_universal_legacy_gate"] += 1
                if row["heavy_atoms"] is not None and int(row["heavy_atoms"]) != mol.GetNumHeavyAtoms():
                    problems.append("heavy_atom_descriptor_mismatch")
                if row["mw"] is not None and abs(float(row["mw"]) - Descriptors.MolWt(mol)) > 0.10:
                    problems.append("mw_descriptor_mismatch_above_0_10_da")
            expected_id = "mol_" + hashlib.sha256(row["smiles"].encode()).hexdigest()[:24]
            if row["molecule_id"] != expected_id:
                problems.append("normalized_identity_id_mismatch")
        except Exception as error:
            problems.append("chemistry_exception:" + type(error).__name__)
        for problem in problems:
            counts[problem] += 1
            if len(examples) < 50:
                examples.append({"molecule_id": row["molecule_id"], "check": problem})
    return dict(counts), examples


def numeric(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def gate_violations(properties, quality):
    problems = []
    for key in ("mw", "qed", "heavy_atoms", "sa_score"):
        if not numeric(properties.get(key)):
            problems.append("missing_or_nonfinite_" + key)
    if problems:
        return problems
    checks = [
        (quality["min_mw"] <= properties["mw"] <= quality["max_mw"], "mw_gate"),
        (quality["min_heavy_atoms"] <= properties["heavy_atoms"] <= quality["max_heavy_atoms"], "heavy_atom_gate"),
        (properties["qed"] + 1e-8 >= quality["qed_min"], "qed_gate"),
        (properties["sa_score"] <= quality["max_sa"] + 1e-8, "sa_gate"),
    ]
    return [name for valid, name in checks if not valid]


def audit_new_relations(connection, sample):
    counts, categories, difficulty, verification, methods = Counter(), Counter(), Counter(), Counter(), Counter()
    history_status, failures, examples = Counter(), Counter(), []
    sql = """SELECT r.*,m.smiles AS child_smiles,m.connectivity_smiles,m.scaffold_key,
        m.mw,m.logp,m.qed,m.sa_score,m.heavy_atoms,m.training_eligible,
        j.config_json AS job_config,c.properties_json AS cached_properties_json,
        p.name AS reference_name,p.smiles AS reference_smiles,
        p.generation_eligible AS reference_eligible
        FROM relations r JOIN molecules m ON r.molecule_id=m.molecule_id
        LEFT JOIN jobs j ON r.job_id=j.job_id
        LEFT JOIN descriptor_cache c ON c.smiles=m.smiles
        LEFT JOIN references_catalogue p ON r.reference_id=p.reference_id
        WHERE r.job_id IS NOT NULL ORDER BY r.job_id,r.relation_id"""
    previous_job, config = None, None
    for row in connection.execute(sql):
        counts["job_linked_relations"] += 1
        methods[row["method"]] += 1
        categories[row["category"] or ""] += 1
        difficulty[row["difficulty"] or ""] += 1
        verification[row["verification_status"] or ""] += 1
        problems = []
        if row["method"] not in {"M", "E"}:
            problems.append("unexpected_new_generation_method")
        if not row["reference_id"] or not (row["reference_name"] or "").strip():
            problems.append("new_relation_without_named_reference")
        if row["reference_eligible"] != 1 or row["training_eligible"] != 1:
            problems.append("ineligible_new_relation_endpoint")
        if row["parent_smiles"] != row["reference_smiles"]:
            problems.append("new_relation_parent_reference_mismatch")
        if row["evidence_tier"] != "synthetic_generator_relation":
            problems.append("unexpected_new_relation_evidence_tier")
        try:
            history = json.loads(row["operator_history_json"] or "null")
            if not isinstance(history, list) or not history or not all(isinstance(item, dict) and item for item in history):
                problems.append("missing_or_empty_new_operator_history")
            elif row["method"] == "M" and not all(item.get("rule_name") and item.get("before_smiles") and item.get("after_smiles") for item in history):
                problems.append("incomplete_recorded_m_reaction_history")
            elif row["method"] == "E" and not all(item.get("operator") and item.get("input_smiles") and item.get("output_smiles") for item in history):
                problems.append("incomplete_recorded_e_producer_events")
            objective = json.loads(row["objective_json"] or "{}")
            history_status[objective.get("operator_history_status", "not_stated")] += 1
        except (ValueError, TypeError):
            problems.append("invalid_new_history_or_objective_json")
        try:
            if row["job_id"] != previous_job:
                config = json.loads(row["job_config"] or "null")
                previous_job = row["job_id"]
            if not isinstance(config, dict) or "quality" not in config or "band" not in config:
                problems.append("missing_actual_job_quality_configuration")
            else:
                quality = config["quality"]
                props = {key: row[key] for key in ("mw", "logp", "qed", "sa_score", "heavy_atoms")}
                problems.extend(gate_violations(props, quality))
                cached = json.loads(row["cached_properties_json"] or "{}")
                alerts = cached.get("structural_alerts")
                if not numeric(alerts):
                    problems.append("missing_or_nonfinite_cached_new_structural_alert_count")
                elif alerts > quality["max_alerts"]:
                    problems.append("cached_new_structural_alert_gate")
                if not numeric(row["similarity"]) or not config["band"]["sim_lo"] <= row["similarity"] <= config["band"]["sim_hi"]:
                    problems.append("stored_normalized_similarity_outside_job_band")
                sample.add({"molecule_id": row["molecule_id"], "smiles": row["child_smiles"],
                    **props, "quality": quality, "job_id": row["job_id"], "sample_scope": "new_job_linked_molecule"})
        except (ValueError, TypeError, KeyError):
            problems.append("invalid_actual_job_configuration")
        for problem in problems:
            failures[problem] += 1
            if len(examples) < 100:
                examples.append({"relation_id": row["relation_id"], "molecule_id": row["molecule_id"], "check": problem})
    return {**dict(counts), "methods": dict(methods), "categories": dict(categories),
        "difficulty": dict(difficulty), "verification_status": dict(verification),
        "history_status": dict(history_status), "failures": dict(failures), "failure_examples": examples,
        "history_interpretation": "M recorded reaction intermediates; E recorded immediate producer events, not guaranteed complete ancestry or verified final edit class."}


def descriptor_sample_check(row):
    from evolutionary_generator import sascorer, structural_alerts, _ALERT_CATALOG
    if sascorer is None or _ALERT_CATALOG is None:
        raise RuntimeError("SA scorer or structural alert catalogue unavailable for descriptor audit")
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is None or not mol.GetNumAtoms():
        return {"molecule_id": row["molecule_id"], "sample_scope": row["sample_scope"],
                "failures": ["descriptor_sample_invalid_or_empty_smiles"],
                "stored_missing_descriptors": [], "recomputed": {}}
    actual = {"mw": float(Descriptors.MolWt(mol)), "logp": float(Descriptors.MolLogP(mol)),
        "qed": float(Descriptors.qed(mol)), "sa_score": float(sascorer.calculateScore(mol)),
        "heavy_atoms": mol.GetNumHeavyAtoms(), "structural_alerts": int(structural_alerts(mol))}
    tolerances = {"mw": 0.10, "logp": 0.03, "qed": 0.001, "sa_score": 0.01, "heavy_atoms": 0}
    failures, missing = [], []
    for key, tolerance in tolerances.items():
        if row.get(key) is None:
            missing.append(key)
        elif not numeric(row[key]) or abs(row[key] - actual[key]) > tolerance:
            failures.append("recomputed_" + key + "_disagrees_with_stored")
    if row.get("quality"):
        failures.extend("recomputed_" + item for item in gate_violations(actual, row["quality"]))
        if actual["structural_alerts"] > row["quality"]["max_alerts"]:
            failures.append("recomputed_structural_alert_gate")
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            failures.append("recomputed_new_radical_gate")
    return {"molecule_id": row["molecule_id"], "sample_scope": row["sample_scope"],
            "failures": failures, "stored_missing_descriptors": missing, "recomputed": actual}


def sample_audit(connection):
    @lru_cache(maxsize=10000)
    def molecule_exists(molecule_id):
        return connection.execute("SELECT 1 FROM molecules WHERE molecule_id=?", (molecule_id,)).fetchone() is not None

    @lru_cache(maxsize=10000)
    def declared_molecule(reference_id):
        result = connection.execute("SELECT molecule_id FROM references_catalogue WHERE reference_id=?", (reference_id,)).fetchone()
        return result[0] if result else None

    counts, scenarios, tiers, failures, examples = Counter(), Counter(), Counter(), Counter(), []
    for row in connection.execute("SELECT * FROM samples ORDER BY sample_id"):
        counts["sample_records"] += 1
        scenarios[row["scenario"] or ""] += 1
        tiers[row["evidence_tier"] or ""] += 1
        problems = []
        parent_molecule = declared_molecule(row["declared_reference_id"])
        if parent_molecule is None:
            problems.append("sample_without_registered_declared_reference")
        try:
            observed = json.loads(row["observed_components_json"])
            quantities = json.loads(row["quantities_json"])
            detail = json.loads(row["metadata_json"])
            if not isinstance(observed, list) or not isinstance(detail, dict) or not isinstance(quantities, dict):
                raise ValueError("wrong JSON container")
            ids = [component.get("molecule_id") for component in observed]
            if any(not identity or not molecule_exists(identity) for identity in ids):
                problems.append("sample_unknown_or_empty_observed_molecule")
            if len(ids) != len(set(ids)):
                problems.append("sample_duplicate_component_identity")
            if detail.get("protocol") == "claim_conditioned_scenarios_v1":
                if row["evidence_tier"] != "synthetic_scenario" or detail.get("confirmed_real_case") is not False or detail.get("clinical_authenticity_or_fraud_label") is not None:
                    problems.append("synthetic_sample_claims_real_fraud_or_authenticity")
                if quantities.get("unit") != "dimensionless_relative_component_amount":
                    problems.append("synthetic_sample_unexpected_quantity_unit")
                qobs = quantities.get("observed", [])
                if sorted(q.get("molecule_id") for q in qobs) != sorted(ids):
                    problems.append("sample_quantity_component_identity_mismatch")
                if any(not numeric(q.get("relative_amount")) or q["relative_amount"] <= 0 for q in qobs):
                    problems.append("sample_nonpositive_or_invalid_relative_amount")
                if not observed:
                    if row["scenario"] != "simulated_declared_api_absence" or detail.get("declared_api_presence") != "simulated_absent" or detail.get("inactive_matrix_status") != "unknown_unspecified":
                        problems.append("empty_component_list_without_explicit_absence_unknown_matrix")
                if row["scenario"] == "synthetic_identity_consistent_control":
                    if ids != [parent_molecule] or row["composition_mismatch"] != 0 or detail.get("identity_mismatch") is not False or detail.get("quantity_mismatch") is not False:
                        problems.append("invalid_identity_consistent_control")
                if row["scenario"] == "simulated_relative_quantity_drift":
                    if ids != [parent_molecule] or detail.get("identity_mismatch") is not False or detail.get("quantity_mismatch") is not True:
                        problems.append("quantity_drift_conflated_with_identity_mismatch")
        except (ValueError, TypeError, KeyError, AttributeError):
            problems.append("invalid_sample_json_or_component_schema")
        for problem in problems:
            failures[problem] += 1
            if len(examples) < 50:
                examples.append({"sample_id": row["sample_id"], "check": problem})
    return {**dict(counts), "scenarios": dict(scenarios), "evidence_tiers": dict(tiers),
            "failures": dict(failures), "failure_examples": examples,
            "sample_rows_count_as_new_molecules": False}


def sql_audit(connection, holdout):
    counts = table_counts(connection)
    issues = []
    integrity = [row[0] for row in connection.execute("PRAGMA quick_check")]
    if integrity != ["ok"]:
        issues.append("sqlite_quick_check_failed")
    foreign = [dict(row) for row in connection.execute("PRAGMA foreign_key_check")]
    if foreign:
        issues.append("foreign_key_violations")
    unique = connection.execute("SELECT COUNT(DISTINCT molecule_id),COUNT(DISTINCT smiles),COUNT(DISTINCT connectivity_smiles),COUNT(DISTINCT scaffold_key) FROM molecules").fetchone()
    if unique[0] != counts["molecules"] or unique[1] != counts["molecules"]:
        issues.append("molecular_identity_uniqueness_failure")
    identity = sorted(set(holdout.get("blocked_connectivity_smiles", [])))
    scaffold = sorted(set(holdout.get("blocked_scaffold_keys", [])))
    clause = " OR ".join(part for part in (
        "m.connectivity_smiles IN (" + ",".join("?" for _ in identity) + ")" if identity else "",
        "m.scaffold_key IN (" + ",".join("?" for _ in scaffold) + ")" if scaffold else "") if part) or "0"
    params = identity + scaffold
    eligible_overlap = connection.execute(f"SELECT COUNT(*) FROM molecules m WHERE m.training_eligible=1 AND ({clause})", params).fetchone()[0]
    new_overlap = connection.execute(f"SELECT COUNT(DISTINCT m.molecule_id) FROM molecules m JOIN relations r ON r.molecule_id=m.molecule_id WHERE r.job_id IS NOT NULL AND ({clause})", params).fetchone()[0]
    if eligible_overlap:
        issues.append("external_holdout_structure_marked_training_eligible")
    if new_overlap:
        issues.append("new_generation_overlaps_external_holdout")
    reference_mismatch = connection.execute("""SELECT COUNT(*) FROM references_catalogue r
        LEFT JOIN molecules m ON r.molecule_id=m.molecule_id
        WHERE m.molecule_id IS NULL OR r.smiles<>m.smiles OR
        (r.generation_eligible=1 AND (r.name IS NULL OR TRIM(r.name)='' OR m.training_eligible<>1))""").fetchone()[0]
    if reference_mismatch:
        issues.append("registered_reference_structure_or_name_inconsistency")
    lineage_orphans = connection.execute("""SELECT COUNT(*) FROM molecules m
        WHERE first_source='named_reference_generation' AND NOT EXISTS
          (SELECT 1 FROM relations r WHERE r.molecule_id=m.molecule_id AND r.job_id IS NOT NULL AND r.reference_id IS NOT NULL)""").fetchone()[0]
    if lineage_orphans:
        issues.append("new_named_generation_without_registered_job_lineage")
    flags = connection.execute("SELECT SUM(is_generated),SUM(is_reference),SUM(is_generated AND is_reference),SUM(training_eligible) FROM molecules").fetchone()
    legacy = connection.execute("""SELECT COUNT(*),SUM(r.reference_id IS NULL),
        SUM(r.reference_id IS NOT NULL AND (p.name IS NULL OR TRIM(p.name)=''))
        FROM relations r LEFT JOIN references_catalogue p ON r.reference_id=p.reference_id WHERE r.job_id IS NULL""").fetchone()
    return {"table_counts": counts, "unique_molecule_ids": unique[0], "unique_isomeric_active_moiety_smiles": unique[1],
        "distinct_connectivity_smiles_ignoring_stereo": unique[2], "distinct_scaffold_keys": unique[3],
        "origin_flags": {"generated_origin": flags[0] or 0, "reference_origin": flags[1] or 0,
                         "both_origins": flags[2] or 0, "training_eligible": flags[3] or 0},
        "method_origin_counts": dict(connection.execute("SELECT method,COUNT(*) FROM molecule_methods GROUP BY method")),
        "reference_status_counts": dict(connection.execute("SELECT status,COUNT(*) FROM references_catalogue GROUP BY status")),
        "relation_counts_by_method": dict(connection.execute("SELECT method,COUNT(*) FROM relations GROUP BY method")),
        "relation_counts_by_category": dict(connection.execute("SELECT COALESCE(category,''),COUNT(*) FROM relations GROUP BY category")),
        "relation_counts_by_difficulty": dict(connection.execute("SELECT COALESCE(difficulty,''),COUNT(*) FROM relations GROUP BY difficulty")),
        "relation_counts_by_verification_status": dict(connection.execute("SELECT COALESCE(verification_status,''),COUNT(*) FROM relations GROUP BY verification_status")),
        "source_record_counts_by_source_type": dict(connection.execute("SELECT source_type,COUNT(*) FROM molecule_sources GROUP BY source_type")),
        "molecule_counts_by_first_source": dict(connection.execute("SELECT first_source,COUNT(*) FROM molecules GROUP BY first_source")),
        "jobs_by_state": dict(connection.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state")),
        "descriptor_cache_versions": [dict(row) for row in connection.execute("SELECT normalization_version,rdkit_version,COUNT(*) AS records FROM descriptor_cache GROUP BY normalization_version,rdkit_version")],
        "legacy_relation_coverage": {"relations": legacy[0], "null_reference": legacy[1] or 0,
                                     "registered_but_unnamed_reference": legacy[2] or 0},
        "external_holdout": {"training_eligible_overlap": eligible_overlap, "new_job_generation_overlap": new_overlap},
        "new_named_molecules_without_job_lineage": lineage_orphans,
        "reference_inconsistencies": reference_mismatch, "sqlite_quick_check": integrity,
        "foreign_key_violations": foreign[:100], "issues": issues}


def stream_table(connection, table, path):
    cursor = connection.execute(f"SELECT * FROM {table} ORDER BY " + {
        "references_catalogue": "reference_id", "relations": "relation_id", "samples": "sample_id",
        "molecule_sources": "molecule_id,source_path,source_id,raw_smiles", "molecule_methods": "molecule_id,method",
        "jobs": "job_id", "import_sources": "source_id", "metadata": "key"}[table])
    fields = [column[0] for column in cursor.description]
    count = 0
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        while batch := cursor.fetchmany(2000):
            writer.writerows(tuple(row) for row in batch)
            count += len(batch)
    return count


def export_and_audit(db, out, *, minimum=1_000_000, holdout_path=Path("real_world_falsification/v1/external_holdout.json"),
                     workers=4, chunk_size=2000, descriptor_sample=4096, skip_chemistry=False):
    db, out, holdout_path = Path(db), Path(out), Path(holdout_path)
    if minimum < 1 or workers < 1 or chunk_size < 1 or descriptor_sample < 0:
        raise ValueError("Minimum, workers and chunk size must be positive; sample size cannot be negative")
    if skip_chemistry and minimum >= 1_000_000:
        raise ValueError("--skip-chemistry is restricted to small fixtures, never a million-molecule release")
    if not db.is_file():
        raise FileNotFoundError(db)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Release directory already contains files; choose a new version directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + out.name + ".staging_", dir=out.parent))
    observer = open_readonly(db)
    connection = open_readonly(db)
    audit = {"created_utc": now(), "source_database": str(db.resolve()), "minimum_unique_molecules": minimum,
             "fixture_skip_chemistry": skip_chemistry, "python_version": platform.python_version(),
             "rdkit_version": rdBase.rdkitVersion, "pipeline_sha256": checksum(__file__), "passed": False}
    try:
        version_before = observer.execute("PRAGMA data_version").fetchone()[0]
        counts_before = table_counts(observer)
        files_before = database_fingerprint(db)
        if observer.execute("PRAGMA data_version").fetchone()[0] != version_before:
            raise RuntimeError("Source database is moving; stop writers before audit/export")
        audit["source_snapshot_before"] = {"table_counts": counts_before, "files": files_before,
                                           "observer_data_version": version_before}
        if counts_before["molecules"] < minimum:
            raise ValueError(f"Only {counts_before['molecules']:,} catalogue molecules; minimum is {minimum:,}. Samples/relations are not molecules.")
        connection.execute("BEGIN")
        if table_counts(connection) != counts_before:
            raise RuntimeError("Source database changed before the export snapshot began")
        holdout = json.loads(holdout_path.read_text())
        audit["holdout_file"] = str(holdout_path)
        audit["holdout_sha256"] = checksum(holdout_path)
        stored_holdout = connection.execute("SELECT value FROM metadata WHERE key='external_holdout_sha256'").fetchone()
        if stored_holdout and stored_holdout[0] != audit["holdout_sha256"]:
            raise ValueError("External holdout differs from the pinned generation/import policy")
        sql = sql_audit(connection, holdout)
        audit["sql"] = sql
        legacy_sample = SmallestHashSample(descriptor_sample // 2)
        new_sample = SmallestHashSample(descriptor_sample - descriptor_sample // 2)
        audit["new_generation"] = audit_new_relations(connection, new_sample)
        audit["samples"] = sample_audit(connection)
        chemistry_counts, chemistry_examples = Counter(), []
        molecule_export_count = 0
        pool = None if skip_chemistry else ProcessPoolExecutor(max_workers=workers, initializer=init_chemistry)
        pending = set()

        def collect(ready):
            for future in ready:
                counts, examples = future.result()
                chemistry_counts.update(counts)
                chemistry_examples.extend(examples[:max(0, 100 - len(chemistry_examples))])

        try:
            cursor = connection.execute("""SELECT m.*,
                COALESCE((SELECT GROUP_CONCAT(method,'|') FROM
                  (SELECT method FROM molecule_methods x WHERE x.molecule_id=m.molecule_id ORDER BY method)), '') AS method_tags
                FROM molecules m ORDER BY m.molecule_id""")
            fields = [column[0] for column in cursor.description]
            with (staging / "molecules.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                while chunk := cursor.fetchmany(chunk_size):
                    records = [dict(row) for row in chunk]
                    writer.writerows(records)
                    molecule_export_count += len(records)
                    for row in records:
                        if row["first_source"] != "named_reference_generation":
                            legacy_sample.add({**row, "sample_scope": "legacy_or_reference_catalogue"})
                    if pool:
                        pending.add(pool.submit(check_chemistry_chunk, records))
                        if len(pending) >= workers * 2:
                            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                            collect(ready)
                    if molecule_export_count % (chunk_size * 25) == 0:
                        print(f"Exported/queued identity audit {molecule_export_count:,}/{counts_before['molecules']:,} molecules", flush=True)
            if pending:
                ready, _ = wait(pending)
                collect(ready)
            descriptor_results = []
            if pool:
                sampled = dict(legacy_sample.rows)
                for identity, row in new_sample.rows.items():
                    if identity in sampled:
                        row = {**row, "sample_scope": "new_job_linked_molecule_and_legacy_origin"}
                    sampled[identity] = row
                rows = sorted(sampled.values(), key=lambda row: (row["sample_scope"], row["molecule_id"]))
                descriptor_results = list(pool.map(descriptor_sample_check, rows, chunksize=32))
        finally:
            if pool:
                pool.shutdown(wait=True)
        audit["chemistry"] = {"scope": "all catalogue molecular identities" if not skip_chemistry else "skipped_fixture_only",
            "counts": dict(chemistry_counts), "failure_examples": chemistry_examples,
            "normalization_not_reapplied": True,
            "policy": "Parse and canonicalize stored normalized isomeric SMILES; check connectivity, scaffold, component count, identity hash, heavy atoms and MW within0.10Da. No universal new-profile gate is imposed on legacy data."}
        descriptor_failures = Counter(problem for result in descriptor_results for problem in result["failures"])
        audit["descriptor_recomputation"] = {"scope": "deterministic bounded smallest-SHA256 identity sample; not full-catalogue descriptor validation",
            "sampled_records": len(descriptor_results), "requested_sample_limit": descriptor_sample,
            "sample_scopes": dict(Counter(r["sample_scope"] for r in descriptor_results)),
            "failures": dict(descriptor_failures),
            "failure_examples": [r for r in descriptor_results if r["failures"]][:50],
            "legacy_missing_stored_descriptors": dict(Counter(key for r in descriptor_results for key in r["stored_missing_descriptors"])),
            "tolerances": {"mw": 0.10, "logp": 0.03, "qed": 0.001, "sa_score": 0.01, "heavy_atoms": 0}}
        with (staging / "descriptor_recomputation_sample.jsonl").open("w") as stream:
            for result in descriptor_results:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
        export_counts = {"molecules.csv": molecule_export_count}
        for table, filename in EXPORTS.items():
            export_counts[filename] = stream_table(connection, table, staging / filename)
            print(f"Exported {filename}: {export_counts[filename]:,} rows", flush=True)
        errors = list(sql["issues"])
        errors.extend("new_generation:" + key for key in audit["new_generation"]["failures"])
        errors.extend("samples:" + key for key in audit["samples"]["failures"])
        errors.extend("identity:" + key for key in chemistry_counts if key not in {
            "molecules_checked", "radical_containing_structures_reported_not_universal_legacy_gate"} and chemistry_counts[key])
        errors.extend("sampled_descriptor:" + key for key in descriptor_failures)
        if not skip_chemistry and chemistry_counts["molecules_checked"] != counts_before["molecules"]:
            errors.append("incomplete_full_identity_parsing")
        if molecule_export_count != counts_before["molecules"]:
            errors.append("molecule_export_row_count_mismatch")
        for table, filename in EXPORTS.items():
            if export_counts[filename] != counts_before[table]:
                errors.append("export_count_mismatch:" + table)
        after_counts = table_counts(observer)
        after_files = database_fingerprint(db)
        after_version = observer.execute("PRAGMA data_version").fetchone()[0]
        audit["source_snapshot_after"] = {"table_counts": after_counts, "files": after_files,
                                          "observer_data_version": after_version}
        if counts_before != after_counts or files_before != after_files or version_before != after_version:
            errors.append("source_database_moved_during_audit_export")
        audit["errors"] = sorted(set(errors))
        audit["export_counts"] = export_counts
        if errors:
            raise ValueError("Bank audit failed: " + ", ".join(sorted(set(errors))))
        audit["passed"] = True
        audit["completed_utc"] = now()
        (staging / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        summary = {
            "release_status": "audited_frozen_snapshot", "created_utc": audit["completed_utc"],
            "unique_catalogue_molecules": counts_before["molecules"], "minimum": minimum,
            "minimum_reached": counts_before["molecules"] >= minimum,
            "count_basis": "unique canonical isomeric active-moiety SMILES; relations, samples, forms and conformers do not count as additional molecules",
            "distinct_connectivity_smiles_ignoring_stereo": sql["distinct_connectivity_smiles_ignoring_stereo"],
            "distinct_scaffolds": sql["distinct_scaffold_keys"], "table_counts": counts_before,
            "origin_flags": sql["origin_flags"], "method_origin_counts": sql["method_origin_counts"],
            "new_job_linked_relation_count": audit["new_generation"].get("job_linked_relations", 0),
            "legacy_relation_coverage": sql["legacy_relation_coverage"],
            "full_identity_parse_completed": not skip_chemistry,
            "descriptor_qc_sample_records": len(descriptor_results),
            "intrinsic_clinical_counterfeit_labels_assigned": False,
            "legacy_quality_status": "retained source provenance; not all legacy structures are asserted to pass the new production profile",
            "output_files": export_counts, "source_snapshot": audit["source_snapshot_after"],
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        legacy = sql["legacy_relation_coverage"]
        (staging / "README.md").write_text(f"""# Bancă moleculară consolidată: snapshot auditat

Acest export are **{counts_before['molecules']:,} molecule unice** după politica
de identitate a catalogului: SMILES izomeric canonic al formei active normalizate.
Sunt {sql['distinct_connectivity_smiles_ignoring_stereo']:,} conectivități distincte
când stereochimia este ignorată. Numărul de molecule nu include relațiile,
amestecurile, observațiile simulate sau conformerii.

`molecules.csv` păstrează structura normalizată, descriptorii, eligibilitatea
pentru antrenare și indicatorii de proveniență. `is_generated`, `is_reference`
și `method_tags` descriu originea și pot coexista pentru aceeași substanță.
Ele nu sunt etichete clinice de medicament autentic/contrafăcut.

`references.csv` păstrează identitatea declarată și statutul sursei.
`relations.csv` păstrează perechile de generare, dificultatea, metoda și istoricul
operațiilor. Sunt {audit['new_generation'].get('job_linked_relations', 0):,} relații
legate de joburi noi, auditate pentru referință cu nume, eligibilitate și istoric
nevid. Pentru E, evenimentele producătorului imediat nu demonstrează întreaga
ancestralitate și nici verificarea categoriei chimice finale. Valorile exacte de
verificare sunt păstrate și numărate în `audit.json`.

Relațiile legacy sunt raportate separat: {legacy['relations']:,} relații,
{legacy['null_reference']:,} fără referință înregistrată și
{legacy['registered_but_unnamed_reference']:,} cu referință înregistrată fără nume.
Numele și statutul clinic lipsă nu au fost inventate. Structurile legacy nu sunt
declarate automat conforme cu toate pragurile noului profil de generare.

`samples.csv` păstrează observații/compoziții cu nivelul dovezii și cantitățile
simulate. O observație nu este o moleculă suplimentară. Controalele sintetice de
identitate nu sunt produse autentice verificate, iar eticheta de neconcordanță a
compoziției nu dovedește intenția de falsificare. Absența unui API cu matrice
necunoscută nu este transformată într-un graf molecular gol.

`sources.csv`, `molecule_methods.csv`, `jobs.csv`, `import_sources.csv` și
`metadata.csv` păstrează proveniența și configurațiile. Sursele originale își
păstrează termenii de utilizare, inclusiv atribuirea ChEMBL acolo unde se aplică.

Auditul SQL verifică unicitatea, integritatea și referințele. Auditul chimic
{'a fost omis explicit doar pentru această probă de test' if skip_chemistry else 'a parcurs toate identitățile moleculare'}.
Descriptorii au fost recalculați pe un eșantion determinist de
{len(descriptor_results):,} înregistrări, separat de verificarea pragurilor stocate
pentru fiecare relație nouă. `descriptor_recomputation_sample.jsonl` păstrează
rezultatele eșantionului; această verificare nu pretinde recalcularea completă a
tuturor descriptorilor din bancă.

Identitățile/scheletele rezervate cazurilor externe sunt excluse din eligibilitatea
pentru antrenare și din generarea nouă. Cazurile reale, referințele și simulările
nu devin o singură etichetă clinică. Detectorul de contrafaceri reale necesită
validare independentă cu observații măsurate și controale corespunzătoare.

Baza SQLite a fost verificată ca snapshot nemodificat pe durata exportului,
folosind numărători, data_version și hashurile fișierelor DB/WAL înainte și după.
Hashurile exporturilor sunt în `manifest_sha256.json`.
""")
        manifest = {path.name: {"bytes": path.stat().st_size, "sha256": checksum(path)}
                    for path in sorted(staging.iterdir()) if path.is_file()}
        (staging / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if observer.execute("PRAGMA data_version").fetchone()[0] != version_before:
            raise RuntimeError("Source database changed before publishing the release")
        connection.rollback()
        if out.exists():
            out.rmdir()  # only an existing empty directory is permitted
        staging.replace(out)
        return summary
    except Exception as error:
        audit["passed"] = False
        audit["failure"] = str(error)
        audit["failed_utc"] = now()
        failure_path = out.parent / (out.name + "_failed_audit.json")
        failure_path.write_text(json.dumps(audit, indent=2) + "\n")
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        connection.close()
        observer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("molecule_bank_1m/release"))
    parser.add_argument("--minimum", type=int, default=1_000_000)
    parser.add_argument("--holdout", type=Path, default=Path("real_world_falsification/v1/external_holdout.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--descriptor-sample", type=int, default=4096)
    parser.add_argument("--skip-chemistry", action="store_true", help="Restricted to small fixtures, never a 1M release")
    args = parser.parse_args()
    summary = export_and_audit(args.db, args.out, minimum=args.minimum, holdout_path=args.holdout,
        workers=args.workers, chunk_size=args.chunk_size, descriptor_sample=args.descriptor_sample,
        skip_chemistry=args.skip_chemistry)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
