#!/usr/bin/env python3
"""Recover legacy synthetic parent context into the normalized scalable bank.

Children must already exist in the chemical catalogue. Missing parents may be
inserted as public-reference-origin chemicals, without inventing a drug name
or product-authenticity label. All source annotations are retained verbatim
inside objective_json. Imports commit in chunks and are safe to resume/repeat.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from scalable_bank_store import (BankStore, chemistry_init, file_sha256,
                                 normalize_identity, prepare_catalogue_row, stable_hash)


def infer_method(row, path, override=None):
    explicit = (row.get("method") or row.get("generation_method") or "").strip()
    if explicit and explicit not in {"M", "E"}:
        raise ValueError(f"unsupported_explicit_method:{explicit}")
    if override and explicit and explicit != override:
        raise ValueError(f"method_override_conflicts_with_source:{explicit}->{override}")
    if explicit or override:
        return explicit or override
    protocol = (row.get("generation_protocol") or "").lower()
    if ("genetic" in protocol or "evolution" in protocol or
            (row.get("category") or "").startswith("evolved_") or "version_ga" in Path(path).parts):
        return "E"
    if row.get("rule_name") or "version_m" in Path(path).parts:
        return "M"
    raise ValueError("method_unknown_use_explicit_method")


class MolecularLookup:
    """Bounded successful-result cache; missing children are never cached."""
    def __init__(self, connection, maximum=50000):
        self.connection, self.maximum, self.cache = connection, maximum, OrderedDict()

    def find(self, raw):
        if raw in self.cache:
            self.cache.move_to_end(raw)
            return self.cache[raw]
        found = self.connection.execute("""SELECT DISTINCT m.molecule_id,m.smiles,m.training_eligible
            FROM molecule_sources s JOIN molecules m ON m.molecule_id=s.molecule_id
            WHERE s.raw_smiles=? LIMIT 2""", (raw,)).fetchall()
        if len(found) > 1:
            raise ValueError("ambiguous_raw_structure_lookup")
        result = found[0] if found else None
        if result is None:
            normalized = normalize_identity(raw)[1]
            result = self.connection.execute("SELECT molecule_id,smiles,training_eligible FROM molecules WHERE smiles=?",
                                             (normalized,)).fetchone()
        if result is not None:
            self.cache[raw] = dict(result)
            if len(self.cache) > self.maximum:
                self.cache.popitem(last=False)
            return self.cache[raw]
        return None


def import_legacy_relations(store, input_path, *, method=None, chunk_size=1000,
                            holdout=None, retry_errors=False):
    """Import one source; accepts an existing BankStore or a SQLite path.

    --method supplies missing labels but cannot silently overwrite an explicit
    different generator. Named-reference matching is a snapshot at import start;
    run after registering the named reference catalogue and importing children.
    """
    if method not in {None, "M", "E"} or chunk_size < 1:
        raise ValueError("method must be M/E or None; chunk_size must be positive")
    path = Path(input_path)
    checksum = file_sha256(path)
    own_store = not isinstance(store, BankStore)
    bank = BankStore(store) if own_store else store
    connection = bank.connection
    source_id = "relation_import_" + stable_hash(str(path.resolve()) + "|" + checksum + "|" + (method or "auto"))
    ledger_path = "relation_import:" + str(path.resolve()) + "|method=" + (method or "auto")
    stats_key = "relation_import_stats:" + source_id
    try:
        if holdout is None:
            default_holdout = Path("real_world_falsification/v1/external_holdout.json")
            holdout = json.loads(default_holdout.read_text()) if default_holdout.exists() else {}
        chemistry_init(holdout)
        prior = connection.execute("SELECT * FROM import_sources WHERE path=?", (ledger_path,)).fetchone()
        if prior and prior["sha256"] != checksum:
            raise ValueError("Legacy relation source changed; use an explicit new source path/version")
        saved = connection.execute("SELECT value FROM metadata WHERE key=?", (stats_key,)).fetchone()
        counts = Counter(json.loads(saved[0])["counts"]) if saved else Counter()
        attempt = json.loads(saved[0]).get("attempt", 1) if saved else 1
        if saved and json.loads(saved[0])["chunk_size"] != chunk_size:
            raise ValueError("Cannot resume with different chunk_size")
        if prior and prior["state"] in {"complete", "complete_with_errors"} and not retry_errors:
            return {"source_id": source_id, "cached": True, "state": prior["state"], "import_attempt": attempt, **dict(counts)}
        if retry_errors and prior:
            connection.execute("INSERT INTO events(kind,payload_json) VALUES('legacy_relation_retry',?)",
                (json.dumps({"source_id": source_id, "prior_attempt": attempt, "prior_counts": dict(counts)}, sort_keys=True),))
            attempt += 1
            connection.execute("DELETE FROM import_chunks WHERE source_id=?", (source_id,))
            connection.execute("UPDATE import_sources SET state='running',rows_processed=0,errors=0 WHERE source_id=?", (source_id,))
            counts = Counter()
        connection.execute("INSERT OR IGNORE INTO import_sources(source_id,path,sha256,state) VALUES(?,?,?,'running')",
                           (source_id, ledger_path, checksum))
        connection.commit()
        completed = {r[0] for r in connection.execute("SELECT chunk_no FROM import_chunks WHERE source_id=?", (source_id,))}
        lookup = MolecularLookup(connection)
        named = defaultdict(list)
        for reference in connection.execute("""SELECT r.reference_id,r.name,r.smiles,r.generation_eligible,
                m.smiles AS normalized_smiles FROM references_catalogue r
                JOIN molecules m ON m.molecule_id=r.molecule_id
                WHERE r.name IS NOT NULL AND r.name<>''"""):
            named[reference["normalized_smiles"]].append(dict(reference))
        for references in named.values():
            references.sort(key=lambda r: (-int(r["generation_eligible"] or 0), r["reference_id"]))
        reference_cache = {}

        def reference_for(parent_raw, source_row_id):
            parent = lookup.find(parent_raw)
            added = 0
            if parent is None:
                prepared = prepare_catalogue_row({"smiles": parent_raw, "label": "0",
                    "molecule_source": "legacy_parent_reference", "source_id": source_row_id})
                if "error" in prepared:
                    raise ValueError("invalid_parent:" + prepared["error"])
                added = bank.insert_molecule(prepared, str(path))
                parent = lookup.find(parent_raw)
                if parent is None:
                    raise ValueError("parent_insertion_failed")
            canonical = parent["smiles"]
            if canonical in reference_cache:
                return reference_cache[canonical], added, 0
            if canonical in named:
                result = (named[canonical][0]["reference_id"], canonical,
                          [r["reference_id"] for r in named[canonical]])
                reference_cache[canonical] = result
                return result, added, 0
            reference_id = "legacy_reference_" + stable_hash(canonical)
            before = connection.total_changes
            connection.execute("""INSERT OR IGNORE INTO references_catalogue
                (reference_id,name,smiles,molecule_id,source,source_id,status,generation_eligible,metadata_json)
                VALUES(?,'',?,?,?,?,'unresolved_public_reference',0,?)""",
                (reference_id, canonical, parent["molecule_id"], "legacy_generator_parent", source_row_id,
                 json.dumps({"first_source_path": str(path), "identity_name_unresolved": True,
                             "not_an_approved_api_or_authenticated_product_claim": True}, sort_keys=True)))
            reference_added = int(connection.total_changes > before)
            result = (reference_id, canonical, [])
            reference_cache[canonical] = result
            return result, added, reference_added

        def process(bucket, chunk_no):
            if chunk_no in completed:
                return
            delta = Counter(rows_processed=len(bucket))
            with connection:
                for row_number, row in bucket:
                    source_row_id = row.get("relation_id") or row.get("source_relation_id") or row.get("molecule_id") or ""
                    try:
                        child_raw = row.get("counterfeit_smiles") or row.get("smiles") or ""
                        parent_raw = row.get("parent_smiles") or ""
                        if not child_raw:
                            raise ValueError("missing_child_structure")
                        child = lookup.find(child_raw)
                        if child is None:
                            delta["missing_children"] += 1
                            raise ValueError("child_not_in_catalogue")
                        if not parent_raw:
                            raise ValueError("missing_parent_structure")
                        source_method = infer_method(row, path, method)
                        (reference_id, canonical_parent, matches), added, reference_added = reference_for(parent_raw, source_row_id)
                        delta["parent_molecules_added"] += added
                        delta["references_added"] += reference_added
                        category = (row.get("category") or "other").removeprefix("evolved_")
                        subcategory = row.get("subcategory") or row.get("rule_name") or "legacy_unspecified"
                        annotation = {"source_relation_id": source_row_id, "source_path": str(path),
                            "source_sha256": checksum, "source_annotations": row,
                            "source_edit_mask_coordinate_system": "raw source child SMILES atom order",
                            "source_edit_mask_remapped_to_normalized_structure": False,
                            "verification_status_scope": "source annotation preserved; normalized pair not reverified",
                            "matching_named_reference_ids": matches,
                            "operator_history_status": row.get("operator_history_status") or "legacy_history_unavailable",
                            "chemical_relation_key": stable_hash("|".join((canonical_parent, child["smiles"], source_method, category, subcategory)))}
                        history = row.get("operator_history_json") or "[]"
                        json.loads(history)  # reject malformed provenance instead of silently dropping it
                        identity = {"source": source_id, "parent": canonical_parent, "child": child["smiles"],
                                    "method": source_method, "category": category, "subcategory": subcategory,
                                    "source_relation_id": source_row_id, "annotations": row}
                        relation_id = "legacy_relation_" + stable_hash(json.dumps(identity, sort_keys=True))
                        similarity = row.get("similarity_to_parent") or None
                        if similarity is not None:
                            similarity = float(similarity)
                        verification = "legacy_source_annotation_not_reverified"
                        connection.execute("UPDATE molecules SET is_generated=1 WHERE molecule_id=?", (child["molecule_id"],))
                        connection.execute("INSERT OR IGNORE INTO molecule_methods VALUES(?,?)", (child["molecule_id"], source_method))
                        connection.execute("INSERT OR IGNORE INTO molecule_sources VALUES(?,?,?,?,?,?)",
                            (child["molecule_id"], str(path), source_row_id or relation_id, child_raw,
                             "legacy_synthetic_generator_relation", json.dumps({"source_relation_id": source_row_id,
                              "parent_smiles": canonical_parent, "category": category, "subcategory": subcategory}, sort_keys=True)))
                        before = connection.total_changes
                        connection.execute("""INSERT OR IGNORE INTO relations
                            (relation_id,molecule_id,reference_id,parent_smiles,method,category,subcategory,scenario,difficulty,
                             protocol,operator_history_json,verification_status,evidence_tier,similarity,objective_json,source_id,job_id)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
                            (relation_id, child["molecule_id"], reference_id, canonical_parent, source_method, category, subcategory,
                             row.get("scenario") if (row.get("scenario") or "").startswith("synthetic_") else "synthetic_structural_analogue",
                             row.get("difficulty") or "legacy_unspecified", row.get("generation_protocol") or "legacy_generator_import",
                             history, verification, "legacy_synthetic_generator_relation", similarity,
                             json.dumps(annotation, sort_keys=True), source_row_id))
                        delta["relations_added" if connection.total_changes > before else "relations_already_present"] += 1
                        delta["rows_imported"] += 1
                        delta["named_reference_rows" if matches else "unresolved_reference_rows"] += 1
                    except Exception as error:
                        delta["errors"] += 1
                        connection.execute("INSERT INTO events(kind,payload_json) VALUES(?,?)",
                            (f"legacy_relation_error:{source_id}:{attempt}", json.dumps({"source_id": source_id, "import_attempt": attempt,
                             "source_path": str(path), "row_number": row_number, "error": str(error),
                             "source_relation_id": source_row_id, "row": row}, sort_keys=True)))
                counts.update(delta)
                connection.execute("INSERT INTO import_chunks VALUES(?,?,?)", (source_id, chunk_no, len(bucket)))
                connection.execute("UPDATE import_sources SET rows_processed=rows_processed+?,new_molecules=new_molecules+?,errors=errors+? WHERE source_id=?",
                    (len(bucket), delta["parent_molecules_added"], delta["errors"], source_id))
                connection.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    (stats_key, json.dumps({"chunk_size": chunk_size, "attempt": attempt, "counts": dict(counts)}, sort_keys=True)))
            if (chunk_no + 1) % 10 == 0:
                print(f"Legacy relations {path.name}: processed={counts['rows_processed']:,} added={counts['relations_added']:,} missing_children={counts['missing_children']:,} parents_added={counts['parent_molecules_added']:,}", flush=True)

        with path.open(newline="") as stream:
            bucket, chunk_no = [], 0
            for row_number, row in enumerate(csv.DictReader(stream), 1):
                bucket.append((row_number, row))
                if len(bucket) == chunk_size:
                    process(bucket, chunk_no)
                    bucket, chunk_no = [], chunk_no + 1
            if bucket:
                process(bucket, chunk_no)
        state = "complete_with_errors" if counts["errors"] else "complete"
        connection.execute("UPDATE import_sources SET state=? WHERE source_id=?", (state, source_id))
        connection.commit()
        errors_path = bank.path.parent / f"{source_id}_attempt_{attempt}_errors.jsonl"
        with errors_path.open("w") as stream:
            for error in connection.execute("SELECT payload_json FROM events WHERE kind=? ORDER BY event_id", (f"legacy_relation_error:{source_id}:{attempt}",)):
                stream.write(error[0] + "\n")
        result = {"source_id": source_id, "input": str(path), "state": state, "import_attempt": attempt, **dict(counts),
                  "parent_molecules_added_all_attempts": connection.execute("SELECT new_molecules FROM import_sources WHERE source_id=?", (source_id,)).fetchone()[0],
                  "errors_file": str(errors_path), "new_molecule_count_is_parent_recovery_only": True}
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        if own_store:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("molecule_bank_1m/bank.sqlite"))
    parser.add_argument("--input", type=Path, default=Path("final_dataset_400k/relations.csv"))
    parser.add_argument("--method", choices=("M", "E"), default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--holdout", type=Path, default=Path("real_world_falsification/v1/external_holdout.json"))
    parser.add_argument("--retry-errors", action="store_true", help="Reprocess all chunks idempotently after missing children have been catalogued")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    holdout = json.loads(args.holdout.read_text()) if args.holdout.exists() else {}
    result = import_legacy_relations(args.db, args.input, method=args.method, chunk_size=args.chunk_size,
                                    holdout=holdout, retry_errors=args.retry_errors)
    if result.get("cached"):
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
