#!/usr/bin/env python3
"""Register valid named source aliases and resolve legacy reference links.

The full source registry remains unchanged. Non-generation aliases, including
withdrawn/holdout records, retain their exclusion and status metadata. Naming a
chemical reference never establishes authenticity of a marketed product.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem

from scalable_bank_store import (BankStore, chemistry_init, file_sha256,
                                 normalize_identity, prepare_catalogue_row)


DISALLOWED_ASSOCIATIONS = {"ambiguous_multi_organic_component_source",
                         "no_carbon_active_moiety", "outside_configured_organic_element_set"}
ALLOWED_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}


def _metadata(value):
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {"prior_metadata_value": parsed}
    except (ValueError, TypeError):
        return {"prior_metadata_unparsed": value}


def register_reference_aliases(store, input_path, *, holdout=None, batch_size=500, relink=True):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    path = Path(input_path)
    checksum = file_sha256(path)
    own_store = not isinstance(store, BankStore)
    bank = BankStore(store) if own_store else store
    connection = bank.connection
    try:
        if holdout is None:
            p = Path("real_world_falsification/v1/external_holdout.json")
            holdout = json.loads(p.read_text()) if p.exists() else {}
        chemistry_init(holdout)
        counts = Counter()
        skipped = Counter()
        errors = []
        eligible_before = {r[0] for r in connection.execute("SELECT reference_id FROM references_catalogue WHERE generation_eligible=1")}
        eligible_input = set()
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"reference_id", "name", "smiles", "generation_eligible"}
            if required - set(reader.fieldnames or []):
                raise ValueError(f"Registry lacks required columns: {sorted(required-set(reader.fieldnames or []))}")
            for row_number, row in enumerate(reader, 1):
                counts["source_rows"] += 1
                reasons = {r for r in (row.get("generation_exclusion_reasons") or "").split(";") if r}
                if reasons & DISALLOWED_ASSOCIATIONS or any(r.startswith("structure_normalization_failed") for r in reasons):
                    for reason in reasons & DISALLOWED_ASSOCIATIONS or {"structure_normalization_failed"}:
                        skipped[reason] += 1
                    counts["skipped_source_records"] += 1
                    continue
                if not row["reference_id"].strip() or not row["name"].strip():
                    skipped["missing_named_identity"] += 1
                    counts["skipped_source_records"] += 1
                    continue
                try:
                    normalized = normalize_identity(row["smiles"])[1]
                    molecule = Chem.MolFromSmiles(normalized)
                    if not any(a.GetAtomicNum() == 6 for a in molecule.GetAtoms()):
                        raise ValueError("no_carbon_active_moiety")
                    if any(a.GetAtomicNum() not in ALLOWED_ELEMENTS for a in molecule.GetAtoms()):
                        raise ValueError("outside_configured_organic_element_set")
                    raw = row.get("original_smiles") or row["smiles"]
                    try:
                        original_matches = normalize_identity(raw)[1] == normalized
                    except ValueError:
                        original_matches = False
                    prepared = prepare_catalogue_row({**row, "smiles": raw if original_matches else normalized,
                        "label": "0", "molecule_id": row["reference_id"],
                        "molecule_source": "named_reference_alias_registry"})
                    if "error" in prepared:
                        raise ValueError(prepared["error"])
                    if prepared["smiles"] != normalized:
                        raise ValueError("source_alias_normalization_disagrees")
                    prepared["source_metadata"] = {"reference_id": row["reference_id"],
                        "registry_sha256": checksum, "original_form_preserved_in_reference_metadata": True,
                        "original_form_associated_to_molecule": original_matches,
                        "not_product_authenticity_evidence": True}
                    current = connection.execute("SELECT * FROM references_catalogue WHERE reference_id=?", (row["reference_id"],)).fetchone()
                    if current is not None and current["molecule_id"] != prepared["molecule_id"]:
                        raise ValueError("existing_reference_id_has_different_normalized_chemical_identity")
                    counts["new_reference_molecules"] += bank.insert_molecule(prepared, str(path))
                    requested_eligible = str(row["generation_eligible"]).strip().lower() in {"1", "1.0", "true", "yes"}
                    # Source exclusions remain exclusions even when they were
                    # omitted from a stale boolean column.
                    incoming_eligible = int(requested_eligible and not reasons and prepared["training_eligible"])
                    if incoming_eligible:
                        eligible_input.add(row["reference_id"])
                    source_metadata = {"registry_path": str(path), "registry_sha256": checksum,
                        "source_record": row, "original_form_associated_to_molecule": original_matches,
                        "name_semantics": "named chemical reference, not authenticated marketed product"}
                    if current is None:
                        connection.execute("INSERT INTO references_catalogue VALUES(?,?,?,?,?,?,?,?,?)",
                            (row["reference_id"], row["name"], normalized, prepared["molecule_id"],
                             row.get("source", ""), row.get("source_id", ""), row.get("status", ""),
                             incoming_eligible, json.dumps({"full_alias_registry_record": source_metadata}, sort_keys=True)))
                        counts["references_added"] += 1
                    else:
                        metadata = _metadata(current["metadata_json"])
                        metadata["full_alias_registry_record"] = source_metadata
                        merged = (current["name"] or row["name"], current["source"] or row.get("source", ""),
                                  current["source_id"] or row.get("source_id", ""), current["status"] or row.get("status", ""),
                                  max(int(current["generation_eligible"] or 0), incoming_eligible),
                                  json.dumps(metadata, sort_keys=True))
                        old = (current["name"], current["source"], current["source_id"], current["status"],
                               int(current["generation_eligible"] or 0), current["metadata_json"])
                        if merged != old:
                            connection.execute("""UPDATE references_catalogue SET name=?,source=?,source_id=?,status=?,
                                generation_eligible=?,metadata_json=? WHERE reference_id=?""", (*merged, row["reference_id"]))
                            counts["references_updated"] += 1
                        else:
                            counts["references_unchanged"] += 1
                        if current["generation_eligible"] and not incoming_eligible:
                            counts["existing_eligible_preserved"] += 1
                    counts["associated_source_records"] += 1
                except Exception as error:
                    counts["errors"] += 1
                    errors.append({"row_number": row_number, "reference_id": row.get("reference_id", ""),
                                   "error": str(error), "source_record": row})
                if row_number % batch_size == 0:
                    connection.commit()
        connection.commit()

        if relink:
            named = defaultdict(list)
            for row in connection.execute("""SELECT r.reference_id,r.name,r.generation_eligible,m.smiles
                    FROM references_catalogue r JOIN molecules m ON m.molecule_id=r.molecule_id
                    WHERE r.name IS NOT NULL AND r.name<>''"""):
                named[row["smiles"]].append(dict(row))
            for aliases in named.values():
                aliases.sort(key=lambda r: (-int(r["generation_eligible"] or 0), r["reference_id"]))
            cursor = connection.execute("""SELECT rel.relation_id,rel.reference_id,rel.parent_smiles
                FROM relations rel LEFT JOIN references_catalogue ref ON ref.reference_id=rel.reference_id
                WHERE rel.evidence_tier='legacy_synthetic_generator_relation'
                  AND (ref.name IS NULL OR ref.name='' OR ref.status='unresolved_public_reference')
                ORDER BY rel.relation_id""")
            while batch := cursor.fetchmany(batch_size):
                with connection:
                    for relation in batch:
                        counts["unresolved_legacy_relations_scanned"] += 1
                        matches = named.get(relation["parent_smiles"])
                        if not matches:
                            continue
                        new_id = matches[0]["reference_id"]
                        value = connection.execute("SELECT objective_json FROM relations WHERE relation_id=?", (relation["relation_id"],)).fetchone()[0]
                        objective = _metadata(value)
                        if not isinstance(objective.get("reference_alias_link_history", []), list):
                            objective["prior_reference_alias_history_unparsed"] = objective["reference_alias_link_history"]
                            objective["reference_alias_link_history"] = []
                        objective.setdefault("reference_alias_link_history", []).append({
                            "prior_reference_id": relation["reference_id"], "named_reference_id": new_id,
                            "basis": "exact_normalized_parent_molecular_identity", "registry_path": str(path),
                            "registry_sha256": checksum, "not_product_authenticity_evidence": True})
                        objective["matching_named_reference_ids"] = sorted({r["reference_id"] for r in matches})
                        connection.execute("UPDATE relations SET reference_id=?,objective_json=? WHERE relation_id=?",
                            (new_id, json.dumps(objective, sort_keys=True), relation["relation_id"]))
                        counts["legacy_relations_relinked"] += 1

        eligible_after = {r[0] for r in connection.execute("SELECT reference_id FROM references_catalogue WHERE generation_eligible=1")}
        if not eligible_after - eligible_before <= eligible_input:
            raise ValueError("Alias registration expanded eligibility beyond explicit eligible source IDs")
        summary = {"input": str(path), "source_sha256": checksum, **dict(counts),
            "skipped_reason_counts": dict(skipped), "eligible_reference_ids_before": len(eligible_before),
            "eligible_reference_ids_after": len(eligible_after),
            "eligibility_policy": "Source exclusions remain zero; existing eligible references are never downgraded; full CSV unchanged.",
            "old_blank_reference_records_retained": True,
            "new_molecules_are_known_reference_alias_identities_not_generated_analogues": True,
            "name_semantics": "Source chemical names/status do not establish product authenticity.", "errors_detail": errors}
        with connection:
            connection.execute("INSERT INTO events(kind,payload_json) VALUES('named_reference_alias_registration',?)",
                               (json.dumps(summary, sort_keys=True),))
        output = bank.path.parent / "reference_alias_registration_summary.json"
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: value for key, value in summary.items() if key != "errors_detail"}, indent=2), flush=True)
        return summary
    finally:
        if own_store:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("molecule_bank_1m/bank.sqlite"))
    parser.add_argument("--input", type=Path, default=Path("molecule_bank_1m/reference_registry/references.csv"))
    parser.add_argument("--holdout", type=Path, default=Path("real_world_falsification/v1/external_holdout.json"))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-relink", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    holdout = json.loads(args.holdout.read_text()) if args.holdout.exists() else {}
    register_reference_aliases(args.db, args.input, holdout=holdout, batch_size=args.batch_size,
                               relink=not args.skip_relink)


if __name__ == "__main__":
    main()
