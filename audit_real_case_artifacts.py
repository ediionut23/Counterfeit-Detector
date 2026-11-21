#!/usr/bin/env python3
"""Check evidence separation, source hashes and measured-data linkage."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from rdkit import Chem


def read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path("real_world_falsification")
cases = read(root / "v1/cases.csv")
molecules = read(root / "v1/molecules.csv")
pairs = read(root / "v1/structure_pairs.csv")
assert len({r["case_id"] for r in cases}) == len(cases)
assert all(r["partition"] == "external_case_challenge_only" for r in cases)
assert all(Chem.MolFromSmiles(r["normalized_smiles"]) is not None for r in molecules)
for case in cases:
    absent = set(json.loads(case["not_detected_declared_compound_ids"]))
    detected = set(json.loads(case["detected_compound_ids"]))
    assert not absent & detected
by_id = {r["case_id"]: r for r in cases}
# Guard against transferring composition to a different product in one alert.
assert by_id["WHO2015_04_DIAZEPAM_TWO"]["single_graph_pair_eligible"] == "0"
assert not json.loads(by_id["WHO2021_03_CYTOTEC_B14660"]["not_detected_declared_compound_ids"])
assert json.loads(by_id["WHO2024_04_PROPYLENE_GLYCOL"]["not_detected_declared_compound_ids"]) == []
assert len(json.loads(by_id["NSW2024_ANTENEX_MIXTURE"]["detected_compound_ids"])) == 2
assert all(r["case_id"] != "NSW2024_ANTENEX_MIXTURE" for r in pairs)

source_manifest = json.loads((root / "source_cache/manifest.json").read_text())
source_files = 0
for record in source_manifest["records"]:
    if record["status"] in {"downloaded", "cached"}:
        assert digest(root / "source_cache" / record["file"]) == record["sha256"]
        source_files += 1

field = root / "field_spectra_doxy"
manifest = json.loads((field / "raw/download_manifest.json").read_text())
for record in manifest["files"]:
    assert digest(field / "raw" / record["name"]) == record["sha256"]
samples = read(field / "field_samples.csv")
scans = read(field / "field_scans.csv")
sample_by_id = {r["source_sample_record_id"]: r for r in samples}
axis = read(field / "spectral_axis.csv")
wavelength = [float(r["wavelength_nm"]) for r in axis]
assert all(b > a for a, b in zip(wavelength, wavelength[1:]))
channels = [r["derived_csv_column"] for r in axis]
assert len(channels) == 228 and all(name.startswith("nir_") for name in channels)
assert len({r["scan_id"] for r in scans}) == len(scans)
vector_groups = defaultdict(set)
for scan in scans:
    sample = sample_by_id[scan["source_sample_record_id"]]
    assert float(scan["hplc_assay_mg"]) == float(sample["hplc_assay_mg"])
    assert scan["conservative_batch_group_id"] == sample["conservative_batch_group_id"]
    assert scan["confirmed_fraud_status"] == "not_adjudicated_by_this_export"
    vector = [float(scan[k]) for k in channels]
    assert all(math.isfinite(value) for value in vector) and max(vector) > min(vector)
    vector_groups[tuple(round(value, 8) for value in vector)].add(scan["conservative_specimen_group_id"])
cross_group_duplicate_vectors = [sorted(groups) for groups in vector_groups.values() if len(groups) > 1]
assert not cross_group_duplicate_vectors, cross_group_duplicate_vectors
sc027 = {r["conservative_batch_group_id"] for r in samples if r["lot_as_reported"] == "SC-027"}
assert len(sc027) == 1, "Inconsistent manufacturer names must not split the same printed batch."
summary = {"checks_passed": True, "case_records": len(cases), "resolved_compounds": len(molecules),
    "single_substitution_case_records": len(pairs), "archived_official_documents_verified": source_files,
    "field_records": len(samples), "assay_linked_scans": len(scans), "spectral_channels": len(channels),
    "cross_specimen_duplicate_spectral_vectors": len(cross_group_duplicate_vectors),
    "raw_author_files_hash_verified": len(manifest["files"]),
    "scope": "Artifact integrity and evidence/measurement separation; not independent clinical validation."}
(root / "quality_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
