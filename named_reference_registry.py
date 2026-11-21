#!/usr/bin/env python3
"""Build a named ChEMBL reference registry for claim-conditioned generation.

Source status is retained verbatim enough to distinguish ChEMBL max_phase=4
drug records from phase-3 clinical candidates. It is not a product-authenticity
label or a guarantee of current approval in a particular jurisdiction.
Original pharmaceutical forms/stereochemistry remain in the source fields;
the generation SMILES is an explicitly normalized active-moiety representative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
STATUS_API = "https://www.ebi.ac.uk/chembl/api/data/status.json"
DOCS = "https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services"
STATUS_DOCS = "https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/drug-and-compound-questions"
LICENSE_DOCS = "https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions"
FIELDS = [
    "reference_id", "name", "smiles", "source", "source_id", "status", "scaffold",
    "generation_eligible", "generation_exclusion_reasons", "representative_reference_id",
    "source_parent_chembl_id", "source_pref_name", "source_molecule_type", "max_phase",
    "withdrawn_flag", "first_approval", "original_smiles", "original_standard_inchi",
    "original_standard_inchi_key", "canonical_original_smiles", "normalized_inchikey",
    "connectivity_smiles", "normalization_basis", "original_component_count",
    "unassigned_stereocenters", "mw", "heavy_atoms", "source_url", "source_request_url",
    "source_cache_file", "registry_role",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_cached(url: str, cache_path: Path, offline: bool = False) -> dict:
    if cache_path.exists():
        wrapper = json.loads(cache_path.read_text())
        if wrapper["request_url"] != url:
            raise ValueError(f"Cache URL mismatch: {cache_path}")
        response = wrapper["response"]
        if sha256(json.dumps(response, sort_keys=True).encode()) != wrapper["response_sha256"]:
            raise ValueError(f"Cached payload changed: {cache_path}")
        return wrapper
    if offline:
        raise FileNotFoundError(f"Offline registry build requires {cache_path}")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "www.ebi.ac.uk":
        raise ValueError("Registry downloader only accesses the official EMBL-EBI HTTPS API")
    last_error = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Counterfeit-Detector named-reference registry research", "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=90) as stream:
                response = json.load(stream)
            wrapper = {"request_url": url, "fetched_utc": datetime.now(timezone.utc).isoformat(),
                       "response_sha256": sha256(json.dumps(response, sort_keys=True).encode()),
                       "response": response}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(wrapper, indent=2) + "\n")
            return wrapper
        except Exception as error:
            last_error = error
            if attempt < 5:
                time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Official ChEMBL request failed: {url}: {last_error}")


def fetch_phase(phase: int, cache: Path, page_size: int, offline: bool,
                max_records: int | None = None):
    records, offset, expected = [], 0, None
    while expected is None or offset < expected:
        params = {"max_phase": phase, "molecule_type": "Small molecule",
                  "molecule_structures__isnull": "false", "limit": page_size,
                  "offset": offset, "order_by": "molecule_chembl_id"}
        url = API + "?" + urllib.parse.urlencode(params)
        path = cache / f"phase{phase}_offset{offset:06d}.json"
        wrapper = fetch_cached(url, path, offline)
        response = wrapper["response"]
        count = int(response["page_meta"]["total_count"])
        if expected is not None and count != expected:
            raise ValueError("ChEMBL total count changed between cached pages; use a clean registry directory to refresh")
        expected = count
        page = response.get("molecules", [])
        for source in page:
            if float(source.get("max_phase", -1)) != phase:
                raise ValueError("ChEMBL phase filter returned unexpected source status")
            records.append((source, url, str(path)))
        print(f"ChEMBL phase {phase}: {min(offset + len(page), count):,}/{count:,} source records", flush=True)
        if not page or max_records is not None and len(records) >= max_records:
            break
        offset += len(page)
        if not offline:
            time.sleep(0.2)
    return records[:max_records] if max_records is not None else records, expected


def normalize(raw: str):
    original = Chem.MolFromSmiles(raw)
    if original is None or not original.GetNumAtoms():
        raise ValueError("invalid source structure")
    original_canonical = Chem.MolToSmiles(original, canonical=True, isomericSmiles=True)
    fragments = Chem.GetMolFrags(original, asMols=True, sanitizeFrags=True)
    molecule = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(original))
    molecule = rdMolStandardize.Uncharger().uncharge(molecule)
    Chem.SanitizeMol(molecule)
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False) or "ACYCLIC:" + connectivity
    organic_fragments = [f for f in fragments if any(a.GetAtomicNum() == 6 for a in f.GetAtoms()) and f.GetNumHeavyAtoms() >= 6]
    return molecule, original_canonical, smiles, connectivity, scaffold, len(fragments), len(organic_fragments)


def source_name(source):
    if source.get("pref_name"):
        return str(source["pref_name"]).strip()
    synonyms = source.get("molecule_synonyms") or []
    for category in ("INN", "USAN", "BAN", "JAN", "DCF"):
        names = sorted(s.get("molecule_synonym") or s.get("synonyms") or ""
                       for s in synonyms if str(s.get("syn_type", "")).upper() == category)
        if names and names[0].strip():
            return names[0].strip()
    return ""


def process_records(source_records, holdout, min_mw: float, max_mw: float):
    blocked_identity = set(holdout.get("blocked_connectivity_smiles", []))
    blocked_scaffold = set(holdout.get("blocked_scaffold_keys", []))
    by_source = {r[0]["molecule_chembl_id"]: r[0] for r in source_records}
    if len(by_source) != len(source_records):
        raise ValueError("Source pagination repeats a ChEMBL molecule ID; refresh in a clean registry directory")
    rows = []
    for source, request_url, cache_path in source_records:
        source_id = source["molecule_chembl_id"]
        structures = source.get("molecule_structures") or {}
        raw = structures.get("canonical_smiles") or ""
        name = source_name(source)
        hierarchy = source.get("molecule_hierarchy") or {}
        parent_id = hierarchy.get("parent_chembl_id") or source_id
        phase = float(source.get("max_phase", -1))
        status = "chembl_max_phase_4_drug_record" if phase == 4 else "chembl_max_phase_3_clinical_candidate_not_approved"
        withdrawn = source.get("withdrawn_flag")
        reasons = []
        row = {key: "" for key in FIELDS}
        row.update({"reference_id": "chembl_" + source_id, "name": name,
            "source": "ChEMBL", "source_id": source_id, "status": status,
            "max_phase": phase, "withdrawn_flag": "" if withdrawn is None else int(bool(withdrawn)),
            "first_approval": source.get("first_approval") or "", "original_smiles": raw,
            "source_pref_name": source.get("pref_name") or "", "source_molecule_type": source.get("molecule_type") or "",
            "original_standard_inchi": structures.get("standard_inchi") or "",
            "original_standard_inchi_key": structures.get("standard_inchi_key") or "",
            "source_parent_chembl_id": parent_id,
            "source_url": f"https://www.ebi.ac.uk/chembl/explore/compound/{source_id}",
            "source_request_url": request_url, "source_cache_file": cache_path,
            "registry_role": "named_reference_for_synthetic_generation_not_authentic_product_label"})
        if not name:
            reasons.append("no_preferred_or_curated_generic_name")
        if withdrawn is True or withdrawn == 1:
            reasons.append("source_withdrawn_flag")
        try:
            mol, original_canonical, smiles, connectivity, scaffold, components, organic_count = normalize(raw)
            row["canonical_original_smiles"] = original_canonical
            row["original_component_count"] = components
            basis = "RDKit Cleanup + FragmentParent + Uncharger; original isomeric form retained separately"
            # A known source parent can disambiguate pharmaceutical salts and
            # combination-like disconnected representations. No parent identity
            # is guessed when the parent source record is unavailable.
            parent = by_source.get(parent_id)
            if components > 1 and parent and parent_id != source_id:
                parent_raw = (parent.get("molecule_structures") or {}).get("canonical_smiles")
                if parent_raw:
                    p_mol, _, p_smiles, p_conn, p_scaffold, p_components, _ = normalize(parent_raw)
                    if p_components == 1 and p_conn == connectivity:
                        mol, smiles, connectivity, scaffold = p_mol, p_smiles, p_conn, p_scaffold
                        basis = "Explicit ChEMBL molecule_hierarchy parent structure + active-moiety normalization; source salt retained separately"
                    elif organic_count > 1:
                        reasons.append("ambiguous_multi_organic_component_source")
                elif organic_count > 1:
                    reasons.append("ambiguous_multi_organic_component_source")
            elif organic_count > 1:
                reasons.append("ambiguous_multi_organic_component_source")
            mw = float(Descriptors.MolWt(mol))
            heavy = mol.GetNumHeavyAtoms()
            row.update({"smiles": smiles, "connectivity_smiles": connectivity, "scaffold": scaffold,
                "normalized_inchikey": Chem.MolToInchiKey(mol), "normalization_basis": basis,
                "mw": round(mw, 6), "heavy_atoms": heavy,
                "unassigned_stereocenters": sum(label == "?" for _, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True))})
            if connectivity in blocked_identity:
                reasons.append("external_case_identity_holdout")
            if scaffold in blocked_scaffold:
                reasons.append("external_case_scaffold_holdout")
            if not min_mw <= mw <= max_mw:
                reasons.append("outside_configured_seed_mw_range")
            if not 7 <= heavy <= 70:
                reasons.append("outside_configured_seed_heavy_atom_range")
            allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
            if any(atom.GetAtomicNum() not in allowed for atom in mol.GetAtoms()):
                reasons.append("outside_configured_organic_element_set")
            if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
                reasons.append("no_carbon_active_moiety")
        except Exception as error:
            reasons.append("structure_normalization_failed:" + type(error).__name__)
        row["generation_eligible"] = int(not reasons)
        row["generation_exclusion_reasons"] = ";".join(reasons)
        rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        if row["smiles"]:
            groups[row["smiles"]].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda r: (not bool(r["generation_eligible"]),
            -float(r["max_phase"]), r["source_id"] != r["source_parent_chembl_id"],
            r["original_component_count"] != 1, r["source_id"]))
        representative = ordered[0]
        for row in ordered:
            row["representative_reference_id"] = representative["reference_id"]
            if row is not representative and row["generation_eligible"]:
                row["generation_eligible"] = 0
                row["generation_exclusion_reasons"] = "duplicate_normalized_active_moiety"
    return sorted(rows, key=lambda r: (-r["max_phase"], r["reference_id"]))


def load_reference_registry(path: str | Path, eligible_only: bool = True):
    """Load registry rows, interpreting generation eligibility explicitly."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"reference_id", "name", "smiles", "source", "source_id", "status", "scaffold", "generation_eligible"}
    if rows and required - rows[0].keys():
        raise ValueError("Registry lacks required generation/provenance fields")
    return [row for row in rows if not eligible_only or row["generation_eligible"] == "1"]


def write_csv(path: Path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("molecule_bank_1m/reference_registry"))
    parser.add_argument("--holdout", type=Path, default=Path("real_world_falsification/v1/external_holdout.json"))
    parser.add_argument("--min-seeds", type=int, default=1000, help="Supplement with separately labeled phase-3 references if needed")
    parser.add_argument("--supplement-limit", type=int, default=3000)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--min-mw", type=float, default=100)
    parser.add_argument("--max-mw", type=float, default=850)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.min_seeds < 1 or args.supplement_limit < 0 or not 1 <= args.page_size <= 1000 or args.min_mw < 0 or args.max_mw <= args.min_mw:
        parser.error("Invalid registry size/property configuration")
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "raw"
    holdout = json.loads(args.holdout.read_text())
    status = fetch_cached(STATUS_API, cache / "chembl_status.json", args.offline)
    source_records, phase4_total = fetch_phase(4, cache, args.page_size, args.offline)
    rows = process_records(source_records, holdout, args.min_mw, args.max_mw)
    eligible = [r for r in rows if r["generation_eligible"]]
    phase3_total = None
    if len(eligible) < args.min_seeds and args.supplement_limit:
        extra, phase3_total = fetch_phase(3, cache, args.page_size, args.offline, args.supplement_limit)
        source_records.extend(extra)
        rows = process_records(source_records, holdout, args.min_mw, args.max_mw)
        eligible = [r for r in rows if r["generation_eligible"]]
    write_csv(args.out / "references.csv", rows)
    write_csv(args.out / "eligible_seeds.csv", eligible)
    with (args.out / "eligible_seeds.smi").open("w") as stream:
        for row in eligible:
            stream.write(row["smiles"] + "\t" + row["reference_id"] + "\n")
    rejection = Counter(reason for r in rows for reason in r["generation_exclusion_reasons"].split(";") if reason)
    summary = {
        "schema_version": 1, "python_version": platform.python_version(),
        "rdkit_version": rdBase.rdkitVersion,
        "registry_script_sha256": sha256(Path(__file__).read_bytes()),
        "source": "EMBL-EBI ChEMBL official molecule web service", "api_url": API,
        "source_documentation_url": DOCS, "status_documentation_url": STATUS_DOCS,
        "source_license_documentation_url": LICENSE_DOCS,
        "chembl_status": status["response"], "phase4_source_records_available": phase4_total,
        "phase3_source_records_available_if_queried": phase3_total,
        "source_records_retained": len(rows), "eligible_unique_active_moieties": len(eligible),
        "eligible_by_source_status": dict(Counter(r["status"] for r in eligible)),
        "eligible_unique_scaffolds": len({r["scaffold"] for r in eligible}),
        "minimum_seed_target": args.min_seeds, "minimum_seed_target_reached": len(eligible) >= args.min_seeds,
        "exclusion_reason_counts": dict(sorted(rejection.items())),
        "holdout_file": str(args.holdout), "holdout_sha256": sha256(args.holdout.read_bytes()),
        "seed_property_policy": {"mw": [args.min_mw, args.max_mw], "heavy_atoms": [7, 70],
            "qed_or_sa_gate": None, "excluded_withdrawn_flagged": True,
            "external_identity_match": "normalized connectivity ignoring stereo", "external_scaffold_match": True},
        "identity_policy": "Preserve raw pharmaceutical forms/stereo and explicit ChEMBL parent ID; generate from Cleanup/FragmentParent/Uncharger active-moiety representative, no tautomer or stereo collapse. Ambiguous multi-organic-component references excluded.",
        "status_semantics": "max_phase=4 means a ChEMBL drug record that has reached approval; it does not verify current jurisdiction-specific authorization or a genuine product. Phase-3 supplement is separately marked clinical candidate, not approved.",
        "license": "ChEMBL data are distributed under CC BY-SA 3.0; retain ChEMBL attribution and source terms.",
        "reference_csv": str(args.out / "references.csv"), "eligible_csv": str(args.out / "eligible_seeds.csv"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "manifest_sha256.json").write_text(json.dumps({str(p.relative_to(args.out)): sha256(p.read_bytes())
        for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "manifest_sha256.json"}, indent=2) + "\n")
    (args.out / "README.md").write_text("""# Named molecular reference registry

This registry preserves ChEMBL preferred/generic names, source IDs, historical
maximum clinical phase, parent hierarchy, pharmaceutical forms and source
structures. `references.csv` retains excluded and duplicate source records;
`eligible_seeds.csv` contains one eligible representative per normalized active
moiety. `eligible_seeds.smi` is a compatibility export; the CSV is the identity
source of truth and must travel with generated relations.

The official API documents max_phase=4 as the approved-drug query. The registry
calls this a source-reported drug record, not proof of a genuine product or
current approval in every jurisdiction. Any phase-3 supplement is labeled
clinical candidate and is never called approved. Withdrawn-flagged records are
retained but excluded from generation.

Raw JSON responses, request URLs, fetch dates, response hashes and the ChEMBL
release/status response are cached under `raw/`. Raw salts and stereochemistry
remain in source columns. Generation uses the explicitly documented active
moiety; ambiguous disconnected organic combinations are excluded. No QED or SA
filter is used to erase legitimate drug-like reference diversity.

The external-case identity and scaffold holdout is applied conservatively.
Excluded cases remain inspectable with reasons. `summary.json` contains exact
counts and policy; molecule and scaffold counts are different quantities.

Rebuild from cached source responses:

```bash
venv/bin/python named_reference_registry.py --offline
```

Source documentation:
https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/drug-and-compound-questions
ChEMBL data license: CC BY-SA 3.0 (attribute ChEMBL and retain its source terms).
https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions
""")
    # Include the human-readable policy in the final artifact manifest.
    manifest_path = args.out / "manifest_sha256.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["README.md"] = sha256((args.out / "README.md").read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
