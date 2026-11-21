#!/usr/bin/env python3
"""Fetch measured doxycycline field spectra and assay metadata from the authors.

Only an explicit allowlist of CSV/XLSX/PDF data, metadata and the MIT license is
downloaded from the original author's public GitHub repository. No author code
is downloaded or executed. A commit pin, source URLs and SHA-256 hashes preserve
provenance. Downloads are limited to <20 MB in total and <10 MB per response.

Reference: Awotunde et al., Analytical Methods (2024), DOI 10.1039/D4AY00001C.
Field-collected HPLC-out-of-specification products are not automatically
confirmed fraudulent products; lab-formulated calibration examples are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile


REPOSITORY = "aawotund/Mitigating-gelatin-capsule-impact-on-detection-of-substandard-and-falsified-pharmaceuticals-with-NIR"
REPOSITORY_URL = "https://github.com/" + REPOSITORY
PAPER_URL = "https://pubs.rsc.org/en/content/articlehtml/2024/ay/d4ay00001c"
ALLOWLIST = [
    "external_set_doxy.csv", "external_set_doxy_conc.csv",
    "branded_doxy_samples_andpythonperformance.xlsx",
    "Capsule NIR metadata final.xlsx", "ALL_EXTERNAL_DE_CAPSULES.csv",
    "ALL_EXTERNAL_DE_CAPSULES_cap.csv", "Liberia Doxycycline Final Report .pdf", "LICENSE",
]
MAX_RESPONSE_BYTES = 9_000_000
MAX_TOTAL_BYTES = 19_000_000
XML_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Downloader:
    def __init__(self):
        self.total = 0

    def get(self, url: str) -> bytes:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.netloc not in {"api.github.com", "raw.githubusercontent.com"}:
            raise ValueError(f"Unexpected download source: {url}")
        request = urllib.request.Request(url, headers={
            "User-Agent": "Counterfeit-Detector measured-data provenance audit",
            "Accept": "application/vnd.github+json" if parsed.netloc == "api.github.com" else "*/*",
        })
        with urllib.request.urlopen(request, timeout=45) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) >= MAX_RESPONSE_BYTES:
                raise ValueError(f"Response too large: {url}")
            payload = bytearray()
            while chunk := response.read(65536):
                payload.extend(chunk)
                self.total += len(chunk)
                if len(payload) >= MAX_RESPONSE_BYTES or self.total >= MAX_TOTAL_BYTES:
                    raise ValueError("Download budget exceeded; no partial data will be saved")
        return bytes(payload)

    def json(self, url: str):
        return json.loads(self.get(url))


def fetch_raw(out: Path, offline: bool, ref: str | None) -> dict:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("repository") != REPOSITORY:
            raise ValueError("Existing download manifest names a different repository")
        if ref and ref != manifest["commit_sha"]:
            raise ValueError("Use a new output directory to fetch a different source commit")
        for record in manifest["files"]:
            path = raw / record["name"]
            if not path.exists() or digest(path.read_bytes()) != record["sha256"]:
                raise ValueError(f"Missing or changed raw file: {path}")
        return manifest
    if offline:
        raise ValueError("--offline needs a complete raw/download_manifest.json")

    downloader = Downloader()
    chosen_ref = urllib.parse.quote(ref or "main", safe="")
    commit = downloader.json(f"https://api.github.com/repos/{REPOSITORY}/commits/{chosen_ref}")
    commit_sha = commit["sha"]
    root_listing = downloader.json(f"https://api.github.com/repos/{REPOSITORY}/contents?ref={commit_sha}")
    by_name = {record["name"]: record for record in root_listing}
    records = []
    expected_total = sum(int(by_name[name]["size"]) for name in ALLOWLIST)
    if expected_total >= MAX_TOTAL_BYTES or any(int(by_name[name]["size"]) >= MAX_RESPONSE_BYTES for name in ALLOWLIST):
        raise ValueError("Selected source files exceed the strict download budget")
    for name in ALLOWLIST:
        record = by_name[name]
        if record["type"] != "file":
            raise ValueError(f"Selected source is not a regular file: {name}")
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit_sha}/" + urllib.parse.quote(name)
        data = downloader.get(url)
        if len(data) != int(record["size"]):
            raise ValueError(f"File size differs from pinned manifest: {name}")
        path = raw / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to replace existing different raw data: {path}")
        path.write_bytes(data)
        records.append({"name": name, "bytes": len(data), "sha256": digest(data),
                        "github_blob_sha": record["sha"], "download_url": url})
        print(f"Downloaded {name}: {len(data):,} bytes", flush=True)
    manifest = {
        "repository": REPOSITORY, "repository_url": REPOSITORY_URL,
        "commit_sha": commit_sha, "files": records,
        "data_bytes": sum(record["bytes"] for record in records),
        "network_payload_bytes_including_github_metadata": downloader.total,
        "strict_max_response_bytes": MAX_RESPONSE_BYTES, "strict_max_total_bytes": MAX_TOTAL_BYTES,
        "author_code_downloaded_or_executed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def sheet_rows(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read cells/cached numeric values with stdlib; never evaluate Excel formulas."""
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in cell.findall(".//m:t", XML_NS))
                       for cell in root.findall("m:si", XML_NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("m:sheets/m:sheet", XML_NS)
        sheet = next((s for s in sheets if s.get("name") == sheet_name), None)
        if sheet is None:
            raise ValueError(f"Missing worksheet {sheet_name!r} in {path.name}")
        rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels if r.get("Id") == rid)
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        root = ET.fromstring(archive.read(target))
        rows = []
        for row in root.findall("m:sheetData/m:row", XML_NS):
            values = {"__row__": row.get("r", "")}
            for cell in row.findall("m:c", XML_NS):
                value_node = cell.find("m:v", XML_NS)
                value = value_node.text if value_node is not None else ""
                if cell.get("t") == "s" and value:
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//m:t", XML_NS))
                if value:
                    values[re.sub(r"\d+", "", cell.get("r", ""))] = value
            rows.append(values)
        return rows


def csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.reader(stream))


def write_table(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def column_number(column: str) -> int:
    value = 0
    for char in column:
        value = value * 26 + ord(char) - ord("A") + 1
    return value


def finite_number(value: str, description: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {description}")
    return number


def parse_field_data(out: Path, manifest: dict) -> dict:
    raw = out / "raw"
    metadata_filename = "Capsule NIR metadata final.xlsx"
    metadata_rows = sheet_rows(raw / metadata_filename, "Metadata")
    header = metadata_rows[0]
    if header.get("A") != "NDID number" or header.get("H") != "doxycyline assay (mg)":
        raise ValueError("Unexpected source Metadata worksheet schema")
    columns = sorted((key for key in header if key != "__row__" and column_number(key) >= 12), key=column_number)
    axis = [finite_number(header[key], "spectral wavelength") for key in columns]
    if len(axis) != 228 or any(a >= b for a, b in zip(axis, axis[1:])):
        raise ValueError("Expected 228 strictly increasing source wavelength channels")
    intensity_fields = [f"nir_{wavelength:.10g}_nm" for wavelength in axis]

    # Author workbook explicitly calls these five records Poor_Samples. Other
    # records are not automatically assigned an author pass/authenticity label.
    secondary_filename = "branded_doxy_samples_andpythonperformance.xlsx"
    poor_ids = {r["A"] for r in sheet_rows(raw / secondary_filename, "Poor_Samples")[1:] if r.get("A")}
    doses = {}
    for name in ("Branded_doxy", "Branded_2", "Poor_Samples"):
        for row in sheet_rows(raw / secondary_filename, name)[1:]:
            if not row.get("A"):
                continue
            if row.get("G") != "100":
                raise ValueError(f"Unexpected nominal dose for {row['A']}")
            dose = finite_number(row["G"], "nominal dose")
            if row["A"] in doses and doses[row["A"]] != dose:
                raise ValueError("Conflicting nominal dose metadata")
            doses[row["A"]] = dose

    replicate_filename = "ALL_EXTERNAL_DE_CAPSULES.csv"
    replicate_rows = csv_rows(raw / replicate_filename)
    replicate_axis = [float(value) for value in replicate_rows[0][5:]]
    if len(replicate_axis) != len(axis) or max(abs(a-b) for a,b in zip(axis, replicate_axis)) > 1e-6:
        raise ValueError("Replicate and metadata spectral axes disagree")
    replicates = defaultdict(list)
    for line, row in enumerate(replicate_rows[1:], 2):
        if len(row) != 5 + len(axis):
            raise ValueError("Replicate CSV has a malformed row")
        values = [finite_number(v, "NIR intensity") for v in row[5:]]
        replicates[row[0]].append((line, values))

    samples, scans, matched_raw_lines, group_consistency = [], [], set(), {}
    grouping_members = defaultdict(list)
    for row in metadata_rows[1:]:
        if not row.get("A"):
            continue
        source_id = row["A"].strip()
        intensity = [finite_number(row[key], "embedded NIR intensity") for key in columns]
        assay = finite_number(row["H"], "HPLC assay mg")
        mass = finite_number(row["G"], "total pill mass mg")
        dose = doses.get(source_id, 100.0)
        dose_source = secondary_filename if source_id in doses else "article section 2.2: all commercial products nominal 100 mg"
        base_id = re.sub(r"(?:bw|cg)$", "", source_id.lower().replace("-", "_"))
        group_id = "source_base_" + base_id
        grouping_members[group_id].append(source_id)
        signature = (row["C"].strip().lower(), row["D"].strip().lower(), row["E"].strip(), assay, mass, dose)
        if group_id in group_consistency and signature != group_consistency[group_id]:
            raise ValueError(f"Conservative source-ID aliases have conflicting assay metadata: {source_id}")
        group_consistency[group_id] = signature
        lot_key = "|".join(signature[:3])
        batch_id = "source_lot_" + digest(lot_key.encode())[:12]
        # Source metadata assigns SC-027 to two different manufacturers. Group
        # the same nominal API and printed lot conservatively to prevent a
        # source inconsistency from placing related records in different folds.
        conservative_lot = re.sub(r"[^a-z0-9]", "", row["E"].lower())
        conservative_batch_key = "doxycycline|" + (conservative_lot or group_id)
        conservative_batch_id = "conservative_batch_" + digest(conservative_batch_key.encode())[:12]
        assay_status = "below_90_mg" if assay < 90 else "above_120_mg" if assay > 120 else "within_90_to_120_mg"
        source_replicates = replicates.get(source_id, [])
        matching_lines = [line for line, values in source_replicates
                          if max(abs(a-b) for a,b in zip(intensity, values)) <= 1e-8]
        if source_replicates and not matching_lines:
            raise ValueError(f"Embedded representative spectrum does not match a source replicate for {source_id}")
        shared = {
            "source_sample_record_id": source_id, "conservative_specimen_group_id": group_id,
            "source_lot_group_id": batch_id, "conservative_batch_group_id": conservative_batch_id,
            "capsule_type": row.get("B", ""),
            "brand_as_reported": row["C"].strip(), "manufacturer_as_reported": row["D"].strip(),
            "lot_as_reported": row["E"].strip(), "expiry_excel_serial_as_reported": row.get("F", ""),
            "nominal_dose_mg": dose, "nominal_dose_source": dose_source,
            "hplc_assay_mg": assay, "total_pill_mass_mg_as_reported": mass,
            "hplc_mass_fraction_as_reported": finite_number(row["I"], "assay mass fraction"),
            "paper_specific_assay_90_to_120_mg_status": assay_status,
            "assay_status_semantics": "API assay conformance only; not authenticity or fraudulent-intent label",
            "author_poor_samples_sheet_member": int(source_id in poor_ids),
            "confirmed_fraud_status": "not_adjudicated_by_this_export",
            "source_metadata_file": metadata_filename, "source_metadata_sheet": "Metadata",
            "source_metadata_excel_row": row["__row__"],
        }
        samples.append({**shared,
                        "matching_replicate_csv_lines": ";".join(map(str, matching_lines)),
                        "matched_replicate_scan_count": len(source_replicates),
                        **dict(zip(intensity_fields, intensity, strict=True))})
        if source_replicates:
            for line, values in source_replicates:
                matched_raw_lines.add(line)
                scans.append({**shared, "scan_id": f"all_external_csv_line_{line}",
                              "scan_source_file": replicate_filename, "scan_source_row": line,
                              "spectrum_selection": "source replicate matched by exact sample record ID",
                              **dict(zip(intensity_fields, values, strict=True))})
        else:
            scans.append({**shared, "scan_id": f"metadata_excel_row_{row['__row__']}",
                          "scan_source_file": metadata_filename, "scan_source_row": row["__row__"],
                          "spectrum_selection": "single embedded representative; replicate CSV unavailable for this ID",
                          **dict(zip(intensity_fields, intensity, strict=True))})
    if len({s["source_sample_record_id"] for s in samples}) != len(samples):
        raise ValueError("Metadata repeats a sample record ID")
    if not poor_ids <= {s["source_sample_record_id"] for s in samples}:
        raise ValueError("Poor sample metadata is not completely represented")

    # This separate external_set file's MG values are nominal doses, not HPLC
    # assays. Its scan IDs do not provide a clean join to the main assay table.
    loose_spectra = csv_rows(raw / "external_set_doxy.csv")
    loose_ids = csv_rows(raw / "external_set_doxy_conc.csv")
    if len(loose_spectra) != len(loose_ids):
        raise ValueError("External CSV spectral rows and accompanying ID rows differ")
    if any(row[1] != "100" for row in loose_ids[1:]):
        raise ValueError("External nominal-dose column changed unexpectedly")
    duplicate_loose_ids = {key: n for key,n in Counter(r[0] for r in loose_ids[1:]).items() if n > 1}
    conservative_records = {sample["conservative_specimen_group_id"]: sample for sample in samples}
    unmatched = [{"source_sample_record_id": key, "scan_rows": len(values)}
                 for key, values in sorted(replicates.items()) if key not in {s["source_sample_record_id"] for s in samples}]
    write_table(out / "field_samples.csv", samples)
    write_table(out / "field_scans.csv", scans)
    write_table(out / "spectral_axis.csv", [{"channel_index": i, "wavelength_nm": value, "derived_csv_column": intensity_fields[i]}
                                          for i,value in enumerate(axis)])
    summary = {
        "article_doi": "10.1039/D4AY00001C", "article_url": PAPER_URL,
        "source_repository": REPOSITORY_URL, "source_commit": manifest["commit_sha"],
        "source_license": "MIT at repository level; original LICENSE preserved; no separate data-specific license file supplied",
        "raw_data_bytes": manifest["data_bytes"],
        "field_source_sample_records": len(samples), "conservative_source_specimen_groups": len(conservative_records),
        "raw_brand_manufacturer_lot_groups": len({s["source_lot_group_id"] for s in samples}),
        "conservative_source_lot_groups": len({s["conservative_batch_group_id"] for s in samples}),
        "conservative_batch_group_definition": "Nominal API doxycycline plus normalized printed lot; source manufacturer inconsistencies such as SC-027 are kept in the same group. This may overgroup unrelated batches and does not prove shared physical origin.",
        "field_scans_with_unambiguous_assay_metadata": len(scans),
        "matched_replicate_csv_scans": len(matched_raw_lines),
        "metadata_only_representative_spectra": sum(not s["matched_replicate_scan_count"] for s in samples),
        "spectral_channels": len(axis), "actual_wavelength_nm_range": [axis[0], axis[-1]],
        "axis_orientation": "one acquisition per row, 228 wavelength columns; actual source axis preserved without interpolation",
        "signal_units": "source numeric NIR signal, no additional calibration or transformation applied by this export",
        "same_specimen_alias_groups_inferred_from_base_id_and_identical_assay": {key: values for key,values in grouping_members.items() if len(values) > 1},
        "paper_specific_assay_rule": {"nominal_dose_mg": 100, "lower_mg_inclusive": 90, "upper_mg_inclusive": 120,
                                      "source": "article discussion of monograph/API-content evaluation; this is an assay-conformance flag, not fraud"},
        "assay_status_counts_by_sample_record": dict(Counter(s["paper_specific_assay_90_to_120_mg_status"] for s in samples)),
        "assay_status_counts_by_conservative_specimen_group": dict(Counter(s["paper_specific_assay_90_to_120_mg_status"] for s in conservative_records.values())),
        "author_poor_samples_sheet_members": sorted(poor_ids),
        "publication_count_discrepancy": "Article describes 23 products (18 pass/5 fail); source Metadata has 26 spectral records. Three pairs of bw/cg source-ID variants share identical assays/mass/lot and conservatively group to 23, but this inferred grouping is not a documented author sample manifest. The 121.6 mg record remains above the stated 120 mg limit, so our derived flags must not be presented as reproducing 18/5.",
        "publication_mass_fraction_discrepancy": "The article describes field-sample API mass fractions of 31–51% except five failed samples at 24–27%. Source workbook records 19K-0551, 19_0554 and 19_0556 report fractions of approximately 70–73% (assay mg divided by listed mass mg). We preserve the source values; the imported workbook cannot be claimed identical to the publication's exact 23-product analysis subset.",
        "unmapped_replicate_csv_sample_records": unmatched,
        "archived_external_set_doxy": {"spectra_rows": len(loose_spectra)-1, "wavelength_columns": len(loose_spectra[0]),
                                        "target_column_name": "MG", "all_target_values": "100 (nominal dose; NOT measured HPLC)",
                                        "duplicate_scan_identifiers": duplicate_loose_ids, "included_in_derived_labeled_scans": False},
        "excluded_raw_files_from_derived_export": {
            "ALL_EXTERNAL_DE_CAPSULES_cap.csv": "contains lab formulations and ambiguous sparse identifiers; archived only",
            "external_set_doxy.csv": "separate scan set lacks unambiguous measured-HPLC mapping; archived only",
            "external_set_doxy_conc.csv": "constant nominal dose and duplicate scan IDs; not assay labels",
        },
        "caveats": [
            "Measured field samples are quantitative medicine-quality observations; no new clinical counterfeit/authenticity labels are assigned.",
            "Keep all scans and source-ID aliases from the same specimen and conservative API/lot group together; they are not independent samples.",
            "Any later spectral model must select only nir_* signal columns as predictors; assay values, derived statuses, brand/lot and identifiers are targets/provenance, not predictive features.",
            "All five author-designated poor samples share the same brand, manufacturer and lot 150502, so performance may follow batch/excipient differences.",
            "Embedded workbook spectra are representative individual scans, not proven averages: 15 match individual replicate CSV rows within 1e-8.",
            "A single molecular graph cannot represent varying doxycycline amounts, talc adulteration or capsule effects; these inputs require spectral/composition models.",
            "The dataset is small and is an external evaluation candidate, not a source of 100k independent real cases.",
        ],
        "derived_files_sha256": {name: digest((out/name).read_bytes()) for name in ("field_samples.csv", "field_scans.csv", "spectral_axis.csv")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text(f"""# Measured doxycycline field spectra and HPLC assays

Original source: [Awotunde et al., Analytical Methods 2024, DOI 10.1039/D4AY00001C]({PAPER_URL}) and [the authors' data repository]({REPOSITORY_URL}), pinned at `{manifest['commit_sha']}`. The repository-level MIT license is preserved in `raw/LICENSE`; no separate data-specific license was supplied. Raw source URLs, byte counts and SHA-256 hashes are in `raw/download_manifest.json`.

`field_samples.csv` contains **{len(samples)} source metadata records**, one representative embedded NIR spectrum per record, HPLC assay mg, nominal dose and source metadata. `field_scans.csv` contains **{len(scans)} measured scan rows** that have an unambiguous assay association: {len(matched_raw_lines)} exact-ID replicate CSV rows plus {sum(not s['matched_replicate_scan_count'] for s in samples)} metadata-only representatives. It excludes duplicate inclusion of embedded spectra already present among the replicates. `spectral_axis.csv` maps the 228 actual source wavelengths from {axis[0]:.4f} to {axis[-1]:.4f} nm; no resampling or spectral preprocessing was applied.

The source tables associate several bw/cg capsule-type variants with identical base IDs, assay values, mass and lot. A **conservative inferred grouping into {len(conservative_records)} specimen groups** prevents treating these variants as independent products. Preserve `conservative_specimen_group_id` and `conservative_batch_group_id` in any later split or uncertainty calculation. The latter groups nominal API plus normalized printed lot, resulting in {len({s['conservative_batch_group_id'] for s in samples})} conservative batch groups. The raw `source_lot_group_id` retains {len({s['source_lot_group_id'] for s in samples})} brand/manufacturer/lot groups, but source lot SC-027 has inconsistent manufacturer names. Conservative grouping keeps those records together, potentially overgrouping unrelated batches; it does not assert shared physical origin.

For a future spectral model, select only the `nir_*` columns as measured input features. HPLC assay values and derived status are evaluation targets; brand, lot and source identifiers are provenance and grouping information. Including these fields as predictors would leak the target or source batch.

The article describes 23 products with 18 reported pass/5 fail. The workbook has 26 spectral records. Three inferred alias pairs can explain the difference in record count, but this is not an author-supplied sample manifest. One source record has 121.6 mg, above the paper-discussed 120 mg bound. The workbook also has three reported mass fractions around 70–73%, whereas the article describes 31–51% for the non-failing field samples. We preserve both numeric assays and source mass fractions and do not claim that this workbook is identical to the publication's exact 23-product analysis subset. The derived 90–120 mg flag is explicitly **API assay conformance, not authenticity or fraudulent intent**. The five records in the authors' `Poor_Samples` sheet are identified separately; other records are not assigned author pass/authenticity labels.

Do not use `external_set_doxy_conc.csv` as measured target labels: its MG column is nominal 100 throughout. The 208 spectra in that separate file, the unmatched replicate rows and the mixed laboratory/capsule table remain archived with mapping limitations in `summary.json`. No laboratory calibration examples are included in the derived field set.

Reproduce from pinned raw data without networking:

```bash
venv/bin/python fetch_field_spectra.py --offline
```

This script downloads only allowlisted data/metadata/license files, never executes author code, and enforces a strict total network payload below 19 MB and an individual response below 9 MB. No model was trained on these data.
""")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("real_world_falsification/field_spectra_doxy"))
    parser.add_argument("--offline", action="store_true", help="Validate/parse previously pinned raw files without networking")
    parser.add_argument("--ref", default=None, help="Git commit/tag to pin; defaults to main for the first fetch")
    args = parser.parse_args()
    manifest = fetch_raw(args.out, args.offline, args.ref)
    metadata = {
        "article_doi": "10.1039/D4AY00001C", "article_url": PAPER_URL,
        "repository_url": REPOSITORY_URL, "source_commit": manifest["commit_sha"],
        "source_article_field_sample_description": "23 commercial 100 mg doxycycline products collected in Western Kenya and Liberia during 2016–2021; NIR and HPLC measured; 18 passed and 5 failed the reported HPLC assay",
        "field_status_semantics": "HPLC assay pass/fail or out-of-specification status is product-quality evidence; it does not establish fraudulent intent",
        "simulated_calibration_excluded": "Dedicated lab-made doxycycline/lactose and isoniazid/cellulose calibration files are not fetched; one mixed capsule file is archived only and lab rows are excluded from derived field exports",
        "source_license_path": "raw/LICENSE", "source_license_identifier": "MIT",
        "independent_sample_count_is_not_spectrum_count": True,
    }
    (args.out / "raw" / "source_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = parse_field_data(args.out, manifest)
    print(json.dumps({"out": str(args.out), "data_bytes": manifest["data_bytes"],
                      "commit_sha": manifest["commit_sha"], "parse_status": "field spectra and quantitative assays exported",
                      "source_sample_records": summary["field_source_sample_records"],
                      "conservative_specimen_groups": summary["conservative_source_specimen_groups"],
                      "labeled_scan_rows": summary["field_scans_with_unambiguous_assay_metadata"]}, indent=2))


if __name__ == "__main__":
    main()
