"""Add quality-filtered ChEMBL reference compounds for larger balanced sets.

The ChEMBL structures are public, literature-curated reference compounds; this
script does not label them as approved medicines or as experimentally verified
authentic products. They are used as the non-generated reference class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from rdkit import Chem, RDLogger

from build_final_counterfeit_bank import child_quality

RDLogger.DisableLog("rdApp.*")
API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
STATUS = "https://www.ebi.ac.uk/chembl/api/data/status.json"
FIELDS = ["molecule_id", "smiles", "label", "reference_source", "source_id",
          "max_phase", "mw", "logp", "qed", "sa_score", "heavy_atoms",
          "smiles_length", "child_scaffold"]


def read_smiles(path: Path, column: str) -> set[str]:
    values = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            raw = row.get(column, "")
            mol = Chem.MolFromSmiles(raw) if raw else None
            if mol is not None:
                values.add(Chem.MolToSmiles(mol, canonical=True))
    return values


def fetch_page(offset: int, limit: int, max_attempts: int = 6) -> tuple[int, list[dict]]:
    params = {
        "molecule_type": "Small molecule",
        "molecule_structures__isnull": "false",
        "molecule_properties__full_mwt__range": "150,800",
        "limit": limit, "offset": offset,
        "only": "molecule_chembl_id,max_phase,molecule_structures",
    }
    last_error = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(API, params=params, timeout=90)
            response.raise_for_status()
            return offset, response.json().get("molecules", [])
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"ChEMBL page offset={offset} failed: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=80000,
                        help="Number of new quality-filtered reference molecules")
    parser.add_argument("--out", type=Path,
                        default=Path("molecule_bank_500k/chembl_controls.csv"))
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()
    if args.target <= 0 or args.workers <= 0 or not 1 <= args.page_size <= 1000:
        parser.error("target/workers must be positive and page-size must be 1..1000")

    exclusions = set()
    for path, column in [
        (Path("final_dataset_400k/authentic.csv"), "smiles"),
        (Path("final_dataset_400k/molecules.csv"), "smiles"),
        (Path("version_ga/counterfeits_all.csv"), "counterfeit_smiles"),
        (Path("version_m/counterfeits_all.csv"), "counterfeit_smiles"),
        (Path("version_e/counterfeits_evolved.csv"), "counterfeit_smiles"),
    ]:
        if path.exists():
            exclusions.update(read_smiles(path, column))
    existing_controls = set(exclusions)
    candidates: dict[str, dict] = {}
    rejected = {name: 0 for name in ("missing_smiles", "invalid_smiles", "overlap_or_duplicate",
                                     "quality_filter")}

    status_response = requests.get(STATUS, timeout=60)
    status_response.raise_for_status()
    chembl_status = status_response.json()
    query = {
        "molecule_type": "Small molecule",
        "molecule_structures__isnull": "false",
        "molecule_properties__full_mwt__range": "150,800",
        "limit": 1, "offset": 0,
        "only": "molecule_chembl_id",
    }
    response = requests.get(API, params=query, timeout=90)
    response.raise_for_status()
    total_count = int(response.json()["page_meta"]["total_count"])
    page_count = math.ceil(total_count / args.page_size)
    offsets = list(range(0, page_count * args.page_size, args.page_size))
    random.Random(args.seed).shuffle(offsets)

    # Pull random pages across the catalog and stop once enough new compounds
    # pass exactly the project quality gate used for generated molecules.
    cursor = 0
    while len(candidates) < args.target and cursor < len(offsets):
        batch = offsets[cursor:cursor + args.workers * 6]
        cursor += len(batch)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_page, offset, args.page_size) for offset in batch]
            pages = []
            for future in as_completed(futures):
                pages.append(future.result())
        for offset, records in sorted(pages):
            for record in records:
                structures = record.get("molecule_structures") or {}
                raw = structures.get("canonical_smiles")
                if not raw:
                    rejected["missing_smiles"] += 1; continue
                mol = Chem.MolFromSmiles(raw)
                if mol is None:
                    rejected["invalid_smiles"] += 1; continue
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if smiles in exclusions:
                    rejected["overlap_or_duplicate"] += 1; continue
                quality = child_quality(smiles)
                if not isinstance(quality, dict):
                    rejected["quality_filter"] += 1; continue
                exclusions.add(smiles)
                source_id = record["molecule_chembl_id"]
                molecule_id = "mol_" + hashlib.sha256(smiles.encode()).hexdigest()[:20]
                candidates[smiles] = {
                    "molecule_id": molecule_id, "smiles": smiles, "label": 0,
                    "reference_source": "ChEMBL", "source_id": source_id,
                    "max_phase": record.get("max_phase") or "",
                    "mw": quality["mw"], "logp": quality["logp"],
                    "qed": quality["qed"], "sa_score": quality["sa_score"],
                    "heavy_atoms": quality["heavy_atoms"],
                    "smiles_length": quality["smiles_length"],
                    "child_scaffold": quality["child_scaffold"],
                }
        print(f"Pages {cursor:,}/{len(offsets):,}; accepted {len(candidates):,}/{args.target:,}", flush=True)

    if len(candidates) < args.target:
        raise RuntimeError(f"Only {len(candidates):,} ChEMBL controls passed; wanted {args.target:,}")
    rows = sorted(candidates.values(),
                  key=lambda row: hashlib.sha256(f"{args.seed}:{row['source_id']}".encode()).digest())[:args.target]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "source": "ChEMBL small-molecule web service",
        "source_url": API, "status_url": STATUS,
        "chembl_status": chembl_status,
        "query": {"molecule_type": "Small molecule", "structure_present": True,
                  "full_mwt_range": [150, 800]},
        "project_quality_gate": "canonical, single component, MW 150-800, 10-70 heavy atoms, QED >= 0.35, SA <= 3.8, no configured structural alerts",
        "generated_and_existing_reference_overlaps_excluded": len(existing_controls) > 0,
        "excluded_base_structures": len(existing_controls),
        "pages_scanned": cursor, "catalog_records_available": total_count,
        "candidates_passing_quality": len(candidates), "written": len(rows),
        "rejected": rejected, "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_semantics": "non-generated public reference compound; not a claim of approved medicine or product authenticity",
        "license": "CC BY-SA 3.0; attribute ChEMBL and share adaptations under the same license",
        "csv": str(args.out),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
