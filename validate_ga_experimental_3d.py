"""Compare generated GA conformers with exact-match experimental PDB ligands.

RCSB exact InChIKey matching finds CCD components in experimental PDB entries.
The CCD model coordinates are the ligand coordinates from the first PDB entry
that used the component; idealized CCD coordinates are deliberately ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import rdMolAlign

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_CCD = "https://files.rcsb.org/ligands/download/{component}.cif"
CATEGORIES = ["bioisostere", "halogen-walk", "scaffold-hop", "homologation", "other"]


def read_cif(text: str) -> tuple[dict[str, str], dict[str, list[dict[str, str]]]]:
    # Keep table accumulation separate from scalar values.
    tables: dict[str, list[dict[str, str]]] = {}
    scalar: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("_"):
            parts = shlex.split(line, comments=True)
            if len(parts) >= 2:
                scalar[parts[0]] = parts[1]
            i += 1
            continue
        if line.lower() != "loop_":
            i += 1
            continue
        i += 1
        columns = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            columns.append(shlex.split(lines[i].strip(), comments=True)[0])
            i += 1
        if not columns:
            continue
        table = columns[0].split(".", 1)[0]
        rows = []
        while i < len(lines):
            row_line = lines[i].strip()
            if not row_line or row_line.startswith("#") or row_line.lower() == "loop_" or row_line.startswith("_"):
                break
            values = shlex.split(row_line, comments=True)
            if len(values) == len(columns):
                rows.append(dict(zip(columns, values)))
            i += 1
        if rows:
            tables.setdefault(table, []).extend(rows)
    return scalar, tables


def cif_descriptor(tables: dict[str, list[dict[str, str]]], descriptor_type: str) -> str | None:
    for row in tables.get("_pdbx_chem_comp_descriptor", []):
        if row.get("_pdbx_chem_comp_descriptor.type", "").upper() == descriptor_type.upper():
            val = row.get("_pdbx_chem_comp_descriptor.descriptor", "")
            if val not in {"", ".", "?"}:
                return val
    return None


def parse_reference_molecule(cif_text: str, target_inchikey: str):
    scalar, tables = read_cif(cif_text)
    atom_rows = tables.get("_chem_comp_atom", [])
    bond_rows = tables.get("_chem_comp_bond", [])
    if not atom_rows or not bond_rows:
        raise ValueError("CCD CIF is missing atom or bond rows")
    component_key = cif_descriptor(tables, "InChIKey")
    if component_key != target_inchikey:
        raise ValueError(f"CCD InChIKey mismatch: {component_key} != {target_inchikey}")
    if scalar.get("_chem_comp.pdbx_model_coordinates_missing_flag", "Y") != "N":
        raise ValueError("CCD component has no complete model coordinates")

    rw = Chem.RWMol()
    atom_ids: dict[str, int] = {}
    coords: list[list[float]] = []
    for row in atom_rows:
        element = row.get("_chem_comp_atom.type_symbol", "").capitalize()
        atom_id = row.get("_chem_comp_atom.atom_id", "")
        xyz = [row.get(f"_chem_comp_atom.model_Cartn_{axis}", "?") for axis in "xyz"]
        if element == "H":
            continue
        if element in {"", "?", "."} or any(v in {"?", ".", ""} for v in xyz):
            raise ValueError("CCD model coordinates are missing for a heavy atom")
        atom = Chem.Atom(element)
        charge = row.get("_chem_comp_atom.charge", "0")
        atom.SetFormalCharge(int(charge) if charge not in {"?", ".", ""} else 0)
        idx = rw.AddAtom(atom)
        atom_ids[atom_id] = idx
        coords.append([float(v) for v in xyz])

    order = {
        "SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC,
        "DELO": Chem.BondType.AROMATIC,
    }
    for row in bond_rows:
        a = row.get("_chem_comp_bond.atom_id_1", "")
        b = row.get("_chem_comp_bond.atom_id_2", "")
        if a not in atom_ids or b not in atom_ids:
            continue
        code = row.get("_chem_comp_bond.value_order", "SING").upper()
        if code not in order:
            continue
        rw.AddBond(atom_ids[a], atom_ids[b], order[code])
        if code in {"AROM", "DELO"}:
            bond = rw.GetBondBetweenAtoms(atom_ids[a], atom_ids[b])
            bond.SetIsAromatic(True)
            rw.GetAtomWithIdx(atom_ids[a]).SetIsAromatic(True)
            rw.GetAtomWithIdx(atom_ids[b]).SetIsAromatic(True)

    ref = rw.GetMol()
    Chem.SanitizeMol(ref)
    conf = Chem.Conformer(ref.GetNumAtoms())
    for atom_id, idx in atom_ids.items():
        row = next(r for r in atom_rows if r.get("_chem_comp_atom.atom_id") == atom_id)
        xyz = [float(row[f"_chem_comp_atom.model_Cartn_{axis}"]) for axis in "xyz"]
        conf.SetAtomPosition(idx, xyz)
    conf.Set3D(True)
    ref.AddConformer(conf, assignId=True)
    # Exact InChIKey was already checked against the CCD descriptor above.
    # Coordinate-derived stereo can be ambiguous in a heavy-atom-only graph
    # (implicit H positions are absent), so atom mapping below uses connectivity
    # and element identity; stereochemical identity is guaranteed by that key.
    pdb_id = scalar.get("_chem_comp.pdbx_model_coordinates_db_code", "?")
    if pdb_id in {"", ".", "?"}:
        raise ValueError("CCD has no source PDB entry for model coordinates")
    return ref, pdb_id, scalar, tables


def best_symmetry_rmsd(pred, ref, max_matches: int = 10000) -> tuple[float, int]:
    # The structures have an exact InChIKey match. Ignore RDKit's incomplete
    # coordinate-derived stereo tags here because CCD omits explicit H atoms.
    matches = ref.GetSubstructMatches(pred, uniquify=False, useChirality=False,
                                      maxMatches=max_matches)
    if not matches:
        raise ValueError("No heavy-atom graph mapping; likely a different tautomer representation")
    rmsd = rdMolAlign.GetBestRMS(pred, ref, maxMatches=max_matches)
    return float(rmsd), len(matches)


def query_batch(keys: list[str]) -> list[str]:
    query = {
        "query": {"type": "terminal", "service": "text_chem", "parameters": {
            "attribute": "rcsb_chem_comp_descriptor.InChIKey", "operator": "in", "value": keys}},
        "return_type": "mol_definition",
        "request_options": {"results_content_type": ["experimental"]},
    }
    url = requests.Request("GET", RCSB_SEARCH, params={"json": json.dumps(query)}).prepare().url
    if url is None or len(url) > 7000:
        raise ValueError("RCSB query batch URL exceeded safe size")
    last_error = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=45)
            if response.status_code == 204:
                return []
            response.raise_for_status()
            data = response.json()
            return sorted({row["identifier"] for row in data.get("result_set", [])})
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"RCSB search batch failed after retries: {last_error}")


def fetch_cif(component: str, cache: Path) -> str:
    path = cache / f"{component}.cif"
    if path.exists():
        return path.read_text()
    last_error = None
    for attempt in range(5):
        try:
            response = requests.get(RCSB_CCD.format(component=component), timeout=90)
            response.raise_for_status()
            path.write_text(response.text)
            return response.text
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"CCD download failed after retries for {component}: {last_error}")


def load_generated(root: Path, categories: list[str]):
    molecules: dict[str, dict[str, Any]] = {}
    graphs: dict[str, Any] = {}
    for category in categories:
        graph_path = root / "3d" / category / "graphs_3d.npz"
        with np.load(graph_path, allow_pickle=False) as z:
            # Use global coordinate storage once, preserving molecule slices.
            graph_meta = {"atom_ptr": z["atom_ptr"].copy(), "positions": z["positions_angstrom"].copy(),
                          "ids": z["molecule_id"].tolist()}
        graphs[category] = graph_meta
        with (root / f"counterfeits_evolved_{category}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        id_set = set(graph_meta["ids"])
        id_index = {mid: i for i, mid in enumerate(graph_meta["ids"])}
        for row in rows:
            mol = Chem.MolFromSmiles(row["counterfeit_smiles"])
            if mol is None:
                continue
            key = Chem.MolToInchiKey(mol)
            if key in molecules:
                continue
            mol_id = __import__("hashlib").sha256(Chem.MolToSmiles(mol).encode()).hexdigest()[:20]
            if mol_id not in id_set:
                continue
            idx = id_index[mol_id]
            lo, hi = graph_meta["atom_ptr"][idx:idx+2]
            molecules[key] = {"category": category, "smiles": Chem.MolToSmiles(mol),
                              "mol_id": mol_id, "graph_start": int(lo), "graph_end": int(hi),
                              "parent_smiles": row.get("parent_smiles", "")}
    return molecules, graphs


def predicted_molecule(item: dict[str, Any], graphs: dict[str, Any]):
    mol = Chem.MolFromSmiles(item["smiles"])
    mh = Chem.AddHs(mol)
    coords = graphs[item["category"]]["positions"][item["graph_start"]:item["graph_end"]]
    if len(coords) != mh.GetNumAtoms():
        raise ValueError("Stored graph atom count does not match SMILES plus explicit hydrogens")
    conf = Chem.Conformer(mh.GetNumAtoms())
    for atom_idx, xyz in enumerate(coords):
        conf.SetAtomPosition(atom_idx, xyz.tolist())
    conf.Set3D(True)
    mh.AddConformer(conf, assignId=True)
    return Chem.RemoveHs(mh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("version_ga"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional query limit for smoke checks")
    args = parser.parse_args()
    molecules, graphs = load_generated(args.root, CATEGORIES)
    keys = sorted(molecules)
    if args.limit:
        keys = keys[:args.limit]
    batches = [keys[i:i+args.batch_size] for i in range(0, len(keys), args.batch_size)]
    print(f"Querying {len(keys):,} exact InChIKeys in {len(batches):,} RCSB batches", flush=True)
    ccd_ids: set[str] = set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(query_batch, batch) for batch in batches]
        for i, future in enumerate(as_completed(futures), 1):
            ccd_ids.update(future.result())
            if i % 25 == 0 or i == len(futures):
                print(f"RCSB search batches: {i:,}/{len(futures):,}; CCD components={len(ccd_ids):,}", flush=True)

    out_dir = args.root / "experimental_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ccd_search_hits.json").write_text(json.dumps(sorted(ccd_ids), indent=2) + "\n")

    ccd_cache = args.root / "experimental_reference" / "ccd"
    ccd_cache.mkdir(parents=True, exist_ok=True)
    matched_components: dict[str, list[str]] = {}
    fetch_errors = []
    for i, component in enumerate(sorted(ccd_ids), 1):
        try:
            cif = fetch_cif(component, ccd_cache)
            _, tables = read_cif(cif)
            key = cif_descriptor(tables, "InChIKey")
            if key and key in molecules:
                matched_components.setdefault(key, []).append(component)
        except Exception as exc:
            fetch_errors.append({"component": component, "error": str(exc)})
        if i % 50 == 0 or i == len(ccd_ids):
            print(f"CCD CIFs checked: {i:,}/{len(ccd_ids):,}; exact keys={len(matched_components):,}", flush=True)

    comparisons = []
    coordinate_errors = []
    for i, (key, components) in enumerate(sorted(matched_components.items()), 1):
        item = molecules[key]
        options = []
        for component in components:
            try:
                cif = fetch_cif(component, ccd_cache)
                ref, pdb_id, scalar, tables = parse_reference_molecule(cif, key)
                rmsd, n_mappings = best_symmetry_rmsd(predicted_molecule(item, graphs), ref)
                options.append({"component": component, "pdb_id": pdb_id, "rmsd_angstrom": rmsd,
                                "mapping_count": n_mappings})
            except Exception as exc:
                coordinate_errors.append({"molecule_id": item["mol_id"], "category": item["category"],
                                          "component": component, "error": str(exc)})
        if options:
            # CCD model coordinates come from the first PDB entry for each
            # component; if the same exact key has multiple CCD IDs, retain
            # the lowest-RMSD exact component and record all considered IDs.
            best = min(options, key=lambda x: x["rmsd_angstrom"])
            comparisons.append({"molecule_id": item["mol_id"], "category": item["category"],
                                "inchi_key": key, "smiles": item["smiles"],
                                "ccd_id": best["component"], "source_pdb_id": best["pdb_id"],
                                "rmsd_angstrom": best["rmsd_angstrom"],
                                "symmetry_mappings_tested": best["mapping_count"],
                                "all_exact_ccd_ids": ";".join(components)})
        if i % 50 == 0 or i == len(matched_components):
            print(f"Experimental coordinate comparisons: {i:,}/{len(matched_components):,}; "
                  f"usable={len(comparisons):,}", flush=True)

    with (out_dir / "rmsd_comparison.csv").open("w", newline="") as stream:
        columns = ["molecule_id", "category", "inchi_key", "smiles", "ccd_id", "source_pdb_id",
                   "rmsd_angstrom", "symmetry_mappings_tested", "all_exact_ccd_ids"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(comparisons)
    with (out_dir / "search_errors.json").open("w") as stream:
        json.dump({"cif_fetch_errors": fetch_errors, "coordinate_errors": coordinate_errors}, stream, indent=2)

    summary = {"reference_database": "wwPDB Chemical Component Dictionary (model coordinates)",
               "match_method": "Exact standardized InChIKey; experimental CCD components only",
               "coordinate_method": "CCD model_Cartn_x/y/z from first PDB entry for component; ideal coordinates excluded",
               "rmsd_method": "Heavy-atom, symmetry-aware least-squares RMSD after optimal rigid alignment",
               "molecules_with_3d_coordinates": sum(len(g["ids"]) for g in graphs.values()),
               "queried_distinct_generated_inchikeys": len(keys),
               "exact_ccd_key_matches": len(matched_components),
               "usable_experimental_coordinate_comparisons": len(comparisons),
               "coordinate_or_fetch_errors": len(coordinate_errors) + len(fetch_errors),
               "coverage": len(comparisons) / sum(len(g["ids"]) for g in graphs.values()) if graphs else 0.0}
    if comparisons:
        values = np.asarray([x["rmsd_angstrom"] for x in comparisons], dtype=float)
        summary.update({"rmsd_mean_angstrom": float(values.mean()),
                        "rmsd_median_angstrom": float(np.median(values)),
                        "rmsd_q25_angstrom": float(np.quantile(values, .25)),
                        "rmsd_q75_angstrom": float(np.quantile(values, .75)),
                        "fraction_rmsd_le_0_5_angstrom": float(np.mean(values <= .5)),
                        "fraction_rmsd_le_1_angstrom": float(np.mean(values <= 1.0)),
                        "fraction_rmsd_le_2_angstrom": float(np.mean(values <= 2.0))})
    by_category = {}
    for category in CATEGORIES:
        vals = np.asarray([x["rmsd_angstrom"] for x in comparisons if x["category"] == category], dtype=float)
        by_category[category] = {"n_exact_comparisons": int(len(vals)),
                                 "coverage_of_generated_category": float(len(vals) / 10000),
                                 "median_rmsd_angstrom": float(np.median(vals)) if len(vals) else None,
                                 "mean_rmsd_angstrom": float(vals.mean()) if len(vals) else None}
    summary["by_category"] = by_category
    (out_dir / "rmsd_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
