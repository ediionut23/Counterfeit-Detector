"""Check coverage and tensor consistency for packed GA 3D graph archives."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from run_version_ga import CATEGORIES

RDLogger.DisableLog("rdApp.*")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def audit_one(root: Path, category: str) -> dict:
    folder = root / "3d" / category
    report_path, graph_path, sdf_path = (
        folder / "conformers_3d_report.json", folder / "graphs_3d.npz", folder / "conformers.sdf.gz")
    missing = [str(p) for p in (report_path, graph_path, sdf_path) if not p.exists()]
    if missing:
        return {"missing_files": missing, "passed": False}
    report = json.loads(report_path.read_text())
    issues = []
    with np.load(graph_path, allow_pickle=False) as graph:
        n_mol = len(graph["molecule_id"])
        n_atoms = len(graph["atom_features"])
        arrays = {
            "positions_finite": bool(np.isfinite(graph["positions_angstrom"]).all()),
            "atom_shape": graph["atom_features"].shape == (n_atoms, 8),
            "position_shape": graph["positions_angstrom"].shape == (n_atoms, 3),
            "atom_ptr": len(graph["atom_ptr"]) == n_mol + 1 and graph["atom_ptr"][0] == 0 and graph["atom_ptr"][-1] == n_atoms,
            "edge_ptr": len(graph["edge_ptr"]) == n_mol + 1 and graph["edge_ptr"][0] == 0 and graph["edge_ptr"][-1] == graph["edge_index"].shape[1],
            "radius_ptr": len(graph["radius_ptr"]) == n_mol + 1 and graph["radius_ptr"][0] == 0 and graph["radius_ptr"][-1] == graph["radius_edge_index"].shape[1],
            "angle_ptr": len(graph["angle_ptr"]) == n_mol + 1 and graph["angle_ptr"][0] == 0 and graph["angle_ptr"][-1] == graph["angle_index"].shape[1],
            "edge_index_shape": graph["edge_index"].shape[0] == 2,
            "radius_edge_index_shape": graph["radius_edge_index"].shape[0] == 2,
            "angle_index_shape": graph["angle_index"].shape[0] == 3,
            "bond_features_shape": graph["edge_features"].shape == (graph["edge_index"].shape[1], 5),
            "distance_shape": len(graph["edge_distance_angstrom"]) == graph["edge_index"].shape[1],
            "radius_distance_shape": len(graph["radius_edge_distance_angstrom"]) == graph["radius_edge_index"].shape[1],
            "angle_shape": len(graph["angle_radians"]) == graph["angle_index"].shape[1],
            "label_count": len(graph["label"]) == n_mol and bool(np.all(graph["label"] == 1)),
            "radius_distances_positive": bool(np.all(graph["radius_edge_distance_angstrom"] > 0)),
            "angles_in_range": bool(np.all((graph["angle_radians"] >= 0) & (graph["angle_radians"] <= np.pi))),
            "indices_in_range": bool(
                np.all((graph["edge_index"] >= 0) & (graph["edge_index"] < n_atoms))
                and np.all((graph["radius_edge_index"] >= 0) & (graph["radius_edge_index"] < n_atoms))
                and np.all((graph["angle_index"] >= 0) & (graph["angle_index"] < n_atoms))
            ),
            "ids_smiles_counts_match": len(graph["smiles"]) == n_mol and len(set(graph["molecule_id"].tolist())) == n_mol,
        }
        issues.extend(name for name, passed in arrays.items() if not passed)
        n_sdf = 0
        n_sdf_3d = 0
        with gzip.open(sdf_path, "rb") as stream:
            supplier = Chem.ForwardSDMolSupplier(stream, removeHs=False)
            for mol in supplier:
                n_sdf += 1
                if mol is not None and mol.GetNumConformers() and mol.GetConformer().Is3D():
                    n_sdf_3d += 1
        arrays["sdf_count_matches_graphs"] = n_sdf == n_mol
        arrays["sdf_all_records_have_3d"] = n_sdf_3d == n_sdf
        arrays["report_count_matches_graphs"] = report.get("embedded") == n_mol
        issues.extend(name for name in ("sdf_count_matches_graphs", "sdf_all_records_have_3d",
                                        "report_count_matches_graphs") if not arrays[name])
    return {
        "category": category, "molecules": n_mol, "sdf_records": n_sdf,
        "report_coverage": report.get("coverage"), "checks": arrays,
        "issues": issues, "passed": not issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("version_ga"))
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = parser.parse_args()
    results = {category: audit_one(args.root, category) for category in args.categories}
    report = {
        "categories": results,
        "all_present_and_passed": all(r.get("passed", False) for r in results.values()),
    }
    (args.root / "3d" / "audit_3d.json").write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n")
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
