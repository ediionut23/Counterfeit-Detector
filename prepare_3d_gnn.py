"""Create optimized 3D conformers and packed graph tensors for GA datasets.

The selected coordinates are predicted gas-phase conformers (ETKDGv3 followed
by MMFF94s minimization, with UFF as a parameter fallback), not experimental
structures. All atoms, including hydrogens, are retained so bond lengths and
valence angles are available to 3D message-passing models.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
_WORKER_N_CONFS = 8
_WORKER_SEED = 20261024
_WORKER_RADIUS_CUTOFF = 5.0


def _init_worker(n_confs: int, seed: int, radius_cutoff: float) -> None:
    global _WORKER_N_CONFS, _WORKER_SEED, _WORKER_RADIUS_CUTOFF
    RDLogger.DisableLog("rdApp.*")
    _WORKER_N_CONFS = n_confs
    _WORKER_SEED = seed
    _WORKER_RADIUS_CUTOFF = radius_cutoff


def _embed_one(row: Dict[str, str]) -> Optional[Dict]:
    smiles = row.get("counterfeit_smiles") or row.get("smiles")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    mol_id = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    mol_h = Chem.AddHs(mol)
    seed = int(hashlib.sha256(f"{_WORKER_SEED}:{canonical}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF

    params = AllChem.ETKDGv3()
    params.randomSeed = seed or 1
    params.numThreads = 1
    params.pruneRmsThresh = 0.5
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=_WORKER_N_CONFS, params=params))
    if not conf_ids:
        params.useRandomCoords = True
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=max(2, _WORKER_N_CONFS // 2), params=params))
    if not conf_ids:
        return None

    ff_name = "MMFF94s"
    optimized: list[Tuple[int, int, float]] = []
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant=ff_name)
        if props is not None:
            for cid, (status, energy) in enumerate(AllChem.MMFFOptimizeMoleculeConfs(
                mol_h, numThreads=1, maxIters=600, mmffVariant=ff_name
            )):
                optimized.append((int(cid), int(status), float(energy)))
    except Exception:
        optimized = []

    if not optimized:
        ff_name = "UFF"
        try:
            for cid, (status, energy) in enumerate(
                AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=1, maxIters=600)
            ):
                optimized.append((int(cid), int(status), float(energy)))
        except Exception:
            return None
    if not optimized:
        return None

    # Prefer converged structures, then choose the lowest force-field energy.
    converged = [x for x in optimized if x[1] == 0]
    chosen = min(converged or optimized, key=lambda x: x[2])
    conf_id, status, energy = chosen
    conf = mol_h.GetConformer(conf_id)
    if not conf.Is3D():
        return None
    pos = np.asarray(conf.GetPositions(), dtype=np.float32)

    atom_features = []
    for atom in mol_h.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetDegree(),
            int(atom.GetIsAromatic()), int(atom.GetHybridization()),
            int(atom.GetChiralTag()), atom.GetTotalNumHs(), int(atom.IsInRing()),
        ])
    atom_features = np.asarray(atom_features, dtype=np.int16)

    edges, edge_features, distances = [], [], []
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = [int(bond.GetBondType()), int(bond.GetIsAromatic()),
                   int(bond.GetIsConjugated()), int(bond.IsInRing()), int(bond.GetStereo())]
        distance = float(np.linalg.norm(pos[i] - pos[j]))
        edges.extend([[i, j], [j, i]])
        edge_features.extend([feature, feature])
        distances.extend([distance, distance])

    # Spatial/radius graph contains both covalent and noncovalent neighbors.
    delta = pos[:, None, :] - pos[None, :, :]
    pairwise = np.linalg.norm(delta, axis=2)
    src, dst = np.where((pairwise <= _WORKER_RADIUS_CUTOFF) & (pairwise > 1e-8))
    radius_edges = np.column_stack((src, dst)).astype(np.int32, copy=False)
    radius_distances = pairwise[src, dst].astype(np.float32, copy=False)

    angle_indices, angles = [], []
    neighbors = [sorted(a.GetNeighbors(), key=lambda x: x.GetIdx()) for a in mol_h.GetAtoms()]
    for center, adjacent in enumerate(neighbors):
        ids = [a.GetIdx() for a in adjacent]
        for a_idx in range(len(ids)):
            for b_idx in range(a_idx + 1, len(ids)):
                i, k = ids[a_idx], ids[b_idx]
                v1, v2 = pos[i] - pos[center], pos[k] - pos[center]
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom <= 1e-8:
                    continue
                cos_theta = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                angle_indices.append([i, center, k])
                angles.append(float(np.arccos(cos_theta)))

    record = Chem.Mol(mol_h)
    record.RemoveAllConformers()
    record.AddConformer(Chem.Conformer(conf), assignId=True)
    record.SetProp("smiles", canonical)
    record.SetProp("molecule_id", mol_id)
    record.SetIntProp("label", int(row.get("label", 1) or 1))
    for key in ("category", "subcategory", "generation_method", "generator_variant",
                "generation_protocol", "parent_smiles", "similarity_to_parent", "qed",
                "sa_score", "objective_score", "split", "reference_source", "source_id"):
        if row.get(key) not in (None, ""):
            record.SetProp(key, str(row[key]))
    record.SetProp("forcefield", ff_name)
    record.SetProp("forcefield_energy_kcal_mol", f"{energy:.6f}")
    record.SetProp("optimization_converged", str(status == 0).lower())
    record.SetProp("conformers_attempted", str(len(conf_ids)))

    return {
        "row": row, "mol": record, "mol_id": mol_id, "smiles": canonical,
        "atom_features": atom_features, "pos": pos,
        "edges": np.asarray(edges, dtype=np.int32).reshape(-1, 2),
        "edge_features": np.asarray(edge_features, dtype=np.int16).reshape(-1, 5),
        "edge_distances": np.asarray(distances, dtype=np.float32),
        "radius_edges": radius_edges,
        "radius_distances": radius_distances,
        "angle_indices": np.asarray(angle_indices, dtype=np.int32).reshape(-1, 3),
        "angles": np.asarray(angles, dtype=np.float32),
        "forcefield": ff_name, "energy": energy, "converged": status == 0,
    }


def _append_with_offset(target: list, value: np.ndarray, offset: int) -> None:
    if value.size == 0:
        return
    target.append(value + offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Generated category CSV")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--conformers", type=int, default=8, help="ETKDG conformers per molecule")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Molecules assigned to a worker at a time")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a staged run")
    parser.add_argument("--radius-cutoff", type=float, default=5.0,
                        help="Spatial graph radius in Angstrom")
    args = parser.parse_args()
    if args.workers < 1 or args.conformers < 1 or args.chunk_size < 1 or args.radius_cutoff <= 0:
        parser.error("--workers, --conformers, --chunk-size and --radius-cutoff must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.limit is not None:
        rows = rows[:args.limit]

    output_sdf = args.out / "conformers.sdf.gz"
    partial_sdf = args.out / "conformers.partial.sdf"
    atom_chunks, pos_chunks = [], []
    edge_chunks, edge_feature_chunks, edge_distance_chunks = [], [], []
    radius_chunks, radius_distance_chunks = [], []
    angle_chunks, angle_value_chunks = [], []
    atom_ptr, edge_ptr, radius_ptr, angle_ptr = [0], [0], [0], [0]
    molecule_ids, input_ids, smiles_values, parents = [], [], [], []
    categories, subcategories, labels, splits, generation_methods = [], [], [], [], []
    forcefields, energies, converged_values = [], [], []
    failed_rows = []
    n_failed = n_converged = 0

    with partial_sdf.open("w") as sdf_stream:
        writer = Chem.SDWriter(sdf_stream)
        try:
            with mp.Pool(args.workers, initializer=_init_worker,
                         initargs=(args.conformers, args.seed, args.radius_cutoff)) as pool:
                for index, result in enumerate(pool.imap(_embed_one, rows, chunksize=args.chunk_size), 1):
                    if result is None:
                        n_failed += 1
                        failed_rows.append({
                            **rows[index - 1],
                            "failure_reason": "SMILES parsing, embedding, or force-field optimization failed",
                        })
                    else:
                        writer.write(result["mol"])
                        n_converged += int(result["converged"])
                        atom_chunks.append(result["atom_features"])
                        pos_chunks.append(result["pos"])
                        atom_offset = atom_ptr[-1]
                        _append_with_offset(edge_chunks, result["edges"], atom_offset)
                        _append_with_offset(radius_chunks, result["radius_edges"], atom_offset)
                        edge_feature_chunks.append(result["edge_features"])
                        edge_distance_chunks.append(result["edge_distances"])
                        _append_with_offset(angle_chunks, result["angle_indices"], atom_offset)
                        angle_value_chunks.append(result["angles"])
                        atom_ptr.append(atom_offset + len(result["atom_features"]))
                        edge_ptr.append(edge_ptr[-1] + len(result["edges"]))
                        radius_ptr.append(radius_ptr[-1] + len(result["radius_edges"]))
                        radius_distance_chunks.append(result["radius_distances"])
                        angle_ptr.append(angle_ptr[-1] + len(result["angles"]))
                        row = result["row"]
                        molecule_ids.append(result["mol_id"])
                        input_ids.append(row.get("molecule_id") or row.get("id", ""))
                        smiles_values.append(result["smiles"])
                        parents.append(row.get("parent_smiles", ""))
                        categories.append(row.get("category", ""))
                        subcategories.append(row.get("subcategory", ""))
                        splits.append(row.get("split", ""))
                        generation_methods.append(row.get("generation_method", ""))
                        labels.append(int(row.get("label", 1) or 1))
                        forcefields.append(result["forcefield"])
                        energies.append(result["energy"])
                        converged_values.append(result["converged"])
                    if index % 1000 == 0:
                        print(f"{index:,}/{len(rows):,} molecules processed; "
                              f"{len(molecule_ids):,} embedded", flush=True)
        finally:
            writer.close()

    with partial_sdf.open("rb") as source, gzip.open(output_sdf, "wb", compresslevel=6) as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
    partial_sdf.unlink()

    def cat(chunks, shape, dtype):
        return np.concatenate(chunks, axis=0) if chunks else np.empty(shape, dtype=dtype)

    graph_path = args.out / "graphs_3d.npz"
    np.savez_compressed(
        graph_path,
        molecule_id=np.asarray(molecule_ids, dtype="U20"),
        source_id=np.asarray(input_ids, dtype="U64"),
        smiles=np.asarray(smiles_values, dtype="U"),
        parent_smiles=np.asarray(parents, dtype="U"),
        category=np.asarray(categories, dtype="U"),
        subcategory=np.asarray(subcategories, dtype="U"),
        generation_method=np.asarray(generation_methods, dtype="U"),
        split=np.asarray(splits, dtype="U"),
        label=np.asarray(labels, dtype=np.uint8),
        atom_ptr=np.asarray(atom_ptr, dtype=np.int64),
        atom_features=cat(atom_chunks, (0, 8), np.int16),
        positions_angstrom=cat(pos_chunks, (0, 3), np.float32),
        edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
        edge_index=cat(edge_chunks, (0, 2), np.int32).T,
        edge_features=cat(edge_feature_chunks, (0, 5), np.int16),
        edge_distance_angstrom=cat(edge_distance_chunks, (0,), np.float32),
        radius_ptr=np.asarray(radius_ptr, dtype=np.int64),
        radius_edge_index=cat(radius_chunks, (0, 2), np.int32).T,
        radius_edge_distance_angstrom=cat(radius_distance_chunks, (0,), np.float32),
        angle_ptr=np.asarray(angle_ptr, dtype=np.int64),
        angle_index=cat(angle_chunks, (0, 3), np.int32).T,
        angle_radians=cat(angle_value_chunks, (0,), np.float32),
        forcefield=np.asarray(forcefields, dtype="U8"),
        forcefield_energy_kcal_mol=np.asarray(energies, dtype=np.float32),
        optimization_converged=np.asarray(converged_values, dtype=np.bool_),
    )
    report = {
        "input_csv": str(args.input), "requested": len(rows), "embedded": len(molecule_ids),
        "embedded_label_counts": {str(label): int(sum(value == label for value in labels))
                                  for label in (0, 1)},
        "embedded_split_counts": {split: int(sum(value == split for value in splits))
                                  for split in sorted(set(splits))},
        "failed": n_failed, "coverage": round(len(molecule_ids) / max(len(rows), 1), 6),
        "mmff_or_uff_converged": n_converged,
        "conformer_method": f"ETKDGv3 ensemble (up to {args.conformers}); lowest-energy converged MMFF94s, UFF fallback",
        "hydrogens_retained": True,
        "graph_file": str(graph_path), "sdf_file": str(output_sdf),
        "graph_contents": ["atom features", "3D positions in Angstrom", "directed bonds and bond features",
                           "bond lengths in Angstrom", "radius-graph neighbors and distances",
                           "neighbor triplets and valence angles in radians"],
        "atom_feature_columns": ["atomic_number", "formal_charge", "degree", "aromatic",
                                 "hybridization_code", "chiral_tag_code", "total_hydrogens", "in_ring"],
        "bond_feature_columns": ["bond_type_code", "aromatic", "conjugated", "in_ring", "stereo_code"],
        "labels": "label=1 marks generated analogues; label=0 marks public reference controls when supplied in the input.",
        "radius_cutoff_angstrom": args.radius_cutoff,
        "coordinate_interpretation": "predicted force-field-minimized conformer; not experimental coordinates",
    }
    if failed_rows:
        failure_path = args.out / "failed_embeddings.csv"
        fields = list(dict.fromkeys(key for row in failed_rows for key in row))
        with failure_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(failed_rows)
        report["failed_rows_csv"] = str(failure_path)
    (args.out / "conformers_3d_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
