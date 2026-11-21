#!/usr/bin/env python3
"""Prepare versioned, source-aligned GAT chemistry sidecars for geometric-v2.

Only derived chemistry arrays are written. Existing NPZ coordinates, spatial
edges, cohort rows and previous caches remain immutable. Resume occurs at
completed source-shard boundaries; incomplete/limited manifests cannot train.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from rdkit import Chem, RDLogger

import shared_molecular_features_v2 as chemistry


FORMAT_VERSION = 1
ARRAY_NAMES = ("z", "x", "atom_ptr", "bond_ptr", "bond_edge_index",
               "bond_edge_attr", "selected_mask", "prepared_mask", "source_id",
               "molecule_id", "canonical_to_heavy")
DEFAULT_INDEX = "balanced_datasets/training_200k/overnight_strict_converged_v2/graph_index.csv"
DEFAULT_CACHE = "training_cache/shared_geometric_v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def interval(budget, label):
    return budget.interval(label) if budget is not None else nullcontext()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def source_lock(path):
    with Path(path).open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another chemistry-sidecar writer holds this source lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sdf_path(source):
    for filename in ("conformers.sdf.gz", "conformers.sdf"):
        candidate = source.parent / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No ordered source SDF accompanies {source}; atom mapping cannot be guessed")


def _save_array(directory, name, value):
    with (directory / (name + ".npy")).open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def validate_sidecar(directory, expected=None, allow_partial=False, budget=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("feature_version") != chemistry.FEATURE_VERSION:
        raise ValueError("Unsupported shared-chemistry sidecar version")
    if not manifest.get("complete") and not allow_partial:
        raise ValueError("Incomplete chemistry sidecar is not eligible for training")
    if expected is not None and manifest.get("specification") != expected:
        raise ValueError("Sidecar provenance does not match the requested source/cohort/encoder")
    with interval(budget, "sidecar:validate:" + directory.name):
        for filename, detail in manifest["files"].items():
            path = directory / filename
            if path.stat().st_size != detail["bytes"] or sha256(path) != detail["sha256"]:
                raise ValueError(f"Chemistry sidecar checksum mismatch: {filename}")
    return manifest


def _build_source(source, selected, graph_index_sha256, cache_root, code_pins,
                  max_records_per_source=None, budget=None):
    source = Path(source).resolve()
    sdf_path = _sdf_path(source)
    with interval(budget, "feature-source:hash:" + source.parent.name):
        source_sha = sha256(source)
        sdf_sha = sha256(sdf_path)
    selected_rows = sorted(selected.values(), key=lambda row: int(row["graph_index"]))
    prepared_rows = selected_rows[:max_records_per_source] if max_records_per_source is not None else selected_rows
    prepared_numbers = {int(row["graph_index"]) for row in prepared_rows}
    specification = {
        "source": str(source), "source_sha256": source_sha,
        "sdf": str(sdf_path), "sdf_sha256": sdf_sha,
        "graph_index_sha256": graph_index_sha256,
        "selected_rows_sha256": json_digest(selected_rows),
        "prepared_graph_numbers": sorted(prepared_numbers),
        "feature_version": chemistry.FEATURE_VERSION,
        "alignment_version": chemistry.ALIGNMENT_VERSION,
        "rdkit_version": chemistry.RDKIT_VERSION, "code_pins": code_pins,
    }
    directory = cache_root / "sources" / json_digest(specification)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with source_lock(directory.with_suffix(".lock")):
        if (directory / "manifest.json").exists():
            existing = validate_sidecar(directory, specification, allow_partial=True, budget=budget)
            return _source_reference(directory, existing)
        if directory.exists():
            raise ValueError("Unpublished sidecar directory exists; refusing to overwrite unknown files")
        with interval(budget, "feature-source:load:" + source.parent.name):
            names = ("atom_ptr", "atom_features", "positions_angstrom", "edge_ptr",
                     "edge_index", "edge_features", "source_id", "molecule_id", "smiles")
            with np.load(source, allow_pickle=False) as archive:
                missing = set(names) - set(archive.files)
                if missing:
                    raise ValueError(f"Source NPZ lacks alignment evidence: {sorted(missing)}")
                arrays = {name: archive[name] for name in names}
        count = len(arrays["source_id"])
        if arrays["atom_ptr"].shape != (count + 1,) or arrays["edge_ptr"].shape != (count + 1,):
            raise ValueError("Source NPZ pointer lengths do not preserve graph numbers")
        if len(arrays["molecule_id"]) != count or len(arrays["smiles"]) != count:
            raise ValueError("Source NPZ identity columns have different lengths")
        for number, row in selected.items():
            if number < 0 or number >= count:
                raise ValueError("Cohort graph number is outside source NPZ")
            if str(arrays["source_id"][number]) != row["source_id"]:
                raise ValueError("Cohort/source row identity mismatch")
            if str(arrays["molecule_id"][number]) != row["molecule_id"]:
                raise ValueError("Cohort/source molecule identity mismatch")

        temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".partial.", dir=directory.parent))
        atom_ptr = np.zeros(count + 1, dtype=np.int64)
        bond_ptr = np.zeros(count + 1, dtype=np.int64)
        selected_mask = np.zeros(count, dtype=np.bool_)
        prepared_mask = np.zeros(count, dtype=np.bool_)
        selected_mask[list(selected)] = True
        z_chunks, x_chunks, edge_chunks, attr_chunks, mapping_chunks = [], [], [], [], []
        atom_count = bond_count = processed = scanned = 0
        coordinate_max = 0.0
        last_number = max(prepared_numbers, default=-1)
        try:
            opener = gzip.open if sdf_path.suffix == ".gz" else open
            with opener(sdf_path, "rb") as stream:
                supplier = iter(Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False, strictParsing=True))
                for number in range(last_number + 1):
                    with interval(budget, "feature-sdf-read:" + source.parent.name + ":" + str(number)):
                        try:
                            sdf_mol = next(supplier)
                        except StopIteration:
                            break
                    scanned = number + 1
                    atom_ptr[number] = atom_count
                    bond_ptr[number] = bond_count
                    if number in prepared_numbers:
                        row = selected[number]
                        with interval(budget, "feature:" + row["source_id"]):
                            if sdf_mol is None:
                                raise chemistry.AlignmentError(f"Unreadable selected SDF record {number}")
                            if sdf_mol.HasProp("molecule_id") and sdf_mol.GetProp("molecule_id") != row["molecule_id"]:
                                raise chemistry.AlignmentError("SDF record/molecule identity mismatch")
                            begin, end = map(int, arrays["atom_ptr"][number:number + 2])
                            edge_begin, edge_end = map(int, arrays["edge_ptr"][number:number + 2])
                            try:
                                result = chemistry.aligned_features(
                                    str(arrays["smiles"][number]), sdf_mol,
                                    arrays["atom_features"][begin:end], arrays["positions_angstrom"][begin:end],
                                    arrays["edge_index"][:, edge_begin:edge_end] - begin,
                                    arrays["edge_features"][edge_begin:edge_end])
                            except chemistry.AlignmentError as error:
                                raise chemistry.AlignmentError(
                                    f"{source}: graph {number}, source_id={row['source_id']}: {error}") from error
                            z_chunks.append(result["z"])
                            x_chunks.append(result["x"])
                            edge_chunks.append(result["bond_edge_index"] + atom_count)
                            attr_chunks.append(result["bond_edge_attr"])
                            mapping_chunks.append(result["canonical_to_heavy"])
                            atom_count += len(result["z"])
                            bond_count += result["bond_edge_index"].shape[1]
                            coordinate_max = max(coordinate_max, result["audit"]["coordinate_max_abs_error_angstrom"])
                            processed += 1
                            prepared_mask[number] = True
                            if processed % 100 == 0:
                                atomic_json(temporary / "progress.json", {"status": "preparing",
                                    "processed": processed, "requested": len(selected),
                                    "last_graph_number": number, "source": str(source)})
                    atom_ptr[number + 1] = atom_count
                    bond_ptr[number + 1] = bond_count
            if processed != len(prepared_numbers):
                raise chemistry.AlignmentError("Source SDF ended before every requested NPZ record was aligned")
            atom_ptr[scanned:] = atom_count
            bond_ptr[scanned:] = bond_count
            # Check the sources again before publishing derived chemistry.
            with interval(budget, "feature-source:recheck:" + source.parent.name):
                if sha256(source) != source_sha or sha256(sdf_path) != sdf_sha:
                    raise ValueError("Immutable source NPZ/SDF changed during chemistry preparation")
            derived = {
                "z": np.concatenate(z_chunks) if z_chunks else np.empty(0, dtype=np.int64),
                "x": np.concatenate(x_chunks) if x_chunks else np.empty((0, chemistry.NODE_DIM), dtype=np.float32),
                "atom_ptr": atom_ptr, "bond_ptr": bond_ptr,
                "bond_edge_index": np.concatenate(edge_chunks, axis=1) if edge_chunks else np.empty((2, 0), dtype=np.int64),
                "bond_edge_attr": np.concatenate(attr_chunks) if attr_chunks else np.empty((0, chemistry.BOND_DIM), dtype=np.float32),
                "selected_mask": selected_mask, "prepared_mask": prepared_mask,
                "source_id": arrays["source_id"], "molecule_id": arrays["molecule_id"],
                "canonical_to_heavy": np.concatenate(mapping_chunks) if mapping_chunks else np.empty(0, dtype=np.int64),
            }
            files = {}
            for name in ARRAY_NAMES:
                with interval(budget, "feature-source:publish-array:" + name):
                    _save_array(temporary, name, derived[name])
                    path = temporary / (name + ".npy")
                    files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            manifest = {
                "format_version": FORMAT_VERSION, "feature_version": chemistry.FEATURE_VERSION,
                "alignment_version": chemistry.ALIGNMENT_VERSION,
                "complete": processed == len(selected), "source_sha256": source_sha,
                "sdf_sha256": sdf_sha, "specification": specification,
                "graph_count": count, "selected_rows": len(selected), "prepared_rows": processed,
                "atoms": atom_count, "directed_bonds": bond_count,
                "sdf_records_scanned": scanned,
                "coordinate_max_abs_error_angstrom": coordinate_max,
                "node_dim": chemistry.NODE_DIM, "bond_dim": chemistry.BOND_DIM,
                "node_columns": list(chemistry.NODE_COLUMNS), "bond_columns": list(chemistry.BOND_COLUMNS),
                "files": files,
            }
            with interval(budget, "feature-source:publish-manifest:" + source.parent.name):
                atomic_json(temporary / "manifest.json", manifest)
                if (temporary / "progress.json").exists():
                    (temporary / "progress.json").unlink()
                os.replace(temporary, directory)
            return _source_reference(directory, manifest)
        except BaseException:
            # This process owns this unpublished temporary directory only.
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def _source_reference(directory, manifest):
    return {"directory": str(directory.resolve()),
            "manifest_sha256": sha256(directory / "manifest.json"),
            "source_sha256": manifest["source_sha256"],
            "complete": manifest["complete"], "graph_count": manifest["graph_count"],
            "selected_rows": manifest["selected_rows"], "prepared_rows": manifest["prepared_rows"]}


def prepare_shared_features(graph_index, cache_root, splits=("train", "val"),
                            limit_sources=None, max_records_per_source=None, budget=None):
    """Return a stable, versioned manifest with source->sidecar lookup.

    Test-source preparation is opt-in by explicitly including ``test`` in
    ``splits``; the CLI additionally requires ``--allow-test``. This function
    performs no model inference or metric computation.
    """
    graph_index, cache_root = Path(graph_index).resolve(), Path(cache_root).resolve()
    splits = tuple(sorted(set(splits)))
    if not splits or not set(splits).issubset({"train", "val", "test"}):
        raise ValueError("Feature preparation requires explicit valid splits")
    for value in (limit_sources, max_records_per_source):
        if value is not None and value < 1:
            raise ValueError("Preparation limits must be positive")
    cache_root.mkdir(parents=True, exist_ok=True)
    with interval(budget, "feature-cohort:read-index"):
        graph_index_sha = sha256(graph_index)
        with graph_index.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"source_id", "molecule_id", "graph_file", "graph_index", "split"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Graph index lacks fixed-cohort provenance fields")
            rows = list(reader)
        if len({row["source_id"] for row in rows}) != len(rows):
            raise ValueError("Cohort contains duplicate source IDs")
        groups = defaultdict(dict)
        for row in rows:
            if row["split"] not in splits:
                continue
            source = str(Path(row["graph_file"]).resolve())
            number = int(row["graph_index"])
            if number in groups[source]:
                raise ValueError("Cohort repeats a source graph number")
            groups[source][number] = row
        gat_reference = Path(__file__).resolve().parent / "train_gatv2_search.py"
        code_pins = {"encoder_sha256": sha256(chemistry.__file__), "preparer_sha256": sha256(__file__),
                     "gat_reference_sha256": sha256(gat_reference)}
    if not groups:
        raise ValueError("Requested cohort splits contain no molecules")
    sources = sorted(groups)
    chosen_sources = sources[:limit_sources] if limit_sources is not None else sources
    source_specs = {}
    prepared_counts = Counter()
    for source in chosen_sources:
        source_specs[source] = _build_source(source, groups[source], graph_index_sha, cache_root,
                                             code_pins, max_records_per_source, budget)
        numbers = sorted(groups[source])
        if max_records_per_source is not None:
            numbers = numbers[:max_records_per_source]
        prepared_counts.update(groups[source][number]["split"] for number in numbers)
    requested_counts = Counter(row["split"] for row in rows if row["split"] in splits)
    manifest = {
        "format_version": FORMAT_VERSION, "feature_version": chemistry.FEATURE_VERSION,
        "gat_reference_feature_version": chemistry.GAT_REFERENCE_FEATURE_VERSION,
        "alignment_version": chemistry.ALIGNMENT_VERSION,
        "graph_index": str(graph_index), "graph_index_sha256": graph_index_sha,
        "splits": list(splits), "sources": source_specs, "code_pins": code_pins,
        "rdkit_version": chemistry.RDKIT_VERSION,
        "counts": dict(sorted(prepared_counts.items())),
        "requested_counts": dict(sorted(requested_counts.items())),
        "cohort_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "complete": len(chosen_sources) == len(sources)
                    and dict(prepared_counts) == dict(requested_counts)
                    and all(value["complete"] for value in source_specs.values()),
        "test_feature_preparation_requested": "test" in splits,
    }
    filename = cache_root / ("cohort_" + json_digest(manifest) + ".json")
    manifest["manifest_path"] = str(filename)
    with interval(budget, "feature-cohort:publish-manifest"):
        atomic_json(filename, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-index", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--cache-root", type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    parser.add_argument("--allow-test", action="store_true", help="Explicitly authorize preparing test chemistry, without inference")
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--max-records-per-source", type=int)
    args = parser.parse_args(argv)
    if "test" in args.splits and not args.allow_test:
        parser.error("Preparing test chemistry requires explicit --allow-test")
    RDLogger.DisableLog("rdApp.*")
    # The CLI always respects the exclusive compute lease and cooling policy.
    from shared_geometry_resources_v2 import ComputeBudget, exclusive_compute
    status_dir = args.cache_root.resolve() / "preparation_status"
    with exclusive_compute(Path(__file__).resolve().parent, status_dir):
        with ComputeBudget(status_dir) as budget:
            manifest = prepare_shared_features(args.graph_index, args.cache_root, args.splits,
                                               args.limit_sources, args.max_records_per_source, budget)
    print(json.dumps({"manifest_path": manifest["manifest_path"], "complete": manifest["complete"],
                      "counts": manifest["counts"], "requested_counts": manifest["requested_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
