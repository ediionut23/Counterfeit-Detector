#!/usr/bin/env python3
"""Distance-dependent invariant GNN with lazy conformer-file geometry access.

Coordinates come exclusively from the existing conformer NPZ files. They are
ETKDG/force-field predictions, not experimentally measured product geometries.
The supervised labels remain generated analogue versus public reference. The
model is invariant to rigid rotations/translations through scalar distances;
it is not a coordinate-equivariant model or a validated counterfeit detector.

Training never evaluates test. Each epoch atomically saves last.pt and, when
validation improves, best.pt; decision thresholds are chosen only on validation.
Run --evaluate-test explicitly after the model/configuration search is frozen.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sqlite3
import tempfile
import time
import zipfile
from collections import Counter, OrderedDict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


LOGGER = logging.getLogger("geometric_gnn")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
FORMAT_VERSION = 1
GEOMETRY_VERSION = "heavy_atom_distance_radius_v1"
NEEDED_ARRAYS = ("atom_features", "positions_angstrom", "atom_ptr", "radius_ptr",
                 "radius_edge_index", "source_id", "molecule_id", "label", "split",
                 "optimization_converged")
SPLITS = ("train", "val", "test")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def readonly(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def close_arrays(arrays):
    for array in arrays.values():
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def spatial_edges(positions, atom_features, stored_edges, cutoff, max_neighbors, heavy_atoms):
    """Filter actual stored radius edges, preserving their atom provenance.

    Incoming neighbors are truncated by distance with source-index tie-breaking.
    No spatial neighbor beyond the stored source radius is fabricated.
    """
    mask = atom_features[:, 0] > 1 if heavy_atoms else np.ones(len(atom_features), dtype=bool)
    if not mask.any():
        raise ValueError("Geometry contains no selected atoms")
    mapping = np.full(len(mask), -1, dtype=np.int32)
    mapping[mask] = np.arange(int(mask.sum()), dtype=np.int32)
    if stored_edges.size == 0:
        return mask, np.empty((2, 0), dtype=np.int32)
    if stored_edges.min() < 0 or stored_edges.max() >= len(mask):
        raise ValueError("Stored radius-edge indices cross molecular boundaries")
    src, dst = stored_edges
    keep = mask[src] & mask[dst]
    src, dst = src[keep], dst[keep]
    distances = np.linalg.norm(positions[src] - positions[dst], axis=1)
    keep = (distances > 1e-8) & (distances <= cutoff)
    src, dst, distances = src[keep], dst[keep], distances[keep]
    if len(src):
        order = np.lexsort((src, distances, dst))
        src, dst = src[order], dst[order]
        starts = np.r_[0, np.flatnonzero(np.diff(dst)) + 1]
        lengths = np.diff(np.r_[starts, len(dst)])
        ranks = np.arange(len(dst)) - np.repeat(starts, lengths)
        keep = ranks < max_neighbors
        src, dst = src[keep], dst[keep]
    return mask, np.vstack((mapping[src], mapping[dst])).astype(np.int32, copy=False)


def _cache_arrays(directory):
    return {name: np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            for name in ("atom_features", "positions_angstrom", "atom_ptr", "edge_ptr", "edge_index",
                         "source_id", "molecule_id", "label", "split", "optimization_converged")}


def _metadata_arrays(directory):
    return {name: np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            for name in ("source_id", "molecule_id", "label", "split", "optimization_converged")}


def materialize_geometry(source, cache_root, source_sha, geometry):
    """Create a reusable memmapped cache with bounded per-molecule work buffers."""
    key = json_digest({"source_sha256": source_sha, "geometry": geometry})
    final = cache_root / "geometry" / key
    final.parent.mkdir(parents=True, exist_ok=True)
    with (final.parent / (key + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = final / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["source_sha256"] != source_sha or manifest["geometry"] != geometry:
                raise ValueError("Existing geometry cache metadata disagrees")
            for name, spec in manifest["files"].items():
                path = final / name
                if not path.exists() or path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
                    raise ValueError(f"Geometry cache changed: {path}")
            return final, manifest
        if final.exists():
            raise ValueError("Incomplete final geometry cache directory; inspect it before reusing")
        LOGGER.info("Preparing actual geometry cache: %s", source)
        with tempfile.TemporaryDirectory(prefix=".geometric_build_", dir=final.parent) as temporary:
            temporary = Path(temporary)
            raw_dir, derived = temporary / "raw", temporary / "derived"
            raw_dir.mkdir()
            derived.mkdir()
            with zipfile.ZipFile(source) as archive:
                for name in NEEDED_ARRAYS:
                    member = name + ".npy"
                    if member not in archive.namelist():
                        raise ValueError(f"Source geometry lacks {member}; no fallback coordinates are created")
                    with archive.open(member) as reader, (raw_dir / member).open("wb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
            raw = {name: np.load(raw_dir / (name + ".npy"), mmap_mode="r", allow_pickle=False)
                   for name in NEEDED_ARRAYS}
            try:
                n_graphs = len(raw["label"])
                if not all(len(raw[k]) == n_graphs for k in ("source_id", "molecule_id", "split", "optimization_converged")):
                    raise ValueError("Geometry metadata lengths disagree")
                if raw["positions_angstrom"].shape != (len(raw["atom_features"]), 3):
                    raise ValueError("Geometry position shape disagrees with atom features")
                if not np.isfinite(raw["positions_angstrom"]).all():
                    raise ValueError("Nonfinite coordinates in source geometry")
                node_ptr, edge_ptr = [0], [0]
                for i in range(n_graphs):
                    begin, end = map(int, raw["atom_ptr"][i:i+2])
                    eb, ee = map(int, raw["radius_ptr"][i:i+2])
                    mask, edges = spatial_edges(raw["positions_angstrom"][begin:end], raw["atom_features"][begin:end],
                        np.asarray(raw["radius_edge_index"][:, eb:ee]) - begin,
                        geometry["cutoff_angstrom"], geometry["max_neighbors"], geometry["heavy_atoms"])
                    node_ptr.append(node_ptr[-1] + int(mask.sum()))
                    edge_ptr.append(edge_ptr[-1] + edges.shape[1])
                shapes = {"atom_features": (node_ptr[-1], 8), "positions_angstrom": (node_ptr[-1], 3),
                          "edge_index": (2, edge_ptr[-1])}
                arrays = {name: np.lib.format.open_memmap(derived / (name + ".npy"), mode="w+",
                    dtype=np.float32 if name == "positions_angstrom" else np.int32 if name == "edge_index" else np.int16,
                    shape=shape) for name, shape in shapes.items()}
                try:
                    for i in range(n_graphs):
                        begin, end = map(int, raw["atom_ptr"][i:i+2])
                        eb, ee = map(int, raw["radius_ptr"][i:i+2])
                        mask, edges = spatial_edges(raw["positions_angstrom"][begin:end], raw["atom_features"][begin:end],
                            np.asarray(raw["radius_edge_index"][:, eb:ee]) - begin,
                            geometry["cutoff_angstrom"], geometry["max_neighbors"], geometry["heavy_atoms"])
                        nb, ne = node_ptr[i:i+2]
                        rb, re = edge_ptr[i:i+2]
                        arrays["atom_features"][nb:ne] = raw["atom_features"][begin:end][mask]
                        arrays["positions_angstrom"][nb:ne] = raw["positions_angstrom"][begin:end][mask]
                        arrays["edge_index"][:, rb:re] = edges + nb
                    for array in arrays.values():
                        array.flush()
                finally:
                    close_arrays(arrays)
                np.save(derived / "atom_ptr.npy", np.asarray(node_ptr, dtype=np.int64))
                np.save(derived / "edge_ptr.npy", np.asarray(edge_ptr, dtype=np.int64))
                for name in ("source_id", "molecule_id", "label", "split", "optimization_converged"):
                    shutil.copyfile(raw_dir / (name + ".npy"), derived / (name + ".npy"))
                files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                         for path in sorted(derived.glob("*.npy"))}
                manifest = {"source_path": str(source), "source_sha256": source_sha,
                    "geometry": geometry, "source_graphs": n_graphs, "selected_atoms": node_ptr[-1],
                    "spatial_edges": edge_ptr[-1], "files": files,
                    "coordinates": "stored ETKDG/MMFF94s or UFF conformers; no fabricated/fallback coordinates",
                    "nonconverged_source_graphs": int((~np.asarray(raw["optimization_converged"], dtype=bool)).sum())}
                atomic_json(derived / "manifest.json", manifest)
            finally:
                close_arrays(raw)
            os.replace(derived, final)
        return final, manifest


def _prepare_data(root, graph_index, cache_root, geometry):
    root, graph_index, cache_root = Path(root).resolve(), Path(graph_index).resolve(), Path(cache_root).resolve()
    if not graph_index.exists():
        raise FileNotFoundError(f"Missing explicit cohort graph index: {graph_index}")
    cohort_manifest = graph_index.parent / "manifest.json"
    audit = json.loads(cohort_manifest.read_text()) if cohort_manifest.exists() else {}
    selected_spec = audit.get("selected_files", {}).get(graph_index.name)
    if selected_spec and sha256(graph_index) != selected_spec["sha256"]:
        raise ValueError("Strict audited cohort index SHA256 changed")
    known = {}
    original_manifest = root / "manifest_sha256.json"
    if original_manifest.exists():
        known = json.loads(original_manifest.read_text()).get("files", {})
    files = {}
    with graph_index.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source_id", "molecule_id", "graph_file", "graph_index", "label", "split"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Graph index lacks {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            path = Path(row["graph_file"])
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Graph path lies outside declared 3D root: {path}")
            files[str(path)] = None
    for path in sorted(files):
        path_obj = Path(path)
        digest = sha256(path_obj)
        prior = known.get(str(path_obj.relative_to(root)))
        if prior and digest != prior["sha256"]:
            raise ValueError(f"Original audited NPZ hash changed: {path}")
        audit_source = audit.get("graph_shards", {}).get(str(path_obj)) or audit.get("selected_npz_files", {}).get(str(path_obj))
        # Audit manifests have used both names; support either without relaxing a pin.
        audit_source = audit_source or audit.get("npz_files", {}).get(str(path_obj))
        if audit_source and digest != audit_source["sha256"]:
            raise ValueError(f"Strict cohort source NPZ hash changed: {path}")
        cached, manifest = materialize_geometry(path_obj, cache_root, digest, geometry)
        files[path] = {"source_sha256": digest, "cache_dir": str(cached), "cache_manifest_sha256": sha256(cached / "manifest.json")}
    data_spec = {"root": str(root), "graph_index": str(graph_index), "graph_index_sha256": sha256(graph_index),
        "cohort_manifest_sha256": sha256(cohort_manifest) if cohort_manifest.exists() else None,
        "geometry": geometry, "source_files": files}
    key = json_digest(data_spec)
    index_dir = cache_root / "cohorts" / key
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index.sqlite"
    if not index_path.exists() or not (index_dir / "summary.json").exists():
        temporary = index_dir / "index.sqlite.tmp"
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript("""CREATE TABLE records(split TEXT NOT NULL,local_index INTEGER NOT NULL,
              source_id TEXT UNIQUE NOT NULL,molecule_id TEXT NOT NULL,graph_file TEXT NOT NULL,graph_number INTEGER NOT NULL,
              label INTEGER NOT NULL,sample_weight REAL NOT NULL,method TEXT,category TEXT,subcategory TEXT,
              optimization_converged INTEGER NOT NULL,PRIMARY KEY(split,local_index));
              CREATE INDEX by_shard ON records(split,graph_file,local_index);""")
            counts, labels, not_converged = Counter(), Counter(), Counter()
            metadata = {path: _metadata_arrays(Path(spec["cache_dir"])) for path, spec in files.items()}
            bucket = []
            with graph_index.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    split, label = row["split"], int(row["label"])
                    if split not in SPLITS or label not in {0, 1}:
                        raise ValueError("Invalid label/split in fixed graph index")
                    weight = float(row.get("sample_weight", "1") or 1)
                    if not math.isfinite(weight) or weight <= 0:
                        raise ValueError("Invalid sample weight")
                    path = Path(row["graph_file"])
                    path = str(path.resolve() if path.is_absolute() else (root / path).resolve())
                    arrays = metadata[path]
                    number = int(row["graph_index"])
                    if not 0 <= number < len(arrays["source_id"]):
                        raise ValueError("Graph number out of source range")
                    if str(arrays["source_id"][number]) != row["source_id"] or int(arrays["label"][number]) != label:
                        raise ValueError(f"Fixed index does not match stored source identity/label: {row['source_id']}")
                    if str(arrays["split"][number]) != split:
                        raise ValueError("Fixed index attempts to move a molecule to another source split")
                    if str(arrays["molecule_id"][number]).removeprefix("mol_") != row["molecule_id"].removeprefix("mol_"):
                        raise ValueError("Fixed index and source molecular identifiers disagree")
                    converged = int(arrays["optimization_converged"][number])
                    bucket.append((split, counts[split], row["source_id"], row["molecule_id"], path, number,
                        label, weight, row.get("generation_method", ""),
                        row.get("balancing_category") or row.get("category", ""),
                        row.get("balancing_subcategory") or row.get("subcategory", ""), converged))
                    counts[split] += 1
                    labels[split, label] += 1
                    not_converged[split] += not converged
                    if len(bucket) >= 2048:
                        connection.executemany("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", bucket)
                        bucket = []
            for arrays in metadata.values():
                close_arrays(arrays)
            if bucket:
                connection.executemany("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", bucket)
            connection.commit()
            summary = {"data_spec": data_spec, "fingerprint": key, "counts": dict(counts),
                "label_counts": {split: {str(label): labels[split, label] for label in (0, 1)} for split in SPLITS},
                "nonconverged_stored_conformers_used": dict(not_converged),
                "additional_geometry_failures_silently_dropped": 0,
                "model_label_semantics": {"0": "public reference", "1": "generated analogue; not confirmed counterfeit"}}
            digest = hashlib.sha256()
            digest.update(b"[")
            for number, record in enumerate(connection.execute("SELECT source_id,split,label FROM records ORDER BY source_id")):
                if number:
                    digest.update(b",")
                digest.update(json.dumps([record[0], record[1], str(record[2])], separators=(",", ":")).encode())
            digest.update(b"]")
            summary["cohort_sha256"] = digest.hexdigest()
        finally:
            connection.close()
        os.replace(temporary, index_path)
        summary["index_sha256"] = sha256(index_path)
        atomic_json(index_dir / "summary.json", summary)
    summary = json.loads((index_dir / "summary.json").read_text())
    if summary["fingerprint"] != key:
        raise ValueError("Cached cohort fingerprint mismatch")
    if summary.get("index_sha256") != sha256(index_path):
        raise ValueError("Cached cohort SQLite index changed")
    if audit.get("cohort_sha256") and summary["cohort_sha256"] != audit["cohort_sha256"]:
        raise ValueError("Selected cohort identities/labels/splits disagree with audited cohort hash")
    return index_path, summary


def prepare_data(root, graph_index, cache_root, geometry):
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    # Cache builds can be shared by separate configurations; only one writer
    # prepares a given cache root, including its cohort index, at a time.
    with (cache_root / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_data(root, graph_index, cache_root, geometry)


class GeometryDataset(Dataset):
    def __init__(self, index_path, data_summary, split, cache_shards=2):
        if split not in SPLITS or cache_shards not in {1, 2}:
            raise ValueError("Invalid split or cache size")
        self.index_path, self.summary, self.split = Path(index_path), data_summary, split
        self.cache_shards, self._cache = cache_shards, OrderedDict()
        self._connection, self._pid = None, None

    def __len__(self):
        return int(self.summary["counts"].get(self.split, 0))

    def connection(self):
        if self._connection is None or self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            for arrays in self._cache.values():
                close_arrays(arrays)
            self._cache.clear()
            self._connection, self._pid = readonly(self.index_path), os.getpid()
        return self._connection

    def shards(self):
        return [row[0] for row in self.connection().execute("SELECT DISTINCT graph_file FROM records WHERE split=? ORDER BY graph_file", (self.split,))]

    def indices_for_shard(self, source):
        return [row[0] for row in self.connection().execute("SELECT local_index FROM records WHERE split=? AND graph_file=? ORDER BY local_index", (self.split, source))]

    def __getitem__(self, item):
        item = int(item)
        row = self.connection().execute("SELECT * FROM records WHERE split=? AND local_index=?", (self.split, item)).fetchone()
        if row is None:
            raise IndexError(item)
        source = row["graph_file"]
        if source not in self._cache:
            directory = Path(self.summary["data_spec"]["source_files"][source]["cache_dir"])
            self._cache[source] = _cache_arrays(directory)
            while len(self._cache) > self.cache_shards:
                _, arrays = self._cache.popitem(last=False)
                close_arrays(arrays)
        self._cache.move_to_end(source)
        arrays, number = self._cache[source], row["graph_number"]
        begin, end = map(int, arrays["atom_ptr"][number:number+2])
        eb, ee = map(int, arrays["edge_ptr"][number:number+2])
        features = np.asarray(arrays["atom_features"][begin:end], dtype=np.float32)
        z = features[:, 0].astype(np.int64)
        if z.min() < 1 or z.max() > 118:
            raise ValueError("Unsupported atomic number in source geometry")
        scalar = features[:, 1:].copy()
        scalar /= np.asarray([4, 8, 1, 8, 4, 4, 1], dtype=np.float32)
        return Data(z=torch.from_numpy(z), scalar_features=torch.from_numpy(scalar),
            pos=torch.from_numpy(np.asarray(arrays["positions_angstrom"][begin:end]).copy()),
            edge_index=torch.from_numpy((np.asarray(arrays["edge_index"][:, eb:ee], dtype=np.int64) - begin).copy()),
            y=torch.tensor([row["label"]], dtype=torch.float32),
            sample_weight=torch.tensor([row["sample_weight"]], dtype=torch.float32),
            num_nodes=len(z),
            source_id=row["source_id"], molecule_id=row["molecule_id"],
            optimization_converged=torch.tensor([row["optimization_converged"]], dtype=torch.bool))

    def close(self):
        for arrays in self._cache.values():
            close_arrays(arrays)
        self._cache.clear()
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_connection=None, _pid=None, _cache=OrderedDict())
        return state


class GeometryBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False, seed=42):
        self.dataset, self.batch_size, self.shuffle, self.seed, self.epoch = dataset, batch_size, shuffle, seed, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        with closing(readonly(self.dataset.index_path)) as connection:
            sizes = [row[0] for row in connection.execute("SELECT COUNT(*) FROM records WHERE split=? GROUP BY graph_file", (self.dataset.split,))]
        return sum(math.ceil(size / self.batch_size) for size in sizes)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shards = self.dataset.shards()
        if self.shuffle:
            rng.shuffle(shards)
        for source in shards:
            indices = self.dataset.indices_for_shard(source)
            if self.shuffle:
                rng.shuffle(indices)
            for begin in range(0, len(indices), self.batch_size):
                yield indices[begin:begin+self.batch_size]


class RadialInteraction(nn.Module):
    def __init__(self, hidden, radial_basis, dropout):
        super().__init__()
        self.filter = nn.Sequential(nn.Linear(radial_basis, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.update = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, features, radial, cutoff_values, edges):
        src, dst = edges
        messages = self.source(features)[src] * self.filter(radial) * cutoff_values[:, None]
        aggregate = torch.zeros_like(features)
        aggregate.index_add_(0, dst, messages)
        counts = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
        counts.index_add_(0, dst, torch.ones_like(dst, dtype=features.dtype))
        aggregate = aggregate / counts.clamp_min(1).sqrt()[:, None]
        return self.norm(features + self.update(aggregate))


class DistanceGNN(nn.Module):
    """Scalar continuous-filter message passing; rigid-motion invariant."""
    def __init__(self, hidden=64, interactions=3, radial_basis=16, cutoff=4.5, dropout=0.1):
        super().__init__()
        self.cutoff = float(cutoff)
        self.embedding = nn.Embedding(119, hidden)
        self.scalar_projection = nn.Linear(7, hidden)
        self.register_buffer("centres", torch.linspace(0, cutoff, radial_basis))
        self.gamma = float((radial_basis / cutoff) ** 2)
        self.interactions = nn.ModuleList([RadialInteraction(hidden, radial_basis, dropout) for _ in range(interactions)])
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, data):
        features = self.embedding(data.z) + self.scalar_projection(data.scalar_features)
        src, dst = data.edge_index
        distance = torch.linalg.vector_norm(data.pos[src] - data.pos[dst], dim=1)
        radial = torch.exp(-self.gamma * (distance[:, None] - self.centres[None, :]) ** 2)
        cutoff_values = 0.5 * (torch.cos(math.pi * distance.clamp(max=self.cutoff) / self.cutoff) + 1)
        cutoff_values = cutoff_values * (distance < self.cutoff).to(distance.dtype)
        for interaction in self.interactions:
            features = interaction(features, radial, cutoff_values, data.edge_index)
        batch = data.batch if getattr(data, "batch", None) is not None else torch.zeros(len(features), device=features.device, dtype=torch.long)
        n_graphs = data.num_graphs if hasattr(data, "num_graphs") else 1
        summed = torch.zeros((n_graphs, features.shape[1]), device=features.device, dtype=features.dtype)
        summed.index_add_(0, batch, features)
        counts = torch.zeros(n_graphs, device=features.device, dtype=features.dtype)
        counts.index_add_(0, batch, torch.ones(len(features), device=features.device, dtype=features.dtype))
        pooled = torch.cat((summed / counts.clamp_min(1)[:, None], summed / counts.clamp_min(1).sqrt()[:, None]), dim=1)
        return self.head(pooled).reshape(-1)


def seed_everything(seed, threads=2):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    numpy = np.random.get_state()
    state = {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def model_config(args):
    return {"hidden": args.hidden, "interactions": args.interactions, "radial_basis": args.radial_basis,
            "cutoff": args.cutoff, "dropout": args.dropout}


def tiny_geometry():
    return Data(z=torch.tensor([6, 7, 8]), scalar_features=torch.zeros((3, 7)),
        pos=torch.tensor([[0., 0., 0.], [1.3, 0., 0.], [1.3, 1.2, 0.]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]]), num_nodes=3)


def choose_device(requested):
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    choices = [requested] if requested != "auto" else (["cuda"] if available["cuda"] else ["mps", "cpu"] if available["mps"] else ["cpu"])
    for choice in choices:
        if not available.get(choice, False):
            if requested != "auto":
                raise ValueError(f"Requested device {choice} is unavailable")
            continue
        try:
            device = torch.device(choice)
            probe = DistanceGNN(hidden=8, interactions=1, radial_basis=4, dropout=0).to(device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
            loss = probe(tiny_geometry().to(device)).square().sum()
            loss.backward()
            optimizer.step()
            if choice == "mps":
                torch.mps.synchronize()
            if not math.isfinite(float(loss.detach().cpu())):
                raise ValueError("Device probe produced nonfinite loss")
            LOGGER.info("Device %s passed an actual distance-message forward/backward/update probe", choice)
            return device
        except (RuntimeError, ValueError) as error:
            if requested != "auto":
                raise
            LOGGER.warning("Device %s probe failed; selecting CPU: %s", choice, error)
    raise RuntimeError("No usable training device")


def make_loader(dataset, args, epoch=0, shuffle=False):
    sampler = GeometryBatchSampler(dataset, args.batch_size, shuffle=shuffle, seed=args.seed)
    sampler.set_epoch(epoch)
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
        generator=generator, persistent_workers=False, pin_memory=False)


def threshold_on_validation(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    return float(thresholds[best[-1]])  # Deterministic tie: largest threshold.


def classification_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= threshold
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    return {"n": len(labels), "threshold": float(threshold), "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)), "fnr": fn / positives if positives else None,
        "fpr": fp / negatives if negatives else None, "false_negatives": fn, "false_positives": fp,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if positives and negatives else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if positives else None}


@torch.no_grad()
def predict(model, dataset, args, device):
    model.eval()
    ids, molecules, labels, probabilities, converged = [], [], [], [], []
    loss_sum, count = 0.0, 0
    for batch in make_loader(dataset, args):
        batch = batch.to(device)
        logits = model(batch)
        loss_sum += float(F.binary_cross_entropy_with_logits(logits, batch.y.reshape(-1), reduction="sum").cpu())
        count += batch.num_graphs
        ids.extend(batch.source_id)
        molecules.extend(batch.molecule_id)
        labels.extend(batch.y.cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        converged.extend(batch.optimization_converged.cpu().tolist())
    if not count:
        raise ValueError(f"Empty evaluation split: {dataset.split}")
    return {"source_id": ids, "molecule_id": molecules, "label": np.asarray(labels, dtype=np.int64),
        "probability": np.asarray(probabilities, dtype=np.float64), "optimization_converged": converged,
        "loss": loss_sum / count}


def write_predictions(path, predictions, split, threshold):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_id", "molecule_id", "split", "label", "probability", "threshold", "prediction", "optimization_converged"])
        for number, source in enumerate(predictions["source_id"]):
            probability = float(predictions["probability"][number])
            writer.writerow([source, predictions["molecule_id"][number], split, int(predictions["label"][number]),
                format(probability, ".10g"), format(threshold, ".10g"), int(probability >= threshold), int(predictions["optimization_converged"][number])])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_history(out, history):
    atomic_json(out / "history.json", history)
    temporary = out / "metrics.csv.tmp"
    fields = ["epoch", "train_loss", "val_loss", "val_f1", "val_auc", "val_precision", "val_recall", "val_f1_at_0_5", "threshold", "learning_rate", "epoch_seconds", "best_epoch", "best_validation_f1"]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, out / "metrics.csv")


def train_epoch(model, optimizer, dataset, args, device, epoch, out, best_f1):
    model.train()
    numerator, denominator = 0.0, 0.0
    started = time.monotonic()
    for batch_number, batch in enumerate(make_loader(dataset, args, epoch, shuffle=True), 1):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        losses = F.binary_cross_entropy_with_logits(logits, batch.y.reshape(-1), reduction="none")
        weights = batch.sample_weight.reshape(-1) if args.weight_mode == "row" else torch.ones_like(losses)
        # Mean weights are one in the fixed cohort. A fixed batch-size denominator
        # gives each row its declared importance, without a second class weighting.
        loss = (losses * weights).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss in epoch {epoch}, batch {batch_number}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        numerator += float((losses.detach() * weights).sum().cpu())
        denominator += float(weights.sum().cpu())
        if batch_number % 100 == 0:
            atomic_json(out / "progress.json", {"status": "training", "current_epoch": epoch,
                "completed": False, "batch": batch_number, "best_validation_f1": float(best_f1),
                "train_loss_so_far": numerator / denominator, "epoch_seconds_so_far": time.monotonic() - started})
    return numerator / denominator


def locked_config(args, data_summary, device):
    return {"format_version": FORMAT_VERSION, "script_sha256": sha256(__file__),
        "model": model_config(args), "data_fingerprint": data_summary["fingerprint"],
        "cohort_sha256": data_summary["cohort_sha256"], "seed": args.seed,
        "selection": args.selection, "threshold_policy": "maximum validation F1; largest-threshold tie",
        "weight_mode": args.weight_mode, "batch_size": args.batch_size, "learning_rate": args.lr,
        "weight_decay": args.weight_decay, "gradient_clip": args.gradient_clip,
        "min_delta": args.min_delta,
        "patience": args.patience, "scheduler_patience": args.scheduler_patience,
        "threads": args.threads, "num_workers": args.num_workers, "device": str(device),
        "label_semantics": data_summary["model_label_semantics"],
        "coordinates": "existing ETKDG force-field predicted coordinates; selected heavy atoms by default",
        "test_policy": "no test access during fitting; explicit locked final evaluation only"}


def fit(args, index_path, data_summary, device):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "test_selection_lock.json").exists():
        raise ValueError("This run has accessed test; use a new validation-only run directory")
    configuration = locked_config(args, data_summary, device)
    model = DistanceGNN(**configuration["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.scheduler_patience)
    history, start_epoch, best_score, best_f1, best_epoch, threshold, bad_epochs = [], 1, -1.0, 0.0, 0, 0.5, 0
    if args.resume:
        checkpoint = torch.load(Path(args.resume).resolve(), map_location="cpu", weights_only=True)
        if checkpoint["configuration"] != configuration:
            raise ValueError("Resume configuration/data/source code differs from pinned checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        history, start_epoch = checkpoint["history"], checkpoint["epoch"] + 1
        best_score, best_f1, best_epoch = checkpoint["best_score"], checkpoint["best_validation_f1"], checkpoint["best_epoch"]
        threshold, bad_epochs = checkpoint["best_threshold"], checkpoint["bad_epochs"]
        restore_rng(checkpoint["rng_state"])
        if not (out / "best.pt").exists():
            raise ValueError("Resume requires the same run directory containing its best.pt")
        LOGGER.info("Resuming after epoch %d; best validation F1 %.4f", start_epoch - 1, best_f1)
    elif (out / "last.pt").exists() or (out / "best.pt").exists():
        raise ValueError("Run directory already contains checkpoints; pass --resume explicitly")
    atomic_json(out / "configuration.json", configuration)
    atomic_json(out / "data_summary.json", data_summary)
    train, validation = [GeometryDataset(index_path, data_summary, split) for split in ("train", "val")]
    started = time.monotonic()
    current_epoch = start_epoch - 1
    reason = "maximum_epochs"
    try:
        if bad_epochs >= args.patience:
            reason = "early_stopping_already_reached"
        else:
            for epoch in range(start_epoch, args.epochs + 1):
                current_epoch = epoch
                epoch_start = time.monotonic()
                atomic_json(out / "progress.json", {"status": "training", "current_epoch": epoch,
                    "completed": False, "best_validation_f1": best_f1})
                train_loss = train_epoch(model, optimizer, train, args, device, epoch, out, best_f1)
                probabilities = predict(model, validation, args, device)
                candidate_threshold = threshold_on_validation(probabilities["label"], probabilities["probability"])
                metrics = classification_metrics(probabilities["label"], probabilities["probability"], candidate_threshold)
                fixed = classification_metrics(probabilities["label"], probabilities["probability"], 0.5)
                score = metrics["f1"] if args.selection == "val_f1" else metrics["roc_auc"]
                if score is None:
                    raise ValueError("Selected validation metric is undefined")
                improved = score > best_score + args.min_delta
                if improved:
                    best_score, best_f1, best_epoch, threshold, bad_epochs = score, metrics["f1"], epoch, candidate_threshold, 0
                else:
                    bad_epochs += 1
                scheduler.step(score)
                row = {"epoch": epoch, "train_loss": train_loss, "val_loss": probabilities["loss"],
                    "val_f1": metrics["f1"], "val_auc": metrics["roc_auc"], "val_precision": metrics["precision"],
                    "val_recall": metrics["recall"], "val_f1_at_0_5": fixed["f1"], "threshold": candidate_threshold,
                    "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.monotonic() - epoch_start,
                    "best_epoch": best_epoch, "best_validation_f1": best_f1}
                history.append(row)
                checkpoint = {"configuration": configuration, "epoch": epoch,
                    "model_state_dict": cpu_tree(model.state_dict()), "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
                    "scheduler_state_dict": scheduler.state_dict(), "rng_state": rng_state(), "history": history,
                    "best_score": best_score, "best_validation_f1": best_f1, "best_epoch": best_epoch,
                    "best_threshold": threshold, "bad_epochs": bad_epochs, "validation_metrics": metrics,
                    "validation_metrics_at_0_5": fixed, "target_validation_f1": args.target_f1}
                # last is the committed epoch boundary. A failed write never replaces it.
                if improved:
                    atomic_torch_save(out / "best.pt", checkpoint)
                    write_predictions(out / "validation_best_predictions.csv", probabilities, "val", candidate_threshold)
                atomic_torch_save(out / "last.pt", checkpoint)
                write_predictions(out / "validation_last_predictions.csv", probabilities, "val", candidate_threshold)
                write_history(out, history)
                status = {"status": "training", "current_epoch": epoch, "completed": False,
                    "best_validation_f1": best_f1, "best_epoch": best_epoch, "last_validation_f1": metrics["f1"],
                    "last_validation_auc": metrics["roc_auc"], "bad_epochs": bad_epochs, "threshold": threshold,
                    "elapsed_seconds_this_process": time.monotonic() - started, "target_validation_f1": args.target_f1,
                    "cohort_sha256": data_summary["cohort_sha256"], "test_evaluated": False}
                atomic_json(out / "progress.json", status)
                atomic_json(out / "summary.json", status)
                LOGGER.info("epoch %3d train_loss=%.5f val_f1=%.4f val_auc=%.4f threshold=%.4f best_f1=%.4f [%0.1fs]",
                    epoch, train_loss, metrics["f1"], metrics["roc_auc"], candidate_threshold, best_f1, row["epoch_seconds"])
                if bad_epochs >= args.patience:
                    reason = "early_stopping"
                    break
        final = {"status": "completed", "current_epoch": current_epoch, "completed": True,
            "completion_reason": reason, "best_validation_f1": best_f1, "best_epoch": best_epoch,
            "best_selection_metric": best_score, "best_threshold": threshold, "test_evaluated": False,
            "target_validation_f1": args.target_f1, "validation_target_reached": best_f1 >= args.target_f1,
            "elapsed_seconds_this_process": time.monotonic() - started, "cohort_sha256": data_summary["cohort_sha256"],
            "best_checkpoint": str(out / "best.pt"), "last_checkpoint": str(out / "last.pt"),
            "label_semantics": data_summary["model_label_semantics"]}
        atomic_json(out / "progress.json", final)
        atomic_json(out / "summary.json", final)
        return final
    except BaseException as error:
        atomic_json(out / "progress.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "current_epoch": current_epoch, "completed": False, "best_validation_f1": best_f1,
            "error": str(error), "resume_from": str(out / "last.pt")})
        raise
    finally:
        train.close()
        validation.close()


def _evaluate_test(args, index_path, data_summary, device):
    out = Path(args.out).resolve()
    checkpoint_path = Path(args.checkpoint or out / "best.pt").resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    if configuration["data_fingerprint"] != data_summary["fingerprint"]:
        raise ValueError("Test data/configuration differs from the checkpoint's fixed cohort/geometry")
    if configuration["model"] != model_config(args):
        raise ValueError("Test model arguments differ from selected checkpoint")
    if checkpoint["epoch"] != checkpoint["best_epoch"]:
        raise ValueError("Final evaluation requires the selected best checkpoint, not the last epoch")
    selection = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "data_fingerprint": data_summary["fingerprint"], "cohort_sha256": data_summary["cohort_sha256"],
        "best_epoch": checkpoint["best_epoch"], "best_validation_f1": checkpoint["best_validation_f1"],
        "threshold": checkpoint["best_threshold"], "selection": configuration["selection"],
        "test_used_for_selection": False}
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "test_selection_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != selection:
            raise ValueError("Test is already locked to another checkpoint; no test-guided reselection")
        if (out / "test_results.json").exists():
            LOGGER.info("Returning the existing locked test evaluation without recomputing")
            return json.loads((out / "test_results.json").read_text())
    else:
        atomic_json(lock_path, selection)
    model = DistanceGNN(**configuration["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = GeometryDataset(index_path, data_summary, "test")
    try:
        predictions = predict(model, dataset, args, device)
    finally:
        dataset.close()
    metrics = classification_metrics(predictions["label"], predictions["probability"], selection["threshold"])
    result = {**selection, "test_metrics": metrics,
        "test_metrics_at_0_5": classification_metrics(predictions["label"], predictions["probability"], 0.5),
        "test_loss": predictions["loss"], "target_f1": args.target_f1,
        "test_target_reached": metrics["f1"] >= args.target_f1, "label_semantics": data_summary["model_label_semantics"]}
    write_predictions(out / "test_predictions.csv", predictions, "test", selection["threshold"])
    atomic_json(out / "test_results.json", result)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    summary.update(completed=True, test_evaluated=True, test_metrics=metrics, best_validation_f1=checkpoint["best_validation_f1"],
        current_epoch=summary.get("current_epoch", checkpoint["epoch"]))
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "progress.json", summary)
    LOGGER.info("Locked final test: F1=%.4f precision=%.4f recall=%.4f AUC=%.4f threshold=%.4f",
        metrics["f1"], metrics["precision"], metrics["recall"], metrics["roc_auc"], selection["threshold"])
    return result


def evaluate_test(args, index_path, data_summary, device):
    """Hold a family-wide OS lock and immutable selection pin through test access."""
    out = Path(args.out).resolve()
    checkpoint_path = Path(args.checkpoint or out / "best.pt").resolve()
    family_path = Path(args.test_lock).resolve() if args.test_lock else out.parent / "geometric_test_selection.json"
    family_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(family_path) + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
               "cohort_sha256": data_summary["cohort_sha256"], "data_fingerprint": data_summary["fingerprint"]}
        if family_path.exists():
            if json.loads(family_path.read_text()) != pin:
                raise ValueError("Family test evaluation is already pinned to another candidate/cohort")
        else:
            atomic_json(family_path, pin)
        return _evaluate_test(args, index_path, data_summary, device)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="balanced_datasets/training_200k/3d")
    p.add_argument("--graph-index", default="balanced_datasets/training_200k/overnight_strict_v1/graph_index.csv")
    p.add_argument("--cache-root", default="training_cache/geometric_3d")
    p.add_argument("--out", default="training_runs/overnight/geometric")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--scheduler-patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--interactions", type=int, default=3)
    p.add_argument("--radial-basis", type=int, default=16)
    p.add_argument("--cutoff", type=float, default=4.5)
    p.add_argument("--max-neighbors", type=int, default=24)
    p.add_argument("--include-hydrogens", action="store_true")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gradient-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--selection", choices=("val_f1", "val_auc"), default="val_f1")
    p.add_argument("--weight-mode", choices=("row", "none"), default="row")
    p.add_argument("--target-f1", type=float, default=0.85)
    p.add_argument("--resume")
    p.add_argument("--no-test", action="store_true", help="Explicit protocol marker; fitting always excludes test")
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--test-lock", help="Shared family selection JSON; its OS lock is held throughout final test")
    p.add_argument("--prepare-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.cutoff <= 5.0:
        raise ValueError("Cutoff must be in (0,5] Angstrom; source radius edges stop at 5 Angstrom")
    if min(args.epochs, args.patience, args.hidden, args.interactions, args.radial_basis, args.max_neighbors,
           args.batch_size, args.threads) < 1 or args.num_workers < 0:
        raise ValueError("Epoch/model/batch/thread counts must be positive")
    if args.evaluate_test and (args.no_test or args.resume or args.prepare_only):
        raise ValueError("Final test cannot be combined with --no-test, --resume, or --prepare-only")
    seed_everything(args.seed, args.threads)
    geometry = {"version": GEOMETRY_VERSION, "cutoff_angstrom": args.cutoff,
        "max_neighbors": args.max_neighbors, "heavy_atoms": not args.include_hydrogens}
    index_path, data_summary = prepare_data(args.root, args.graph_index, args.cache_root, geometry)
    LOGGER.info("Fixed 3D cohort: %s; cohort SHA256=%s", data_summary["counts"], data_summary["cohort_sha256"])
    if args.prepare_only:
        print(json.dumps({"index": str(index_path), **data_summary}, indent=2))
        return data_summary
    device = choose_device(args.device)
    seed_everything(args.seed, args.threads)  # Device probing must not consume the training RNG.
    # One process owns each run directory; mmap geometry caches are separately shared/locked.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a+") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return evaluate_test(args, index_path, data_summary, device) if args.evaluate_test else fit(args, index_path, data_summary, device)


if __name__ == "__main__":
    main()
