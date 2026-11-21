#!/usr/bin/env python3
"""Train chemical-edge-aware geometry v2 on the fixed large cohort.

Uses immutable existing coordinates and the GAT-compatible sidecar chemistry.
Training is validation-only, durable and resumable. Production work holds the
same exclusive compute lease as Ax/generation and uses conservative cooling.
No historical trainer, conformer, cache or checkpoint is rewritten.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

import train_geometric_gnn as engine
import geometric_bond_model_v2 as models
import prepare_shared_geometric_features_v2 as features
import shared_molecular_features_v2 as encoder
import shared_geometry_resources_v2 as resources


WORKSPACE = Path(__file__).resolve().parent
COHORT_SHA256 = "041936ffe7ad260c110d4bc8e9226d4622d3d0e79ae023b89cae2d1b789e5126"
PROTOCOL_VERSION = "shared_gat_chemistry_geometry_validation_v2"
FEATURE_ARRAYS = ("z", "x", "atom_ptr", "bond_ptr", "bond_edge_index",
                  "bond_edge_attr", "selected_mask", "source_id", "molecule_id")
_ORIGINAL_DATASET = engine.GeometryDataset
_ORIGINAL_MODEL_CONFIG = engine.model_config
_ORIGINAL_LOCKED_CONFIG = engine.locked_config
_ORIGINAL_LOADER = engine.make_loader


def _feature_arrays(directory):
    return {name: np.load(Path(directory) / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            for name in FEATURE_ARRAYS}


class SharedGeometryDataset(_ORIGINAL_DATASET):
    """Keep native heavy-atom coordinate order, add independently verified chemistry."""
    def __init__(self, index_path, data_summary, split, cache_shards=2):
        if split not in {"train", "val"}:
            raise ValueError("This version exposes train/validation only; no test dataset")
        super().__init__(index_path, data_summary, split, cache_shards)
        self._chemical_cache = OrderedDict()
        self._chemical_pid = os.getpid()
        if not data_summary.get("shared_features", {}).get("complete"):
            raise ValueError("Shared chemical features are incomplete")

    def __getitem__(self, item):
        if self._chemical_pid != os.getpid():
            for arrays in self._chemical_cache.values():
                engine.close_arrays(arrays)
            self._chemical_cache.clear()
            self._chemical_pid = os.getpid()
        base = super().__getitem__(item)
        row = self.connection().execute(
            "SELECT * FROM records WHERE split=? AND local_index=?", (self.split, int(item))).fetchone()
        source, number = row["graph_file"], int(row["graph_number"])
        spec = self.summary["shared_features"]["sources"].get(source)
        if spec is None:
            raise ValueError(f"Source has no prepared chemical features: {source}")
        if source not in self._chemical_cache:
            self._chemical_cache[source] = _feature_arrays(spec["directory"])
            while len(self._chemical_cache) > self.cache_shards:
                _, expired = self._chemical_cache.popitem(last=False)
                engine.close_arrays(expired)
        self._chemical_cache.move_to_end(source)
        arrays = self._chemical_cache[source]
        if not 0 <= number < len(arrays["selected_mask"]) or not bool(arrays["selected_mask"][number]):
            raise ValueError("Chemical cache does not contain this selected graph")
        if str(arrays["source_id"][number]) != row["source_id"]:
            raise ValueError("Chemical cache source identity mismatch")
        if str(arrays["molecule_id"][number]).removeprefix("mol_") != row["molecule_id"].removeprefix("mol_"):
            raise ValueError("Chemical cache molecular identity mismatch")
        begin, end = map(int, arrays["atom_ptr"][number:number+2])
        eb, ee = map(int, arrays["bond_ptr"][number:number+2])
        z = np.asarray(arrays["z"][begin:end], dtype=np.int64)
        if end - begin != base.num_nodes or not np.array_equal(z, base.z.numpy()):
            raise ValueError("Chemical atom order disagrees with the original geometry")
        x = np.asarray(arrays["x"][begin:end], dtype=np.float32).copy()
        bonds = (np.asarray(arrays["bond_edge_index"][:, eb:ee], dtype=np.int64) - begin).copy()
        attributes = np.asarray(arrays["bond_edge_attr"][eb:ee], dtype=np.float32).copy()
        if x.shape != (base.num_nodes, 21) or attributes.shape != (bonds.shape[1], 14):
            raise ValueError("Shared feature dimensions are not GAT-compatible")
        if not np.isfinite(x).all() or not np.isfinite(attributes).all():
            raise ValueError("Nonfinite shared chemistry")
        return Data(z=base.z, x=torch.from_numpy(x), pos=base.pos,
            bond_edge_index=torch.from_numpy(bonds), bond_edge_attr=torch.from_numpy(attributes),
            spatial_edge_index=base.edge_index, y=base.y, sample_weight=base.sample_weight,
            num_nodes=base.num_nodes, source_id=base.source_id, molecule_id=base.molecule_id,
            optimization_converged=base.optimization_converged)

    def close(self):
        for arrays in self._chemical_cache.values():
            engine.close_arrays(arrays)
        self._chemical_cache.clear()
        super().close()

    def __getstate__(self):
        result = super().__getstate__()
        result.update(_chemical_cache=OrderedDict(), _chemical_pid=None)
        return result


def model_config(args):
    return {"hidden": args.hidden, "interactions": args.interactions,
            "radial_basis": args.radial_basis, "angular_basis": args.angular_basis,
            "cutoff": args.cutoff, "dropout": args.dropout,
            "geometry_variant": args.geometry_variant, "validate_inputs": True}


def locked_config(args, summary, device):
    result = _ORIGINAL_LOCKED_CONFIG(args, summary, device)
    modules = (engine, models, features, encoder, resources)
    result.update(script_sha256=engine.sha256(__file__),
                  protocol_version=PROTOCOL_VERSION,
                  source_pins={Path(module.__file__).name: engine.sha256(module.__file__) for module in modules},
                  shared_feature_version=summary["shared_features"].get("feature_version"),
                  node_features=21, bond_features=14,
                  edge_relations="separate covalent bonds and spatial neighbors",
                  thermal_policy="exclusive lease; accumulated15s/3xrest nominal,5s/11xrest fair;120s recovery",
                  test_policy="validation-only implementation; separate fresh holdout protocol required")
    # Dependencies implementing the active cooling and angular functions are pinned too.
    for filename in ("geometric_angular_model.py", "accumulated_generation_cooling.py", "generation_job_host.py"):
        result["source_pins"][filename] = engine.sha256(WORKSPACE / filename)
    return result


@contextmanager
def bound_shared_engine(budget=None):
    def make_loader(dataset, args, epoch=0, shuffle=False):
        loader = _ORIGINAL_LOADER(dataset, args, epoch, shuffle)
        return budget.batches(loader, f"{dataset.split}:epoch{epoch}") if budget else loader
    replacements = {"DistanceGNN": models.BondGeometryGNN, "GeometryDataset": SharedGeometryDataset,
                    "model_config": model_config, "locked_config": locked_config, "make_loader": make_loader}
    previous = {key: getattr(engine, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(engine, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(engine, key, value)


def shared_summary(index_path, original, feature_manifest, graph_index):
    """Publish a new fingerprint without changing the old index or coordinate cache."""
    if feature_manifest.get("complete") is not True:
        raise ValueError("Partial feature preparation cannot be used for training")
    if feature_manifest.get("graph_index_sha256") != engine.sha256(graph_index):
        raise ValueError("Shared chemical features belong to a different fixed graph index")
    if not {"train", "val"} <= set(feature_manifest.get("splits", [])):
        raise ValueError("Both train and validation features must be prepared")
    with closing(engine.readonly(index_path)) as connection:
        required = {row[0] for row in connection.execute(
            "SELECT DISTINCT graph_file FROM records WHERE split IN ('train','val')")}
    sources = feature_manifest.get("sources", {})
    if not required <= set(sources):
        raise ValueError("Chemical features are missing train/validation source shards")
    for source in sorted(required):
        spec = sources[source]
        path = Path(spec["directory"]) / "manifest.json"
        if not path.exists() or engine.sha256(path) != spec["manifest_sha256"]:
            raise ValueError("Chemical source manifest changed")
        meta = json.loads(path.read_text())
        if meta.get("source_sha256") != original["data_spec"]["source_files"][source]["source_sha256"]:
            raise ValueError("Chemical and geometry caches derive from different NPZ contents")
    summary = dict(original)
    summary["shared_features"] = feature_manifest
    summary["data_spec"] = {**original["data_spec"], "shared_features_sha256": engine.json_digest(feature_manifest),
                            "feature_protocol": PROTOCOL_VERSION}
    summary["fingerprint"] = engine.json_digest(summary["data_spec"])
    summary["additional_geometry_failures_silently_dropped"] = 0
    return summary


@contextmanager
def cooled_geometry_preparation(budget):
    """Bound checksum chunks and per-molecule radius work without editing the engine."""
    original_sha, original_edges = engine.sha256, engine.spatial_edges
    def cooled_sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while True:
                with budget.interval(f"geometry_hash:{Path(path).name}"):
                    block = stream.read(8 * 1024 * 1024)
                    if block:
                        digest.update(block)
                if not block:
                    break
        return digest.hexdigest()
    def cooled_edges(*args, **kwargs):
        with budget.interval("geometry_radius_one_molecule"):
            return original_edges(*args, **kwargs)
    engine.sha256, engine.spatial_edges = cooled_sha, cooled_edges
    try:
        yield
    finally:
        engine.sha256, engine.spatial_edges = original_sha, original_edges


def preflight_out(args):
    """Never publish sidecars or cooling state into an unrelated historical run."""
    out = Path(args.out).resolve()
    protocol = out / "protocol.json"
    config = out / "configuration.json"
    if protocol.exists():
        if json.loads(protocol.read_text()).get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Output directory belongs to another protocol")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Refusing nonempty output without this version's protocol marker")
    if config.exists() and json.loads(config.read_text()).get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Output contains a historical or incompatible model configuration")
    if args.prepare_only and any((out / name).exists() for name in ("best.pt", "last.pt")):
        raise ValueError("Use a separate output directory for preparation when checkpoints exist")
    if not args.resume and not args.prepare_only and any((out / name).exists() for name in ("best.pt", "last.pt")):
        raise ValueError("Run already has checkpoints; pass --resume explicitly")
    return out


def validate_resume_checkpoints(args, summary, device):
    """Reject foreign or inconsistent best/last artifacts before any training write."""
    if not args.resume:
        return
    out = Path(args.out).resolve()
    if Path(args.resume).resolve() != out / "last.pt":
        raise ValueError("Resume must use this run's committed last.pt checkpoint")
    best_path = out / "best.pt"
    if not best_path.is_file():
        raise ValueError("Resume requires the same run's best.pt")
    last = torch.load(args.resume, map_location="cpu", weights_only=True)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    expected = locked_config(args, summary, device)
    if last.get("configuration") != expected or best.get("configuration") != expected:
        raise ValueError("Resume checkpoint source/configuration/cohort pins disagree")
    if best.get("epoch") != last.get("best_epoch") or best.get("best_epoch") != last.get("best_epoch"):
        raise ValueError("best.pt and committed last.pt do not describe the same best epoch")
    for key in ("best_score", "best_validation_f1", "best_threshold"):
        if best.get(key) != last.get(key):
            raise ValueError(f"best.pt and last.pt disagree on {key}")


def parser():
    p = engine.parser()
    p.description = __doc__
    p.set_defaults(graph_index="balanced_datasets/training_200k/overnight_strict_converged_v2/graph_index.csv",
                   cache_root="training_cache/geometric_overnight_v2",
                   out="training_runs/shared_chemical_geometry_20261110/angular_seed42",
                   weight_mode="none", device="cpu", epochs=100, patience=15, no_test=True)
    p.add_argument("--shared-feature-cache", default=features.DEFAULT_CACHE)
    p.add_argument("--geometry-variant", choices=models.GEOMETRY_VARIANTS, default="angular")
    p.add_argument("--angular-basis", type=int, default=6)
    p.add_argument("--cohort-sha256", default=COHORT_SHA256)
    p.add_argument("--minimum-train-molecules", type=int, default=150000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.evaluate_test or args.checkpoint or args.test_lock:
        raise ValueError("This trainer is validation-only; no test/checkpoint-evaluation route is exposed")
    if args.device != "cpu" or not 1 <= args.threads <= 2 or args.num_workers != 0:
        raise ValueError("This version requires CPU,1–2 threads and zero loader workers")
    if args.include_hydrogens:
        raise ValueError("Shared GAT v2 chemistry uses the verified heavy-atom coordinate order")
    if not 0 < args.cutoff <= 5 or min(args.hidden, args.interactions, args.radial_basis,
            args.angular_basis, args.max_neighbors, args.batch_size, args.epochs, args.patience,
            args.minimum_train_molecules) < 1:
        raise ValueError("Invalid model, budget or geometry hyperparameter")
    out = preflight_out(args)
    with resources.exclusive_compute(WORKSPACE, out):
        out.mkdir(parents=True, exist_ok=True)
        with (out / ".run.lock").open("a+") as stream:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Recheck existing ownership before publishing any per-run state.
            marker = out / "protocol.json"
            if not marker.exists():
                if any(path.name != ".run.lock" for path in out.iterdir()):
                    raise ValueError("Output changed before protocol ownership was established")
                engine.atomic_json(marker, {"protocol_version": PROTOCOL_VERSION})
            else:
                preflight_out(args)
            with resources.ComputeBudget(out / "cooling") as budget:
                engine.seed_everything(args.seed, args.threads)
                feature_manifest = features.prepare_shared_features(args.graph_index, args.shared_feature_cache,
                                                                   splits=("train", "val"), budget=budget)
                geometry = {"version": engine.GEOMETRY_VERSION, "cutoff_angstrom": args.cutoff,
                            "max_neighbors": args.max_neighbors, "heavy_atoms": True}
                with cooled_geometry_preparation(budget):
                    index_path, original = engine.prepare_data(args.root, args.graph_index, args.cache_root, geometry)
                if original["cohort_sha256"] != args.cohort_sha256:
                    raise ValueError("Cohort identity/splits changed from the declared fixed protocol")
                if original["counts"].get("train", 0) < args.minimum_train_molecules:
                    raise ValueError("Full large training set required; partial/smoke feature caches are forbidden")
                with budget.interval("validate_shared_feature_manifest"):
                    summary = shared_summary(index_path, original, feature_manifest, args.graph_index)
                if args.prepare_only:
                    engine.atomic_json(out / "prepared_data_summary.json", summary)
                    print(json.dumps({"index": str(index_path), "counts": summary["counts"],
                                      "cohort_sha256": summary["cohort_sha256"],
                                      "feature_version": feature_manifest.get("feature_version"),
                                      "training_started": False}, indent=2))
                    return summary
                engine.seed_everything(args.seed, args.threads)
                with bound_shared_engine(budget):
                    validate_resume_checkpoints(args, summary, torch.device("cpu"))
                    return engine.fit(args, index_path, summary, torch.device("cpu"))


if __name__ == "__main__":
    main()
