#!/usr/bin/env python3
"""Bond-aware residual GATv2 experiments on fixed molecular splits.

Task: synthetic-origin discrimination (generated analogues versus public
reference molecules), not confirmed medicine fraud. Training uses validation
only. A separate locked test invocation applies the saved validation threshold.
Graph tensors are cached in bounded shards; model/optimizer/RNG checkpoints are
written atomically after every epoch. No downloaded code or model is executed.
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
import platform
import random
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit import rdBase
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gatv2_experiment")
TASK = "synthetic_origin_discrimination_not_confirmed_fraud"
FEATURE_VERSION = "atomic_number_chirality_rings_bond_order_stereo_v1"
DEFAULT_CSV = "balanced_datasets/training_200k/subcategory_weighted/dataset.csv"
META_FIELDS = ["source_id", "split", "label", "sample_weight", "generation_method", "category",
               "subcategory", "balancing_category", "balancing_subcategory"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path):
    """OS-lifetime lock: competing processes fail, crashed owners release it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another trainer/evaluator holds lock {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_rows(dataset_csv, graph_index=None):
    selected = None
    if graph_index:
        with Path(graph_index).open(newline="") as stream:
            selected = {}
            for row in csv.DictReader(stream):
                key = row["source_id"]
                if key in selected:
                    raise ValueError(f"Duplicate graph-index source_id: {key}")
                selected[key] = row
    rows, seen, matched = [], set(), set()
    with Path(dataset_csv).open(newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), 1):
            key = row.get("molecule_id") or row.get("source_id") or row.get("id")
            if not key:
                raise ValueError(f"Missing molecule/source ID at CSV row {row_number}")
            if selected is not None and key not in selected:
                continue
            if key in seen:
                raise ValueError(f"Duplicate selected molecular ID: {key}")
            seen.add(key)
            if row["split"] not in {"train", "val", "test"} or row["label"] not in {"0", "1"}:
                raise ValueError(f"Invalid split/label for {key}")
            index = selected[key] if selected is not None else None
            if index is not None:
                if index["split"] != row["split"] or str(index["label"]) != row["label"]:
                    raise ValueError(f"Graph index and CSV disagree for {key}")
                matched.add(key)
            weight = float((index or {}).get("sample_weight") or row.get("sample_weight") or 1.0)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"Invalid sample weight for {key}")
            rows.append({"source_id": key, "smiles": row["smiles"], "split": row["split"],
                "label": row["label"], "sample_weight": weight,
                **{k: (index or {}).get(k) or row.get(k, "") for k in META_FIELDS[4:]}})
    if selected is not None and matched != set(selected):
        raise ValueError(f"Graph index has {len(set(selected)-matched)} source IDs absent from CSV")
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {"0", "1"}:
            raise ValueError(f"Fixed {split} split must contain both labels")
    return rows


def molecular_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Invalid/empty SMILES in fixed cohort")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
    hybrids = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
               Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
    chirals = [Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_OTHER]
    x = []
    for atom in mol.GetAtoms():
        x.append([atom.GetDegree()/6, atom.GetFormalCharge()/4, atom.GetTotalNumHs()/4,
                  float(atom.GetIsAromatic()), float(atom.IsInRing())]
                 + [float(atom.GetHybridization() == h) for h in hybrids]
                 + [float(atom.GetChiralTag() == c) for c in chirals]
                 + [float(atom.IsInRingSize(s)) for s in range(3, 9)] + [atom.GetMass()/200])
    types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    stereos = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY, Chem.BondStereo.STEREOZ,
               Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS]
    edges, attributes = [], []
    for bond in mol.GetBonds():
        bt, stereo = bond.GetBondType(), bond.GetStereo()
        values = [float(bt == t) for t in types] + [float(bt not in types),
                  float(bond.GetIsConjugated()), float(bond.IsInRing())]
        values += [float(stereo == s) for s in stereos] + [float(stereo not in stereos)]
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((i, j), (j, i)))
        attributes.extend((values, values))
    return {"x": torch.tensor(x, dtype=torch.float32),
            "z": torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long),
            "edge_index": torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.tensor(attributes, dtype=torch.float32) if attributes else torch.empty((0, 14))}


def prepare_cache(dataset_csv, graph_index, cache_dir, shard_size=1024):
    signature = {"dataset_sha256": sha256(dataset_csv),
                 "graph_index_sha256": sha256(graph_index) if graph_index else None,
                 "feature_version": FEATURE_VERSION, "shard_size": shard_size}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = Path(cache_dir) / fingerprint[:20]
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Cache signature conflict")
        with (folder / "metadata.csv").open(newline="") as stream:
            metadata = list(csv.DictReader(stream))
        if len(metadata) != manifest["rows"]:
            raise ValueError("Incomplete cache metadata")
        for item in manifest["shards"]:
            if (folder / item["file"]).stat().st_size != item["bytes"]:
                raise ValueError(f"Changed graph cache shard: {item['file']}")
        return folder, metadata, manifest
    rows = read_rows(dataset_csv, graph_index)
    metadata = [{k: row[k] for k in META_FIELDS} for row in rows]
    shards = []
    for start in range(0, len(rows), shard_size):
        name = f"graphs_{start//shard_size:05d}.pt"
        path = folder / name
        bucket = rows[start:start+shard_size]
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=True)
            if len(existing) != len(bucket):
                raise ValueError(f"Incomplete cached shard: {path}")
        else:
            atomic_torch(path, [molecular_features(row["smiles"]) for row in bucket])
        shards.append({"file": name, "rows": len(bucket), "bytes": path.stat().st_size, "sha256": sha256(path)})
        if start//shard_size % 10 == 0:
            log.info("Graph cache %s: %d/%d", fingerprint[:12], min(start+shard_size, len(rows)), len(rows))
    cohort_records = sorted([r["source_id"], r["split"], str(r["label"])] for r in metadata)
    manifest = {"signature": signature, "fingerprint": fingerprint, "rows": len(rows),
        "cohort_sha256": hashlib.sha256(json.dumps(cohort_records, separators=(",", ":")).encode()).hexdigest(),
        "split_label_counts": {split: dict(Counter(r["label"] for r in metadata if r["split"] == split))
                               for split in ("train", "val", "test")},
        "node_dim": 21, "edge_dim": 14, "shards": shards}
    atomic_csv(folder / "metadata.csv", metadata, META_FIELDS)
    atomic_json(manifest_path, manifest)
    return folder, metadata, manifest


class LazyGraphs(Dataset):
    def __init__(self, folder, metadata, manifest, split, cache_shards=3):
        self.folder, self.metadata = Path(folder), metadata
        self.shard_size = manifest["signature"]["shard_size"]
        self.indices = [i for i, r in enumerate(metadata) if r["split"] == split]
        self.cache_shards, self.loaded = cache_shards, OrderedDict()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        index = self.indices[offset]
        shard, position = divmod(index, self.shard_size)
        if shard not in self.loaded:
            self.loaded[shard] = torch.load(self.folder / f"graphs_{shard:05d}.pt", map_location="cpu", weights_only=True)
            if len(self.loaded) > self.cache_shards:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(shard)
        meta = self.metadata[index]
        return Data(**self.loaded[shard][position], y=torch.tensor([float(meta["label"])], dtype=torch.float32),
                    sample_weight=torch.tensor([float(meta["sample_weight"])], dtype=torch.float32),
                    cohort_row_id=torch.tensor([index], dtype=torch.long))


class ShardWindowShuffle(Sampler):
    """Visit every row once, shuffling within shuffled three-shard windows.

    Ordinary random indexing would reload a large tensor shard for almost every
    molecule. A window fits the bounded graph cache while still mixing rows.
    The dedicated generator is checkpointed, making epoch-boundary resume exact.
    """
    def __init__(self, dataset, generator, window_shards=3):
        self.dataset, self.generator, self.window_shards = dataset, generator, window_shards
        by_shard = {}
        for offset, global_index in enumerate(dataset.indices):
            by_shard.setdefault(global_index // dataset.shard_size, []).append(offset)
        self.groups = list(by_shard.values())

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        order = torch.randperm(len(self.groups), generator=self.generator).tolist()
        for start in range(0, len(order), self.window_shards):
            window = [offset for group in order[start:start+self.window_shards] for offset in self.groups[group]]
            for offset in torch.randperm(len(window), generator=self.generator).tolist():
                yield window[offset]


class BondResidualGATv2(torch.nn.Module):
    def __init__(self, node_dim=21, edge_dim=14, hidden=128, layers=4, heads=4, dropout=.2):
        super().__init__()
        self.dropout = dropout
        self.atomic_embedding = torch.nn.Embedding(119, hidden)
        self.node_projection = torch.nn.Linear(node_dim, hidden)
        self.convs = torch.nn.ModuleList([GATv2Conv(hidden, hidden//heads, heads=heads, dropout=dropout,
            edge_dim=edge_dim, fill_value=0.0) for _ in range(layers)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden*2, hidden), torch.nn.SiLU(),
                                        torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def forward(self, batch):
        h = self.atomic_embedding(batch.z) + self.node_projection(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            message = conv(h, batch.edge_index, edge_attr=batch.edge_attr)
            h = norm(h + F.dropout(F.silu(message), p=self.dropout, training=self.training))
        pooled = torch.cat([global_mean_pool(h, batch.batch), global_add_pool(h, batch.batch)], dim=-1)
        return self.head(pooled).reshape(-1)


def choose_f1_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def binary_metrics(y, probability, threshold):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    pred = probability >= threshold
    pos, neg = int(y.sum()), int((1-y).sum())
    fn, fp = int(((y == 1) & ~pred).sum()), int(((y == 0) & pred).sum())
    return {"n": len(y), "positive": pos, "negative": neg, "threshold": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float((y == pred).mean()), "fnr": fn/pos if pos else None,
        "fpr": fp/neg if neg else None,
        "roc_auc": float(roc_auc_score(y, probability)) if pos and neg else None,
        "pr_auc": float(average_precision_score(y, probability)) if pos and neg else None}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y, p, indices = [], [], []
    for batch in loader:
        batch = batch.to(device)
        probability = torch.sigmoid(model(batch))
        if not torch.isfinite(probability).all():
            raise ValueError("Nonfinite model probabilities")
        y.extend(batch.y.cpu().tolist()); p.extend(probability.cpu().tolist())
        indices.extend(batch.cohort_row_id.cpu().tolist())
    return np.asarray(y, dtype=int), np.asarray(p), indices


def prediction_outputs(folder, split, y, p, indices, metadata, threshold, epoch=None):
    records = [{**metadata[i], "probability_generated": float(score),
                "prediction": int(score >= threshold), "threshold": float(threshold), "epoch": epoch}
               for i, score in zip(indices, p)]
    atomic_csv(folder / f"{split}_probabilities.csv", records,
               META_FIELDS + ["probability_generated", "prediction", "threshold", "epoch"])
    groups = []
    negative = np.flatnonzero(y == 0)
    for kind in ("method", "subcategory"):
        keys = []
        for i in indices:
            row = metadata[i]
            key = row["generation_method"] if kind == "method" else "/".join((row["generation_method"],
                row["balancing_category"] or row["category"], row["balancing_subcategory"] or row["subcategory"]))
            keys.append(key)
        for key in sorted({keys[j] for j in range(len(y)) if y[j] == 1}):
            positive = np.asarray([j for j in range(len(y)) if y[j] == 1 and keys[j] == key], dtype=int)
            selected = np.concatenate((negative, positive))
            groups.append({"group_type": kind, "group": key,
                "reference_pool": "all public-reference rows in this fixed split, shared between groups",
                **binary_metrics(y[selected], p[selected], threshold)})
    atomic_csv(folder / f"{split}_group_metrics.csv", groups,
               ["group_type", "group", "reference_pool", "n", "positive", "negative", "threshold", "f1",
                "precision", "recall", "accuracy", "fnr", "fpr", "roc_auc", "pr_auc"])
    return binary_metrics(y, p, threshold)


def rng_state(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"name": state[0], "keys": state[1].tolist(),
        "position": state[2], "has_gauss": state[3], "cached_gaussian": state[4]},
        "torch": torch.get_rng_state(), "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch"]); generator.set_state(state["loader"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_snapshot(value):
    """Detach mutable optimizer/history state from later training updates."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return value


def write_status(out, value):
    """Flat, stable supervisor contract; no held-out metrics during search."""
    atomic_json(out / "progress.json", value)
    atomic_json(out / "summary.json", value)


def load_checkpoint(path):
    # Hash and deserialize the same open inode, even if an external publisher
    # atomically replaces the path while another output directory evaluates it.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != TASK or checkpoint.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Checkpoint task/features do not match this trainer")
    checkpoint["_loaded_file_sha256"] = digest.hexdigest()
    return checkpoint


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(name)


def evaluate_locked_test(args, checkpoint, out, lock_path, manifest, model, loaders, device, metadata):
    checkpoint_hash = checkpoint["_loaded_file_sha256"]
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["checkpoint_sha256"] != checkpoint_hash or lock["cohort_sha256"] != manifest["cohort_sha256"]:
            raise ValueError("This locked test was already assigned to another checkpoint/cohort")
        if lock.get("state") == "complete":
            log.info("Locked test already completed: %s", lock["result_path"])
            return
    else:
        atomic_json(lock_path, {"state": "running", "checkpoint_sha256": checkpoint_hash,
            "cohort_sha256": manifest["cohort_sha256"]})
    y, p, indices = predict(model, loaders["test"], device)
    threshold = checkpoint["validation_threshold"]
    report = prediction_outputs(out, "test", y, p, indices, metadata, threshold, checkpoint["epoch"])
    result = {"task": TASK, "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
        "cohort_sha256": manifest["cohort_sha256"], "threshold_source": "validation_only",
        "test_metrics": report, "test_metrics_at_0_5": binary_metrics(y, p, .5),
        "target_f1": .85, "target_f1_met": report["f1"] >= .85,
        "interpretation": "Fixed held-out synthetic-origin task; not validated detection of real medicine fraud."}
    atomic_json(out / "locked_test_result.json", result)
    write_status(out, {"state": "locked_test_complete", "task": TASK, "completed": True,
        "current_epoch": checkpoint["epoch"], "best_epoch": checkpoint["epoch"],
        "best_validation_f1": checkpoint["validation_metrics"]["f1"],
        "best_validation_auc": checkpoint["validation_metrics"]["roc_auc"],
        "cohort_sha256": manifest["cohort_sha256"], "test_evaluated": True,
        "test_metrics": report, "test_metrics_at_0_5": result["test_metrics_at_0_5"],
        "best_checkpoint": str(args.checkpoint), "target_f1_met": result["target_f1_met"]})
    atomic_json(lock_path, {"state": "complete", "checkpoint_sha256": checkpoint_hash,
        "cohort_sha256": manifest["cohort_sha256"], "result_path": str(out / "locked_test_result.json")})
    log.info("LOCKED TEST F1=%.4f ROC-AUC=%.4f threshold=%.5f", report["f1"], report["roc_auc"], threshold)
    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path)
    parser.add_argument("--graph-index", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--test-lock", type=Path)
    parser.add_argument("--no-test", action="store_true", help="Validation-only search (also the default training policy)")
    parser.add_argument("--prepare-cache-only", action="store_true")
    defaults = {"hidden": 128, "layers": 4, "heads": 4, "dropout": .2, "lr": .001,
        "weight_decay": 1e-5, "batch_size": 128, "epochs": 80, "patience": 12,
        "lr_patience": 4, "min_lr": 1e-5, "seed": 42, "threads": 2, "loader_workers": 0,
        "shard_size": 1024, "clip_grad": 5.0}
    for name in defaults:
        parser.add_argument("--"+name.replace("_", "-"), type=int if isinstance(defaults[name], int) else float)
    parser.add_argument("--weight-mode", choices=("none", "row"))
    parser.add_argument("--selection-metric", choices=("f1", "roc_auc"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    if args.checkpoint and not args.evaluate_test:
        parser.error("--checkpoint is only used with --evaluate-test; use --resume to continue training")
    if args.resume and args.evaluate_test:
        parser.error("Choose --resume or --evaluate-test, not both")
    out = args.out_dir or (args.checkpoint.parent if args.evaluate_test and args.checkpoint else args.resume.parent if args.resume else Path("training_runs/gatv2_experiment"))
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / ".trainer.lock"):
        configure_and_run(args, parser, out, defaults)


def configure_and_run(args, parser, out, defaults):
    checkpoint = load_checkpoint(args.checkpoint if args.evaluate_test else args.resume) if (args.checkpoint if args.evaluate_test else args.resume) else None
    if args.evaluate_test and (checkpoint is None or args.no_test):
        parser.error("--evaluate-test requires --checkpoint and cannot combine with --no-test")
    saved = checkpoint["config"] if checkpoint else {}
    immutable = {"hidden", "layers", "heads", "dropout", "lr", "weight_decay", "batch_size", "seed", "weight_mode", "selection_metric", "clip_grad", "shard_size"}
    config = {}
    for key, default in {**defaults, "weight_mode": "none", "selection_metric": "f1", "device": "cpu"}.items():
        value = getattr(args, key)
        if checkpoint and not args.evaluate_test and key in immutable and value is not None and value != saved[key]:
            parser.error(f"Cannot change {key} on resume")
        config[key] = saved.get(key, default) if value is None else value
    for key, default in (("dataset_csv", DEFAULT_CSV), ("graph_index", None), ("cache_dir", "training_runs/gatv2_graph_cache"), ("split_manifest", None)):
        value = getattr(args, key)
        path = str(value) if value else saved.get(key, default)
        config[key] = str(Path(path).resolve()) if path else None
    if min(config[k] for k in ("hidden", "layers", "heads", "batch_size", "epochs", "patience", "threads", "shard_size")) < 1:
        parser.error("Architecture, batch, epoch, patience, thread and shard sizes must be positive")
    if config["hidden"] % config["heads"] or not 0 <= config["dropout"] < 1:
        parser.error("Hidden size must divide into heads; dropout must be in [0,1)")
    if config["lr"] <= 0 or config["min_lr"] <= 0 or config["clip_grad"] <= 0 or config["weight_decay"] < 0:
        parser.error("Learning rates/gradient clip must be positive; weight decay nonnegative")
    if config["loader_workers"] < 0 or config["lr_patience"] < 1:
        parser.error("Loader workers must be nonnegative; scheduler patience positive")
    run_experiment(args, checkpoint, config, out, parser)


def run_experiment(args, checkpoint, config, out, parser):
    if not checkpoint and (out / "last.pt").exists() and not args.prepare_cache_only:
        parser.error("Run directory already has last.pt; use --resume or a new --out-dir")
    torch.set_num_threads(config["threads"])
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = device_for(config["device"])
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
    folder, metadata, manifest = prepare_cache(config["dataset_csv"], config["graph_index"], config["cache_dir"], config["shard_size"])
    if checkpoint and checkpoint["cache_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Dataset/index/features changed since checkpoint")
    split_manifest_hash = sha256(config["split_manifest"]) if config["split_manifest"] else None
    if checkpoint and checkpoint.get("split_manifest_sha256") != split_manifest_hash:
        raise ValueError("Split audit manifest changed since checkpoint")
    if config["split_manifest"]:
        audit = json.loads(Path(config["split_manifest"]).read_text())
        if audit["cohort_sha256"] != manifest["cohort_sha256"]:
            raise ValueError("Input cohort differs from the immutable split audit")
        for name, signature_key in (("dataset.csv", "dataset_sha256"), ("graph_index.csv", "graph_index_sha256")):
            recorded = audit.get("selected_files", {}).get(name, {}).get("sha256")
            if recorded and recorded != manifest["signature"][signature_key]:
                raise ValueError(f"Input {name} differs from the immutable split audit")
    atomic_json(out / "cohort_manifest.json", manifest)
    provenance = {"task": TASK, "config": config, "trainer_sha256": sha256(__file__),
        "dataset_sha256": manifest["signature"]["dataset_sha256"],
        "graph_index_sha256": manifest["signature"]["graph_index_sha256"],
        "split_manifest_sha256": split_manifest_hash, "cohort_sha256": manifest["cohort_sha256"],
        "python": platform.python_version(), "torch": str(torch.__version__), "rdkit": rdBase.rdkitVersion,
        "sampler": "shuffled_three_shard_windows_each_training_row_once_per_epoch",
        "label_semantics": {"0": "public reference molecular origin", "1": "generated analogue molecular origin"},
        "test_policy": "standalone_locked_evaluation_only", "selection_uses": "validation_only"}
    if not args.evaluate_test:
        atomic_json(out / "run_metadata.json", provenance)
    if args.prepare_cache_only:
        log.info("Prepared %d cached graphs at %s", len(metadata), folder)
        return
    datasets = {split: LazyGraphs(folder, metadata, manifest, split) for split in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(config["seed"])
    common_loader = {"num_workers": config["loader_workers"], "persistent_workers": config["loader_workers"] > 0}
    if config["loader_workers"]:
        common_loader["prefetch_factor"] = 1
    loaders = {split: DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
        sampler=ShardWindowShuffle(dataset, generator) if split == "train" else None,
        generator=generator if split == "train" else None, **common_loader) for split, dataset in datasets.items()}
    arch = {key: (checkpoint["config"][key] if args.evaluate_test else config[key]) for key in ("hidden", "layers", "heads", "dropout")}
    model = BondResidualGATv2(**arch).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    if args.evaluate_test:
        lock_path = args.test_lock or out / "locked_test.json"
        with exclusive_lock(lock_path.with_name(lock_path.name + ".lock")):
            evaluate_locked_test(args, checkpoint, out, lock_path, manifest, model, loaders, device, metadata)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5,
        patience=config["lr_patience"], min_lr=config["min_lr"])
    start, best_metric, best_epoch, bad, history = 1, -1.0, 0, 0, []
    best_checkpoint = None
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start, best_metric, best_epoch = checkpoint["epoch"]+1, checkpoint["best_metric"], checkpoint["best_epoch"]
        bad, history = checkpoint["bad_epochs"], cpu_snapshot(checkpoint["history"])
        restore_rng(checkpoint["rng_state"], generator)
        if checkpoint.get("best_checkpoint_state") is not None:
            # A crash between the two atomic files must not pair a last state
            # with a best state from a different completed epoch.
            best_checkpoint = cpu_snapshot(checkpoint["best_checkpoint_state"])
            atomic_torch(out / "best.pt", best_checkpoint)
        elif (out / "best.pt").exists():
            best_checkpoint = load_checkpoint(out / "best.pt")
        elif checkpoint.get("kind") == "validation_best":
            best_checkpoint = cpu_snapshot(checkpoint)
        else:
            raise ValueError("Resume directory lacks best.pt; retain both per-epoch checkpoint files")
        if best_checkpoint["epoch"] != best_epoch or abs(best_checkpoint["best_metric"] - best_metric) > 1e-12:
            raise ValueError("Resume best checkpoint is inconsistent with the last completed epoch")
    mean_weight = (sum(float(metadata[i]["sample_weight"]) for i in datasets["train"].indices)/len(datasets["train"])) if config["weight_mode"] == "row" else 1.0
    log.info("%s | train=%d val=%d test_locked=%d | device=%s weight_mode=%s", TASK,
        len(datasets["train"]), len(datasets["val"]), len(datasets["test"]), device, config["weight_mode"])
    for epoch in range(start, config["epochs"]+1):
        started = time.monotonic(); model.train(); numerator = denominator = 0.0
        for batch in loaders["train"]:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            logits = model(batch); losses = F.binary_cross_entropy_with_logits(logits, batch.y, reduction="none")
            weights = batch.sample_weight if config["weight_mode"] == "row" else torch.ones_like(losses)
            loss = (losses*weights).mean()/mean_weight
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            optimizer.step(); numerator += float((losses.detach()*weights).sum().cpu()); denominator += float(weights.sum().cpu())
        y, p, indices = predict(model, loaders["val"], device)
        threshold = choose_f1_threshold(y, p); val = binary_metrics(y, p, threshold)
        score = val[config["selection_metric"]]; scheduler.step(score)
        improved = score > best_metric + 1e-6
        if improved:
            best_metric, best_epoch, bad = score, epoch, 0
        else:
            bad += 1
        entry = {"epoch": epoch, "train_loss": numerator/denominator, "val_f1": val["f1"],
            "val_roc_auc": val["roc_auc"], "val_precision": val["precision"], "val_recall": val["recall"],
            "validation_threshold": threshold, "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic()-started, "best_epoch": best_epoch, "best_metric": best_metric}
        history.append(entry)
        state = {"format_version": 1, "task": TASK, "model": "bond_residual_gatv2", "feature_version": FEATURE_VERSION,
            "provenance": provenance,
            "config": config, "cache_fingerprint": manifest["fingerprint"], "cohort_sha256": manifest["cohort_sha256"],
            "split_manifest_sha256": split_manifest_hash,
            "epoch": epoch, "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state": cpu_snapshot(optimizer.state_dict()), "scheduler_state": cpu_snapshot(scheduler.state_dict()),
            "rng_state": rng_state(generator), "validation_threshold": threshold, "validation_metrics": val,
            "best_metric": best_metric, "best_epoch": best_epoch, "bad_epochs": bad, "history": cpu_snapshot(history),
            "test_policy": "standalone_locked_evaluation_only"}
        if improved:
            best_checkpoint = {**state, "kind": "validation_best"}
            prediction_outputs(out, "val", y, p, indices, metadata, threshold, epoch)
            atomic_json(out / "validation_best_metrics.json", {"epoch": epoch, **val})
        # Maintain both a current restart and the selected best checkpoint on
        # every completed epoch; the best retains its own optimizer/RNG state.
        atomic_torch(out / "best.pt", best_checkpoint)
        atomic_torch(out / "last.pt", {**state, "kind": "last", "best_checkpoint_state": best_checkpoint})
        atomic_json(out / "history.json", history)
        write_status(out, {"state": "training", "task": TASK, "epoch": epoch,
            "current_epoch": epoch, "completed": False,
            "best_validation_f1": best_checkpoint["validation_metrics"]["f1"],
            "best_validation_auc": best_checkpoint["validation_metrics"]["roc_auc"],
            "maximum_epochs": config["epochs"], **entry, "bad_epochs": bad, "patience": config["patience"],
            "cohort_sha256": manifest["cohort_sha256"], "best_checkpoint": str(out / "best.pt"),
            "last_checkpoint": str(out / "last.pt"), "test_evaluated": False})
        log.info("epoch %3d loss=%.4f val_F1=%.4f val_AUC=%.4f threshold=%.4f best=%d lr=%.2g seconds=%.1f",
                 epoch, entry["train_loss"], val["f1"], val["roc_auc"], threshold, best_epoch,
                 entry["learning_rate"], entry["seconds"])
        if bad >= config["patience"]:
            log.info("Early stop after %d epochs without improvement", bad)
            break
    write_status(out, {"state": "validation_search_complete", "task": TASK, "completed": True,
        "current_epoch": history[-1]["epoch"] if history else checkpoint["epoch"],
        "epoch": history[-1]["epoch"] if history else checkpoint["epoch"], "best_epoch": best_epoch,
        "best_metric": best_metric, "selection_metric": config["selection_metric"],
        "best_validation_f1": best_checkpoint["validation_metrics"]["f1"],
        "best_validation_auc": best_checkpoint["validation_metrics"]["roc_auc"],
        "cohort_sha256": manifest["cohort_sha256"], "test_evaluated": False,
        "best_checkpoint": str(out / "best.pt"), "last_checkpoint": str(out / "last.pt")})


if __name__ == "__main__":
    main()
