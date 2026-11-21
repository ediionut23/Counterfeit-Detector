#!/usr/bin/env python3
"""Train a frozen-distance baseline or a new angular geometric GNN.

Reuses the original trainer's unchanged cache, optimizer/checkpoint/RNG code,
validation-only selection, and run/test locks. The runtime binding is confined
to this standalone process (or a context restored after a test call); it never
edits the original trainer. Variant, engine and new module hashes are pinned in
every checkpoint. Existing source coordinates and radius-neighbor caches are
reused without generating coordinates or changing cohort membership/splits.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import torch

import train_geometric_gnn as engine
import geometric_angular_model as geometry_models


_ORIGINAL_PARSER = engine.parser
_ORIGINAL_MODEL_CONFIG = engine.model_config
_ORIGINAL_LOCKED_CONFIG = engine.locked_config


def positive_int(value):
    import argparse
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def parser():
    result = _ORIGINAL_PARSER()
    result.description = __doc__
    result.set_defaults(graph_index="balanced_datasets/training_200k/overnight_strict_converged_v2/graph_index.csv",
                        out="training_runs/expanded/geometric_angular")
    result.add_argument("--geometry-variant", choices=("distance", "angular"), default="angular")
    result.add_argument("--angular-basis", type=positive_int, default=6,
                        help="Radial channels for first/second directional-moment angle summaries")
    return result


def model_config(args):
    result = _ORIGINAL_MODEL_CONFIG(args)
    result.update(geometry_variant=args.geometry_variant, angular_basis=args.angular_basis)
    return result


def locked_config(args, data_summary, device):
    result = _ORIGINAL_LOCKED_CONFIG(args, data_summary, device)
    result.update(engine_sha256=result["script_sha256"], script_sha256=engine.sha256(__file__),
                  geometry_model_sha256=engine.sha256(geometry_models.__file__),
                  geometry_variant=args.geometry_variant,
                  angular_features="weighted distinct-neighbor cos(theta), P2(cos(theta)), radial mass; linear edge work",
                  variant_protocol_version=1)
    return result


def build_model(configuration):
    """Construct selected new-variant or frozen-original checkpoint architecture."""
    model = configuration["model"]
    if "geometry_variant" in model:
        return geometry_models.GeometryVariantGNN(**model)
    return geometry_models.FrozenDistanceGNN(**model)


def infer_selected_checkpoint(checkpoint_path, graph_index, split_manifest, split,
                              device="cpu", cache_root=None, threads=2, batch_size=64):
    """Return aligned raw probabilities for a frozen ensemble member.

    No metric, threshold, prediction file or individual test lock is created.
    The ensemble caller owns the single immutable family-selection pin and OS
    lock before asking for ``split='test'``. This adapter does not select models
    or inspect other splits. All tensor/data/source pins remain enforced.
    Legacy checkpoints from the unchanged original trainer are also supported.
    """
    if split not in engine.SPLITS or threads < 1 or batch_size < 1:
        raise ValueError("Invalid inference split/thread/batch settings")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["configuration"]
    if checkpoint["epoch"] != checkpoint["best_epoch"]:
        raise ValueError("Ensemble inference requires the validation-selected best checkpoint")
    if "geometry_variant" in config["model"]:
        pins = ((config["engine_sha256"], engine.__file__),
                (config["script_sha256"], __file__),
                (config["geometry_model_sha256"], geometry_models.__file__))
    else:
        pins = ((config["script_sha256"], engine.__file__),)
    for expected, filename in pins:
        if expected != engine.sha256(filename):
            raise ValueError(f"Selected checkpoint implementation changed: {filename}")
    stored = json.loads((checkpoint_path.parent / "data_summary.json").read_text())
    if stored["fingerprint"] != config["data_fingerprint"] or stored["cohort_sha256"] != config["cohort_sha256"]:
        raise ValueError("Checkpoint-adjacent pinned geometry summary disagrees")
    graph_index = Path(graph_index).resolve()
    supplied_manifest = Path(split_manifest).resolve() if split_manifest is not None else graph_index.parent / "manifest.json"
    if supplied_manifest.exists():
        expected_manifest = stored["data_spec"]["cohort_manifest_sha256"]
        if expected_manifest and engine.sha256(supplied_manifest) != expected_manifest:
            raise ValueError("Ensemble split manifest differs from training")
        if expected_manifest and supplied_manifest != graph_index.parent / "manifest.json":
            raise ValueError("Split manifest must be the selected index's adjacent audited manifest")
    elif stored["data_spec"]["cohort_manifest_sha256"] is not None:
        raise ValueError("Selected audited split manifest is missing")
    if cache_root is None:
        example = next(iter(stored["data_spec"]["source_files"].values()))
        cache_root = Path(example["cache_dir"]).parent.parent
    index_path, summary = engine.prepare_data(stored["data_spec"]["root"], graph_index, cache_root,
                                              stored["data_spec"]["geometry"])
    if summary["fingerprint"] != config["data_fingerprint"] or summary["cohort_sha256"] != config["cohort_sha256"]:
        raise ValueError("Inference does not match the selected training cohort/geometry")
    torch.set_num_threads(threads)
    selected_device = torch.device(device)
    model = build_model(config).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = engine.GeometryDataset(index_path, summary, split)
    args = SimpleNamespace(batch_size=batch_size, seed=config["seed"], num_workers=0)
    identifiers, molecules, labels, probabilities = [], [], [], []
    try:
        with torch.no_grad():
            for batch in engine.make_loader(dataset, args):
                batch = batch.to(selected_device)
                logits = model(batch)
                if not torch.isfinite(logits).all():
                    raise ValueError("Selected model produced nonfinite probabilities")
                identifiers.extend(batch.source_id)
                molecules.extend(batch.molecule_id)
                labels.extend(batch.y.cpu().tolist())
                probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    finally:
        dataset.close()
    return {"source_id": np.asarray(identifiers), "molecule_id": np.asarray(molecules),
            "label": np.asarray(labels, dtype=np.int64), "probability": np.asarray(probabilities, dtype=np.float64),
            "split": split, "cohort_sha256": summary["cohort_sha256"],
            "checkpoint_sha256": engine.sha256(checkpoint_path)}


@contextmanager
def bound_variant_engine():
    """Install this process's model hooks, restoring them after a library call."""
    replacements = {"DistanceGNN": geometry_models.GeometryVariantGNN, "parser": parser,
                    "model_config": model_config, "locked_config": locked_config}
    original = {name: getattr(engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(engine, name, value)


def main(argv=None):
    with bound_variant_engine():
        return engine.main(argv)


if __name__ == "__main__":
    main()
