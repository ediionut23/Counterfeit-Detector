#!/usr/bin/env python3
"""Small local geometry/model/resume checks; never uses the production bank."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import train_geometric_gnn as geo


def fixture(directory, nonfinite=False):
    root = directory / "source"
    root.mkdir(parents=True)
    rows, hashes = [], {}
    for shard in range(2):
        positions, features, edges, ptr, eptr = [], [], [], [0], [0]
        sources, molecules, labels, splits, converged = [], [], [], [], []
        for number in range(8):
            index = shard * 8 + number
            label = number % 2
            split = "train" if number < 4 else "val" if number < 6 else "test"
            length = 1.2 + label * 0.8 + 0.03 * index
            pos = np.array([[0., 0., 0.], [length, 0., 0.], [length, 1.3, 0.2], [-0.6, 0.3, 0.]], dtype=np.float32)
            if nonfinite and index == 0:
                pos[0, 0] = np.nan
            atoms = np.array([[6, 0, 3, 0, 4, 0, 1, 0], [7, 0, 2, 0, 4, 0, 1, 0],
                              [8, 0, 1, 0, 4, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int16)
            local = np.array([(src, dst) for src in range(4) for dst in range(4) if src != dst], dtype=np.int32).T
            edges.append(local + ptr[-1])
            positions.append(pos)
            features.append(atoms)
            ptr.append(ptr[-1] + 4)
            eptr.append(eptr[-1] + local.shape[1])
            source = f"fixture_{index:03d}"
            molecule = f"molecule_{index:03d}"
            sources.append(source)
            molecules.append(molecule)
            labels.append(label)
            splits.append(split)
            converged.append(index != 0)
            rows.append({"source_id": source, "molecule_id": molecule, "split": split,
                "label": str(label), "graph_file": str(root / f"shard_{shard}.npz"), "graph_index": str(number),
                "sample_weight": str(0.7 if label == 0 else 1.3), "generation_method": "M" if label else "reference",
                "category": "fixture", "subcategory": "fixture"})
        path = root / f"shard_{shard}.npz"
        np.savez_compressed(path, atom_features=np.concatenate(features), positions_angstrom=np.concatenate(positions),
            atom_ptr=np.array(ptr, dtype=np.int64), radius_ptr=np.array(eptr, dtype=np.int64),
            radius_edge_index=np.concatenate(edges, axis=1), source_id=np.array(sources), molecule_id=np.array(molecules),
            label=np.array(labels, dtype=np.int8), split=np.array(splits), optimization_converged=np.array(converged))
        hashes[path.name] = {"sha256": geo.sha256(path), "bytes": path.stat().st_size}
    geo.atomic_json(root / "manifest_sha256.json", {"files": hashes})
    index = directory / "graph_index.csv"
    # Deliberately alternate source shards to check metadata access independently
    # of the source-file row order used for efficient model batching.
    rows = sorted(rows, key=lambda row: (int(row["graph_index"]), row["graph_file"]))
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root, index


def expect_failure(function, message):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert message.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError(f"Expected failure containing {message!r}")


def self_check():
    geo.seed_everything(11, 1)
    model = geo.DistanceGNN(hidden=12, interactions=2, radial_basis=8, cutoff=4.5, dropout=0.15).eval()
    original = geo.tiny_geometry()
    moved = original.clone()
    moved.pos = moved.pos @ torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) + torch.tensor([4., -7., 2.])
    changed = original.clone()
    changed.pos[1, 0] += 0.5
    with torch.no_grad():
        first, rigid, stretched = model(original), model(moved), model(changed)
    assert torch.allclose(first, rigid, atol=1e-6, rtol=1e-6), (first, rigid)
    assert not torch.allclose(first, stretched, atol=1e-7, rtol=1e-7), (first, stretched)

    with tempfile.TemporaryDirectory(prefix="geometric_gnn_check_") as temporary:
        directory = Path(temporary)
        root, graph_index = fixture(directory)
        geometry = {"version": geo.GEOMETRY_VERSION, "cutoff_angstrom": 4.5, "max_neighbors": 2, "heavy_atoms": True}
        cache = directory / "cache"
        index, summary = geo.prepare_data(root, graph_index, cache, geometry)
        assert summary["counts"] == {"train": 8, "val": 4, "test": 4}
        assert summary["nonconverged_stored_conformers_used"]["train"] == 1
        dataset = geo.GeometryDataset(index, summary, "train", cache_shards=1)
        first = dataset[0]
        assert first.num_nodes == 3 and first.pos.shape == (3, 3)
        assert first.edge_index.max() < 3 and first.edge_index.min() >= 0
        assert first.edge_index.shape[1] <= first.num_nodes * 2
        assert abs(float(first.sample_weight) - 0.7) < 1e-6
        for number in range(len(dataset)):
            dataset[number]
            assert len(dataset._cache) == 1
            assert isinstance(next(iter(dataset._cache.values()))["positions_angstrom"], np.memmap)
        sampler = geo.GeometryBatchSampler(dataset, 3, shuffle=True, seed=11)
        sampler.set_epoch(1)
        shuffled = [i for batch in sampler for i in batch]
        assert sorted(shuffled) == list(range(8)) and len(shuffled) == len(set(shuffled))
        dataset.close()
        assert geo.prepare_data(root, graph_index, cache, geometry)[1] == summary

        common = ["--root", str(root), "--graph-index", str(graph_index), "--cache-root", str(cache),
            "--device", "cpu", "--threads", "1", "--hidden", "12", "--interactions", "2",
            "--radial-basis", "8", "--max-neighbors", "2", "--dropout", "0.15", "--seed", "11",
            "--batch-size", "3", "--patience", "10", "--weight-mode", "row"]
        uninterrupted, resumed = directory / "uninterrupted", directory / "resumed"
        geo.main(common + ["--out", str(uninterrupted), "--epochs", "2", "--no-test"])
        geo.main(common + ["--out", str(resumed), "--epochs", "1", "--no-test"])
        geo.main(common + ["--out", str(resumed), "--epochs", "2", "--resume", str(resumed / "last.pt"), "--no-test"])
        left = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
        right = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        assert left["epoch"] == right["epoch"] == 2
        for name, tensor in left["model_state_dict"].items():
            assert torch.equal(tensor, right["model_state_dict"][name]), name
        assert torch.equal(left["rng_state"]["torch"], right["rng_state"]["torch"])
        assert len(right["history"]) == 2 and right["optimizer_state_dict"]["state"]
        for path in (uninterrupted, resumed):
            assert (path / "best.pt").exists() and (path / "last.pt").exists()
            assert not list(path.glob("*.tmp"))
            assert not (path / "test_results.json").exists()
            assert not (path / "test_predictions.csv").exists()
            progress = json.loads((path / "progress.json").read_text())
            assert progress["current_epoch"] == 2 and progress["completed"]

        test_args = common + ["--out", str(resumed), "--evaluate-test", "--checkpoint", str(resumed / "best.pt")]
        result = geo.main(test_args)
        assert result["test_metrics"]["n"] == 4
        predictions = resumed / "test_predictions.csv"
        unchanged = predictions.stat().st_mtime_ns
        assert geo.main(test_args) == result and predictions.stat().st_mtime_ns == unchanged
        expect_failure(lambda: geo.main(common + ["--out", str(uninterrupted), "--evaluate-test",
            "--checkpoint", str(uninterrupted / "best.pt")]), "another candidate")
        expect_failure(lambda: geo.main(common + ["--out", str(resumed), "--epochs", "3", "--resume", str(resumed / "last.pt")]), "accessed test")
        with (root / "shard_0.npz").open("ab") as stream:
            stream.write(b"source change")
        expect_failure(lambda: geo.prepare_data(root, graph_index, cache, geometry), "hash changed")

        bad = directory / "nonfinite"
        bad.mkdir()
        badroot, badindex = fixture(bad, nonfinite=True)
        expect_failure(lambda: geo.prepare_data(badroot, badindex, bad / "cache", geometry), "nonfinite")

    return {"status": "passed", "fixture_graphs": 16, "training_epochs": 2,
        "rotation_translation_invariant": True, "distance_sensitive": True,
        "resume_matches_uninterrupted_weights_and_rng_exactly": True,
        "source_hash_and_nonfinite_coordinate_guards": True, "test_locked_once": True}


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2))
