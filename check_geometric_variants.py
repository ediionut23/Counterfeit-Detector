#!/usr/bin/env python3
"""Fixture checks for angular features, unchanged caches and exact resuming."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import csv

import numpy as np

import torch
from torch_geometric.data import Data

import train_geometric_gnn as frozen
import train_geometric_variants as trainer
from geometric_angular_model import GeometryVariantGNN, angular_features
from check_geometric_gnn import fixture, expect_failure


def angle_graph(angle):
    return Data(z=torch.tensor([6, 7, 8]), scalar_features=torch.zeros((3, 7)),
        pos=torch.tensor([[0., 0., 0.], [1., 0., 0.], [math.cos(angle), math.sin(angle), 0.]]),
        edge_index=torch.tensor([[1, 2, 0, 0], [0, 0, 1, 2]]), num_nodes=3)


def self_check():
    original_sha = frozen.sha256(frozen.__file__)
    frozen.seed_everything(101, 1)
    reference = frozen.DistanceGNN(hidden=12, interactions=2, radial_basis=8, dropout=0).eval()
    frozen.seed_everything(101, 1)
    distance = GeometryVariantGNN(hidden=12, interactions=2, radial_basis=8, dropout=0, geometry_variant="distance").eval()
    for name, value in reference.state_dict().items():
        assert torch.equal(value, distance.state_dict()[name]), name
    right_angle, wider_angle = angle_graph(math.pi / 2), angle_graph(2 * math.pi / 3)
    assert torch.allclose(reference(right_angle), distance(right_angle), atol=0, rtol=0)
    assert torch.allclose(distance(right_angle), distance(wider_angle), atol=1e-7, rtol=1e-7)

    frozen.seed_everything(101, 1)
    angular = GeometryVariantGNN(hidden=12, interactions=2, radial_basis=8, angular_basis=4, dropout=0).eval()
    moved = right_angle.clone()
    moved.pos = moved.pos @ torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) + torch.tensor([4., -7., 2.])
    stretched = right_angle.clone()
    stretched.pos[1, 0] += 0.5
    with torch.no_grad():
        first, rigid, wider, longer = angular(right_angle), angular(moved), angular(wider_angle), angular(stretched)
    assert torch.allclose(first, rigid, atol=2e-6, rtol=2e-6), (first, rigid)
    assert not torch.allclose(first, wider, atol=1e-7, rtol=1e-7), (first, wider)
    assert not torch.allclose(first, longer, atol=1e-7, rtol=1e-7), (first, longer)
    context, node = angular_features(right_angle.pos, right_angle.edge_index, angular.angular_centres,
        angular.angular_gamma, angular.cutoff)
    assert torch.allclose(node[0, :4], torch.zeros(4), atol=2e-6)
    assert torch.allclose(node[0, 4:7], torch.full((3,), -0.5), atol=2e-6)
    context_wider, node_wider = angular_features(wider_angle.pos, wider_angle.edge_index, angular.angular_centres,
        angular.angular_gamma, angular.cutoff)
    assert torch.allclose(node_wider[0, :3], torch.full((3,), -0.5), atol=2e-6)
    assert torch.allclose(node_wider[0, 4:7], torch.full((3,), -0.125), atol=2e-6)
    differentiable = wider_angle.clone()
    differentiable.pos.requires_grad_(True)
    angular(differentiable).sum().backward()
    assert differentiable.pos.grad is not None and torch.isfinite(differentiable.pos.grad).all()
    assert float(differentiable.pos.grad.abs().sum()) > 1e-7
    assert torch.allclose(differentiable.pos.grad.sum(dim=0), torch.zeros(3), atol=2e-6)
    isolated = Data(z=torch.tensor([8]), scalar_features=torch.zeros((1, 7)),
        pos=torch.zeros((1, 3)), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
    assert torch.isfinite(angular(isolated)).all()

    with tempfile.TemporaryDirectory(prefix="angular_gnn_check_") as temporary:
        directory = Path(temporary)
        root, graph_index = fixture(directory)
        common = ["--root", str(root), "--graph-index", str(graph_index), "--cache-root", str(directory / "cache"),
            "--device", "cpu", "--threads", "1", "--hidden", "12", "--interactions", "2",
            "--radial-basis", "8", "--angular-basis", "4", "--geometry-variant", "angular",
            "--max-neighbors", "2", "--dropout", "0.15", "--seed", "11", "--batch-size", "3",
            "--patience", "10", "--weight-mode", "row"]
        uninterrupted, resumed = directory / "uninterrupted", directory / "resumed"
        trainer.main(common + ["--out", str(uninterrupted), "--epochs", "2", "--no-test"])
        trainer.main(common + ["--out", str(resumed), "--epochs", "1", "--no-test"])
        trainer.main(common + ["--out", str(resumed), "--epochs", "2", "--resume", str(resumed / "last.pt"), "--no-test"])
        left = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
        right = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        for name, value in left["model_state_dict"].items():
            assert torch.equal(value, right["model_state_dict"][name]), name
        assert torch.equal(left["rng_state"]["torch"], right["rng_state"]["torch"])
        assert len(right["history"]) == 2 and right["optimizer_state_dict"]["state"]
        config = right["configuration"]
        assert config["geometry_variant"] == "angular" and config["model"]["angular_basis"] == 4
        assert config["engine_sha256"] == original_sha
        assert config["script_sha256"] == frozen.sha256(trainer.__file__)
        assert config["geometry_model_sha256"] == frozen.sha256(Path(__file__).with_name("geometric_angular_model.py"))
        changed = list(common)
        changed[changed.index("angular")] = "distance"
        expect_failure(lambda: trainer.main(changed + ["--out", str(resumed), "--epochs", "3",
            "--resume", str(resumed / "last.pt"), "--no-test"]), "differs")
        for out in (uninterrupted, resumed):
            assert (out / "best.pt").exists() and not list(out.glob("*.tmp"))
            assert not (out / "test_results.json").exists()
            assert not (out / "test_predictions.csv").exists()
        inferred = trainer.infer_selected_checkpoint(resumed / "best.pt", graph_index, None, "val",
            device="cpu", cache_root=directory / "cache", threads=1, batch_size=3)
        with (resumed / "validation_best_predictions.csv").open() as stream:
            saved = list(csv.DictReader(stream))
        assert inferred["source_id"].tolist() == [row["source_id"] for row in saved]
        assert np.allclose(inferred["probability"], [float(row["probability"]) for row in saved], atol=1e-9)
        assert "loss" not in inferred and "test_metrics" not in inferred
        baseline_args = list(common)
        for flag in ("--geometry-variant", "--angular-basis"):
            offset = baseline_args.index(flag)
            del baseline_args[offset:offset + 2]
        original_out = directory / "original_distance"
        frozen.main(baseline_args + ["--out", str(original_out), "--epochs", "1", "--no-test"])
        old_inferred = trainer.infer_selected_checkpoint(original_out / "best.pt", graph_index, None, "val",
            device="cpu", cache_root=directory / "cache", threads=1, batch_size=3)
        assert len(old_inferred["probability"]) == 4 and np.isfinite(old_inferred["probability"]).all()
        test_args = common + ["--out", str(resumed), "--evaluate-test", "--checkpoint", str(resumed / "best.pt")]
        result = trainer.main(test_args)
        assert result["test_metrics"]["n"] == 4
        assert trainer.main(test_args) == result
        expect_failure(lambda: trainer.main(common + ["--out", str(uninterrupted), "--evaluate-test",
            "--checkpoint", str(uninterrupted / "best.pt")]), "another candidate")
        assert frozen.DistanceGNN is not GeometryVariantGNN
        assert frozen.sha256(frozen.__file__) == original_sha
    return {"status": "passed", "rotation_translation_invariant": True,
        "angle_sensitive_at_fixed_edge_distances": True, "distance_sensitive": True,
        "coordinate_gradients_finite_nonzero": True, "distance_baseline_exactly_preserved": True,
        "resumed_weights_rng_exact_match": True, "variant_change_refused_on_resume": True,
        "original_trainer_unchanged": True, "test_lock_verified": True}


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2))
