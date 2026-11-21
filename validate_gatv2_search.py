#!/usr/bin/env python3
"""Tiny engineering fixture: cache, weighted training, exact resume and test lock.

The toy molecules/labels do not estimate scientific performance. No production
datasets or model runs are modified, and all artifacts live in a temporary dir.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch_geometric.data import Batch, Data

from train_gatv2_search import (LazyGraphs, ShardWindowShuffle,
    BASE_PROTOCOL_SHA256, FEATURE_VERSION, TASK, SearchGATv2, atomic_torch,
    choose_f1_threshold, exclusive_lock, infer_selected_checkpoint,
    molecular_features, predict, prepare_cache, read_rows, DataLoader)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, expect_success=True):
    result = subprocess.run(command, capture_output=True, text=True)
    if (result.returncode == 0) != expect_success:
        raise AssertionError(f"Unexpected return code {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


def same_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_state(x, y)
    else:
        assert a == b


def check_variants():
    from train_gatv2_experiment import BondResidualGATv2
    torch.set_num_threads(1)
    torch.manual_seed(19)
    baseline = BondResidualGATv2(hidden=8, layers=2, heads=2, dropout=.1)
    torch.manual_seed(19)
    compatible = SearchGATv2(hidden=8, layers=2, heads=2, dropout=.1)
    same_state(baseline.state_dict(), compatible.state_dict())
    first, second = molecular_features("c1ccncc1O"), molecular_features("CC(=O)N")
    order = torch.arange(len(first["z"])-1, -1, -1)
    inverse = torch.empty_like(order); inverse[order] = torch.arange(len(order))
    permuted = {"x": first["x"][order], "z": first["z"][order],
        "edge_index": inverse[first["edge_index"]], "edge_attr": first["edge_attr"]}
    batch = Batch.from_data_list([Data(**first), Data(**second)])
    rearranged = Batch.from_data_list([Data(**permuted), Data(**second)])
    baseline.eval(); compatible.eval()
    assert torch.equal(baseline(batch), compatible(batch))
    for norm in ("layer", "batch"):
        for readout, pool in (("mean_sum", "mean_sum"), ("mean_max", "mean_sum"),
            ("attention", "mean_sum"), ("jk", "mean_sum"), ("jk", "mean_max"), ("jk", "attention")):
            model = SearchGATv2(hidden=8, layers=2, heads=2, dropout=.1, norm=norm,
                readout=readout, jk_pool=pool, input_dropout=.05)
            model.train()
            logits = model(batch)
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
            assert logits.shape == (2,) and torch.isfinite(logits).all()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            model.eval()
            before = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}
            assert torch.allclose(model(batch), model(rearranged), atol=1e-5, rtol=1e-5)
            same_state(before, {k: v for k, v in model.state_dict().items() if k in before})
    singleton = Batch.from_data_list([Data(**molecular_features("[Na+]"))])
    model = SearchGATv2(hidden=8, layers=2, heads=2, norm="batch", readout="attention")
    model.train()
    assert torch.isfinite(model(singleton)).all()


def main():
    check_variants()
    script = str(Path(__file__).with_name("train_gatv2_search.py"))
    with tempfile.TemporaryDirectory(prefix="gatv2_search_fixture_") as directory:
        root = Path(directory)
        smiles = ["CCO", "CCN", "CCCl", "CCF", "CC(=O)O", "CC(=O)N", "CCCO", "CCCN",
                  "COC", "CNC", "c1ccccc1", "c1ccncc1", "CCCC", "CCOC", "CCBr", "CCS",
                  "C1CCCCC1", "C1CCNCC1", "CC#N", "CC=C"]
        rows = []
        for i, molecule in enumerate(smiles):
            label = i % 2
            rows.append({"molecule_id": f"fixture_{i:02d}", "smiles": molecule,
                "split": "train" if i < 12 else "val" if i < 16 else "test", "label": str(label),
                "generation_method": ("M" if i % 4 == 1 else "E") if label else "reference",
                "category": "fixture", "subcategory": "toy", "sample_weight": .5 if label else 2.0})
        dataset, index = root / "dataset.csv", root / "graph_index.csv"
        write_csv(dataset, rows)
        write_csv(index, [{"source_id": r["molecule_id"], "split": r["split"], "label": r["label"],
            "sample_weight": r["sample_weight"]} for r in rows])
        cohort = sorted([[r["molecule_id"], r["split"], r["label"]] for r in rows])
        audit = root / "manifest.json"
        audit.write_text(json.dumps({"cohort_sha256": hashlib.sha256(json.dumps(cohort, separators=(",", ":")).encode()).hexdigest(),
            "selected_files": {"dataset.csv": {"sha256": sha(dataset)}, "graph_index.csv": {"sha256": sha(index)}}}))
        folder, metadata, manifest = prepare_cache(dataset, index, root / "cache", shard_size=4)
        assert manifest["rows"] == 20 and manifest["node_dim"] == 21 and manifest["edge_dim"] == 14
        train = LazyGraphs(folder, metadata, manifest, "train")
        sampler = ShardWindowShuffle(train, torch.Generator().manual_seed(17))
        assert sorted(list(sampler)) == list(range(12))
        assert molecular_features("[Na+]")["edge_attr"].shape == (0, 14)
        assert molecular_features("C[C@H](O)Cl")["x"].shape[1] == 21
        assert choose_f1_threshold([0, 0, 1, 1], [.1, .4, .5, .9]) == .5
        base = [sys.executable, script, "--dataset-csv", str(dataset), "--graph-index", str(index),
            "--split-manifest", str(audit), "--cache-dir", str(root / "cache"), "--hidden", "8",
            "--layers", "2", "--heads", "2", "--dropout", ".15", "--batch-size", "4",
            "--patience", "10", "--threads", "1", "--shard-size", "4", "--lr", ".003",
            "--seed", "17", "--weight-mode", "row", "--no-test",
            "--readout", "jk", "--jk-pool", "attention", "--norm", "batch",
            "--input-dropout", ".05", "--feature-dropout", ".1",
            "--attention-dropout", ".15", "--readout-dropout", ".2"]
        full, resumed = root / "full", root / "resumed"
        run(base + ["--out-dir", str(full), "--epochs", "3"])
        run(base + ["--out-dir", str(resumed), "--epochs", "1"])
        (resumed / "best.pt").unlink()  # Recover interrupted two-file checkpoint publication.
        run([sys.executable, script, "--resume", str(resumed / "last.pt"), "--epochs", "3", "--no-test"])
        a = torch.load(full / "last.pt", weights_only=True)
        b = torch.load(resumed / "last.pt", weights_only=True)
        for field in ("model_state", "optimizer_state", "scheduler_state", "rng_state", "validation_metrics"):
            same_state(a[field], b[field])
        for output in (full, resumed):
            best = torch.load(output / "best.pt", weights_only=True)
            assert len(best["history"]) == best["epoch"]
            assert best["kind"] == "validation_best"
            summary = json.loads((output / "summary.json").read_text())
            assert summary["completed"] and summary["current_epoch"] == 3
            assert isinstance(summary["best_validation_f1"], float)
            assert not summary["test_evaluated"]
            assert not (output / "test_probabilities.csv").exists()
        raw = infer_selected_checkpoint(full / "best.pt", index, audit, "val", "cpu", root / "cache", 1, 4)
        with (full / "val_probabilities.csv").open(newline="") as stream:
            selected_val = list(csv.DictReader(stream))
        assert raw["source_id"] == [row["source_id"] for row in selected_val]
        assert raw["probability"].tolist() == [float(row["probability_generated"]) for row in selected_val]
        from train_gatv2_experiment import BondResidualGATv2
        legacy_config = {key: value for key, value in a["config"].items() if key not in {
            "readout", "jk_pool", "norm", "input_dropout", "feature_dropout", "attention_dropout", "readout_dropout"}}
        legacy_model = BondResidualGATv2(**{key: legacy_config[key] for key in ("hidden", "layers", "heads", "dropout")})
        legacy_path = root / "legacy_baseline" / "best.pt"
        atomic_torch(legacy_path, {"task": TASK, "feature_version": FEATURE_VERSION,
            "model": "bond_residual_gatv2", "config": legacy_config,
            "provenance": {"trainer_sha256": BASE_PROTOCOL_SHA256},
            "cache_fingerprint": manifest["fingerprint"], "split_manifest_sha256": sha(audit),
            "cohort_sha256": manifest["cohort_sha256"], "model_state": legacy_model.state_dict()})
        legacy_raw = infer_selected_checkpoint(legacy_path, index, audit, "val", "cpu", root / "cache", 1, 4)
        legacy_y, legacy_p, legacy_indices = predict(legacy_model,
            DataLoader(LazyGraphs(folder, metadata, manifest, "val"), batch_size=4), "cpu")
        assert legacy_raw["source_id"] == [metadata[i]["source_id"] for i in legacy_indices]
        assert legacy_raw["probability"].tolist() == legacy_p.tolist()
        assert legacy_raw["label"].tolist() == legacy_y.tolist()
        assert legacy_raw["checkpoint_sha256"] == sha(legacy_path)
        for flag, value in (("--readout", "mean_max"), ("--norm", "layer"), ("--feature-dropout", ".9"), ("--patience", "11")):
            rejected = run([sys.executable, script, "--resume", str(resumed / "last.pt"), flag, value], expect_success=False)
            assert "Cannot change" in rejected.stderr
        lock = root / "global_locked_test.json"
        evaluate = [sys.executable, script, "--evaluate-test", "--checkpoint", str(full / "best.pt"),
            "--test-lock", str(lock), "--threads", "1"]
        with exclusive_lock(full / ".trainer.lock"):
            rejected = run(evaluate, expect_success=False)
            assert "Another trainer/evaluator holds lock" in rejected.stderr
        with exclusive_lock(lock.with_name(lock.name + ".lock")):
            rejected = run(evaluate, expect_success=False)
            assert "Another trainer/evaluator holds lock" in rejected.stderr
        assert not lock.exists()
        run(evaluate)
        result_hash = sha(full / "locked_test_result.json")
        run(evaluate)  # Idempotent rerun: the test is not recomputed.
        assert result_hash == sha(full / "locked_test_result.json")
        assert json.loads((full / "summary.json").read_text())["test_evaluated"]
        rejected = run([sys.executable, script, "--evaluate-test", "--checkpoint", str(resumed / "best.pt"),
            "--test-lock", str(lock), "--threads", "1"], expect_success=False)
        assert "already assigned to another checkpoint/cohort" in rejected.stderr
        changed = rows + [{**rows[0], "molecule_id": "unselected_extra", "split": "train"}]
        write_csv(dataset, changed)
        rejected = run([sys.executable, script, "--resume", str(resumed / "last.pt"), "--epochs", "4"], expect_success=False)
        assert "changed since checkpoint" in rejected.stderr
        # A selected duplicate/mismatch must also fail at input ingestion.
        write_csv(dataset, rows + [rows[0]])
        try:
            read_rows(dataset, index)
        except ValueError as error:
            assert "Duplicate selected" in str(error)
        else:
            raise AssertionError("Duplicate selected molecule was accepted")
    print("PASS: all readout/norm gradients, permutation invariance, baseline parity, BatchNorm buffers, original/new raw adapters, fixed cohort/cache, weighted bit-exact resume, checkpoint recovery, immutable architecture, exclusive process locks and locked test")


if __name__ == "__main__":
    main()
