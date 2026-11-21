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

from train_gatv2_experiment import (LazyGraphs, ShardWindowShuffle,
    choose_f1_threshold, exclusive_lock, molecular_features, prepare_cache, read_rows)


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


def main():
    script = str(Path(__file__).with_name("train_gatv2_experiment.py"))
    with tempfile.TemporaryDirectory(prefix="gatv2_fixture_") as directory:
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
            "--seed", "17", "--weight-mode", "row", "--no-test"]
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
    print("PASS: fixed cohort/cache, weighted training, bit-exact resume, checkpoint history, validation-only search, exclusive process locks, global test lock and changed-input rejection")


if __name__ == "__main__":
    main()
