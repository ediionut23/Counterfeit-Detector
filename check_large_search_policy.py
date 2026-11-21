#!/usr/bin/env python3
"""Temporary-only policy fixture; never launches training, generation or test.

Exercises frozen promotion rankings, declared full-cohort seed clones, real
GAT CLI parsers, stage ordering and ensemble metadata using synthetic summaries.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import make_large_model_search_plan as builder
import run_large_model_search as search
from run_overnight_research import atomic_json, checksum


def flags(command):
    from enforce_large_training_cohort import options
    return options(command)


def publish(run, cohort, score, epoch=20):
    folder = Path(run["out_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "summary.json", {"completed": True, "cohort_sha256": cohort,
        "best_validation_f1": score, "current_epoch": epoch, "best_epoch": epoch,
        "test_evaluated": False})
    (folder / "best.pt").write_bytes(f"fixture checkpoint {run['id']} epoch {epoch}".encode())
    (folder / "val_probabilities.csv").write_text("source_id,split,label,probability_generated\n")
    (folder / "validation_best_predictions.csv").write_text("source_id,split,label,probability\n")


class CaptureParse(Exception):
    def __init__(self, args):
        self.args_parsed = args


def parse_gat_command(command):
    import train_gatv2_experiment
    import train_gatv2_search
    module = {"train_gatv2_experiment.py": train_gatv2_experiment,
              "train_gatv2_search.py": train_gatv2_search}[command[1]]
    original = argparse.ArgumentParser.parse_args
    def intercept(parser, *args, **kwargs):
        parsed = original(parser, command[2:])
        raise CaptureParse(parsed)
    try:
        with patch.object(argparse.ArgumentParser, "parse_args", intercept):
            module.main()
    except CaptureParse as captured:
        return captured.args_parsed
    raise AssertionError("GAT parser was not intercepted before work")


def generated_plan(root, workspace):
    old = json.loads((workspace / "overnight_research_plan.json").read_text())
    old["workspace"] = str(root)
    atomic_json(root / "overnight_research_plan.json", old)
    atomic_json(root / old["out_dir"] / "progress.json", {"status": "stopped", "stages": {}})
    for filename in ("run_large_model_search.py", "make_large_model_search_plan.py", "enforce_large_training_cohort.py",
        "select_validation_ensemble.py", "evaluate_selected_ensemble.py", "train_gatv2_search.py",
        "train_geometric_variants.py", "geometric_angular_model.py"):
        (root / filename).write_text("temporary pin fixture")
    prior = Path.cwd()
    try:
        os.chdir(root)
        with patch.object(builder, "__file__", str(root / "make_large_model_search_plan.py")), redirect_stdout(io.StringIO()):
            builder.main()
    finally:
        os.chdir(prior)
    return json.loads((root / "large_model_search_plan.json").read_text())


class Harness(search.LargeSearch):
    def save(self):
        atomic_json(self.out / "progress.json", self.state)

    def run(self, key, command, *, resume_dir=None):
        if self.state["stages"].get(key, {}).get("status") == "complete":
            return
        options = flags(command)
        self.events.append((key, list(command)))
        if key.startswith(("screen_", "promote_", "repeat_")):
            assert "--no-test" in options and "--evaluate-test" not in options
            assert options["--graph-index"] == self.plan["training_cohort"]["directory"] + "/graph_index.csv"
            if "--dataset-csv" in options:
                assert options["--dataset-csv"] == self.plan["training_cohort"]["directory"] + "/dataset.csv"
            assert int(options["--threads"]) == 2 and options.get("--device", "cpu") == "cpu"
            folder = Path(options.get("--out-dir", options.get("--out")))
            record = {"id": folder.name, "out_dir": str(folder)}
            epoch = int(options["--epochs"])
            score = .60 + (sum(folder.name.encode()) % 15)/100 + (.05 if epoch == 100 else 0)
            publish(record, self.plan["training_cohort"]["cohort_sha256"], score, epoch)
        elif key.endswith("_select_on_validation"):
            assert sum(k.startswith("repeat_") for k, _ in self.events) == 4
            specification = json.loads(Path(options["--spec"]).read_text())
            assert checksum(options["--spec"]) == options["--spec-sha256"]
            ensemble = specification["candidates"][-1]
            members = {m["id"]: m for m in specification["members"]}
            selected = [members[name] for name in ensemble["members"]]
            assert sorted(m["seed"] for m in selected) == [42, 43, 44]
            assert len({m["config_id"] for m in selected}) == 1
            atomic_json(Path(options["--out"]) / "selection.json", {"selection_uses": "validation_only"})
        elif key.endswith("_ensemble_locked_test"):
            assert json.loads(Path(options["--selection"]).read_text())["selection_uses"] == "validation_only"
            assert sum(k.startswith("repeat_") for k, _ in self.events) == 4
            assert sum(k.startswith("promote_") for k, _ in self.events) == 4
            atomic_json(Path(options["--out"]) / "test_result.json", {
                "cohort_sha256": self.plan["training_cohort"]["cohort_sha256"],
                "test_metrics": {"f1": .70, "roc_auc": .80}})
        else:
            raise AssertionError("Unexpected work stage: " + key)
        self.state["stages"][key] = {"status": "complete"}


def main():
    workspace = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="large_search_policy_") as directory:
        root = Path(directory)
        plan = generated_plan(root, workspace)
        assert [len(f["runs"]) for f in plan["families"]] == [8, 4]
        for family in plan["families"]:
            for run in family["runs"]:
                options = flags(run["command"])
                assert options["--seed"] == "42" and options["--epochs"] == "20"
                assert options["--patience"] == "12" and "--no-test" in options
                assert "--evaluate-test" not in options
                if family["id"] == "gatv2":
                    parsed = parse_gat_command(run["command"])
                    assert parsed.hidden % parsed.heads == 0
        # Direct ranking rerun test: overwritten promoted summaries cannot
        # change the locked screening shortlist or its copied evidence.
        ranking = Harness.__new__(Harness)
        ranking.out, ranking.plan = root / "ranking", plan
        runs = [{"id": x, "out_dir": str(root / "ranking_runs" / x), "command": ["fixture", x]} for x in ("a", "b", "c")]
        for run, score in zip(runs, (.80, .70, .80)):
            publish(run, plan["training_cohort"]["cohort_sha256"], score)
        chosen = ranking.freeze_ranking("gatv2", "promotion", runs, 2)
        assert [r["id"] for r in chosen] == ["a", "c"]
        frozen = checksum(ranking.out / "gatv2_promotion.json")
        for run, score in zip(runs, (.10, .99, .20)):
            publish(run, plan["training_cohort"]["cohort_sha256"], score, 100)
        assert [r["id"] for r in ranking.freeze_ranking("gatv2", "promotion", runs, 2)] == ["a", "c"]
        assert checksum(ranking.out / "gatv2_promotion.json") == frozen
        copied = json.loads((ranking.out / "selection_evidence/gatv2/promotion/a/summary.json").read_text())
        assert copied["best_validation_f1"] == .80 and copied["current_epoch"] == 20
        queue = Harness.__new__(Harness)
        queue.root, queue.plan, queue.out = root, plan, root / "executed"
        queue.out.mkdir()
        queue.plan_path = root / "large_model_search_plan.json"
        queue.state, queue.events = {"stages": {}}, []
        # Synthetic guard validates each declared full-cohort command without
        # touching real molecular CSVs or executing a subprocess.
        guards = []
        def audit(path, cohort):
            derived = json.loads(Path(path).read_text())
            for family in derived["families"]:
                for run in family["runs"]:
                    options = flags(run["command"])
                    assert options["--graph-index"] == str(cohort) + "/graph_index.csv"
                    assert "--no-test" in options
                    guards.append((run["id"], int(options["--seed"])))
            return {"actual_train_unique_ids": 161157, "actual_cohort_unique_ids": 191701}
        prior = Path.cwd()
        try:
            os.chdir(root)
            with patch.object(search, "cohort_audit", audit):
                queue.work()
                first_events = list(queue.events)
                ranking_pins = {path.name: checksum(path) for path in queue.out.glob("*_promotion.json")}
                queue.work()  # All stages already completed; no work is relaunched.
                assert queue.events == first_events
                assert ranking_pins == {path.name: checksum(path) for path in queue.out.glob("*_promotion.json")}
        finally:
            os.chdir(prior)
        assert queue.state["status"] == "complete"
        keys = [key for key, _ in queue.events]
        assert sum(key.startswith("screen_") for key in keys) == 12
        assert sum(key.startswith("promote_") for key in keys) == 4
        assert sum(key.startswith("repeat_") for key in keys) == 4
        assert sum(key.endswith("_ensemble_locked_test") for key in keys) == 2
        assert [key.split("_", 2)[1] for key in keys[:4]] == ["gatv2", "geometric", "gatv2", "geometric"]
        for key, command in queue.events:
            if key.startswith("repeat_"):
                options = flags(command)
                assert options["--seed"] in {"43", "44"} and options["--epochs"] == "100"
                assert str(queue.out) in options.get("--out-dir", options.get("--out"))
        unique_repeats = {(ident, seed) for ident, seed in guards if "champion_seed" in ident}
        assert len(unique_repeats) == 4 and sorted(seed for ident, seed in unique_repeats) == [43, 43, 44, 44]
        import torch
        stopped = {"id": "already_stopped", "out_dir": str(root / "already_stopped"),
            "command": ["fixture", "fixture.py", "--patience", "12", "--epochs", "20", "--no-test"]}
        publish(stopped, plan["training_cohort"]["cohort_sha256"], .77)
        torch.save({"bad_epochs": 12}, Path(stopped["out_dir"]) / "last.pt")
        before = len(queue.events)
        assert queue.train(stopped, "promote_already_stopped", 100) == .77
        assert len(queue.events) == before
        assert queue.state["stages"]["promote_already_stopped"]["execution_skipped"]
        report = json.loads((Path(stopped["out_dir"]) / "summary.json").read_text())
        report["test_evaluated"] = True
        atomic_json(Path(stopped["out_dir"]) / "summary.json", report)
        try:
            search.validation_score(stopped, plan["training_cohort"]["cohort_sha256"])
        except ValueError as error:
            assert "already accessed test" in str(error)
        else:
            raise AssertionError("A test-exposed candidate entered validation search")
    print("PASS: real GAT CLI compatibility, immutable ranking after summary overwrite, idempotent full rerun, full-cohort seed clones, literal early stop, interleaved phases and no test before validation freeze")


if __name__ == "__main__":
    main()
