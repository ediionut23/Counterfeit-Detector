#!/usr/bin/env python3
"""Temporary fixture checks; never touches the production molecule bank."""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from generation_job_host import NativeJobHost, NativeJobTimeout, acquire_compute_lease
from yield_aware_scheduling import YieldAwareScheduler


def fixture_job(job):
    if job.get("sleep"):
        ctypes.PyDLL(None).sleep(int(job["sleep"]))  # Native call retains GIL; parent deadline must still work.
    import random
    import numpy as np
    return {"job_id": job["job_id"], "random": [random.random(), float(np.random.random())],
            "payload": b"x" * int(job.get("payload_bytes", 0))}


def frozen_core_job(job):
    import expand_million_bank as core
    core.worker_init(job["fixture_db"], {}, job["quality"])
    return core.run_job(job)


def chemistry_jobs(db):
    from rdkit import Chem
    config = json.loads(Path("million_bank_yield_config.json").read_text())
    reference = {"reference_id": "fixture_ibuprofen", "name": "IBUPROFEN",
                 "smiles": Chem.MolToSmiles(Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")),
                 "source": "local_fixture", "source_id": "fixture", "status": "fixture", "generation_eligible": "1"}
    jobs = []
    for method in ("M", "E"):
        jobs.append({"job_id": "fixture_" + method, "config_hash": "fixture", "method": method,
            "reference": reference, "category": "homologation", "difficulty": "fixture_declared_broad_band",
            "band": {"sim_lo": 0.35, "sim_hi": 0.98, "similarity_target": 0.65, "m_depth": 1},
            "quality": config["quality"], "rules_path": str(Path(config["rules_path"]).resolve()),
            "mutation": {**config["mutation"], "max_candidates": 4},
            "evolution": {**config["evolution"], "pop": 4, "generations": 2, "max_candidates": 4},
            "scenario": "synthetic_structural_analogue_substitution", "seed": 42,
            "timeout_seconds": 30, "fixture_db": str(db)})
    return jobs, config


def core_signatures(result):
    return sorted((c["molecule"]["molecule_id"], c["molecule"]["smiles"],
                   c["proposal"].get("operator_history_json") or c["proposal"].get("operator_history"))
                  for c in result["candidates"])


def lease_owner(directory):
    directory = Path(directory)
    with NativeJobHost(job_function=fixture_job, lease_path=directory / "compute.lock") as host:
        host._start()
        (directory / "owned.json").write_text(json.dumps({"owner": os.getpid(), "child": host.child_pid}))
        host.run({"job_id": "owned_orphan_fixture", "seed": 42, "sleep": 30, "timeout_seconds": 60})


def self_check():
    with tempfile.TemporaryDirectory(prefix="yield_bank_check_") as temporary:
        temporary = Path(temporary)
        lease = temporary / "native.lock"
        with NativeJobHost(job_function=fixture_job, lease_path=lease) as host:
            first = host.run({"job_id": "seed_fixture", "seed": 123, "timeout_seconds": 3})
            pid = host.child_pid
            again = host.run({"job_id": "seed_fixture", "seed": 123, "timeout_seconds": 3})
            assert first["random"] == again["random"] and host.child_pid == pid
            large = host.run({"job_id": "large_pipe_fixture", "seed": 42, "payload_bytes": 8 * 1024 * 1024, "timeout_seconds": 3})
            assert len(large["payload"]) == 8 * 1024 * 1024
            started = time.monotonic()
            try:
                host.run({"job_id": "native_timeout", "seed": 42, "sleep": 10, "timeout_seconds": 0.30})
                raise AssertionError("Native timeout was not enforced")
            except NativeJobTimeout:
                pass
            assert time.monotonic() - started < 2.0 and host.process is None
            recovered = host.run({"job_id": "seed_fixture", "seed": 123, "timeout_seconds": 3})
            assert host.child_pid != pid and recovered["random"] == first["random"]
            for value in (float("nan"), float("inf"), -1):
                try:
                    host.run({"job_id": "invalid_budget", "seed": 1, "timeout_seconds": value})
                    raise AssertionError("Invalid deadline accepted")
                except ValueError:
                    pass

        # Parent death: inherited flock prevents a replacement compute child.
        orphan = temporary / "orphan"
        orphan.mkdir()
        owner = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--lease-owner", str(orphan)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child_pid = None
        try:
            deadline = time.monotonic() + 10
            while not (orphan / "owned.json").exists() and time.monotonic() < deadline:
                if owner.poll() is not None:
                    raise AssertionError("Fixture owner exited before readiness")
                time.sleep(0.02)
            info = json.loads((orphan / "owned.json").read_text())
            assert info["owner"] == owner.pid
            child_pid = info["child"]
            owner.kill()
            owner.wait(timeout=2)
            try:
                unexpected = acquire_compute_lease(orphan / "compute.lock")
                unexpected.close()
                raise AssertionError("Orphan child did not retain its compute lease")
            except RuntimeError as error:
                assert "lease is already held" in str(error)
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=2)
            if child_pid:
                try:
                    os.kill(child_pid, signal.SIGTERM)  # Only the newly created fixture descendant.
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3
        while True:
            try:
                released = acquire_compute_lease(orphan / "compute.lock")
                released.close()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise AssertionError("Fixture orphan did not release its lease after termination")
                time.sleep(0.02)

        # No-production chemical/QC parity against the original frozen backend.
        from scalable_bank_store import BankStore, chemistry_init
        chemistry_init({})
        db = temporary / "fixture.sqlite"
        store = BankStore(db)
        store.connection.close()
        jobs, config = chemistry_jobs(db)
        results = {}
        with NativeJobHost(db, {}, config["quality"]) as host, NativeJobHost(job_function=frozen_core_job) as old:
            for job in jobs:
                reference = old.run(job)
                actual = host.run(job)
                assert actual["produced"] > 0 and actual["candidates"], job["method"]
                assert core_signatures(actual) == core_signatures(reference), job["method"]
                repeated = host.run(job)
                assert core_signatures(repeated) == core_signatures(actual), job["method"]
                for candidate in actual["candidates"]:
                    molecule = candidate["molecule"]
                    assert molecule["training_eligible"] and molecule["structural_alerts"] == 0
                    assert molecule["qed"] >= config["quality"]["qed_min"] and molecule["sa_score"] <= config["quality"]["max_sa"]
                results[job["method"]] = len(actual["candidates"])

        # Productive bands win exploitation; every band remains explored.
        scheduler = YieldAwareScheduler({"minimum_named_category_unique": {"M": 0, "E": 0},
            "minimum_active_time_share": {"M": 0, "E": 0}})
        cells = [("E", "test", band) for band in ("near", "intermediate", "distant")]
        for _ in range(32):
            scheduler.record(cells[0], 0, 60, current_policy=False)
            scheduler.record(cells[1], 15, 60, current_policy=False)
            scheduler.record(cells[2], 45, 60, current_policy=False)
        choices = Counter()
        for _ in range(200):
            selected = scheduler.choose(cells)
            choices[selected[2]] += 1
            scheduler.record(selected, {"near": 0, "intermediate": 15, "distant": 45}[selected[2]], 60)
        assert choices["distant"] >= 160 and choices["near"] >= 10 and choices["intermediate"] >= 10, choices
        time_scheduler = YieldAwareScheduler({"minimum_named_category_unique": {"M": 0, "E": 0}})
        method_choices = [("M", "test", "distant"), ("E", "test", "distant")]
        for _ in range(200):
            selected = time_scheduler.choose(method_choices)
            time_scheduler.record(selected, 50 if selected[0] == "M" else 5, 1)
        assert time_scheduler.active_seconds["E"] >= 0.25 * sum(time_scheduler.active_seconds.values()) - 1
        assert time_scheduler.attempts[method_choices[0]] > time_scheduler.attempts[method_choices[1]]
        coverage_scheduler = YieldAwareScheduler({"minimum_active_time_share": {"M": 0, "E": 0}},
            {("E", "well_covered"): 25000, ("E", "deficit"): 16000})
        assert coverage_scheduler.choose([("E", "well_covered", "distant"), ("E", "deficit", "distant")])[1] == "deficit"

    return {"status": "passed", "hard_native_deadline": True, "persistent_child_restarts_after_timeout": True,
        "large_pipe_result_without_deadlock": True, "orphan_child_retains_single_compute_lease": True,
        "M_E_accepted_same_as_frozen_chemistry": results, "seed_reproducible": True,
        "productive_vs_zero_yield_band_choices": dict(choices), "E_minimum_time_share_preserved": True,
        "production_bank_touched": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lease-owner")
    args = parser.parse_args()
    if args.lease_owner:
        lease_owner(args.lease_owner)
    else:
        print(json.dumps(self_check(), indent=2))
