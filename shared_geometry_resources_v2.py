"""Exclusive, conservatively cooled compute for the new chemistry/geometry pipeline.

The existing Ax/generation lease is respected. This module never signals another
process, changes a queue, or takes over an existing owner's lock.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

from accumulated_generation_cooling import AccumulatedThermalGate


@contextmanager
def exclusive_compute(workspace, status_dir=None):
    """Fail before heavy work if Ax, another trainer, or generation owns compute."""
    workspace = Path(workspace).resolve()
    path = workspace / "molecule_bank_1m" / "bank.chemistry.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Compute is already leased by Ax, generation, or another trainer. "
                "The shared-feature v2 job has not started; retry after that owner finishes."
            ) from error
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class ComputeBudget:
    """Account preparation, batch loading and forward/backward as bounded work.

Native operations are allowed to finish. All actual elapsed work is charged and
its cooling debt is repaid before another interval, including on long overruns.
"""
    def __init__(self, status_dir, *, clock=None, sleeper=None):
        self.status_dir = Path(status_dir)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.gate = AccumulatedThermalGate(
            self.status_dir, rest_seconds=.05, sustained_nominal_seconds=120,
            nominal_active_budget_seconds=15, nominal_rest_ratio=3,
            fair_active_budget_seconds=5, fair_rest_ratio=11,
            clock=clock, sleeper=sleeper,
        )
        self._closed = False

    @contextmanager
    def interval(self, job_id):
        if self._closed:
            raise RuntimeError("Compute budget is already closed")
        self.gate.wait(str(job_id))
        try:
            yield
        finally:
            self.gate.finished()

    def batches(self, loader, job_id):
        iterator = iter(loader)
        index = 0
        while True:
            with self.interval(f"{job_id}:{index}"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                yield batch
            index += 1

    def close(self):
        if self._closed:
            return
        self.gate.finished()
        # Include coordinator work since the last batch/record before settling.
        if self.gate.finished_at is not None:
            self.gate._add_work(self.gate.clock() - self.gate.finished_at)
            self.gate.finished_at = None
        debt = max(.05, self.gate.rest_debt)
        self.gate._rest(debt, f"shared_geometry_v2_finish:{os.getpid()}", None,
                        self.gate.state(), "settling_final_cooling_debt")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
