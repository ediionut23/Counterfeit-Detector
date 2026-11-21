"""Accumulate short-job work before cooling; never pause a native chemistry job."""
from __future__ import annotations
import math
import time
from generation_job_host import ThermalGate as LegacyGate, JobDeferred


class AccumulatedThermalGate(LegacyGate):
    def __init__(self, status_dir, rest_seconds=.05, sustained_nominal_seconds=120,
                 nominal_active_budget_seconds=15, nominal_rest_ratio=3,
                 fair_active_budget_seconds=5, fair_rest_ratio=11,
                 clock=None, sleeper=None):
        super().__init__(status_dir, rest_seconds, sustained_nominal_seconds)
        values = (rest_seconds, sustained_nominal_seconds, nominal_active_budget_seconds,
                  nominal_rest_ratio, fair_active_budget_seconds, fair_rest_ratio)
        if not all(math.isfinite(x) for x in values) or rest_seconds < .05 or sustained_nominal_seconds < 120:
            raise ValueError('Cooling parameters must be finite with minimum rest .05s and recovery120s')
        if not 0 < nominal_active_budget_seconds <= 15 or nominal_rest_ratio < 3 or not 0 < fair_active_budget_seconds <= 5 or fair_rest_ratio < 11:
            raise ValueError('Require nominal<=15s/25%duty and fair<=5s/8.33%duty')
        self.nominal_budget, self.nominal_ratio = nominal_active_budget_seconds, nominal_rest_ratio
        self.fair_budget, self.fair_ratio = fair_active_budget_seconds, fair_rest_ratio
        self.clock, self.sleeper = clock or time.monotonic, sleeper or time.sleep
        self.active_wall = self.rest_debt = 0.0
        self.total_active_wall = self.total_rest_wall = 0.0
        self.compute_started = self.finished_at = None
        self.mode = None
        self.latched = False
        self.nominal_since = None

    def _add_work(self, elapsed):
        elapsed = max(0.0, float(elapsed))
        self.active_wall += elapsed
        self.total_active_wall += elapsed
        self.rest_debt += elapsed * (self.fair_ratio if self.mode == 'fair_or_unknown' else self.nominal_ratio)

    def _control(self, function, *args, **kwargs):
        before = self.clock()
        try:
            return function(*args, **kwargs)
        finally:
            self._add_work(self.clock() - before)

    def _rest(self, seconds, job_id, stop_requested, state, action):
        start = self.clock()
        remaining = max(0.0, seconds)
        self._control(self.record, state, action, job_id, **self.summary())
        while remaining > 1e-8:
            if stop_requested and stop_requested():
                raise JobDeferred('Stop requested during accumulated chemistry cooling')
            before = self.clock()
            self.sleeper(min(1.0, remaining))
            elapsed = max(0.0, self.clock() - before)
            self.total_rest_wall += elapsed
            self.rest_debt = max(0.0, self.rest_debt - elapsed)
            remaining -= elapsed
        return self.clock() - start

    def wait(self, job_id, stop_requested=None):
        if self.compute_started is not None:
            # Defensive recovery for any caller that failed to close a measured
            # interval. NativeJobHost's finally already calls finished on errors.
            self.finished()
        started = self.clock()
        if self.finished_at is not None:
            # Include coordinator/import/progress wall work, conservatively,
            # rather than treating only the chemistry child as heat-producing.
            self._add_work(self.clock() - self.finished_at)
            self.finished_at = None
        while True:
            if stop_requested and stop_requested():
                raise JobDeferred('Stop requested before chemistry starts')
            state = self._control(self.state)
            current = self.clock()
            if state in {2, 3}:
                self.latched, self.nominal_since = True, None
            if self.latched:
                if state == 0:
                    self.nominal_since = self.nominal_since if self.nominal_since is not None else current
                    if current - self.nominal_since >= self.sustained:
                        self.latched = False
                    else:
                        self._rest(1, job_id, stop_requested, state, 'waiting_for120s_sustained_nominal')
                        continue
                else:
                    self.nominal_since = None
                    self._rest(1, job_id, stop_requested, state, 'serious_or_critical_cooling_hold')
                    continue
            mode = 'nominal' if state == 0 else 'fair_or_unknown'
            budget, ratio = ((self.nominal_budget, self.nominal_ratio) if mode == 'nominal'
                             else (self.fair_budget, self.fair_ratio))
            if mode == 'fair_or_unknown' and self.mode != mode:
                self.rest_debt = max(self.rest_debt, self.active_wall * ratio, 55.0)
            self.mode = mode
            if self.active_wall >= budget or self.rest_debt >= 55 and mode == 'fair_or_unknown':
                self._rest(max(self.rest_seconds, self.rest_debt), job_id, stop_requested, state,
                           'repaying_accumulated_cooling_debt')
                self.active_wall = 0.0
                # Recheck pressure after the full cool period before permitting
                # the next native job. Rest is outside its fixed90s deadline.
                continue
            self._rest(self.rest_seconds, job_id, stop_requested, state, 'minimum_interjob_rest')
            self._control(self.record, state, 'running_one_compute_child_accumulated_budget', job_id, **self.summary())
            self.compute_started = self.clock()
            return self.clock() - started

    def finished(self):
        if self.compute_started is not None:
            self._add_work(self.clock() - self.compute_started)
            self.compute_started = None
            self.finished_at = self.clock()

    def summary(self):
        return {'cooling_policy': 'accumulated_active_wall_v1', 'active_wall_in_budget': self.active_wall,
                'cooling_debt_seconds': self.rest_debt, 'total_accounted_active_wall_seconds': self.total_active_wall,
                'total_cooling_wall_seconds': self.total_rest_wall,
                'nominal_active_budget_seconds': self.nominal_budget, 'nominal_rest_ratio': self.nominal_ratio,
                'fair_active_budget_seconds': self.fair_budget, 'fair_rest_ratio': self.fair_ratio,
                'native_job_preemption': False,
                'accounted_work_includes': 'nativejobwall,coordinatorwall,pressureprobe,statuswrites',
                'long_job_overrun_policy': 'native90sdeadlineunchanged;repayfullcoolingdebtbeforeanynextjob'}
