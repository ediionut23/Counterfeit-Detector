#!/usr/bin/env python3
"""Conservative training-only thermal guard with an exclusive chemistry lease.

Uses the frozen guard's birth/token/lease verification. Generation cannot acquire
its compute lease while this guard is active. A guard exit during training holds
the verified trainer paused instead of leaving unguarded compute running.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
from pathlib import Path

import thermal_guard as engine

engine.SUPERVISORS.add('run_validation_continuation.py')


class CoolingGuard(engine.Guard):
    def __init__(self, *args, policy_metadata=None, **kwargs):
        self.policy_metadata = policy_metadata or {}
        super().__init__(*args, **kwargs)

    def save(self):
        self.state.update(cooling_policy=self.policy_metadata,
                          generation_excluded_by_compute_lease=True)
        super().save()

    def resume_all(self, reason):
        if reason != 'guard_exit_recovery':
            return super().resume_all(reason)
        # Fail closed on guard shutdown: never leave a live trainer computing
        # without cooling supervision. Keep durable identities for recovery.
        for path in self.progress:
            owned = engine.owned_training(path, self.workspace, self.reader)
            if owned:
                self.pause(owned, 'guard_exit_holds_training_for_cooling')
        if not self.state['paused']:
            return super().resume_all(reason)
        self.event('guard_exit_training_held', {'paused_count': len(self.state['paused'])})


def terminal_queue(progress):
    state = engine.load_json(progress)
    return state.get('status') in {'validation_complete', 'complete', 'failed', 'stopped'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--state', type=Path, default=Path('training_runs/thermal_guard_20261107/state.json'))
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--chemistry-lock', type=Path, default=Path('molecule_bank_1m/bank.chemistry.lock'))
    parser.add_argument('--compute-seconds', type=float, default=15)
    parser.add_argument('--rest-seconds', type=float, default=45)
    parser.add_argument('--poll-seconds', type=float, default=2)
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.compute_seconds, args.rest_seconds, args.poll_seconds)) or not 0 < args.compute_seconds <= 15 or args.rest_seconds < 45 or not 0 < args.poll_seconds <= 2:
        parser.error('Conservative policy requires compute<=15s, rest>=45s, polling<=2s')
    workspace = args.workspace.resolve()
    lock_path = (workspace / args.chemistry_lock).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = {'nominal_compute_seconds': args.compute_seconds,
                    'nominal_rest_seconds': args.rest_seconds,
                    'fair_or_unknown_compute_seconds': 5,
                    'fair_or_unknown_rest_seconds': 55,
                    'sustained_nominal_recovery_seconds': 120,
                    'poll_seconds': args.poll_seconds,
                    'chemistry_lock': str(lock_path),
                    'guard_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'temperature_celsius_available': False}
        guard = CoolingGuard([args.progress], args.state, workspace,
                             poll=args.poll_seconds,
                             policy=engine.DutyPolicy(compute=args.compute_seconds, rest=args.rest_seconds,
                                                      fair_compute=5, fair_rest=55, recovery=120),
                             policy_metadata=metadata)
        def read_thermal():
            if terminal_queue(args.progress):
                guard.stopping = True
            return engine.thermal_state()
        guard.thermal_reader = read_thermal
        try:
            guard.run()
        except RuntimeError:
            if guard.state.get('paused'):
                guard.state.update(status='stopped_training_held', generation_excluded_by_compute_lease=False)
                guard.save()
            raise
        finally:
            guard.state['generation_excluded_by_compute_lease'] = False
            # Bypass subclass.save only to record that the lease is now released.
            engine.Guard.save(guard)


if __name__ == '__main__':
    main()
