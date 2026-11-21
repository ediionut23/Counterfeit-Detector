#!/usr/bin/env python3
"""Conservative guard that holds leased orphan workloads instead of resuming them.

Default stage scripts are trainers only. Other owned Python stage scripts must
be explicitly allowlisted on the command line. Generation is excluded with the
same exclusive chemistry lease used by its compute host. SIGTERM is a controlled
handoff: verified workloads are paused before this guard releases its lease.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
import os
from pathlib import Path
import signal
import time

import thermal_guard as engine


DEFAULT_STAGE_SCRIPTS = frozenset({'train_gatv2_experiment.py', 'train_gatv2_search.py',
    'train_geometric_gnn.py', 'train_geometric_variants.py'})
SUPERVISORS = frozenset({'run_large_model_search.py', 'run_validation_continuation.py', 'run_overnight_research.py'})
TERMINAL = frozenset({'validation_complete', 'complete', 'failed', 'stopped'})
engine.SUPERVISORS.update(SUPERVISORS)


def stage_candidates(queue):
    current = queue.get('current_stage')
    keys = [current] if current else []
    keys += [key for key, value in queue.get('stages', {}).items()
             if key != current and value.get('status') in {'running', 'starting', 'failed'}]
    return keys


def owned_stage_paths(progress_path, stage):
    directory = Path(progress_path).resolve().parent / 'stages'
    status = Path(stage['stage_status_path']).resolve()
    lease = Path(stage['stage_lock_path']).resolve()
    if status.parent != directory or lease.parent != directory or not status.name.endswith('.status.json'):
        raise ValueError('Stage ownership paths escape the selected queue')
    spec = status.with_name(status.name.replace('.status.json', '.spec.json'))
    return status, lease, spec


def prior_for(prior_records, progress, stage, child_pid, token):
    for record in prior_records:
        if (record.get('progress_path') == str(progress) and record.get('stage') == stage
                and record.get('wrapper_token') == token
                and record.get('fingerprint', {}).get('pid') == child_pid):
            return record
    return None


def verify_leased_training(progress_path, workspace, reader, *, allowed_scripts=DEFAULT_STAGE_SCRIPTS,
                           prior_records=()):
    """Verify a healthy or orphan stage, including wrapper-death cached identity.

An orphan is owned enough to pause, never enough to resume. A dead wrapper
requires a previously verified child birth identity, as its parent chain can no
longer be established from live processes alone.
"""
    progress, workspace = Path(progress_path).resolve(), Path(workspace).resolve()
    queue = engine.load_json(progress)
    candidates = []
    for key in stage_candidates(queue):
        stage = queue['stages'][key]
        if not stage.get('stage_status_path') or not stage.get('stage_lock_path'):
            continue
        status_path, lease_path, spec_path = owned_stage_paths(progress, stage)
        if not lease_path.exists() or not engine.lease_held(lease_path):
            continue
        status, spec = engine.load_json(status_path), engine.load_json(spec_path)
        if status.get('status') != 'running':
            raise ValueError('Live stage lease has no running wrapper status')
        token = stage.get('wrapper_token')
        if not token or not token == status.get('wrapper_token') == spec.get('wrapper_token'):
            raise ValueError('Stage wrapper token mismatch')
        command_hash = stage.get('command_sha256')
        if not command_hash or not command_hash == status.get('command_sha256') == spec.get('command_sha256'):
            raise ValueError('Stage command fingerprint mismatch')
        command = status.get('command')
        if not command or command != spec.get('command') or command != stage.get('command'):
            raise ValueError('Stage command payload mismatch')
        if Path(spec.get('workspace', '')).resolve() != workspace or Path(spec.get('lock_path', '')).resolve() != lease_path:
            raise ValueError('Stage workspace or lease mismatch')
        if Path(spec.get('status_path', '')).resolve() != status_path:
            raise ValueError('Stage status path mismatch')
        script = engine.script_argument(command)
        if script is None or Path(script).name not in allowed_scripts:
            raise ValueError('Live stage script is not explicitly allowlisted')
        if (workspace / script).resolve() != workspace / Path(script).name:
            raise ValueError('Stage script escapes the workspace')

        # Preserve the complete frozen healthy-chain verifier where applicable.
        healthy = None
        if queue.get('status') == 'running' and queue.get('current_stage') == key:
            try:
                engine.TRAINERS.update(allowed_scripts)
                healthy = engine.owned_training(progress, workspace, reader)
            except ProcessLookupError:
                pass  # The independently leased wrapper may outlive its queue.
        if healthy is not None:
            healthy.update(orphan=False, ownership_mode='healthy_queue_chain')
            candidates.append(healthy)
            continue

        child, child_argv = reader.snapshot(status['child_pid'])
        if child['uid'] != os.getuid() or child['executable'] != engine.canonical_executable(command[0]):
            raise ValueError('Stage child UID or executable differs')
        if engine.canonical_command(child_argv) != engine.canonical_command(command):
            raise ValueError('Stage child argv differs from its leased command')
        prior = prior_for(prior_records, progress, key, child['pid'], token)
        if prior and not engine.same_process(prior['fingerprint'], child):
            raise ValueError('Stage child birth identity changed')
        try:
            wrapper, wrapper_argv = reader.snapshot(status['wrapper_pid'])
        except ProcessLookupError:
            if not prior or child['ppid'] != 1:
                raise ValueError('Dead wrapper requires the exact previously verified orphan birth')
            old_wrapper = prior.get('wrapper_fingerprint', {})
            if old_wrapper.get('pid') != status['wrapper_pid']:
                raise ValueError('Cached wrapper identity differs')
            record = {**prior, 'fingerprint': child, 'orphan': True,
                      'ownership_mode': 'cached_birth_after_wrapper_death'}
            candidates.append(record)
            continue
        if wrapper['uid'] != os.getuid() or wrapper['executable'] != engine.canonical_executable(command[0]):
            raise ValueError('Stage wrapper UID or executable differs')
        expected_wrapper = [engine.canonical_executable(command[0]), '-u', str(workspace / 'run_overnight_research.py'),
                            '--stage-wrapper', str(spec_path), '--stage-lease-fd']
        actual_wrapper = engine.canonical_command(wrapper_argv)
        if (len(actual_wrapper) != len(expected_wrapper) + 1 or actual_wrapper[:-1] != expected_wrapper
                or not actual_wrapper[-1].isdigit() or int(actual_wrapper[-1]) < 3
                or status['wrapper_pid'] != stage.get('pid')):
            raise ValueError('Live wrapper command differs from the exact stage invocation')
        if child['ppid'] != wrapper['pid']:
            raise ValueError('Stage child is not a child of its verified wrapper')
        if prior and not engine.same_process(prior['wrapper_fingerprint'], wrapper):
            raise ValueError('Stage wrapper birth identity changed')
        queue_fingerprint = None
        queue_alive = False
        try:
            live_queue, queue_argv = reader.snapshot(queue['pid'])
            queue_alive = (live_queue['uid'] == os.getuid()
                and Path(engine.script_argument(queue_argv) or '').name in SUPERVISORS
                and engine.option_value(queue_argv, '--startup-token') == queue.get('startup_token')
                and live_queue['executable'] == wrapper['executable'])
            if queue_alive:
                queue_fingerprint = live_queue
        except ProcessLookupError:
            pass
        if wrapper['ppid'] != 1 and not (queue_alive and wrapper['ppid'] == queue_fingerprint['pid']):
            raise ValueError('Wrapper has neither its verified queue nor launchd as parent')
        orphan = queue.get('status') != 'running' or not queue_alive
        record = {'fingerprint': child, 'wrapper_fingerprint': wrapper,
            'queue_fingerprint': queue_fingerprint, 'queue_token': queue.get('startup_token'),
            'wrapper_token': token, 'command_sha256': command_hash,
            'progress_path': str(progress), 'stage': key, 'orphan': orphan,
            'ownership_mode': 'leased_wrapper_fallback'}
        candidates.append(record)
    if len(candidates) > 1:
        raise ValueError('Multiple live leased stages require explicit recovery')
    return candidates[0] if candidates else None


class CoolingGuard(engine.Guard):
    def __init__(self, *args, allowed_scripts=DEFAULT_STAGE_SCRIPTS, policy_metadata=None, **kwargs):
        self.allowed_scripts = frozenset(allowed_scripts)
        self.policy_metadata = policy_metadata or {}
        self.compute_lease_held = True
        super().__init__(*args, **kwargs)
        self.cache = {str(record['fingerprint']['pid']): record
                      for record in self.state.get('last_verified_targets', []) + list(self.state['paused'].values())}

    def save(self):
        self.state.update(cooling_policy=self.policy_metadata,
            generation_excluded_by_compute_lease=self.compute_lease_held)
        if hasattr(self, 'cache'):
            self.state['last_verified_targets'] = list(self.cache.values())
        super().save()

    def live_cached(self):
        result = []
        for key, record in list(self.cache.items()):
            try:
                current, _ = self.reader.snapshot(record['fingerprint']['pid'])
            except ProcessLookupError:
                self.cache.pop(key, None)
                self.state['paused'].pop(key, None)
                continue
            if not engine.same_process(record['fingerprint'], current):
                self.event('cached_identity_retired', {'pid': current['pid']})
                self.cache.pop(key, None)
                self.state['paused'].pop(key, None)
                continue
            if current['ppid'] not in {record['fingerprint']['ppid'], 1}:
                raise ValueError('Previously verified child has an unexpected parent')
            record = {**record, 'fingerprint': current}
            self.cache[key] = record
            result.append(record)
        return result

    def pause(self, record, reason):
        # Reassert a verified pause even if an external SIGCONT occurred while
        # its durable ledger entry already existed.
        original = record['fingerprint']
        key = str(original['pid'])
        try:
            current, _ = self.reader.snapshot(original['pid'])
            if not engine.same_process(original, current, parent=True):
                self.cache.pop(key, None)
                self.state['paused'].pop(key, None)
                self.event('pause_rejected_identity_changed', {'pid': original['pid']})
                self.save()
                return
            first = key not in self.state['paused']
            self.state['paused'][key] = {**record, 'pause_intended': True, 'reason': reason}
            self.save()
            current, _ = self.reader.snapshot(original['pid'])
            if not engine.same_process(original, current, parent=True):
                self.cache.pop(key, None)
                self.state['paused'].pop(key, None)
                self.event('pause_rejected_identity_changed', {'pid': original['pid']})
                self.save()
                return
            os.kill(current['pid'], signal.SIGSTOP)
        except ProcessLookupError:
            # Normal stage completion can race a cooling poll. Retire this
            # identity without exiting and leaving the next stage unguarded.
            self.cache.pop(key, None)
            self.state['paused'].pop(key, None)
            self.save()
            return
        self.state['paused'][key]['pause_confirmed'] = True
        self.save()
        if first:
            self.event('paused', {'pid': current['pid'], 'reason': reason, 'stage': record['stage']})

    def resume_verified(self, records):
        for record in records:
            if record.get('orphan'):
                continue
            key = str(record['fingerprint']['pid'])
            if key not in self.state['paused']:
                continue
            # Re-read terminal state and the complete lease/chain immediately
            # before a signal. A cached identity alone can never authorize CONT.
            try:
                fresh = verify_leased_training(record['progress_path'], self.workspace, self.reader,
                    allowed_scripts=self.allowed_scripts, prior_records=list(self.cache.values()))
                if fresh is None or fresh.get('orphan') or not engine.same_process(record['fingerprint'], fresh['fingerprint']):
                    continue
                os.kill(fresh['fingerprint']['pid'], signal.SIGCONT)
            except ProcessLookupError:
                self.cache.pop(key, None)
                self.state['paused'].pop(key, None)
                self.save()
                continue
            except ValueError as error:
                self.state['last_error'] = repr(error)
                self.pause(record, 'ownership_changed_before_resume')
                continue
            self.state['paused'].pop(key, None)
            self.event('resumed', {'pid': fresh['fingerprint']['pid'], 'reason': 'verified_healthy_compute_window'})
            self.save()

    def hold_all(self, reason):
        for record in self.live_cached():
            self.pause(record, reason)

    def discover(self):
        found, errors = [], []
        for path in self.progress:
            try:
                record = verify_leased_training(path, self.workspace, self.reader,
                    allowed_scripts=self.allowed_scripts, prior_records=list(self.cache.values()))
                if record:
                    self.cache[str(record['fingerprint']['pid'])] = record
                    found.append(record)
            except Exception as error:
                errors.append(repr(error))
        return found, errors

    def queues_finished(self):
        for path in self.progress:
            queue = engine.load_json(path)
            if queue.get('status') not in TERMINAL:
                return False
            for stage in queue.get('stages', {}).values():
                if stage.get('stage_lock_path') and stage.get('stage_status_path'):
                    _, lease, _ = owned_stage_paths(path, stage)
                    if lease.exists() and engine.lease_held(lease):
                        return False
        return True

    def run(self):
        def stop(signum, frame):
            self.stopping = True
        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            # No startup SIGCONT: recover old pauses only after healthy ownership
            # and the current thermal policy both permit computation.
            while not self.stopping:
                found, errors = self.discover()
                cached = self.live_cached()
                orphan = any(record.get('orphan') for record in found)
                unknown_live = any(str(record['fingerprint']['pid']) not in {str(item['fingerprint']['pid']) for item in found}
                                   for record in cached)
                if not cached and not errors and self.queues_finished():
                    self.state.update(status='queues_finished', decision='no_live_leased_or_cached_workload')
                    break
                thermal = self.thermal_reader()
                pause, reason = self.policy.decision(thermal, time.monotonic())
                force_hold = bool(errors) or orphan or unknown_live
                if force_hold:
                    pause, reason = True, 'orphan_or_unverified_queue_held_until_controlled_recovery'
                self.state.update(status='orphan_held' if force_hold else 'cooling' if pause else 'compute_window',
                    thermal_state=thermal, thermal_name=engine.THERMAL_NAMES.get(thermal, 'unknown'),
                    decision=reason, verified_targets=found, last_error='; '.join(errors) if errors else None)
                if pause:
                    self.hold_all(reason)
                elif not self.stopping:
                    self.resume_verified(found)
                self.save()
                deadline = time.monotonic() + self.poll
                while not self.stopping and time.monotonic() < deadline:
                    time.sleep(min(.2, max(0, deadline - time.monotonic())))
        finally:
            # Include a stage that appeared since the last poll, then pause all
            # cached birth-verified children even if the supervisor has vanished.
            try:
                _, errors = self.discover()
                if errors:
                    self.state['last_error'] = '; '.join(errors)
                self.hold_all('controlled_guard_exit_holds_owned_workloads')
                self.state.update(status='stopped_training_held' if self.state['paused'] else 'stopped',
                    unrecovered_pause_records=len(self.state['paused']))
                self.save()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, action='append', required=True)
    parser.add_argument('--state', type=Path, default=Path('training_runs/thermal_guard_20261107/state.json'))
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--chemistry-lock', type=Path, default=Path('molecule_bank_1m/bank.chemistry.lock'))
    parser.add_argument('--compute-seconds', type=float, default=15)
    parser.add_argument('--rest-seconds', type=float, default=45)
    parser.add_argument('--poll-seconds', type=float, default=2)
    parser.add_argument('--allow-stage-script', action='append', default=[])
    args = parser.parse_args()
    values = (args.compute_seconds, args.rest_seconds, args.poll_seconds)
    if not all(math.isfinite(value) for value in values) or not 0 < args.compute_seconds <= 15 or args.rest_seconds < 45 or not 0 < args.poll_seconds <= 2:
        parser.error('Require compute<=15s, rest>=45s, polling<=2s')
    if any(Path(name).name != name or not name.endswith('.py') for name in args.allow_stage_script):
        parser.error('Additional stage scripts must be workspace Python basenames')
    workspace = args.workspace.resolve()
    lock_path = (workspace / args.chemistry_lock).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allowed = DEFAULT_STAGE_SCRIPTS | frozenset(args.allow_stage_script)
        metadata = {'nominal_compute_seconds': args.compute_seconds, 'nominal_rest_seconds': args.rest_seconds,
            'fair_or_unknown_compute_seconds': 5, 'fair_or_unknown_rest_seconds': 55,
            'sustained_nominal_recovery_seconds': 120, 'poll_seconds': args.poll_seconds,
            'chemistry_lock': str(lock_path), 'allowed_stage_scripts': sorted(allowed),
            'guard_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'temperature_celsius_available': False, 'orphan_policy': 'hold_paused_with_generation_excluded'}
        guard = CoolingGuard(args.progress, args.state, workspace, poll=args.poll_seconds,
            policy=engine.DutyPolicy(compute=args.compute_seconds, rest=args.rest_seconds, fair_compute=5, fair_rest=55, recovery=120),
            allowed_scripts=allowed, policy_metadata=metadata)
        try:
            guard.run()
        finally:
            guard.compute_lease_held = False
            guard.save()


if __name__ == '__main__':
    main()
