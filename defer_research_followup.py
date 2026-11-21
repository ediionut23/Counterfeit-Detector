#!/usr/bin/env python3
"""Wait for validation training, then run one cooled finalization queue.

The original model queue lease remains held throughout finalization. A separate
cooling guard owns the chemistry lease. Inventory's ready marker is published
only by successful, immutable ensemble finalization.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from run_overnight_research import atomic_json, checksum, load_json, now
from thermal_guard import ProcessReader, lease_held, owned_training, same_process


def verify_plan(path, expected):
    if checksum(path) != expected:
        raise ValueError('Finalization plan changed')
    plan = load_json(path)
    for filename, digest in plan['file_pins'].items():
        if checksum(Path(plan['workspace']) / filename) != digest:
            raise ValueError('Pinned finalization source changed: ' + filename)
    return plan


def spawn(command, log_path, root, pass_fds=()):
    with Path(log_path).open('a') as log:
        return subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds,
            env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))


def run(args):
    plan = verify_plan(args.plan, args.plan_sha256)
    root = Path(plan['workspace']).resolve()
    os.chdir(root)
    directory = root / plan['out_dir']
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / 'deferred_followup.json'
    record = {'status': 'waiting', 'pid': os.getpid(), 'startup_token': args.startup_token,
              'plan_sha256': args.plan_sha256, 'created_at': now(), 'reason': 'waiting_for_validation_complete'}
    reader = ProcessReader()
    guard = queue = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with (directory / 'deferred_followup.lock').open('a+') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(record_path, record)
        try:
            while not stopping:
                state = load_json(args.model_progress)
                if state.get('status') in {'failed', 'stopped'}:
                    raise RuntimeError('Model continuation failed/stopped; finalization remains unstarted')
                hold = load_json(root / 'training_runs/manual_cooling_hold.json')
                if hold.get('status') in {'paused', 'pause_intended'}:
                    record.update(reason='waiting_for_manual_cooling_release')
                elif state.get('status') == 'validation_complete':
                    original_queue = root / Path(args.model_progress).parent / 'queue.lock'
                    with ExitStack() as locks:
                        queue_lease = locks.enter_context(original_queue.open('a+'))
                        try:
                            fcntl.flock(queue_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            record.update(reason='waiting_for_training_queue_exit')
                        else:
                            if lease_held(root / 'molecule_bank_1m/bank.chemistry.lock'):
                                record.update(reason='waiting_for_previous_thermal_guard_release')
                            else:
                                verify_plan(args.plan, args.plan_sha256)
                                progress_path = directory / 'progress.json'
                                if not progress_path.exists():
                                    atomic_json(progress_path, {'status': 'waiting', 'stages': {}})
                                guard_state = directory / 'cooling_guard.json'
                                command = [sys.executable, '-u', 'thermal_guard_cool_v2.py',
                                    '--progress', str(progress_path), '--state', str(guard_state),
                                    '--compute-seconds', '15', '--rest-seconds', '45', '--poll-seconds', '2',
                                    '--allow-stage-script', 'finish_heterogeneous_research.py']
                                guard = spawn(command, directory / 'cooling_guard.log', root)
                                deadline = time.monotonic() + 20
                                while time.monotonic() < deadline:
                                    if guard.poll() is not None and load_json(progress_path).get('status') != 'complete':
                                        raise RuntimeError('Finalization cooling guard failed to start')
                                    thermal = load_json(guard_state)
                                    if thermal.get('guard_pid') == guard.pid and thermal.get('status') in {'compute_window', 'cooling'} and lease_held(root / 'molecule_bank_1m/bank.chemistry.lock'):
                                        break
                                    time.sleep(.2)
                                else:
                                    raise RuntimeError('Finalization cooling guard readiness not acknowledged')
                                token = uuid.uuid4().hex
                                queue = spawn([sys.executable, '-u', 'run_overnight_research.py', '--plan', str(args.plan.resolve()),
                                    '--startup-token', token], directory / 'supervisor.log', root, (queue_lease.fileno(),))
                                record.update(status='finalization_running', queue_pid=queue.pid, guard_pid=guard.pid,
                                              queue_startup_token=token, started_at=now(), reason=None)
                                atomic_json(record_path, record)
                                last_write = 0
                                while queue.poll() is None:
                                    if guard.poll() is not None:
                                        # This queue supervisor is our live direct child;
                                        # the helper checks its exact token/lease/argv chain.
                                        import thermal_guard as ownership
                                        ownership.TRAINERS.add('finish_heterogeneous_research.py')
                                        target = owned_training(progress_path, root, reader)
                                        if target:
                                            current, _ = reader.snapshot(target['fingerprint']['pid'])
                                            if not same_process(current, target['fingerprint'], parent=True):
                                                raise RuntimeError('Finalization identity changed before emergency cooling hold')
                                            record['emergency_pause_intent'] = target
                                            atomic_json(record_path, record)
                                            os.kill(current['pid'], signal.SIGSTOP)
                                        raise RuntimeError('Finalization guard exited; verified active stage held for recovery')
                                    if stopping:
                                        (directory / 'STOP').touch(exist_ok=True)
                                    if time.monotonic() - last_write > 15:
                                        record.update(updated_at=now(), current_stage=load_json(progress_path).get('current_stage'))
                                        atomic_json(record_path, record)
                                        last_write = time.monotonic()
                                    time.sleep(2)
                                record.update(status='complete' if queue.returncode == 0 else 'failed',
                                              returncode=queue.returncode, finished_at=now())
                                atomic_json(record_path, record)
                                return queue.returncode
                else:
                    record.update(reason='waiting_for_validation_complete')
                record.update(updated_at=now())
                atomic_json(record_path, record)
                deadline = time.monotonic() + 30
                while not stopping and time.monotonic() < deadline:
                    time.sleep(.2)
            record.update(status='stopped', updated_at=now())
            atomic_json(record_path, record)
            return 0
        except BaseException as error:
            record.update(status='failed', error=repr(error), updated_at=now())
            atomic_json(record_path, record)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--model-progress', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--startup-token')
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.model_progress = args.model_progress.resolve()
    plan = verify_plan(args.plan, args.plan_sha256)
    out = Path(plan['workspace']) / plan['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        token = uuid.uuid4().hex
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--plan', str(args.plan),
            '--plan-sha256', args.plan_sha256, '--model-progress', str(args.model_progress), '--startup-token', token]
        process = spawn(command, out / 'deferred_followup.log', plan['workspace'])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = load_json(out / 'deferred_followup.json')
            if status.get('pid') == process.pid and status.get('startup_token') == token:
                print(json.dumps({'startup_verified': True, 'pid': process.pid, 'status': status.get('status'), 'progress': str(out / 'deferred_followup.json')}))
                return
            if process.poll() is not None:
                raise RuntimeError('Follow-up waiter failed to start')
            time.sleep(.2)
        raise RuntimeError('Follow-up waiter did not acknowledge startup')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
