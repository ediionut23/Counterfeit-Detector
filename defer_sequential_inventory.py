#!/usr/bin/env python3
"""Wait for completed research and a pinned marker, then run inventory sequentially.

No process signals or old-state edits. The waiter's exclusive model queue and
trainer locks are retained throughout inventory. The model guard separately
owns the bank chemistry lease while training; the bank's existing native host
keeps that lease even if its coordinator dies. STOP_GENERATION is only removed
with the explicit --release-cooling-stop flag and exact owned-hold evidence.
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
from thermal_guard import (ProcessReader, canonical_command, canonical_executable,
                           option_value, script_argument, thermal_state)


class Waiting(Exception):
    pass


def strict_json(path):
    return json.loads(Path(path).read_text())


def check_ready(state, marker, protocol_sha256, model_out):
    if state.get('status') in {'failed', 'stopped'}:
        raise RuntimeError('Model research stopped/failed; inventory remains unstarted')
    if state.get('status') != 'validation_complete':
        raise Waiting('waiting_for_validation_complete')
    if not marker:
        raise Waiting('waiting_for_ready_marker')
    if marker.get('status') != 'done' or marker.get('protocol_sha256') != protocol_sha256:
        raise ValueError('Ready marker must have status=done and the declared protocol SHA256')
    if Path(marker.get('model_out_dir', '')).resolve() != Path(model_out).resolve():
        raise ValueError('Ready marker belongs to a different model output directory')


def lock_nonblocking(stack, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = stack.enter_context(path.open('a+'))
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise Waiting('waiting_for_free_lock:' + str(path))
    return stream


def existing_model_directories(plan, state):
    root = Path(plan['workspace'])
    paths = {root / run['out_dir'] for family in plan.get('families', []) for run in family['runs']}
    paths.update(Path(value) for value in state.get('adaptive_run_dirs', {}).values())
    return sorted(paths)


def verified_model_processes(state, reader):
    """Find only processes proved by queue token or exact stage argv/spec path."""
    found = []

    def snapshot(pid):
        if not isinstance(pid, int) or pid < 1:
            return None
        try:
            return reader.snapshot(pid)
        except ProcessLookupError:
            return None
        # Permission/identity errors fail closed rather than treating a PID as dead.

    queue = snapshot(state.get('pid'))
    if queue and option_value(queue[1], '--startup-token') == state.get('startup_token'):
        found.append({'role': 'model_supervisor', 'fingerprint': queue[0]})
    for stage in state.get('stages', {}).values():
        status_path = stage.get('stage_status_path')
        if not status_path:
            continue
        status = load_json(status_path)
        spec_path = str(Path(status_path).with_name(Path(status_path).name.replace('.status.json', '.spec.json')))
        wrapper = snapshot(status.get('wrapper_pid'))
        if wrapper and option_value(wrapper[1], '--stage-wrapper') == spec_path:
            found.append({'role': 'model_wrapper', 'fingerprint': wrapper[0]})
        child = snapshot(status.get('child_pid'))
        command = status.get('command')
        if child and command and canonical_command(child[1]) == canonical_command(command) and child[0]['executable'] == canonical_executable(command[0]):
            found.append({'role': 'model_stage_child', 'fingerprint': child[0]})
    return found


def verify_plan(path, expected_sha):
    if checksum(path) != expected_sha:
        raise ValueError('Inventory plan SHA256 changed')
    plan = strict_json(path)
    root = Path(plan['workspace']).resolve()
    if plan.get('families'):
        raise ValueError('Deferred inventory plan must contain no model families')
    for name, expected in plan.get('file_pins', {}).items():
        if checksum(root / name) != expected:
            raise ValueError('Pinned inventory source/input changed: ' + name)
    flags = plan['generation_command']
    if flags[flags.index('--workers') + 1] != '1':
        raise ValueError('Deferred inventory must use exactly one chemistry child')
    config = strict_json(root / flags[flags.index('--config') + 1])
    execution = config['execution_policy']
    if execution['compute_children'] != 1 or execution['nominal_rest_seconds'] < 60 or execution['sustained_nominal_seconds'] < 120:
        raise ValueError('Conservative generation cooling policy was weakened')
    return plan


def release_owned_stop(root, hold, permit):
    path = root / 'molecule_bank_1m/STOP_GENERATION'
    if not path.exists():
        return None
    if not permit:
        raise Waiting('waiting_for_STOP_GENERATION_release_authorization')
    record = next((item for item in hold.get('stop_files', []) if Path(item.get('path', '')).resolve() == path.resolve()), None)
    expected = 'Manual cooling hold: laptop reported hot on 2026-11-03. See training_runs/manual_cooling_hold.json.\n'
    if not record or record.get('preexisting') is not False or path.read_text() != expected:
        raise ValueError('STOP_GENERATION is not the exact stop file created by the recorded cooling hold')
    evidence = {'path': str(path), 'sha256': checksum(path), 'removed_at': now(),
                'authorization': 'user resumed research; launcher explicit --release-cooling-stop'}
    path.unlink()
    return evidence


def wait_and_run(args):
    plan = verify_plan(args.plan, args.plan_sha256)
    root = Path(plan['workspace']).resolve()
    os.chdir(root)
    out = root / plan['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'deferred_waiter.json'
    reader = ProcessReader()
    record = {'status': 'waiting', 'pid': os.getpid(), 'startup_token': args.startup_token,
              'plan': str(args.plan.resolve()), 'plan_sha256': args.plan_sha256,
              'model_progress': str(args.model_progress.resolve()),
              'ready_marker': str(args.ready_marker.resolve()), 'protocol_sha256': args.protocol_sha256,
              'created_at': now(), 'inventory_started': False}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with ExitStack() as waiter_stack:
        lock_nonblocking(waiter_stack, out / 'deferred_waiter.lock')
        atomic_json(status_path, record)
        nominal_since = None
        while not stopping:
            try:
                if checksum(args.plan) != args.plan_sha256:
                    raise ValueError('Inventory plan SHA256 changed while waiting')
                state = strict_json(args.model_progress)
                hold = load_json(root / 'training_runs/manual_cooling_hold.json')
                if hold.get('status') in {'paused', 'pause_intended'}:
                    raise Waiting('waiting_for_manual_cooling_hold_release')
                marker = strict_json(args.ready_marker) if args.ready_marker.exists() else None
                model_out = args.model_progress.resolve().parent
                check_ready(state, marker, args.protocol_sha256, model_out)
                if (out / 'STOP').exists():
                    raise Waiting('waiting_for_inventory_STOP_release')
                thermal = thermal_state()
                if thermal != 0:
                    nominal_since = None
                    raise Waiting('waiting_for_nominal_thermal_pressure')
                if nominal_since is None:
                    nominal_since = time.monotonic()
                if time.monotonic() - nominal_since < 120:
                    raise Waiting('waiting_for120s_sustained_nominal')
                with ExitStack() as model_locks:
                    # Retain these locks for the entire inventory supervisor lifetime.
                    queue_lock = lock_nonblocking(model_locks, model_out / 'queue.lock')
                    model_plan_path = root / 'large_model_search_plan.json'
                    model_plan = strict_json(model_plan_path)
                    for directory in existing_model_directories(model_plan, state):
                        for name in ('.trainer.lock', '.run.lock'):
                            lock_nonblocking(model_locks, directory / name)
                    # A check under the queue lock closes the normal supervisor race.
                    state = strict_json(args.model_progress)
                    marker = strict_json(args.ready_marker)
                    check_ready(state, marker, args.protocol_sha256, model_out)
                    live = verified_model_processes(state, reader)
                    if live:
                        raise Waiting('waiting_for_verified_model_process_exit')
                    bank = root / plan['bank_db']
                    with ExitStack() as checks:
                        lock_nonblocking(checks, bank.with_suffix('.generation.lock'))
                        lock_nonblocking(checks, bank.with_suffix('.chemistry.lock'))
                    verify_plan(args.plan, args.plan_sha256)
                    released = release_owned_stop(root, hold, args.release_cooling_stop)
                    # Root's training guard must acquire bank.chemistry.lock before
                    # starting a model, and the model queue is locked here.
                    token = uuid.uuid4().hex
                    command = [sys.executable, '-u', str(root / 'run_overnight_research.py'),
                               '--plan', str(args.plan.resolve()), '--startup-token', token]
                    with (out / 'supervisor.log').open('a') as log:
                        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(queue_lock.fileno(),),
                            env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
                                     OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    record.update(status='inventory_starting', inventory_pid=child.pid,
                                  inventory_startup_token=token, started_at=now(),
                                  released_generation_stop=released)
                    atomic_json(status_path, record)
                    deadline = time.monotonic() + 30
                    verified = False
                    while child.poll() is None and time.monotonic() < deadline:
                        inventory = load_json(out / 'progress.json')
                        if inventory.get('pid') == child.pid and inventory.get('startup_token') == token and inventory.get('plan_sha256') == args.plan_sha256:
                            verified = True
                            break
                        time.sleep(.2)
                    if not verified:
                        raise RuntimeError('Inventory supervisor startup not acknowledged; inspect supervisor.log')
                    record.update(status='inventory_running', inventory_started=True,
                                  startup_verified=True, updated_at=now())
                    atomic_json(status_path, record)
                    while child.poll() is None:
                        record.update(updated_at=now(), inventory_progress=load_json(out / 'progress.json').get('status'))
                        atomic_json(status_path, record)
                        # A stopped waiter keeps locks until the inventory drains.
                        if stopping:
                            (out / 'STOP').touch(exist_ok=True)
                            (root / 'molecule_bank_1m/STOP_GENERATION').touch(exist_ok=True)
                        time.sleep(5)
                    record.update(status='complete' if child.returncode == 0 else 'failed',
                                  returncode=child.returncode, finished_at=now())
                    atomic_json(status_path, record)
                    return child.returncode
            except Waiting as reason:
                if str(reason) not in {'waiting_for120s_sustained_nominal'} and not str(reason).startswith('waiting_for_free_lock:'):
                    nominal_since = None
                record.update(status='waiting', reason=str(reason), updated_at=now())
                atomic_json(status_path, record)
            except Exception as error:
                record.update(status='failed', error=repr(error), updated_at=now())
                atomic_json(status_path, record)
                raise
            deadline = time.monotonic() + args.poll_seconds
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(.2, max(0, deadline - time.monotonic())))
        record.update(status='stopped', updated_at=now())
        atomic_json(status_path, record)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--model-progress', type=Path, required=True)
    parser.add_argument('--ready-marker', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--release-cooling-stop', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--startup-token')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('Polling interval must be between1and60seconds')
    args.plan = args.plan.resolve()
    args.model_progress = args.model_progress.resolve()
    args.ready_marker = args.ready_marker.resolve()
    plan = verify_plan(args.plan, args.plan_sha256)
    out = Path(plan['workspace']) / plan['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        token = uuid.uuid4().hex
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--plan', str(args.plan),
                   '--plan-sha256', args.plan_sha256, '--model-progress', str(args.model_progress),
                   '--ready-marker', str(args.ready_marker), '--protocol-sha256', args.protocol_sha256,
                   '--poll-seconds', str(args.poll_seconds), '--startup-token', token]
        if args.release_cooling_stop:
            command.append('--release-cooling-stop')
        with (out / 'deferred_waiter.log').open('a') as log:
            process = subprocess.Popen(command, cwd=plan['workspace'], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = load_json(out / 'deferred_waiter.json')
            if state.get('pid') == process.pid and state.get('startup_token') == token:
                print(json.dumps({'startup_verified': True, 'pid': process.pid,
                                  'status': state.get('status'), 'progress': str(out / 'deferred_waiter.json')}))
                return
            if process.poll() is not None:
                raise RuntimeError('Deferred launcher failed; inspect deferred_waiter.log')
            time.sleep(.1)
        raise RuntimeError('Deferred launcher startup not acknowledged')
    args.startup_token = args.startup_token or uuid.uuid4().hex
    raise SystemExit(wait_and_run(args))


if __name__ == '__main__':
    main()
