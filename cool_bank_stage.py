#!/usr/bin/env python3
"""Run one non-generation bank stage with a conservative whole-group duty cycle.

The child starts in a new session/process group. Only its birth-verified group
is signalled. Subprocess workers share that group, so audit workers also pause.
A chemistry lease excludes training and generation for this stage's lifetime.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_overnight_research import atomic_json, now
from thermal_guard import (DutyPolicy, ProcessReader, ProcBSDInfo, canonical_command,
                           canonical_executable, same_process, script_argument, thermal_state)


ALLOWED = frozenset({'import_legacy_bank_relations.py', 'build_bank_scenarios.py',
                     'audit_export_million_bank.py', 'bank_graph_shards.py'})


def inner_command(command):
    command = list(command)
    if command[:3] == ['/usr/sbin/taskpolicy', '-c', 'utility']:
        command = command[3:]
    return command


def validate_command(command, workspace):
    command = inner_command(command)
    script = script_argument(command)
    if not command or script is None or Path(script).name not in ALLOWED:
        raise ValueError('Only non-generation bank import/scenario/audit/graph scripts are allowed')
    if (workspace / script).resolve() != workspace / Path(script).name:
        raise ValueError('Bank stage script must be inside the workspace')
    if canonical_executable(command[0]) != canonical_executable(workspace / 'venv/bin/python'):
        raise ValueError('Bank stage must use this workspace Python')
    if '--workers' in command and command[command.index('--workers') + 1] != '1':
        raise ValueError('Bank stage permits at most one worker')
    if any(token.startswith('--workers=') and token != '--workers=1' for token in command):
        raise ValueError('Bank stage permits at most one worker')
    return command


class OwnedGroup:
    def __init__(self, child, expected, reader):
        self.child, self.expected, self.reader = child, expected, reader
        self.leader = None
        self.members = {}
        self.paused = False

    def info(self, pid):
        info = ProcBSDInfo()
        count = self.reader.proc.proc_pidinfo(int(pid), 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        if count != ctypes.sizeof(info):
            raise ProcessLookupError(pid)
        return info

    def pin(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self.child.poll() is None and time.monotonic() < deadline:
            try:
                fingerprint, argv = self.reader.snapshot(self.child.pid)
                info = self.info(self.child.pid)
                if fingerprint['uid'] != os.getuid() or fingerprint['ppid'] != os.getpid() or info.pgid != self.child.pid:
                    raise ValueError('Spawned bank stage owner/parent/group mismatch')
                if (canonical_command(argv) == canonical_command(self.expected)
                        and fingerprint['executable'] == canonical_executable(self.expected[0])):
                    self.leader = fingerprint
                    self.members[str(fingerprint['pid'])] = fingerprint
                    return True
            except ProcessLookupError:
                if self.child.poll() is not None:
                    return False
            time.sleep(.01)
        if self.child.poll() is not None:
            return False
        raise ValueError('Bank stage did not exec the exact expected command')

    def current_leader(self):
        if self.leader is None:
            raise ValueError('Bank stage has no verified leader identity')
        info = self.info(self.leader['pid'])
        if info.status == 5:  # SZOMB: Popen.poll will reap this completed child.
            raise ProcessLookupError(self.leader['pid'])
        current, _ = self.reader.snapshot(self.leader['pid'])
        if not same_process(current, self.leader, parent=True) or info.pgid != self.child.pid:
            raise ValueError('Owned bank process group leader identity changed')
        return current

    def cache_group(self):
        self.current_leader()
        proc = self.reader.proc
        proc.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        proc.proc_listpids.restype = ctypes.c_int
        size = proc.proc_listpids(1, 0, None, 0)
        if size <= 0 or size > 64 * 1024 * 1024:
            raise RuntimeError('Cannot inventory owned process group')
        values = (ctypes.c_int * (size // ctypes.sizeof(ctypes.c_int) + 1024))()
        count = proc.proc_listpids(1, 0, values, ctypes.sizeof(values))
        if count <= 0:
            raise RuntimeError('Cannot inventory owned process group')
        for pid in values[:count // ctypes.sizeof(ctypes.c_int)]:
            if pid <= 0:
                continue
            try:
                info = self.info(pid)
                if info.pgid != self.child.pid or info.uid != os.getuid() or info.status == 5:
                    continue
                fingerprint, _ = self.reader.snapshot(pid)
                if fingerprint['start'] < self.leader['start']:
                    raise ValueError('Group member predates owned leader')
                previous = self.members.get(str(pid))
                if previous and not same_process(previous, fingerprint):
                    raise ValueError('Owned process group member identity changed')
                self.members[str(pid)] = fingerprint
            except ProcessLookupError:
                continue

    def send(self, sig):
        self.cache_group()
        self.current_leader()  # Verify birth/group again immediately before killpg.
        os.killpg(self.child.pid, sig)
        if sig == signal.SIGSTOP:
            self.paused = True
        elif sig == signal.SIGCONT:
            self.paused = False

    def cleanup_members(self, sig):
        """Fallback for cached descendants if the group leader exited first."""
        sent = []
        for original in list(self.members.values()):
            try:
                info = self.info(original['pid'])
                if info.status == 5:
                    continue
                current, _ = self.reader.snapshot(original['pid'])
                if (same_process(current, original) and info.pgid == self.child.pid
                        and current['ppid'] in {original['ppid'], 1}):
                    os.kill(current['pid'], sig)
                    sent.append(current['pid'])
            except ProcessLookupError:
                pass
        return sent


def run_stage(command, state_path, lease_path, *, policy=None, poll=2, thermal_reader=thermal_state,
              reader=None, expected=None, stop_after=None):
    reader = reader or ProcessReader()
    policy = policy or DutyPolicy(compute=15, rest=45, fair_compute=5, fair_rest=55, recovery=120)
    expected = list(expected or inner_command(command))
    record = {'status': 'starting', 'wrapper_pid': os.getpid(), 'started_at': now(),
              'command': command, 'nominal_compute_seconds': policy.compute,
              'nominal_rest_seconds': policy.rest, 'fair_compute_seconds': policy.fair_compute,
              'fair_rest_seconds': policy.fair_rest, 'sustained_nominal_recovery_seconds': policy.recovery,
              'temperature_celsius_available': False, 'events': []}
    stopped = False
    child = group = None
    started = time.monotonic()

    def request_stop(signum, _frame):
        nonlocal stopped
        stopped = True
        record['requested_signal'] = int(signum)

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}

    def save():
        record['updated_at'] = now()
        if group is not None:
            record.update(child_fingerprint=group.leader, owned_group_members=list(group.members.values()),
                          process_group_paused=group.paused)
        atomic_json(state_path, record)

    def send(sig, reason):
        record['signal_intent'] = {'signal': signal.Signals(sig).name, 'reason': reason}
        save()  # Durable intent precedes every group signal.
        try:
            group.send(sig)
        except ProcessLookupError:
            if child.poll() is None:
                raise
            return
        record['events'].append({'signal': signal.Signals(sig).name, 'reason': reason, 'time': now()})
        record.pop('signal_intent', None)
        save()

    lease_path = Path(lease_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lease_path.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Child inherits this lease, preserving exclusion if this wrapper dies.
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True,
                pass_fds=(lease.fileno(),), env=dict(os.environ, OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            group = OwnedGroup(child, expected, reader)
            record.update(child_pid=child.pid, process_group=child.pid)
            save()
            if not group.pin():
                record.update(status='complete' if child.returncode == 0 else 'failed',
                              returncode=child.returncode, finished_at=now())
                save()
                return child.returncode
            group.cache_group()
            save()
            try:
                while child.poll() is None:
                    if stop_after is not None and time.monotonic() - started >= stop_after:
                        stopped = True
                    if stopped:
                        if group.paused:
                            send(signal.SIGCONT, 'controlled_termination_unblocks_owned_group')
                        send(signal.SIGTERM, 'controlled_wrapper_stop')
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            send(signal.SIGKILL, 'owned_group_termination_timeout')
                            child.wait(timeout=5)
                        break
                    thermal = thermal_reader()
                    pause, reason = policy.decision(thermal, time.monotonic())
                    record.update(status='cooling' if pause else 'compute_window', thermal_state=thermal,
                                  decision=reason)
                    if pause and not group.paused:
                        send(signal.SIGSTOP, reason)
                    elif not pause and group.paused:
                        send(signal.SIGCONT, reason)
                    save()
                    deadline = time.monotonic() + poll
                    while not stopped and time.monotonic() < deadline:
                        time.sleep(min(.05, max(0, deadline - time.monotonic())))
                record.update(status='stopped' if stopped else 'complete' if child.returncode == 0 else 'failed',
                              returncode=child.returncode, finished_at=now())
                save()
                return child.returncode
            finally:
                # On exceptions, continue only the identity-verified owned group
                # so it can receive TERM. Never leave an unguarded live job behind.
                if child.poll() is None:
                    if group.paused:
                        send(signal.SIGCONT, 'exception_cleanup_owned_group')
                    send(signal.SIGTERM, 'exception_cleanup_owned_group')
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        send(signal.SIGKILL, 'exception_cleanup_timeout')
                        child.wait(timeout=5)
                # Cached worker identities are safe to clean after a failed main
                # process; do not signal an arbitrary reused process-group ID.
                if child.returncode != 0:
                    group.cleanup_members(signal.SIGCONT)
                    group.cleanup_members(signal.SIGTERM)
    except BaseException as error:
        # Startup verification can fail before the main loop's cleanup becomes
        # active. A fresh same-user direct-child/group identity still proves
        # this Popen-owned process, even while taskpolicy is execing Python.
        if child is not None and child.poll() is None and group is not None:
            try:
                if group.leader is None:
                    fingerprint, _ = reader.snapshot(child.pid)
                    info = group.info(child.pid)
                    if fingerprint['uid'] != os.getuid() or fingerprint['ppid'] != os.getpid() or info.pgid != child.pid:
                        raise ValueError('Cannot verify failed-start child for cleanup')
                    group.leader = fingerprint
                if group.paused:
                    send(signal.SIGCONT, 'startup_or_cleanup_failure')
                send(signal.SIGTERM, 'startup_or_cleanup_failure')
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    send(signal.SIGKILL, 'startup_cleanup_timeout')
                    child.wait(timeout=5)
            except BaseException as cleanup_error:
                record['cleanup_error'] = repr(cleanup_error)
        record.update(status='failed', error=repr(error), finished_at=now())
        save()
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--chemistry-lock', type=Path, default=Path('molecule_bank_1m/bank.chemistry.lock'))
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--poll-seconds', type=float, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 2:
        parser.error('Polling must be positive and at most2seconds')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    workspace = args.workspace.resolve()
    expected = validate_command(command, workspace)
    os.chdir(workspace)
    raise SystemExit(run_stage(command, args.state, args.chemistry_lock, poll=args.poll_seconds, expected=expected))


if __name__ == '__main__':
    main()
