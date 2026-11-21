#!/usr/bin/env python3
"""Small real owned-child cooling and termination fixtures; no production jobs."""
import ctypes
import fcntl
import json
from pathlib import Path
import signal
import sys
import tempfile

from cool_bank_stage import run_stage, validate_command
from thermal_guard import DutyPolicy, ProcessReader, ProcBSDInfo


def main():
    # Pressure and recovery behavior, without sleeping120seconds.
    policy = DutyPolicy(compute=15, rest=45, fair_compute=5, fair_rest=55, recovery=120)
    assert policy.decision(2, 0)[0]
    assert policy.decision(0, 1)[0]
    assert policy.decision(0, 120)[0]
    assert not policy.decision(0, 121)[0]
    assert policy.decision(0, 136)[0]
    assert policy.decision(None, 200)[0]
    assert not policy.decision(None, 255)[0]
    root = Path(__file__).resolve().parent
    try:
        validate_command([str(root/'venv/bin/python'), 'expand_yield_aware_bank.py'], root)
    except ValueError:
        pass
    else:
        raise AssertionError('Generation was allowed inside downstream cooling wrapper')
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        lock = fixture/'chemistry.lock'
        command = [sys.executable, '-c',
            'import subprocess,sys,time; child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(0.5)"]); child.wait()']
        status = fixture/'normal.json'
        seen_paused_worker = []
        reader = ProcessReader()
        def nominal():
            if status.exists():
                current = json.loads(status.read_text())
                members = current.get('owned_group_members', [])
                if current.get('process_group_paused') and len(members) >= 2:
                    states = []
                    for member in members:
                        info = ProcBSDInfo()
                        size = reader.proc.proc_pidinfo(member['pid'], 3, 0, ctypes.byref(info), ctypes.sizeof(info))
                        if size == ctypes.sizeof(info):
                            states.append(info.status)
                    if len(states) >= 2 and all(state == 4 for state in states):
                        seen_paused_worker.append(True)
            return 0
        result = run_stage(command, status, lock,
            policy=DutyPolicy(compute=.05, rest=.08, fair_compute=.02, fair_rest=.10, recovery=.1),
            poll=.01, thermal_reader=nominal)
        assert result == 0
        record = json.loads(status.read_text())
        signals = [event['signal'] for event in record['events']]
        assert 'SIGSTOP' in signals and 'SIGCONT' in signals
        assert record['child_fingerprint']['pid'] == record['process_group']
        assert record['status'] == 'complete'
        assert seen_paused_worker, 'The parent and its worker were not observed paused together'
        # Request termination while the owned process group is paused.
        status = fixture/'terminated.json'
        result = run_stage([sys.executable, '-c', 'import time; time.sleep(30)'], status, lock,
            policy=DutyPolicy(compute=.03, rest=.5, fair_compute=.01, fair_rest=.5, recovery=.1),
            poll=.01, thermal_reader=lambda: 0, stop_after=.12)
        assert result == -signal.SIGTERM
        record = json.loads(status.read_text())
        events = record['events']
        stop_index = next(i for i, event in enumerate(events) if event['signal'] == 'SIGSTOP')
        assert any(event['signal'] == 'SIGCONT' for event in events[stop_index+1:])
        assert record['status'] == 'stopped' and not record['process_group_paused']
        # Both terminal paths release their lease after reaping the child.
        with lock.open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
    print('PASS: pressure/recovery mocks; real birth-verified group STOP/CONT; paused-child TERM cleanup; lease release; generation excluded')


if __name__ == '__main__':
    main()
