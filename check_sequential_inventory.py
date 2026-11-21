#!/usr/bin/env python3
"""Small safety fixtures: no real inventory, database mutation, or process signals."""
from contextlib import ExitStack
import copy
import fcntl
import json
from pathlib import Path
import tempfile

from defer_sequential_inventory import (Waiting, check_ready, lock_nonblocking,
    release_owned_stop, verified_model_processes)
from make_sequential_inventory_plan import build


def raises(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError('Expected ' + kind.__name__)


def main():
    root = Path(__file__).resolve().parent
    config, plan = build(root)
    old = json.loads((root / 'million_bank_yield_config.json').read_text())
    before, after = copy.deepcopy(old), copy.deepcopy(config)
    before.pop('execution_policy'); after.pop('execution_policy')
    assert before == after
    assert config['execution_policy']['nominal_rest_seconds'] == 60
    assert config['execution_policy']['sustained_nominal_seconds'] == 120
    assert not plan['families']
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        model = fixture / 'models'
        marker = {'status': 'done', 'protocol_sha256': 'abc', 'model_out_dir': str(model)}
        check_ready({'status': 'validation_complete'}, marker, 'abc', model)
        raises(Waiting, lambda: check_ready({'status': 'running'}, marker, 'abc', model))
        raises(Waiting, lambda: check_ready({'status': 'validation_complete'}, None, 'abc', model))
        raises(RuntimeError, lambda: check_ready({'status': 'failed'}, marker, 'abc', model))
        raises(ValueError, lambda: check_ready({'status': 'validation_complete'}, marker, 'changed', model))
        raises(ValueError, lambda: check_ready({'status': 'validation_complete'}, marker, 'abc', fixture / 'other'))
        lock = fixture / 'queue.lock'
        with ExitStack() as first:
            lock_nonblocking(first, lock)
            with ExitStack() as second:
                raises(Waiting, lambda: lock_nonblocking(second, lock))
        with ExitStack() as second:
            lock_nonblocking(second, lock)
        stop = fixture / 'molecule_bank_1m/STOP_GENERATION'
        stop.parent.mkdir()
        expected = 'Manual cooling hold: laptop reported hot on 2026-11-03. See training_runs/manual_cooling_hold.json.\n'
        stop.write_text(expected)
        hold = {'stop_files': [{'path': str(stop), 'preexisting': False}]}
        raises(Waiting, lambda: release_owned_stop(fixture, hold, False))
        assert stop.exists()
        raises(ValueError, lambda: release_owned_stop(fixture, {'stop_files': []}, True))
        evidence = release_owned_stop(fixture, hold, True)
        assert evidence and not stop.exists()
        stop.write_text('New user stop')
        raises(ValueError, lambda: release_owned_stop(fixture, hold, True))
        assert stop.read_text() == 'New user stop'

        class Reader:
            def snapshot(self, pid):
                if pid == 101:
                    return {'pid': 101}, ['python', 'run_validation_continuation.py', '--startup-token', 'correct']
                if pid == 102:
                    return {'pid': 102}, ['python', 'unrelated.py', '--startup-token', 'different']
                raise ProcessLookupError(pid)

        assert len(verified_model_processes({'pid': 101, 'startup_token': 'correct'}, Reader())) == 1
        assert not verified_model_processes({'pid': 102, 'startup_token': 'correct'}, Reader())
        assert not verified_model_processes({'pid': 999, 'startup_token': 'correct'}, Reader())
    print('PASS: identical chemistry; strict readiness/digest/model identity; retained lock conflict; exact owned STOP removal only')


if __name__ == '__main__':
    main()
