#!/usr/bin/env python3
"""Ownership/orphan/lease fixtures; never signals a real process."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import thermal_guard as engine
import thermal_guard_cool_v2 as guard_module


class Reader:
    def __init__(self, records):
        self.records = records

    def snapshot(self, pid):
        if pid not in self.records:
            raise ProcessLookupError(pid)
        return copy.deepcopy(self.records[pid])


class GuardChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='guard_v2_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        stages = self.root / 'queue/stages'
        stages.mkdir(parents=True)
        self.progress = self.root / 'queue/progress.json'
        self.status_path, self.spec_path, self.lease = stages / 'fixture.status.json', stages / 'fixture.spec.json', stages / 'fixture.lock'
        self.lease.touch()
        self.command = [sys.executable, 'train_geometric_variants.py', '--no-test']
        self.qargv = [sys.executable, '-u', str(self.root / 'run_validation_continuation.py'), '--startup-token', 'queue-token']
        self.wargv = [sys.executable, '-u', str(self.root / 'run_overnight_research.py'), '--stage-wrapper', str(self.spec_path), '--stage-lease-fd', '4']
        def fingerprint(pid, ppid, argv):
            return {'pid': pid, 'ppid': ppid, 'uid': os.getuid(), 'start': [1000 + pid, 123],
                'executable': engine.canonical_executable(sys.executable), 'argv_sha256': engine.digest(argv)}
        self.reader = Reader({10: (fingerprint(10, 1, self.qargv), list(self.qargv)),
            20: (fingerprint(20, 10, self.wargv), list(self.wargv)),
            30: (fingerprint(30, 20, self.command), list(self.command))})
        self.stage = {'status': 'running', 'pid': 20, 'wrapper_token': 'wrapper-token',
            'command_sha256': 'frozen-command', 'command': self.command,
            'stage_status_path': str(self.status_path), 'stage_lock_path': str(self.lease)}
        self.queue = {'status': 'failed', 'pid': 10, 'startup_token': 'queue-token',
            'current_stage': 'fixture', 'stages': {'fixture': self.stage}}
        self.status = {'status': 'running', 'wrapper_pid': 20, 'child_pid': 30,
            'wrapper_token': 'wrapper-token', 'command_sha256': 'frozen-command', 'command': self.command}
        self.spec = {'wrapper_token': 'wrapper-token', 'command_sha256': 'frozen-command',
            'command': self.command, 'workspace': str(self.root), 'lock_path': str(self.lease), 'status_path': str(self.status_path)}
        self.write()

    def write(self):
        for path, value in [(self.progress, self.queue), (self.status_path, self.status), (self.spec_path, self.spec)]:
            path.write_text(json.dumps(value))

    def verify(self, **kwargs):
        with patch.object(engine, 'lease_held', return_value=True):
            return guard_module.verify_leased_training(self.progress, self.root, self.reader, **kwargs)

    def fake_guard(self, record):
        guard = object.__new__(guard_module.CoolingGuard)
        guard.progress, guard.workspace, guard.reader = [self.progress], self.root, self.reader
        guard.allowed_scripts = guard_module.DEFAULT_STAGE_SCRIPTS
        guard.cache = {'30': record}
        guard.state = {'paused': {}, 'last_verified_targets': [record]}
        guard.save, guard.event, guard.lock = Mock(), Mock(), Mock()
        guard.stopping, guard.poll = False, .01
        guard.thermal_reader = lambda: 0
        guard.policy = engine.DutyPolicy(compute=15, rest=45, fair_compute=5, fair_rest=55, recovery=120)
        return guard

    def test_failed_queue_live_stage_is_owned_but_never_resumed(self):
        record = self.verify()
        self.assertTrue(record['orphan'])
        guard = self.fake_guard(record)
        guard.state['paused']['30'] = record
        spoofed_healthy = {**record, 'orphan': False}
        with patch.object(engine, 'lease_held', return_value=True), patch.object(guard_module.os, 'kill') as kill:
            guard.resume_verified([spoofed_healthy])
            kill.assert_not_called()
            self.assertFalse(guard.queues_finished())
        self.assertIn('30', guard.state['paused'])

    def test_dead_supervisor_reparented_wrapper_is_held(self):
        self.reader.records.pop(10)
        self.reader.records[20][0]['ppid'] = 1
        self.queue['status'] = 'running'
        self.write()
        self.assertTrue(self.verify()['orphan'])

    def test_dead_wrapper_requires_cached_exact_child_birth(self):
        prior = self.verify()
        self.reader.records.pop(20)
        self.reader.records[30][0]['ppid'] = 1
        with self.assertRaises(ValueError):
            self.verify()
        self.assertTrue(self.verify(prior_records=[prior])['orphan'])
        self.reader.records[30][0]['start'][0] += 1
        with self.assertRaises(ValueError):
            self.verify(prior_records=[prior])

    def test_wrong_wrapper_command_token_and_argv_are_rejected(self):
        self.reader.records[20][1][-1] = 'unrelated'
        with self.assertRaises(ValueError):
            self.verify()
        self.reader.records[20] = (self.reader.records[20][0], self.wargv)
        self.status['wrapper_token'] = 'wrong-token'
        self.write()
        with self.assertRaises(ValueError):
            self.verify()
        self.status['wrapper_token'] = 'wrapper-token'
        self.write()
        self.reader.records[30][1].append('--different-command')
        with self.assertRaises(ValueError):
            self.verify()

    def test_missing_live_lease_authorizes_no_target(self):
        with patch.object(engine, 'lease_held', return_value=False):
            self.assertIsNone(guard_module.verify_leased_training(self.progress, self.root, self.reader))

    def test_healthy_queue_and_explicit_extra_stage(self):
        self.queue['status'] = 'running'
        self.write()
        self.assertFalse(self.verify()['orphan'])
        self.queue['status'] = 'failed'
        self.command[1] = 'finish_heterogeneous_research.py'
        self.reader.records[30] = ({**self.reader.records[30][0], 'argv_sha256': engine.digest(self.command)}, self.command)
        self.write()
        with self.assertRaises(ValueError):
            self.verify()
        self.assertTrue(self.verify(allowed_scripts={'finish_heterogeneous_research.py'})['orphan'])

    def test_orphan_loop_and_exit_only_issue_stop_signals(self):
        record = self.verify()
        guard = self.fake_guard(record)
        def end_poll(_):
            guard.stopping = True
        with patch.object(engine, 'lease_held', return_value=True), \
             patch.object(guard_module.os, 'kill') as kill, \
             patch.object(guard_module.signal, 'signal', return_value=signal.SIG_DFL), \
             patch.object(guard_module.time, 'sleep', side_effect=end_poll):
            guard.run()
        self.assertTrue(kill.called)
        self.assertTrue(all(call.args == (30, signal.SIGSTOP) for call in kill.call_args_list))
        self.assertEqual(guard.state['status'], 'stopped_training_held')
        self.assertIn('30', guard.state['paused'])

    def test_no_live_stage_and_completed_queue_may_finish(self):
        self.queue.update(status='validation_complete', current_stage=None)
        self.queue['stages']['fixture']['status'] = 'complete'
        self.write()
        guard = object.__new__(guard_module.CoolingGuard)
        guard.progress = [self.progress]
        with patch.object(engine, 'lease_held', return_value=False):
            self.assertTrue(guard.queues_finished())
        with patch.object(engine, 'lease_held', return_value=True):
            self.assertFalse(guard.queues_finished())

    def test_chemistry_lease_blocks_guard_start_before_process_reader(self):
        import fcntl
        lock = self.root / 'chemistry.lock'
        argv = ['thermal_guard_cool_v2.py', '--progress', str(self.progress), '--workspace', str(self.root), '--chemistry-lock', str(lock)]
        with lock.open('a+') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(sys, 'argv', argv), patch.object(guard_module, 'CoolingGuard') as guard:
                with self.assertRaises(BlockingIOError):
                    guard_module.main()
                guard.assert_not_called()

    def test_existing_pause_is_reasserted(self):
        record = self.verify()
        guard = self.fake_guard(record)
        guard.state['paused']['30'] = record
        with patch.object(guard_module.os, 'kill') as kill:
            guard.pause(record, 'fixture-rest')
        kill.assert_called_once_with(30, signal.SIGSTOP)

    def test_child_finishing_during_pause_does_not_kill_guard(self):
        record = self.verify()
        guard = self.fake_guard(record)
        self.reader.records.pop(30)
        with patch.object(guard_module.os, 'kill') as kill:
            guard.pause(record, 'completion-race')
        kill.assert_not_called()
        self.assertNotIn('30', guard.cache)
        self.assertNotIn('30', guard.state['paused'])


if __name__ == '__main__':
    unittest.main()
