"""Small policy/ownership checks without touching live processes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import thermal_guard as engine
from thermal_guard_cool import CoolingGuard, terminal_queue


class CoolingPolicyChecks(unittest.TestCase):
    def test_compute_and_pressure_recovery(self):
        policy = engine.DutyPolicy(compute=15, rest=45, fair_compute=5, fair_rest=55, recovery=120)
        self.assertFalse(policy.decision(0, 0)[0])
        self.assertTrue(policy.decision(0, 15)[0])
        self.assertTrue(policy.decision(0, 59.99)[0])
        self.assertFalse(policy.decision(0, 60)[0])
        self.assertTrue(policy.decision(2, 61)[0])
        self.assertTrue(policy.decision(0, 62)[0])
        self.assertTrue(policy.decision(0, 181.9)[0])
        self.assertFalse(policy.decision(0, 182)[0])

    def test_shutdown_does_not_resume_live_compute(self):
        guard = object.__new__(CoolingGuard)
        guard.progress = [Path('fixture-progress')]
        guard.workspace = Path.cwd()
        guard.reader = object()
        guard.state = {'paused': {}}
        calls = []
        guard.pause = lambda record, reason: (calls.append(reason), guard.state['paused'].update({'fixture': record}))
        guard.event = lambda *args: None
        with patch('thermal_guard.owned_training', return_value={'fingerprint': {'pid': 123}}), patch.object(engine.Guard, 'resume_all') as resume:
            guard.resume_all('guard_exit_recovery')
            resume.assert_not_called()
        self.assertEqual(calls, ['guard_exit_holds_training_for_cooling'])

    def test_only_terminal_states_end_guard(self):
        for status in ['running', 'starting', None]:
            with patch('thermal_guard.load_json', return_value={'status': status}):
                self.assertFalse(terminal_queue('fixture'))
        for status in ['validation_complete', 'failed', 'stopped', 'complete']:
            with patch('thermal_guard.load_json', return_value={'status': status}):
                self.assertTrue(terminal_queue('fixture'))


if __name__ == '__main__':
    unittest.main()
