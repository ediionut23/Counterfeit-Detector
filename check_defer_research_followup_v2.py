"""Check the known guard-exit and restart races without running any models."""
import tempfile
import unittest
from pathlib import Path

from defer_research_followup_v2 import startup_acknowledged, unexpected_guard_exit, completed_followup
from run_overnight_research import atomic_json, checksum


class FollowupLifecycleChecks(unittest.TestCase):
    def test_clean_guard_exit_after_complete_does_not_trigger_hold(self):
        self.assertFalse(unexpected_guard_exit(0, {'status': 'complete'}))
        self.assertFalse(unexpected_guard_exit(None, {'status': 'running'}))
        self.assertTrue(unexpected_guard_exit(0, {'status': 'running'}))
        self.assertTrue(unexpected_guard_exit(1, {'status': 'complete'}))

    def test_startup_ack_requires_process_nonce_and_plan(self):
        state = {'pid': 1, 'startup_token': 'nonce', 'plan_sha256': 'pin', 'status': 'running'}
        self.assertTrue(startup_acknowledged(state, 1, 'nonce', 'pin'))
        self.assertFalse(startup_acknowledged(state, 2, 'nonce', 'pin'))
        self.assertFalse(startup_acknowledged(state, 1, 'stale', 'pin'))
        self.assertFalse(startup_acknowledged(state, 1, 'nonce', 'changed'))

    def test_completed_restart_requires_pinned_result_and_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = root / 'result.json'
            atomic_json(result, {'fixed': True})
            marker_path = root / 'ready.json'
            manifest_path = root / 'manifest.json'
            manifest = {'inventory_ready_marker': str(marker_path), 'continuation_protocol_sha256': 'protocol'}
            atomic_json(manifest_path, manifest)
            manifest_hash = checksum(manifest_path)
            plan = {'stages': [{'command': ['python', 'bridge', '--manifest', str(manifest_path), '--manifest-sha256', manifest_hash]}]}
            atomic_json(root / 'progress.json', {'status': 'complete', 'plan_sha256': 'plan'})
            with self.assertRaises(ValueError):
                completed_followup(root, plan, 'plan')
            marker = {'status': 'done', 'followup_manifest_sha256': manifest_hash,
                      'protocol_sha256': 'protocol', 'result_path': str(result), 'result_sha256': checksum(result)}
            atomic_json(marker_path, marker)
            self.assertTrue(completed_followup(root, plan, 'plan'))
            atomic_json(result, {'tampered': True})
            with self.assertRaises(ValueError):
                completed_followup(root, plan, 'plan')


if __name__ == '__main__':
    unittest.main()
