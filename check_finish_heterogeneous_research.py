#!/usr/bin/env python3
"""Verify finalization order on tiny synthetic fixtures, without real inference."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from check_heterogeneous_ensemble import EnsembleChecks
import heterogeneous_validation_ensemble as selector
import evaluate_heterogeneous_ensemble as evaluator
import finish_heterogeneous_research as bridge


class FinalizationChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = EnsembleChecks('test_fixed_weights_and_ties_are_deterministic')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.protocol = self.root / 'protocol.json'
        self.fixture.write(self.protocol, {'fixture': 'fixed continuation'})
        self.protocol_hash = selector.legacy.sha256(self.protocol)
        self.fixture.write(self.fixture.search / 'progress.json', {'status': 'validation_complete',
                          'continuation_protocol': {'sha256': self.protocol_hash}})
        self.marker = self.root / 'inventory_ready.json'
        self.manifest = self.root / 'followup.json'
        self.fixture.write(self.manifest, {'schema_version': 1,
             'final_test_after_frozen_validation_selection': True,
             'workspace': str(self.root), 'continuation_protocol': str(self.protocol),
             'continuation_protocol_sha256': self.protocol_hash,
             'source_pins': {str(self.fixture.policy): selector.legacy.sha256(self.fixture.policy),
                             str(Path(bridge.__file__).resolve()): selector.legacy.sha256(bridge.__file__)},
             'model_out_dir': str(self.fixture.search), 'ensemble_out_dir': 'ensemble',
             'ensemble_policy': str(self.fixture.policy),
             'ensemble_policy_sha256': selector.legacy.sha256(self.fixture.policy),
             'cohort_dir': str(self.fixture.cohort), 'test_lock': 'test_lock.json',
             'inventory_ready_marker': str(self.marker)})
        self.manifest_hash = selector.legacy.sha256(self.manifest)

    def finalize(self):
        with patch.object(selector.legacy, 'checkpoint_details', side_effect=self.fixture.checkpoint):
            return bridge.finalize(self.manifest, self.manifest_hash)

    def test_missing_member_never_calls_test_or_publishes_ready(self):
        path = self.fixture.search / 'b' / 'summary.json'
        value = selector.load(path)
        value['completed'] = False
        self.fixture.write(path, value)
        with patch.object(evaluator, 'evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, 'not completed'):
                self.finalize()
            evaluate.assert_not_called()
        self.assertFalse(self.marker.exists())
        self.assertEqual(selector.load(self.root / 'ensemble/pipeline_progress.json')['status'], 'failed')

    def test_evaluation_failure_does_not_release_inventory(self):
        def fail(selection_path, selection_hash, *args, **kwargs):
            self.assertTrue(selection_path.is_file())
            self.assertEqual(selector.legacy.sha256(selection_path), selection_hash)
            self.assertEqual(selector.load(selection_path)['mode'], 'final')
            raise RuntimeError('isolated mocked inference failed')
        with patch.object(evaluator, 'evaluate', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'mocked inference failed'):
                self.finalize()
        self.assertFalse(self.marker.exists())
        self.assertEqual(selector.load(self.root / 'ensemble/pipeline_progress.json')['status'], 'failed')

    def test_success_releases_inventory_only_after_result_is_published(self):
        def complete(selection_path, selection_hash, test_lock, out, **kwargs):
            self.assertFalse(self.marker.exists())
            self.assertTrue(selection_path.is_file())
            selection = selector.load(selection_path)
            self.assertEqual(selection['winner'], 'mean')
            self.assertEqual(selection_hash, selector.legacy.sha256(selection_path))
            out.mkdir(parents=True)
            answer = {'selection_sha256': selection_hash, 'winner': selection['winner'],
                      'test_metrics': {'f1': 1.0}}
            self.fixture.write(out / 'test_result.json', answer)
            return answer
        with patch.object(evaluator, 'evaluate', side_effect=complete):
            result = self.finalize()
        marker = selector.load(self.marker)
        self.assertEqual(marker['status'], 'done')
        self.assertEqual(marker['selection_sha256'], result['selection_sha256'])
        self.assertEqual(marker['result_sha256'], selector.legacy.sha256(marker['result_path']))

    def test_incomplete_validation_state_rejected_before_any_selection(self):
        self.fixture.write(self.fixture.search / 'progress.json', {'status': 'running'})
        with patch.object(evaluator, 'evaluate') as evaluate, patch.object(selector, 'build_spec') as build:
            with self.assertRaisesRegex(ValueError, 'finish first'):
                self.finalize()
            evaluate.assert_not_called()
            build.assert_not_called()
        self.assertFalse(self.marker.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
