#!/usr/bin/env python3
"""Small synthetic fixtures for immutable heterogeneous selection and locks.

No real molecular graphs, model forwards, or production test data are used.
"""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import heterogeneous_validation_ensemble as selector
import evaluate_heterogeneous_ensemble as evaluator


class EnsembleChecks(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='heterogeneous-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.search = self.root / 'runs'
        self.cohort = self.root / 'cohort'
        self.cohort.mkdir()
        self.ids = ['a', 'b', 'c', 'd']
        self.labels = [0, 1, 0, 1]
        self.values = {'a': [.1, .9, .8, .4], 'b': [.7, .4, .1, .9]}
        dataset = self.cohort / 'dataset.csv'
        with dataset.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_id', 'molecule_id', 'split', 'label'])
            for split in ('val', 'test'):
                for name, label in zip(self.ids, self.labels):
                    writer.writerow([name, name, split, label])
        index = self.cohort / 'graph_index.csv'
        index.write_text('fixture only; not opened for inference\n')
        self.write(self.cohort / 'manifest.json', {'cohort_sha256': 'fixture-cohort',
                   'split_label_counts': {'val': {'0': 2, '1': 2}},
                   'selected_files': {p.name: {'sha256': selector.legacy.sha256(p)} for p in (dataset, index)}})
        self.trainer = self.root / 'mock_trainer.py'
        self.trainer.write_text('# isolated fake trainer; no real model code\n')
        definitions = []
        self.checkpoints = {}
        for name, seed in [('a', 42), ('b', 43)]:
            folder = self.search / name
            folder.mkdir(parents=True)
            (folder / 'best.pt').write_text('fake checkpoint ' + name)
            probability = np.asarray(self.values[name])
            report = selector.legacy.metrics(self.labels, probability, selector.legacy.choose_threshold(self.labels, probability))
            self.write(folder / 'summary.json', {'completed': True, 'test_evaluated': False,
                       'cohort_sha256': 'fixture-cohort', 'best_epoch': 1, 'best_validation_f1': report['f1']})
            with (folder / 'val_probabilities.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['source_id', 'split', 'label', 'probability_generated', 'epoch'])
                for identity, label, score in zip(self.ids, self.labels, probability):
                    writer.writerow([identity, 'val', label, score, 1])
            definitions.append({'id': name, 'run_dir': name, 'family': 'gatv2', 'seed': seed,
                                'trainer': self.trainer.name, 'inference_adapter': self.trainer.name})
            self.checkpoints[str(folder / 'best.pt')] = {'epoch': 1, 'best_epoch': 1,
                  'config': {'seed': seed}, 'validation_metrics': report,
                  'provenance': {'trainer_sha256': selector.legacy.sha256(self.trainer)}}
        self.policy = self.root / 'policy.json'
        self.write(self.policy, {'schema_version': 1, 'policy_id': 'fixture', 'selection_uses': 'validation_only',
                    'aggregation': selector.AGGREGATION, 'all_members_required_for_final': True,
                    'cohort_sha256': 'fixture-cohort', 'members': definitions,
                    'candidates': [{'id': 'single_a', 'weights': {'a': '1'}},
                                   {'id': 'single_b', 'weights': {'b': '1'}},
                                   {'id': 'mean', 'weights': {'a': '1/2', 'b': '1/2'}}]})

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')

    def checkpoint(self, path):
        value = self.checkpoints[str(path)]
        return value, value['config'], 'fixture-cohort', selector.legacy.sha256(path)

    def spec(self, mode='final', name='spec.json'):
        path = self.root / name
        selector.build_spec(self.policy, selector.legacy.sha256(self.policy), self.root, self.search,
                            self.cohort, mode, path)
        return path

    def selected(self, mode='final', name='selection'):
        spec = self.spec(mode, name + '_spec.json')
        out = self.root / name
        with patch.object(selector.legacy, 'checkpoint_details', side_effect=self.checkpoint):
            result = selector.select(spec, selector.legacy.sha256(spec), out)
        return out, result

    def test_complete_alignment_is_order_invariant_and_rejects_bad_ids_labels(self):
        expected = dict(zip(self.ids, self.labels))
        np.testing.assert_array_equal(selector.align_prediction(['d', 'a', 'c', 'b'], [1, 0, 0, 1], [.4, .1, .8, .9], expected), self.values['a'])
        for ids, labels, scores in [(['a', 'a', 'c', 'd'], self.labels, self.values['a']),
                                   (self.ids[:3], self.labels[:3], self.values['a'][:3]),
                                   (self.ids, [1, 1, 0, 1], self.values['a']),
                                   (self.ids, self.labels, [float('nan'), .9, .8, .4])]:
            with self.assertRaises(ValueError):
                selector.align_prediction(ids, labels, scores, expected)

    def test_fixed_weights_and_ties_are_deterministic(self):
        for value in ({'a': '1/3'}, {'a': '-1', 'b': '2'}, {'missing': '1'}, {'a': '0/0'}):
            with self.assertRaises(ValueError):
                selector.weights({'weights': value}, ['a', 'b'])
        p = np.asarray([.1, .9, .2, .8])
        winner, _, _ = selector.choose_candidate([{'id': 'first', 'weights': {'a': '1'}},
                                                  {'id': 'second', 'weights': {'b': '1'}}], {'a': p, 'b': p}, self.labels)
        self.assertEqual(winner['id'], 'first')
        self.assertEqual(selector.legacy.choose_threshold(self.labels, p), .8)

    def test_selection_reproduces_mean_and_is_idempotent_and_source_pinned(self):
        out, result = self.selected()
        self.assertEqual(result['winner'], 'mean')
        self.assertEqual(result['validation_metrics']['f1'], 1)
        again = selector.select(result['candidate_specification'], result['candidate_specification_sha256'], out)
        self.assertEqual(again, result)
        self.trainer.write_text('# source changed\n')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            selector.select(result['candidate_specification'], result['candidate_specification_sha256'], out)

    def test_missing_member_rejected_final_and_exploratory_never_test_eligible(self):
        summary = self.search / 'b' / 'summary.json'
        value = selector.load(summary)
        value['completed'] = False
        self.write(summary, value)
        with self.assertRaisesRegex(ValueError, 'not completed'):
            self.spec()
        out, result = self.selected('exploratory')
        self.assertFalse(result['eligible_for_final_test'])
        with patch.object(selector.legacy, 'fixed_split', side_effect=AssertionError('must not access test')):
            with self.assertRaisesRegex(ValueError, 'test eligible'):
                evaluator.validate_selection(out / 'selection.json', selector.legacy.sha256(out / 'selection.json'))

    def test_changed_policy_weights_rejected_even_if_spec_hash_is_recomputed(self):
        path = self.spec()
        value = selector.load(path)
        value['candidates'][-1]['weights'] = {'a': '3/4', 'b': '1/4'}
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, 'fixed policy'):
            selector.validate_spec(path, selector.legacy.sha256(path))

    def test_final_lock_precedes_mock_forward_and_rejects_reselection(self):
        out, result = self.selected()
        path = out / 'selection.json'
        lock = self.root / 'final_lock.json'
        final = self.root / 'final_result'
        forwards = []

        def infer(**kwargs):
            self.assertEqual(selector.load(lock)['state'], 'running')
            self.assertEqual(kwargs['split'], 'test')
            name = Path(kwargs['checkpoint_path']).parent.name
            forwards.append(name)
            return {'source_id': self.ids[::-1], 'label': self.labels[::-1],
                    'probability': self.values[name][::-1], 'cohort_sha256': 'fixture-cohort',
                    'checkpoint_sha256': selector.legacy.sha256(kwargs['checkpoint_path'])}

        with patch.object(evaluator, 'import_trainer', return_value=SimpleNamespace(infer_selected_checkpoint=infer)):
            answer = evaluator.evaluate(path, selector.legacy.sha256(path), lock, final, threads=1)
            self.assertEqual(answer['test_metrics']['f1'], 1)
            self.assertEqual(len(forwards), 2)
            self.assertFalse(answer['individual_member_test_metrics_computed'])
            self.assertEqual(evaluator.evaluate(path, selector.legacy.sha256(path), lock, final, threads=1), answer)
            self.assertEqual(len(forwards), 2)
            changed = selector.load(path)
            changed['threshold'] = .5
            changed_path = self.root / 'changed_selection.json'
            self.write(changed_path, changed)
            with self.assertRaisesRegex(ValueError, 'another frozen selection'):
                evaluator.evaluate(changed_path, selector.legacy.sha256(changed_path), lock, final, threads=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
