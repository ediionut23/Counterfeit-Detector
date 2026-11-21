#!/usr/bin/env python3
"""Explicitly evaluate ONE frozen heterogeneous ensemble under a shared lock.

Never called automatically by validation search. Reports only aggregate test
metrics. Fixed validation weights and threshold cannot be fitted on test.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from evaluate_selected_ensemble import import_trainer
import heterogeneous_validation_ensemble as selector
from run_overnight_research import atomic_json


def validate_selection(path, expected_hash):
    path = Path(path).resolve()
    if selector.legacy.sha256(path) != expected_hash:
        raise ValueError('Selection differs from the explicit final-test hash')
    selection = selector.load(path)
    if (selection.get('schema') != selector.SCHEMA or selection.get('mode') != 'final'
            or selection.get('eligible_for_final_test') is not True
            or selection.get('exploratory') is not False
            or selection.get('selection_uses') != 'validation_only'
            or selection.get('test_predictions_accessed') is not False
            or selection.get('aggregation') != selector.AGGREGATION):
        raise ValueError('Only a final, frozen validation-only selection is test eligible')
    if (selection.get('selector_sha256') != selector.legacy.sha256(selector.__file__)
            or selection.get('helper_sha256') != selector.legacy.sha256(selector.legacy.__file__)):
        raise ValueError('Selected validation implementation changed')
    spec = selector.validate_spec(selection['candidate_specification'], selection['candidate_specification_sha256'])
    if spec['mode'] != 'final':
        raise ValueError('Exploratory spec cannot be used for final test')
    candidates = {candidate['id']: candidate for candidate in spec['candidates']}
    if selection['winner'] not in candidates:
        raise ValueError('Selected winner is absent from fixed candidates')
    declared = candidates[selection['winner']]
    member_weights = {m['id']: m['aggregation_weight'] for m in selection['members']}
    if len(member_weights) != len(selection['members']) or member_weights != declared['weights']:
        raise ValueError('Selected member weights differ from frozen candidate')
    selector.weights(declared, [m['id'] for m in spec['members']])
    if not np.isfinite(selection['threshold']) or not 0 <= selection['threshold'] <= 1:
        raise ValueError('Invalid frozen validation threshold')
    if (selection['cohort_sha256'] != spec['cohort_sha256']
            or selection['policy_sha256'] != spec['policy_sha256']):
        raise ValueError('Selection/spec cohort or policy disagrees')
    selector.verify_hashes(selection['frozen_input_sha256'])
    for member in selection['members']:
        for key in ('checkpoint', 'trainer', 'inference_adapter'):
            if selector.legacy.sha256(member[key]) != member[key + '_sha256']:
                raise ValueError('Frozen selected member changed')
    return selection


def evaluate(selection_path, selection_hash, test_lock, out, *, device='cpu', threads=2, batch_size=None):
    if threads < 1 or batch_size is not None and batch_size < 1:
        raise ValueError('Inference thread/batch settings must be positive')
    selection_path, test_lock, out = (Path(p).resolve() for p in (selection_path, test_lock, out))
    selection = validate_selection(selection_path, selection_hash)
    pin = {'selection_path': str(selection_path), 'selection_sha256': selection_hash,
           'cohort_sha256': selection['cohort_sha256'], 'winner': selection['winner'],
           'threshold': selection['threshold'], 'aggregation': selector.AGGREGATION,
           'members': [{'id': m['id'], 'weight': m['aggregation_weight'],
                        'checkpoint_sha256': m['checkpoint_sha256'],
                        'trainer_sha256': m['trainer_sha256'],
                        'inference_adapter_sha256': m['inference_adapter_sha256']}
                       for m in selection['members']], 'result_directory': str(out),
           'evaluator_sha256': selector.legacy.sha256(__file__),
           'inference_loader_sha256': selector.legacy.sha256(Path(__file__).with_name('evaluate_selected_ensemble.py'))}
    test_lock.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(test_lock) + '.lock').open('a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = selector.load(test_lock) if test_lock.exists() else None
        if previous:
            if previous.get('pin') != pin:
                raise ValueError('Shared test lock already belongs to another frozen selection')
            if previous.get('state') == 'complete':
                path = Path(previous['result_path'])
                if selector.legacy.sha256(path) != previous['result_sha256']:
                    raise ValueError('Existing locked test result changed')
                return selector.load(path)
        elif out.exists() and any(out.iterdir()):
            raise FileExistsError('Unowned nonempty final-test output directory')
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(test_lock, {'state': 'running', 'pin': pin})
        # The first test-ID/label access and first model forward occur only
        # after the immutable selection has acquired its global test lease.
        try:
            expected = selector.legacy.fixed_split(selection['dataset_csv'], 'test')
            ids = sorted(expected)
            labels = np.asarray([expected[name] for name in ids], dtype=int)
            total = np.zeros(len(ids), dtype=np.float64)
            for member in selection['members']:
                adapter = import_trainer(member['inference_adapter'])
                kwargs = {'checkpoint_path': member['checkpoint'], 'graph_index': selection['graph_index'],
                          'split_manifest': selection['cohort_manifest'], 'split': 'test',
                          'device': device, 'threads': threads}
                if batch_size is not None:
                    kwargs['batch_size'] = batch_size
                prediction = adapter.infer_selected_checkpoint(**kwargs)
                if (prediction.get('cohort_sha256') != selection['cohort_sha256']
                        or prediction.get('checkpoint_sha256') != member['checkpoint_sha256']):
                    raise ValueError('Inference checkpoint/cohort differs from selected member')
                values = selector.align_prediction(prediction['source_id'], prediction['label'], prediction['probability'], expected)
                total += float(Fraction(member['aggregation_weight'])) * values
            validate_selection(selection_path, selection_hash)
            if selector.legacy.sha256(Path(__file__).with_name('evaluate_selected_ensemble.py')) != pin['inference_loader_sha256']:
                raise ValueError('Inference adapter loader changed during final evaluation')
            report = selector.legacy.metrics(labels, total, selection['threshold'])
            result = {'pin': pin, 'cohort_sha256': selection['cohort_sha256'], 'winner': selection['winner'],
                      'selection_sha256': selection_hash, 'selected_members': pin['members'],
                      'test_metrics': report, 'test_metrics_at_0_5': selector.legacy.metrics(labels, total, .5),
                      'threshold_source': 'frozen_validation_only', 'individual_member_test_metrics_computed': False,
                      'target_f1': .85, 'test_target_reached': report['f1'] >= .85,
                      'interpretation': 'Fixed synthetic molecular-origin benchmark; real product fraud detection remains unvalidated.'}
            temporary = out / 'ensemble_test_predictions.csv.tmp'
            with temporary.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['source_id', 'split', 'label', 'probability_generated', 'threshold', 'prediction'])
                for name, label, value in zip(ids, labels, total):
                    writer.writerow([name, 'test', int(label), format(float(value), '.17g'), selection['threshold'], int(value >= selection['threshold'])])
            temporary.replace(out / 'ensemble_test_predictions.csv')
            atomic_json(out / 'test_result.json', result)
            atomic_json(test_lock, {'state': 'complete', 'pin': pin, 'result_path': str(out / 'test_result.json'),
                                    'result_sha256': selector.legacy.sha256(out / 'test_result.json'),
                                    'predictions_sha256': selector.legacy.sha256(out / 'ensemble_test_predictions.csv')})
            return result
        except BaseException as error:
            atomic_json(test_lock, {'state': 'failed', 'pin': pin, 'error': repr(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--test-lock', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    result = evaluate(args.selection, args.selection_sha256, args.test_lock, args.out,
                      device=args.device, threads=args.threads, batch_size=args.batch_size)
    print(json.dumps({'winner': result['winner'], 'test_metrics': result['test_metrics']}, indent=2))


if __name__ == '__main__':
    main()
