#!/usr/bin/env python3
"""Declare and select fixed heterogeneous ensembles using validation only.

No model forwards or test rows are accessed. Final selection requires every
member of the previously frozen policy; exploratory subsets cannot be tested.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import select_validation_ensemble as legacy


AGGREGATION = 'fixed_weight_probability_mean'
SCHEMA = 'heterogeneous_validation_ensemble_v1'


def load(path):
    return json.loads(Path(path).read_text())


def source_pins():
    return {str(Path(p).resolve()): legacy.sha256(p)
            for p in (__file__, legacy.__file__, Path(__file__).with_name('audit_overnight_training_split.py'))}


def verify_hashes(pins):
    for path, expected in pins.items():
        if legacy.sha256(path) != expected:
            raise ValueError('Frozen input changed: ' + path)


def weights(definition, member_ids):
    """Require exact positive rational weights summing to one."""
    declared = definition.get('weights')
    if not isinstance(declared, dict) or not declared:
        raise ValueError('Missing fixed candidate weights')
    if not set(declared) <= set(member_ids):
        raise ValueError('Candidate names an unknown member')
    try:
        result = {name: Fraction(value) for name, value in declared.items()}
    except (ValueError, ZeroDivisionError, TypeError) as error:
        raise ValueError('Invalid rational candidate weight') from error
    if any(value <= 0 for value in result.values()) or sum(result.values()) != 1:
        raise ValueError('Candidate weights must be positive and sum exactly to one')
    return result


def policy_at(path, expected_hash):
    path = Path(path).resolve()
    if legacy.sha256(path) != expected_hash:
        raise ValueError('Policy differs from the predeclared hash')
    policy = load(path)
    if (policy.get('schema_version') != 1 or policy.get('selection_uses') != 'validation_only'
            or policy.get('aggregation') != AGGREGATION
            or policy.get('all_members_required_for_final') is not True):
        raise ValueError('Unsupported frozen heterogeneous policy')
    ids = [member['id'] for member in policy['members']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Policy member IDs are empty or duplicated')
    candidate_ids = []
    for candidate in policy['candidates']:
        candidate_ids.append(candidate['id'])
        weights(candidate, ids)
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('Policy candidate IDs are empty or duplicated')
    return path, policy


def declarations(policy, mode, available):
    ids = [m['id'] for m in policy['members']]
    if mode == 'final':
        if available != ids:
            raise ValueError('Final selection requires all predeclared members')
        return policy['candidates']
    if mode != 'exploratory' or not available:
        raise ValueError('Exploration requires at least one completed member')
    result = [{'id': 'single_' + name, 'weights': {name: '1'}} for name in available]
    if len(available) > 1:
        result.append({'id': 'exploratory_all_available_equal',
                       'weights': dict.fromkeys(available, '1/' + str(len(available)))})
    return result


def member_paths(definition, workspace, search_root):
    folder = (search_root / definition['run_dir']).resolve()
    if folder.parent != search_root:
        raise ValueError('Member run directory escapes the selected search root')
    prediction = 'val_probabilities.csv' if definition['family'] == 'gatv2' else 'validation_best_predictions.csv'
    return {**definition, 'checkpoint': str(folder / 'best.pt'),
            'predictions': str(folder / prediction), 'summary': str(folder / 'summary.json'),
            'trainer': str((workspace / definition['trainer']).resolve()),
            'inference_adapter': str((workspace / definition['inference_adapter']).resolve())}


def auxiliary_paths(member, workspace):
    """Pin local source/config dependencies required by the frozen adapters."""
    result = []
    if Path(member['trainer']).name == 'train_gatv2_search.py':
        result.append(workspace / 'train_gatv2_experiment.py')
    if member['family'] == 'geometric':
        result.extend(workspace / name for name in ('train_geometric_gnn.py', 'geometric_angular_model.py'))
        result.append(Path(member['checkpoint']).parent / 'data_summary.json')
    return [str(path.resolve()) for path in result]


def publish_json(path, value):
    path = Path(path).resolve()
    if path.exists():
        if load(path) != value:
            raise ValueError('Immutable declaration already exists with different inputs: ' + str(path))
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    # Link rather than overwrite: competing builders cannot replace a frozen spec.
    try:
        os.link(temporary, path)
    except FileExistsError:
        if load(path) != value:
            raise ValueError('Concurrent immutable declaration differs')
    finally:
        temporary.unlink(missing_ok=True)
    return value


def build_spec(policy_path, policy_hash, workspace, search_root, cohort_dir, mode, out):
    policy_path, policy = policy_at(policy_path, policy_hash)
    workspace, search_root, cohort_dir = (Path(p).resolve() for p in (workspace, search_root, cohort_dir))
    manifest_path = cohort_dir / 'manifest.json'
    manifest = load(manifest_path)
    if manifest['cohort_sha256'] != policy['cohort_sha256']:
        raise ValueError('Policy and cohort differ')
    pins = {str(policy_path): policy_hash, str(manifest_path): legacy.sha256(manifest_path), **source_pins()}
    for name in ('dataset.csv', 'graph_index.csv'):
        path = cohort_dir / name
        digest = legacy.sha256(path)
        if digest != manifest['selected_files'][name]['sha256']:
            raise ValueError('Selected cohort file differs from audited manifest')
        pins[str(path)] = digest
    members = []
    for definition in policy['members']:
        member = member_paths(definition, workspace, search_root)
        summary_path = Path(member['summary'])
        summary = load(summary_path) if summary_path.exists() else {}
        if summary.get('completed') is not True:
            if mode == 'final':
                raise ValueError('Required final member is not completed: ' + member['id'])
            continue
        if summary.get('test_evaluated', False) or 'test_metrics' in summary:
            raise ValueError('Member accessed test: ' + member['id'])
        if summary.get('cohort_sha256') != policy['cohort_sha256']:
            raise ValueError('Member summary cohort differs: ' + member['id'])
        for key in ('checkpoint', 'predictions', 'summary', 'trainer', 'inference_adapter'):
            pins[member[key]] = legacy.sha256(member[key])
        for path in auxiliary_paths(member, workspace):
            pins[path] = legacy.sha256(path)
        members.append(member)
    candidates = declarations(policy, mode, [m['id'] for m in members])
    spec = {'schema': SCHEMA, 'mode': mode, 'workspace': str(workspace),
            'search_root': str(search_root), 'policy': str(policy_path), 'policy_sha256': policy_hash,
            'cohort_manifest': str(manifest_path), 'dataset_csv': str(cohort_dir / 'dataset.csv'),
            'graph_index': str(cohort_dir / 'graph_index.csv'), 'cohort_sha256': policy['cohort_sha256'],
            'members': members, 'candidates': candidates, 'aggregation': AGGREGATION,
            'frozen_spec_input_sha256': pins}
    verify_hashes(pins)
    return publish_json(out, spec)


def align_prediction(names, labels, probability, expected):
    """Validate complete ID/label alignment before combining model outputs."""
    names = [str(value) for value in names]
    y, p = np.asarray(labels), np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(names) != len(y) or len(y) != len(p):
        raise ValueError('Prediction arrays have incompatible shapes')
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError('Prediction IDs are duplicated, missing or unexpected')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Prediction probabilities are invalid')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Prediction labels are invalid')
    table = {name: (int(label), float(value)) for name, label, value in zip(names, y, p)}
    ids = sorted(expected)
    if any(table[name][0] != expected[name] for name in ids):
        raise ValueError('Prediction labels disagree with fixed cohort')
    return np.asarray([table[name][1] for name in ids], dtype=float)


def choose_candidate(candidates, probabilities, labels):
    reports, winner, winner_values, best = [], None, None, -1.
    for candidate in candidates:
        fixed = weights(candidate, probabilities)
        values = sum((float(weight) * probabilities[name] for name, weight in fixed.items()),
                     np.zeros(len(labels), dtype=np.float64))
        report = {'id': candidate['id'], 'weights': candidate['weights'],
                  **legacy.metrics(labels, values, legacy.choose_threshold(labels, values))}
        reports.append(report)
        if report['f1'] > best + 1e-12:
            best, winner, winner_values = report['f1'], report, values
    if winner is None:
        raise ValueError('No fixed candidate was declared')
    return winner, winner_values, reports


def validate_spec(spec_path, expected_hash):
    spec_path = Path(spec_path).resolve()
    if legacy.sha256(spec_path) != expected_hash:
        raise ValueError('Candidate spec differs from pinned hash')
    spec = load(spec_path)
    if spec.get('schema') != SCHEMA or spec.get('aggregation') != AGGREGATION:
        raise ValueError('Unsupported heterogeneous specification')
    _, policy = policy_at(spec['policy'], spec['policy_sha256'])
    if spec['cohort_sha256'] != policy['cohort_sha256']:
        raise ValueError('Spec cohort differs from policy')
    ids = [m['id'] for m in spec['members']]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated member in spec')
    if spec['candidates'] != declarations(policy, spec['mode'], ids):
        raise ValueError('Spec candidates differ from the fixed policy')
    by_id = {m['id']: m for m in policy['members']}
    if ids != [m['id'] for m in policy['members'] if m['id'] in ids]:
        raise ValueError('Spec member order differs from policy')
    for member in spec['members']:
        expected = member_paths(by_id[member['id']], Path(spec['workspace']), Path(spec['search_root']))
        if member != expected:
            raise ValueError('Spec member definition differs from policy')
        required = [member[key] for key in ('checkpoint', 'predictions', 'summary', 'trainer', 'inference_adapter')]
        required.extend(auxiliary_paths(member, Path(spec['workspace'])))
        if any(path not in spec['frozen_spec_input_sha256'] for path in required):
            raise ValueError('Spec omits required member input/source pins')
    if any(spec[key] not in spec['frozen_spec_input_sha256'] for key in ('policy', 'cohort_manifest', 'dataset_csv', 'graph_index')):
        raise ValueError('Spec omits required policy/cohort pins')
    for path, digest in source_pins().items():
        if spec['frozen_spec_input_sha256'].get(path) != digest:
            raise ValueError('Specification builder/selector implementation changed')
    verify_hashes(spec['frozen_spec_input_sha256'])
    return spec


def select(spec_path, spec_hash, out):
    spec_path, out = Path(spec_path).resolve(), Path(out).resolve()
    spec = validate_spec(spec_path, spec_hash)
    if out.exists():
        saved = load(out / 'selection.json')
        if saved.get('candidate_specification_sha256') != spec_hash or saved.get('selector_sha256') != legacy.sha256(__file__):
            raise ValueError('Existing immutable selection belongs to another input')
        verify_hashes(saved['frozen_input_sha256'])
        if legacy.sha256(out / 'validation_ensemble_probabilities.csv') != saved['validation_ensemble_predictions_sha256']:
            raise ValueError('Saved validation predictions changed')
        return saved
    manifest = load(spec['cohort_manifest'])
    for name, field in (('dataset.csv', 'dataset_csv'), ('graph_index.csv', 'graph_index')):
        if legacy.sha256(spec[field]) != manifest['selected_files'][name]['sha256']:
            raise ValueError('Cohort file differs from audited manifest')
    expected = legacy.fixed_split(spec['dataset_csv'], 'val')
    if len(expected) != sum(manifest['split_label_counts']['val'].values()):
        raise ValueError('Actual full validation size differs from audited manifest')
    ids = sorted(expected)
    labels = np.asarray([expected[name] for name in ids], dtype=int)
    probabilities, members, member_reports = {}, {}, {}
    pins = {**spec['frozen_spec_input_sha256'], str(spec_path): spec_hash}
    for declared in spec['members']:
        member = dict(declared)
        summary = load(member['summary'])
        if summary.get('completed') is not True or summary.get('test_evaluated', False) or 'test_metrics' in summary:
            raise ValueError('Member must be completed and validation-only')
        checkpoint, config, cohort, digest = legacy.checkpoint_details(member['checkpoint'])
        if cohort != spec['cohort_sha256'] or int(config['seed']) != int(member['seed']):
            raise ValueError('Member checkpoint seed/cohort differs')
        epoch = int(checkpoint['epoch'])
        if epoch != int(checkpoint.get('best_epoch', epoch)) or epoch != int(summary['best_epoch']):
            raise ValueError('Member checkpoint is not its validation-selected best')
        table, epochs = legacy.read_probabilities(member['predictions'], expected)
        if epochs and epochs != {epoch}:
            raise ValueError('Prediction epoch differs from selected checkpoint')
        values = np.asarray([table[name] for name in ids], dtype=float)
        report = legacy.metrics(labels, values, legacy.choose_threshold(labels, values))
        recorded = checkpoint.get('validation_metrics', {}).get('f1')
        if (recorded is None or abs(float(recorded) - report['f1']) > 1e-6
                or abs(float(summary['best_validation_f1']) - report['f1']) > 1e-6):
            raise ValueError('Predictions do not reproduce completed checkpoint F1')
        trainer_hash = legacy.sha256(member['trainer'])
        pinned_trainer = checkpoint.get('provenance', {}).get('trainer_sha256', config.get('script_sha256'))
        if pinned_trainer and pinned_trainer != trainer_hash:
            raise ValueError('Checkpoint trainer source pin changed')
        extra_pins = []
        if checkpoint.get('provenance', {}).get('base_protocol_sha256'):
            extra_pins.append((str(Path(spec['workspace']) / 'train_gatv2_experiment.py'),
                               checkpoint['provenance']['base_protocol_sha256']))
        for key, filename in (('engine_sha256', 'train_geometric_gnn.py'),
                              ('geometry_model_sha256', 'geometric_angular_model.py')):
            if config.get(key):
                extra_pins.append((str(Path(spec['workspace']) / filename), config[key]))
        if any(legacy.sha256(path) != digest for path, digest in extra_pins):
            raise ValueError('Checkpoint auxiliary engine/model source pin changed')
        if digest != pins[member['checkpoint']]:
            raise ValueError('Checkpoint changed after spec declaration')
        member.update(checkpoint_sha256=digest, trainer_sha256=trainer_hash,
                      inference_adapter_sha256=legacy.sha256(member['inference_adapter']),
                      predictions_sha256=legacy.sha256(member['predictions']),
                      summary_sha256=legacy.sha256(member['summary']), best_epoch=epoch)
        members[member['id']], probabilities[member['id']], member_reports[member['id']] = member, values, report
    winner, values, reports = choose_candidate(spec['candidates'], probabilities, labels)
    verify_hashes(pins)
    selected = []
    for name, weight in winner['weights'].items():
        selected.append({**members[name], 'aggregation_weight': weight})
    result = {'schema': SCHEMA, 'mode': spec['mode'], 'exploratory': spec['mode'] == 'exploratory',
              'eligible_for_final_test': spec['mode'] == 'final', 'selection_uses': 'validation_only',
              'test_predictions_accessed': False, 'aggregation': AGGREGATION,
              'candidate_specification': str(spec_path), 'candidate_specification_sha256': spec_hash,
              'policy': spec['policy'], 'policy_sha256': spec['policy_sha256'],
              'cohort_sha256': spec['cohort_sha256'], 'cohort_manifest': spec['cohort_manifest'],
              'cohort_manifest_sha256': legacy.sha256(spec['cohort_manifest']),
              'dataset_csv': spec['dataset_csv'], 'dataset_csv_sha256': legacy.sha256(spec['dataset_csv']),
              'graph_index': spec['graph_index'], 'graph_index_sha256': legacy.sha256(spec['graph_index']),
              'validation_ids_sha256': hashlib.sha256(legacy.encoded(ids).encode()).hexdigest(),
              'winner': winner['id'], 'members': selected, 'threshold': winner['threshold'],
              'validation_metrics': winner, 'candidate_validation_metrics': reports,
              'member_validation_metrics': member_reports, 'frozen_input_sha256': pins,
              'selector_sha256': legacy.sha256(__file__), 'helper_sha256': legacy.sha256(legacy.__file__),
              'interpretation': 'Exploratory validation evidence; not eligible for final test.' if spec['mode'] == 'exploratory' else
                  'Frozen validation-only selection. Repeated validation search is optimistic; report final test separately.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + out.name, dir=out.parent))
    try:
        with (staging / 'validation_ensemble_probabilities.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_id', 'split', 'label', 'probability_generated', 'threshold', 'prediction'])
            for identity, label, value in zip(ids, labels, values):
                writer.writerow([identity, 'val', int(label), format(float(value), '.17g'), winner['threshold'], int(value >= winner['threshold'])])
        result['validation_ensemble_predictions_sha256'] = legacy.sha256(staging / 'validation_ensemble_probabilities.csv')
        (staging / 'selection.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    declaration = sub.add_parser('spec')
    declaration.add_argument('--policy', type=Path, required=True)
    declaration.add_argument('--policy-sha256', required=True)
    declaration.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parent)
    declaration.add_argument('--search-root', type=Path, default=Path('training_runs/large_search_20261101'))
    declaration.add_argument('--cohort-dir', type=Path, default=Path('balanced_datasets/training_200k/overnight_strict_converged_v2'))
    declaration.add_argument('--mode', choices=('exploratory', 'final'), required=True)
    declaration.add_argument('--out', type=Path, required=True)
    selection = sub.add_parser('select')
    selection.add_argument('--spec', type=Path, required=True)
    selection.add_argument('--spec-sha256', required=True)
    selection.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'spec':
        result = build_spec(args.policy, args.policy_sha256, args.workspace, args.search_root, args.cohort_dir, args.mode, args.out)
        print(json.dumps({'spec': str(args.out), 'sha256': legacy.sha256(args.out),
                          'mode': result['mode'], 'members': len(result['members']), 'candidates': len(result['candidates'])}))
    else:
        result = select(args.spec, args.spec_sha256, args.out)
        print(json.dumps({key: result[key] for key in ('mode', 'eligible_for_final_test', 'winner', 'validation_metrics')}, indent=2))


if __name__ == '__main__':
    main()
