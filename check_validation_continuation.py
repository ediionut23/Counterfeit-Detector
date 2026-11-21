#!/usr/bin/env python3
"""Temporary-only continuation fixtures: no chemistry, training, audit or test."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import run_large_model_search as base
import run_validation_continuation as continuation
from run_overnight_research import atomic_json, checksum


def expect_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Expected rejection')


def protocol_checks(root):
    root.mkdir()
    for name in continuation.REQUIRED_SOURCE_PINS:
        (root / name).write_text('temporary source fixture: ' + name)
    plan = {'workspace': str(root), 'out_dir': 'runs', 'training_cohort': {'cohort_sha256': 'fixture-cohort'}}
    plan_path, path = root / 'plan.json', root / 'protocol.json'
    atomic_json(plan_path, plan)
    protocol = {'schema_version': 1, 'protocol_id': 'validation_fixture', 'mode': 'validation_only',
        'test_evaluation': False, 'workspace': str(root), 'out_dir': 'runs',
        'plan_sha256': checksum(plan_path), 'training_cohort_sha256': 'fixture-cohort',
        'source_pins': {name: checksum(root / name) for name in continuation.REQUIRED_SOURCE_PINS}}
    atomic_json(path, protocol)
    digest = checksum(path)
    with patch.object(continuation, '__file__', str(root / 'run_validation_continuation.py')):
        verified, _, out = continuation.verified_protocol(plan_path, path, digest)
        saved = continuation.freeze_protocol(path, digest, verified, out)
        assert checksum(saved) == digest
        assert continuation.freeze_protocol(path, digest, verified, out) == saved
        bad = copy.deepcopy(protocol)
        bad['test_evaluation'] = True
        atomic_json(path, bad)
        expect_error(lambda: continuation.verified_protocol(plan_path, path, checksum(path)))
        expect_error(lambda: continuation.verified_protocol(plan_path, path, digest))
        atomic_json(path, protocol)
        (root / 'run_large_model_search.py').write_text('tampered source')
        expect_error(lambda: continuation.verified_protocol(plan_path, path, checksum(path)))
        expect_error(lambda: continuation.freeze_protocol(path, '0' * 64, protocol, out))


class Harness(continuation.ValidationContinuation):
    def verify_derived(self, run, family):
        self.derived.append((family, copy.deepcopy(run)))
        assert '--no-test' in run['command']
        assert run['command'][run['command'].index('--graph-index') + 1] == 'fixture_cohort/graph_index.csv'

    def run(self, key, command, *, resume_dir=None):
        self.validate_command(key, command)
        if self.state['stages'].get(key, {}).get('status') == 'complete':
            return
        command = list(command)
        self.events.append((key, command))
        if key.startswith(('screen_', 'promote_', 'repeat_')):
            flag = '--out-dir' if '--out-dir' in command else '--out'
            folder = Path(command[command.index(flag) + 1])
            folder.mkdir(parents=True, exist_ok=True)
            epochs = int(command[command.index('--epochs') + 1])
            score = .70 + (sum(folder.name.encode()) % 10) / 100 + (.02 if epochs == 100 else 0)
            atomic_json(folder / 'summary.json', {'completed': True,
                'cohort_sha256': self.plan['training_cohort']['cohort_sha256'],
                'best_validation_f1': score, 'test_evaluated': False, 'best_epoch': epochs})
            (folder / 'best.pt').write_bytes(b'fake checkpoint; never loaded')
            for name in ('val_probabilities.csv', 'validation_best_predictions.csv'):
                (folder / name).write_text('fake validation probabilities; never evaluated\n')
        else:
            assert key.endswith('_select_on_validation')
            assert sum(event.startswith('repeat_') for event, _ in self.events) == 4
            specification = json.loads(Path(command[command.index('--spec') + 1]).read_text())
            assert checksum(command[command.index('--spec') + 1]) == command[command.index('--spec-sha256') + 1]
            members = {member['id']: member for member in specification['members']}
            selected = [members[name] for name in specification['candidates'][-1]['members']]
            assert sorted(member['seed'] for member in selected) == [42, 43, 44]
            assert len({member['config_id'] for member in selected}) == 1
            atomic_json(Path(command[command.index('--out') + 1]) / 'selection.json', {'selection_uses': 'validation_only'})
        self.state['stages'][key] = {'status': 'complete'}


def workflow_checks(root):
    root.mkdir()
    runner = Harness.__new__(Harness)
    runner.root, runner.out, runner.plan_path = root, root / 'runs', root / 'plan.json'
    runner.out.mkdir()
    runner.events, runner.derived, runner.saved_statuses = [], [], []
    runner.continuation_protocol = {'protocol_id': 'validation_fixture'}
    runner.protocol_sha256, runner.protocol_snapshot = 'fixture-digest', root / 'protocol.json'
    runner.state = {'stages': {}, 'status': 'running', 'experiments': {}}
    families = []
    for family, script, flag in [('gatv2', 'train_gatv2_search.py', '--out-dir'),
        ('geometric', 'train_geometric_variants.py', '--out')]:
        runs = []
        for slot in range(2):
            identity = family + '_' + str(slot)
            folder = str(runner.out / identity)
            command = ['python', script, '--graph-index', 'fixture_cohort/graph_index.csv',
                '--no-test', '--patience', '12', '--seed', '42', '--epochs', '20', flag, folder]
            runs.append({'id': identity, 'out_dir': folder, 'command': command})
        families.append({'id': family, 'runs': runs})
    runner.plan = {'training_cohort': {'directory': 'fixture_cohort', 'cohort_sha256': 'fixture-cohort'},
        'families': families, 'search_policy': {'screen_epochs': 20, 'final_epochs': 100,
            'promote_per_family': 1, 'repeat_seeds': [43, 44]}, 'target_f1': .85}
    def parent_save(instance):
        instance.saved_statuses.append(instance.state['status'])
    with patch.object(base, 'cohort_audit', return_value={'fixture': True}) as audit, \
         patch.object(base.LargeSearch, 'save', parent_save), \
         patch('subprocess.Popen', side_effect=AssertionError('No subprocess may run in this fixture')):
        runner.work()
        assert audit.call_count == 1
        assert runner.state['status'] == 'validation_complete'
        assert 'complete' not in runner.saved_statuses
        assert runner.state['continuation_protocol']['test_evaluation'] is False
        assert len(runner.events) == 12  # 4 screens + 2 promotions + 4 repeats + 2 selectors.
        assert sum(key.startswith('screen_') for key, _ in runner.events) == 4
        assert sum(key.startswith('promote_') for key, _ in runner.events) == 2
        assert sum(key.startswith('repeat_') for key, _ in runner.events) == 4
        assert len(runner.derived) == 4
        assert all('evaluate_selected_ensemble.py' not in command for _, command in runner.events)
        assert all('test_metrics' not in json.loads(path.read_text()) for path in runner.out.glob('*/summary.json'))
        events = copy.deepcopy(runner.events)
        runner.work()
        assert runner.events == events
        assert runner.state['status'] == 'validation_complete'
    expect_error(lambda: runner.validate_command('gatv2_ensemble_locked_test', ['python', 'evaluate_selected_ensemble.py']))
    expect_error(lambda: runner.validate_command('screen_bad', ['python', 'train_gatv2_search.py']))
    expect_error(lambda: runner.validate_command('screen_bad', ['python', 'train_gatv2_search.py', '--no-test', '--evaluate-test']))
    expect_error(lambda: runner.validate_command('gatv2_select_on_validation', ['python', 'evaluate_selected_ensemble.py']))


def main():
    with tempfile.TemporaryDirectory(prefix='validation_continuation_') as directory:
        root = Path(directory)
        protocol_checks(root / 'pins')
        workflow_checks(root / 'workflow')
    print(json.dumps({'passed': True, 'protocol_hash_and_source_guards': True,
        'immutable_protocol_snapshot': True, 'all_validation_stages_retained': True,
        'final_state': 'validation_complete', 'test_command_rejected': True,
        'idempotent_rerun': True, 'real_training_audit_and_test_executed': False}))


if __name__ == '__main__':
    main()
