#!/usr/bin/env python3
"""Durable Ax ask/train/tell search for one angular geometric architecture.

All fitting uses the same frozen161157 training molecules. Ax helpers are
separate, thermally guardable Queue stages. One atomic bundle stores the Ax
snapshot AND its trial ledger, so interrupted ask/tell operations are idempotent.
No ensemble or test evaluation is implemented in this campaign.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

from enforce_large_training_cohort import audit as cohort_audit
from run_overnight_research import Queue, atomic_json, checksum, load_json, now


PARAMETER_SPACE = {
    'lr': {'kind': 'range', 'bounds': [3e-4, 1.2e-3], 'log': True},
    'dropout': {'kind': 'range', 'bounds': [.05, .30], 'log': False},
    'weight_decay': {'kind': 'range', 'bounds': [1e-6, 1e-3], 'log': True},
    'hidden': {'kind': 'choice', 'values': [64, 96, 128]},
    'interactions': {'kind': 'choice', 'values': [2, 3, 4]},
    'angular_basis': {'kind': 'choice', 'values': [4, 6, 8, 12]},
}
FLAGS = {'lr': '--lr', 'dropout': '--dropout', 'weight_decay': '--weight-decay',
         'hidden': '--hidden', 'interactions': '--interactions', 'angular_basis': '--angular-basis'}


def argument(command, flag, value):
    command = list(command)
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])
    return command


def normalized_parameters(parameters):
    if set(parameters) != set(PARAMETER_SPACE):
        raise ValueError('Ax parameter names differ from frozen domain')
    result = {}
    for name, definition in PARAMETER_SPACE.items():
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError('Nonfinite or nonnumeric Ax parameter: ' + name)
        if definition['kind'] == 'choice':
            if value not in definition['values']:
                raise ValueError('Ax choice outside frozen domain: ' + name)
            result[name] = int(value)
        else:
            low, high = definition['bounds']
            if not low <= float(value) <= high:
                raise ValueError('Ax range outside frozen domain: ' + name)
            result[name] = float(value)
    return result


def verify_plan(path, expected_sha=None):
    path = Path(path).resolve()
    if expected_sha and checksum(path) != expected_sha:
        raise ValueError('Ax plan changed after stage declaration')
    plan = json.loads(path.read_text())
    root = Path(plan['workspace']).resolve()
    if plan.get('protocol') != 'ax_angular_validation_only_v1' or plan.get('test_evaluation') is not False:
        raise ValueError('Campaign must explicitly forbid test evaluation')
    if plan.get('parameter_space') != PARAMETER_SPACE:
        raise ValueError('HPO parameter domain differs from declared implementation')
    policy = plan['search_policy']
    if (policy['new_trials'] != 8 or policy['sobol_trials'] != 4 or policy['screen_epochs'] != 40
            or policy['promote_count'] != 2 or policy['final_epochs'] != 100
            or policy['repeat_seeds'] != [43, 44] or policy['training_seed'] != 42
            or policy['patience'] != 15 or policy['selection_metric'] != 'validation_f1'):
        raise ValueError('Campaign budget/selection protocol changed')
    command = plan['template_command']
    fixed = {'--geometry-variant': 'angular', '--radial-basis': '16', '--cutoff': '4.5',
             '--max-neighbors': '24', '--batch-size': '64', '--patience': '15',
             '--scheduler-patience': '3', '--threads': '2', '--num-workers': '0',
             '--device': 'cpu', '--weight-mode': 'none', '--selection': 'val_f1', '--seed': '42'}
    if len(command) < 2 or command[1] != 'train_geometric_variants.py' or '--no-test' not in command:
        raise ValueError('Expected the unchanged validation-only geometric trainer')
    if any(flag in command for flag in ['--evaluate-test', '--include-hydrogens', '--prepare-only']):
        raise ValueError('Test/data-selection flags are outside the frozen HPO protocol')
    for flag, value in fixed.items():
        if flag not in command or command[command.index(flag) + 1] != value:
            raise ValueError('Fixed HPO setting changed: ' + flag)
    required_sources = {'run_ax_geometry_search.py', 'make_ax_geometry_search_plan.py',
        'run_overnight_research.py', 'enforce_large_training_cohort.py',
        'train_geometric_variants.py', 'train_geometric_gnn.py', 'geometric_angular_model.py',
        'thermal_guard_ax.py', 'thermal_guard_cool_v2.py', 'thermal_guard.py'}
    if not required_sources <= plan['file_pins'].keys():
        raise ValueError('Missing required HPO implementation/thermal source pins')
    for filename, digest in plan['file_pins'].items():
        if checksum(root / filename) != digest:
            raise ValueError('Pinned HPO input/source changed: ' + filename)
    runtime = plan['ax_runtime']
    if runtime['version'] != '1.3.1' or checksum(root / runtime['manifest']) != runtime['manifest_sha256']:
        raise ValueError('Ax runtime version/manifest changed')
    return plan


def verify_runtime(plan):
    root = Path(plan['workspace'])
    runtime = plan['ax_runtime']
    directory = (root / runtime['directory']).resolve()
    manifest = json.loads((root / runtime['manifest']).read_text())
    files = manifest.get('installed_files_sha256')
    if not isinstance(files, dict) or not files:
        raise ValueError('Ax runtime manifest lacks installed file hashes')
    for relative, expected in files.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or checksum(path) != expected:
            raise ValueError('Isolated Ax runtime file changed: ' + relative)
    return directory


def ax_api(plan):
    directory = verify_runtime(plan)
    sys.path.insert(0, str(directory))
    import torch
    import ax
    from ax.api.client import Client
    from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
    from importlib.metadata import version
    if version('ax-platform') != '1.3.1' or not Path(ax.__file__).resolve().is_relative_to(directory):
        raise ValueError('Ax did not load from the pinned isolated runtime')
    torch.set_num_threads(1)
    return Client, RangeParameterConfig, ChoiceParameterConfig


def client_snapshot(client):
    # Keep the exact public Ax JSON as a string. Ax may encode missing SEM as
    # NaN internally; the surrounding atomic bundle remains strict JSON.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'client.json'
        client.save_to_json_file(str(path))
        return path.read_text()


def restore_client(Client, snapshot):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'client.json'
        path.write_text(snapshot)
        return Client.load_from_json_file(str(path))


def ledger_hash(trials):
    return hashlib.sha256(json.dumps(trials, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_bundle(bundle, plan_sha):
    if (bundle.get('schema_version') != 1 or bundle.get('plan_sha256') != plan_sha
            or not isinstance(bundle.get('ax_snapshot_json'), str)
            or hashlib.sha256(bundle['ax_snapshot_json'].encode()).hexdigest() != bundle.get('ax_snapshot_sha256')):
        raise ValueError('Ax snapshot schema/plan/checksum changed')
    trials = bundle.get('trials')
    if not isinstance(trials, list) or not 1 <= len(trials) <= 8 or ledger_hash(trials) != bundle.get('ledger_sha256'):
        raise ValueError('Ax trial ledger checksum/size changed')
    indices = []
    pending = []
    for ordinal, trial in enumerate(trials):
        if trial.get('ordinal') != ordinal or type(trial.get('ax_trial_index')) is not int or type(trial.get('told')) is not bool:
            raise ValueError('Ax trial ledger ordering/status changed')
        normalized_parameters(trial['parameters'])
        indices.append(trial['ax_trial_index'])
        if not trial['told']:
            pending.append(ordinal)
        else:
            score = trial.get('screening_result', {}).get('best_validation_f1')
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Completed Ax trial has no finite recorded objective')
    if len(set(indices)) != len(indices) or pending not in ([], [len(trials) - 1]):
        raise ValueError('Ax ledger contains duplicate trial indices or multiple in-flight trials')
    return bundle


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Immutable HPO declaration changed: ' + str(path))
    else:
        atomic_json(path, value)
    return value


def new_client(plan, api):
    Client, Range, Choice = api
    client = Client(random_seed=plan['search_policy']['ax_seed'])
    parameters = []
    for name, definition in PARAMETER_SPACE.items():
        if definition['kind'] == 'choice':
            parameters.append(Choice(name=name, values=definition['values'], parameter_type='int', is_ordered=True))
        else:
            parameters.append(Range(name=name, bounds=tuple(definition['bounds']), parameter_type='float',
                                    scaling='log' if definition['log'] else 'linear'))
    client.configure_experiment(parameters=parameters, name=plan['experiment_name'],
                                description='Angular geometric GNN, full frozen train; validation F1 only')
    client.configure_optimization(objective='validation_f1')
    baseline = plan['historical_baseline']
    history = json.loads(Path(plan['workspace'], baseline['history']).read_text())
    first = [row for row in history if int(row['epoch']) <= plan['search_policy']['screen_epochs']]
    if not first or max(int(row['epoch']) for row in first) != 40:
        raise ValueError('Historical warm observation must contain the full first40 epochs')
    warm_score = max(float(row['val_f1']) for row in first)
    index = client.attach_trial(parameters=normalized_parameters(baseline['parameters']), arm_name='historical_angular_seed42_first40')
    client.complete_trial(trial_index=index, raw_data={'validation_f1': warm_score})
    client.configure_generation_strategy(method='fast', initialization_budget=4,
        initialization_random_seed=plan['search_policy']['ax_seed'], initialize_with_center=False,
        use_existing_trials_for_initialization=False, min_observed_initialization_trials=4,
        allow_exceeding_initialization_budget=False, torch_device='cpu')
    return client, {'trial_index': index, 'score': warm_score, 'budget_epochs': 40,
                    'source_history_sha256': checksum(Path(plan['workspace']) / baseline['history']),
                    'not_counted_among_eight_new_trials': True}


def training_run(plan, trial, *, seed=42):
    ordinal = int(trial['ordinal'])
    ident = f'ax_trial_{ordinal:02d}' if seed == 42 else f'winner_seed{seed}'
    directory = Path(plan['workspace']) / plan['out_dir'] / ident
    command = list(plan['template_command'])
    command = argument(command, '--out', directory)
    command = argument(command, '--seed', seed)
    for name, value in normalized_parameters(trial['parameters']).items():
        command = argument(command, FLAGS[name], value)
    return {'id': ident, 'out_dir': str(directory), 'command': command,
            'trial_ordinal': ordinal, 'parameters': trial['parameters'], 'seed': seed}


def validation_result(run, cohort):
    directory = Path(run['out_dir'])
    summary = load_json(directory / 'summary.json')
    score = summary.get('best_validation_f1')
    if (summary.get('completed') is not True or summary.get('cohort_sha256') != cohort
            or summary.get('test_evaluated') or 'test_metrics' in summary
            or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Missing finite full-cohort validation-only result: ' + run['id'])
    if not (directory / 'best.pt').exists() or not (directory / 'last.pt').exists():
        raise ValueError('Completed HPO run lacks atomic model checkpoints')
    return {'best_validation_f1': float(score), 'best_epoch': summary['best_epoch'],
            'completed_epoch': summary['current_epoch'], 'summary_sha256': checksum(directory / 'summary.json'),
            'best_checkpoint_sha256': checksum(directory / 'best.pt')}


def retain_incumbent(new_seed42, new_mean, incumbent_seed42, incumbent_mean):
    """Do not replace the old single model with a worse or seed-lucky candidate."""
    return new_seed42 <= incumbent_seed42 + 1e-6 or new_mean <= incumbent_mean + 1e-6


def ax_step(plan_path, expected_sha, mode, ordinal):
    plan = verify_plan(plan_path, expected_sha)
    if not 0 <= ordinal < plan['search_policy']['new_trials']:
        raise ValueError('Ax ordinal outside declared trial budget')
    out = Path(plan['workspace']) / plan['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'ax_bundle.json'
    with Path(str(path) + '.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle = load_json(path)
        if bundle:
            validate_bundle(bundle, expected_sha)
        # Fast idempotent retries do not import/re-fit Ax at all.
        trials = bundle.get('trials', [])
        if ordinal < len(trials) and (mode == 'ask' or trials[ordinal].get('told')):
            return trials[ordinal]
        api = ax_api(plan)
        if bundle:
            client = restore_client(api[0], bundle['ax_snapshot_json'])
        else:
            if mode != 'ask' or ordinal != 0:
                raise ValueError('Ax search has no durable initial ask')
            client, warm = new_client(plan, api)
            bundle = {'schema_version': 1, 'plan_sha256': expected_sha, 'trials': [],
                      'historical_warm_observation': warm, 'created_at': now()}
        if mode == 'ask':
            if ordinal != len(bundle['trials']) or any(not item.get('told') for item in bundle['trials']):
                raise ValueError('A previous Ax trial is pending; do not generate a duplicate ask')
            import torch
            torch.manual_seed(plan['search_policy']['ax_seed'] + ordinal)
            random.seed(plan['search_policy']['ax_seed'] + ordinal)
            import numpy as np
            np.random.seed(plan['search_policy']['ax_seed'] + ordinal)
            proposal = client.get_next_trials(max_trials=1)
            if len(proposal) != 1:
                raise ValueError('Ax did not produce exactly one sequential trial')
            index, parameters = next(iter(proposal.items()))
            trial = {'ordinal': ordinal, 'ax_trial_index': int(index),
                     'parameters': normalized_parameters(parameters), 'told': False,
                     'declared_generation_phase': 'sobol' if ordinal < 4 else 'bayesian_fast', 'asked_at': now()}
            if any(item['parameters'] == trial['parameters'] for item in bundle['trials']):
                raise ValueError('Ax repeated an already declared configuration')
            bundle['trials'].append(trial)
        else:
            if ordinal >= len(bundle['trials']):
                raise ValueError('Cannot tell an unasked trial')
            trial = bundle['trials'][ordinal]
            run = training_run(plan, trial)
            result = validation_result(run, plan['training_cohort']['cohort_sha256'])
            if result['completed_epoch'] > plan['search_policy']['screen_epochs']:
                raise ValueError('Ax objective may only use its frozen screening epoch budget')
            client.complete_trial(trial_index=trial['ax_trial_index'],
                                  raw_data={'validation_f1': result['best_validation_f1']})
            trial.update(told=True, told_at=now(), screening_result=result,
                         objective='best_validation_f1_within40epochs_or_early_stop')
        bundle['ax_snapshot_json'] = client_snapshot(client)
        bundle['ax_snapshot_sha256'] = hashlib.sha256(bundle['ax_snapshot_json'].encode()).hexdigest()
        bundle['ledger_sha256'] = ledger_hash(bundle['trials'])
        bundle['updated_at'] = now()
        atomic_json(path, bundle)
        return trial


class AxGeometrySearch(Queue):
    def __init__(self, plan_path, startup_token=None):
        self.dynamic_runs = {}
        self.current_phase = 'initializing'
        verify_plan(plan_path)
        super().__init__(plan_path, startup_token)
        self.plan_hash = checksum(self.plan_path)
        self.bundle_path = self.out / 'ax_bundle.json'
        frozen_json(self.out / 'plan_snapshot.json', self.plan)

    def save(self):
        super().save()
        self.state['adaptive_run_dirs'] = dict(self.state.get('adaptive_run_dirs', {}))
        for ident, run in self.dynamic_runs.items():
            self.state['adaptive_run_dirs'][ident] = run['out_dir']
        self.state['experiments'] = {ident: load_json(Path(folder) / 'progress.json')
                                     for ident, folder in self.state['adaptive_run_dirs'].items()}
        self.state.update(campaign='Ax1.3.1/angular/full_train/validation_only',
                          phase=self.current_phase, test_evaluated=False, ensemble=False)
        bundle = load_json(self.out / 'ax_bundle.json')
        self.state['ax_trials'] = [{key: trial.get(key) for key in ('ordinal', 'ax_trial_index', 'parameters', 'told', 'screening_result')}
                                   for trial in bundle.get('trials', [])]
        atomic_json(self.out / 'progress.json', self.state)

    def verify_stage(self, key, run):
        verify_plan(self.plan_path, self.plan_hash)
        derived = {**self.plan, 'families': [{'id': 'angular', 'runs': [run]}]}
        path = self.out / 'derived_plans' / (key + '.json')
        atomic_json(path, derived)
        atomic_json(path.with_suffix('.guard.json'), cohort_audit(path, self.plan['training_cohort']['directory']))

    def helper(self, mode, ordinal):
        key = f'ax_{mode}_{ordinal:02d}'
        template = {'id': 'full_cohort_guard_template', 'out_dir': str(self.out / 'guard_template'),
                    'command': argument(self.plan['template_command'], '--out', self.out / 'guard_template')}
        self.verify_stage(key, template)
        self.current_phase = mode
        self.run(key, [sys.executable, 'run_ax_geometry_search.py', '--plan', str(self.plan_path),
                      '--plan-sha256', self.plan_hash, '--ax-step', mode, '--ordinal', str(ordinal)])

    def train(self, run, key, epochs):
        self.dynamic_runs[run['id']] = run
        command = argument(run['command'], '--epochs', epochs)
        guarded = {**run, 'command': command}
        self.verify_stage(key, guarded)
        if '--no-test' not in command or '--evaluate-test' in command or command[1] != 'train_geometric_variants.py':
            raise ValueError('Only the frozen validation-only angular trainer may run')
        self.current_phase = key
        self.run(key, command, resume_dir=run['out_dir'])
        if key.startswith('ax_train_'):
            trial = self.bundle()['trials'][run['trial_ordinal']]
            if trial.get('told'):
                # Promotion deliberately reuses this seed42 directory. Its
                # current summary may be incomplete after a later interruption;
                # the completed40-epoch observation must remain frozen.
                frozen = trial['screening_result']
                promotion = load_json(self.out / 'promotion.json')
                if promotion:
                    evidence = next(item for item in promotion['candidate_evidence'] if item['ordinal'] == trial['ordinal'])
                    for filename, digest in evidence['evidence_sha256'].items():
                        if checksum(filename) != digest:
                            raise ValueError('Frozen screening evidence changed: ' + filename)
                elif validation_result(run, self.plan['training_cohort']['cohort_sha256']) != frozen:
                    raise ValueError('Screening outputs changed before promotion was frozen')
                return dict(frozen)
        return validation_result(run, self.plan['training_cohort']['cohort_sha256'])

    def bundle(self):
        bundle = load_json(self.bundle_path)
        return validate_bundle(bundle, self.plan_hash)

    def freeze_promotions(self):
        path = self.out / 'promotion.json'
        trials = self.bundle()['trials']
        if len(trials) != 8 or not all(t['told'] for t in trials):
            raise ValueError('Every new Ax trial must be told before promotion')
        ranked = sorted(trials, key=lambda t: (-t['screening_result']['best_validation_f1'], t['ordinal']))
        selected = [t['ordinal'] for t in ranked[:2]]
        if path.exists():
            prior = load_json(path)
            if prior.get('selected_ordinals') != selected or {x['ordinal'] for x in prior.get('candidate_evidence', [])} != set(range(8)):
                raise ValueError('Frozen promotion differs from the durable screening ranking')
            for item in prior['candidate_evidence']:
                if item['screening_result'] != trials[item['ordinal']]['screening_result']:
                    raise ValueError('Frozen promotion objective changed')
                for filename, digest in item['evidence_sha256'].items():
                    if checksum(filename) != digest:
                        raise ValueError('Frozen promotion evidence changed: ' + filename)
            return selected
        evidence = []
        for trial in trials:
            run = training_run(self.plan, trial)
            if validation_result(run, self.plan['training_cohort']['cohort_sha256']) != trial['screening_result']:
                raise ValueError('Screening output differs from its durable Ax objective')
            folder = self.out / 'screening_evidence' / run['id']
            folder.mkdir(parents=True, exist_ok=True)
            pins = {}
            for name in ['summary.json', 'history.json', 'validation_best_predictions.csv', 'best.pt']:
                source = Path(run['out_dir']) / name
                destination = folder / name
                shutil.copy2(source, destination)
                pins[str(destination)] = checksum(destination)
            evidence.append({'ordinal': trial['ordinal'], 'screening_result': trial['screening_result'], 'evidence_sha256': pins})
        frozen_json(path, {'selected_ordinals': selected, 'criterion': 'screen validation F1 descending; ordinal resolves ties',
                           'candidate_evidence': evidence, 'test_read': False})
        return selected

    def work(self):
        for ordinal in range(self.plan['search_policy']['new_trials']):
            self.helper('ask', ordinal)
            trial = self.bundle()['trials'][ordinal]
            self.train(training_run(self.plan, trial), f'ax_train_{ordinal:02d}', 40)
            self.helper('tell', ordinal)
        promoted = self.freeze_promotions()
        results = []
        for ordinal in promoted:
            trial = self.bundle()['trials'][ordinal]
            run = training_run(self.plan, trial)
            result = self.train(run, f'ax_promote_{ordinal:02d}', 100)
            results.append({'ordinal': ordinal, 'parameters': trial['parameters'], **result})
        winner_path = self.out / 'single_configuration_selection.json'
        selected = sorted(results, key=lambda r: (-r['best_validation_f1'], r['ordinal']))[0]
        winner = {**selected, 'cohort_sha256': self.plan['training_cohort']['cohort_sha256'],
                  'selected_on': 'validation_only', 'seed': 42, 'candidate_results': results,
                  'role': 'best_new_challenger_before_incumbent_and_repeat_check',
                  'test_evaluated': False, 'ensemble': False}
        frozen_json(winner_path, winner)
        trial = self.bundle()['trials'][winner['ordinal']]
        baseline_scores = [float(load_json(Path(self.root) / path)['best_validation_f1'])
                           for path in self.plan['historical_baseline']['repeat_summaries']]
        repeat_results = []
        # If no promoted challenger beats the already fixed seed42 incumbent,
        # reuse its existing three-seed evidence instead of repeating old work.
        if winner['best_validation_f1'] > baseline_scores[0] + 1e-6:
            for seed in self.plan['search_policy']['repeat_seeds']:
                run = training_run(self.plan, trial, seed=seed)
                repeat_results.append({'seed': seed, **self.train(run, f'ax_repeat_seed{seed}', 100)})
        scores = [winner['best_validation_f1']] + [x['best_validation_f1'] for x in repeat_results]
        old_kept = retain_incumbent(scores[0], statistics.mean(scores), baseline_scores[0], statistics.mean(baseline_scores))
        selected_scores = baseline_scores if old_kept else scores
        challenger_seed_results = [{'seed': 42, 'best_validation_f1': scores[0]}] + repeat_results
        selected_seed_results = ([{'seed': seed, 'best_validation_f1': score}
                                  for seed, score in zip([42, 43, 44], baseline_scores)] if old_kept else challenger_seed_results)
        selected_parameters = self.plan['historical_baseline']['parameters'] if old_kept else winner['parameters']
        selected_checkpoint = (str(self.root / self.plan['historical_baseline']['checkpoint']) if old_kept else
                               str(self.out / f"ax_trial_{winner['ordinal']:02d}" / 'best.pt'))
        report = {'configuration': selected_parameters, 'selected_trial_ordinal': None if old_kept else winner['ordinal'],
                  'single_model_seed42_checkpoint': selected_checkpoint,
                  'single_model_checkpoint_sha256': checksum(selected_checkpoint),
                  'incumbent_retained': old_kept,
                  'final_selection_rule': 'Replace incumbent only if new seed42 F1 AND three-seed mean F1 both improve by more than1e-6',
                  'search_winner_configuration': winner['parameters'], 'search_winner_trial_ordinal': winner['ordinal'],
                  'seed_results': selected_seed_results,
                  'validation_f1_mean': statistics.mean(selected_scores), 'validation_f1_sample_sd': statistics.stdev(selected_scores),
                  'challenger_seed_results': challenger_seed_results,
                  'challenger_validation_f1_mean': statistics.mean(scores) if len(scores) == 3 else None,
                  'challenger_validation_f1_sample_sd': statistics.stdev(scores) if len(scores) == 3 else None,
                  'challenger_repeats_skipped_because_incumbent_seed42_was_not_beaten': len(scores) == 1,
                  'historical_baseline_seed_scores': baseline_scores,
                  'historical_baseline_mean': statistics.mean(baseline_scores),
                  'historical_baseline_sample_sd': statistics.stdev(baseline_scores),
                  'historical_protocol_comparison_limit': self.plan['historical_baseline']['comparison_limit'],
                  'cohort_sha256': self.plan['training_cohort']['cohort_sha256'],
                  'test_evaluated': False, 'ensemble': False,
                  'old_test_already_consulted': True,
                  'new_independent_final_assessment_required': True}
        frozen_json(self.out / 'validation_result.json', report)
        frozen_json(self.out / 'final_single_model_selection.json', {
            'parameters': selected_parameters, 'checkpoint': selected_checkpoint,
            'checkpoint_sha256': checksum(selected_checkpoint), 'incumbent_retained': old_kept,
            'selection_uses': 'validation_only', 'test_evaluated': False, 'ensemble': False,
            'validation_result_sha256': checksum(self.out / 'validation_result.json')})
        self.current_phase = 'validation_complete'
        self.state.update(status='validation_complete', current_stage=None, finished_at=now())
        self.save()
        frozen_json(self.out / 'ALLOW_INVENTORY.json', {'status': 'done', 'protocol_sha256': self.plan_hash,
                    'model_out_dir': str(self.out.resolve()), 'validation_only': True, 'test_evaluated': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--startup-token')
    parser.add_argument('--ax-step', choices=('ask', 'tell'))
    parser.add_argument('--ordinal', type=int)
    args = parser.parse_args()
    plan = verify_plan(args.plan, args.plan_sha256)
    if args.ax_step:
        if args.ordinal is None or not args.plan_sha256 or args.detach:
            parser.error('Ax helper requires ordinal+plan hash and cannot detach')
        print(json.dumps(ax_step(args.plan, args.plan_sha256, args.ax_step, args.ordinal)))
        return
    out = Path(plan['workspace']) / plan['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        token = uuid.uuid4().hex
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--plan', str(args.plan.resolve()), '--startup-token', token]
        with (out / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=plan['workspace'], stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = load_json(out / 'progress.json')
            if state.get('pid') == process.pid and state.get('startup_token') == token and state.get('status') in {'running', 'validation_complete'} and (state.get('current_stage') or state.get('status') == 'validation_complete'):
                print(json.dumps({'startup_verified': True, 'pid': process.pid, 'progress': str(out / 'progress.json')}))
                return
            if process.poll() is not None:
                raise RuntimeError('Ax search failed to start; inspect supervisor.log')
            time.sleep(.2)
        raise RuntimeError('Ax search did not acknowledge its first stage')
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    queue = AxGeometrySearch(args.plan, args.startup_token)
    awake = None
    try:
        if sys.platform == 'darwin':
            awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        queue.work()
    except BaseException as error:
        queue.state.update(status='stopped' if isinstance(error, InterruptedError) else 'failed', error=repr(error))
        queue.save()
        raise
    finally:
        if awake and awake.poll() is None:
            awake.terminate()


if __name__ == '__main__':
    main()
