#!/usr/bin/env python3
"""Crash/restart, budget and no-test checks; no molecular model is trained."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_ax_geometry_search as runner
from run_overnight_research import atomic_json, checksum


BASE = {'lr': .001, 'dropout': .1, 'weight_decay': 1e-5,
        'hidden': 64, 'interactions': 3, 'angular_basis': 6}


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeClient:
    asks = 0
    def __init__(self, **kwargs):
        self.trials = []
    def configure_experiment(self, **kwargs):
        self.experiment = kwargs
    def configure_optimization(self, **kwargs):
        self.optimization = kwargs
    def configure_generation_strategy(self, **kwargs):
        self.strategy = kwargs
    def attach_trial(self, parameters, **kwargs):
        self.trials.append({'parameters': parameters, 'complete': False})
        return len(self.trials) - 1
    def get_next_trials(self, max_trials):
        FakeClient.asks += 1
        parameters = {**BASE, 'lr': .0004 + .00001 * len(self.trials)}
        index = self.attach_trial(parameters)
        return {index: parameters}
    def complete_trial(self, trial_index, raw_data):
        if self.trials[trial_index]['complete']:
            raise AssertionError('Duplicate tell on durable client')
        self.trials[trial_index].update(complete=True, data=raw_data)
    def save_to_json_file(self, path):
        Path(path).write_text(json.dumps({'trials': self.trials}))
    @classmethod
    def load_from_json_file(cls, path):
        result = cls()
        result.trials = json.loads(Path(path).read_text())['trials']
        return result


class DurableAxChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        history = self.root / 'baseline_history.json'
        atomic_json(history, [{'epoch': i, 'val_f1': .99 if i == 47 else .7 + min(i, 40) / 1000} for i in range(1, 60)])
        self.plan = {'workspace': str(self.root), 'out_dir': 'campaign',
            'experiment_name': 'fixture', 'training_cohort': {'cohort_sha256': 'cohort'},
            'historical_baseline': {'history': str(history), 'parameters': BASE},
            'search_policy': {'screen_epochs': 40, 'ax_seed': 42, 'new_trials': 8},
            'template_command': ['python', 'train_geometric_variants.py', '--out', 'unused', '--seed', '42', '--no-test']}
        self.plan_path = self.root / 'plan.json'
        atomic_json(self.plan_path, self.plan)
        self.plan_sha = checksum(self.plan_path)
        self.api = (FakeClient, FakeConfig, FakeConfig)
        self.patches = [patch.object(runner, 'verify_plan', return_value=self.plan),
                        patch.object(runner, 'ax_api', return_value=self.api)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        FakeClient.asks = 0

    def ask(self, ordinal=0):
        return runner.ax_step(self.plan_path, self.plan_sha, 'ask', ordinal)

    def tell(self, ordinal=0):
        return runner.ax_step(self.plan_path, self.plan_sha, 'tell', ordinal)

    def complete_fixture_training(self, trial, score=.8, epoch=40):
        run = runner.training_run(self.plan, trial)
        directory = Path(run['out_dir'])
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / 'summary.json', {'completed': True, 'cohort_sha256': 'cohort',
            'test_evaluated': False, 'best_validation_f1': score, 'best_epoch': min(epoch, 30), 'current_epoch': epoch})
        for name in ('best.pt', 'last.pt'):
            (directory / name).write_bytes(b'fixture checkpoint')

    def test_pending_ask_retry_does_not_ask_or_import_again(self):
        first = self.ask()
        with patch.object(runner, 'ax_api', side_effect=AssertionError('Should not reimport Ax')):
            second = self.ask()
        self.assertEqual(first, second)
        self.assertEqual(FakeClient.asks, 1)
        with self.assertRaisesRegex(ValueError, 'pending'):
            self.ask(1)

    def test_tell_retry_is_idempotent_and_atomic(self):
        trial = self.ask()
        self.complete_fixture_training(trial)
        original = runner.atomic_json
        with patch.object(runner, 'atomic_json', side_effect=RuntimeError('crash before atomic publication')):
            with self.assertRaises(RuntimeError):
                self.tell()
        before = json.loads((self.root / 'campaign/ax_bundle.json').read_text())
        self.assertFalse(before['trials'][0]['told'])
        result = self.tell()
        self.assertTrue(result['told'])
        with patch.object(runner, 'ax_api', side_effect=AssertionError('Should not tell twice')):
            self.assertEqual(self.tell(), result)

    def test_warm_observation_uses_only_first40_epochs(self):
        self.ask()
        bundle = json.loads((self.root / 'campaign/ax_bundle.json').read_text())
        self.assertAlmostEqual(bundle['historical_warm_observation']['score'], .74)
        self.assertTrue(bundle['historical_warm_observation']['not_counted_among_eight_new_trials'])

    def test_tell_rejects_larger_budget_and_test_access(self):
        trial = self.ask()
        self.complete_fixture_training(trial, epoch=41)
        with self.assertRaisesRegex(ValueError, 'screening epoch budget'):
            self.tell()
        self.complete_fixture_training(trial)
        p = self.root / 'campaign/ax_trial_00/summary.json'
        data = json.loads(p.read_text())
        data['test_evaluated'] = True
        atomic_json(p, data)
        with self.assertRaisesRegex(ValueError, 'validation-only'):
            self.tell()

    def test_corrupt_snapshot_or_ledger_rejected_even_on_fast_retry(self):
        self.ask()
        p = self.root / 'campaign/ax_bundle.json'
        original = json.loads(p.read_text())
        damaged = copy.deepcopy(original)
        damaged['ax_snapshot_json'] += ' '
        atomic_json(p, damaged)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.ask()
        damaged = copy.deepcopy(original)
        damaged['trials'][0]['parameters']['hidden'] = 96
        atomic_json(p, damaged)
        with self.assertRaisesRegex(ValueError, 'ledger checksum'):
            self.ask()

    def test_earlystopped_run_is_valid_equal_maximum_budget(self):
        trial = self.ask()
        self.complete_fixture_training(trial, epoch=24)
        self.assertTrue(self.tell()['told'])

    def test_parameter_domain_and_incumbent_retention(self):
        for parameters in [{**BASE, 'hidden': 65}, {**BASE, 'lr': float('nan')}, {**BASE, 'dropout': .5}]:
            with self.assertRaises(ValueError):
                runner.normalized_parameters(parameters)
        self.assertTrue(runner.retain_incumbent(.80, .80, .818, .814))
        self.assertTrue(runner.retain_incumbent(.83, .81, .818, .814))
        self.assertFalse(runner.retain_incumbent(.83, .82, .818, .814))

    def test_completed_screen_replay_uses_frozen_result_during_partial_promotion(self):
        trial = self.ask()
        self.complete_fixture_training(trial)
        told = self.tell()
        run = runner.training_run(self.plan, trial)
        directory = Path(run['out_dir'])
        evidence = self.root / 'screen_summary.json'
        evidence.write_bytes((directory/'summary.json').read_bytes())
        atomic_json(self.root/'campaign/promotion.json', {'candidate_evidence':[
            {'ordinal':0,'evidence_sha256':{str(evidence):checksum(evidence)}}]})
        atomic_json(directory/'summary.json', {'completed':False,'current_epoch':55})
        queue = object.__new__(runner.AxGeometrySearch)
        queue.plan,self.plan['training_cohort']['directory'] = self.plan,str(self.root/'cohort')
        queue.plan_hash,self.plan_sha = self.plan_sha,self.plan_sha
        queue.out = self.root/'campaign'
        queue.bundle_path = queue.out/'ax_bundle.json'
        queue.dynamic_runs = {}
        queue.state = {'stages':{'ax_train_00':{'status':'complete'},'ax_promote_00':{'status':'failed'}}}
        queue.verify_stage = lambda *args:None
        calls = []
        def run_stage(key,command,resume_dir=None):
            calls.append((key,command,resume_dir))
            if key=='ax_promote_00':self.complete_fixture_training(trial,score=.83,epoch=100)
        queue.run = run_stage
        self.assertEqual(queue.train(run,'ax_train_00',40),told['screening_result'])
        self.assertAlmostEqual(queue.train(run,'ax_promote_00',100)['best_validation_f1'],.83)
        self.assertEqual([x[0] for x in calls],['ax_train_00','ax_promote_00'])
        self.assertEqual(calls[1][2],run['out_dir'])
        self.assertEqual(FakeClient.asks,1)

    def campaign(self, incumbent):
        summaries = []
        for seed, score in zip([42, 43, 44], [incumbent, incumbent-.001, incumbent-.002]):
            path = self.root / f'baseline_{seed}.json'
            atomic_json(path, {'best_validation_f1': score})
            summaries.append(str(path))
        incumbent_checkpoint = self.root / 'baseline.pt'
        incumbent_checkpoint.write_bytes(b'incumbent')
        self.plan.update(file_pins={}, families=[], target_unique_molecules=1000000,
                         target_f1=.85, bank_db='unused.sqlite')
        self.plan['training_cohort']['directory'] = str(self.root / 'cohort')
        self.plan['search_policy'].update(repeat_seeds=[43,44], final_epochs=100)
        self.plan['historical_baseline'].update(repeat_summaries=summaries, checkpoint=str(incumbent_checkpoint),
                                                comparison_limit='fixture patience differs')
        atomic_json(self.plan_path, self.plan)
        self.plan_sha = checksum(self.plan_path)
        queue = runner.AxGeometrySearch(self.plan_path)
        calls = []
        def run(key, command, resume_dir=None):
            if queue.state['stages'].get(key, {}).get('status') == 'complete':
                return
            calls.append((key, list(command)))
            if command[1] == 'run_ax_geometry_search.py':
                mode = command[command.index('--ax-step')+1]
                ordinal = int(command[command.index('--ordinal')+1])
                runner.ax_step(self.plan_path, self.plan_sha, mode, ordinal)
            else:
                directory = Path(command[command.index('--out')+1])
                directory.mkdir(parents=True, exist_ok=True)
                ordinal = int(directory.name[-2:]) if directory.name.startswith('ax_trial_') else 7
                seed = int(command[command.index('--seed')+1])
                epochs = int(command[command.index('--epochs')+1])
                score = .79 + .002*ordinal + (.025 if epochs==100 else 0) - .001*(seed-42)
                atomic_json(directory/'summary.json', {'completed':True,'cohort_sha256':'cohort','test_evaluated':False,
                    'best_validation_f1':score,'best_epoch':epochs,'current_epoch':epochs})
                atomic_json(directory/'history.json', [{'epoch':epochs,'val_f1':score}])
                (directory/'validation_best_predictions.csv').write_text('fixture\n')
                for filename in ['best.pt','last.pt']:(directory/filename).write_text(str(score))
            queue.state['stages'][key] = {'status':'complete'}
        queue.run = run
        with patch.object(runner, 'cohort_audit', return_value={'passed':True,'actual_train_unique_ids':161157}):
            queue.work()
            first_count = len(calls)
            queue.work()
        self.assertEqual(first_count,len(calls))
        self.assertEqual(FakeClient.asks,8)
        self.assertEqual(queue.state['status'],'validation_complete')
        self.assertTrue((queue.out/'ALLOW_INVENTORY.json').exists())
        self.assertFalse(any('--evaluate-test' in command for _,command in calls))
        return queue,calls

    def test_full_mock_campaign_retains_budget_promotes_and_repeats_without_test(self):
        queue,calls = self.campaign(.8)
        self.assertEqual(sum(key.startswith('ax_train_') for key,_ in calls),8)
        self.assertEqual(sum(key.startswith('ax_promote_') for key,_ in calls),2)
        self.assertEqual(sum(key.startswith('ax_repeat_') for key,_ in calls),2)
        self.assertEqual(len(queue.state['adaptive_run_dirs']),10)
        result = json.loads((queue.out/'validation_result.json').read_text())
        self.assertFalse(result['incumbent_retained'])
        self.assertEqual(len(result['seed_results']),3)
        self.assertEqual(json.loads((queue.out/'promotion.json').read_text())['selected_ordinals'],[7,6])

    def test_worse_campaign_reuses_incumbent_stats_without_retraining_it(self):
        queue,calls = self.campaign(.85)
        self.assertFalse(any(key.startswith('ax_repeat_') for key,_ in calls))
        result = json.loads((queue.out/'validation_result.json').read_text())
        self.assertTrue(result['incumbent_retained'])
        self.assertEqual(result['configuration'],BASE)
        self.assertAlmostEqual(result['validation_f1_mean'],.849)
        self.assertIsNone(result['challenger_validation_f1_mean'])
        self.assertTrue(result['challenger_repeats_skipped_because_incumbent_seed42_was_not_beaten'])


def real_ax_smoke(runtime_directory):
    """Eight virtual observations, not molecular training, exercise4Sobol→4BO."""
    import sys
    sys.path.insert(0, str(Path(runtime_directory).resolve()))
    from ax.api.client import Client
    from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
    import torch
    torch.set_num_threads(1)
    client = Client(random_seed=42)
    parameters = []
    for name, definition in runner.PARAMETER_SPACE.items():
        if definition['kind'] == 'choice':
            parameters.append(ChoiceParameterConfig(name=name, values=definition['values'], parameter_type='int', is_ordered=True))
        else:
            parameters.append(RangeParameterConfig(name=name, bounds=tuple(definition['bounds']), parameter_type='float', scaling='log' if definition['log'] else 'linear'))
    client.configure_experiment(parameters=parameters, name='temporary_synthetic_api_check')
    client.configure_optimization(objective='validation_f1')
    index = client.attach_trial(parameters=BASE)
    client.complete_trial(trial_index=index, raw_data={'validation_f1': .8})
    client.configure_generation_strategy(method='fast', initialization_budget=4, initialization_random_seed=42,
        initialize_with_center=False, use_existing_trials_for_initialization=False,
        min_observed_initialization_trials=4, allow_exceeding_initialization_budget=False, torch_device='cpu')
    nodes = []
    for ordinal in range(8):
        torch.manual_seed(42 + ordinal)
        trial_index, parameters = next(iter(client.get_next_trials(max_trials=1).items()))
        runner.normalized_parameters(parameters)
        # Private inspection is confined to the smoke check of this pinned
        # version; production orchestration uses only public Client APIs.
        trial = client._experiment.trials[trial_index]
        node = trial.generator_run._generation_node_name
        nodes.append(node)
        value = .75 + .03 * (1 - abs(float(parameters['dropout']) - .16)) + .001 * ordinal
        client.complete_trial(trial_index=trial_index, raw_data={'validation_f1': value})
        snapshot = runner.client_snapshot(client)
        client = runner.restore_client(Client, snapshot)
    if len(set(nodes[:4])) != 1 or nodes[4] == nodes[0] or len(set(nodes[4:])) != 1:
        raise AssertionError('Expected exactly4Sobol then4Bayesian nodes: ' + str(nodes))
    print(json.dumps({'real_ax_api_check': True, 'synthetic_observations': 8,
                      'generation_nodes': nodes, 'molecular_training_or_test_access': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-ax-runtime', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DurableAxChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.real_ax_runtime:
        real_ax_smoke(args.real_ax_runtime)
