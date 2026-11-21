#!/usr/bin/env python3
"""Create an immutable, validation-only Ax campaign for one angular GNN."""
import argparse
import json
from pathlib import Path

from run_ax_geometry_search import PARAMETER_SPACE
from run_overnight_research import atomic_json, checksum


def build(root, runtime_manifest, runtime_directory, out_dir):
    original = json.loads((root / 'large_model_search_plan.json').read_text())
    cohort = original['training_cohort']
    baseline_root = Path('training_runs/large_search_20261101')
    base = baseline_root / 'geometric_angular'
    template = [str(root / 'venv/bin/python'), 'train_geometric_variants.py',
        '--root', 'balanced_datasets/training_200k/3d',
        '--graph-index', cohort['directory'] + '/graph_index.csv',
        '--cache-root', 'training_cache/geometric_overnight_v2',
        '--out', str(root / out_dir / 'guard_template'),
        '--geometry-variant', 'angular', '--radial-basis', '16', '--angular-basis', '6',
        '--cutoff', '4.5', '--max-neighbors', '24', '--hidden', '64', '--interactions', '3',
        '--dropout', '0.1', '--lr', '0.001', '--weight-decay', '0.00001',
        '--batch-size', '64', '--epochs', '40', '--patience', '15', '--scheduler-patience', '3',
        '--seed', '42', '--threads', '2', '--num-workers', '0', '--device', 'cpu',
        '--weight-mode', 'none', '--selection', 'val_f1', '--no-test',
        '--test-lock', str(root / out_dir / 'UNUSED_test_lock.json')]
    names = ['run_ax_geometry_search.py', 'make_ax_geometry_search_plan.py', 'run_overnight_research.py',
             'enforce_large_training_cohort.py', 'audit_overnight_training_split.py',
             'train_geometric_variants.py', 'train_geometric_gnn.py', 'geometric_angular_model.py',
             'thermal_guard_ax.py', 'thermal_guard_cool_v2.py', 'thermal_guard.py',
             cohort['directory'] + '/dataset.csv', cohort['directory'] + '/graph_index.csv',
             cohort['directory'] + '/manifest.json', str(runtime_manifest)]
    assessment_policy = 'fresh_holdout_policy_20261109.json'
    if (root / assessment_policy).exists():
        names.append(assessment_policy)
    repeats = []
    for name in ['geometric_angular', 'geometric_champion_seed43', 'geometric_champion_seed44']:
        folder = baseline_root / name
        for filename in ['configuration.json', 'history.json', 'summary.json', 'best.pt']:
            names.append(str(folder / filename))
        repeats.append(str(folder / 'summary.json'))
    plan = {'schema_version': 1, 'protocol': 'ax_angular_validation_only_v1', 'test_evaluation': False,
            'workspace': str(root), 'out_dir': out_dir, 'experiment_name': 'angular_gnn_ax_20261109_v1',
            'bank_db': 'molecule_bank_1m/bank.sqlite', 'target_unique_molecules': 1000000, 'target_f1': .85,
            'families': [], 'stages': [], 'training_cohort': cohort,
            'file_pins': {name: checksum(root / name) for name in names},
            'ax_runtime': {'version': '1.3.1', 'directory': str(runtime_directory),
                           'manifest': str(runtime_manifest), 'manifest_sha256': checksum(root / runtime_manifest)},
            'parameter_space': PARAMETER_SPACE, 'template_command': template,
            'search_policy': {'new_trials': 8, 'sobol_trials': 4, 'screen_epochs': 40,
                              'promote_count': 2, 'final_epochs': 100, 'patience': 15,
                              'training_seed': 42, 'repeat_seeds': [43, 44], 'ax_seed': 42,
                              'selection_metric': 'validation_f1'},
            'historical_baseline': {'history': str(base / 'history.json'),
                'configuration': str(base / 'configuration.json'), 'checkpoint': str(base / 'best.pt'),
                'parameters': {'lr': .001, 'dropout': .1, 'weight_decay': 1e-5,
                               'hidden': 64, 'interactions': 3, 'angular_basis': 6},
                'repeat_summaries': repeats, 'warm_observation_budget': 40,
                'baseline_patience': 12, 'new_patience': 15,
                'comparison_limit': 'Historical protocol differs in patience; comparison is not a pure causal hyperparameter estimate'},
            'scientific_policy': {'objective': 'best validation F1 within equal40-epoch screening budget or early stop',
                'final_candidate': 'one configuration and one seed42 model; no ensemble',
                'incumbent_retention': 'Keep baseline unless challenger improves both seed42 F1 and the three-seed mean by more than1e-6',
                'fresh_holdout_policy': assessment_policy,
                'old_test_already_consulted': True, 'new_independent_assessment_required': True,
                'test_predictions_read_during_search': False},
            'resource_policy': {'generation_during_training': False, 'trainer_threads': 2,
                'ax_helper_threads': 1, 'compute_seconds': 15, 'rest_seconds': 45,
                'serial_stage_scripts': ['train_geometric_variants.py', 'run_ax_geometry_search.py']}}
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-manifest', type=Path, default=Path('.hpo_runtime/ax_1_3_1/runtime_manifest.json'))
    parser.add_argument('--runtime-directory', type=Path, default=Path('.hpo_runtime/ax_1_3_1/site-packages'))
    parser.add_argument('--out-dir', default='training_runs/ax_angular_20261109')
    parser.add_argument('--plan', type=Path, default=Path('ax_angular_search_plan_20261109.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = build(root, args.runtime_manifest, args.runtime_directory, args.out_dir)
    if args.plan.exists():
        if json.loads(args.plan.read_text()) != plan:
            raise ValueError('Refusing to replace an existing versioned Ax plan')
    else:
        atomic_json(args.plan, plan)
    print(json.dumps({'plan': str(args.plan), 'sha256': checksum(args.plan), 'out_dir': args.out_dir}))


if __name__ == '__main__':
    main()
