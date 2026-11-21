#!/usr/bin/env python3
"""Prepare the same bounded inventory after the angular Ax campaign finishes.

Does not launch, signal, edit pointers or change molecular generation settings.
The caller supplies the final immutable HPO plan; its hash is the readiness pin.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from run_overnight_research import atomic_json, checksum


EXPECTED_MODEL_OUT = 'training_runs/ax_angular_20261109'
INVENTORY_OUT = 'molecule_bank_1m/after_ax_angular_inventory_20261109'
DESTINATION = 'million_inventory_after_ax_plan_20261109.json'


def build(root, hpo_plan_path, predecessor_path=None):
    root = Path(root).resolve()
    hpo_plan_path = Path(hpo_plan_path).resolve()
    predecessor_path = (Path(predecessor_path).resolve() if predecessor_path is not None else
                        root / 'million_inventory_bounded_plan_20261109.json')
    if not hpo_plan_path.is_relative_to(root):
        raise ValueError('HPO plan must be a workspace file')
    old = json.loads(predecessor_path.read_text())
    hpo = json.loads(hpo_plan_path.read_text())
    if Path(old['workspace']).resolve() != root or Path(hpo['workspace']).resolve() != root:
        raise ValueError('Inventory and HPO workspaces differ')
    if hpo.get('out_dir') != EXPECTED_MODEL_OUT:
        raise ValueError('After-Ax inventory requires the declared angular HPO output directory')
    if old.get('families'):
        raise ValueError('Predecessor must contain inventory only')
    old_cohort = old.get('training_cohort', {}).get('cohort_sha256')
    if old_cohort and hpo.get('training_cohort', {}).get('cohort_sha256') != old_cohort:
        raise ValueError('HPO cohort differs from the existing full-data cohort')
    for filename, expected in old.get('file_pins', {}).items():
        if checksum(root / filename) != expected:
            raise ValueError('Pinned inventory source/input changed: ' + filename)
    plan = copy.deepcopy(old)
    plan.update(out_dir=INVENTORY_OUT, supersedes_plan_sha256=checksum(predecessor_path))
    old_relative, old_absolute = old['out_dir'] + '/', str(root / old['out_dir']) + '/'
    for command in [plan['generation_command']] + [stage['command'] for stage in plan['stages']]:
        for index, token in enumerate(command):
            if token.startswith(old_relative):
                command[index] = INVENTORY_OUT + '/' + token[len(old_relative):]
            elif token.startswith(old_absolute):
                command[index] = str(root / INVENTORY_OUT) + '/' + token[len(old_absolute):]
    if plan['generation_command'] != old['generation_command']:
        raise ValueError('Deferred continuation must preserve the exact generation command/configuration')
    hpo_relative = str(hpo_plan_path.relative_to(root))
    plan['after_ax_dependency'] = {
        'model_plan': hpo_relative,
        'model_plan_sha256': checksum(hpo_plan_path),
        'model_progress': EXPECTED_MODEL_OUT + '/progress.json',
        'ready_marker': EXPECTED_MODEL_OUT + '/ALLOW_INVENTORY.json',
        'protocol_sha256': checksum(hpo_plan_path),
        'required_model_status': 'validation_complete',
        'model_test_evaluation_required': False,
    }
    plan.setdefault('resource_policy', {}).update(
        initial_generation_requires='angular Ax validation_complete + exact HPO ready marker + exclusive model locks +120s nominal',
        overlap_with_training=False)
    plan.setdefault('scientific_policy', {}).update(
        generation_configuration_and_chemistry_unchanged=True,
        selection='Inventory resumes only after the declared validation-only angular Ax campaign; no model score changes molecule labels')
    plan.setdefault('file_pins', {}).update({name: checksum(root / name) for name in (
        'make_after_ax_inventory_plan_20261109.py', 'thermal_guard_ax.py', hpo_relative)})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hpo-plan', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = build(root, args.hpo_plan)
    path = root / DESTINATION
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('Refusing to replace a versioned after-Ax plan')
    else:
        atomic_json(path, plan)
    print(json.dumps({'plan': str(path), 'plan_sha256': checksum(path),
                      'out_dir': plan['out_dir'], 'dependency': plan['after_ax_dependency'],
                      'production_started': False}))


if __name__ == '__main__':
    main()
