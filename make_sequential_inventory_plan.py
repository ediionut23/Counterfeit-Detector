#!/usr/bin/env python3
"""Prepare a versioned, cooler inventory plan without starting it or editing pointers."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from run_overnight_research import atomic_json, checksum


CONFIG = 'million_bank_sequential_cool_config_20261107.json'
PLAN = 'million_inventory_sequential_plan_20261107.json'
OUT = 'molecule_bank_1m/sequential_inventory_20261107'


def build(root):
    root = Path(root).resolve()
    old_config = json.loads((root / 'million_bank_yield_config.json').read_text())
    config = copy.deepcopy(old_config)
    config['execution_policy'].update(
        nominal_rest_seconds=60,
        sustained_nominal_seconds=120,
        thermal_gate='nominal_only;120s_sustained_after_elevated_or_unknown')
    # The complete chemistry, coverage and scheduling policy must stay identical.
    assert {k: v for k, v in config.items() if k != 'execution_policy'} == {
        k: v for k, v in old_config.items() if k != 'execution_policy'}
    assert config['execution_policy']['compute_children'] == 1
    plan = copy.deepcopy(json.loads((root / 'million_inventory_yield_plan.json').read_text()))
    plan.update(out_dir=OUT,
                supersedes_plan_sha256=checksum(root / 'million_inventory_yield_plan.json'))
    for command in [plan['generation_command']] + [stage['command'] for stage in plan['stages']]:
        if '--config' in command:
            command[command.index('--config') + 1] = CONFIG
    plan['resource_policy'].update(
        generation_rest_seconds=60,
        new_generation_job_requires='nominal thermal state; recovery after120s nominal',
        initial_generation_requires='validation_complete + pinned ready marker + exclusive model locks +120s nominal',
        overlap_with_training=False,
        reason='Sequential work after training; one chemistry child with60s interjob rest; unchanged chemical gates')
    plan['scientific_policy']['selection'] = (
        'Inventory only, deferred until all declared model and inference work is done; model scores do not change chemistry')
    return config, plan


def write_new_or_identical(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Refusing to replace a versioned file: ' + str(path))
    else:
        atomic_json(path, value)


def main():
    root = Path(__file__).resolve().parent
    if (root / OUT / 'progress.json').exists():
        raise ValueError('Successor already started; preserve its pinned plan')
    config, plan = build(root)
    write_new_or_identical(root / CONFIG, config)
    plan['file_pins'].update({name: checksum(root / name) for name in (
        CONFIG, 'make_sequential_inventory_plan.py', 'defer_sequential_inventory.py')})
    write_new_or_identical(root / PLAN, plan)
    print(json.dumps({'plan': PLAN, 'plan_sha256': checksum(root / PLAN),
                     'out_dir': OUT, 'config': CONFIG, 'started': False,
                     'changes': {'nominal_rest_seconds': 60, 'sustained_nominal_seconds': 120},
                     'chemistry_and_scheduler_unchanged': True}))


if __name__ == '__main__':
    main()
