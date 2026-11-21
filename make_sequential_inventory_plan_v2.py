#!/usr/bin/env python3
"""Prepare a new plan that cools all downstream stages as well as generation."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from run_overnight_research import atomic_json, checksum
from make_sequential_inventory_plan import write_new_or_identical


PLAN = 'million_inventory_sequential_plan_20261107_v2.json'
OUT = 'molecule_bank_1m/sequential_inventory_20261107_v2'


def build(root):
    root = Path(root).resolve()
    plan = copy.deepcopy(json.loads((root / 'million_inventory_sequential_plan_20261107.json').read_text()))
    plan.update(out_dir=OUT,
                supersedes_plan_sha256=checksum(root / 'million_inventory_sequential_plan_20261107.json'))
    for stage in plan['stages']:
        if stage['id'] == 'ensure_named_category_coverage':
            continue
        stage['command'] = [str(root / 'venv/bin/python'), 'cool_bank_stage.py',
                            '--state', OUT + '/cooling/' + stage['id'] + '.json', '--'] + stage['command']
    plan['resource_policy'].update(
        downstream_stages_nominal_duty='15s compute/45s rest',
        downstream_stages_fair_or_unknown_duty='5s compute/55s rest',
        downstream_stages_recovery_seconds=120,
        downstream_subprocess_workers_pause_together=True)
    plan['file_pins'].update({name: checksum(root / name) for name in (
        'cool_bank_stage.py', 'thermal_guard.py', 'make_sequential_inventory_plan_v2.py')})
    return plan


def main():
    root = Path(__file__).resolve().parent
    if (root / OUT / 'progress.json').exists():
        raise ValueError('Successor already started; preserve its pinned plan')
    plan = build(root)
    write_new_or_identical(root / PLAN, plan)
    print(json.dumps({'plan': PLAN, 'plan_sha256': checksum(root / PLAN),
                     'out_dir': OUT, 'started': False, 'generation_configuration_unchanged': True,
                     'downstream_duty': '15s compute/45s rest'}))


if __name__ == '__main__':
    main()
