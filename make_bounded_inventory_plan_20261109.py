#!/usr/bin/env python3
"""Version inventory allocation/cooling repair; preserve original plans and chemistry."""
import copy
import json
from pathlib import Path
from run_overnight_research import atomic_json, checksum


def main():
    root = Path(__file__).resolve().parent
    predecessor = root / 'million_inventory_sequential_plan_20261107_v2.json'
    old = json.loads(predecessor.read_text())
    plan = copy.deepcopy(old)
    new_out = 'molecule_bank_1m/bounded_cooling_inventory_20261109'
    plan.update(out_dir=new_out, supersedes_plan_sha256=checksum(predecessor))
    new_config = 'million_bank_bounded_cool_config_20261109.json'
    for command in [plan['generation_command']] + [stage['command'] for stage in plan['stages']]:
        for index, token in enumerate(command):
            if token == 'ensure_yield_bank.py':
                command[index] = 'ensure_bounded_bank.py'
            elif token == 'million_bank_sequential_cool_config_20261107.json':
                command[index] = new_config
            elif token.startswith(old['out_dir'] + '/'):
                command[index] = new_out + token[len(old['out_dir']):]
    plan['resource_policy'].pop('generation_rest_seconds', None)
    plan['resource_policy'].update(
        generation_minimum_rest_seconds=.05,
        nominal_accumulated_active_budget_seconds=15,
        nominal_cooling_debt_ratio=3,
        fair_accumulated_active_budget_seconds=5,
        fair_cooling_debt_ratio=11,
        native_job_deadline_seconds=90,
        generation_compute_children=1,
        new_generation_job_requires='accumulated cooling debt repaid; pressure gating;120s nominal recovery',
        reason='Bound deficient-category allocation, broaden truthful M paths and accumulate short jobs within the cooling budget')
    plan['scientific_policy'].update(
        chemistry_quality_caps_holdout_and_normalization_unchanged=True,
        named_category_floors_unchanged=True,
        generated_molecular_origin_is_not_confirmed_product_fraud=True)
    files = [new_config, 'bounded_bank_scheduling.py', 'accumulated_generation_cooling.py',
             'expand_bounded_cool_bank.py', 'ensure_bounded_bank.py', 'defer_bounded_inventory.py',
             'make_bounded_inventory_plan_20261109.py']
    plan['file_pins'].update({name: checksum(root / name) for name in files})
    destination = root / 'million_inventory_bounded_plan_20261109.json'
    if destination.exists():
        if json.loads(destination.read_text()) != plan:
            raise ValueError('Refusing to replace an existing versioned inventory plan')
    else:
        atomic_json(destination, plan)
    print(json.dumps({'plan': str(destination), 'sha256': checksum(destination), 'out_dir': new_out}))


if __name__ == '__main__':
    main()
