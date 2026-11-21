#!/usr/bin/env python3
"""Version a lower-load inventory queue without changing chemistry or models."""
import copy
import json
from pathlib import Path
from run_overnight_research import atomic_json, checksum, now


def main():
    root=Path(__file__).resolve().parent
    old=json.loads((root/'million_inventory_plan.json').read_text())
    state=json.loads((root/old['out_dir']/'progress.json').read_text())
    if state['status']!='stopped' or state['stages']:
        raise ValueError('Old inventory must be stopped before any stage was launched')
    out='molecule_bank_1m/overnight_inventory_cool_20261101'
    if (root/out/'progress.json').exists():raise ValueError('Cooling inventory already started; retain its plan')
    plan=copy.deepcopy(old)
    plan.update(out_dir=out,generation_command=['/usr/sbin/taskpolicy','-c','background',
        str(root/'venv/bin/python'),'run_cool_generation.py','--workers','1'],
        supersedes_plan_sha256=checksum(root/'million_inventory_plan.json'),
        resource_policy={'generation_workers':1,'audit_workers':1,'priority':'background',
            'new_generation_job_requires':'nominal thermal state; recovery after30s nominal',
            'generation_rest_seconds':1,'training_parameters_changed':False,
            'reason':'User requested avoiding laptop overheating'})
    for stage in plan['stages']:
        command=stage['command']
        if '--workers' in command:command[command.index('--workers')+1]='1'
        stage['command']=['/usr/sbin/taskpolicy','-c','background']+command
    plan['file_pins']['run_cool_generation.py']=checksum(root/'run_cool_generation.py')
    plan['file_pins']['make_cool_inventory_plan.py']=checksum(root/'make_cool_inventory_plan.py')
    atomic_json(root/'million_inventory_cool_plan.json',plan)
    active=json.loads((root/'active_research_run.json').read_text())
    active.update(updated_at=now(),inventory_out_dir=out,inventory_plan='million_inventory_cool_plan.json',
        thermal_policy='generation1worker; training2threads with automatic thermal duty-cycle guard')
    atomic_json(root/'active_research_run.json',active)
    atomic_json(root/old['out_dir']/'handoff.json',{'updated_at':now(),'successor_out_dir':out,
        'successor_plan':'million_inventory_cool_plan.json','reason':'Reduced CPU load at user request; all batches retained'})
    print(json.dumps({'generation_workers':1,'audit_workers':1,'new_plan':'million_inventory_cool_plan.json','out_dir':out}))


if __name__=='__main__':main()
