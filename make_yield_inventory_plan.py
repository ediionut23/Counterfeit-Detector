#!/usr/bin/env python3
"""Version the throughput repair while preserving the low-heat model run."""
import copy
import json
from pathlib import Path
from run_overnight_research import atomic_json,checksum,now


def main():
    root=Path(__file__).resolve().parent
    old=json.loads((root/'million_inventory_cool_plan.json').read_text())
    state=json.loads((root/old['out_dir']/'progress.json').read_text())
    if state['status'] not in {'failed','stopped'} or not (root/old['out_dir']/'STOP').exists():
        raise ValueError('Old inventory must be stopped with its handoff request retained')
    out='molecule_bank_1m/yield_aware_inventory_20261102'
    if (root/out/'progress.json').exists():raise ValueError('Successor already started; do not replace its pinned plan')
    plan=copy.deepcopy(old)
    command=['/usr/sbin/taskpolicy','-c','utility',str(root/'venv/bin/python'),
        'ensure_yield_bank.py','--db','molecule_bank_1m/bank.sqlite',
        '--config','million_bank_yield_config.json','--workers','1']
    plan.update(out_dir=out,generation_command=command,
        supersedes_plan_sha256=checksum(root/'million_inventory_cool_plan.json'))
    for stage in plan['stages']:
        cmd=stage['command']
        if cmd[:3]==['/usr/sbin/taskpolicy','-c','background']:cmd=cmd[3:]
        stage['command']=['/usr/sbin/taskpolicy','-c','utility']+cmd
    # The outer queue uses only the volume test on restart. This first stage
    # independently fills/validates coverage even when volume already exceeds1M.
    plan['stages'].insert(0,{'id':'ensure_named_category_coverage','command':command})
    plan['resource_policy'].update(priority='utility',generation_compute_children=1,
        hard_native_deadline_seconds=90,training_parameters_changed=False,
        reason='Reduce zero-yield work/native hangs without increasing CPU concurrency or loosening chemistry')
    additions=['expand_yield_aware_bank.py','generation_job_host.py','yield_aware_scheduling.py',
        'million_bank_yield_config.json','ensure_yield_bank.py','make_yield_inventory_plan.py']
    plan['file_pins'].update({p:checksum(root/p) for p in additions})
    atomic_json(root/'million_inventory_yield_plan.json',plan)
    active=json.loads((root/'active_research_run.json').read_text())
    active.update(updated_at=now(),inventory_out_dir=out,inventory_plan='million_inventory_yield_plan.json',
        generation_policy='yield-aware category floors; one chemistry child; hard90s deadline; unchanged thermal gate')
    atomic_json(root/'active_research_run.json',active)
    atomic_json(root/old['out_dir']/'handoff.json',{'updated_at':now(),'successor_out_dir':out,
        'successor_plan':'million_inventory_yield_plan.json','committed_data_preserved':True,
        'reason':'Repair zero-yield allocation and native-code deadline; old worker reaped before coordinator'})
    print(json.dumps({'out_dir':out,'plan':'million_inventory_yield_plan.json','compute_children':1,
                     'chemistry_quality_changed':False,'training_dataset_changed':False}))


if __name__=='__main__':main()
