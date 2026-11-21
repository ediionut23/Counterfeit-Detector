#!/usr/bin/env python3
"""Read-only live view of the durable generation/training queue."""
import argparse
import json
from pathlib import Path
import time


def read(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=None,help='Default follows active_research_run.json')
    p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    while True:
        active=read('active_research_run.json')
        angular_hpo=active.get('research_phase')=='ax_angular_hpo'
        followup=read(Path(active['followup_out_dir'])/'deferred_followup.json') if active.get('followup_out_dir') else {}
        followup_state=read(Path(active['followup_out_dir'])/'progress.json') if active.get('followup_out_dir') else {}
        root=a.root or Path(active.get('model_out_dir','training_runs/overnight_20261101'))
        state=read(root/'progress.json')
        hold=read('training_runs/manual_cooling_hold.json')
        cooling_hold=hold.get('status')=='paused'
        inventory={};deferred={}
        bank=read('molecule_bank_1m/progress.json')
        count=bank.get('unique_molecules',0)
        print(time.strftime('%Y-%m-%d %H:%M:%S'),'MANUAL COOLING HOLD' if cooling_hold else state.get('status','preparing'),
              '| bank',f'{count:,}', '/',f"{bank.get('target_unique_molecules',1000000):,}",
              '| stage',state.get('current_stage','not started'),flush=True)
        if active.get('inventory_out_dir'):
            inventory=read(Path(active['inventory_out_dir'])/'progress.json')
            deferred=read(Path(active['inventory_out_dir'])/'deferred_waiter.json')
            inventory_status=('paused' if cooling_hold else
                inventory.get('status','starting') if deferred.get('inventory_started') else
                'deferred; no generation during training' if active.get('inventory_phase') in {'deferred_until_models_finished','deferred_until_ax_hpo_finished'} else
                inventory.get('status','preparing'))
            print(' Inventory:',inventory_status,inventory.get('current_stage','pending'),flush=True)
            if deferred.get('status')=='waiting':print(' Inventory wait:',deferred.get('reason','pending'),flush=True)
        if cooling_hold:
            print(' Training and generation suspended; automatic resume disabled. Thermal guard snapshot is stale.',flush=True)
        elif deferred.get('inventory_started') and inventory.get('status')=='running':
            stage=inventory.get('stages',{}).get(inventory.get('current_stage'),{})
            command=stage.get('command',[])
            if 'cool_bank_stage.py' in command and '--state' in command:
                thermal=read(command[command.index('--state')+1])
                print(' Thermal: cooled bank stage |',thermal.get('status','starting'),
                      '| pressure',thermal.get('thermal_state','pending'),flush=True)
            else:
                try:host_pid=int(Path('molecule_bank_1m/bank.generation.lock').read_text().strip())
                except (OSError,ValueError):host_pid=None
                thermal=read(Path('molecule_bank_1m/thermal_control_yield')/('yield_aware_host_'+str(host_pid)+'.json'))
                print(' Thermal: generation |',thermal.get('action','initializing'),
                      '| pressure',thermal.get('thermal_state','pending'),
                      '| cooling debt seconds',round(thermal.get('cooling_debt_seconds',0),1),flush=True)
        elif active.get('thermal_guard_state'):
            thermal_path=(Path(active['followup_out_dir'])/'cooling_guard.json'
                          if followup.get('status')=='finalization_running' else active['thermal_guard_state'])
            thermal=read(thermal_path)
            print(' Thermal:',thermal.get('thermal_name','unknown'),'|',thermal.get('status','preparing'),
                  '| paused workloads',len(thermal.get('paused',{})),flush=True)
        for name,run in state.get('experiments',{}).items():
            if run:
                print(' ',name,'epoch',run.get('current_epoch',run.get('epoch','?')),
                      'best validation F1',run.get('best_validation_f1','pending'),
                      'state',run.get('state',run.get('completed','running')),flush=True)
        if angular_hpo:
            trials=state.get('ax_trials',[])
            print(' Ax angular single-model search:',sum(bool(t.get('told')) for t in trials),
                  '/ 8 completed configurations | phase',state.get('phase','initializing'),flush=True)
            pending=next((trial for trial in trials if not trial.get('told')),None)
            if pending:print(' Current Ax parameters:',pending.get('parameters'),flush=True)
            result=read(root/'validation_result.json')
            if result:print(' Selected single-model validation F1 mean:',result.get('validation_f1_mean'),
                            '| incumbent retained',result.get('incumbent_retained'),flush=True)
        for family in ('gatv2','geometric'):
            result=read(root/(family+'_result.json'))
            if result:
                print(' ',family,'locked test',result.get('test_metrics'),flush=True)
        if followup and not angular_hpo:
            print(' Follow-up:',followup.get('status'),followup_state.get('current_stage') or followup.get('reason'),flush=True)
        if active.get('ensemble_out_dir'):
            ensemble_root=Path(active['ensemble_out_dir'])
            final=read(ensemble_root/'final_test'/'test_result.json')
            selected=read(ensemble_root/'final_selection'/'selection.json')
            exploratory=read(ensemble_root/'exploratory_selection'/'selection.json')
            if final:print(' Historical ensemble locked test:' if angular_hpo else ' Heterogeneous locked test:',final.get('test_metrics'),flush=True)
            elif selected:print(' Heterogeneous selected validation F1:',selected.get('validation_metrics',{}).get('f1'),flush=True)
            elif exploratory:print(' Heterogeneous exploratory validation F1:',exploratory.get('validation_metrics',{}).get('f1'),'| final test reserved',flush=True)
        if state.get('error'):print(' Error:',state['error'],flush=True)
        if not a.watch:break
        try:time.sleep(5)
        except KeyboardInterrupt:break


if __name__=='__main__':main()
