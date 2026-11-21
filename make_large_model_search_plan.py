#!/usr/bin/env python3
"""Version the authorized expanded search and independent inventory queue."""
import copy
import json
from pathlib import Path
from run_overnight_research import atomic_json, checksum, now
from run_large_model_search import argument


def main():
    root=Path(__file__).resolve().parent
    old=json.loads((root/'overnight_research_plan.json').read_text())
    previous=json.loads((root/old['out_dir']/'progress.json').read_text())
    if previous['status']!='stopped' or previous['stages']:
        raise ValueError('Expected untouched stopped predecessor; inspect before migrating')
    out='training_runs/large_search_20261101'
    inventory_out='molecule_bank_1m/overnight_inventory_20261101'
    if (root/out/'progress.json').exists() or (root/inventory_out/'progress.json').exists():
        raise ValueError('Versioned successors already started; do not overwrite their plans')
    plan=copy.deepcopy(old)
    plan.update(out_dir=out,stages=[],search_policy={'screen_epochs':20,'promote_per_family':2,
        'final_epochs':100,'repeat_seeds':[43,44],'training_minimum':150000,
        'screening_changes_only_epoch_budget':True,'interleave_model_families':True,
        'inventory_runs_independently':True},supersedes_plan_sha256=checksum('overnight_research_plan.json'))
    gatbase=old['families'][0]['runs'][0]['command']
    geobase=[r['command'] for r in old['families'][1]['runs']]
    gat=[];geo=[]

    def record(ident,command,family):
        dest=out+'/'+ident
        command=argument(command,'--out-dir' if family=='gatv2' else '--out',dest)
        command=argument(command,'--epochs',20)
        command=argument(command,'--patience',12)
        command=argument(command,'--seed',42)
        command=argument(command,'--device','cpu')
        command=argument(command,'--test-lock',out+'/'+family+'_member_test_lock_UNUSED.json')
        return {'id':ident,'out_dir':dest,'command':command}

    for i,run in enumerate(old['families'][0]['runs']):
        gat.append(record(['gatv2_base','gatv2_base_weighted','gatv2_base_deep'][i],list(run['command']),'gatv2'))
    variants=[
        ('gatv2_jk',128,3,4,'jk','mean_sum','layer',.001,'none',.1,.1,.2),
        ('gatv2_attention_weighted',128,4,8,'attention','mean_sum','layer',.0007,'row',.1,.1,.2),
        ('gatv2_jk_attention',128,4,4,'jk','attention','batch',.0005,'none',.15,.05,.2),
        ('gatv2_max_weighted',96,3,4,'mean_max','mean_sum','layer',.001,'row',.1,.1,.2),
        ('gatv2_jk_wide',192,4,8,'jk','mean_max','layer',.0005,'none',.1,.1,.25)]
    for ident,hidden,layers,heads,readout,jk,norm,lr,weight,feature_drop,attention_drop,readout_drop in variants:
        command=list(gatbase);command[1]='train_gatv2_search.py'
        for flag,value in [('--hidden',hidden),('--layers',layers),('--heads',heads),('--dropout',.15),
            ('--readout',readout),('--jk-pool',jk),('--norm',norm),('--lr',lr),('--weight-decay',.0001),
            ('--weight-mode',weight),('--input-dropout',.05),('--feature-dropout',feature_drop),
            ('--attention-dropout',attention_drop),('--readout-dropout',readout_drop)]:
            command=argument(command,flag,value)
        gat.append(record(ident,command,'gatv2'))
    for i,command in enumerate(geobase):
        geo.append(record(['geometric_distance','geometric_distance_weighted'][i],list(command),'geometric'))
    for i,base in enumerate(geobase):
        command=list(base);command[1]='train_geometric_variants.py'
        command=argument(command,'--geometry-variant','angular')
        command=argument(command,'--angular-basis',6)
        geo.append(record(['geometric_angular','geometric_angular_weighted'][i],command,'geometric'))
    plan['families']=[{'id':'gatv2','runs':gat},{'id':'geometric','runs':geo}]
    plan['scientific_policy'].update(selection='12 full-data screens; top2 per family promoted; champions repeated at seeds43/44; singletons and equal-seed ensemble selected on validation; one locked test per family',
        training_size_constraint='Every production run: exactly161157 train molecules,191701 total; no data subsampling',
        inventory_parallelism='generation6workers + at most one trainer2threads; model cohort already frozen')
    extra=['run_large_model_search.py','make_large_model_search_plan.py','enforce_large_training_cohort.py',
           'select_validation_ensemble.py','evaluate_selected_ensemble.py','train_gatv2_search.py',
           'train_geometric_variants.py','geometric_angular_model.py']
    plan['file_pins'].update({p:checksum(root/p) for p in extra})
    atomic_json(root/'large_model_search_plan.json',plan)
    inventory=copy.deepcopy(old)
    inventory.update(out_dir=inventory_out,families=[],supersedes_plan_sha256=checksum('overnight_research_plan.json'))
    inventory['scientific_policy']['selection']='Inventory pipeline only; model research runs under large_model_search_plan.json'
    atomic_json(root/'million_inventory_plan.json',inventory)
    atomic_json(root/'active_research_run.json',{'updated_at':now(),'model_out_dir':out,'inventory_out_dir':inventory_out,
        'model_plan':'large_model_search_plan.json','inventory_plan':'million_inventory_plan.json',
        'predecessor_out_dir':old['out_dir'],'handoff_reason':'User authorized expanded full150–200k-data research; no predecessor model/test stage had started'})
    atomic_json(root/old['out_dir']/'handoff.json',json.loads((root/'active_research_run.json').read_text()))
    print(json.dumps({'large_data_configurations':len(gat)+len(geo),'models_out':out,
        'inventory_out':inventory_out,'screen_epochs':20,'promoted_max_epochs':100,'champion_seeds':[42,43,44]}))


if __name__=='__main__':main()
