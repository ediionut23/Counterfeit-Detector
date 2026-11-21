#!/usr/bin/env python3
"""Freeze the bounded, validation-only research schedule before any model run."""
import json
from pathlib import Path
import sys
from scalable_bank_store import file_sha256


def main():
    root=Path(__file__).resolve().parent
    cohort='balanced_datasets/training_200k/overnight_strict_converged_v2'
    out='training_runs/overnight_20261101'
    python=str(root/'venv/bin/python')
    manifest=json.loads((root/cohort/'manifest.json').read_text())
    gatbase=[python,'train_gatv2_experiment.py','--dataset-csv',cohort+'/dataset.csv',
        '--graph-index',cohort+'/graph_index.csv','--split-manifest',cohort+'/manifest.json',
        '--cache-dir','training_cache/gatv2_overnight_v2','--heads','4','--batch-size','128',
        '--epochs','80','--patience','12','--threads','2','--device','cpu',
        '--selection-metric','f1','--test-lock',out+'/gatv2_test_lock.json','--no-test']
    gatruns=[]
    for ident,hidden,layers,dropout,lr,weight,seed in [
        ('gatv2_unweighted',96,3,.2,.001,'none',42),
        ('gatv2_weighted',96,3,.2,.001,'row',42),
        ('gatv2_deep',128,4,.15,.0007,'none',43)]:
        dest=out+'/'+ident
        gatruns.append({'id':ident,'out_dir':dest,'command':gatbase+['--out-dir',dest,
            '--hidden',str(hidden),'--layers',str(layers),'--dropout',str(dropout),'--lr',str(lr),
            '--weight-mode',weight,'--seed',str(seed)]})
    geobase=[python,'train_geometric_gnn.py','--root','balanced_datasets/training_200k/3d',
        '--graph-index',cohort+'/graph_index.csv','--cache-root','training_cache/geometric_overnight_v2',
        '--cutoff','4.5','--max-neighbors','24','--batch-size','64','--epochs','60','--patience','10',
        '--threads','2','--selection','val_f1','--test-lock',out+'/geometric_test_lock.json','--no-test']
    georuns=[]
    for ident,hidden,interactions,radial,lr,weight,seed in [
        ('geometric_unweighted',64,3,16,.001,'none',42),
        ('geometric_weighted',96,4,32,.0007,'row',42)]:
        dest=out+'/'+ident
        georuns.append({'id':ident,'out_dir':dest,'command':geobase+['--out',dest,
            '--hidden',str(hidden),'--interactions',str(interactions),'--radial-basis',str(radial),
            '--lr',str(lr),'--weight-mode',weight,'--seed',str(seed)]})
    stages=[]
    for i,path in enumerate(sorted((root/'version_ga/supplements').glob('*/counterfeits*.csv'))):
        stages.append({'id':f'legacy_context_{i+1:02d}','command':[python,'import_legacy_bank_relations.py',
            '--db','molecule_bank_1m/bank.sqlite','--input',str(path.relative_to(root)),'--method','E']})
    stages += [
        {'id':'composition_scenarios','command':[python,'build_bank_scenarios.py','--db','molecule_bank_1m/bank.sqlite']},
        {'id':'audit_and_export_bank','command':[python,'audit_export_million_bank.py','--db','molecule_bank_1m/bank.sqlite',
            '--out','molecule_bank_1m/release','--minimum','1000000','--workers','4','--descriptor-sample','4096']},
        {'id':'disk_graph_shards','command':[python,'bank_graph_shards.py','prepare','--db','molecule_bank_1m/bank.sqlite',
            '--out','molecule_bank_1m/graphs_2d','--shard-size','2048']}]
    pins=['run_overnight_research.py','expand_million_bank.py','scalable_bank_store.py','evolutionary_generator.py',
        'evolve_category_pilot.py','m_proposal_engine.py','million_bank_config.json','mined_transformations.py',
        'named_reference_registry.py','import_legacy_bank_relations.py','build_bank_scenarios.py',
        'audit_export_million_bank.py','bank_graph_shards.py','train_gnn.py',
        'train_gatv2_experiment.py','train_geometric_gnn.py',
        'real_world_falsification/v1/external_holdout.json',cohort+'/manifest.json',cohort+'/dataset.csv',cohort+'/graph_index.csv']
    plan={'schema_version':1,'workspace':str(root),'out_dir':out,'bank_db':'molecule_bank_1m/bank.sqlite',
        'target_unique_molecules':1000000,'target_f1':.85,
        'generation_command':[python,'expand_million_bank.py','--workers','6'],
        'stages':stages,'families':[{'id':'gatv2','runs':gatruns},{'id':'geometric','runs':georuns}],
        'file_pins':{path:file_sha256(root/path) for path in pins},
        'training_cohort':{'directory':cohort,'rows':manifest['rows'],'train_rows':manifest['train_rows'],
            'split_label_counts':manifest['split_label_counts'],'cohort_sha256':manifest['cohort_sha256']},
        'scientific_policy':{
            'task':'public-reference molecular origin versus generated analogue origin; not verified medicine fraud',
            'bank_role':'1M is inventory size; supervised experiments use the frozen common 191k cohort with existing 3D assets',
            'geometry':'converged finite-energy predicted ETKDG/force-field conformers, not experimental coordinates',
            'selection':'predeclared 3 GATv2 + 2 geometric configurations, validation F1 only; one locked test per family',
            'target':'F1 >= 0.85 is a research target, never guaranteed or enforced by label/test changes',
            'weights':'trial names identify hierarchical train-row weighting or unit weights; heldout evaluation unweighted',
            'checkpointing':'atomic best/last model+optimizer+scheduler+RNG each epoch; SQLite transaction each generation job'}}
    path=root/'overnight_research_plan.json'
    if (root/out/'progress.json').exists():raise RuntimeError('Existing queue: do not overwrite its immutable plan')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'plan':str(path),'training':plan['training_cohort'],'configurations':5}))


if __name__=='__main__':main()
