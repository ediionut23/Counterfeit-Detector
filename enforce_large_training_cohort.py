#!/usr/bin/env python3
"""Read-only guard: every declared experiment uses the complete frozen cohort."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_overnight_training_split import encoded, read_csv, sha256


def options(command):
    result={}
    for i,token in enumerate(command):
        if token.startswith('--'):
            key,sep,value=token.partition('=')
            if key in result:
                raise ValueError('Repeated argument: '+key)
            result[key]=value if sep else command[i+1] if i+1<len(command) and not command[i+1].startswith('--') else True
    return result


def resolve(root,value):
    return (root/Path(value)).resolve()


def audit(plan_path,cohort,expected_train=161157,expected_total=191701):
    plan_path,cohort=Path(plan_path).resolve(),Path(cohort).resolve()
    plan=json.loads(plan_path.read_text());root=Path(plan['workspace']).resolve()
    manifest=json.loads((cohort/'manifest.json').read_text())
    for name,pin in manifest['selected_files'].items():
        if sha256(cohort/name)!=pin['sha256']:
            raise ValueError('Frozen cohort file changed: '+name)
    rows,_=read_csv(cohort/'dataset.csv');graphs,_=read_csv(cohort/'graph_index.csv')
    records={(r['molecule_id'],r['split'],r['label'],r['sample_weight']) for r in rows}
    indexed={(r['source_id'],r['split'],r['label'],r['sample_weight']) for r in graphs}
    if len(records)!=len(rows) or len(indexed)!=len(graphs) or records!=indexed:
        raise ValueError('Dataset/index cohort tuples are not unique and equal')
    if len({r['molecule_id'] for r in rows})!=len(rows):
        raise ValueError('Source IDs are not unique')
    train={r['molecule_id'] for r in rows if r['split']=='train'}
    if len(rows)!=expected_total or len(train)!=expected_train:
        raise ValueError(f'Actual cohort is total={len(rows)}, train={len(train)}; expected {expected_total}/{expected_train}')
    digest=hashlib.sha256(encoded([[r['molecule_id'],r['split'],r['label']] for r in sorted(rows,key=lambda r:r['molecule_id'])]).encode()).hexdigest()
    if digest!=manifest['cohort_sha256']:
        raise ValueError('Actual cohort hash differs from frozen manifest')
    pins={resolve(root,key):value for key,value in plan.get('file_pins',{}).items()}
    for name in ('dataset.csv','graph_index.csv','manifest.json'):
        path=(cohort/name).resolve()
        if pins.get(path)!=sha256(path):
            raise ValueError('Plan must pin exact common cohort file: '+str(path))
    checked=[];out_dirs=set();run_ids=set()
    for family in plan.get('families',[]):
        for run in family['runs']:
            if run['id'] in run_ids:
                raise ValueError('Repeated global run ID: '+run['id'])
            run_ids.add(run['id']);flags=options(run['command'])
            forbidden=[]
            for flag in flags:
                token=flag.lower()
                subset=any(word in token for word in ('subset','fraction','smoke','debug','fast-dev','max-samples','max-train','train-limit','row-limit','max-graphs','graph-limit','sample-limit'))
                if subset or token in {'--limit','--evaluate-test','--prepare-cache-only'}:
                    forbidden.append(flag)
            if forbidden:
                raise ValueError(f"Subset/debug/test arguments in {run['id']}: {forbidden}")
            if '--no-test' not in flags:
                raise ValueError('Search run must explicitly forbid test: '+run['id'])
            if resolve(root,flags.get('--graph-index','__missing__'))!=cohort/'graph_index.csv':
                raise ValueError('Run does not use the full fixed graph index: '+run['id'])
            if '--dataset-csv' in flags and resolve(root,flags['--dataset-csv'])!=cohort/'dataset.csv':
                raise ValueError('Run uses a different molecular CSV: '+run['id'])
            if '--split-manifest' in flags and resolve(root,flags['--split-manifest'])!=cohort/'manifest.json':
                raise ValueError('Run uses a different split manifest: '+run['id'])
            output=resolve(root,flags.get('--out-dir',flags.get('--out','__missing__')))
            if output!=resolve(root,run['out_dir']) or output in out_dirs:
                raise ValueError('Missing, reused or inconsistent run output directory: '+run['id'])
            out_dirs.add(output)
            if '--seed' not in flags:
                raise ValueError('Seed must be explicit: '+run['id'])
            checked.append({'family':family['id'],'run_id':run['id'],'seed':int(flags['--seed']),
                'actual_train_unique_ids':len(train),'actual_cohort_unique_ids':len(rows),
                'command_sha256':hashlib.sha256(encoded(run['command']).encode()).hexdigest()})
    if not checked:
        raise ValueError('Plan contains no declared model runs')
    return {'passed':True,'plan_sha256':sha256(plan_path),'cohort_sha256':digest,
        'manifest_sha256':sha256(cohort/'manifest.json'),'actual_train_unique_ids':len(train),
        'actual_cohort_unique_ids':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),
        'runs_checked':checked,'model_results_or_test_predictions_read':False,
        'scope':'Preflight verifies declared commands and immutable full-cohort inputs; trainer cache/checkpoint cohort checks still apply at runtime.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--cohort',type=Path,default=Path('balanced_datasets/training_200k/overnight_strict_converged_v2'))
    p.add_argument('--expected-train',type=int,default=161157)
    p.add_argument('--expected-total',type=int,default=191701)
    p.add_argument('--out',type=Path)
    a=p.parse_args();result=audit(a.plan,a.cohort,a.expected_train,a.expected_total)
    if a.out:
        with a.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
