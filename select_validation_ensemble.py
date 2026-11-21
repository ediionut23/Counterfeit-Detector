#!/usr/bin/env python3
"""Freeze one predeclared singleton/equal-mean model using validation only.

No test predictions are read or computed. Every member must contain exactly
the fixed validation IDs and labels, and correspond to its completed best
checkpoint. Candidate ordering and equal weights are declared in a pinned JSON.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from audit_overnight_training_split import encoded, sha256


def file_path(base,value):
    return (base/Path(value)).resolve()


def fixed_split(dataset,split):
    result={}
    with Path(dataset).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split']!=split:
                continue
            identity=row.get('molecule_id') or row.get('source_id')
            if not identity or identity in result or row['label'] not in {'0','1'}:
                raise ValueError('Invalid, duplicate or contradictory fixed-split identity')
            result[identity]=int(row['label'])
    if set(result.values())!={0,1}:
        raise ValueError('Fixed evaluation split must contain both labels')
    return result


def read_probabilities(path,expected,split='val'):
    records={}
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        columns=set(reader.fieldnames or [])
        probability='probability_generated' if 'probability_generated' in columns else 'probability' if 'probability' in columns else None
        if not probability or not {'source_id','split','label'}<=columns:
            raise ValueError('Unsupported probability artifact schema: '+str(path))
        epochs=set()
        for row in reader:
            identity=row['source_id']
            if row['split']!=split or identity in records or identity not in expected or int(row['label'])!=expected[identity]:
                raise ValueError('Prediction split/ID/label does not match the full fixed validation cohort')
            value=float(row[probability])
            if not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Nonfinite or out-of-range probability')
            records[identity]=value
            if row.get('epoch'):
                epochs.add(int(row['epoch']))
    if records.keys()!=expected.keys():
        raise ValueError(f'Incomplete fixed validation predictions: {len(records)}/{len(expected)}')
    return records,epochs


def choose_threshold(labels,probabilities):
    precision,recall,thresholds=precision_recall_curve(labels,probabilities)
    scores=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    best=np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-12))[-1]
    return float(thresholds[best])


def metrics(labels,probabilities,threshold):
    y=np.asarray(labels,dtype=int);p=np.asarray(probabilities,dtype=float);pred=p>=threshold
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    return {'n':len(y),'threshold':float(threshold),'f1':2*tp/max(2*tp+fp+fn,1),
        'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'accuracy':(tp+tn)/len(y),
        'fnr':fn/max(tp+fn,1),'fpr':fp/max(fp+tn,1),
        'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
        'true_positive':tp,'false_positive':fp,'false_negative':fn,'true_negative':tn}


def checkpoint_details(path):
    import torch
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        stream.seek(0);checkpoint=torch.load(stream,map_location='cpu',weights_only=True)
    config=checkpoint.get('config',checkpoint.get('configuration',{}))
    cohort=checkpoint.get('cohort_sha256',config.get('cohort_sha256'))
    return checkpoint,config,cohort,digest.hexdigest()


def select(spec_path,spec_sha256,out):
    spec_path,out=Path(spec_path).resolve(),Path(out).resolve()
    if sha256(spec_path)!=spec_sha256:
        raise ValueError('Candidate specification differs from the predeclared pinned hash')
    if out.exists():
        saved_path=out/'selection.json'
        if not saved_path.is_file():
            raise FileExistsError('Unowned or incomplete selection directory: '+str(out))
        saved=json.loads(saved_path.read_text())
        if saved.get('candidate_specification_sha256')!=spec_sha256 or saved.get('selector_sha256')!=sha256(__file__):
            raise ValueError('Existing immutable selection belongs to another specification or selector')
        for filename,digest in saved['frozen_input_sha256'].items():
            if sha256(filename)!=digest:
                raise ValueError('A previously selected input changed: '+filename)
        if sha256(out/'validation_ensemble_probabilities.csv')!=saved['validation_ensemble_predictions_sha256']:
            raise ValueError('Previously published validation ensemble probabilities changed')
        return saved
    spec=json.loads(spec_path.read_text())
    base=Path(spec['workspace']).resolve() if spec.get('workspace') else spec_path.parent
    manifest_path=file_path(base,spec['cohort_manifest'])
    dataset=file_path(base,spec['dataset_csv']);graph_index=file_path(base,spec['graph_index'])
    manifest=json.loads(manifest_path.read_text())
    for path,name in ((dataset,'dataset.csv'),(graph_index,'graph_index.csv')):
        if sha256(path)!=manifest['selected_files'][name]['sha256']:
            raise ValueError('Fixed cohort file changed: '+str(path))
    expected=fixed_split(dataset,'val');ids=sorted(expected);labels=np.asarray([expected[i] for i in ids])
    expected_count=sum(manifest['split_label_counts']['val'].values())
    if len(expected)!=expected_count:
        raise ValueError('Actual validation rows differ from frozen cohort manifest')
    input_hashes={str(p):sha256(p) for p in (spec_path,manifest_path,dataset,graph_index)}
    members={};probabilities={};member_metrics={}
    for definition in spec['members']:
        member=dict(definition);identity=member['id']
        if identity in members:
            raise ValueError('Repeated member ID: '+identity)
        path_fields=['checkpoint','predictions','summary','trainer']
        if member.get('inference_adapter'):
            path_fields.append('inference_adapter')
        for key in path_fields:
            member[key]=str(file_path(base,member[key]))
        member.setdefault('inference_adapter',member['trainer'])
        summary=json.loads(Path(member['summary']).read_text())
        if summary.get('completed') is not True or summary.get('test_evaluated',False) or 'test_metrics' in summary:
            raise ValueError('Member must be completed and validation-only: '+identity)
        if summary.get('cohort_sha256')!=manifest['cohort_sha256']:
            raise ValueError('Summary cohort differs: '+identity)
        checkpoint,config,cohort,checkpoint_hash=checkpoint_details(member['checkpoint'])
        if cohort!=manifest['cohort_sha256'] or int(config['seed'])!=int(member['seed']):
            raise ValueError('Checkpoint seed/cohort differs: '+identity)
        epoch=int(checkpoint['epoch'])
        if epoch!=int(checkpoint.get('best_epoch',epoch)) or epoch!=int(summary['best_epoch']):
            raise ValueError('Member is not its completed validation-selected checkpoint: '+identity)
        table,epochs=read_probabilities(member['predictions'],expected)
        if epochs and epochs!={epoch}:
            raise ValueError('Validation probability epoch differs from checkpoint: '+identity)
        values=np.asarray([table[i] for i in ids]);threshold=choose_threshold(labels,values)
        report=metrics(labels,values,threshold)
        checkpoint_f1=checkpoint.get('validation_metrics',{}).get('f1')
        if checkpoint_f1 is None or abs(float(checkpoint_f1)-report['f1'])>1e-6 or abs(float(summary['best_validation_f1'])-report['f1'])>1e-6:
            raise ValueError('Validation probabilities do not reproduce completed checkpoint F1: '+identity)
        trainer_hash=sha256(member['trainer'])
        checkpoint_trainer=checkpoint.get('provenance',{}).get('trainer_sha256',config.get('script_sha256'))
        if checkpoint_trainer and checkpoint_trainer!=trainer_hash:
            raise ValueError('Trainer implementation differs from checkpoint: '+identity)
        member.update(checkpoint_sha256=checkpoint_hash,trainer_sha256=trainer_hash,
            inference_adapter_sha256=sha256(member['inference_adapter']),
            predictions_sha256=sha256(member['predictions']),summary_sha256=sha256(member['summary']),
            best_epoch=epoch,validation_f1=report['f1'],cohort_sha256=cohort)
        members[identity]=member;probabilities[identity]=values;member_metrics[identity]=report
        for key in ('checkpoint','predictions','summary','trainer','inference_adapter'):
            input_hashes[member[key]]=member[key+'_sha256']
    if not members:
        raise ValueError('No declared ensemble members')
    required=sorted(int(seed) for seed in spec.get('required_ensemble_seeds',[42,43,44]))
    candidate_ids=set();candidate_reports=[];winner=None;winner_values=None;best=-1.
    for definition in spec['candidates']:
        candidate=dict(definition);names=candidate['members']
        if candidate['id'] in candidate_ids or not names or len(set(names))!=len(names) or any(name not in members for name in names):
            raise ValueError('Invalid or repeated declared candidate')
        candidate_ids.add(candidate['id'])
        selected=[members[name] for name in names]
        if len(selected)>1:
            if len({m['family'] for m in selected})!=1 or len({m['config_id'] for m in selected})!=1 or sorted(int(m['seed']) for m in selected)!=required:
                raise ValueError('Equal-mean seed ensemble must use the same family/config and exactly the predeclared seeds')
        values=np.mean([probabilities[name] for name in names],axis=0)
        report=metrics(labels,values,choose_threshold(labels,values))
        candidate_reports.append({'id':candidate['id'],'members':names,'aggregation':'equal_probability_mean',**report})
        if report['f1']>best+1e-12:
            best=report['f1'];winner=candidate;winner_values=values
    if winner is None:
        raise ValueError('No predeclared candidates')
    for path,digest in input_hashes.items():
        if sha256(path)!=digest:
            raise RuntimeError('Input changed during ensemble selection: '+path)
    selected_report=next(r for r in candidate_reports if r['id']==winner['id'])
    result={'schema_version':1,'selection_uses':'validation_only','test_predictions_accessed':False,
        'candidate_specification':str(spec_path),'candidate_specification_sha256':spec_sha256,
        'cohort_manifest':str(manifest_path),'cohort_manifest_sha256':sha256(manifest_path),
        'dataset_csv':str(dataset),'dataset_csv_sha256':sha256(dataset),'graph_index':str(graph_index),'graph_index_sha256':sha256(graph_index),
        'cohort_sha256':manifest['cohort_sha256'],'validation_ids_sha256':hashlib.sha256(encoded(ids).encode()).hexdigest(),
        'winner':winner['id'],'members':[members[name] for name in winner['members']],
        'aggregation':'equal_probability_mean','threshold':selected_report['threshold'],
        'threshold_policy':'maximum validation F1; ties choose largest threshold',
        'candidate_tie_policy':'declared candidate order','validation_metrics':selected_report,
        'candidate_validation_metrics':candidate_reports,'member_validation_metrics':member_metrics,
        'frozen_input_sha256':input_hashes,
        'selector_sha256':sha256(__file__),'label_semantics':'public-reference versus generated molecular origin; no verified product-fraud inference',
        'statistical_limit':'Repeated validation-driven configuration/epoch/threshold selection is optimistic; final test remains inaccessible until this specification is frozen.'}
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.'+out.name,dir=out.parent))
    try:
        with (stage/'validation_ensemble_probabilities.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['source_id','split','label','probability_generated','threshold','prediction'])
            for identity,label,value in zip(ids,labels,winner_values):writer.writerow([identity,'val',int(label),format(float(value),'.17g'),result['threshold'],int(value>=result['threshold'])])
        result['validation_ensemble_predictions_sha256']=sha256(stage/'validation_ensemble_probabilities.csv')
        (stage/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
        recipe={'command':[sys.executable,str(Path(__file__).with_name('evaluate_selected_ensemble.py').resolve()),
            '--selection',str(out/'selection.json'),'--family-test-lock',str(out.parent/'global_ensemble_test_lock.json'),
            '--out',str(out/'final_test'),'--device','cpu','--threads','2'],
            'execute_only_after_this_selection_is_frozen':True,'member_standalone_test_invocations':False,
            'only_aggregate_test_metrics_are_reported':True}
        (stage/'final_inference_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
        stage.replace(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=select(a.spec,a.spec_sha256,a.out)
    print(json.dumps({k:r[k] for k in ('winner','threshold','validation_metrics','cohort_sha256')},indent=2))


if __name__=='__main__':main()
