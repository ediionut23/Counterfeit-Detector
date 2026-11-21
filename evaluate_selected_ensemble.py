#!/usr/bin/env python3
"""Evaluate one frozen validation-selected ensemble under a shared test lock.

This is the only helper that accesses test predictions. It never reports
individual member test metrics and never chooses members or thresholds on test.
Calling it again with the same frozen pin returns the existing result.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from run_overnight_research import atomic_json
from select_validation_ensemble import fixed_split,metrics,sha256


def import_trainer(path):
    name='frozen_ensemble_trainer_'+sha256(path)[:24]
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot import frozen trainer: '+str(path))
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    if not callable(getattr(module,'infer_selected_checkpoint',None)):
        raise ValueError('Trainer lacks the raw-inference ensemble adapter: '+str(path))
    return module


def evaluate(selection_path,family_lock,out,*,device='cpu',threads=2,batch_size=None,cache_root=None,selection_sha256=None):
    selection_path,family_lock,out=map(lambda p:Path(p).resolve(),(selection_path,family_lock,out))
    selection=json.loads(selection_path.read_text());selection_hash=sha256(selection_path)
    if selection_sha256 and selection_hash!=selection_sha256:
        raise ValueError('Frozen selection hash differs from the final inference command')
    if selection.get('selection_uses')!='validation_only' or selection.get('test_predictions_accessed') is not False or selection.get('aggregation')!='equal_probability_mean' or not selection.get('members'):
        raise ValueError('Not a frozen validation-only equal-mean selection')
    if selection.get('selector_sha256')!=sha256(Path(__file__).with_name('select_validation_ensemble.py')):
        raise ValueError('Validation selector implementation differs from frozen selection')
    for key,pin_key in (('cohort_manifest','cohort_manifest_sha256'),('dataset_csv','dataset_csv_sha256'),('graph_index','graph_index_sha256')):
        if sha256(selection[key])!=selection[pin_key]:
            raise ValueError('Frozen cohort input changed: '+key)
    for member in selection['members']:
        for key in ('checkpoint','trainer','inference_adapter'):
            if sha256(member[key])!=member[key+'_sha256']:
                raise ValueError('Frozen member changed before test: '+member['id']+'/'+key)
    pin={'selection_path':str(selection_path),'selection_sha256':selection_hash,
        'cohort_sha256':selection['cohort_sha256'],'winner':selection['winner'],
        'aggregation':selection['aggregation'],'threshold':selection['threshold'],
        'members':[{'id':m['id'],'checkpoint_sha256':m['checkpoint_sha256'],'trainer_sha256':m['trainer_sha256'],
                    'inference_adapter':m['inference_adapter'],'inference_adapter_sha256':m['inference_adapter_sha256']} for m in selection['members']],
        'result_directory':str(out),'evaluator_sha256':sha256(__file__)}
    family_lock.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(family_lock)+'.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads(family_lock.read_text()) if family_lock.exists() else None
        if previous:
            if previous.get('pin')!=pin:
                raise ValueError('Shared test lock already belongs to another model/ensemble/cohort; test-guided reselection is forbidden')
            if previous.get('state')=='complete':
                path=Path(previous['result_path'])
                if not path.is_file() or sha256(path)!=previous['result_sha256']:
                    raise ValueError('Existing locked test result is missing or changed; refusing another evaluation')
                return json.loads(path.read_text())
        elif out.exists() and any(out.iterdir()):
            raise FileExistsError('Unowned nonempty final-test directory: '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(family_lock,{'state':'running','pin':pin})
        # Test IDs/labels and model forwards are accessed only after selection
        # and the shared immutable pin are locked above.
        expected=fixed_split(selection['dataset_csv'],'test');ids=sorted(expected)
        labels=np.asarray([expected[i] for i in ids],dtype=int)
        total=np.zeros(len(ids),dtype=np.float64)
        try:
            for member in selection['members']:
                module=import_trainer(member['inference_adapter'])
                kwargs={'checkpoint_path':member['checkpoint'],'graph_index':selection['graph_index'],
                    'split_manifest':selection['cohort_manifest'],'split':'test','device':device,'threads':threads}
                if batch_size is not None:kwargs['batch_size']=batch_size
                if cache_root is not None:kwargs['cache_root']=str(Path(cache_root).resolve())
                prediction=module.infer_selected_checkpoint(**kwargs)
                if prediction.get('cohort_sha256')!=selection['cohort_sha256'] or prediction.get('checkpoint_sha256')!=member['checkpoint_sha256']:
                    raise ValueError('Inference checkpoint/cohort differs from the selected member: '+member['id'])
                names=[str(value) for value in prediction['source_id']]
                y=np.asarray(prediction['label'],dtype=int);p=np.asarray(prediction['probability'],dtype=float)
                if y.ndim!=1 or p.ndim!=1 or len(names)!=len(y) or len(y)!=len(p) or len(set(names))!=len(names) or set(names)!=set(expected):
                    raise ValueError('Inference did not return exactly the complete fixed test IDs')
                if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                    raise ValueError('Inference returned invalid probabilities')
                table={identity:(int(label),float(value)) for identity,label,value in zip(names,y,p)}
                if any(table[identity][0]!=expected[identity] for identity in ids):
                    raise ValueError('Inference labels differ from the fixed cohort')
                total+=np.asarray([table[identity][1] for identity in ids])
            probability=total/len(selection['members'])
            # Verify immutable inputs again before any final score is published.
            if sha256(selection_path)!=selection_hash:
                raise ValueError('Frozen selection changed during final inference')
            for member in selection['members']:
                for key in ('checkpoint','trainer','inference_adapter'):
                    if sha256(member[key])!=member[key+'_sha256']:
                        raise ValueError('Frozen member changed during final inference')
            for key,pin_key in (('cohort_manifest','cohort_manifest_sha256'),('dataset_csv','dataset_csv_sha256'),('graph_index','graph_index_sha256')):
                if sha256(selection[key])!=selection[pin_key]:
                    raise ValueError('Frozen cohort changed during final inference')
            report=metrics(labels,probability,float(selection['threshold']))
            result={'pin':pin,'cohort_sha256':selection['cohort_sha256'],'winner':selection['winner'],
                'selection_sha256':selection_hash,'selected_members':pin['members'],
                'test_metrics':report,'test_metrics_at_0_5':metrics(labels,probability,.5),
                'threshold_source':'frozen_validation_only','individual_member_test_metrics_computed':False,
                'target_f1':.85,'test_target_reached':report['f1']>=.85,
                'interpretation':'Fixed synthetic molecular-origin task; not established detection of counterfeit medicine products.'}
            temporary=out/'ensemble_test_predictions.csv.tmp'
            with temporary.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['source_id','split','label','probability_generated','threshold','prediction'])
                for identity,label,value in zip(ids,labels,probability):
                    writer.writerow([identity,'test',int(label),format(float(value),'.17g'),selection['threshold'],int(value>=selection['threshold'])])
            temporary.replace(out/'ensemble_test_predictions.csv')
            atomic_json(out/'test_result.json',result)
            atomic_json(family_lock,{'state':'complete','pin':pin,'result_path':str(out/'test_result.json'),
                'result_sha256':sha256(out/'test_result.json'),'predictions_sha256':sha256(out/'ensemble_test_predictions.csv')})
            return result
        except BaseException as error:
            atomic_json(family_lock,{'state':'failed','pin':pin,'error':repr(error)})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-sha256')
    p.add_argument('--family-test-lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int);p.add_argument('--cache-root',type=Path)
    a=p.parse_args()
    if a.threads<1 or a.batch_size is not None and a.batch_size<1:p.error('Threads and optional batch size must be positive')
    r=evaluate(a.selection,a.family_test_lock,a.out,device=a.device,threads=a.threads,
        batch_size=a.batch_size,cache_root=a.cache_root,selection_sha256=a.selection_sha256)
    print(json.dumps({'winner':r['pin']['winner'],'test_metrics':r['test_metrics'],'test_target_reached':r['test_target_reached']},indent=2))


if __name__=='__main__':main()
