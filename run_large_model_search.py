#!/usr/bin/env python3
"""Resumable full-cohort screening, promotion, seed repeats and locked ensembles.

All trials use the same 161,157 training molecules. Screening changes the
epoch budget only. Configuration and seed selection never read test results.
The independent inventory queue can continue expanding the molecular bank.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from run_overnight_research import Queue, atomic_json, checksum, load_json, now
from enforce_large_training_cohort import audit as cohort_audit


def argument(command,name,value):
    command=list(command)
    if name in command:command[command.index(name)+1]=str(value)
    else:command.extend([name,str(value)])
    return command


def validation_score(run,expected):
    report=load_json(Path(run['out_dir'])/'summary.json')
    score=report.get('best_validation_f1')
    if not report.get('completed') or report.get('cohort_sha256')!=expected:
        raise ValueError('Missing completed full-cohort validation result: '+run['id'])
    if report.get('test_evaluated') or report.get('test_metrics'):
        raise ValueError('A candidate already accessed test and cannot enter validation search')
    if not isinstance(score,(int,float)) or not 0<=score<=1:
        raise ValueError('Invalid validation F1: '+run['id'])
    return float(score)


class LargeSearch(Queue):
    def save(self):
        super().save()
        adaptive={ident:load_json(Path(folder)/'progress.json') for ident,folder in self.state.get('adaptive_run_dirs',{}).items()}
        self.state['experiments'].update(adaptive)
        atomic_json(self.out/'progress.json',self.state)
        if adaptive:
            with (self.out/'progress_report.txt').open('a') as report:
                report.write('\nRepetări independente:\n')
                for ident,value in adaptive.items():
                    report.write(ident+': epoca='+str(value.get('current_epoch','în așteptare'))+
                        ', F1 validare='+str(value.get('best_validation_f1','în așteptare'))+'\n')

    def verify_derived(self,run,family):
        # Generated repeat commands must pass the same guard as the base trials.
        derived={**self.plan,'families':[{'id':family,'runs':[run]}]}
        target=self.out/'derived_plans'/(run['id']+'.json')
        atomic_json(target,derived)
        result=cohort_audit(target,self.plan['training_cohort']['directory'])
        atomic_json(target.with_suffix('.guard.json'),result)

    def train(self,run,key,epochs):
        cmd=argument(run['command'],'--epochs',epochs)
        last=Path(run['out_dir'])/'last.pt'
        if self.state['stages'].get(key,{}).get('status')!='complete' and key.startswith('promote_') and last.exists():
            import torch
            checkpoint=torch.load(last,map_location='cpu',weights_only=True)
            patience=int(cmd[cmd.index('--patience')+1])
            if checkpoint.get('bad_epochs',0)>=patience:
                self.state['stages'][key]={'status':'complete','returncode':0,
                    'command_sha256':checksum_text_argv(cmd),'finished_at':now(),
                    'execution_skipped':'screening already reached the fixed early-stopping patience',
                    'checkpoint_sha256':checksum(last),'requested_maximum_epochs':epochs}
                self.save()
                return validation_score(run,self.plan['training_cohort']['cohort_sha256'])
        self.run(key,cmd,resume_dir=run['out_dir'])
        return validation_score(run,self.plan['training_cohort']['cohort_sha256'])

    def freeze_ranking(self,family,kind,runs,count):
        path=self.out/(family+'_'+kind+'.json')
        prior=load_json(path)
        if prior:
            by_id={run['id']:run for run in runs}
            if prior['cohort_sha256']!=self.plan['training_cohort']['cohort_sha256']:
                raise ValueError('Locked cohort changed')
            return [by_id[ident] for ident in prior['selected_ids']]
        expected=self.plan['training_cohort']['cohort_sha256']
        scores=[(validation_score(run,expected),run) for run in runs]
        ranked=sorted(scores,key=lambda pair:-pair[0])
        chosen=[run for score,run in ranked[:count]]
        # Promotion overwrites best.pt; retain the actual screening evidence.
        evidence=self.out/'selection_evidence'/family/kind
        evidence.mkdir(parents=True,exist_ok=True)
        records=[]
        for score,run in scores:
            src=Path(run['out_dir']);dest=evidence/run['id'];dest.mkdir(exist_ok=True)
            for name in ('summary.json','val_probabilities.csv','validation_best_predictions.csv'):
                if (src/name).exists():shutil.copy2(src/name,dest/name)
            if run in chosen:shutil.copy2(src/'best.pt',dest/'best.pt')
            records.append({'id':run['id'],'best_validation_f1':score,'checkpoint_sha256':checksum(src/'best.pt'),
                            'command_sha256':checksum_text(run['command'])})
        atomic_json(path,{'family':family,'selection_stage':kind,'cohort_sha256':expected,
            'selected_ids':[r['id'] for r in chosen],'candidates':records,
            'criterion':'validation F1 descending; declared order resolves ties','test_used':False})
        return chosen

    def repeat(self,champion,family,seed):
        ident=family+'_champion_seed'+str(seed)
        dest=str(self.out/ident)
        cmd=argument(champion['command'],'--seed',seed)
        flag='--out-dir' if '--out-dir' in cmd else '--out'
        cmd=argument(cmd,flag,dest)
        run={'id':ident,'out_dir':dest,'command':cmd,'config_id':champion['id']}
        self.verify_derived(run,family)
        self.state.setdefault('adaptive_run_dirs',{})[ident]=dest
        self.train(run,'repeat_'+ident,self.plan['search_policy']['final_epochs'])
        return run

    def ensemble(self,family,runs,champion,repeats):
        members=[]
        for run in runs+repeats:
            validation_score(run,self.plan['training_cohort']['cohort_sha256'])
            cmd=run['command'];seed=int(cmd[cmd.index('--seed')+1])
            folder=Path(run['out_dir'])
            predictions=folder/('val_probabilities.csv' if family=='gatv2' else 'validation_best_predictions.csv')
            members.append({'id':run['id'],'family':family,'config_id':run.get('config_id',run['id']),
                'seed':seed,'checkpoint':str(folder/'best.pt'),'predictions':str(predictions),
                'summary':str(folder/'summary.json'),'trainer':cmd[1],
                'inference_adapter':'train_gatv2_search.py' if family=='gatv2' else 'train_geometric_variants.py'})
        candidates=[{'id':'single_'+member['id'],'members':[member['id']]} for member in members]
        candidates.append({'id':'champion_equal_seed_mean','members':[champion['id']]+[r['id'] for r in repeats]})
        cohort=self.plan['training_cohort']['directory']
        spec={'schema_version':1,'workspace':str(self.root),'cohort_manifest':cohort+'/manifest.json','dataset_csv':cohort+'/dataset.csv',
            'graph_index':cohort+'/graph_index.csv','members':members,'candidates':candidates,
            'required_ensemble_seeds':[42,43,44]}
        spec_path=self.out/(family+'_ensemble_candidates.json')
        if spec_path.exists() and load_json(spec_path)!=spec:
            raise ValueError('Ensemble candidate declaration changed after selection')
        atomic_json(spec_path,spec)
        selected=self.out/(family+'_ensemble_selection')
        self.run(family+'_select_on_validation',[sys.executable,'select_validation_ensemble.py',
            '--spec',str(spec_path),'--spec-sha256',checksum(spec_path),'--out',str(selected)])
        destination=self.out/(family+'_final_test')
        self.run(family+'_ensemble_locked_test',[sys.executable,'evaluate_selected_ensemble.py',
            '--selection',str(selected/'selection.json'),'--family-test-lock',str(self.out/(family+'_family_test_lock.json')),
            '--out',str(destination),'--device','cpu','--threads','2'])
        result=load_json(destination/'test_result.json')
        metrics=result.get('test_metrics',{})
        if result.get('cohort_sha256')!=self.plan['training_cohort']['cohort_sha256'] or not all(
            isinstance(metrics.get(key),(int,float)) and 0<=metrics[key]<=1 for key in ('f1','roc_auc')):
            raise ValueError('Final ensemble test returned no valid full-cohort metrics')
        result['target_f1']=self.plan['target_f1']
        result['test_target_reached']=metrics['f1']>=self.plan['target_f1']
        atomic_json(self.out/(family+'_result.json'),result)

    def work(self):
        check=cohort_audit(self.plan_path,self.plan['training_cohort']['directory'])
        atomic_json(self.out/'full_cohort_guard.json',check)
        policy=self.plan['search_policy']
        families=self.plan['families']
        # Interleave model families so geometric training gets an early turn.
        for slot in range(max(len(f['runs']) for f in families)):
            for family in families:
                if slot<len(family['runs']):
                    run=family['runs'][slot]
                    self.train(run,'screen_'+run['id'],policy['screen_epochs'])
        promoted={f['id']:self.freeze_ranking(f['id'],'promotion',f['runs'],policy['promote_per_family']) for f in families}
        for slot in range(policy['promote_per_family']):
            for family in families:
                run=promoted[family['id']][slot]
                self.train(run,'promote_'+run['id'],policy['final_epochs'])
        champions={f['id']:self.freeze_ranking(f['id'],'champion',promoted[f['id']],1)[0] for f in families}
        repeats={f['id']:[] for f in families}
        for seed in policy['repeat_seeds']:
            for family in families:
                ident=family['id']
                repeats[ident].append(self.repeat(champions[ident],ident,seed))
        for family in families:
            ident=family['id']
            self.ensemble(ident,family['runs'],champions[ident],repeats[ident])
        self.state.update(status='complete',current_stage=None,finished_at=now(),
            protocol='full-cohort screening/promotion; validation-only selection; one locked test per family')
        self.save()


def checksum_text(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def checksum_text_argv(value):
    # Same serialization as Queue.run; preserve its resume/idempotence ledger.
    import hashlib
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('large_model_search_plan.json'))
    p.add_argument('--detach',action='store_true')
    p.add_argument('--startup-token')
    a=p.parse_args()
    if a.detach:
        plan=load_json(a.plan);out=Path(plan['workspace'])/plan['out_dir'];out.mkdir(parents=True,exist_ok=True)
        token=uuid.uuid4().hex
        with (out/'supervisor.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--plan',str(a.plan.resolve()),'--startup-token',token],
                cwd=plan['workspace'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            state=load_json(out/'progress.json')
            if state.get('startup_token')==token and state.get('current_stage') and state.get('status')=='running':
                print(json.dumps({'startup_verified':True,'pid':process.pid,'progress':str(out/'progress.json')}));return
            if process.poll() is not None:raise RuntimeError('Search startup failed; inspect '+str(out/'supervisor.log'))
            time.sleep(.2)
        raise RuntimeError('Search did not acknowledge first stage')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    queue=LargeSearch(a.plan,a.startup_token)
    awake=None
    try:
        if sys.platform=='darwin':awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        queue.work()
    except BaseException as error:
        queue.state.update(status='stopped' if isinstance(error,InterruptedError) else 'failed',error=repr(error));queue.save();raise
    finally:
        if awake and awake.poll() is None:awake.terminate()


if __name__=='__main__':main()
