#!/usr/bin/env python3
"""Durable, resumable local bank/training queue; no shell or test-driven search.

Stages checkpoint atomically. Trainers checkpoint every epoch independently.
A STOP file drains the current command; it prevents launching subsequent work.
Only validation F1 chooses a configuration; each family has one locked test run.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return {}


def lease_held(path):
    """The open-file lease proves live ownership, without trusting reused PIDs."""
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream,fcntl.LOCK_UN)
        return False


def process_alive(pid):
    if not isinstance(pid,int) or pid<1:
        return False
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stage_wrapper(spec_path,lease_fd=None):
    """Own one command independently of its replaceable queue supervisor."""
    spec=json.loads(Path(spec_path).read_text())
    status_path=Path(spec['status_path'])
    lock=os.fdopen(lease_fd,'a+') if lease_fd is not None else Path(spec['lock_path']).open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        return 75
    prior=load_json(status_path)
    if prior.get('command_sha256')==spec['command_sha256'] and prior.get('status')=='complete':
        return 0
    status={'command_sha256':spec['command_sha256'],'wrapper_token':spec['wrapper_token'],
            'wrapper_pid':os.getpid(),'status':'running','started_at':now(),
            'attempt':spec['attempt'],'command':spec['command']}
    atomic_json(status_path,status)
    process=None
    pending_signal=None

    def relay(signum,frame):
        nonlocal pending_signal
        pending_signal=signum
        if process is not None and process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,relay)
    signal.signal(signal.SIGINT,relay)
    try:
        os.chdir(spec['workspace'])
        with Path(spec['log_path']).open('a',buffering=1) as log:
            log.write('\n'+now()+' '+json.dumps(spec['command'])+'\n')
            if pending_signal is not None:
                raise InterruptedError('Stage wrapper stopped before launching child')
            # An orphan command retains the lease even if its wrapper is killed.
            process=subprocess.Popen(spec['command'],stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            status['child_pid']=process.pid
            atomic_json(status_path,status)
            code=process.wait()
        status.update(status='complete' if code==0 else 'failed',returncode=code,finished_at=now())
        atomic_json(status_path,status)
        return code if code>=0 else 128-code
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        status.update(status='failed',returncode=1,error=repr(error),finished_at=now())
        atomic_json(status_path,status)
        raise
    finally:
        lock.close()


class Queue:
    def __init__(self,plan_path,startup_token=None):
        self.plan_path=Path(plan_path).resolve()
        self.plan=json.loads(self.plan_path.read_text())
        self.root=Path(self.plan['workspace']).resolve()
        os.chdir(self.root)
        self.out=self.root/self.plan['out_dir']
        self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'queue.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.lock.seek(0); self.lock.truncate(); self.lock.write(str(os.getpid())); self.lock.flush()
        self.state=load_json(self.out/'progress.json') or {'created_at':now(),'stages':{}}
        pin=checksum(self.plan_path)
        if self.state.get('plan_sha256',pin)!=pin:
            raise ValueError('Plan changed: use a new versioned output directory')
        self.state.update(plan_sha256=pin,pid=os.getpid(),status='running',updated_at=now(),
                          startup_token=startup_token or uuid.uuid4().hex,current_stage=None)
        for path,digest in self.plan.get('file_pins',{}).items():
            if checksum(path)!=digest:
                raise ValueError('Pinned implementation/input changed: '+path)
        self.stop=self.out/'STOP'
        self.save()

    def save(self):
        self.state['updated_at']=now()
        self.state['bank']=load_json(self.root/'molecule_bank_1m/progress.json')
        self.state['experiments']={run['id']:load_json(self.root/run['out_dir']/'progress.json')
            for family in self.plan.get('families',[]) for run in family['runs']}
        atomic_json(self.out/'progress.json',self.state)
        cohort=self.plan.get('training_cohort',{})
        lines=['Rulare autonomă: bancă moleculară + GATv2 + GNN geometric',
            'Actualizare UTC: '+self.state['updated_at'],
            'Stare: '+self.state['status'],
            'Etapă: '+str(self.state.get('current_stage')),
            'Molecule unice salvate: '+str(self.state['bank'].get('unique_molecules','în curs')),
            'Țintă bancă: '+str(self.plan.get('target_unique_molecules')),
            'Țintă F1: '+str(self.plan.get('target_f1')),
            'Exemple de antrenare în cohorta comună: '+str(cohort.get('train_rows','fixture')),
            '', 'Modele:']
        for ident,run in self.state['experiments'].items():
            lines.append(ident+': epoca='+str(run.get('current_epoch',run.get('epoch','în așteptare')))+
                ', F1 validare maxim='+str(run.get('best_validation_f1','în așteptare')))
        for family in self.plan.get('families',[]):
            result=load_json(self.out/(family['id']+'_result.json'))
            if result:lines.append(family['id']+' test blocat: '+json.dumps(result.get('test_metrics'),ensure_ascii=False))
        lines += ['', 'Datele, joburile și proveniența sunt salvate în SQLite la fiecare lot.',
            'Modelele, optimizerul, schedulerul și RNG sunt salvate în best.pt/last.pt la fiecare epocă.',
            'Configurațiile și pragurile se aleg exclusiv pe validare; testul nu conduce căutarea.',
            'Etichetele descriu originea moleculară publică/generată; nu confirmă contrafaceri reale.',
            'Geometriile sunt conformeri prezise cu optimizare convergentă, nu coordonate experimentale.',
            'Banca de 1M și cohorta de antrenare sunt inventare diferite; noile molecule nu au toate conformeri 3D.',
            'Plan și hashuri: overnight_research_plan.json. Metrici și comenzi complete: progress.json și logurile etapelor.']
        if self.state.get('error'):lines+=['Eroare: '+self.state['error']]
        report=self.out/'progress_report.txt'
        temporary=report.with_suffix('.tmp')
        temporary.write_text('\n'.join(lines)+'\n');temporary.replace(report)

    def run(self,key,command,*,resume_dir=None):
        if self.stop.exists():
            raise InterruptedError('STOP file present; no subsequent command launched')
        for path,digest in self.plan.get('file_pins',{}).items():
            if checksum(path)!=digest:
                raise ValueError('Pinned implementation/input changed: '+path)
        stage=self.state['stages'].setdefault(key,{})
        command=list(command)
        command_hash=hashlib.sha256(json.dumps(command).encode()).hexdigest()
        if stage.get('command_sha256',command_hash)!=command_hash:
            raise ValueError('Completed/planned command changed: '+key)
        if stage.get('status')=='complete':
            return
        stage_dir=self.out/'stages'
        stage_dir.mkdir(exist_ok=True)
        basename=hashlib.sha256(key.encode()).hexdigest()[:12]+'_'+command_hash[:12]
        status_path=stage_dir/(basename+'.status.json')
        spec_path=stage_dir/(basename+'.spec.json')
        lease_path=stage_dir/(basename+'.lock')
        wrapper_status=load_json(status_path)
        if wrapper_status and wrapper_status.get('command_sha256')!=command_hash:
            raise ValueError('Stage ownership record changed: '+key)
        active=lease_held(lease_path)
        # A prior wrapper may still be acquiring its lease immediately after Popen.
        if not active and stage.get('status')=='running' and process_alive(stage.get('pid')):
            deadline=time.monotonic()+10
            while not active and time.monotonic()<deadline:
                time.sleep(.1)
                wrapper_status=load_json(status_path)
                active=lease_held(lease_path)
                if wrapper_status.get('status') in {'complete','failed'}:
                    break
            if not active and wrapper_status.get('status') not in {'complete','failed'}:
                raise RuntimeError('Prior stage PID is live without a verified command lease; refusing duplicate launch: '+key)
        if not active and wrapper_status.get('status')=='running':
            raise RuntimeError('Stage lease vanished without a durable exit record; inspect the orphan before retrying: '+key)
        if not active and wrapper_status.get('status') not in {'complete'}:
            if resume_dir and (Path(resume_dir)/'last.pt').exists():
                command+=['--resume',str(Path(resume_dir)/'last.pt')]
            token=uuid.uuid4().hex
            spec={'command_sha256':command_hash,'command':command,'wrapper_token':token,
                  'workspace':str(self.root),'status_path':str(status_path),'lock_path':str(lease_path),
                  'log_path':str(self.out/(key+'.log')),'attempt':stage.get('attempts',0)+1}
            # Acquire before spawning and transfer the same open-file lease to
            # the wrapper: there is no unowned fork/startup race on restart.
            with lease_path.open('a+') as launch_lease:
                try:
                    fcntl.flock(launch_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError('Stage acquired a live lease during startup; restart to adopt it: '+key)
                atomic_json(spec_path,spec)
                atomic_json(status_path,{'status':'starting','command_sha256':command_hash,
                                         'wrapper_token':token,'attempt':spec['attempt']})
                stage.update(status='starting',started_at=now(),command=command,
                    command_sha256=command_hash,attempts=spec['attempt'],wrapper_token=token,
                    stage_status_path=str(status_path),stage_lock_path=str(lease_path))
                self.state['current_stage']=key
                self.save()
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--stage-wrapper',str(spec_path),'--stage-lease-fd',str(launch_lease.fileno())],
                    cwd=self.root,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,start_new_session=True,pass_fds=(launch_lease.fileno(),))
                stage.update(status='running',pid=process.pid)
                self.save()
            print(now(),key,'START',flush=True)
        elif active:
            stage.update(status='running',command_sha256=command_hash,
                pid=wrapper_status.get('wrapper_pid'),adopted_at=now(),
                stage_status_path=str(status_path),stage_lock_path=str(lease_path))
            self.state['current_stage']=key
            self.save()
            print(now(),key,'ADOPT verified stage lease',flush=True)
        deadline=time.monotonic()+10
        last_save=time.monotonic()
        while True:
            wrapper_status=load_json(status_path)
            if wrapper_status and wrapper_status.get('command_sha256')!=command_hash:
                raise ValueError('Stage status command fingerprint changed: '+key)
            if wrapper_status.get('status') in {'complete','failed'} and not lease_held(lease_path):
                code=wrapper_status.get('returncode',0 if wrapper_status['status']=='complete' else 1)
                break
            if not lease_held(lease_path) and time.monotonic()>deadline:
                raise RuntimeError('Stage lost its lease without a durable exit record: '+key)
            if time.monotonic()-last_save>=5:
                self.save()
                last_save=time.monotonic()
            time.sleep(.25)
        stage.update(status='complete' if code==0 else 'failed',returncode=code,finished_at=now())
        stage['wrapper_status']=wrapper_status
        self.save()
        print(now(),key,'EXIT',code,flush=True)
        if code:
            raise RuntimeError(f'{key} failed with status {code}; inspect {key}.log')

    def bank_count(self):
        with sqlite3.connect('file:'+str(Path(self.plan['bank_db']).resolve())+'?mode=ro',uri=True,timeout=30) as c:
            return c.execute('SELECT COUNT(*) FROM molecules').fetchone()[0]

    def expand_bank(self):
        # A short root-launched production benchmark can be adopted safely.
        lock_path=Path(self.plan['bank_db']).with_suffix('.generation.lock')
        while True:
            if self.stop.exists():
                raise InterruptedError('STOP file present')
            with lock_path.open('a+') as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    self.state['current_stage']='wait_for_existing_generation'
                    self.save()
                    time.sleep(10)
                    continue
            break
        if self.bank_count()<self.plan['target_unique_molecules']:
            self.run('expand_bank',self.plan['generation_command'])
        count=self.bank_count()
        if count<self.plan['target_unique_molecules']:
            raise RuntimeError(f'Generation ended below target: {count}; chemistry gates were not relaxed')
        self.state['bank_target_verified_count']=count
        self.save()

    def family(self,family):
        for run in family['runs']:
            self.run(run['id'],run['command'],resume_dir=run['out_dir'])
        ranked=[]
        for run in family['runs']:
            report=load_json(Path(run['out_dir'])/'summary.json')
            score=report.get('best_validation_f1')
            if not isinstance(score,(float,int)) or not 0<=score<=1:
                raise ValueError('Missing finite validation F1 for '+run['id'])
            ranked.append((score,run))
        score,winner=max(ranked,key=lambda pair:pair[0])
        checkpoint=Path(winner['out_dir'])/'best.pt'
        selection={'family':family['id'],'criterion':'highest validation F1; ties follow predeclared run order',
                   'winner':winner['id'],'best_validation_f1':score,'checkpoint':str(checkpoint),
                   'checkpoint_sha256':checksum(checkpoint),'target_f1':self.plan['target_f1'],
                   'validation_target_reached':score>=self.plan['target_f1'],
                   'candidate_validation_f1':{r['id']:s for s,r in ranked}}
        selection_path=self.out/(family['id']+'_selection.json')
        prior=load_json(selection_path)
        if prior and prior!=selection:
            raise ValueError('Locked model selection changed; refusing another test-driven choice')
        atomic_json(selection_path,selection)
        command=[x for x in winner['command'] if x!='--no-test']
        command+=['--evaluate-test','--checkpoint',str(checkpoint)]
        self.run(family['id']+'_locked_test',command)
        final=load_json(Path(winner['out_dir'])/'summary.json')
        metrics=final.get('test_metrics')
        if not isinstance(metrics,dict) or not isinstance(metrics.get('f1'),(int,float)) or not math.isfinite(metrics['f1']):
            raise ValueError('Locked test exited without finite test F1: '+winner['id'])
        expected=self.plan.get('training_cohort',{}).get('cohort_sha256')
        if expected and (final.get('cohort_sha256')!=expected or not isinstance(metrics.get('roc_auc'),(float,int)) or not math.isfinite(metrics['roc_auc'])):
            raise ValueError('Locked test cohort/AUC does not match the predeclared plan')
        selection['test_metrics']=metrics
        selection['test_target_reached']=metrics['f1']>=self.plan['target_f1']
        atomic_json(self.out/(family['id']+'_result.json'),selection)

    def work(self):
        self.expand_bank()
        for stage in self.plan.get('stages',[]):
            self.run(stage['id'],stage['command'])
        for family in self.plan.get('families',[]):
            self.family(family)
        self.state.update(status='complete',current_stage=None,finished_at=now(),
            target_semantics='85% is a research target, never a guaranteed/edited test score')
        self.save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('overnight_research_plan.json'))
    p.add_argument('--detach',action='store_true')
    p.add_argument('--stage-wrapper',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--stage-lease-fd',type=int,help=argparse.SUPPRESS)
    p.add_argument('--startup-token',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.stage_wrapper:
        raise SystemExit(stage_wrapper(args.stage_wrapper,args.stage_lease_fd))
    if args.detach:
        plan=json.loads(args.plan.read_text())
        out=Path(plan['workspace'])/plan['out_dir']
        out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        token=uuid.uuid4().hex
        with (out/'supervisor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--plan',str(args.plan.resolve()),
                '--startup-token',token],
                cwd=plan['workspace'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=env)
        # A fresh token prevents confusing a stale progress file or reused PID with
        # successful startup. Constructor pin/lock errors must be surfaced here.
        deadline=time.monotonic()+15
        while True:
            progress=load_json(out/'progress.json')
            fresh=progress.get('pid')==child.pid and progress.get('startup_token')==token and progress.get('plan_sha256')==checksum(args.plan)
            if fresh and progress.get('status') in {'failed','stopped'}:
                raise RuntimeError('Detached supervisor failed during startup; inspect '+str(out/'supervisor.log'))
            if fresh and (progress.get('current_stage') or progress.get('status')=='complete'):
                if child.poll() not in (None,0):
                    raise RuntimeError('Detached supervisor exited during startup; inspect '+str(out/'supervisor.log'))
                break
            if child.poll() is not None:
                raise RuntimeError('Detached supervisor did not start successfully; inspect '+str(out/'supervisor.log'))
            if time.monotonic()>deadline:
                raise RuntimeError('Detached supervisor startup was not acknowledged; PID '+str(child.pid)+'; inspect '+str(out/'supervisor.log'))
            time.sleep(.1)
        print(json.dumps({'pid':child.pid,'startup_verified':True,'progress':str(out/'progress.json'),'log':str(out/'supervisor.log')}))
        return
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    queue=Queue(args.plan,startup_token=args.startup_token)
    # Keep an authorized overnight local computation awake while this process exists.
    awake=None
    try:
        if sys.platform=='darwin':
            awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        queue.work()
    except BaseException as error:
        queue.state.update(status='stopped' if isinstance(error,InterruptedError) else 'failed',error=repr(error))
        queue.save()
        raise
    finally:
        if awake and awake.poll() is None:
            awake.terminate()


if __name__=='__main__':
    main()
