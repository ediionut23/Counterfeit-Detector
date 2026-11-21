#!/usr/bin/env python3
"""Conservative macOS cooling pauses for verified, owned training stages only.

Generation is never stopped: its wall-clock job alarms would expire. Ownership
uses the queue token, wrapper lease/spec/token, exact argv and libproc process
birth time. No process inventory, UI automation or unrelated apps are used.
"""
from __future__ import annotations
import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

TRAINERS={'train_gnn.py','train_gatv2_experiment.py','train_gatv2_search.py',
          'train_geometric_gnn.py','train_geometric_variants.py',
          'select_validation_ensemble.py','evaluate_selected_ensemble.py'}
SUPERVISORS={'run_large_model_search.py','run_overnight_research.py'}
THERMAL_NAMES={0:'nominal',1:'fair',2:'serious',3:'critical',None:'unknown'}


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


class ProcBSDInfo(ctypes.Structure):
    # Local Apple SDK sys/proc_info.h, PROC_PIDTBSDINFO=3; struct is136bytes.
    _fields_=[(key,ctypes.c_uint32) for key in 'flags status xstatus pid ppid uid gid ruid rgid svuid svgid rfu'.split()]+[
        ('comm',ctypes.c_char*16),('name',ctypes.c_char*32)]+[
        (key,ctypes.c_uint32) for key in 'nfiles pgid pjobc tdev tpgid'.split()]+[
        ('nice',ctypes.c_int32),('start_sec',ctypes.c_uint64),('start_usec',ctypes.c_uint64)]


class ProcessReader:
    def __init__(self):
        if sys.platform!='darwin':raise RuntimeError('Process ownership reader requires macOS libproc')
        self.proc=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
        self.proc.proc_pidinfo.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int]
        self.proc.proc_pidinfo.restype=ctypes.c_int
        self.proc.proc_pidpath.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
        self.proc.proc_pidpath.restype=ctypes.c_int
        self.libc=ctypes.CDLL(None,use_errno=True)
        self.libc.sysctl.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_uint,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p,ctypes.c_size_t]
        self.libc.sysctlbyname.argtypes=[ctypes.c_char_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p,ctypes.c_size_t]
        value=ctypes.c_int();size=ctypes.c_size_t(ctypes.sizeof(value))
        if self.libc.sysctlbyname(b'kern.argmax',ctypes.byref(value),ctypes.byref(size),None,0)!=0 or not 4096<=value.value<=16*1024*1024:
            raise RuntimeError('Cannot read a safe kern.argmax process-argument bound')
        self.argmax=value.value

    def snapshot(self,pid):
        pid=int(pid);info=ProcBSDInfo();ctypes.set_errno(0)
        count=self.proc.proc_pidinfo(pid,3,0,ctypes.byref(info),ctypes.sizeof(info))
        if count!=ctypes.sizeof(info):
            raise ProcessLookupError(ctypes.get_errno() or errno.ESRCH,'Cannot verify process birth/parent',pid)
        path=ctypes.create_string_buffer(4096)
        if self.proc.proc_pidpath(pid,path,len(path))<=0:
            raise PermissionError('Cannot verify executable for PID '+str(pid))
        buffer=ctypes.create_string_buffer(self.argmax);size=ctypes.c_size_t(len(buffer));mib=(ctypes.c_int*3)(1,49,pid)
        if self.libc.sysctl(mib,3,buffer,ctypes.byref(size),None,0)!=0:
            raise PermissionError('Cannot verify process argv for PID '+str(pid))
        raw=buffer.raw[:size.value]
        if len(raw)<5:raise ValueError('Truncated process argument record')
        argc=int.from_bytes(raw[:4],sys.byteorder,signed=True)
        if not 1<=argc<=16384:raise ValueError('Invalid process argument count')
        position=raw.find(b'\0',4)+1
        if position<=4:raise ValueError('Missing executable argument prefix')
        while position<len(raw) and raw[position]==0:position+=1
        argv=[]
        for _ in range(argc):
            end=raw.find(b'\0',position)
            if end<position:raise ValueError('Truncated process argv')
            argv.append(raw[position:end].decode('utf-8','surrogateescape'));position=end+1
        # Never decode the environment that follows the exact argc strings.
        fingerprint={'pid':pid,'ppid':int(info.ppid),'uid':int(info.uid),
            'start':[int(info.start_sec),int(info.start_usec)],
            'executable':str(Path(os.fsdecode(path.value)).resolve()),'argv_sha256':digest(argv)}
        return fingerprint,argv


def thermal_state():
    try:
        result=subprocess.run(['/usr/bin/osascript','-l','JavaScript','-e',
            'ObjC.import("Foundation"); Number($.NSProcessInfo.processInfo.thermalState)'],
            capture_output=True,text=True,timeout=5,check=True)
        value=int(result.stdout.strip())
        return value if value in {0,1,2,3} else None
    except (OSError,ValueError,subprocess.SubprocessError):return None


def same_process(left,right,*,parent=False):
    keys=('pid','uid','start','executable','argv_sha256')+('ppid',) if parent else ('pid','uid','start','executable','argv_sha256')
    return all(left.get(key)==right.get(key) for key in keys)


def canonical_command(command):
    command=list(command)
    if command:command[0]=canonical_executable(command[0])
    return command


def canonical_executable(path):
    path=Path(path).resolve()
    # Homebrew framework Python's bin launcher execs the matching app binary,
    # rewriting argv0. Accept only that exact same-framework/version endpoint.
    if path.parent.name=='bin' and path.name.startswith('python') and '/Python.framework/Versions/' in str(path):
        app=path.parent.parent/'Resources/Python.app/Contents/MacOS/Python'
        if app.is_file():return str(app.resolve())
    return str(path)


def script_argument(argv):
    index=1
    while index<len(argv) and argv[index] in {'-u','-B'}:index+=1
    return argv[index] if index<len(argv) and not argv[index].startswith('-') else None


def option_value(argv,name):
    return argv[argv.index(name)+1] if name in argv and argv.index(name)+1<len(argv) else None


def lease_held(path):
    with Path(path).open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(stream,fcntl.LOCK_UN);return False


def owned_training(progress_path,workspace,reader):
    """Return a fully verified training child, orNone while queues are idle."""
    progress_path,workspace=Path(progress_path).resolve(),Path(workspace).resolve()
    queue=load_json(progress_path)
    if queue.get('status')!='running' or not queue.get('current_stage'):return None
    stage=queue.get('stages',{}).get(queue['current_stage'],{})
    if stage.get('status')!='running':return None
    status_path=Path(stage['stage_status_path']).resolve();lease_path=Path(stage['stage_lock_path']).resolve()
    stage_dir=progress_path.parent/'stages'
    if status_path.parent!=stage_dir or lease_path.parent!=stage_dir:
        raise ValueError('Stage ownership paths escape the selected queue')
    status=load_json(status_path)
    if status.get('status')!='running':return None
    spec_path=status_path.with_name(status_path.name.replace('.status.json','.spec.json'))
    spec=load_json(spec_path)
    token=stage.get('wrapper_token')
    if not token or token!=status.get('wrapper_token') or token!=spec.get('wrapper_token'):
        raise ValueError('Stage wrapper ownership token mismatch')
    if not stage.get('command_sha256') or not stage['command_sha256']==status.get('command_sha256')==spec.get('command_sha256'):
        raise ValueError('Stage command hash mismatch')
    if status.get('command')!=spec.get('command') or status.get('command')!=stage.get('command'):
        raise ValueError('Stage command payload mismatch')
    if Path(spec.get('workspace','')).resolve()!=workspace or Path(spec.get('lock_path','')).resolve()!=lease_path or not lease_held(lease_path):
        raise ValueError('Stage workspace or live command lease mismatch')
    command=status['command'];script=script_argument(command)
    if script is None or Path(script).name not in TRAINERS:return None
    if (workspace/script).resolve()!=workspace/Path(script).name:
        raise ValueError('Training script is outside the selected workspace')
    q,qargv=reader.snapshot(queue['pid']);wrapper,wargv=reader.snapshot(status['wrapper_pid']);child,cargv=reader.snapshot(status['child_pid'])
    if any(p['uid']!=os.getuid() for p in (q,wrapper,child)):
        raise ValueError('A stage process has a different user identity')
    if Path(script_argument(qargv) or '').name not in SUPERVISORS or option_value(qargv,'--startup-token')!=queue.get('startup_token'):
        raise ValueError('Queue process startup token or executable mismatch')
    if status['wrapper_pid']!=stage.get('pid') or Path(script_argument(wargv) or '').name!='run_overnight_research.py' or option_value(wargv,'--stage-wrapper')!=str(spec_path):
        raise ValueError('Wrapper PID or exact wrapper command mismatch')
    if child['ppid']!=wrapper['pid'] or wrapper['ppid']!=q['pid'] and not (wrapper['ppid']==1 and stage.get('adopted_at')):
        raise ValueError('Owned queue/wrapper/training parent chain mismatch')
    if canonical_command(cargv)!=canonical_command(command) or child['executable']!=canonical_executable(command[0]):
        raise ValueError('Live training argv/executable differs from the owned command')
    return {'fingerprint':child,'queue_fingerprint':q,'wrapper_fingerprint':wrapper,
        'queue_token':queue['startup_token'],'wrapper_token':token,'command_sha256':stage['command_sha256'],
        'progress_path':str(progress_path),'stage':queue['current_stage']}


class DutyPolicy:
    def __init__(self,compute=60,rest=30,fair_compute=15,fair_rest=45,recovery=60,latched=False):
        if min(compute,rest,fair_compute,fair_rest,recovery)<=0:raise ValueError('Cooling periods must be positive')
        self.compute,self.rest,self.fair_compute,self.fair_rest,self.recovery=compute,rest,fair_compute,fair_rest,recovery
        self.latched=latched;self.nominal_since=None;self.anchor=None;self.mode=None

    def decision(self,state,now):
        if state in {2,3}:
            self.latched=True;self.nominal_since=None
        if self.latched:
            if state==0:
                if self.nominal_since is None:self.nominal_since=now
                if now-self.nominal_since>=self.recovery:
                    self.latched=False;self.anchor=now;self.mode='nominal'
                else:return True,'waiting_for_sustained_nominal_thermal_state'
            else:
                self.nominal_since=None
                return True,'serious_or_critical_latched_until_sustained_nominal'
        mode='nominal' if state==0 else 'fair_or_unknown'
        if mode!=self.mode or self.anchor is None:self.mode,self.anchor=mode,now
        elapsed=now-self.anchor
        if mode=='nominal':
            pause=elapsed%(self.compute+self.rest)>=self.compute
        else:
            # Fair/unknown starts with a rest, then allows only25% duty bydefault.
            pause=elapsed%(self.fair_compute+self.fair_rest)<self.fair_rest
        return pause,('scheduled_cooling_rest' if pause else 'scheduled_compute_window')


class Guard:
    def __init__(self,progress,state_path,workspace,*,poll=5,policy=None,reader=None,thermal_reader=None):
        self.progress=[Path(path).resolve() for path in progress];self.state_path=Path(state_path).resolve()
        self.workspace=Path(workspace).resolve();self.poll=poll
        self.reader=reader or ProcessReader();self.thermal_reader=thermal_reader or thermal_state
        self.state_path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=Path(str(self.state_path)+'.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.state=load_json(self.state_path) or {'paused':{}}
        self.state.setdefault('paused',{})
        self.policy=policy or DutyPolicy(latched=bool(self.state.get('thermal_latched')))
        if self.state.get('thermal_latched'):self.policy.latched=True
        self.stopping=False
        self.state.update(guard_pid=os.getpid(),instance_token=uuid.uuid4().hex,status='starting',
            progress_paths=[str(path) for path in self.progress],workspace=str(self.workspace))
        self.save()

    def save(self):
        self.state['updated_unix']=time.time();self.state['thermal_latched']=self.policy.latched
        atomic_json(self.state_path,self.state)

    def event(self,kind,detail):
        with self.state_path.with_suffix('.events.jsonl').open('a') as stream:
            stream.write(encode({'time_unix':time.time(),'kind':kind,**detail})+'\n')

    def resume_all(self,reason):
        for key,record in list(self.state['paused'].items()):
            original=record['fingerprint']
            try:
                current,_=self.reader.snapshot(original['pid'])
                if not same_process(original,current):
                    self.event('resume_rejected_process_identity_changed',{'pid':original['pid']})
                else:
                    os.kill(original['pid'],signal.SIGCONT)
                    self.event('resumed',{'pid':original['pid'],'reason':reason})
            except ProcessLookupError:pass
            except Exception as error:
                self.state['last_error']=repr(error)
                continue  # Keep an unverifiable pause record for explicit recovery.
            del self.state['paused'][key]
            self.save()

    def pause(self,record,reason):
        original=record['fingerprint'];key=str(original['pid'])
        if key in self.state['paused']:return
        current,_=self.reader.snapshot(original['pid'])
        if not same_process(original,current,parent=True):raise ValueError('Training PID changed before cooling pause')
        # Write intent before SIGSTOP, so aSIGKILL between the signal and the next
        # save still leaves enough identity evidence to recover SIGCONT safely.
        self.state['paused'][key]={**record,'pause_intended':True,'reason':reason}
        self.save()
        current,_=self.reader.snapshot(original['pid'])
        if not same_process(original,current,parent=True):
            del self.state['paused'][key];self.save();raise ValueError('Training PID reused during cooling pause')
        os.kill(original['pid'],signal.SIGSTOP)
        self.state['paused'][key]['pause_confirmed']=True;self.save()
        self.event('paused',{'pid':original['pid'],'reason':reason,'stage':record['stage']})

    def run(self,recover_only=False,max_seconds=None):
        def stop(signum,frame):self.stopping=True
        prior={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
        started=time.monotonic()
        try:
            self.resume_all('guard_startup_recovery')
            if recover_only:return
            while not self.stopping and (max_seconds is None or time.monotonic()-started<max_seconds):
                state=self.thermal_reader();pause,reason=self.policy.decision(state,time.monotonic())
                self.state.update(status='cooling' if pause else 'compute_window',thermal_state=state,
                    thermal_name=THERMAL_NAMES.get(state,'unknown'),decision=reason,verified_targets=[])
                if not pause:self.resume_all('scheduled_or_thermal_recovery')
                for path in self.progress:
                    try:
                        owned=owned_training(path,self.workspace,self.reader)
                        if owned:
                            self.state['verified_targets'].append(owned)
                            if pause:self.pause(owned,reason)
                    except Exception as error:
                        self.state['last_error']=repr(error)
                self.save()
                deadline=time.monotonic()+self.poll
                while not self.stopping and time.monotonic()<deadline:time.sleep(min(.2,max(0,deadline-time.monotonic())))
        finally:
            self.resume_all('guard_exit_recovery')
            self.state.update(status='stopped',unrecovered_pause_records=len(self.state['paused']))
            self.save()
            for sig,handler in prior.items():signal.signal(sig,handler)
            self.lock.close()
            if self.state['paused']:
                raise RuntimeError('Some owned pause records could not be verified/resumed; rerun --recover-only with the same guard state and process permissions')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--progress',type=Path,action='append',required=True)
    p.add_argument('--state',type=Path,default=Path('training_runs/thermal_guard/state.json'))
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--poll-seconds',type=float,default=5)
    p.add_argument('--compute-seconds',type=float,default=60);p.add_argument('--rest-seconds',type=float,default=30)
    p.add_argument('--fair-compute-seconds',type=float,default=15);p.add_argument('--fair-rest-seconds',type=float,default=45)
    p.add_argument('--nominal-recovery-seconds',type=float,default=60)
    p.add_argument('--recover-only',action='store_true');p.add_argument('--probe',action='store_true')
    a=p.parse_args()
    if a.poll_seconds<=0:p.error('Polling interval must be positive')
    reader=ProcessReader()
    if a.probe:
        state=thermal_state();targets=[]
        for path in a.progress:
            try:targets.append({'progress':str(path),'owned_training':owned_training(path,a.workspace,reader)})
            except Exception as error:targets.append({'progress':str(path),'verification_error':repr(error)})
        print(json.dumps({'thermal_state':state,'thermal_name':THERMAL_NAMES[state],'targets':targets},indent=2));return
    policy=DutyPolicy(a.compute_seconds,a.rest_seconds,a.fair_compute_seconds,a.fair_rest_seconds,a.nominal_recovery_seconds)
    guard=Guard(a.progress,a.state,a.workspace,poll=a.poll_seconds,policy=policy,reader=reader)
    guard.run(recover_only=a.recover_only)


if __name__=='__main__':main()
