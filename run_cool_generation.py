#!/usr/bin/env python3
"""Run the unchanged chemistry engine with one thermal-aware worker.

Thermal waits occur BEFORE the engine's per-job alarm and deterministic seed.
Committed batches/checkpoints and the chemistry configuration remain unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import expand_million_bank as engine

ORIGINAL_RUN_JOB=engine.run_job
LAST_FINISHED=0.0
NOMINAL_SINCE=None
STATUS=Path(os.environ.get('CD_THERMAL_STATUS_DIR','molecule_bank_1m/thermal_control'))


def thermal_state():
    try:
        result=subprocess.run(['/usr/bin/osascript','-l','JavaScript','-e',
            'ObjC.import("Foundation"); Number($.NSProcessInfo.processInfo.thermalState)'],
            text=True,capture_output=True,timeout=5)
        value=int(result.stdout.strip())
        return value if result.returncode==0 and value in (0,1,2,3) else None
    except (OSError,ValueError,subprocess.TimeoutExpired):return None


def record(state,action,job_id=None):
    STATUS.mkdir(parents=True,exist_ok=True)
    path=STATUS/('generation_worker_'+str(os.getpid())+'.json')
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps({'pid':os.getpid(),'updated_epoch':time.time(),'thermal_state':state,
        'action':action,'job_id':job_id,'max_generation_workers':1,'nominal_rest_seconds':1,
        'fair_serious_critical_unknown_policy':'defer new work until nominal for30s',
        'temperature_celsius_available':False},indent=2)+'\n')
    temp.replace(path)


def wait_for_thermal_budget(job_id):
    global NOMINAL_SINCE
    # Rest between batches; the preceding batch has already been returned to
    # the coordinator for transactional persistence.
    if LAST_FINISHED:
        time.sleep(max(0,1-(time.monotonic()-LAST_FINISHED)))
    cooling=False
    while True:
        state=thermal_state()
        if state==0:
            if not cooling:
                record(state,'running_one_worker',job_id);return
            NOMINAL_SINCE=NOMINAL_SINCE or time.monotonic()
            if time.monotonic()-NOMINAL_SINCE>=30:
                record(state,'running_after_cooldown',job_id);return
        else:
            cooling=True;NOMINAL_SINCE=None
        record(state,'waiting_for_sustained_nominal',job_id)
        time.sleep(5)


def cool_job(job):
    global LAST_FINISHED
    wait_for_thermal_budget(job['job_id'])
    try:return ORIGINAL_RUN_JOB(job)
    finally:LAST_FINISHED=time.monotonic()


# Picklable top-level adapter: only scheduling changes, before seeded chemistry.
engine.run_job=cool_job


def main():
    args=sys.argv[1:]
    if '--workers' in args:
        if int(args[args.index('--workers')+1])!=1:raise ValueError('Cool generation requires exactly one worker')
    else:sys.argv.extend(['--workers','1'])
    STATUS.mkdir(parents=True,exist_ok=True)
    manifest={'policy':'one worker; nominal1s rest; defer new jobs for fair/serious/critical/unknown until30s nominal',
        'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'chemistry_engine_sha256':hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
        'wall_clock_alarm_starts_after_thermal_wait':True,'created_epoch':time.time()}
    (STATUS/'generation_policy.json').write_text(json.dumps(manifest,indent=2)+'\n')
    engine.main()


if __name__=='__main__':main()
