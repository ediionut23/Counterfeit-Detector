#!/usr/bin/env python3
"""Isolated orchestration fixture: selection, one test, and restart idempotence."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time


def wait_for(predicate,timeout=12):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Timed out waiting for fixture state')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return {}


def durability_fixture(runner,root,base_plan):
    from run_overnight_research import lease_held
    script=root/'blocking_stage.py'
    script.write_text('''import os,pathlib,sys,time
r=pathlib.Path(sys.argv[1]);r.mkdir(exist_ok=True)
with (r/'launches.txt').open('a') as f:f.write(str(os.getpid())+'\\n')
(r/'ready').write_text(str(os.getpid()))
deadline=time.monotonic()+30
while not (r/'release').exists():
 if time.monotonic()>deadline:raise RuntimeError('Fixture stage not released')
 time.sleep(.05)
''')
    for kill_wrapper in (False,True):
        tag='orphan_lease' if kill_wrapper else 'supervisor_restart'
        work=root/tag
        plan={**base_plan,'out_dir':tag+'_queue','families':[],
              'stages':[{'id':'blocking','command':[sys.executable,str(script),str(work)]}]}
        plan_path=root/(tag+'.json');plan_path.write_text(json.dumps(plan))
        progress=root/plan['out_dir']/'progress.json'
        command=[sys.executable,str(runner),'--plan',str(plan_path)]
        first=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        second=None
        try:
            wait_for(lambda:(work/'ready').exists())
            state=read_json(progress)['stages']['blocking']
            assert lease_held(state['stage_lock_path'])
            if kill_wrapper:
                os.kill(state['pid'],signal.SIGKILL)
                # The real command inherits the lease: killing both queue and
                # wrapper cannot make a still-live command look safe to rerun.
                wait_for(lambda:lease_held(state['stage_lock_path']))
            first.kill();first.wait(timeout=5)
            second=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            wait_for(lambda:read_json(progress).get('stages',{}).get('blocking',{}).get('adopted_at'))
            assert len((work/'launches.txt').read_text().splitlines())==1
            (work/'release').touch()
            code=second.wait(timeout=20)
            final=read_json(progress)
            assert len((work/'launches.txt').read_text().splitlines())==1
            if kill_wrapper:
                assert code!=0 and final['status']=='failed'
                assert 'durable exit record' in final['error']
            else:
                assert code==0 and final['status']=='complete'
                assert final['stages']['blocking']['attempts']==1
        finally:
            (work/'release').touch()
            for process in (first,second):
                if process is not None and process.poll() is None:
                    process.terminate();process.wait(timeout=5)

    # Detach must surface constructor/pin errors, rather than print success PID.
    bad={**base_plan,'out_dir':'bad_detach','families':[],
         'file_pins':{str(script):'not_the_fixture_hash'}}
    bad_path=root/'bad_detach.json';bad_path.write_text(json.dumps(bad))
    failed=subprocess.run([sys.executable,str(runner),'--plan',str(bad_path),'--detach'],capture_output=True,text=True)
    assert failed.returncode!=0 and 'did not start successfully' in failed.stderr
    good={**base_plan,'out_dir':'good_detach','families':[]}
    good_path=root/'good_detach.json';good_path.write_text(json.dumps(good))
    passed=subprocess.run([sys.executable,str(runner),'--plan',str(good_path),'--detach'],capture_output=True,text=True,check=True)
    assert json.loads(passed.stdout)['startup_verified'] is True


def main():
    runner=Path(__file__).with_name('run_overnight_research.py').resolve()
    with tempfile.TemporaryDirectory(prefix='overnight_queue_fixture_') as d:
        root=Path(d)
        db=sqlite3.connect(root/'bank.sqlite')
        db.execute('create table molecules(molecule_id text)')
        db.execute("insert into molecules values('fixture_identity')")
        db.commit(); db.close()
        fake=root/'fixture_trainer.py'
        fake.write_text('''import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--out-dir');p.add_argument('--score',type=float);p.add_argument('--no-test',action='store_true');p.add_argument('--evaluate-test',action='store_true');p.add_argument('--checkpoint');p.add_argument('--resume');a=p.parse_args()
r=pathlib.Path(a.out_dir);r.mkdir(parents=True,exist_ok=True)
s={'best_validation_f1':a.score,'completed':True,'current_epoch':1}
if a.evaluate_test:
 with (r/'test_count.txt').open('a') as f:f.write('fixture-test\\n')
 s['test_metrics']={'f1':0.1,'fixture_only':True}
else:(r/'best.pt').write_text('fixture checkpoint')
(r/'summary.json').write_text(json.dumps(s));(r/'progress.json').write_text(json.dumps(s))
''')
        runs=[]
        for ident,score in [('smaller',.7),('larger',.8)]:
            out=root/ident
            runs.append({'id':ident,'out_dir':str(out),'command':[sys.executable,str(fake),'--out-dir',str(out),'--score',str(score),'--no-test']})
        plan={'workspace':str(root),'out_dir':'queue','bank_db':'bank.sqlite',
              'target_unique_molecules':1,'target_f1':.85,'generation_command':['false'],
              'families':[{'id':'fixture_family','runs':runs}]}
        path=root/'plan.json';path.write_text(json.dumps(plan))
        command=[sys.executable,str(runner),'--plan',str(path)]
        subprocess.run(command,check=True,capture_output=True)
        subprocess.run(command,check=True,capture_output=True)
        result=json.loads((root/'queue/fixture_family_result.json').read_text())
        assert result['winner']=='larger' and not result['validation_target_reached']
        assert (root/'larger/test_count.txt').read_text().splitlines()==['fixture-test']
        assert not (root/'smaller/test_count.txt').exists()
        assert json.loads((root/'queue/progress.json').read_text())['status']=='complete'
        durability_fixture(runner,root,plan)
        print(json.dumps({'fixture_passed':True,'validation_selection':True,'single_test_after_restart':True,
                          'target_failure_reported_honestly':True,'supervisor_kill_live_stage_adopted':True,
                          'killed_wrapper_child_lease_prevents_duplicate':True,
                          'indeterminate_orphan_exit_fails_closed':True,'detach_health_check':True}))


if __name__=='__main__':main()
