#!/usr/bin/env python3
"""Recover missing E producer events by exact deterministic job replay.

The original operator output may have an invalid symmetric stereocentre that
RDKit removes on re-parsing. No operator is inferred or invented from a pair.
Only actually replayed events for the same normalized molecule are accepted.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import time

from expand_million_bank import worker_init, run_job
from scalable_bank_store import BankStore, file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,default=Path('molecule_bank_1m/bank.sqlite'))
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    with a.db.with_suffix('.generation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=BankStore(a.db)
        rows=[dict(r) for r in store.connection.execute('''SELECT r.relation_id,r.molecule_id,r.job_id,
            r.operator_history_json,r.objective_json,j.config_json FROM relations r JOIN jobs j USING(job_id)
            WHERE r.method='E' AND r.operator_history_json='[]' ORDER BY r.relation_id''')]
        if not rows:
            print('No missing E producer histories remain.',flush=True)
            return
        jobs={r['job_id']:json.loads(r['config_json']) for r in rows}
        config=json.loads(Path('million_bank_config.json').read_text())
        holdout=json.loads(Path(config['holdout']).read_text())
        events={}
        with ProcessPoolExecutor(max_workers=a.workers,initializer=worker_init,
              initargs=(str(a.db),holdout,config['quality'])) as pool:
            futures={pool.submit(run_job,j):key for key,j in jobs.items()}
            for future in as_completed(futures):
                key=futures[future]
                result=future.result()
                events[key]={c['molecule']['molecule_id']:json.loads(c['proposal']['operator_history_json'])
                             for c in result['candidates']}
                print('Replayed',key,len(result['candidates']),'accepted identities',flush=True)
        changes=[]
        for row in rows:
            history=events[row['job_id']].get(row['molecule_id'])
            if not history:
                raise ValueError('Replay failed to recover actual history for '+row['relation_id'])
            changes.append({**row,'recovered_history':history})
        artifact=a.db.parent/('e_history_replay_repair_'+str(time.time_ns())+'.json')
        artifact.write_text(json.dumps({'policy':__doc__,'changes':changes,'script_sha256':file_sha256(__file__),
            'generator_sha256':file_sha256('evolutionary_generator.py')},indent=2)+'\n')
        with store.connection:
            for row in changes:
                objective=json.loads(row['objective_json'])
                objective.update(operator_history_status='recovered_immediate_producer_events_by_deterministic_job_replay_not_complete_ancestry',
                    provenance_repair_artifact=str(artifact),provenance_repair_artifact_sha256=file_sha256(artifact))
                store.connection.execute('UPDATE relations SET operator_history_json=?,objective_json=? WHERE relation_id=? AND operator_history_json=?',
                    (json.dumps(row['recovered_history'],sort_keys=True),json.dumps(objective,sort_keys=True),row['relation_id'],row['operator_history_json']))
            store.connection.execute('INSERT INTO events(kind,payload_json) VALUES(?,?)',
                ('e_history_repair_by_deterministic_replay',json.dumps({'relations':len(changes),'jobs':len(jobs),
                    'artifact':str(artifact),'sha256':file_sha256(artifact),'molecular_identities_changed':False})))
        print(json.dumps({'recovered_relations':len(changes),'replayed_jobs':len(jobs),'artifact':str(artifact)}),flush=True)


if __name__=='__main__':main()
