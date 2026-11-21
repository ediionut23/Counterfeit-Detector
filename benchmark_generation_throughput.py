#!/usr/bin/env python3
"""Temporary-scale chemistry benchmark; never writes the production bank."""
import argparse
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from expand_million_bank import worker_init, run_job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--out', default='molecule_bank_1m/throughput_benchmark.json')
    args = p.parse_args()
    cfg = json.loads(Path('million_bank_config.json').read_text())
    import evolutionary_generator as eg
    eligible = []
    with open(cfg['reference_csv']) as f:
        for ref in csv.DictReader(f):
            mol = Chem.MolFromSmiles(ref['smiles'])
            if (mol and .5 <= Descriptors.qed(mol) and eg.sascorer.calculateScore(mol) <= 3.5
                    and eg.structural_alerts(mol) == 0 and 180 <= Descriptors.MolWt(mol) <= 500):
                eligible.append(ref)
    random.Random(20261101).shuffle(eligible)
    refs = eligible[:3]
    print('Feasible named reference parents:',len(eligible),[r['name'] for r in refs],flush=True)
    jobs=[]
    for n,ref in enumerate(refs):
        for method,pop,generations in [('M',40,8),('E',40,8),('E',128,4)]:
            job={'reference':ref,'job_id':f'benchmark_{n}_{method}_{pop}', 'method':method,
                 'category':'bioisostere','difficulty':'intermediate',
                 'band':cfg['difficulty_bands']['intermediate'],'seed':20261101+n,
                 'quality':cfg['quality'],'evolution':dict(cfg['evolution'],pop=pop,generations=generations,max_candidates=128),
                 'mutation':cfg['mutation'],'rules_path':cfg['rules_path'],
                 'scenario':cfg['scenario'],'config_hash':'readonly_throughput_benchmark'}
            jobs.append(job)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,
          initargs=('molecule_bank_1m/bank.sqlite',json.loads(Path(cfg['holdout']).read_text()),cfg['quality'])) as pool:
        futures={pool.submit(run_job,j):j for j in jobs}
        for future in as_completed(futures):
            j=futures[future]
            try:
                r=future.result()
                out={'name':j['reference']['name'],'method':j['method'],'pop':j['evolution']['pop'],
                     'generations':j['evolution']['generations'],'produced':r['produced'],
                     'accepted':len(r['candidates']),'seconds':r['seconds'],'rejections':r['rejections']}
            except Exception as e:
                out={'name':j['reference']['name'],'method':j['method'],'error':repr(e)}
            results.append(out)
            print(json.dumps(out),flush=True)
            Path(args.out).write_text(json.dumps({'feasible_reference_count':len(eligible),'results':results},indent=2)+'\n')


if __name__=='__main__':
    main()
