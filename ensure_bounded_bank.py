#!/usr/bin/env python3
"""Idempotent inventory gate: require volume AND named-category coverage.

The frozen outer queue checks catalogue size alone. This unconditional first
stage also recovers category floors after an interruption beyond 1M molecules.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from yield_aware_scheduling import category_coverage


def inspect(db,config):
    connection=sqlite3.connect('file:'+str(Path(db).resolve())+'?mode=ro',uri=True,timeout=120)
    try:
        total=connection.execute('SELECT COUNT(*) FROM molecules').fetchone()[0]
        coverage=category_coverage(connection)
    finally:connection.close()
    floors=config['yield_scheduler']['minimum_named_category_unique']
    unmet={method+'/'+category:max(0,int(minimum)-coverage.get((method,category),0))
        for method,minimum in floors.items() for category in config['categories']
        if coverage.get((method,category),0)<int(minimum)}
    return {'unique_molecules':total,'target_unique_molecules':config['target_unique_molecules'],
        'coverage':{method+'/'+category:coverage.get((method,category),0) for method in floors for category in config['categories']},
        'unmet_named_category_floors':unmet,'coverage_floors_met':not unmet,
        'objective_met':total>=config['target_unique_molecules'] and not unmet}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=Path('molecule_bank_1m/bank.sqlite'))
    parser.add_argument('--config',type=Path,default=Path('million_bank_bounded_cool_config_20261109.json'))
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.workers!=1:parser.error('This profile permits exactly one chemistry child')
    config=json.loads(args.config.read_text())
    before=inspect(args.db,config);print(json.dumps(before,indent=2),flush=True)
    if before['objective_met']:return
    if args.verify_only:raise SystemExit(2)
    result=subprocess.run([sys.executable,'-u','expand_bounded_cool_bank.py','--db',str(args.db),
        '--config',str(args.config),'--workers','1'])
    if result.returncode:raise SystemExit(result.returncode)
    after=inspect(args.db,config);print(json.dumps(after,indent=2),flush=True)
    if not after['objective_met']:raise SystemExit(2)


if __name__=='__main__':main()
