#!/usr/bin/env python3
"""Resume import of the previous bank and unmerged generated extensions."""
import argparse
import json
from pathlib import Path

from scalable_bank_store import BankStore,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",type=Path,default=Path("molecule_bank_1m/bank.sqlite"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--legacy",type=Path,default=Path("molecule_bank_500k/molecules.csv"))
    args=parser.parse_args()
    holdout=json.loads(Path("real_world_falsification/v1/external_holdout.json").read_text())
    store=BankStore(args.db)
    holdout_hash=file_sha256("real_world_falsification/v1/external_holdout.json")
    previous=store.connection.execute("SELECT value FROM metadata WHERE key='external_holdout_sha256'").fetchone()
    if previous and previous[0]!=holdout_hash:
        raise ValueError("Holdout changed; re-audit existing eligibility before resuming imports")
    store.connection.execute("INSERT OR IGNORE INTO metadata VALUES('external_holdout_sha256',?)",(holdout_hash,))
    store.connection.commit()
    store.import_catalogue(args.legacy,workers=args.workers,holdout=holdout)
    for path in sorted(Path("version_ga/supplements").glob("*/counterfeits_evolved_*.csv")):
        store.import_catalogue(path,workers=args.workers,holdout=holdout,method_override="E")
    summary=store.summary()
    (args.db.parent/"initialization_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
