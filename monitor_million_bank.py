#!/usr/bin/env python3
"""Read-only live bank progress, usable in the user's own visible terminal."""
import argparse
import json
import time
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--progress",type=Path,default=Path("molecule_bank_1m/progress.json"))
parser.add_argument("--watch",action="store_true")
args=parser.parse_args()
while True:
    try:
        data=json.loads(args.progress.read_text())
        print(time.strftime("%H:%M:%S"),f"unique={data['unique_molecules']:,}/{data['target_unique_molecules']:,}",
              f"eligible={data['training_eligible_molecules']:,}","methods="+str(data["method_origin_counts"]),
              "jobs="+str(data["jobs_by_state"]),flush=True)
    except FileNotFoundError:
        print("Waiting for initialization/generation progress file",flush=True)
    if not args.watch:
        break
    time.sleep(5)
