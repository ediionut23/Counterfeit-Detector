#!/usr/bin/env python3
"""Resume named-reference M/E generation using coverage-deficit scheduling.

The SQLite job ledger is the checkpoint. Finished jobs and imported references
are idempotent. Molecules, parent relations and hypothetical observations are
counted separately. No generated item is asserted to be a real counterfeit.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import random
import signal
import sqlite3
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scalable_bank_store import BankStore, chemistry_init, normalize_identity, prepare_catalogue_row, stable_hash, file_sha256

_CACHE_DB = None
_QUALITY = None
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048)


def worker_init(db_path, holdout, quality):
    global _CACHE_DB,_QUALITY
    chemistry_init(holdout)
    _CACHE_DB = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro",uri=True,timeout=120)
    _QUALITY=quality


def checked_candidate(proposal, reference, job):
    history=proposal.get("operator_history_json") or proposal.get("operator_history") or "[]"
    try:
        history=json.loads(history) if isinstance(history,str) else history
    except (TypeError,ValueError):
        return None,"invalid_operator_history"
    if not isinstance(history,list) or not history:
        return None,"missing_recorded_operator_history"
    proposal={**proposal,"generation_method":job["method"],"label":"1"}
    row=prepare_catalogue_row(proposal)
    if "error" in row:
        return None,"normalization_failure"
    if not row["training_eligible"]:
        return None,"external_case_holdout"
    if row["smiles"]==reference["smiles"]:
        return None,"unchanged_reference"
    q=_QUALITY
    if not q["min_heavy_atoms"]<=row["heavy_atoms"]<=q["max_heavy_atoms"]:
        return None,"heavy_atom_gate"
    if not q["min_mw"]<=row["mw"]<=q["max_mw"] or row["qed"]<q["qed_min"]:
        return None,"property_gate"
    cached=_CACHE_DB.execute("SELECT properties_json FROM descriptor_cache WHERE smiles=?",(row["smiles"],)).fetchone()
    cached=json.loads(cached[0]) if cached else {}
    sa=row.get("sa_score") if row.get("sa_score") is not None and row["smiles"]==row["raw_canonical_smiles"] else cached.get("sa_score")
    mol=Chem.MolFromSmiles(row["smiles"])
    if sa is None:
        import evolutionary_generator as eg
        if eg.sascorer is None:
            raise RuntimeError("SA scorer required; no silent substitute score permitted")
        sa=float(eg.sascorer.calculateScore(mol))
    if sa>q["max_sa"]:
        return None,"sa_gate"
    alerts=cached.get("structural_alerts")
    if alerts is None:
        import evolutionary_generator as eg
        alerts=eg.structural_alerts(mol)
    if alerts>q["max_alerts"]:
        return None,"structural_alert_gate"
    parent_mol=Chem.MolFromSmiles(reference["smiles"])
    similarity=float(DataStructs.TanimotoSimilarity(_FP.GetFingerprint(parent_mol),_FP.GetFingerprint(mol)))
    band=job["band"]
    if not band["sim_lo"]<=similarity<=band["sim_hi"]:
        return None,"normalized_similarity_gate"
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return None,"radical_gate"
    row.update(sa_score=sa,structural_alerts=int(alerts),similarity_to_parent=similarity)
    row["source_id"]=job["job_id"]
    row["source_type"]="named_reference_"+job["method"]
    row["source_metadata"]={"reference_id":reference["reference_id"],"reference_name":reference["name"],
        "category":job["category"],"difficulty":job["difficulty"],"config_hash":job["config_hash"],
        "scenario":job["scenario"],"chemical_category_verification":proposal.get("chemical_category_verification",proposal.get("verification_status","not_verified"))}
    return {"molecule":row,"proposal":proposal},None


def _run_job(job):
    reference=job["reference"]
    q=job["quality"]
    band=job["band"]
    start=time.monotonic()
    report={}
    if job["method"]=="M":
        from m_proposal_engine import generate_with_report
        settings=job["mutation"]
        proposals,report=generate_with_report(reference,job["category"],job["seed"],settings["max_candidates"],
            depth=band["m_depth"],sim_lo=band["sim_lo"],sim_hi=band["sim_hi"],difficulty=job["difficulty"],
            subcategory=job.get("target_subcategory"),rules_path=job["rules_path"],
            beam_width=settings["beam_width"],max_rule_applications_per_node=settings["max_rule_applications_per_node"],
            max_products_per_rule=settings["max_products_per_rule"],qed_min=q["qed_min"],max_sa=q["max_sa"],max_alerts=q["max_alerts"],
            min_heavy_atoms=q["min_heavy_atoms"],max_heavy_atoms=q["max_heavy_atoms"],min_mw=q["min_mw"],max_mw=q["max_mw"])
    else:
        from evolve_category_pilot import generate_for_reference_e
        settings=job["evolution"]
        proposals=generate_for_reference_e(reference,job["category"],job["seed"],settings["max_candidates"],
            sim_lo=band["sim_lo"],sim_hi=band["sim_hi"],similarity_target=band["similarity_target"],
            scaffold_penalty=0.0 if job["category"]=="scaffold-hop" or job["difficulty"]=="distant" else .35,
            pop=settings["pop"],generations=settings["generations"],qed_min=q["qed_min"],max_sa=q["max_sa"],
            rules_path=job["rules_path"],difficulty=job["difficulty"],scenario=job["scenario"],
            crossover_op=settings["crossover_op"],max_pairwise_similarity=settings["max_pairwise_similarity"],
            fitness_weights=settings.get("fitness_weights"),
            child_mw_min=q["min_mw"],child_mw_max=q["max_mw"],child_heavy_min=q["min_heavy_atoms"],child_heavy_max=q["max_heavy_atoms"])
    accepted=[]
    rejected=Counter()
    for proposal in proposals:
        candidate,reason=checked_candidate(proposal,reference,job)
        if reason:
            rejected[reason]+=1
        else:
            accepted.append(candidate)
    return {"job_id":job["job_id"],"candidates":accepted,"produced":len(proposals),
            "rejections":dict(rejected),"proposal_report":report,"seconds":time.monotonic()-start}


def run_job(job):
    # pymoo's seed alone does not seed the Python operators used by crossover.
    import numpy as np
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    def timed_out(signum, frame):
        raise TimeoutError("Generation exceeded the recorded per-job time budget")
    previous=signal.signal(signal.SIGALRM,timed_out)
    signal.alarm(int(job.get("timeout_seconds",90)))
    try:
        return _run_job(job)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)


def reference_priority(reference, quality):
    """Prioritize feasible seeds; never replace the candidate acceptance gates."""
    import evolutionary_generator as eg
    from rdkit.Chem import Descriptors
    mol=Chem.MolFromSmiles(reference["smiles"])
    mw=float(Descriptors.MolWt(mol))
    qed=float(Descriptors.qed(mol))
    sa=float(eg.sascorer.calculateScore(mol))
    alerts=eg.structural_alerts(mol)
    basic=(quality["min_mw"]<=mw<=quality["max_mw"] and
           quality["min_heavy_atoms"]<=mol.GetNumHeavyAtoms()<=quality["max_heavy_atoms"])
    tier=0 if basic and qed>=quality["qed_min"] and sa<=quality["max_sa"] and alerts==0 else 1 if basic and alerts==0 else 2
    return tier,{"mw":mw,"qed":qed,"sa_score":sa,"structural_alerts":alerts,
                 "priority_basis":"parent_feasibility_only; candidate_gates_unchanged"}


def register_references(store,rows):
    for reference in rows:
        row=prepare_catalogue_row({**reference,"label":"0","molecule_id":reference["reference_id"],"molecule_source":"ChEMBL_named_reference"})
        if "error" in row:
            raise ValueError(f"Invalid named reference {reference['reference_id']}: {row['error']}")
        if not row["training_eligible"]:
            raise ValueError(f"Named generation seed overlaps external holdout: {reference['reference_id']}")
        store.insert_molecule(row,"reference_registry")
        store.connection.execute("INSERT OR IGNORE INTO references_catalogue VALUES(?,?,?,?,?,?,?,?,?)",
            (reference["reference_id"],reference["name"],row["smiles"],row["molecule_id"],reference["source"],reference["source_id"],
             reference["status"],int(reference["generation_eligible"]),json.dumps(reference,sort_keys=True)))
    store.connection.commit()


def cell_counts(store,config_hash):
    # Allocate production budget by genuinely new catalogue identities, not
    # by repeated parent relations or cross-cell appearances of one molecule.
    return {(r[0],r[1],r[2]):r[3] for r in store.connection.execute("""SELECT method,category,difficulty,SUM(new_molecules)
        FROM jobs WHERE state='complete' AND json_extract(config_json,'$.config_hash')=? GROUP BY method,category,difficulty""",(config_hash,))}


def fine_counts(store):
    # Imported primary-category counts are an initial coverage signal; they do
    # not magically verify the old records' chemical edit class.
    rows=store.connection.execute("""WITH coverage AS (
        SELECT molecule_id,json_extract(metadata_json,'$.category') AS category,
          json_extract(metadata_json,'$.subcategory') AS subcategory
        FROM molecule_sources WHERE source_type='final_dataset_400k'
        UNION SELECT molecule_id,category,subcategory FROM relations WHERE method='M' AND job_id IS NOT NULL)
        SELECT category,subcategory,COUNT(*) FROM coverage GROUP BY category,subcategory""")
    return Counter({(row[0],row[1]):row[2] for row in rows})


def build_round(store,refs,config,config_hash,round_no):
    from m_proposal_engine import applicable_subcategories
    inserted=0
    for reference in refs:
        priority=reference.get("scheduling_priority",2)
        feasibility=reference.get("scheduling_feasibility",{})
        reference={k:reference.get(k,"") for k in ("reference_id","name","smiles","source","source_id","status","generation_eligible")}
        for category in config["categories"]:
            applicable=applicable_subcategories(reference,category,config["rules_path"])
            if not applicable:
                continue
            for difficulty,band in config["difficulty_bands"].items():
                for method in ("M","E"):
                    job_key=f"{config_hash}|{round_no}|{reference['reference_id']}|{category}|{difficulty}|{method}"
                    job_id="job_"+stable_hash(job_key)
                    job={"job_id":job_id,"method":method,"reference":reference,"category":category,
                        "difficulty":difficulty,"band":band,"seed":int(stable_hash(job_key,8),16),
                        "config_hash":config_hash,"rules_path":config["rules_path"],"quality":config["quality"],
                        "evolution":config["evolution"],"mutation":config["mutation"],"scenario":config["scenario"],
                        "applicable_subcategories":applicable,"round":round_no,
                        "reference_priority":priority,"reference_feasibility":feasibility,
                        "timeout_seconds":config.get("job_timeout_seconds",90)}
                    before=store.connection.total_changes
                    store.connection.execute("""INSERT OR IGNORE INTO jobs(job_id,method,reference_id,category,difficulty,round,seed,config_json)
                        VALUES(?,?,?,?,?,?,?,?)""",(job_id,method,reference["reference_id"],category,difficulty,round_no,job["seed"],json.dumps(job,sort_keys=True)))
                    inserted+=store.connection.total_changes>before
    store.connection.commit()
    print(f"Round {round_no}: {inserted:,} new resumable jobs",flush=True)


def write_progress(store,target,start,completed,output):
    summary=store.summary()
    summary.update(target_unique_molecules=target,target_reached=summary["unique_molecules"]>=target,
        jobs_by_state=dict(store.connection.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state")),
        session_jobs_completed=completed,session_elapsed_seconds=round(time.monotonic()-start,1),
        generated_label_semantics="computational synthetic origin, not clinical/product falsification")
    temporary=output.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary,indent=2)+"\n")
    temporary.replace(output)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",type=Path,default=Path("molecule_bank_1m/bank.sqlite"))
    parser.add_argument("--config",type=Path,default=Path("million_bank_config.json"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--max-jobs",type=int,default=None,help="Optional staged/pilot session limit; ledger remains resumable")
    parser.add_argument("--reference-limit",type=int,default=None,help="Pilot seed subset; do not change a production run's reference file")
    parser.add_argument("--prepare-only",action="store_true",help="Register references/jobs without running generation")
    parser.add_argument("--stop-file",type=Path,default=Path("molecule_bank_1m/STOP_GENERATION"),help="Drain current jobs and stop when this file exists")
    args=parser.parse_args()
    args.db.parent.mkdir(parents=True,exist_ok=True)
    coordinator_lock=args.db.with_suffix(".generation.lock").open("a+")
    try:
        fcntl.flock(coordinator_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another generation coordinator is already using this bank; no job states were reset.")
    coordinator_lock.seek(0)
    coordinator_lock.truncate()
    coordinator_lock.write(str(os.getpid())+"\n")
    coordinator_lock.flush()
    config=json.loads(args.config.read_text())
    holdout=json.loads(Path(config["holdout"]).read_text())
    if config["quality"]["max_alerts"]!=0:
        raise ValueError("The current E analogue backend supports an absolute structural-alert gate of zero; use a separate explicitly implemented scenario profile for other gates.")
    chemistry_init(holdout)
    from named_reference_registry import load_reference_registry
    refs=load_reference_registry(config["reference_csv"])
    if args.reference_limit:
        refs=refs[:args.reference_limit]
    for reference in refs:
        reference["scheduling_priority"],reference["scheduling_feasibility"]=reference_priority(reference,config["quality"])
    print("Named-parent feasibility tiers:",dict(Counter(r["scheduling_priority"] for r in refs)),flush=True)
    store=BankStore(args.db)
    holdout_hash=file_sha256(config["holdout"])
    previous_holdout=store.connection.execute("SELECT value FROM metadata WHERE key='external_holdout_sha256'").fetchone()
    if previous_holdout and previous_holdout[0]!=holdout_hash:
        raise ValueError("External holdout changed; rebuild/re-audit catalogue eligibility before resuming.")
    if not previous_holdout:
        blocked_identity=holdout.get("blocked_connectivity_smiles",[])
        blocked_scaffold=holdout.get("blocked_scaffold_keys",[])
        predicates=[]
        values=[]
        if blocked_identity:
            predicates.append("connectivity_smiles IN ("+",".join("?" for _ in blocked_identity)+")")
            values.extend(blocked_identity)
        if blocked_scaffold:
            predicates.append("scaffold_key IN ("+",".join("?" for _ in blocked_scaffold)+")")
            values.extend(blocked_scaffold)
        if predicates:
            store.connection.execute("UPDATE molecules SET training_eligible=0 WHERE "+" OR ".join(predicates),values)
        store.connection.execute("INSERT INTO events(kind,payload_json) VALUES('first_holdout_pin_conservative_recheck',?)",(json.dumps({"sha256":holdout_hash,"unblocks_old_exclusions":False}),))
    store.connection.execute("INSERT OR IGNORE INTO metadata VALUES('external_holdout_sha256',?)",(holdout_hash,))
    # Repair any pre-fix imported SA metadata conservatively; complete values
    # are recomputed and persisted when a normalized structure is used.
    if not store.connection.execute("SELECT 1 FROM metadata WHERE key='normalized_sa_cache_policy_v1'").fetchone():
        store.connection.execute("""UPDATE molecules SET sa_score=NULL WHERE molecule_id IN
          (SELECT s.molecule_id FROM molecule_sources s JOIN molecules m ON m.molecule_id=s.molecule_id WHERE s.raw_smiles<>m.smiles)""")
        store.connection.execute("""UPDATE descriptor_cache SET properties_json=json_set(properties_json,'$.sa_score',NULL)
          WHERE smiles IN (SELECT smiles FROM molecules WHERE sa_score IS NULL)""")
        store.connection.execute("INSERT INTO metadata VALUES('normalized_sa_cache_policy_v1','invalidated_source_form_SA')")
    store.connection.commit()
    register_references(store,refs)
    input_pins={"rules_sha256":file_sha256(config["rules_path"]),"reference_csv_sha256":file_sha256(config["reference_csv"]),
        "holdout_sha256":holdout_hash,"code_sha256":{path:file_sha256(path) for path in
        ("expand_million_bank.py","scalable_bank_store.py","m_proposal_engine.py","evolutionary_generator.py","evolve_category_pilot.py","named_reference_registry.py")}}
    config_hash=stable_hash(json.dumps({"config":config,"pins":input_pins},sort_keys=True))
    metadata_key="production_config_"+config_hash
    store.connection.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)",(metadata_key,json.dumps({"config":config,"pins":input_pins},sort_keys=True)))
    store.connection.execute("UPDATE jobs SET state='superseded_configuration' WHERE state='pending' AND json_extract(config_json,'$.config_hash')<>?",(config_hash,))
    store.connection.execute("UPDATE jobs SET state='superseded_configuration' WHERE state='running' AND json_extract(config_json,'$.config_hash')<>?",(config_hash,))
    # Recover only jobs belonging to this configuration after interruption.
    store.connection.execute("UPDATE jobs SET state='pending' WHERE state='running' AND json_extract(config_json,'$.config_hash')=?",(config_hash,))
    store.connection.commit()
    target=int(config["target_unique_molecules"])
    baseline_key="baseline_"+config_hash
    saved=store.connection.execute("SELECT value FROM metadata WHERE key=?",(baseline_key,)).fetchone()
    if saved:
        baseline=json.loads(saved[0])
    else:
        baseline=store.summary()
        store.connection.execute("INSERT INTO metadata VALUES(?,?)",(baseline_key,json.dumps(baseline)))
        store.connection.commit()
    remaining=max(0,target-baseline["unique_molecules"])
    origins=baseline["method_origin_counts"]
    equal_target=(origins.get("M",0)+origins.get("E",0)+remaining)/2
    needed={method:max(0,equal_target-origins.get(method,0)) for method in ("M","E")}
    if sum(needed.values())>remaining and remaining:
        needed={k:v*remaining/sum(needed.values()) for k,v in needed.items()}
    n_cells=len(config["categories"])*len(config["difficulty_bands"])
    desired={(method,category,difficulty):max(1,needed[method]/n_cells)
             for method in ("M","E") for category in config["categories"] for difficulty in config["difficulty_bands"]}
    print(f"Bank={store.count():,}; target={target:,}; new-method budgets={needed}",flush=True)
    rounds={r[0] for r in store.connection.execute("SELECT DISTINCT round FROM jobs WHERE json_extract(config_json,'$.config_hash')=?",(config_hash,))}
    build_round(store,refs,config,config_hash,0)
    rounds.add(0)
    start=time.monotonic()
    completed=0
    progress=args.db.parent/"progress.json"
    counts=cell_counts(store,config_hash)
    attempts=Counter({(r[0],r[1],r[2]):r[3] for r in store.connection.execute("""SELECT method,category,difficulty,SUM(attempts)
        FROM jobs WHERE json_extract(config_json,'$.config_hash')=? GROUP BY method,category,difficulty""",(config_hash,))})
    current_total=store.count()
    last_progress=start
    fine=fine_counts(store)
    fine_failed=Counter()
    store.connection.execute("CREATE INDEX IF NOT EXISTS relation_reference_molecule ON relations(reference_id,molecule_id)")
    store.connection.execute("CREATE INDEX IF NOT EXISTS relation_molecule_cell ON relations(molecule_id,method,category,subcategory)")
    store.connection.commit()
    reference_counts=dict(store.connection.execute("SELECT reference_id,COUNT(DISTINCT molecule_id) FROM relations WHERE job_id IS NOT NULL GROUP BY reference_id"))
    scaffold_counts=dict(store.connection.execute("SELECT scaffold_key,COUNT(*) FROM molecules WHERE first_source='named_reference_generation' GROUP BY scaffold_key"))
    pending_by_cell=defaultdict(list)

    def refresh_pending():
        pending_by_cell.clear()
        for row in store.connection.execute("SELECT job_id,method,category,difficulty,config_json FROM jobs WHERE state='pending' AND json_extract(config_json,'$.config_hash')=? ORDER BY CAST(json_extract(config_json,'$.reference_priority') AS INTEGER) DESC,job_id DESC",(config_hash,)):
            pending_by_cell[row["method"],row["category"],row["difficulty"]].append((row["job_id"],row["config_json"]))

    refresh_pending()
    inflight=Counter()
    if args.prepare_only:
        print(json.dumps(write_progress(store,target,start,completed,progress),indent=2))
        return

    def next_job():
        cells=[cell for cell,queue in pending_by_cell.items() if queue]
        if not cells:
            return None
        # Relative coverage deficit; queued work counts toward scheduling only,
        # never toward delivered molecule counts.
        cell=max(cells,key=lambda key:(max(1,desired[key]-counts.get(key,0))/math.sqrt(1+attempts[key]+inflight[key]),-counts.get(key,0),key))
        job_id,payload=pending_by_cell[cell].pop()
        job=json.loads(payload)
        if job["method"]=="M" and not job.get("target_subcategory"):
            options=job["applicable_subcategories"]
            selected=min(options,key=lambda sub:(fine[(job["category"],sub)]+fine_failed[job["category"],sub]*5,sub))
            job["target_subcategory"]=selected
        store.connection.execute("UPDATE jobs SET state='running',attempts=attempts+1,started_at=CURRENT_TIMESTAMP,config_json=? WHERE job_id=?",
                                 (json.dumps(job,sort_keys=True),job_id))
        store.connection.commit()
        inflight[cell]+=1
        return job

    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(str(args.db),holdout,config["quality"])) as pool:
        running={}
        while True:
            while not args.stop_file.exists() and len(running)<args.workers and current_total<target and (args.max_jobs is None or completed+len(running)<args.max_jobs):
                job=next_job()
                if job is None:
                    break
                running[pool.submit(run_job,job)]=job
            if not running:
                if args.stop_file.exists() or current_total>=target or (args.max_jobs is not None and completed>=args.max_jobs):
                    break
                round_no=max(rounds)+1
                if round_no>=config["max_rounds"]:
                    print("All configured rounds exhausted; inspect coverage/yield before changing chemistry gates.",flush=True)
                    break
                build_round(store,refs,config,config_hash,round_no)
                rounds.add(round_no)
                refresh_pending()
                continue
            ready,_=wait(running,timeout=15,return_when=FIRST_COMPLETED)
            if not ready:
                write_progress(store,target,start,completed,progress)
                continue
            for future in ready:
                job=running.pop(future)
                cell=(job["method"],job["category"],job["difficulty"])
                inflight[cell]-=1
                try:
                    result=future.result()
                    added=relations_added=0
                    diversity_rejections=Counter()
                    with store.connection:
                        for candidate in result["candidates"]:
                            row,proposal=candidate["molecule"],candidate["proposal"]
                            reference_id=job["reference"]["reference_id"]
                            prior_reference=store.connection.execute("SELECT 1 FROM relations WHERE reference_id=? AND molecule_id=? LIMIT 1",(reference_id,row["molecule_id"])).fetchone()
                            if not prior_reference and reference_counts.get(reference_id,0)>=config["diversity"]["max_named_molecules_per_reference"]:
                                diversity_rejections["reference_cap"]+=1
                                continue
                            exists=store.connection.execute("SELECT 1 FROM molecules WHERE molecule_id=?",(row["molecule_id"],)).fetchone()
                            if not exists and scaffold_counts.get(row["scaffold_key"],0)>=config["diversity"]["max_new_molecules_per_child_scaffold"]:
                                diversity_rejections["child_scaffold_cap"]+=1
                                continue
                            is_new=store.insert_molecule(row,"named_reference_generation")
                            added+=is_new
                            if is_new:
                                scaffold_counts[row["scaffold_key"]]=scaffold_counts.get(row["scaffold_key"],0)+1
                            # Persistent full-quality cache is reusable by later
                            # jobs and future resumed worker processes.
                            properties={k:row.get(k) for k in ("mw","logp","qed","sa_score","heavy_atoms","scaffold_key","structural_alerts")}
                            store.connection.execute("UPDATE descriptor_cache SET properties_json=? WHERE smiles=?",(json.dumps(properties,sort_keys=True),row["smiles"]))
                            history=proposal.get("operator_history_json") or proposal.get("operator_history") or "[]"
                            if isinstance(history,(list,dict)):
                                history=json.dumps(history,sort_keys=True)
                            # Validate stored lineage instead of silently losing
                            # a generator-specific history representation.
                            json.loads(history)
                            verification=proposal.get("chemical_category_verification") or proposal.get("verification_status","not_verified")
                            subcategory=proposal.get("subcategory",job["category"])
                            relation_id="rel_"+stable_hash("|".join((row["molecule_id"],job["reference"]["reference_id"],job["method"],job["category"],job["difficulty"],history)))
                            before=store.connection.total_changes
                            prior_fine=True
                            if job["method"]=="M":
                                prior_fine=bool(store.connection.execute("SELECT 1 FROM relations WHERE molecule_id=? AND method='M' AND category=? AND subcategory=? LIMIT 1",(row["molecule_id"],job["category"],subcategory)).fetchone()
                                    or store.connection.execute("""SELECT 1 FROM molecule_sources WHERE molecule_id=? AND source_type='final_dataset_400k'
                                       AND json_extract(metadata_json,'$.category')=? AND json_extract(metadata_json,'$.subcategory')=? LIMIT 1""",(row["molecule_id"],job["category"],subcategory)).fetchone())
                            store.connection.execute("""INSERT OR IGNORE INTO relations
                                (relation_id,molecule_id,reference_id,parent_smiles,method,category,subcategory,scenario,difficulty,
                                 protocol,operator_history_json,verification_status,evidence_tier,similarity,objective_json,source_id,job_id)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (relation_id,row["molecule_id"],job["reference"]["reference_id"],job["reference"]["smiles"],job["method"],job["category"],subcategory,
                                 job["scenario"],job["difficulty"],proposal.get("generation_protocol","named_reference_generation"),history,verification,
                                 "synthetic_generator_relation",row["similarity_to_parent"],
                                 json.dumps({"config_hash":config_hash,"objective_score":proposal.get("objective_score"),"proposal_fitness":proposal.get("objectives"),
                                             "operator_history_status":proposal.get("operator_history_status", "recorded_mmp_intermediates")},sort_keys=True),
                                 proposal.get("source_id",""),job["job_id"]))
                            inserted=store.connection.total_changes>before
                            relations_added+=inserted
                            if inserted and not prior_reference:
                                reference_counts[reference_id]=reference_counts.get(reference_id,0)+1
                            if inserted and job["method"]=="M" and not prior_fine:
                                fine[job["category"],subcategory]+=1
                        store.connection.execute("""UPDATE jobs SET state='complete',produced=?,new_molecules=?,relations_added=?,rejections_json=?,finished_at=CURRENT_TIMESTAMP WHERE job_id=?""",
                            (result["produced"],added,relations_added,json.dumps({"quality":result["rejections"],"diversity":dict(diversity_rejections),"proposal_report":result["proposal_report"],"seconds":result["seconds"]},sort_keys=True),job["job_id"]))
                    if not relations_added and job["method"]=="M":
                        fine_failed[job["category"],job.get("target_subcategory","")]+=1
                    current_total+=added
                    counts[cell]=counts.get(cell,0)+added
                except Exception:
                    error=traceback.format_exc()
                    # The transaction may have rolled back after changing
                    # provisional in-memory counters. Restore all counters
                    # from committed state before allocating another job.
                    counts=cell_counts(store,config_hash)
                    current_total=store.count()
                    fine=fine_counts(store)
                    reference_counts=dict(store.connection.execute("SELECT reference_id,COUNT(DISTINCT molecule_id) FROM relations WHERE job_id IS NOT NULL GROUP BY reference_id"))
                    scaffold_counts=dict(store.connection.execute("SELECT scaffold_key,COUNT(*) FROM molecules WHERE first_source='named_reference_generation' GROUP BY scaffold_key"))
                    store.connection.execute("UPDATE jobs SET state='error',error=?,finished_at=CURRENT_TIMESTAMP WHERE job_id=?",(error,job["job_id"]))
                    store.connection.commit()
                    print(f"Job failed {job['job_id']}: {error.splitlines()[-1]}",flush=True)
                    added=relations_added=0
                completed+=1
                attempts[cell]+=1
                if completed%50==0 or time.monotonic()-last_progress>=30:
                    summary=write_progress(store,target,start,completed,progress)
                    print(f"jobs={completed:,} bank={summary['unique_molecules']:,}/{target:,} last={job['method']}/{job['category']}/{job['difficulty']} +{added} molecules ({relations_added} relations)",flush=True)
                    last_progress=time.monotonic()
        summary=write_progress(store,target,start,completed,progress)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
