#!/usr/bin/env python3
"""Bounded coverage and accumulated cooling over unchanged molecular chemistry.

Uses unchanged chemistry, QC, holdout, caps and lineage from the frozen core.
Category coverage has explicit floors; difficulty and subsequent method
allocation learn actual unique yield per active chemistry second. Completed
bank identities/history remain unchanged. Only this version's pending jobs
are recovered; old pending configurations are explicitly superseded.
"""
from __future__ import annotations
import argparse,csv,fcntl,json,math,os,random,sqlite3,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
from scalable_bank_store import BankStore,chemistry_init,prepare_catalogue_row,stable_hash,file_sha256
from expand_million_bank import (register_references,reference_priority,cell_counts,fine_counts,build_round,write_progress as core_write_progress)
from generation_job_host import NativeJobHost,ThermalGate,NativeJobTimeout,JobDeferred,JobHostError,acquire_compute_lease
from yield_aware_scheduling import category_coverage
from bounded_bank_scheduling import BoundedCoverageScheduler as YieldAwareScheduler,load_scheduler,DEFAULT_POLICY
from accumulated_generation_cooling import AccumulatedThermalGate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",type=Path,default=Path("molecule_bank_1m/bank.sqlite"))
    parser.add_argument("--config",type=Path,default=Path("million_bank_bounded_cool_config_20261109.json"))
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--max-jobs",type=int,default=None,help="Optional staged/pilot session limit; ledger remains resumable")
    parser.add_argument("--reference-limit",type=int,default=None,help="Pilot seed subset; do not change a production run's reference file")
    parser.add_argument("--prepare-only",action="store_true",help="Register references/jobs without running generation")
    parser.add_argument("--stop-file",type=Path,default=Path("molecule_bank_1m/STOP_GENERATION"),help="Drain current jobs and stop when this file exists")
    parser.add_argument("--thermal-status-dir",type=Path,default=None)
    parser.add_argument("--progress-seconds",type=float,default=120.0)
    args=parser.parse_args()
    if args.workers!=1:
        raise ValueError("Yield-aware cool generation permits exactly one chemistry compute child")
    if not math.isfinite(args.progress_seconds) or args.progress_seconds<30:
        raise ValueError("Full inventory reports must be at least30seconds apart")
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
    # Acquire before any DB recovery. A child orphaned by SIGKILL retains this
    # lease and makes a replacement coordinator fail closed without DB writes.
    compute_lease=acquire_compute_lease(args.db.with_suffix(".chemistry.lock"))
    config=json.loads(args.config.read_text())
    config.setdefault("yield_scheduler",dict(DEFAULT_POLICY))
    config.setdefault("execution_policy",{"version":"owned_native_child_v1","compute_children":1,
        "hard_wall_seconds":config.get("job_timeout_seconds",90),"nominal_rest_seconds":1,
        "sustained_nominal_seconds":30})
    execution=config["execution_policy"]
    execution.setdefault("nominal_rest_seconds",1)
    execution.setdefault("sustained_nominal_seconds",30)
    execution.setdefault("python_hash_seed",0)
    if execution["compute_children"]!=1 or execution["hard_wall_seconds"]!=config.get("job_timeout_seconds",90):
        raise ValueError("Execution policy must match one child and the recorded per-job deadline")
    if not math.isfinite(execution["hard_wall_seconds"]) or execution["hard_wall_seconds"]<=0:
        raise ValueError("Execution deadline must be finite and positive")
    if execution.get("version")!="accumulated_active_wall_v1":
        raise ValueError("This version requires explicit accumulated active-wall cooling metadata")
    thermal=AccumulatedThermalGate(args.thermal_status_dir or args.db.parent/"thermal_control_yield",
        rest_seconds=execution["minimum_rest_seconds"],
        sustained_nominal_seconds=execution["sustained_nominal_seconds"],
        nominal_active_budget_seconds=execution["nominal_active_budget_seconds"],
        nominal_rest_ratio=execution["nominal_rest_ratio"],
        fair_active_budget_seconds=execution["fair_active_budget_seconds"],
        fair_rest_ratio=execution["fair_rest_ratio"])
    def cooled_call(label,function,*values):
        thermal.wait(label,args.stop_file.exists)
        try:
            return function(*values)
        finally:
            thermal.finished()
    YieldAwareScheduler(config["yield_scheduler"])  # Validate before bank mutations.
    holdout=json.loads(Path(config["holdout"]).read_text())
    if config["quality"]["max_alerts"]!=0:
        raise ValueError("The current E analogue backend supports an absolute structural-alert gate of zero; use a separate explicitly implemented scenario profile for other gates.")
    chemistry_init(holdout)
    from named_reference_registry import load_reference_registry
    refs=load_reference_registry(config["reference_csv"])
    if args.reference_limit:
        refs=refs[:args.reference_limit]
    for offset in range(0,len(refs),32):
        def prepare_reference_batch():
            for reference in refs[offset:offset+32]:
                reference["scheduling_priority"],reference["scheduling_feasibility"]=reference_priority(reference,config["quality"])
        cooled_call("startup_reference_feasibility_"+str(offset),prepare_reference_batch)
    print("Named-parent feasibility tiers:",dict(Counter(r["scheduling_priority"] for r in refs)),flush=True)
    store=BankStore(args.db)
    # Coordinator-local indexes avoid rescanning old configurations and large
    # lineage payloads during recovery/progress/coverage queries.
    store.connection.execute("CREATE INDEX IF NOT EXISTS jobs_config_state_yield ON jobs(json_extract(config_json,'$.config_hash'),state)")
    store.connection.execute("CREATE INDEX IF NOT EXISTS jobs_state_finished_yield ON jobs(state,finished_at)")
    store.connection.execute("CREATE INDEX IF NOT EXISTS relation_named_coverage_yield ON relations(method,category,molecule_id) WHERE job_id IS NOT NULL")
    store.connection.commit()
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
    for offset in range(0,len(refs),128):
        cooled_call("startup_register_references_"+str(offset),register_references,store,refs[offset:offset+128])
    input_pins={"rules_sha256":file_sha256(config["rules_path"]),"reference_csv_sha256":file_sha256(config["reference_csv"]),
        "holdout_sha256":holdout_hash,"code_sha256":{path:file_sha256(path) for path in
        ("expand_bounded_cool_bank.py","bounded_bank_scheduling.py","accumulated_generation_cooling.py","expand_yield_aware_bank.py","generation_job_host.py","yield_aware_scheduling.py","expand_million_bank.py","scalable_bank_store.py","m_proposal_engine.py","evolutionary_generator.py","evolve_category_pilot.py","named_reference_registry.py")}}
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
    print(f"Bank={store.count():,}; target={target:,}; method parity soft; category/time-floor policy={config['yield_scheduler']}",flush=True)
    rounds={r[0] for r in store.connection.execute("SELECT DISTINCT round FROM jobs WHERE json_extract(config_json,'$.config_hash')=?",(config_hash,))}
    cooled_call("startup_build_round_0",build_round,store,refs,config,config_hash,0)
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
    scheduler=cooled_call("startup_scheduler_history",load_scheduler,store.connection,config,config_hash,baseline["method_origin_counts"])
    pending_by_cell=defaultdict(list)

    def refresh_pending():
        pending_by_cell.clear()
        for row in store.connection.execute("SELECT job_id,method,category,difficulty,config_json FROM jobs WHERE state='pending' AND json_extract(config_json,'$.config_hash')=? ORDER BY CAST(json_extract(config_json,'$.reference_priority') AS INTEGER) DESC,job_id DESC",(config_hash,)):
            pending_by_cell[row["method"],row["category"],row["difficulty"]].append((row["job_id"],row["config_json"]))

    refresh_pending()
    if args.prepare_only:
        print(json.dumps(core_write_progress(store,target,start,completed,progress),indent=2))
        return

    def next_job():
        cells=[cell for cell,queue in pending_by_cell.items() if queue]
        if not cells:
            return None
        scheduler.volume_target_reached=current_total>=target
        cell=scheduler.choose(cells)
        job_id,payload=pending_by_cell[cell].pop()
        job=json.loads(payload)
        job["scheduler_decision"]=scheduler.last_decision
        job["execution_backend"]="owned_persistent_native_child_hard_deadline_v1"
        job["execution_policy_version"]=execution["version"]
        if job["method"]=="M":
            job["target_subcategory"]=scheduler.choose_fine_target(job,fine)
            job["fine_target_decision"]=dict(scheduler.last_fine_decision)
        store.connection.execute("UPDATE jobs SET state='running',attempts=attempts+1,started_at=CURRENT_TIMESTAMP,config_json=? WHERE job_id=?",
                                 (json.dumps(job,sort_keys=True),job_id))
        store.connection.commit()
        return job

    with NativeJobHost(args.db,holdout,config["quality"],thermal_gate=thermal,
                       stop_requested=args.stop_file.exists,compute_lease=compute_lease,
                       python_hash_seed=execution["python_hash_seed"]) as host:
        while current_total<target or scheduler.unmet_coverage(config["categories"]):
            if args.stop_file.exists() or (args.max_jobs is not None and completed>=args.max_jobs):
                break
            job=next_job()
            if job is None:
                round_no=max(rounds)+1
                if round_no>=config["max_rounds"]:
                    print("All configured rounds exhausted; inspect yield/coverage without relaxing chemical gates.",flush=True)
                    break
                cooled_call("build_round_"+str(round_no),build_round,store,refs,config,config_hash,round_no)
                rounds.add(round_no)
                refresh_pending()
                continue
            cell=(job["method"],job["category"],job["difficulty"])
            try:
                result=host.run(job)
                added=relations_added=category_memberships_added=0
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
                        prior_named_category=store.connection.execute("""SELECT 1 FROM relations WHERE molecule_id=?
                            AND method=? AND category=? AND job_id IS NOT NULL LIMIT 1""",
                            (row["molecule_id"],job["method"],job["category"])).fetchone()
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
                        if inserted and not prior_named_category:
                            category_memberships_added+=1
                        if inserted and not prior_reference:
                            reference_counts[reference_id]=reference_counts.get(reference_id,0)+1
                        if inserted and job["method"]=="M" and not prior_fine:
                            fine[job["category"],subcategory]+=1
                    store.connection.execute("""UPDATE jobs SET state='complete',produced=?,new_molecules=?,relations_added=?,rejections_json=?,finished_at=CURRENT_TIMESTAMP WHERE job_id=?""",
                        (result["produced"],added,relations_added,json.dumps({"category_memberships_added":category_memberships_added,"quality":result["rejections"],"diversity":dict(diversity_rejections),"proposal_report":result["proposal_report"],"seconds":result["seconds"],"job_host_seconds":result["job_host_seconds"],"thermal_wait_seconds":result["thermal_wait_seconds"]},sort_keys=True),job["job_id"]))
                if not relations_added and job["method"]=="M":
                    fine_failed[job["category"],job.get("target_subcategory","")]+=1
                current_total+=added
                counts[cell]=counts.get(cell,0)+added
                scheduler.record(cell,added,result["job_host_seconds"],category_memberships=category_memberships_added)
                scheduler.record_fine(job,result["produced"])
                scheduler.coverage[cell[:2]]+=category_memberships_added-added
            except NativeJobTimeout:
                # This happens before the bank transaction starts. Its only
                # committed mutation so far is running-state/attempt accounting.
                error=traceback.format_exc()
                timing=json.dumps({"job_host_seconds":host.last_job_seconds,
                    "thermal_wait_seconds":host.last_thermal_wait_seconds,"hard_native_timeout":True},sort_keys=True)
                store.connection.execute("UPDATE jobs SET state='error',error=?,rejections_json=?,finished_at=CURRENT_TIMESTAMP WHERE job_id=?",
                    (error,timing,job["job_id"]))
                store.connection.commit()
                scheduler.record(cell,0,max(0.001,host.last_job_seconds))
                print(f"Hard native deadline {job['job_id']}: {host.last_job_seconds:.3f}s; owned child reaped",flush=True)
                added=relations_added=0
            except JobDeferred:
                store.connection.execute("UPDATE jobs SET state='pending' WHERE job_id=?",(job["job_id"],))
                store.connection.commit()
                break
            except JobHostError:
                store.connection.execute("UPDATE jobs SET state='pending' WHERE job_id=?",(job["job_id"],))
                store.connection.commit()
                raise  # Infrastructure faults must never consume the queue.
            except Exception as failure:
                error=traceback.format_exc()
                # The transaction may have rolled back after changing
                # provisional in-memory counters. Restore all counters
                # from committed state before allocating another job.
                counts=cell_counts(store,config_hash)
                current_total=store.count()
                fine=fine_counts(store)
                scheduler.coverage=category_coverage(store.connection)
                reference_counts=dict(store.connection.execute("SELECT reference_id,COUNT(DISTINCT molecule_id) FROM relations WHERE job_id IS NOT NULL GROUP BY reference_id"))
                scaffold_counts=dict(store.connection.execute("SELECT scaffold_key,COUNT(*) FROM molecules WHERE first_source='named_reference_generation' GROUP BY scaffold_key"))
                failed_timing=json.dumps({"job_host_seconds":host.last_job_seconds,
                    "thermal_wait_seconds":host.last_thermal_wait_seconds,
                    "hard_native_timeout":isinstance(failure,NativeJobTimeout)},sort_keys=True)
                store.connection.execute("UPDATE jobs SET state='error',error=?,rejections_json=?,finished_at=CURRENT_TIMESTAMP WHERE job_id=?",
                    (error,failed_timing,job["job_id"]))
                store.connection.commit()
                scheduler.record(cell,0,max(0.001,host.last_job_seconds))
                print(f"Job failed {job['job_id']}: {error.splitlines()[-1]}",flush=True)
                added=relations_added=0
            completed+=1
            attempts[cell]+=1
            if completed%50==0 or time.monotonic()-last_progress>=args.progress_seconds:
                summary=core_write_progress(store,target,start,completed,progress)
                summary["yield_scheduler"]=scheduler.summary()
                summary["accumulated_cooling"]=thermal.summary()
                summary["unmet_named_category_floors"]=scheduler.unmet_coverage(config["categories"])
                summary["coverage_floors_met"]=not summary["unmet_named_category_floors"]
                summary["execution_backend"]="one_owned_persistent_compute_child;90s_parent_deadline;accumulated_cooling;bounded_allocation"
                temporary=progress.with_suffix(".tmp")
                temporary.write_text(json.dumps(summary,indent=2)+"\n")
                temporary.replace(progress)
                print(f"jobs={completed:,} bank={summary['unique_molecules']:,}/{target:,} last={job['method']}/{job['category']}/{job['difficulty']} +{added} molecules ({relations_added} relations)",flush=True)
                last_progress=time.monotonic()
        summary=core_write_progress(store,target,start,completed,progress)
        summary["yield_scheduler"]=scheduler.summary()
        summary["accumulated_cooling"]=thermal.summary()
        summary["unmet_named_category_floors"]=scheduler.unmet_coverage(config["categories"])
        summary["coverage_floors_met"]=not summary["unmet_named_category_floors"]
        temporary=progress.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary,indent=2)+"\n")
        temporary.replace(progress)
    print(json.dumps(summary,indent=2))
    if not args.stop_file.exists() and args.max_jobs is None and (current_total<target or not summary["coverage_floors_met"]):
        raise SystemExit("Configured jobs exhausted before both inventory target and named-category coverage floors were met")


if __name__=="__main__":
    main()
