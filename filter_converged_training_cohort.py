#!/usr/bin/env python3
"""Derive the common cohort using converged, finite-energy force-field geometries.

Uses no model outputs. Both 2D and 3D runs receive the same remaining examples.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import numpy as np
from audit_overnight_training_split import read_csv, write_csv, sha256, rank, encoded, set_training_weights, overlap_report


def build(source,out):
    source,out=Path(source),Path(out)
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((source/'manifest.json').read_text())
    for name,pin in manifest['selected_files'].items():
        if sha256(source/name)!=pin['sha256']:raise ValueError('Source cohort changed: '+name)
    rows,fields=read_csv(source/'dataset.csv')
    graphs,gfields=read_csv(source/'graph_index.csv')
    reasons={}
    by_file={}
    for g in graphs:by_file.setdefault(g['graph_file'],[]).append(g)
    for filename,items in by_file.items():
        if sha256(filename)!=manifest['graph_shards'][filename]['sha256']:
            raise ValueError('Original graph source changed: '+filename)
        with np.load(filename,allow_pickle=False) as archive:
            converged=archive['optimization_converged']
            energies=archive['forcefield_energy_kcal_mol']
            identifiers=archive['source_id']
            for g in items:
                i=int(g['graph_index']);why=[]
                if str(identifiers[i])!=g['source_id']:raise ValueError('Source graph ID mismatch')
                if not bool(converged[i]):why.append('forcefield_optimization_not_converged')
                if not math.isfinite(float(energies[i])):why.append('nonfinite_forcefield_energy')
                if why:reasons[g['source_id']]=why
    selected=[r for r in rows if r['molecule_id'] not in reasons]
    for split in ('val','test'):
        groups={label:sorted([r for r in selected if r['split']==split and r['label']==label],
                key=lambda r:rank(r['molecule_id'],manifest['seed'],'converged_heldout_balance')) for label in ('0','1')}
        n=min(map(len,groups.values()))
        if not n:raise ValueError('Empty heldout class after convergence filtering')
        for group in groups.values():
            for row in group[n:]:reasons[row['molecule_id']]=['deterministic_convergence_heldout_label_balance']
    selected=[r for r in selected if r['molecule_id'] not in reasons]
    weights=set_training_weights(selected)
    by_id={r['molecule_id']:r for r in selected}
    selected_graphs=[dict(g,sample_weight=by_id[g['source_id']]['sample_weight']) for g in graphs if g['source_id'] in by_id]
    overlaps=overlap_report(selected)
    if any(v for pair in overlaps.values() for v in pair.values()):raise ValueError('Derived cohort split overlap')
    if sum(r['split']=='train' for r in selected)<=100000:raise ValueError('Training below required 100k')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name,dir=out.parent))
    try:
        write_csv(stage/'dataset.csv',selected,fields)
        write_csv(stage/'graph_index.csv',selected_graphs,gfields)
        excluded,efields=read_csv(source/'excluded_rows.csv')
        excluded += [{'source_id':r['molecule_id'],'original_split':r['split'],'original_label':r['label'],
                      'reasons_json':encoded(reasons[r['molecule_id']])} for r in rows if r['molecule_id'] in reasons]
        write_csv(stage/'excluded_rows.csv',excluded,efields)
        shard_counts=Counter(g['graph_file'] for g in selected_graphs)
        manifest.update(rows=len(selected),train_rows=sum(r['split']=='train' for r in selected),
            split_label_counts={s:dict(Counter(r['label'] for r in selected if r['split']==s)) for s in ('train','val','test')},
            weight_policy=weights,after_overlap=overlaps,excluded_unique_source_rows=len(excluded),
            parent_manifest_sha256=sha256(source/'manifest.json'),
            derived_geometry_filter={'required_convergence':True,'required_finite_energy':True,
                'removed_rows':len(reasons),'reason_counts':dict(Counter(x for v in reasons.values() for x in v)),
                'source_cohort':str(source),'model_results_used':False,'script_sha256':sha256(__file__)},
            cohort_sha256=hashlib.sha256(encoded([[r['molecule_id'],r['split'],r['label']] for r in sorted(selected,key=lambda r:r['molecule_id'])]).encode()).hexdigest())
        manifest['exclusion_reasons']=dict(Counter(manifest['exclusion_reasons'])+Counter(x for v in reasons.values() for x in v))
        manifest['graph_shards']={f:dict(pin,selected_graphs=shard_counts[f]) for f,pin in manifest['graph_shards'].items() if shard_counts[f]}
        manifest['selected_files']={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in stage.iterdir()}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        stage.replace(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    print(json.dumps({k:manifest[k] for k in ('rows','train_rows','split_label_counts','derived_geometry_filter','cohort_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('balanced_datasets/training_200k/overnight_strict_v1'))
    p.add_argument('--out',type=Path,default=Path('balanced_datasets/training_200k/overnight_strict_converged_v2'))
    a=p.parse_args();build(a.source,a.out)
