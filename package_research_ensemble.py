#!/usr/bin/env python3
"""Copy a frozen research ensemble and byte-exact provenance into a new bundle.

Uses only the standard library. Never loads a model, starts training, computes
predictions, chooses a threshold, or edits an original research artifact.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / 'bundle_manifest.json'
    expected_manifest_hash = (directory / 'bundle_manifest.sha256').read_text().split()[0]
    if digest(manifest_path) != expected_manifest_hash:
        raise ValueError('Bundle manifest checksum differs')
    manifest = read(manifest_path)
    listed = set()
    for entry in manifest['files']:
        path = directory / entry['relative_path']
        if not path.resolve().is_relative_to(directory) or path.is_symlink():
            raise ValueError('Bundle file escapes its directory')
        if entry['relative_path'] in listed:
            raise ValueError('Duplicate bundle path')
        listed.add(entry['relative_path'])
        if path.stat().st_size != entry['size_bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Bundle file changed: ' + entry['relative_path'])
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != listed | {'bundle_manifest.json', 'bundle_manifest.sha256'}:
        raise ValueError('Unlisted or missing bundle files')
    return {'verified': True, 'bundle': str(directory), 'files': len(listed),
            'members': len(manifest['members']), 'payload_bytes': sum(e['size_bytes'] for e in manifest['files']),
            'manifest_sha256': expected_manifest_hash, 'test_f1': manifest['test_metrics']['f1']}


def package(workspace, out):
    workspace, out = Path(workspace).resolve(), Path(out).resolve()
    ensemble = workspace / 'training_runs/heterogeneous_20261107'
    selection_path = ensemble / 'final_selection/selection.json'
    result_path = ensemble / 'final_test/test_result.json'
    lock_path = ensemble / 'global_heterogeneous_test_lock.json'
    selection, result, lock = (read(p) for p in (selection_path, result_path, lock_path))
    selection_hash = digest(selection_path)
    if out.exists():
        manifest = read(out / 'bundle_manifest.json')
        if manifest['selection_sha256'] != selection_hash:
            raise ValueError('Existing bundle belongs to another frozen selection')
        return verify_bundle(out)
    if (selection.get('mode') != 'final' or selection.get('eligible_for_final_test') is not True
            or selection.get('exploratory') is not False or selection.get('test_predictions_accessed') is not False):
        raise ValueError('Bundle requires the completed final validation-only selection')
    if (lock.get('state') != 'complete' or lock.get('pin') != result.get('pin')
            or result.get('selection_sha256') != selection_hash
            or lock.get('result_sha256') != digest(result_path)
            or result['pin']['threshold'] != selection['threshold']
            or result['cohort_sha256'] != selection['cohort_sha256']):
        raise ValueError('Final selection, lock and result do not agree')
    if len(selection['members']) != 8 or selection['winner'] != 'all_members_equal':
        raise ValueError('This release declaration expects the frozen eight-member winner')
    pins = dict(selection['frozen_input_sha256'])
    pins[str(selection_path)] = selection_hash
    pins[str(result_path)] = lock['result_sha256']
    pins[str(ensemble / 'final_test/ensemble_test_predictions.csv')] = lock['predictions_sha256']
    pins[str(ensemble / 'final_selection/validation_ensemble_probabilities.csv')] = selection['validation_ensemble_predictions_sha256']
    expected_code = {'evaluate_heterogeneous_ensemble.py': result['pin']['evaluator_sha256'],
                     'evaluate_selected_ensemble.py': result['pin']['inference_loader_sha256'],
                     'heterogeneous_validation_ensemble.py': selection['selector_sha256'],
                     'select_validation_ensemble.py': selection['helper_sha256']}
    pins.update({str(workspace / filename): sha for filename, sha in expected_code.items()})
    original_plan = workspace / 'large_model_search_plan.json'
    plan_pins = {str((workspace / path).resolve()): sha for path, sha in read(original_plan)['file_pins'].items()}
    original_records = {}
    relative_records = {}

    def add(source, relative, role, required_hash=None):
        source = Path(source).resolve()
        relative = str(Path(relative))
        if not source.is_relative_to(workspace) or not source.is_file():
            raise ValueError('Source is not a workspace file: ' + str(source))
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Bundle path must be relative')
        if str(source) in original_records:
            return original_records[str(source)]['relative_path']
        if relative in relative_records:
            raise ValueError('Two different sources map to the same bundle path')
        expected = required_hash or pins.get(str(source)) or plan_pins.get(str(source))
        observed = digest(source)
        if expected is not None and observed != expected:
            raise ValueError('Source differs from frozen pin: ' + str(source))
        record = {'relative_path': relative, 'original_path': str(source), 'sha256': observed,
                  'size_bytes': source.stat().st_size, 'role': role,
                  'frozen_research_pin_verified': expected is not None}
        original_records[str(source)] = record
        relative_records[relative] = record
        return relative

    member_records = []
    raw_geometry = {}
    cache_paths = set()
    for member in selection['members']:
        identity = member['id']
        folder = Path(member['checkpoint']).parent
        checkpoint = add(member['checkpoint'], f'models/{identity}/best.pt', 'frozen_best_checkpoint', member['checkpoint_sha256'])
        configuration_files = []
        for name in ('run_metadata.json', 'configuration.json', 'data_summary.json', 'cohort_manifest.json',
                     'summary.json', 'validation_best_metrics.json', 'history.json'):
            path = folder / name
            if path.is_file():
                configuration_files.append(add(path, f'models/{identity}/{name}', 'model_configuration_and_training_evidence'))
        prediction_path = Path(member['predictions'])
        add(prediction_path, f'models/{identity}/{prediction_path.name}', 'frozen_validation_predictions', member['predictions_sha256'])
        for key in ('trainer', 'inference_adapter'):
            add(member[key], 'source/' + Path(member[key]).name, 'pinned_model_source', member[key + '_sha256'])
        member_records.append({'id': identity, 'family': member['family'], 'seed': member['seed'],
                               'weight': member['aggregation_weight'], 'checkpoint_relative_path': checkpoint,
                               'checkpoint_original_path': member['checkpoint'], 'checkpoint_sha256': member['checkpoint_sha256'],
                               'trainer_relative_path': 'source/' + Path(member['trainer']).name,
                               'inference_adapter_relative_path': 'source/' + Path(member['inference_adapter']).name,
                               'configuration_relative_paths': configuration_files})
        metadata = folder / 'run_metadata.json'
        if metadata.exists():
            cache = read(metadata).get('config', {}).get('cache_dir')
            if cache:
                cache_paths.add(cache)
        data_summary = folder / 'data_summary.json'
        if data_summary.exists():
            for original, details in read(data_summary)['data_spec']['source_files'].items():
                if original in raw_geometry and raw_geometry[original] != details['source_sha256']:
                    raise ValueError('Geometry-source pins disagree between members')
                raw_geometry[original] = details['source_sha256']
                cache_paths.add(details['cache_dir'])

    for key, name in (('cohort_manifest', 'manifest.json'), ('dataset_csv', 'dataset.csv'), ('graph_index', 'graph_index.csv')):
        add(selection[key], 'cohort/' + name, 'frozen_cohort_metadata', selection[key + '_sha256'])
    for source, relative in ((selection_path, 'provenance/final_selection.json'),
                             (selection['candidate_specification'], 'provenance/final_candidates.json'),
                             (selection['policy'], 'provenance/heterogeneous_ensemble_policy_20261107.json'),
                             (result_path, 'provenance/test_result.json'),
                             (lock_path, 'provenance/global_heterogeneous_test_lock.json'),
                             (ensemble / 'final_test/ensemble_test_predictions.csv', 'evidence/ensemble_test_predictions.csv'),
                             (ensemble / 'final_selection/validation_ensemble_probabilities.csv', 'evidence/validation_ensemble_probabilities.csv'),
                             (original_plan, 'provenance/large_model_search_plan.json'),
                             (workspace / 'research_continuation_protocol_20261107.json', 'provenance/research_continuation_protocol_20261107.json'),
                             (workspace / 'research_continuation_20261107/ALLOW_INVENTORY.json', 'provenance/ALLOW_INVENTORY.json')):
        add(source, relative, 'immutable_provenance_or_fixed_evidence')
    # Include every selected local Python dependency without executing imports.
    for path in pins:
        if Path(path).suffix == '.py':
            add(path, 'source/' + Path(path).name, 'pinned_research_source')
    add(workspace / 'evaluate_heterogeneous_ensemble.py', 'source/evaluate_heterogeneous_ensemble.py', 'pinned_evaluator')
    add(workspace / 'evaluate_selected_ensemble.py', 'source/evaluate_selected_ensemble.py', 'pinned_adapter_loader')
    add(__file__, 'source/package_research_ensemble.py', 'release_packaging_tool')
    inspected = set()
    while True:
        pending = [Path(path) for path in original_records if Path(path).suffix == '.py' and path not in inspected]
        if not pending:
            break
        for path in pending:
            inspected.add(str(path))
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
                for name in names:
                    candidate = workspace / (name.split('.')[0] + '.py')
                    if candidate.is_file():
                        add(candidate, 'source/' + candidate.name, 'local_source_dependency')
    readiness = read(workspace / 'research_continuation_20261107/ALLOW_INVENTORY.json')
    for candidate in sorted(workspace.glob('research_followup_manifest*.json')):
        if digest(candidate) == readiness['followup_manifest_sha256']:
            add(candidate, 'provenance/' + candidate.name, 'frozen_finalization_manifest', readiness['followup_manifest_sha256'])

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + out.name + '.', dir=out.parent))
    try:
        for record in relative_records.values():
            source = Path(record['original_path'])
            target = staging / record['relative_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if (digest(source) != record['sha256'] or digest(target) != record['sha256']
                    or target.stat().st_size != record['size_bytes']):
                raise ValueError('Source changed during byte-exact copy: ' + str(source))
        packages = {}
        for name in ('torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn', 'rdkit', 'pandas'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        environment = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                       'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                       'installed_distribution_versions': packages,
                       'note': 'Observed packaging environment; original training versions also remain in per-model run_metadata.json.'}
        save(staging / 'environment.json', environment)
        readme = f'''Pachet de cercetare: ansamblu GATv2 + GNN geometric, 9 noiembrie 2026

Conține cele opt checkpointuri best.pt selectate, configurațiile, codul local,
cohorta tabelară și copii identice ale dovezilor de selecție și evaluare.
Fiecare model are ponderea 1/8. Pragul fixat pe validare este
{selection['threshold']:.17g}. Scorul final pe 16.788 molecule de test este
F1={100 * result['test_metrics']['f1']:.2f}%, ROC-AUC={100 * result['test_metrics']['roc_auc']:.2f}%.
Ținta F1 de 85% nu a fost atinsă. Testul a fost calculat numai pentru ansamblul
fixat; selecția și pragul nu au fost modificate după rezultat.

Etichetele reprezintă originea moleculară: referințe publice versus analogi
generați. Acest rezultat nu validează detectarea contrafacerilor farmaceutice
reale. Coordonatele 3D sunt predicții ETKDG și optimizări de câmp de forță,
nu măsurători experimentale ale produselor.

Fișierul bundle_manifest.json mapează căile relative la căile originale și
înregistrează SHA256, dimensiuni, ponderi, prag, cohortă și metrici. Fișierele
originale nu au fost modificate. Copiile de proveniență, configurațiile și
indicii păstrează textual căile absolute originale; nu au fost rescrise.

Cache-urile de antrenare și shardurile NPZ cu grafuri/coordonate nu sunt incluse.
Dependențele externe și hashurile deja cunoscute sunt enumerate în manifest;
nu s-a repetat verificarea integrală a acelor cache-uri/sharduri la ambalare.
Mutarea pe alt calculator necesită restaurarea datelor externe și adaptarea
căilor într-un strat separat de încărcare. Pachetul păstrează modelele și
proveniența; nu include un serviciu de inferență gata de lansare.

Verificare exclusiv a integrității pachetului, fără inferență sau retestare:
  python source/package_research_ensemble.py --verify-only --out .

Încărcarea checkpointurilor pe alt calculator necesită bibliotecile indicate
în environment.json. Nu s-au executat antrenări sau predicții la ambalare.
'''
        (staging / 'README.txt').write_text(readme)
        for name in ('README.txt', 'environment.json'):
            target = staging / name
            relative_records[name] = {'relative_path': name, 'original_path': None,
                                      'sha256': digest(target), 'size_bytes': target.stat().st_size,
                                      'role': 'generated_release_metadata', 'frozen_research_pin_verified': False}
        external = [{'kind': 'training_cache', 'original_path': path, 'copied': False,
                     'verification_at_packaging': 'not rehashed; referenced by original configurations/data_summary'}
                    for path in sorted(cache_paths)]
        external += [{'kind': 'source_3d_graph_shard', 'original_path': path, 'copied': False,
                      'sha256_from_training_metadata': sha, 'verification_at_packaging': 'not rehashed'}
                     for path, sha in sorted(raw_geometry.items())]
        manifest = {'schema_version': 1, 'release_id': 'ensemble_20261109',
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'release_kind': 'portable_research_checkpoint_and_provenance_bundle',
                    'selection_original_path': str(selection_path), 'selection_sha256': selection_hash,
                    'result_original_path': str(result_path), 'result_sha256': digest(result_path),
                    'winner': selection['winner'], 'aggregation': selection['aggregation'],
                    'threshold': selection['threshold'], 'threshold_source': 'frozen_validation_only',
                    'cohort_sha256': selection['cohort_sha256'], 'policy_sha256': selection['policy_sha256'],
                    'members': member_records, 'validation_metrics': selection['validation_metrics'],
                    'test_metrics': result['test_metrics'], 'target_f1': .85,
                    'test_target_reached': result['test_target_reached'],
                    'label_semantics': {'0': 'public reference molecular origin', '1': 'generated analogue molecular origin'},
                    'geometry_semantics': 'ETKDG/force-field predicted conformers; not experimental product measurements',
                    'files': sorted(relative_records.values(), key=lambda entry: entry['relative_path']),
                    'external_dependencies_not_copied': external,
                    'training_cache_and_graph_shards_included': False,
                    'original_absolute_paths_preserved': True, 'relocation_requires_external_data_and_path_adapter': True,
                    'inference_wrapper_included': False, 'model_inference_performed_for_packaging': False,
                    'integrity_verification': 'All copied payload files checked byte-for-byte by SHA256 and size; external graph/cache data not rehashed.'}
        save(staging / 'bundle_manifest.json', manifest)
        (staging / 'bundle_manifest.sha256').write_text(digest(staging / 'bundle_manifest.json') + '  bundle_manifest.json\n')
        verify_bundle(staging)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_bundle(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path, default=Path('model_releases/ensemble_20261109'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    result = verify_bundle(args.out) if args.verify_only else package(args.workspace, args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
