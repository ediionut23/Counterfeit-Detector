#!/usr/bin/env python3
"""Select a fixed validation ensemble, evaluate it once, then release inventory.

Run as one durable Queue stage under the cooling guard. All declaration and
selection steps precede the single locked test call. No test-driven tuning.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from run_overnight_research import atomic_json, checksum, load_json, now


def verify_manifest(path, expected):
    if checksum(path) != expected:
        raise ValueError('Follow-up manifest changed')
    manifest = load_json(path)
    if manifest.get('schema_version') != 1 or manifest.get('final_test_after_frozen_validation_selection') is not True:
        raise ValueError('Unsupported finalization protocol')
    for filename, digest in manifest['source_pins'].items():
        if checksum(filename) != digest:
            raise ValueError('Pinned finalization input changed: ' + filename)
    if checksum(manifest['continuation_protocol']) != manifest['continuation_protocol_sha256']:
        raise ValueError('Training continuation protocol changed')
    state = load_json(Path(manifest['workspace']) / manifest['model_out_dir'] / 'progress.json')
    if state.get('status') != 'validation_complete':
        raise ValueError('All declared training and validation selection must finish first')
    if state.get('continuation_protocol', {}).get('sha256') != manifest['continuation_protocol_sha256']:
        raise ValueError('Training finished under a different protocol')
    return manifest


def finalize(manifest_path, manifest_hash):
    manifest = verify_manifest(manifest_path, manifest_hash)
    import heterogeneous_validation_ensemble as selector
    import evaluate_heterogeneous_ensemble as evaluator
    root = Path(manifest['workspace']).resolve()
    destination = root / manifest['ensemble_out_dir']
    destination.mkdir(parents=True, exist_ok=True)
    progress = destination / 'pipeline_progress.json'
    def report(phase, **fields):
        atomic_json(progress, {'status': phase, 'updated_at': now(), 'manifest_sha256': manifest_hash, **fields})
    try:
        report('build_final_fixed_candidates', test_accessed=False)
        spec_path = destination / 'final_candidates.json'
        selector.build_spec(root / manifest['ensemble_policy'], manifest['ensemble_policy_sha256'], root,
                            root / manifest['model_out_dir'], root / manifest['cohort_dir'], 'final', spec_path)
        report('select_on_validation', test_accessed=False)
        selector.select(spec_path, checksum(spec_path), destination / 'final_selection')
        selection_path = destination / 'final_selection' / 'selection.json'
        verify_manifest(manifest_path, manifest_hash)
        report('evaluate_one_frozen_candidate', selection_sha256=checksum(selection_path))
        result = evaluator.evaluate(selection_path, checksum(selection_path),
                                    root / manifest['test_lock'], destination / 'final_test',
                                    device='cpu', threads=2)
        verify_manifest(manifest_path, manifest_hash)
        result_path = destination / 'final_test' / 'test_result.json'
        if not result_path.exists() or result['selection_sha256'] != checksum(selection_path):
            raise RuntimeError('Final evaluation did not publish the frozen selection result')
        marker = {'status': 'done', 'protocol_sha256': manifest['continuation_protocol_sha256'],
                  'model_out_dir': str((root / manifest['model_out_dir']).resolve()),
                  'followup_manifest_sha256': manifest_hash, 'completed_at': now(),
                  'selection_sha256': checksum(selection_path),
                  'result_path': str(result_path.resolve()), 'result_sha256': checksum(result_path),
                  'test_selection_finalized_before_evaluation': True}
        atomic_json(root / manifest['inventory_ready_marker'], marker)
        report('complete', selection_sha256=checksum(selection_path), test_metrics=result['test_metrics'],
               inventory_ready_marker=manifest['inventory_ready_marker'])
        return result
    except BaseException as error:
        report('failed', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    result = finalize(args.manifest.resolve(), args.manifest_sha256)
    print(json.dumps({'completed': True, 'winner': result['winner'], 'test_metrics': result['test_metrics']}))


if __name__ == '__main__':
    main()
