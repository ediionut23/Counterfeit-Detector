#!/usr/bin/env python3
"""Resume the frozen large-data search and finish with validation ensembles only.

The original plan, stage IDs, source files and checkpoint configurations remain
unchanged. A separately pinned protocol explicitly disables test evaluation.
Required protocol keys: schema_version=1, protocol_id, mode='validation_only',
test_evaluation=false, workspace, out_dir, plan_sha256,
training_cohort_sha256, and source_pins (see REQUIRED_SOURCE_PINS).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import run_large_model_search as base
from run_overnight_research import atomic_json, checksum, load_json


REQUIRED_SOURCE_PINS = frozenset({
    'run_validation_continuation.py', 'run_large_model_search.py',
    'run_overnight_research.py', 'select_validation_ensemble.py',
    'enforce_large_training_cohort.py',
})
TRAINERS = frozenset({
    'train_gatv2_experiment.py', 'train_gatv2_search.py',
    'train_geometric_gnn.py', 'train_geometric_variants.py',
})


def verified_protocol(plan_path, protocol_path, protocol_sha256):
    """Validate all declared inputs without altering queue or stage ledgers."""
    plan_path, protocol_path = Path(plan_path).resolve(), Path(protocol_path).resolve()
    if checksum(protocol_path) != protocol_sha256:
        raise ValueError('Continuation protocol differs from its pinned hash')
    protocol, plan = load_json(protocol_path), load_json(plan_path)
    if protocol.get('schema_version') != 1 or protocol.get('mode') != 'validation_only' or protocol.get('test_evaluation') is not False:
        raise ValueError('Continuation requires an explicit validation-only, no-test protocol')
    identity = protocol.get('protocol_id', '')
    if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', identity):
        raise ValueError('Invalid continuation protocol_id')
    workspace = Path(plan['workspace']).resolve()
    if Path(protocol.get('workspace', '')).resolve() != workspace or protocol.get('out_dir') != plan['out_dir']:
        raise ValueError('Continuation must reuse the exact original workspace and output directory')
    if Path(__file__).resolve() != workspace / 'run_validation_continuation.py':
        raise ValueError('Continuation implementation is outside the pinned workspace')
    if protocol.get('plan_sha256') != checksum(plan_path):
        raise ValueError('Original search plan changed')
    if protocol.get('training_cohort_sha256') != plan['training_cohort']['cohort_sha256']:
        raise ValueError('Continuation cohort differs from the original frozen cohort')
    pins = protocol.get('source_pins', {})
    if not isinstance(pins, dict) or not REQUIRED_SOURCE_PINS <= pins.keys():
        raise ValueError('Continuation protocol is missing required implementation pins')
    for filename, digest in pins.items():
        path = (workspace / filename).resolve()
        if not path.is_relative_to(workspace) or checksum(path) != digest:
            raise ValueError('Pinned continuation source changed: ' + filename)
    out = workspace / plan['out_dir']
    if not out.resolve().is_relative_to(workspace):
        raise ValueError('Original search output escapes the workspace')
    return protocol, plan, out


def freeze_protocol(protocol_path, protocol_sha256, protocol, out):
    """Publish the exact manifest bytes once; refuse replacement on restart."""
    directory = Path(out) / 'continuation_protocols' / protocol['protocol_id']
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / 'protocol.json'
    payload = Path(protocol_path).read_bytes()
    try:
        with snapshot.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        pass
    if checksum(snapshot) != protocol_sha256:
        raise ValueError('Existing immutable continuation protocol changed')
    return snapshot


class ValidationContinuation(base.LargeSearch):
    def __init__(self, plan_path, protocol_path, protocol_sha256, startup_token=None):
        self.protocol_path = Path(protocol_path).resolve()
        self.protocol_sha256 = protocol_sha256
        self.continuation_protocol, _, out = verified_protocol(plan_path, protocol_path, protocol_sha256)
        self.protocol_snapshot = freeze_protocol(protocol_path, protocol_sha256, self.continuation_protocol, out)
        super().__init__(plan_path, startup_token)

    def save(self):
        # The inherited orchestration retains every validation stage, while its
        # final state is rewritten BEFORE any parent save can persist it.
        if self.state.get('status') == 'complete':
            self.state.update(status='validation_complete', current_stage=None,
                protocol='full-cohort screening/promotion/repeats; validation-only ensemble selection; test remains reserved')
        self.state['continuation_protocol'] = {
            'id': self.continuation_protocol['protocol_id'],
            'sha256': self.protocol_sha256,
            'snapshot': str(self.protocol_snapshot),
            'mode': 'validation_only', 'test_evaluation': False,
        }
        super().save()

    @staticmethod
    def validate_command(key, command):
        """Positive allowlist: no command can reach an evaluation executable."""
        command = list(command)
        script = Path(command[1]).name if len(command) >= 2 else ''
        if key.startswith(('screen_', 'promote_', 'repeat_')):
            if script not in TRAINERS or '--no-test' not in command or '--evaluate-test' in command:
                raise ValueError('Only validation-only training commands may continue')
            return
        if key in {'gatv2_select_on_validation', 'geometric_select_on_validation'} and script == 'select_validation_ensemble.py':
            return
        raise ValueError('Stage is outside the validation-only continuation allowlist: ' + key)

    def run(self, key, command, *, resume_dir=None):
        verified_protocol(self.plan_path, self.protocol_path, self.protocol_sha256)
        if checksum(self.protocol_snapshot) != self.protocol_sha256:
            raise ValueError('Frozen continuation protocol changed')
        self.validate_command(key, command)
        return super().run(key, command, resume_dir=resume_dir)

    def ensemble(self, family, runs, champion, repeats):
        # Same immutable member and candidate declaration as the pinned base.
        # There is intentionally no final-test stage in this implementation.
        members = []
        for run in runs + repeats:
            base.validation_score(run, self.plan['training_cohort']['cohort_sha256'])
            command = run['command']
            seed = int(command[command.index('--seed') + 1])
            folder = Path(run['out_dir'])
            predictions = folder / ('val_probabilities.csv' if family == 'gatv2' else 'validation_best_predictions.csv')
            members.append({'id': run['id'], 'family': family,
                'config_id': run.get('config_id', run['id']), 'seed': seed,
                'checkpoint': str(folder / 'best.pt'), 'predictions': str(predictions),
                'summary': str(folder / 'summary.json'), 'trainer': command[1],
                'inference_adapter': 'train_gatv2_search.py' if family == 'gatv2' else 'train_geometric_variants.py'})
        candidates = [{'id': 'single_' + member['id'], 'members': [member['id']]} for member in members]
        candidates.append({'id': 'champion_equal_seed_mean', 'members': [champion['id']] + [run['id'] for run in repeats]})
        cohort = self.plan['training_cohort']['directory']
        spec = {'schema_version': 1, 'workspace': str(self.root),
            'cohort_manifest': cohort + '/manifest.json', 'dataset_csv': cohort + '/dataset.csv',
            'graph_index': cohort + '/graph_index.csv', 'members': members,
            'candidates': candidates, 'required_ensemble_seeds': [42, 43, 44]}
        spec_path = self.out / (family + '_ensemble_candidates.json')
        if spec_path.exists() and load_json(spec_path) != spec:
            raise ValueError('Ensemble candidate declaration changed after selection')
        atomic_json(spec_path, spec)
        selected = self.out / (family + '_ensemble_selection')
        self.run(family + '_select_on_validation', [sys.executable, 'select_validation_ensemble.py',
            '--spec', str(spec_path), '--spec-sha256', checksum(spec_path), '--out', str(selected)])

    def work(self):
        # Base work supplies all screening, promotion and independent repeats;
        # dynamic dispatch selects the validation-only ensemble implementation.
        super().work()
        if self.state.get('status') != 'validation_complete':
            raise RuntimeError('Continuation did not publish its validation-only finish state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--startup-token')
    args = parser.parse_args()
    protocol, plan, out = verified_protocol(args.plan, args.protocol, args.protocol_sha256)
    freeze_protocol(args.protocol, args.protocol_sha256, protocol, out)
    if args.detach:
        token = uuid.uuid4().hex
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--plan', str(args.plan.resolve()),
            '--protocol', str(args.protocol.resolve()), '--protocol-sha256', args.protocol_sha256,
            '--startup-token', token]
        with (out / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=plan['workspace'], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = load_json(out / 'progress.json')
            fresh = state.get('startup_token') == token and state.get('pid') == process.pid
            pinned = state.get('continuation_protocol', {}).get('sha256') == args.protocol_sha256
            acknowledged = state.get('status') == 'validation_complete' or state.get('status') == 'running' and state.get('current_stage')
            if fresh and pinned and acknowledged:
                print(json.dumps({'startup_verified': True, 'pid': process.pid,
                    'protocol_sha256': args.protocol_sha256, 'progress': str(out / 'progress.json')}))
                return
            if process.poll() is not None:
                raise RuntimeError('Continuation startup failed; inspect ' + str(out / 'supervisor.log'))
            time.sleep(.2)
        raise RuntimeError('Continuation did not acknowledge its first stage; inspect ' + str(out / 'supervisor.log'))
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    queue = ValidationContinuation(args.plan, args.protocol, args.protocol_sha256, args.startup_token)
    awake = None
    try:
        if sys.platform == 'darwin':
            awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        queue.work()
    except BaseException as error:
        queue.state.update(status='stopped' if isinstance(error, InterruptedError) else 'failed', error=repr(error))
        queue.save()
        raise
    finally:
        if awake and awake.poll() is None:
            awake.terminate()


if __name__ == '__main__':
    main()
