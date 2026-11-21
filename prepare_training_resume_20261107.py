#!/usr/bin/env python3
"""Reconcile a dead held stage before checkpoint retry, retaining original evidence.

Never sends process signals and never changes trainer, plan, cohort or checkpoint.
"""
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from run_overnight_research import atomic_json, checksum, now
from thermal_guard import ProcessReader, same_process


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    active = json.loads(Path('active_research_run.json').read_text())
    hold_path = Path('training_runs/manual_cooling_hold.json')
    hold = json.loads(hold_path.read_text())
    if hold.get('status') != 'paused':
        raise RuntimeError('Expected original cooling hold; inspect before another recovery')
    plan_path = Path(active['model_plan'])
    plan = json.loads(plan_path.read_text())
    for path, expected in plan['file_pins'].items():
        if checksum(path) != expected:
            raise RuntimeError('Pinned implementation/input changed: ' + path)
    reader = ProcessReader()
    identity_evidence = []
    for target in hold['targets']:
        try:
            current, _ = reader.snapshot(target['fingerprint']['pid'])
        except ProcessLookupError:
            identity_evidence.append({'role': target['role'], 'original': target['fingerprint'], 'state': 'absent'})
        else:
            if same_process(current, target['fingerprint']):
                raise RuntimeError('Original held process is still present: ' + target['role'])
            identity_evidence.append({'role': target['role'], 'original': target['fingerprint'],
                                      'state': 'pid_reused_by_different_process', 'current': current})
    out = Path(active['model_out_dir'])
    progress_path = out / 'progress.json'
    state = json.loads(progress_path.read_text())
    if state['plan_sha256'] != checksum(plan_path):
        raise RuntimeError('Plan identity changed')
    key = state['current_stage']
    stage = state['stages'][key]
    status_path = Path(stage['stage_status_path'])
    status = json.loads(status_path.read_text())
    if stage['status'] != 'running' or status['status'] != 'running':
        raise RuntimeError('Only the confirmed dead running stage may be reconciled')
    if stage['wrapper_token'] != status['wrapper_token'] or stage['command_sha256'] != status['command_sha256']:
        raise RuntimeError('Stage ownership evidence differs')
    command = status['command']
    run_dir = Path(command[command.index('--out') + 1] if '--out' in command else command[command.index('--out-dir') + 1])
    checkpoint_path = run_dir / 'last.pt'
    checkpoint_hash = checksum(checkpoint_path)
    with ExitStack() as locks:
        for path in (out / 'queue.lock', Path(stage['stage_lock_path']), run_dir / '.run.lock',
                     Path('molecule_bank_1m/bank.generation.lock'), Path('molecule_bank_1m/bank.chemistry.lock')):
            stream = locks.enter_context(path.open('a+'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        torch.set_num_threads(1)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['configuration']['cohort_sha256'] != plan['training_cohort']['cohort_sha256']:
            raise RuntimeError('Checkpoint cohort differs')
        if checksum(checkpoint_path) != checkpoint_hash:
            raise RuntimeError('Checkpoint changed during recovery verification')
        archive = out / ('recovery_' + time.strftime('%Y%m%dT%H%M%S', time.gmtime()))
        archive.mkdir()
        for path in (progress_path, status_path, status_path.with_name(status_path.name.replace('.status.json', '.spec.json')),
                     hold_path, Path('active_research_run.json'), run_dir / 'progress.json'):
            (archive / path.name).write_bytes(path.read_bytes())
        evidence = {'created_at': now(), 'action': 'reconcile_dead_stage_for_checkpoint_retry',
                    'stage': key, 'identity_evidence': identity_evidence,
                    'queue_stage_trainer_generation_and_chemistry_locks_verified_free': True,
                    'checkpoint': str(checkpoint_path.resolve()), 'checkpoint_sha256': checkpoint_hash,
                    'checkpoint_epoch': checkpoint['epoch'], 'best_epoch': checkpoint['best_epoch'],
                    'cohort_sha256': checkpoint['configuration']['cohort_sha256'],
                    'original_plan_sha256': checksum(plan_path), 'test_evaluation': False,
                    'models_data_and_plan_unchanged': True}
        atomic_json(archive / 'recovery.json', evidence)
        status.update(status='failed', returncode=75, finished_at=now(),
                      error='Original processes absent after cooling hold; explicit checkpoint retry authorized on 2026-11-07',
                      recovery_evidence=str((archive / 'recovery.json').resolve()))
        atomic_json(status_path, status)
        stage.update(status='failed', returncode=75, wrapper_status=status,
                     recovery_evidence=str((archive / 'recovery.json').resolve()))
        state.update(status='resume_prepared', updated_at=now(), error=None)
        atomic_json(progress_path, state)
        print(json.dumps({'prepared': True, 'stage': key, 'resume_after_epoch': checkpoint['epoch'],
                          'recovery_evidence': str(archive / 'recovery.json')}))


if __name__ == '__main__':
    main()
