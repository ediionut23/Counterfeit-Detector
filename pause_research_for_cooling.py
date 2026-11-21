#!/usr/bin/env python3
"""Suspend only identity-verified active research processes; retain an audit ledger.

This is an indefinite manual cooling hold, with no automatic resume timer.
Resume must reverify every recorded process birth and account for a chemistry
job's elapsed watchdog deadline. Do not signal recorded PIDs without verification.
"""
import json
import os
from pathlib import Path
import signal
import time

from thermal_guard import ProcessReader, ProcBSDInfo, owned_training, same_process, option_value, script_argument
import ctypes


ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / 'training_runs' / 'manual_cooling_hold.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(value):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    temporary = LEDGER.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, LEDGER)


def main():
    os.chdir(ROOT)
    reader = ProcessReader()
    if LEDGER.exists() and read(LEDGER).get('status') == 'paused':
        print(json.dumps({'status': 'already_paused', 'ledger': str(LEDGER)}))
        return
    active = read('active_research_run.json')
    training = owned_training(Path(active['model_out_dir']) / 'progress.json', ROOT, reader)
    if training is None:
        raise RuntimeError('No verifiable active training stage; inspect before pausing')
    targets = []

    def capture(role, pid, expected_script=None, parent=None):
        fingerprint, argv = reader.snapshot(pid)
        if fingerprint['uid'] != os.getuid():
            raise RuntimeError('Unexpected process owner: ' + role)
        if expected_script and Path(script_argument(argv) or '').name != expected_script:
            raise RuntimeError('Unexpected script: ' + role)
        if parent is not None and fingerprint['ppid'] != parent:
            raise RuntimeError('Unexpected parent: ' + role)
        record = {'role': role, 'fingerprint': fingerprint}
        targets.append(record)
        return record, argv

    guard_state = read(active['thermal_guard_state'])
    guard, argv = capture('thermal_guard', guard_state['guard_pid'], 'thermal_guard.py')
    if Path(option_value(argv, '--state') or '').resolve() != (ROOT / active['thermal_guard_state']).resolve():
        raise RuntimeError('Unexpected thermal guard state path')
    capture('model_supervisor', training['queue_fingerprint']['pid'], 'run_large_model_search.py')

    inventory_path = ROOT / active['inventory_out_dir'] / 'progress.json'
    queue = read(inventory_path)
    stage = queue['stages'][queue['current_stage']]
    status = read(stage['stage_status_path'])
    spec = read(Path(stage['stage_status_path']).with_name(Path(stage['stage_status_path']).name.replace('.status.json', '.spec.json')))
    if not (queue['status'] == stage['status'] == status['status'] == 'running'):
        raise RuntimeError('Inventory no longer running')
    if not (stage['wrapper_token'] == status['wrapper_token'] == spec['wrapper_token']):
        raise RuntimeError('Inventory wrapper token mismatch')
    if not (stage['command_sha256'] == status['command_sha256'] == spec['command_sha256']):
        raise RuntimeError('Inventory command hash mismatch')
    _, argv = capture('inventory_supervisor', queue['pid'], 'run_overnight_research.py')
    if option_value(argv, '--startup-token') != queue['startup_token']:
        raise RuntimeError('Inventory startup token mismatch')
    capture('model_wrapper', training['wrapper_fingerprint']['pid'], 'run_overnight_research.py', training['queue_fingerprint']['pid'])
    capture('inventory_wrapper', status['wrapper_pid'], 'run_overnight_research.py', queue['pid'])
    capture('inventory_ensure', status['child_pid'], 'ensure_yield_bank.py', status['wrapper_pid'])
    lease = read('molecule_bank_1m/bank.chemistry.lock')
    capture('generation_coordinator', lease['host_pid'], 'expand_yield_aware_bank.py', status['child_pid'])
    child, argv = capture('chemistry_child', lease['child_pid'], parent=lease['host_pid'])
    identity = lease['child_identity']
    if child['fingerprint']['start'] != [identity['start_seconds'], identity['start_microseconds']]:
        raise RuntimeError('Chemistry child birth mismatch')
    if 'multiprocessing.spawn' not in ' '.join(argv) or child['fingerprint']['executable'] != training['fingerprint']['executable']:
        raise RuntimeError('Unexpected chemistry child executable/argv')
    trainer, _ = capture('training_child', training['fingerprint']['pid'], parent=training['wrapper_fingerprint']['pid'])
    if not same_process(trainer['fingerprint'], training['fingerprint'], parent=True):
        raise RuntimeError('Training identity changed')
    hold = {'status': 'pause_intended', 'created_epoch': time.time(),
            'reason': 'User reports hot laptop; suspend generation and training together',
            'automatic_resume': False, 'targets': targets, 'stop_files': [],
            'resume_note': 'Reverify identities; chemistry watchdog may require job timeout recovery after a long hold.'}
    save(hold)
    for path in (ROOT / active['model_out_dir'] / 'STOP', ROOT / active['inventory_out_dir'] / 'STOP', ROOT / 'molecule_bank_1m/STOP_GENERATION'):
        preexisting = path.exists()
        if not preexisting:
            path.write_text('Manual cooling hold: laptop reported hot on 2026-11-03. See training_runs/manual_cooling_hold.json.\n')
        hold['stop_files'].append({'path': str(path), 'preexisting': preexisting})
        save(hold)
    for target in targets:
        current, _ = reader.snapshot(target['fingerprint']['pid'])
        if not same_process(current, target['fingerprint'], parent=True):
            raise RuntimeError('Process identity changed before pause: ' + target['role'])
        os.kill(current['pid'], signal.SIGSTOP)
        target['signal_sent'] = 'SIGSTOP'
        save(hold)
    time.sleep(.25)
    for target in targets:
        current, _ = reader.snapshot(target['fingerprint']['pid'])
        if not same_process(current, target['fingerprint'], parent=True):
            raise RuntimeError('Process identity changed during pause')
        info = ProcBSDInfo()
        if reader.proc.proc_pidinfo(current['pid'], 3, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
            raise RuntimeError('Cannot read suspended process state')
        target['darwin_process_status'] = int(info.status)
        if info.status != 4:  # Darwin SSTOP
            raise RuntimeError('Pause not confirmed: ' + target['role'])
    hold.update(status='paused', verified_epoch=time.time())
    save(hold)
    print(json.dumps({'status': hold['status'], 'paused_processes': len(targets), 'automatic_resume': False, 'ledger': str(LEDGER)}))


if __name__ == '__main__':
    main()
