#!/usr/bin/env python3
"""Mock ownership and plan-identity checks; never signals or launches a process."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import check_thermal_guard_cool_v2 as fixtures
import thermal_guard_ax as adapter
import thermal_guard_cool_v2 as base
from make_after_ax_inventory_plan_20261109 import build, EXPECTED_MODEL_OUT, INVENTORY_OUT
from run_overnight_research import atomic_json, checksum


class AxOwnershipChecks(unittest.TestCase):
    def fixture(self, helper=False):
        f = fixtures.GuardChecks()
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.queue['status'] = 'running'
        f.qargv[2] = str(f.root / adapter.AX_SUPERVISOR)
        f.reader.records[10] = ({**f.reader.records[10][0], 'argv_sha256': base.engine.digest(f.qargv)}, list(f.qargv))
        if helper:
            f.command[:] = [sys.executable, adapter.AX_SUPERVISOR, '--ax-step', 'ask', '--ordinal', '0']
            f.reader.records[30] = ({**f.reader.records[30][0], 'argv_sha256': base.engine.digest(f.command)}, list(f.command))
        f.write()
        return f

    def test_new_ax_supervisor_is_healthy_and_unknown_queue_is_rejected(self):
        f = self.fixture()
        with adapter.bound_ax_ownership():
            self.assertFalse(f.verify(allowed_scripts=adapter.AX_STAGE_SCRIPTS)['orphan'])
            f.qargv[2] = str(f.root / 'unrelated_supervisor.py')
            f.reader.records[10] = ({**f.reader.records[10][0], 'argv_sha256': base.engine.digest(f.qargv)}, list(f.qargv))
            with self.assertRaises(ValueError):
                f.verify(allowed_scripts=adapter.AX_STAGE_SCRIPTS)

    def test_same_script_helper_needs_explicit_stage_allowlist(self):
        f = self.fixture(helper=True)
        with adapter.bound_ax_ownership():
            with self.assertRaises(ValueError):
                f.verify(allowed_scripts=base.DEFAULT_STAGE_SCRIPTS)
            record = f.verify(allowed_scripts=adapter.AX_STAGE_SCRIPTS)
            self.assertFalse(record['orphan'])
            f.reader.records[30][1].append('--different-step')
            with self.assertRaises(ValueError):
                f.verify(allowed_scripts=adapter.AX_STAGE_SCRIPTS)

    def test_adapter_restores_globals_and_rejects_extra_executables(self):
        before = (base.SUPERVISORS, base.engine.SUPERVISORS, base.engine.TRAINERS,
                  base.DEFAULT_STAGE_SCRIPTS, base.CoolingGuard)
        with adapter.bound_ax_ownership():
            self.assertIn(adapter.AX_SUPERVISOR, base.SUPERVISORS)
            self.assertIn(adapter.AX_SUPERVISOR, base.engine.SUPERVISORS)
            self.assertEqual(base.DEFAULT_STAGE_SCRIPTS, {'train_geometric_variants.py'})
        after = (base.SUPERVISORS, base.engine.SUPERVISORS, base.engine.TRAINERS,
                 base.DEFAULT_STAGE_SCRIPTS, base.CoolingGuard)
        self.assertTrue(all(x is y for x, y in zip(before, after)))
        adapter.validate_extra_stages(['--allow-stage-script', adapter.AX_SUPERVISOR])
        with self.assertRaises(ValueError):
            adapter.validate_extra_stages(['--allow-stage-script=unrelated.py'])


class AfterAxPlanChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='after_ax_plan_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ('frozen_core.py', 'make_after_ax_inventory_plan_20261109.py', 'thermal_guard_ax.py'):
            (self.root / name).write_text('# fixture\n')
        self.old = {'workspace': str(self.root), 'out_dir': 'molecule_bank_1m/old',
            'generation_command': ['python', 'ensure_bounded_bank.py', '--config', 'same_config.json'],
            'stages': [{'id': 'audit', 'command': ['python', 'cool_bank_stage.py', '--state',
                'molecule_bank_1m/old/cooling/audit.json', '--', 'audit_export_million_bank.py',
                '--out', 'molecule_bank_1m/release']}], 'families': [],
            'training_cohort': {'cohort_sha256': 'cohort'},
            'file_pins': {'frozen_core.py': checksum(self.root / 'frozen_core.py')}}
        self.predecessor = self.root / 'million_inventory_bounded_plan_20261109.json'
        atomic_json(self.predecessor, self.old)
        self.hpo = {'workspace': str(self.root), 'out_dir': EXPECTED_MODEL_OUT,
                    'training_cohort': {'cohort_sha256': 'cohort'}}
        self.hpo_path = self.root / 'hpo_plan.json'
        atomic_json(self.hpo_path, self.hpo)

    def test_only_output_and_readiness_change_generation_is_identical(self):
        new = build(self.root, self.hpo_path)
        self.assertEqual(new['generation_command'], self.old['generation_command'])
        self.assertEqual(new['out_dir'], INVENTORY_OUT)
        self.assertEqual(new['stages'][0]['command'][3], INVENTORY_OUT + '/cooling/audit.json')
        self.assertEqual(new['stages'][0]['command'][-1], 'molecule_bank_1m/release')
        self.assertEqual(new['file_pins']['frozen_core.py'], self.old['file_pins']['frozen_core.py'])
        dependency = new['after_ax_dependency']
        self.assertEqual(dependency['protocol_sha256'], checksum(self.hpo_path))
        self.assertEqual(dependency['model_progress'], EXPECTED_MODEL_OUT + '/progress.json')
        self.assertEqual(dependency['ready_marker'], EXPECTED_MODEL_OUT + '/ALLOW_INVENTORY.json')
        self.assertFalse(dependency['model_test_evaluation_required'])
        self.assertEqual(json.loads(self.predecessor.read_text()), self.old)

    def test_wrong_hpo_destination_cohort_or_changed_source_is_rejected(self):
        hpo = copy.deepcopy(self.hpo)
        hpo['out_dir'] = 'training_runs/old_completed_run'
        atomic_json(self.hpo_path, hpo)
        with self.assertRaises(ValueError):
            build(self.root, self.hpo_path)
        hpo['out_dir'] = EXPECTED_MODEL_OUT
        hpo['training_cohort']['cohort_sha256'] = 'wrong'
        atomic_json(self.hpo_path, hpo)
        with self.assertRaises(ValueError):
            build(self.root, self.hpo_path)
        atomic_json(self.hpo_path, self.hpo)
        (self.root / 'frozen_core.py').write_text('# changed\n')
        with self.assertRaises(ValueError):
            build(self.root, self.hpo_path)


if __name__ == '__main__':
    unittest.main()
