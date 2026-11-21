"""Isolated allocation, cooled-time and error-path checks; no production bank writes."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from accumulated_generation_cooling import AccumulatedThermalGate
from bounded_bank_scheduling import BoundedCoverageScheduler, load_scheduler
from generation_job_host import NativeJobHost, NativeJobTimeout


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def gate(clock):
    result = AccumulatedThermalGate('/unused-fixture', clock=clock, sleeper=clock.sleep)
    result.record = Mock()
    result.state = Mock(return_value=0)
    return result


class CoolingChecks(unittest.TestCase):
    def test_short_jobs_accumulate_without_sixty_second_per_job_wait(self):
        clock = FakeClock(); cooling = gate(clock)
        for i in range(100):
            cooling.wait(str(i)); clock.sleep(.005); cooling.finished()
        self.assertLess(clock(), 6)
        self.assertAlmostEqual(cooling.total_active_wall, .5, places=6)

    def test_budget_debt_and_coordinator_work_are_accounted(self):
        clock = FakeClock(); cooling = gate(clock)
        cooling.wait('first'); clock.sleep(10); cooling.finished()
        clock.sleep(5)  # Coordinator/import work is deliberately conservative.
        before = clock(); cooling.wait('second')
        self.assertGreaterEqual(clock() - before, 45)
        self.assertEqual(cooling.active_wall, 0)
        self.assertEqual(cooling.rest_debt, 0)

    def test_pressure_probe_and_status_writes_are_charged_as_work(self):
        clock = FakeClock(); cooling = gate(clock)
        def pressure():
            clock.sleep(2)
            return 0
        cooling.state = pressure
        cooling.record = lambda *args, **kwargs: clock.sleep(1)
        cooling.wait('control-overhead')
        self.assertAlmostEqual(cooling.total_active_wall, 4)
        self.assertGreater(cooling.rest_debt, 11)

    def test_native_timeout_finally_charges_full_ninety_second_debt(self):
        clock = FakeClock(); cooling = gate(clock)
        host = NativeJobHost(thermal_gate=cooling)
        host._start = Mock()
        host.connection = Mock()
        host._discard = Mock()
        def timeout(_deadline):
            clock.sleep(90)
            raise NativeJobTimeout('fixture')
        host._receive = timeout
        with self.assertRaises(NativeJobTimeout):
            host.run({'job_id': 'fixture', 'timeout_seconds': 90})
        self.assertIsNone(cooling.compute_started)
        self.assertAlmostEqual(cooling.rest_debt, 270)
        before = clock(); cooling.wait('retry')
        self.assertGreaterEqual(clock() - before, 270)

    def test_unknown_starts_with_rest_and_fair_budget_repaid(self):
        clock = FakeClock(); cooling = gate(clock); cooling.state = Mock(return_value=None)
        before = clock(); cooling.wait('unknown')
        self.assertGreaterEqual(clock() - before, 55)
        clock.sleep(5); cooling.finished()
        before = clock(); cooling.wait('next')
        self.assertGreaterEqual(clock() - before, 55)

    def test_critical_requires_sustained_nominal_recovery(self):
        clock = FakeClock(); cooling = gate(clock)
        cooling.state = lambda: 3 if clock() < 2 else 0
        cooling.wait('recover')
        self.assertGreaterEqual(clock(), 122)


class AllocationChecks(unittest.TestCase):
    def test_zero_yield_coverage_cannot_starve_volume(self):
        scheduler = BoundedCoverageScheduler(coverage={('M', 'bioisostere'): 441, ('E', 'bioisostere'): 30000})
        cells = [('M', 'bioisostere', 'near'), ('E', 'bioisostere', 'near')]
        for _ in range(1000):
            cell = scheduler.choose(cells)
            scheduler.record(cell, 25 if cell[0] == 'E' else 0, 5 if cell[0] == 'E' else .04)
        self.assertGreater(scheduler.attempts[cells[1]], 500)
        self.assertEqual(scheduler.mode_attempts, {'volume': 600, 'coverage': 250, 'exploration': 150})
        self.assertEqual(scheduler.coverage['M', 'bioisostere'], 441)
        scheduler.volume_target_reached = True
        self.assertEqual(scheduler.choose(cells)[0], 'M')

    def test_fine_targets_bounded_and_known_zero_pairs_deferred(self):
        scheduler = BoundedCoverageScheduler()
        job = {'method': 'M', 'category': 'bioisostere', 'reference': {'reference_id': 'ref'},
               'applicable_subcategories': ['rare', 'common']}
        counts = Counter({('bioisostere', 'common'): 100000})
        targets = []
        for _ in range(100):
            target = scheduler.choose_fine_target(job, counts)
            targets.append(target)
            scheduler.record_fine({**job, 'target_subcategory': target}, 0)
        self.assertEqual(targets.count('rare'), 2)
        self.assertEqual(targets.count('common'), 2)
        self.assertGreaterEqual(targets.count(None), 75)

    def test_load_reconstructs_persistent_fine_failure_and_allocation(self):
        c = sqlite3.connect(':memory:')
        c.execute('CREATE TABLE relations(method,category,molecule_id,job_id)')
        c.execute('CREATE TABLE jobs(method,category,difficulty,config_json,state,new_molecules,rejections_json,started_at,finished_at,produced,job_id)')
        config = {'quality': {}, 'evolution': {}, 'mutation': {}, 'difficulty_bands': {'near': {}}, 'rules_path': 'rules.py'}
        base = {**config, 'method': 'M', 'category': 'bioisostere', 'difficulty': 'near', 'band': {},
                'config_hash': 'current', 'reference': {'reference_id': 'ref'}, 'target_subcategory': 'rare',
                'scheduler_decision': {'allocation_mode': 'coverage'}}
        for i in range(2):
            c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)', ('M', 'bioisostere', 'near', json.dumps(base),
                'complete', 0, json.dumps({'job_host_seconds': .01, 'category_memberships_added': 0}),
                '2026-01-01T00:00:00', '2026-01-01T00:00:01', 0, str(i)))
        loaded = load_scheduler(c, config, 'current')
        self.assertEqual(loaded.fine_outcomes['ref', 'bioisostere', 'rare'], [2, 0])
        self.assertEqual(loaded.mode_attempts['coverage'], 2)
        self.assertEqual(loaded.fine_attempts, 2)
        self.assertEqual(loaded.coverage_history['M', 'bioisostere', 'near'][0][0], 0)


if __name__ == '__main__':
    unittest.main()
