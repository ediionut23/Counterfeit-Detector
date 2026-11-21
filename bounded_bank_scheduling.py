"""Bound coverage work, learn yield per cooled wall second, retain actual labels."""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import yield_aware_scheduling as legacy


DEFAULT_POLICY = {**legacy.DEFAULT_POLICY, 'version': 'bounded_coverage_wall_v1',
    'coverage_fraction': .25, 'exploration_fraction': .15,
    'minimum_active_time_share': {'E': 0.0, 'M': 0.0},
    'minimum_job_share': {'E': .25, 'M': .10},
    'wall_overhead_seconds': .25, 'fine_target_fraction': .25,
    'fine_pair_zero_skip_after': 2}


class BoundedCoverageScheduler(legacy.YieldAwareScheduler):
    def __init__(self, policy=None, coverage=None, method_origins=None):
        policy = {**DEFAULT_POLICY, **(policy or {})}
        super().__init__(policy, coverage, method_origins)
        c, e = policy['coverage_fraction'], policy['exploration_fraction']
        if not 0 < c <= .5 or c + e >= 1 or not 0 <= policy['fine_target_fraction'] <= .5:
            raise ValueError('Coverage/fine allocations must stay bounded')
        shares = policy['minimum_job_share']
        if set(shares) != {'M', 'E'} or min(shares.values()) < 0 or sum(shares.values()) >= 1:
            raise ValueError('Invalid method job shares')
        if policy['wall_overhead_seconds'] <= 0 or policy['fine_pair_zero_skip_after'] < 1:
            raise ValueError('Wall overhead and fine retry cap must be positive')
        self.mode_attempts = Counter()
        self.coverage_history = defaultdict(lambda: deque(maxlen=self.policy['history_window']))
        self.fine_outcomes = defaultdict(lambda: [0, 0])
        self.fine_attempts = 0
        self.last_fine_decision = None
        self.volume_target_reached = False

    def cooled_cost(self, seconds):
        return 4 * max(.001, float(seconds)) + self.policy['wall_overhead_seconds']

    def wall_rate(self, cell, *, coverage=False):
        history = self.coverage_history if coverage else self.history
        all_rows = [row for values in history.values() for row in values]
        total_new = sum(v[0] for v in all_rows)
        total_cost = sum(self.cooled_cost(v[1]) for v in all_rows)
        prior = total_new / total_cost if total_cost else .1
        values = history[cell]
        # Two average-job pseudocounts fade with observed JOBS rather than only
        # CPU time, so millisecond failures cannot retain an optimistic prior.
        mean_cost = total_cost / len(all_rows) if all_rows else 1.0
        pseudo_cost = 2 * mean_cost
        return (sum(v[0] for v in values) + prior * pseudo_cost) / (sum(self.cooled_cost(v[1]) for v in values) + pseudo_cost)

    def _mode(self):
        shares = {'coverage': self.policy['coverage_fraction'],
                  'exploration': self.policy['exploration_fraction'],
                  'volume': 1 - self.policy['coverage_fraction'] - self.policy['exploration_fraction']}
        n = sum(self.mode_attempts.values()) + 1
        return max(shares, key=lambda key: (shares[key] * n - self.mode_attempts[key], key))

    def choose(self, cells):
        cells = sorted(set(cells))
        if not cells:
            return None
        methods = sorted({cell[0] for cell in cells})
        attempts = Counter()
        for cell, n in self.attempts.items():
            attempts[cell[0]] += n
        mode = self._mode()
        n = sum(attempts.values())
        untried = [method for method in methods if not attempts[method]]
        groups = sorted({cell[:2] for cell in cells})
        missing = {group: max(0, self.policy['minimum_named_category_unique'][group[0]] - self.coverage[group]) for group in groups}
        under = [group for group in groups if missing[group]]
        allowed = set(methods)
        reason = mode
        if self.volume_target_reached and under:
            # The volume objective is complete; finish the original floors,
            # retaining truthful same-category generation and all chemical gates.
            mode, reason = 'coverage', 'coverage_after_volume_target'
        elif untried:
            allowed = {untried[0]}
            reason += '+initial_method_probe'
        else:
            method_deficits = {m: self.policy['minimum_job_share'][m] * (n + 1) - attempts[m] for m in methods}
            if max(method_deficits.values()) > 1e-8:
                method = max(methods, key=lambda m: (method_deficits[m], m))
                allowed = {method}
                reason += '+method_job_share'
        choices = [cell for cell in cells if cell[0] in allowed]
        groups = sorted({cell[:2] for cell in choices})
        group_attempts = {g: sum(v for key, v in self.attempts.items() if key[:2] == g) for g in groups}
        under = [g for g in groups if missing[g]]
        if mode == 'coverage' and under:
            # Yield/coverage utility breaks ties; a zero-yield fine target cannot
            # monopolize volume slots merely because its category is deficient.
            group = max(under, key=lambda g: (missing[g] / self.policy['minimum_named_category_unique'][g[0]],
                max(self.wall_rate(c, coverage=True) for c in choices if c[:2] == g), -group_attempts[g], g))
        elif mode == 'exploration':
            group = min(groups, key=lambda g: (group_attempts[g], self.coverage[g], g))
        else:
            group = max(groups, key=lambda g: (max(self.wall_rate(c) for c in choices if c[:2] == g), -group_attempts[g], g))
        bands = [cell for cell in choices if cell[:2] == group]
        if mode == 'exploration':
            cell = min(bands, key=lambda c: (self.attempts[c], c))
        else:
            cell = max(bands, key=lambda c: (self.wall_rate(c, coverage=mode == 'coverage'), -self.attempts[c], c))
        self.last_decision = {'cell': list(cell), 'reason': reason, 'allocation_mode': mode,
            'expected_new_per_cooled_wall_second': self.wall_rate(cell),
            'expected_new_category_memberships_per_cooled_wall_second': self.wall_rate(cell, coverage=True),
            'exploration': mode == 'exploration', 'volume_target_reached': self.volume_target_reached}
        return cell

    def record(self, cell, new_molecules, seconds, *, current_policy=True, category_memberships=0):
        super().record(cell, new_molecules, seconds, current_policy=current_policy)
        self.coverage_history[cell].append((int(category_memberships), max(.001, float(seconds))))
        if current_policy:
            mode = (self.last_decision or {}).get('allocation_mode', 'volume')
            self.mode_attempts[mode] += 1

    def choose_fine_target(self, job, fine_counts):
        self.fine_attempts += 1
        fraction = self.policy['fine_target_fraction']
        target_slot = int(self.fine_attempts * fraction) > int((self.fine_attempts - 1) * fraction)
        options = list(job['applicable_subcategories'])
        reference = job['reference']['reference_id']
        def stats(sub):
            return self.fine_outcomes[reference, job['category'], sub]
        viable = [sub for sub in options if not (stats(sub)[0] >= self.policy['fine_pair_zero_skip_after'] and not stats(sub)[1])]
        if not target_slot or not viable:
            result = None
            reason = 'broad_same_category' if not target_slot else 'known_zero_yield_pairs_deferred'
        else:
            # Normalized count only gently favors rarity; observed feasibility
            # dominates repeated failures. Actual engine subcategory is retained.
            result = max(viable, key=lambda sub: ((stats(sub)[1] + 1) / (stats(sub)[0] + 2) /
                math.sqrt(1 + math.log1p(fine_counts[job['category'], sub])), sub))
            reason = 'bounded_feasibility_aware_fine_target'
        self.last_fine_decision = {'policy': 'bounded_fine_target_v1', 'target_fraction': fraction,
            'method_attempt': self.fine_attempts, 'target_subcategory': result, 'reason': reason}
        return result

    def record_fine(self, job, produced):
        target = job.get('target_subcategory')
        if job['method'] == 'M' and target:
            outcome = self.fine_outcomes[job['reference']['reference_id'], job['category'], target]
            outcome[0] += 1
            outcome[1] += int(produced)

    def summary(self):
        result = super().summary()
        result.update(policy_version='bounded_coverage_wall_v1', allocation_counts=dict(self.mode_attempts),
            expected_new_per_cooled_wall_second={'/'.join(cell): self.wall_rate(cell) for cell in self.history},
            cooling_cost_estimate='4x chemistry wall + fixed coordinator overhead; actual cooling separately measured',
            fine_target_attempts=self.fine_attempts, last_fine_decision=self.last_fine_decision,
            volume_target_reached=self.volume_target_reached)
        return result


def load_scheduler(connection, config, config_hash, method_origins=None):
    scheduler = BoundedCoverageScheduler(config.get('yield_scheduler'), legacy.category_coverage(connection), method_origins)
    retained = Counter()
    rows = connection.execute("""SELECT method,category,difficulty,config_json,state,new_molecules,
        rejections_json,started_at,finished_at,produced FROM jobs WHERE state IN ('complete','error')
        AND finished_at IS NOT NULL ORDER BY finished_at DESC,job_id DESC""")
    for row in rows:
        job = json.loads(row[3])
        if (job.get('quality') != config['quality'] or job.get('evolution') != config['evolution'] or
                job.get('mutation') != config['mutation'] or job.get('band') != config['difficulty_bands'].get(job.get('difficulty')) or
                Path(job.get('rules_path', '')).resolve() != Path(config['rules_path']).resolve()):
            continue
        cell = tuple(row[:3]); report = json.loads(row[6]) if row[6] else {}
        try:
            elapsed = (datetime.fromisoformat(row[8]) - datetime.fromisoformat(row[7])).total_seconds()
        except (TypeError, ValueError):
            elapsed = float(job.get('timeout_seconds', 90))
        seconds = max(.001, float(report.get('job_host_seconds', report.get('seconds', elapsed))))
        if retained[cell] < scheduler.policy['history_window']:
            scheduler.history[cell].appendleft((int(row[5] or 0), seconds))
            scheduler.coverage_history[cell].appendleft((int(report.get('category_memberships_added', row[5] or 0)), seconds))
            retained[cell] += 1
        if row[4] == 'complete':
            scheduler.record_fine(job, row[9] or 0)
        if job.get('config_hash') == config_hash:
            scheduler.attempts[cell] += 1
            scheduler.active_seconds[cell[0]] += seconds
            scheduler.mode_attempts[job.get('scheduler_decision', {}).get('allocation_mode', 'volume')] += 1
            if job['method'] == 'M':
                scheduler.fine_attempts += 1
    return scheduler
