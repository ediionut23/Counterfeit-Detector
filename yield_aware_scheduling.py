"""Deterministic coverage floors and yield/time allocation, without gate changes."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime
import json
from pathlib import Path


DEFAULT_POLICY = {"version": "adaptive_yield_active_time_v1", "exploration_fraction": 0.15,
    "history_window": 32, "smoothing_seconds": 60.0,
    "minimum_active_time_share": {"E": 0.25, "M": 0.10},
    "minimum_named_category_unique": {"E": 20000, "M": 1000},
    "method_parity_bonus": 0.0}


def category_coverage(connection):
    # These are unique molecule/category associations with actual named
    # generator relations, not verified clinical or final-pair edit classes.
    return Counter({(r[0], r[1]): r[2] for r in connection.execute("""SELECT method,category,
        COUNT(DISTINCT molecule_id) FROM relations WHERE job_id IS NOT NULL GROUP BY method,category""")})


class YieldAwareScheduler:
    def __init__(self, policy=None, coverage=None, method_origins=None):
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        exploration = self.policy["exploration_fraction"]
        if not 0.10 <= exploration <= 0.50:
            raise ValueError("Exploration fraction must be in [0.10,0.50]")
        shares = self.policy["minimum_active_time_share"]
        if set(shares) != {"M", "E"} or min(shares.values()) < 0 or sum(shares.values()) >= 1:
            raise ValueError("Method time floors must be nonnegative and sum to less than one")
        if self.policy["history_window"] < 1 or self.policy["smoothing_seconds"] <= 0:
            raise ValueError("History window and smoothing must be positive")
        self.history = defaultdict(lambda: deque(maxlen=self.policy["history_window"]))
        self.coverage = Counter(coverage or {})
        self.method_origins = Counter(method_origins or {})
        self.attempts, self.active_seconds = Counter(), Counter()
        self.last_decision = None

    def rate(self, cell):
        method_rows = [entry for key, values in self.history.items() if key[0] == cell[0] for entry in values]
        amount = sum(entry[0] for entry in method_rows)
        seconds = sum(entry[1] for entry in method_rows)
        prior_rate = max(0.01, amount / seconds) if seconds else 0.10
        smoothing = self.policy["smoothing_seconds"]
        values = self.history[cell]
        return (sum(v[0] for v in values) + smoothing * prior_rate) / (sum(v[1] for v in values) + smoothing)

    def choose(self, cells):
        cells = sorted(set(cells))
        if not cells:
            return None
        methods = sorted({c[0] for c in cells})
        method_attempts = Counter()
        for cell, n in self.attempts.items():
            method_attempts[cell[0]] += n
        # One real probe of each available method precedes time-ratio decisions.
        untried = [m for m in methods if method_attempts[m] == 0]
        total_time = sum(self.active_seconds.values())
        deficits = {m: self.policy["minimum_active_time_share"][m] * total_time - self.active_seconds[m]
                    for m in methods}
        if untried:
            allowed = {untried[0]}
            reason = "method_initial_probe"
        elif total_time and max(deficits.values()) > 1e-8:
            allowed = {max(methods, key=lambda m: (deficits[m], m))}
            reason = "method_active_time_floor"
        else:
            allowed, reason = set(methods), "yield_exploitation"
        choices = [cell for cell in cells if cell[0] in allowed]
        groups = sorted({cell[:2] for cell in choices})
        group_attempts = {group: sum(n for cell, n in self.attempts.items() if cell[:2] == group) for group in groups}
        total_attempts = sum(self.attempts.values())
        # A deterministic accumulator provides approximately the declared share
        # over every prefix. No global RNG is consumed by the scheduler.
        explore = int((total_attempts + 1) * self.policy["exploration_fraction"]) > int(total_attempts * self.policy["exploration_fraction"])
        missing = {group: max(0, self.policy["minimum_named_category_unique"][group[0]] - self.coverage[group])
                   for group in groups}
        undercovered = [g for g in groups if missing[g]]
        if undercovered:
            group = max(undercovered, key=lambda g: (missing[g] / max(1, self.policy["minimum_named_category_unique"][g[0]]) /
                (1 + group_attempts[g]) ** 0.5, -group_attempts[g], g))
            reason += "+category_coverage_floor"
        elif explore:
            group = min(groups, key=lambda g: (group_attempts[g], self.coverage[g], g))
            reason += "+category_exploration"
        else:
            def group_rate(g):
                value = max(self.rate(cell) for cell in choices if cell[:2] == g)
                total = sum(self.method_origins.values())
                other = sum(n for m, n in self.method_origins.items() if m != g[0])
                bonus = self.policy["method_parity_bonus"] * max(0, other - self.method_origins[g[0]]) / max(1, total)
                return value * (1 + bonus)
            group = max(groups, key=lambda g: (group_rate(g), -group_attempts[g], g))
        bands = [cell for cell in choices if cell[:2] == group]
        if explore:
            cell = min(bands, key=lambda c: (self.attempts[c], c))
            reason += "+difficulty_exploration"
        else:
            cell = max(bands, key=lambda c: (self.rate(c), -self.attempts[c], c))
        self.last_decision = {"cell": list(cell), "reason": reason, "expected_new_per_active_second": self.rate(cell),
                              "exploration": explore}
        return cell

    def record(self, cell, new_molecules, seconds, *, current_policy=True):
        seconds = max(0.001, float(seconds))
        self.history[cell].append((int(new_molecules), seconds))
        if current_policy:
            self.attempts[cell] += 1
            self.active_seconds[cell[0]] += seconds
            self.coverage[cell[:2]] += int(new_molecules)  # Conservative: no duplicate relation credit.
            self.method_origins[cell[0]] += int(new_molecules)

    def summary(self):
        return {"policy": self.policy, "named_category_unique": {"/".join(k): v for k, v in self.coverage.items()},
            "active_chemistry_wall_seconds": dict(self.active_seconds), "current_policy_attempts": {"/".join(k): v for k, v in self.attempts.items()},
            "expected_new_per_active_second": {"/".join(k): self.rate(k) for k in list(self.history)},
            "last_decision": self.last_decision, "method_parity_is_mandatory": False,
            "category_count_semantics": "named operator-associated unique molecules; not verified final chemical classes"}

    def unmet_coverage(self, categories):
        return {f"{method}/{category}": max(0, self.policy["minimum_named_category_unique"][method] - self.coverage[method, category])
                for method in ("M", "E") for category in categories
                if self.coverage[method, category] < self.policy["minimum_named_category_unique"][method]}


def load_scheduler(connection, config, config_hash, method_origins=None):
    scheduler = YieldAwareScheduler(config.get("yield_scheduler"), category_coverage(connection), method_origins)
    retained = Counter()
    rows = connection.execute("""SELECT method,category,difficulty,config_json,state,new_molecules,
        rejections_json,started_at,finished_at FROM jobs WHERE state IN ('complete','error')
        AND finished_at IS NOT NULL ORDER BY finished_at DESC,job_id DESC""")
    current = []
    for row in rows:
        job = json.loads(row[3])
        if (job.get("quality") != config["quality"] or job.get("evolution") != config["evolution"] or
            job.get("mutation") != config["mutation"] or job.get("band") != config["difficulty_bands"].get(job.get("difficulty")) or
            Path(job.get("rules_path", "")).resolve() != Path(config["rules_path"]).resolve()):
            continue  # Pilot chemistry/search settings do not become production priors.
        cell = tuple(row[:3])
        report = json.loads(row[6]) if row[6] else {}
        try:
            wall = (datetime.fromisoformat(row[8]) - datetime.fromisoformat(row[7])).total_seconds()
        except (TypeError, ValueError):
            wall = float(job.get("timeout_seconds", 90))
        seconds = report.get("job_host_seconds", report.get("seconds", wall))
        if row[4] == "error":
            seconds = report.get("job_host_seconds", wall)
        if retained[cell] < scheduler.policy["history_window"]:
            # Rows are newest-first; appendleft retains chronological order.
            scheduler.history[cell].appendleft((int(row[5] or 0), max(0.001, float(seconds))))
            retained[cell] += 1
        if job.get("config_hash") == config_hash:
            current.append((cell, max(0.001, float(seconds))))
    for cell, seconds in current:
        scheduler.attempts[cell] += 1
        scheduler.active_seconds[cell[0]] += seconds
    return scheduler
