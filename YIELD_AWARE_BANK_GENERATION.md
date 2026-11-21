# Yield-aware, cool bank generation

The overnight run was active, but allocated most work to a poor-yield similarity
band. In the inspected session, 263 completed E/near jobs added327 unique
molecules, compared with1,789 from39 distant jobs and1,469 from98 intermediate
jobs. Seven native-code timeouts consumed approximately1.65hours: Python's
SIGALRM could not enforce a90second wall deadline while blocked inside native
RDKit code. One background worker and repeated bank-wide queries further
reduced throughput. Thermal status was nominal at inspection; there was no
evidence of a permanently stalled ledger or a thermal pause at that moment.

New files address scheduling and execution without editing the frozen core:

- `expand_yield_aware_bank.py`: serial coordinator and transactional persistence.
- `generation_job_host.py`: persistent isolated chemistry child, hard parent
  deadline, conservative thermal gate and inherited per-bank compute lease.
- `yield_aware_scheduling.py`: explicit category floors and adaptive yield/time
  allocation, including deterministic exploration.
- `million_bank_yield_config.json`: versioned production policy and unchanged
  chemistry settings.
- `check_yield_aware_bank.py`: temporary local fixtures; no production writes.

## Scientific and data constraints

The child invokes the frozen `expand_million_bank._run_job`, including the same
M/E proposal generation, active-moiety normalization, quality and similarity
gates, external holdout, recorded operations and synthetic-evidence labels.
The coordinator preserves the original global deduplication, reference caps,
child-scaffold caps, transaction boundaries and rollback recovery. Existing
completed jobs, molecules and relations are retained. Only pending/running jobs
of older configurations are superseded after acquiring the exclusive bank
coordinator and compute leases.

No category or generation-method label is renamed. A generator's operator
category remains an operator association, not independent verification of a
final chemical edit class or a real counterfeit product. The frozen supervised
training cohort and its balanced training/evaluation protocol are unchanged.

## Scheduling policy

Equal M/E molecule inventory is not a requirement. The default method-parity
bonus is zero. The policy instead protects sufficient named operator coverage:

| Policy field | Default |
|---|---:|
| `minimum_named_category_unique.E` |20,000 per configured category |
| `minimum_named_category_unique.M` |1,000 per configured category |
| `minimum_active_time_share.E` |0.25 |
| `minimum_active_time_share.M` |0.10 |
| `exploration_fraction` |0.15 |
| `history_window` |32 comparable jobs per cell |
| `smoothing_seconds` |60 |
| `method_parity_bonus` |0 |

These fields live under `yield_scheduler` in the config and are included in the
production configuration hash. Category deficits remain a priority; difficulty
allocation uses observed new globally unique molecules per active chemistry
wall second, with15% deterministic exploration across available categories and
bands. This preserves exploration of near analogues without forcing equal
near/intermediate/distant molecule budgets. High-yield M proposals can fill the
inventory while E retains a minimum share of active generation time.

Historical priors use only jobs matching quality, mutation/evolution settings,
rules path and the exact declared difficulty band. Current policy attempts and
active seconds reload from committed jobs, including actual timeout cost;
thermal waiting is recorded separately and excluded. The active time measure
is wall time inside the chemistry host, not CPU utilization or energy.

Coverage counts use `COUNT(DISTINCT molecule_id)` for actual named
method/category relations. New membership of an existing bank molecule is
counted once, while repeated relations do not inflate coverage. The coordinator
stops successfully only when both the global unique inventory target and all
configured category floors are satisfied. Staged `--max-jobs` runs and explicit
drains may exit earlier; progress reports `coverage_floors_met` and
`unmet_named_category_floors`. Configured-round exhaustion below either target
returns failure rather than claiming completion.

## Hard native deadline and resource limits

At most one persistent chemistry compute child runs. It is started with spawn,
has only a read-only SQLite connection, and returns proposals/QC results through
a Pipe. The parent reads complete results using a receiver thread before any
join; a large result cannot deadlock on a full Pipe. A monotonic90second deadline
starts after child readiness and the thermal gate. At expiry the parent
terminates and reaps only its owned child, escalating to kill if required,
before starting any replacement. Child initialization has a separate30second
deadline. A native timeout occurs before any candidate bank transaction, so its
error path avoids unnecessary full-bank counter reconstruction. Infrastructure
errors restore the current job to pending and stop the coordinator.

A per-bank `.chemistry.lock` is acquired before DB recovery and inherited by
the child using spawn's file-descriptor transfer. If the coordinator is killed,
an orphan child retains that lease: a second coordinator fails closed before
DB changes or spawning a second compute child. The90second watchdog requires a
live parent. After parent death, explicit recovery must verify the child PID,
UID, parent and process birth identity recorded in the lease; no PID-only
automatic orphan kill is attempted.

Python and NumPy receive each job's existing deterministic seed. The child's
Python hash seed is also pinned (`execution_policy.python_hash_seed=0`) so
native-child replacement cannot change hash-based iteration order.

The thermal gate retains the previous conservative policy: one second rest
between jobs; no new job at fair/serious/critical/unknown state; after elevated
or unknown state, require30continuous seconds of nominal state. Waiting occurs
outside the chemical deadline. Thermal pressure is not a temperatureC sensor.
The coordinator does not change model threads or thermal-duty settings.

The root supervisor may launch this one-child generator with utility QoS to
avoid severe background I/O throttling, while retaining the compute and
thermal limits. QoS alone is not a CPU or temperature cap. Coordinator-local
SQLite expression/state and named-coverage indexes avoid repeated scans of
superseded job JSON or large lineage payloads. Full progress snapshots default
to120second intervals.

## CLI and monitoring

After the previous generator has drained and been verified stopped:

```sh
/usr/sbin/taskpolicy -c utility ./venv/bin/python expand_yield_aware_bank.py \
  --db molecule_bank_1m/bank.sqlite \
  --config million_bank_yield_config.json --workers 1 \
  --thermal-status-dir molecule_bank_1m/thermal_control_yield
```

`--max-jobs N`, `--prepare-only`, `--reference-limit`, `--stop-file` and the
existing core CLI meanings are retained. Any worker count other than1 is
refused. Do not use `--reference-limit` for a production continuation.

Progress remains in `molecule_bank_1m/progress.json`, augmented with scheduler
policy, actual coverage, rates and unmet floors. Per-host thermal state is in
the selected thermal-status directory. The compute lease records child identity
for explicit recovery. Code/config/rules/reference/holdout pins are stored in
`metadata.production_config_<hash>`; altered source or policy creates a distinct
configuration rather than silently changing old job provenance.

An inventory supervisor must verify both unique count and category floors on
every restart, even if the bank already exceeds1million. Read-only helpers:

```python
from yield_aware_scheduling import category_coverage
coverage = category_coverage(sqlite_connection)
floors = config["yield_scheduler"]["minimum_named_category_unique"]
```

## Verification

`./venv/bin/python check_yield_aware_bank.py` verifies:

- deadline enforcement for a native sleep retaining the Python GIL;
- persistence, replacement after timeout and deterministic seeded replay;
- an8MiB Pipe result without joining before reception;
- SIGKILL of a fixture owner, inherited lease refusal, and explicit fixture-child
  cleanup before the lease can be acquired again;
- identical accepted M/E structures and recorded histories compared with the
  frozen chemistry backend (4M and2E fixture candidates), with unchanged QC;
- a productive-band allocation of170distant/15intermediate/15near jobs in a
  200-job fixture, plus E minimum active-time and category-coverage priority.

A separate temporary coordinator integration committed two jobs (4M and3E
candidates), producing8total molecules and7relations, and verified new source
pins and separate chemistry/thermal timing in the job ledger. No production
bank or production process was changed by these checks.
