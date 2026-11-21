# Fixed heterogeneous ensemble search

The frozen policy is `heterogeneous_ensemble_policy_20261107.json`, SHA256
`95cf859a671d93d7db53e1e89f35350a614c225313a934b6d84e97a7ca10c152`.
It was declared before the remaining seed 43/44 runs completed. The original
trainers, cohort and original ensemble helpers were not edited.

Every final member uses the same audited cohort of 191,701 molecules, including
161,157 training molecules and 13,756 validation molecules. The selector combines saved
validation predictions by source ID, verifies labels and complete membership,
checks the selected checkpoint epoch/seed/cohort/F1, and pins the actual
checkpoint, prediction, summary, source, cohort and policy files. Its reuse of
the original helper is itself pinned. It performs no model forward passes.

## Predeclared members and candidates

Eight members are required for final selection:

- GATv2 JK attention champion, seeds 42, 43, 44.
- GATv2 JK wide, seed 42.
- Angular geometric champion, seeds 42, 43, 44.
- Distance geometric baseline, seed 42.

Sixteen candidates are declared, with exact rational weights:

1. Eight individual members.
2. The three-seed GAT mean and three-seed angular-geometric mean.
3. GAT/geometric seed-mean blends at 50/50, 75/25 and 25/75.
4. 75% GAT seed mean plus 25% wide GAT.
5. Equal weight for four blocks: GAT seed mean, wide GAT, angular seed mean,
   distance geometric baseline.
6. Equal weight for all eight members.

The threshold maximizes validation F1; ties choose the largest threshold.
Candidate ties retain the first declared candidate. There is no continuous
weight optimization, test-guided tuning, or replacement of absent members.

## Commands

Build the final declaration after all eight members finish:

```sh
./venv/bin/python heterogeneous_validation_ensemble.py spec \
  --policy heterogeneous_ensemble_policy_20261107.json \
  --policy-sha256 95cf859a671d93d7db53e1e89f35350a614c225313a934b6d84e97a7ca10c152 \
  --workspace /Users/itanasa/Counterfeit-Detector \
  --search-root training_runs/large_search_20261101 \
  --cohort-dir balanced_datasets/training_200k/overnight_strict_converged_v2 \
  --mode final \
  --out training_runs/heterogeneous_20261107/final_candidates.json
```

The command prints the actual frozen specification SHA. Pass that SHA as
`--spec-sha256` to the selection command:

```sh
./venv/bin/python heterogeneous_validation_ensemble.py select \
  --spec training_runs/heterogeneous_20261107/final_candidates.json \
  --spec-sha256 SPEC_SHA_PRINTED_BY_THE_PREVIOUS_COMMAND \
  --out training_runs/heterogeneous_20261107/final_selection
```

Selection writes immutable `selection.json` and validation probabilities. It
does not start evaluation. The separate evaluator requires the exact SHA of
that selection file and a shared test lock:

```sh
./venv/bin/python evaluate_heterogeneous_ensemble.py \
  --selection training_runs/heterogeneous_20261107/final_selection/selection.json \
  --selection-sha256 EXACT_FROZEN_SELECTION_SHA \
  --test-lock training_runs/heterogeneous_20261107/global_heterogeneous_test_lock.json \
  --out training_runs/heterogeneous_20261107/final_test \
  --device cpu --threads 2
```

The evaluator locks the exact selection, threshold, weights, member hashes and
cohort before accessing test labels or invoking model adapters. It computes
only aggregate test metrics. A different selection cannot reuse that lock.
An unchanged completed invocation returns the saved result. No production
test invocation was performed while implementing these files.

## Exploratory evidence

Exploratory mode permits completed available members, compares their singles
and equal mean, and sets `eligible_for_final_test=false`. The evaluator rejects
such artifacts. It never interprets a missing seed as a completed member.

The actual artifact at
`training_runs/heterogeneous_20261107/exploratory_selection/selection.json`
uses five completed models: GAT42, GAT43, wide42, angular42, distance42. It has:

| Validation metric | Value |
|---|---:|
| F1 | 0.8405367136403479 |
| ROC-AUC | 0.9097313828575139 |
| Precision | 0.7837294102854269 |
| Recall | 0.9062227391683629 |
| Reference false-positive rate | 0.2500726955510323 |
| Threshold | 0.25447643920765867 |
| Molecules | 13,756 |

These are exploratory validation scores. Repeated configuration, epoch and
threshold selection makes validation estimates optimistic. The label denotes
public-reference versus generated molecular origin; actual medicine-product
fraud detection is not established by this benchmark.

## Verification

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv/bin/python check_heterogeneous_ensemble.py
```

Six isolated fixtures pass: shuffled complete alignment and invalid-ID/label/
NaN rejection; exact rational weights and deterministic ties; immutable source
pins and idempotence; missing final members and exploratory test rejection;
altered candidate-policy rejection; shared lock before mocked inference and
reselection rejection. They use fake checkpoint metadata and small synthetic
tables, without production graphs, trained model forwards, or production test
data.

`check_finish_heterogeneous_research.py` also checks the production finalization
bridge on these synthetic fixtures: unfinished validation blocks selection and
test; a missing member blocks evaluation and inventory release; an evaluation
failure cannot publish the readiness marker; success publishes that marker
only after the frozen selection's result exists. Its 10 checks include the six
selector checks above. No real model forwards or production test rows are used.
