# GATv2 experiment protocol

`train_gatv2_experiment.py` measures **synthetic molecular origin**: public
reference structures (0) versus generated analogues (1). These labels do not
establish authentic products or confirmed counterfeits. F1 ≥ 0.85 is a target,
not a promised result.

The common 2D/3D cohort is the immutable
`balanced_datasets/training_200k/overnight_strict_v1/` directory: 191,819 rows,
161,225 train, 13,774 validation and 16,820 test. The audited cohort SHA256 is
`c83f6845dd3afbf13b82950468266befbb87f7a7d2aa03b2b6fcb9c49edf9248`.
`dataset.csv`, `graph_index.csv` and `manifest.json` are passed together; the
trainer checks source hashes and the exact source-ID/split/label cohort hash.
The index chooses the same molecules as the geometric trainer. GATv2 builds 2D
bond graphs from SMILES and does not consume 3D coordinates.

## Model and data handling

Atomic-number embeddings and 21 atom features include charge, aromaticity,
hybridization, canonical-order chirality, ring membership and mass. Fourteen
edge features encode bond order, aromaticity, conjugation, rings and stereo.
GATv2 attention consumes these bond features. Residual connections, layer
normalization and dropout precede concatenated mean/sum pooling and a binary
prediction head.

Graphs are cached in bounded tensor shards, loaded lazily. Training shuffles
three-shard windows and rows within each window; each training row is visited
once per epoch, without replacement. This keeps the disk cache bounded and
avoids rereading an entire shard for each randomly chosen molecule. Use the
default `--loader-workers 0` for deterministic CPU restart and modest memory.
Prebuild the shared cache once before running trials; do not concurrently build
the same cache from different output directories.

`--weight-mode none` gives each training row equal weight. `--weight-mode row`
uses train-only weights from the strict graph index, already balancing
reference/generated mass, M/E mass and subgroups within each generator. There
is no extra class weighting and no weighted sampler. The weighted loss divides
by the fixed global training mean weight. Validation and test metrics always
use physical rows with equal weight.

## Validation search and final test

Use three predeclared configurations: baseline without weights, the same model
with row weights, then a deeper model without weights. Keep the same cohort and
seed and cap each at 80 epochs with patience 12. The supervisor selects the
highest `best_validation_f1` across these runs. Do not expand this search in
response to test scores. Architecture and runtime settings are explicit CLI
arguments so CPU resource limits can be chosen before launching the schedule.

Example validation-only command:

```sh
./venv/bin/python train_gatv2_experiment.py \
  --dataset-csv balanced_datasets/training_200k/overnight_strict_v1/dataset.csv \
  --graph-index balanced_datasets/training_200k/overnight_strict_v1/graph_index.csv \
  --split-manifest balanced_datasets/training_200k/overnight_strict_v1/manifest.json \
  --out-dir training_runs/overnight/gatv2_trial1 \
  --weight-mode none --hidden 128 --layers 4 --heads 4 --dropout .2 \
  --lr .001 --batch-size 128 --epochs 80 --patience 12 \
  --seed 42 --device cpu --threads 2 --no-test
```

Add `--prepare-cache-only` for cache construction without training. Training is
validation-only even when `--no-test` is omitted. Each epoch chooses the F1
decision threshold using validation probabilities only. The default checkpoint
selection metric is validation F1; `--selection-metric roc_auc` is an optional
predeclared alternative. Scheduler and early stopping follow that selected
metric. Threshold ties choose the highest threshold. No test predictions are
computed during training.

Resume a run, allowing a larger epoch cap while preserving the model, optimizer,
scheduler, Python/NumPy/Torch RNG and sampler RNG:

```sh
./venv/bin/python train_gatv2_experiment.py \
  --resume training_runs/overnight/gatv2_trial1/last.pt --epochs 80 --no-test
```

After selecting exactly one configuration using validation, evaluate its
`best.pt` with a **shared** lock path across all candidate directories:

```sh
./venv/bin/python train_gatv2_experiment.py --evaluate-test \
  --checkpoint training_runs/overnight/gatv2_trial1/best.pt \
  --test-lock training_runs/overnight/gatv2_global_locked_test.json
```

The selected checkpoint's saved validation threshold is applied unchanged.
The test JSON pins the checkpoint and cohort hashes. Another checkpoint is
rejected; an already completed invocation returns the existing result. An
interrupted test may restart only with the same pinned candidate. OS `flock`
locks protect each entire output run and the shared test evaluation, and release
automatically when a process exits. An overlapping trainer/evaluator fails
instead of writing concurrent checkpoints or test files.

## Outputs and supervisor schema

`last.pt` and `best.pt` are written atomically after every completed epoch. Both
include model, optimizer, scheduler, RNG, configuration, dataset hashes and
history for their own epoch. `last.pt` also embeds the selected best checkpoint,
so a crash between writing the two files can be repaired consistently at resume.
`run_metadata.json` records trainer/source hashes and software versions.

`progress.json` and `summary.json` share the same flat schema and are replaced
atomically each epoch and at completion:

| Field | Meaning |
| --- | --- |
| `state` | `training`, `validation_search_complete` or `locked_test_complete` |
| `current_epoch` | Integer last completed epoch; locked test uses selected epoch |
| `completed` | Boolean: this validation run or final test invocation completed |
| `best_validation_f1` | Numeric F1 of the selected validation checkpoint |
| `best_validation_auc` | Numeric ROC-AUC of that same checkpoint |
| `best_epoch` | Selected validation epoch |
| `best_checkpoint` | Path to selected checkpoint |
| `cohort_sha256` | Exact immutable cohort identifier |
| `test_evaluated` | False during validation search; true after final test |
| `test_metrics` | Present only after locked test; includes F1, AUC, precision, recall, FNR and counts |

With the default F1 selection, `best_validation_f1` is the maximum validation F1.
With optional AUC selection, it is F1 at the AUC-selected checkpoint.
`history.json` contains epoch loss, validation metrics, fitted threshold,
learning rate and elapsed seconds. `val_probabilities.csv` contains raw
probabilities from the best validation epoch; `test_probabilities.csv` is only
created by locked final evaluation. Group CSVs report M/E and subcategory
results, with the **same reference negative pool shared across groups**.
Subcategory precision/F1 therefore also depend on the group's prevalence;
recall/FNR directly describe errors for that generated subgroup.

The engineering fixture runs only tiny temporary toy data, verifies bit-exact
CPU epoch-boundary resume, weighted loss execution, source guards and exclusive
test/run locks. It does not estimate scientific performance:

```sh
./venv/bin/python validate_gatv2_experiment.py
```
