# Geometric GNN training protocol

`train_geometric_gnn.py` trains a scalar continuous-filter molecular network using
the existing 3D conformers. Coordinates are the stored ETKDGv3 conformations after
MMFF94s/UFF minimization, in Angstrom. They are predicted conformations, not
experimental measurements. The network recomputes interatomic distances from
these positions in every forward pass, expands them into Gaussian radial basis
features, and uses distance-dependent messages with a smooth cutoff. Three
residual interaction layers and mean plus size-normalized sum pooling are the
default. Atomic number and seven stored atomic properties supply chemical input.

Rotating or translating a molecule leaves its prediction unchanged, within
floating-point precision. The model uses scalar invariant features; it does not
predict equivariant coordinates. Pair distances and repeated spatial messages
provide geometric information; explicit angle features are not used in this
first architecture.

## Fixed data and honest interpretation

Both GATv2 and this network use the strict shared cohort in
`balanced_datasets/training_200k/overnight_strict_converged_v2/graph_index.csv`. It contains
191,701 molecules: 161,157 training, 13,756 validation and 16,788 test. The audited
cohort hash is
`041936ffe7ad260c110d4bc8e9226d4622d3d0e79ae023b89cae2d1b789e5126`.
The selected source identities, labels and scaffold splits are preserved. The
audit excludes conflicting normalized identities, unavailable graphs, external
holdout overlap and normalized-connectivity/scaffold overlap between splits.
The predeclared v2 filter additionally requires converged force-field
optimization and finite energy in both architectures' common cohort.
See the adjacent audited `manifest.json` for counts, source hashes and rules.

Label 0 means public reference molecular origin and label 1 means generated
analogue molecular origin. These labels do not establish authenticity or fraud
of a measured pharmaceutical product. A test F1 target of 0.85 is a target,
not a guaranteed result or a real counterfeit detection claim.

Training row weights already balance weighted label mass, M/E mass and groups
within each generation method. `--weight-mode row` applies those weights once in
binary cross entropy, with unit class weights. `--weight-mode none` is a
predeclared ablation. Validation and test metrics are unweighted.

The default spatial graph retains heavy atoms and at most 24 incoming nearest
neighbors per atom within 4.5 Angstrom. Positions are selected from the original
NPZ; no replacement coordinates or new conformations are created. Explicit
hydrogens can be retained with `--include-hydrogens`. A cutoff above the stored
source radius of 5 Angstrom is refused. The trainer reports convergence status
in `data_summary.json` and with predictions. The v2 cohort excludes optimizer
failures before any model metrics are observed; earlier v1 did not.

## Disk use and reproducibility

The first invocation streams only the required arrays out of each compressed
NPZ into a temporary disk directory, then writes a geometry cache of `.npy`
arrays. Training memory maps the derived arrays and opens at most two shards at
a time. Shard-local shuffled batches prevent decompressing a whole NPZ per
molecule. Molecules and graph tensors are not all loaded into RAM.

Source NPZ SHA256 hashes, the strict graph-index hash, cohort hash, geometry
settings, derived-array hashes and SQLite index hash are checked. The cache
refuses changed data. Preparation has an OS writer lock, and each run directory
has a nonblocking OS lock held during fitting or evaluation. The shared cache
is reusable across configurations with identical graph settings.

Every completed epoch atomically saves `last.pt`; an improved selected
validation metric atomically saves `best.pt`. Checkpoints contain model,
optimizer, scheduler, Python/NumPy/Torch RNG states, history, best epoch,
validation threshold and pinned configuration. Resuming requires the same
configuration, source code, data and run directory. `--epochs` can be increased
without resetting the optimizer. Mid-epoch interruption resumes from the last
completed epoch, replaying that partial epoch deterministically on CPU.

Outputs include `progress.json`, `summary.json`, `metrics.csv`, `history.json`,
raw best/last validation probabilities, configuration and data summaries.
`progress.json` updates during training and at every completed epoch. Both
progress and summary expose numeric `best_validation_f1`, `current_epoch` and
boolean `completed` for the supervisor.

## Commands

Prepare the actual existing 3D corpus without fitting:

```sh
./venv/bin/python train_geometric_gnn.py \
  --root balanced_datasets/training_200k/3d \
  --graph-index balanced_datasets/training_200k/overnight_strict_converged_v2/graph_index.csv \
  --cache-root training_cache/geometric_3d --prepare-only
```

Validation-only training:

```sh
./venv/bin/python train_geometric_gnn.py \
  --root balanced_datasets/training_200k/3d \
  --graph-index balanced_datasets/training_200k/overnight_strict_converged_v2/graph_index.csv \
  --cache-root training_cache/geometric_3d \
  --out training_runs/overnight/geometric_64 \
  --hidden 64 --interactions 3 --radial-basis 16 \
  --cutoff 4.5 --max-neighbors 24 --batch-size 64 \
  --epochs 60 --patience 10 --threads 2 --device cpu \
  --selection val_f1 --weight-mode row --seed 42 --no-test
```

Always pass the explicit v2 graph index shown above. The frozen trainer's
backward-compatible default path points at v1.

To resume, repeat that exact command with
`--resume training_runs/overnight/geometric_64/last.pt`. On automatic device
selection, a real forward/backward/optimizer probe must pass before CUDA or MPS
is used. CPU is supported; the local machine reported MPS unavailable.

Choose a configuration using validation only, lock that choice externally for
the entire model family, then repeat the same model/data arguments with
`--evaluate-test --checkpoint training_runs/overnight/geometric_64/best.pt`
and remove `--no-test`. The validation-selected threshold is frozen. Optional
`--test-lock training_runs/overnight/geometric_test_selection.json` specifies
the family-wide immutable candidate pin; its sibling OS lock is held for the
whole evaluation. A crash may resume only that same candidate. Repeating a
successful evaluation returns saved results without evaluating test again.
A different candidate is refused, and training a test-accessed run is refused.

The final outputs are `test_predictions.csv`, `test_results.json` and
`test_metrics` in summary/progress. No test metrics are computed during fitting.

## Verification

```sh
./venv/bin/python check_geometric_gnn.py
```

The temporary 16-molecule fixture checks geometry identity/edge offsets,
heavy-atom selection, bounded mmap loading, deterministic complete sampling,
rotation/translation invariance, sensitivity to changed distances, source hash
and nonfinite-coordinate failure guards, atomic epoch checkpoint outputs,
validation-only fitting, family test pinning, and exact CPU weight/RNG equality
between two uninterrupted epochs and one epoch followed by resume.
