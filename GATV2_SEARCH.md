# Configurable GATv2 search

`train_gatv2_search.py` is a separate architecture-search trainer. The original
`train_gatv2_experiment.py` remains unchanged, SHA256
`198bb0b562b6aa010644d36b1aa62edfaafa9d7d1e4a06c0335cc75d7acdc783`.
It forks that checkpoint/cache/validation protocol and records this base hash,
its own source hash, architecture version and all architecture settings.

This is synthetic molecular-origin discrimination, not confirmed counterfeit
medicine detection. A higher F1 on this task does not establish real-world fraud
detection. The target F1 is an objective, not an assured score.

## Common full cohort

Defaults point to `balanced_datasets/training_200k/overnight_strict_converged_v2/`:
191,701 molecules, including all **161,157 training molecules**. The exact CSV,
graph-index and audited split hashes are verified. Every production trial uses
this entire training split, including the short screening stage. Fixtures use
temporary toy data solely to test implementation.

Atom/bond features and their disk-cache fingerprint are identical to the frozen
base trainer, so an existing cache is reused. Training visits each row once per
epoch with the same bounded shard-window shuffle. Row weights, if requested,
already contain class/method/subgroup balancing; no second class weight or
weighted sampler is applied.

## Architecture controls

| Argument | Choices and behavior |
| --- | --- |
| `--readout mean_sum` | Final-layer mean plus sum, matching the base architecture |
| `--readout mean_max` | Final-layer mean plus maximum |
| `--readout attention` | Learned, permutation-invariant weighted sum of atom states |
| `--readout jk` | Concatenate graph representations from every convolutional layer |
| `--jk-pool` | `mean_sum`, `mean_max`, or `attention` applied separately at every JK layer |
| `--norm` | Per-atom `layer` normalization or atom-batch `batch` normalization |
| `--input-dropout` | Dropout on the initial embedding; default 0 |
| `--feature-dropout` | Residual message dropout; default inherits `--dropout` |
| `--attention-dropout` | GAT attention coefficient dropout; default inherits `--dropout` |
| `--readout-dropout` | Prediction-head dropout; default inherits `--dropout` |

JK excludes the input embedding and includes every convolutional layer,
including the final layer. Attention pooling has one independent gate per JK
layer. BatchNorm acts on atoms, never on pooled graph batches. Its running mean,
variance and update count are checkpointed. A singleton atom uses saved running
statistics rather than silently dropping a molecule. Validation/test use eval
mode, so held-out data never update BatchNorm statistics.

All ordinary base flags remain available. Architecture, optimizer, scheduler and
patience settings are immutable on resume/evaluation. The epoch cap may grow for
successive halving. Increasing it does not restart a trial that already stopped
under its patience criterion. Base-trainer checkpoints cannot be loaded into
the search trainer. Search checkpoints require the exact source version that
created them; code changes require a new versioned trainer/run.

## Bounded full-data schedule

Freeze eight configurations before examining their validation scores. The
existing three frozen base configurations can be retained, with these five
additional contrasts:

| New configuration | Readout | Norm | Layers | Weight mode |
| --- | --- | --- | --- | --- |
| Max pooling | `mean_max` | `layer` | 4 | `none` |
| Attention pooling | `attention` | `layer` | 4 | `none` |
| Multi-layer readout | `jk`, pool `mean_sum` | `layer` | 4 | `none` |
| Multi-layer max + BatchNorm | `jk`, pool `mean_max` | `batch` | 4 | `none` |
| Multi-layer attention + BatchNorm | `jk`, pool `attention` | `batch` | 6 | `none` |

Use the same declared hidden size, head count and optimization budget when
comparing readouts. An example uses hidden 128, four heads, learning rate .001,
batch size 128 and dropout .2; CPU settings must be chosen before launching the
plan. The supervisor owns the final immutable configuration list.

1. Train all eight configurations with seed 42, a 20-epoch cap and patience 12,
   on the full 161,157-row training split.
2. Select the best two using validation F1 only; resume their `last.pt` to a
   100-epoch cap without changing other hyperparameters.
3. Repeat the validation-selected champion configuration with seeds 43 and 44,
   each on the same full training split with a 100-epoch cap and patience 12.
4. Select one final model or predeclared validation-only ensemble. Apply its
   locked validation threshold to the test set once, under the shared family
   test lock. Member test scores are not used to choose an ensemble.

Example screening command:

```sh
./venv/bin/python train_gatv2_search.py \
  --dataset-csv balanced_datasets/training_200k/overnight_strict_converged_v2/dataset.csv \
  --graph-index balanced_datasets/training_200k/overnight_strict_converged_v2/graph_index.csv \
  --split-manifest balanced_datasets/training_200k/overnight_strict_converged_v2/manifest.json \
  --out-dir training_runs/gatv2_search/jk_attention_batch_seed42 \
  --readout jk --jk-pool attention --norm batch \
  --hidden 128 --layers 6 --heads 4 --dropout .2 --input-dropout .05 \
  --lr .001 --batch-size 128 --weight-mode none \
  --epochs 20 --patience 12 --seed 42 --threads 2 --device cpu --no-test
```

Promotion preserves optimizer, scheduler, Python/NumPy/Torch and shuffle RNG:

```sh
./venv/bin/python train_gatv2_search.py \
  --resume training_runs/gatv2_search/jk_attention_batch_seed42/last.pt \
  --epochs 100 --no-test
```

The base protocol's atomic `best.pt`/`last.pt`, checkpoint-recovery behavior,
`progress.json`, `summary.json`, raw validation probabilities and subgroup
metrics remain available. Training never predicts test probabilities. Standalone
final evaluation retains `--evaluate-test --checkpoint ... --test-lock ...`.

## Ensemble inference hook

`build_model(config)` restores the architecture declared in a search checkpoint.
`infer_selected_checkpoint(checkpoint_path, graph_index, split_manifest, split,
device, cache_root, threads, batch_size)` returns `source_id`, `label`, raw
`probability`, exact `checkpoint_sha256` and `cohort_sha256`. It checks source,
cohort, architecture and trainer-code pins while holding the member run lock.
It writes no metrics or test-probability files and chooses no threshold.

The calling evaluator must hold the one shared family OS lock, pin the complete
ensemble membership/weights/validation threshold before test access, compare
each returned checkpoint hash with that pin and report aggregate test metrics
only. The adapter itself is not an independent authorization to evaluate test
members outside that locked evaluation.

For the three original baseline runs, the same raw adapter also dispatches
original `bond_residual_gatv2` checkpoints to the byte-frozen original model.
It verifies the original trainer source hash and checkpoint provenance against
the base protocol pin. The ensemble records the original training script and
this separate inference adapter with their own hashes. This compatibility is
restricted to raw inference; search training/resume still reject base checkpoints.

## Verification

```sh
./venv/bin/python validate_gatv2_search.py
```

The tiny fixture checks every readout/norm branch for finite gradients and node
permutation invariance; default-model parity with the frozen base; BatchNorm
buffer behavior; weighted training; exact epoch-boundary resume; best-checkpoint
recovery; raw-adapter agreement; immutable configuration; process locks; and
idempotent, globally pinned final test. It does not estimate training accuracy.
No production training is launched by the fixture.
