# Angular geometric GNN variants

`train_geometric_variants.py` adds a new angular model while preserving the
original `train_geometric_gnn.py` file byte for byte. It binds model/parser
extensions only inside its own Python process and reuses the unchanged engine's
lazy data access, epoch checkpoints, optimizer/scheduler, RNG restoration,
validation-only selection and OS locks. Checkpoints additionally pin the new
trainer and model module SHA256 hashes and record the geometry variant.

## What changes in the model

The existing model filters messages using interatomic distance. The new
`--geometry-variant angular` also describes the arrangement of neighbors around
each central atom. For each radial channel it computes weighted averages of
`cos(theta)` and `P2(cos(theta)) = (3 cos(theta)^2 - 1)/2` over distinct neighbors.
Self-pairs are excluded. These scalars enter the central atom features and the
filters of every message-passing layer. The filters still depend on actual
distances with a smooth 4.5 Angstrom cutoff by default.

First and second moments of unit neighbor vectors give these angular averages
without enumerating all neighbor pairs. For radial weights `w_j`, the first
angular numerator is `|sum_j w_j u_j|^2 - sum_j w_j^2`; its denominator is
`(sum_j w_j)^2 - sum_j w_j^2`. Contracting second moments similarly gives the
second Legendre-polynomial average. Directed-edge context excludes that edge's
own source before averaging other neighbors. Channels with negligible mass
receive zero angular context. Work and intermediate memory scale linearly in
the number of radius edges for a fixed number of radial channels.

The angular features are differentiable contractions of the actual coordinate
vectors. Rigid rotations and translations leave the returned scalars and model
predictions unchanged. The network is an invariant classifier; it does not
predict equivariant coordinates or encode every possible angular/chiral
configuration. These features summarize a neighborhood's angular distribution.

`--geometry-variant distance` exactly preserves the original model architecture
and seeded initial weights, providing a compatible control in the new wrapper.
The angular variant adds only 3,521 parameters at hidden64/3interactions/rbf16/
angular-basis6: 73,346 total versus 69,825 in that distance control.

## Same real coordinate files and common cohort

Both variants reuse the existing coordinate/neighbor mmap caches. They never
generate positions or change molecular identity, labels or splits. The source
positions remain predicted ETKDG/force-field conformations, not experimental
measurements. The production cohort is the common audited and converged v2
cohort: 191,701 total, 161,157 train, 13,756 validation and 16,788 test, with
cohort SHA256
`041936ffe7ad260c110d4bc8e9226d4622d3d0e79ae023b89cae2d1b789e5126`.
The task remains molecular origin classification: generated analogue versus
public reference, rather than validated medicine fraud detection.

## Predeclared full-data search

All runs use the entire 161,157-row training partition. No smaller training
subset is used to screen a configuration. Recommended paired configurations:

| Configuration | Trainer | Hidden/layers | Radial/angular channels | Weights | LR |
|---|---|---|---|---|---|
| D64 | original frozen distance | 64 / 3 | 16 / unused | none | 0.001 |
| D96 | original frozen distance | 96 / 4 | 32 / unused | row | 0.0007 |
| A64 | new angular | 64 / 3 | 16 / 6 | none | 0.001 |
| A96 | new angular | 96 / 4 | 32 / 6 | row | 0.0007 |

Common settings are cutoff4.5, neighbors24, batch64, dropout0.1, seed42,
CPU2threads and fixed patience12. Screen all four for up to20epochs, promote the
two best validation-F1 configurations by resuming their original checkpoints
with only the epoch cap increased to100, then repeat the selected champion
with seeds43 and44. An already triggered patience stop remains stopped on
resume; promotion never resets optimizer/scheduler/RNG or silently resets
patience. The decision threshold is selected on validation only. Test is
reserved for the externally locked final ensemble.

Example new-model screen:

```sh
./venv/bin/python train_geometric_variants.py \
  --root balanced_datasets/training_200k/3d \
  --graph-index balanced_datasets/training_200k/overnight_strict_converged_v2/graph_index.csv \
  --cache-root training_cache/geometric_overnight_v2 \
  --out training_runs/expanded/geometric_a64_seed42 \
  --geometry-variant angular --angular-basis 6 \
  --hidden 64 --interactions 3 --radial-basis 16 \
  --cutoff 4.5 --max-neighbors 24 --batch-size 64 --dropout 0.1 \
  --lr 0.001 --weight-mode none --epochs 20 --patience 12 \
  --threads 2 --device cpu --selection val_f1 --seed 42 --no-test
```

Promotion repeats identical arguments with `--epochs 100` and
`--resume training_runs/expanded/geometric_a64_seed42/last.pt`. Fresh seed
repetitions use distinct output directories and do not resume seed42.

## Ensemble inference API

The final selector can import these functions from the new trainer:

```python
model = build_model(checkpoint["configuration"])
predictions = infer_selected_checkpoint(
    checkpoint_path, graph_index, split_manifest, split="val", device="cpu",
    cache_root="training_cache/geometric_overnight_v2", threads=2, batch_size=64,
)
```

Both frozen-original and new-variant checkpoints are supported. The adapter
returns `source_id`, `molecule_id`, `label`, `probability` NumPy arrays plus
split/cohort/checkpoint hashes. It does not calculate individual classification
metrics or loss, select a threshold, write prediction files or create individual
test locks. It validates checkpoint implementation and data pins. The ensemble
caller must hold its single immutable ensemble-selection pin and OS lock for
the entire final test pass. `cache_root=None` reuses the original cache path
from the checkpoint-adjacent `data_summary.json`.

## Verification and CPU benchmark

`./venv/bin/python check_geometric_variants.py` passed. Checks include exact
distance-control equivalence, rigid-motion invariance, changed predictions
when an angle changes at fixed represented edge distances, distance sensitivity,
finite nonzero coordinate gradients, isolated atoms, exact resumed model/RNG
equivalence, refusal to change variants during resume, atomic checkpoint
outputs, unchanged original trainer, both inference adapters and test locking.

On 2026-11-01, a CPU2thread probe of512 existing strict-cohort molecules used
12,937 heavy atoms and135,968 directed spatial edges. Median forward/backward/
optimizer time per64 molecules after the first batch was:

| Model | Median training batch | Extrapolated train computation/epoch |
|---|---:|---:|
| Distance64, 3layers, rbf16 | 0.0253s | 63.7s |
| Angular64, 3layers, rbf16/angular6 | 0.0313s | 78.9s |
| Angular96, 4layers, rbf16/angular6 | 0.0567s | 142.7s |

The last benchmark used rbf16, while the proposed A96 search uses rbf32.
These extrapolations exclude loading, validation and competing processes and
are not wall-clock guarantees. The probe did not launch a full training run
or write production model checkpoints.
