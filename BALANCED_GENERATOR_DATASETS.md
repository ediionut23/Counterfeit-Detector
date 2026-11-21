# Balanced M/E datasets and Version M 3D graphs

> This page documents the first ~53k balanced build. The expanded datasets
> with at least 200k rows each supersede those CSV outputs and are documented
> in [`LARGE_DATASET_EXPANSION.md`](LARGE_DATASET_EXPANSION.md).

## Balanced datasets

Build the three balanced CSV datasets from `final_dataset/molecules.csv`,
`final_dataset/relations.csv`, and `final_dataset/authentic_eligible.csv`:

```bash
venv/bin/python build_generator_balanced_datasets.py \
  --bank final_dataset --out balanced_datasets/generator_versions \
  --seed 20261030
```

The script creates `generator_m/`, `generator_e/`, and `combined/`; each has a
`dataset.csv` and `summary.json`. The shared `summary_all.json` records source
pool counts and dataset-level totals.

- M-only and E-only use molecules whose `methods` provenance is exactly `M`
  or exactly `E`. The 4,522 molecules produced by both methods are omitted from
  these source-specific comparisons.
- The combined set draws equal M and E generated analogues within each split.
- Label 0 is an eligible authentic source molecule. Label 1 means generated
  analogue; it does not mean a medicine was confirmed counterfeit.
- Each split is label-balanced. Source parent Bemis-Murcko scaffolds stay in
  one 80/10/10 train/validation/test split. Test rows whose own child scaffold
  occurs in train or validation are removed before balancing.
- The generator-category and method counts are retained in each summary for
  stratified evaluation.

The supplied run used seed `20261030`. It produced 53,708 rows for M-only,
53,848 for E-only, and 53,496 for combined. Every split is exactly balanced;
combined has equal M and E positive counts in each split. See the JSON files
for exact counts and scaffold audits.

## Version M 3D tensors

Prepare the per-category Version M graphs with:

```bash
venv/bin/python prepare_version_m_3d.py --root version_m \
  --workers 6 --conformers 2 --radius-cutoff 5.0 --seed 20261030
venv/bin/python audit_version_m_3d.py --root version_m
```

Each category writes `version_m/3d/<category>/graphs_3d.npz`,
`conformers.sdf.gz`, and `conformers_3d_report.json`. The packed graphs contain
atom and bond features, explicit hydrogen coordinates, directed covalent
edges, radius edges and distances, and valence-angle indices and values. The
selected conformer is the lower-energy converged MMFF94s geometry among the
ETKDGv3 candidates, with UFF fallback. The Version M run uses two candidates
per molecule to make the 50,000-molecule preparation practical; Version GA's
existing bank used eight.

The completed Version M run produced 49,960 of 50,000 graph/SDF records
(99.92% coverage). All five category archives passed the structural and
index consistency audit. The 40 molecules without tensors are listed in
`version_m/3d/failed_embeddings.csv`.

These are force-field predictions. They are not experimentally measured
geometries. The aggregate `final_dataset/` also has conformer SDF shards for
243,426 unique molecule structures, as recorded in
`final_dataset/conformers_report.json`; those SDF conformers do not provide
the packed graph arrays created here.

The balanced CSVs retain `molecule_id` and all source/split fields. Version M
graph tensors are packed in the original five 10,000-row category banks; the
balanced M set is sampled from the larger aggregate M provenance in
`final_dataset/`, so not every balanced M row is guaranteed to have a tensor
in the dedicated `version_m/3d/` bank. Use the aggregate SDF coverage or create
a matching graph archive before training a 3D model on those balanced CSVs.
