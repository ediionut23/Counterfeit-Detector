# Large molecule bank and 200k balanced datasets

## Why the first balanced sets were small

The first builder used `final_dataset/authentic_eligible.csv`, which contains
27,261 reference molecules after strict quality filters. It balanced each
generator against that smaller class and then removed test rows with child
scaffolds seen in train or validation. This gave about 53k rows per set even
though the aggregate generated bank is much larger.

The updated build uses the 400k aggregate, the raw Version M bank, and the
category-conditioned GA bank. It adds 80,000 ChEMBL reference molecules that pass the same quality gate as
the generated bank, while excluding exact overlaps with existing authentic and
generated structures. ChEMBL records are public literature-curated compounds;
their use as label 0 means “public reference, not generated”, not “approved
medicine” or “verified genuine product”.

## Outputs

- `molecule_bank_500k/molecules.csv` contains **657,106 unique molecules**:
  512,104 generated analogues and 145,002 public reference compounds.
- `balanced_datasets/generator_versions_200k/generator_m/dataset.csv` contains
  213,422 rows, balanced 106,711/106,711.
- `balanced_datasets/generator_versions_200k/generator_e/dataset.csv` contains
  205,396 rows, balanced 102,698/102,698. The E pool includes 45,836 unique
  category-conditioned GA molecules in addition to pure Version E provenance.
- `balanced_datasets/generator_versions_200k/combined/dataset.csv` contains
  213,160 rows, balanced 106,580/106,580; its generated class has equal M and
  E-family counts within each split.
- `balanced_datasets/training_200k/train.csv` is the ready-to-train split
  with **171,616 molecules** (85,808 per label); `val.csv` and `test.csv` each
  remain separate for model selection and final evaluation. These files are
  exported from the combined set without changing the scaffold split.

Every split has balanced labels. Source-parent scaffold groups stay disjoint
across train/validation/test, and test molecules whose own scaffold appears in
train or validation are removed. The larger ChEMBL pool lets all versions stay
above 200k with this strict child-scaffold filter enabled. Summaries record
actual split sizes, removed rows, categories, source composition and overlap
checks.

## Rebuild

```bash
venv/bin/python download_chembl_controls.py \
  --target 80000 --out molecule_bank_500k/chembl_controls.csv \
  --seed 20261031 --workers 4
venv/bin/python build_large_molecule_bank.py \
  --bank final_dataset_400k --ga-root version_ga \
  --m-root version_m \
  --controls molecule_bank_500k/chembl_controls.csv \
  --out molecule_bank_500k --minimum 500000
venv/bin/python build_generator_balanced_datasets.py \
  --bank final_dataset_400k --ga-root version_ga \
  --authentic-extra molecule_bank_500k/chembl_controls.csv \
  --out balanced_datasets/generator_versions_200k \
  --seed 20261031 --min-per-class 100000
venv/bin/python export_training_splits.py \
  --source balanced_datasets/generator_versions_200k/combined/dataset.csv \
  --out balanced_datasets/training_200k
```

The separate split files and their row counts, label counts, source checksum,
and SHA-256 manifest are recorded under `balanced_datasets/training_200k/`.
For an all-rows training view with macro-subcategory loss weights, use
`balanced_datasets/training_200k/subcategory_weighted/dataset.csv`; its
weighting logic and cautions are in
[`MOLECULE_BANK_AND_TRAINING_REPORT.md`](MOLECULE_BANK_AND_TRAINING_REPORT.md).

## 3D GNN representation

The 3D build is stored under `balanced_datasets/training_200k/3d/`, with
separate `train/`, `val/`, and `test/` shard directories. Each shard contains
`graphs_3d.npz` for packed graph arrays and `conformers.sdf.gz` for the
corresponding hydrogen-complete structures. The arrays include atom and bond
features, 3D positions in Å, covalent and radius edges with distances, and
bond-neighbor angles in radians. Each graph also carries the binary label,
split, generator source, category, and subcategory. Of 213,160 input rows,
213,150 received a graph (99.9953% coverage); the ten failures are in
`failed_embeddings.csv`. `balanced_graph_index.csv` selects 213,146 unique
graphs and exactly balances the labels within each split after dropping four
majority-label graphs. The selected train split contains **171,614 graphs**
(85,807 per label); validation has 21,448 and test has 20,084. Counts and
excluded IDs are in `balanced_index_summary.json`; archive and index hashes
are in `manifest_sha256.json`.

Coordinates are ETKDGv3 conformer predictions minimized with MMFF94s (UFF
fallback), not experimentally measured structures. The build retains
hydrogens and uses the lowest-energy result among two attempted conformers.
The 3D pipeline is reproducible with:

```bash
venv/bin/python prepare_balanced_training_3d.py \
  --input balanced_datasets/generator_versions_200k/combined/dataset.csv \
  --out balanced_datasets/training_200k/3d \
  --workers 6 --conformers 2 --shard-size 15000 --seed 20261026
venv/bin/python build_balanced_graph_index.py \
  --root balanced_datasets/training_200k/3d --seed 20261026
```

Use `load_3d_graph_dataset.iter_graphs(root, split)` to stream graph records
using the balanced index; edge and angle indices are localized to each
molecule. Train only on `train`, select models on `val`, and use `test` once
for final evaluation. The fine-grained subcategory-balanced view remains a
separate, smaller dataset described in
[`SUBCATEGORY_BALANCED_DATASET.md`](SUBCATEGORY_BALANCED_DATASET.md).

The downloader queries the ChEMBL small-molecule web service for structures
with 150–800 Da molecular weight, then applies the project filters for a single
component, 10–70 heavy atoms, QED ≥ 0.35, SA ≤ 3.8 and no configured structural
alerts. It excludes exact canonical-SMILES overlap against the local authentic
and generated banks. The 80,000 output rows came from 216 randomly selected
catalog pages; 87,556 passed the quality filters before the final seeded
selection. The query and current ChEMBL release are captured in
`molecule_bank_500k/chembl_controls.summary.json`.

ChEMBL documents its [molecule API and pagination](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)
and [CC BY-SA 3.0 license](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions).
Attribute ChEMBL when redistributing the added references and preserve the
license for adapted ChEMBL data.

## Interpretation

The binary benchmark separates generated analogues from public reference
molecules. It does not train or validate a detector of real counterfeit
medicines. Split by scaffold to limit molecular-family leakage, and report
metrics by generator source as well as overall.
