# Counterfeit Detector Benchmark

Molecular-graph datasets for studying detection of structurally modified
medicine candidates. The counterfeit labels describe **generated analogues of
authentic molecules**, not laboratory-confirmed counterfeit products. See
[`WIKI.md`](WIKI.md) for the construction protocol and
[`benchmark/README.md`](benchmark/README.md) for the CSV and SDF formats.

The larger, deduplicated source bank for constructing new training datasets is
in [`final_dataset/`](final_dataset/README.md). It keeps rule-based (`M`) and
evolutionary (`E`) candidates in selectable subcategories, with relation
provenance and separate chemical-quality and 3D audits. Its `summary.json` is
the source of truth for the final unique-molecule count; subcategory counts
can overlap. The audited counts are in
[`final_dataset/REPORT.md`](final_dataset/REPORT.md) and the
[`final dataset PDF`](output/pdf/Raport_dataset_final_2026-09-26.pdf). A possible
follow-on study of molecule-induced changes in human cell communication is outlined in
[`RESEARCH_IDEA_CELL_COMMUNICATION.md`](RESEARCH_IDEA_CELL_COMMUNICATION.md).
The category-conditioned genetic extension and packed 3D GNN representation
workflow are documented in [`GENETIC_ALGORITHM_AND_3D.md`](GENETIC_ALGORITHM_AND_3D.md).
Generated analogues are synthetic labels and their 3D structures are
force-field predictions, not experimental observations.
An evidence-grounded product-case pilot now records declared and detected
ingredients from official reports, with a claimed-versus-observed graph loader
and a separate measured NIR/HPLC data import. See
[`real_world_falsification/`](real_world_falsification/README.md) and
[`the working report`](REAL_WORLD_FALSIFICATION_REPORT.md) for scope,
external-holdout policy, data quality issues and the prior model's limitations.
The [generation scaling plan](SCALABLE_SCENARIO_GENERATION_PLAN.md) describes
API-conditioned synthetic scenarios, large-bank pretraining, coverage targets
and the generator throughput fixes applied on 2026-10-31.
The expanded bank of 657,106 unique molecules and three balanced datasets with
more than 200,000 rows each are documented in
[`LARGE_DATASET_EXPANSION.md`](LARGE_DATASET_EXPANSION.md).
The fine-grained dataset balanced across 85 generator/category/subcategory
labels, including its rare-class limits and split counts, is documented in
[`SUBCATEGORY_BALANCED_DATASET.md`](SUBCATEGORY_BALANCED_DATASET.md).
The >100k training split and sharded 3D GNN graph representation are documented
in [`LARGE_DATASET_EXPANSION.md`](LARGE_DATASET_EXPANSION.md).
The molecule-bank inventory and an explanation of binary versus subcategory
balancing are summarized in
[`MOLECULE_BANK_AND_TRAINING_REPORT.md`](MOLECULE_BANK_AND_TRAINING_REPORT.md).
The full-size weighted training view is at
[`subcategory_weighted/dataset.csv`](balanced_datasets/training_200k/subcategory_weighted/dataset.csv);
its per-row loss weights are consumed by `train_gnn.py` and the 3D graph loader.
An extension toward prospective in-body nanosensing is described in
[`RESEARCH_IDEA_NANOSENSORS.md`](RESEARCH_IDEA_NANOSENSORS.md).
A focused literature map and testable active-inspection proposal for vascular
microrobots is in
[`RESEARCH_IDEA_VASCULAR_MICROROBOT.md`](RESEARCH_IDEA_VASCULAR_MICROROBOT.md).
The first six-architecture training comparison on a balanced subset of the
final bank is in [`training_runs/mixed_20k/`](training_runs/mixed_20k/README.md),
with a [three-seed results table](training_runs/mixed_20k/architecture_summary_80.md).
The [validation-calibrated low-alarm analysis](training_runs/mixed_20k/fpr_budget_summary_80.md)
shows the sensitivity lost when the allowed false-alarm rate is 1%.
The [dissertation and publication research direction](DISSERTATION_PUBLICATION_DIRECTION.md)
proposes a generator-shift and hard-control benchmark grounded in the final bank;
its [MMP control feasibility audit](reports/near_analogue_controls_feasibility.json)
records the available near-analogue pairs and their limits.
The [first dedicated MMP challenge pilot](research_runs/MMP_PILOT_REPORT.md)
uses 1,000 parent-matched source controls, molecule/scaffold/component-disjoint
splits, three model seeds, and a hard-negative training ablation.
The [matched generator-transfer pilot](research_runs/GENERATOR_TRANSFER_REPORT.md)
trains on exclusive rule-based analogues and compares matched rule-based and
evolutionary exact-edit analogues across eight elemental substitutions. It
includes stricter parent matching, repeated split selections, and a third
split requiring identical local environments around the edited atom. Much
of the apparent transfer gap in the initial splits concentrates in pairs
with different local environments.
The [HPC training and tuning plan](HPC_TRAINING_PLAN.md) gives the next
experimental protocol, model search spaces, compute workflow, and reporting
criteria for a dissertation-scale comparison.

## Current snapshot

- 90 balanced datasets, 781,692 total rows; fixed splits targeting 80/10/10 are
  disjoint on the stored scaffold, which is the **parent scaffold for
  counterfeits**. An audit of each molecule's own scaffold and a stricter E
  test subset are in `benchmark/child_scaffold_overlap_audit.json` and
  `benchmark/strict_child_scaffold/`.
- 249,803 tagged 3D conformers for 249,975 unique molecules (99.9312%),
  archived in five `benchmark/conformers_part_*.sdf.gz` shards. Missing cases
  are characterized in `benchmark/conformers_missing_summary.json`.
- `benchmark/cross_version/` contains a globally split, conflict-free
  four-category M source, a 572-row E test slice unseen by source SMILES and
  stored parent scaffold, and a stricter 448-row slice unseen by each
  molecule's own scaffold as well.
- `benchmark/realism_audit.json` and `baseline_results.json` contain current
  diagnostics and fixed-split baselines. Small subcategories can retain
  property shortcuts; inspect their `summary.json` before comparing models.
- `benchmark/edit_mask_corrections.csv` pins corrected atom-level annotations
  that would otherwise require ambiguous MCS alignment to regenerate;
  `benchmark/halogen_mask_audit.json` identifies exactly verified halogen swaps.
  `benchmark/edit_mask_size_audit.json` flags large remaining masks, so
  atom-level scores should use verified subsets.
- `benchmark/verified_edits/` contains 23,826 uniquely reversible single-atom
  parent/child edits. Their masks were applied to 49,741 table rows, correcting
  40,298 prior masks; the directory includes the manifest, audits and the
  before/after agreement reports.
- `benchmark/reproduction_audit.json` records an exact rebuild comparison:
  90/90 CSV tables, summaries and the index matched this snapshot.
- `benchmark/environment.json` records the Python, library and compute versions
  used for the runs in this snapshot.
- `benchmark/prevalence_analysis.json` projects fixed-threshold precision at
  rarer counterfeit prevalences from observed test error rates.
- `benchmark/imbalance/` keeps evolved validation and test fixed while
  downsampling only training counterfeits to 1:10, 1:100 and 1:1000. Its
  baseline summary and plots report missed counterfeits, alarm precision, and
  performance under a 1% validation false-alarm budget.
- `benchmark/factor_analysis/` records evolved RF errors by molecule and edit
  characteristics; `benchmark/graph_size/` tests fixed-count small/large
  counterfeit training subsets; `benchmark/scaffold_diversity/` contrasts
  matched-size training samples from 10 versus 3,700 counterfeit scaffolds.
- `benchmark/models/` contains reusable Cl→F and evolved GIN checkpoints and
  test predictions, with loading commands and exact training settings.
- `benchmark/explanations/` compares that GIN's atom saliency with uniquely
  verified Cl→F edit sites and their immediate neighbors, and repeats the
  check on the exactly verified subset of the evolved test split.

## Setup and checks

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python audit_benchmark_integrity.py benchmark
python audit_child_scaffold_overlap.py
python audit_halogen_masks.py
python audit_verified_single_atom_edits.py
python audit_cross_version.py
```

On macOS, XGBoost also requires `brew install libomp`. To restore the ignored
uncompressed source files and central SDF from their committed archives:

```bash
zstd -d intelligent_pharma_50k_v2.pt.zst
zstd -d mmp_mining_100k/mined.mmpdb.zst
zstd -d mmp_mining_100k/fragments.fragdb.zst
for part in benchmark/conformers_part_*.sdf.gz; do gzip -dc "$part"; done > benchmark/conformers_full.sdf
python audit_conformers.py --library benchmark/conformers_full.sdf
```

## Baselines

```bash
python train_baseline.py --dataset evolved --out /tmp/evolved_baseline.json
python train_baseline.py --train-csv benchmark/cross_version/M_four_categories.csv \
  --test-csv benchmark/cross_version/evolved_unseen_from_M.csv --xgboost \
  --out /tmp/cross_version_baseline.json
python train_gnn.py --dataset 'subcategory_halogen-walk.Cl->F' --model gin
```

The last command supports `--model gcn` and `--model gat`. The recorded
three-seed comparison in `benchmark/gnn_comparison/` is exploratory; the
final bank now has a separate fixed-setting comparison in
`training_runs/mixed_20k/`. The proposal's equal hyperparameter-search budget
and evaluation across additional chemistry modules are still future work.
`--ring-size-features` adds six explicit atom
features for a first C3 ablation; its three-seed Cl→F result is in the same
comparison folder.
