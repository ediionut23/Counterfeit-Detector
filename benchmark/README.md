# Counterfeit-Detection Benchmark — Dataset Format Specification

A family of datasets for detecting counterfeit medicines from molecular structure
with Graph Neural Networks. Each dataset pairs authentic drugs (label `0`) with
counterfeit variants (label `1`) produced by structural modifications. The
`edited_atoms` field records the inferred modification region, but only a
subset of those atom indices can be verified unambiguously from each
parent/child pair; see the annotation quality notes below.

This document specifies the **format**. For the *methodology* (how counterfeits
are generated and mined, the evolutionary search, the anti-shortcut and split
protocols), see [`../WIKI.md`](../WIKI.md).

## Directory layout

```
benchmark/
├── index.json                     # summary of every dataset (sizes, metrics)
├── conformers_part_000..004.sdf.gz # 249,803 archived 3D conformers
├── conformers_report.json         # coverage and integrity report
├── conformers_missing.smi         # 172 molecules without an embedded conformer
├── conformers_missing_summary.json # size and fragment profile of failures
├── edit_mask_corrections.csv       # versioned parent/child mask corrections
├── halogen_mask_audit.json         # exact reverse-swap mask verification
├── edit_mask_size_audit.json       # per-dataset mask-size warning indicators
├── child_scaffold_overlap_audit.json # row-SMILES scaffold overlap by dataset
├── reproduction_audit.json         # exact 90-dataset rebuild comparison
├── prevalence_analysis.json        # fixed-error-rate precision projections
├── realism_audit.json             # generation-quality / realism / bias metrics
├── environment.json               # software and compute versions for this snapshot
├── cross_version/                # globally split M source and strict E test slice
├── gnn_comparison/              # exploratory Cl->F model runs and summary
├── imbalance/                   # evolved train-only 1:10/100/1000 variants
├── graph_size/                  # small/large counterfeit training stress tests
├── scaffold_diversity/          # matched-size 10-vs-3700-scaffold train test
├── strict_child_scaffold/       # balanced E test with unseen child scaffolds
├── factor_analysis/             # evolved RF test scores and error strata
├── models/                      # reusable Cl->F GIN checkpoint and predictions
├── explanations/                # atom saliency versus verified Cl->F edits
├── pair_ordering/               # E parent-child score ordering across models
├── verified_edits/              # exact one-atom edit sites and audit
├── category_<name>/               # one dataset per transformation category
│   ├── dataset.csv
│   ├── summary.json
│   └── conformers.sdf             # 150 current CSV rows with 3D coordinates
├── subcategory_<cat>.<sub>/       # one dataset per well-populated subcategory
│   └── ...
├── evolved/                       # Version E — evolutionary-search counterfeits
│   └── ...
└── pooled/                        # all categories combined
    └── ...
```

## `dataset.csv` columns

| Column | Type | Meaning |
|---|---|---|
| `id` | str | Unique row id within the dataset |
| `smiles` | str | Canonical SMILES (RDKit) of the molecule |
| `label` | int | `0` = authentic, `1` = counterfeit |
| `category` | str | Transformation category, or `authentic` |
| `subcategory` | str | Finer chemical subcategory, or `authentic` |
| `parent_smiles` | str | For a counterfeit: the genuine drug it was derived from (empty for authentic) |
| `edited_atoms` | str | Space-separated **atom indices in `smiles`** marking the edit and its boundary (empty for authentic); exactness varies by transformation |
| `scaffold` | str | Bemis–Murcko scaffold of `parent_smiles` for counterfeits, or of `smiles` for authentic rows; used for splitting |
| `split` | str | `train` / `val` / `test` (disjoint on stored parent scaffold for counterfeits) |
| `mw` | float | Molecular weight |
| `logp` | float | Crippen logP |
| `qed` | float | Quantitative Estimate of Drug-likeness |
| `heavy_atoms` | int | Heavy-atom count |
| `smiles_length` | int | Length of the SMILES string |

Atom indices in `edited_atoms` index into the molecule parsed from `smiles`
(`Chem.MolFromSmiles(smiles)`), including explicit isotopic hydrogens such as
`[2H]`. `AddHs` preserves those parsed atom indices and appends newly generated
hydrogens afterward.

## 3D conformers

The five `conformers_part_*.sdf.gz` shards contain **249,803** distinct tagged
conformers for **249,975** unique SMILES in the 90 datasets (99.9312% coverage).
All records have 3D coordinates and a `smiles` property; the audit found zero
invalid, duplicate or extraneous records. The 172 failures are listed in
`conformers_missing.smi`. Embedding uses ETKDGv3 and attempts MMFF94 relaxation. The uncompressed
`conformers_full.sdf` is ignored by Git and can be restored from all five shards.
The missing molecules average 28.51 heavy atoms and 421.51 Da, compared with
26.18 and 372.79 for embedded molecules; 39/172 missing SMILES have multiple
disconnected components. Geometry experiments should report this coverage.

```python
import gzip
from glob import glob
from rdkit import Chem
conf = {}
for path in sorted(glob("benchmark/conformers_part_*.sdf.gz")):
    with gzip.open(path, "rb") as stream:
        supplier = Chem.ForwardSDMolSupplier(stream, removeHs=False)
        conf.update({m.GetProp("smiles"): m for m in supplier if m is not None})
mol3d = conf[row["smiles"]]           # 3D coords via mol3d.GetConformer()
```

For smaller jobs, stream one shard at a time rather than holding the full
library in memory.

## Realism audit

`realism_audit.json` covers all 90 datasets. Its `novelty` compares generated
counterfeits with the full 58,616-molecule authentic reference library, rather
than just the matched negatives in each CSV. All counterfeit rows are valid and
unique within their datasets. Novelty is 0.983 for `evolved` and 0.976 for
`pooled`; small subcategories vary, including 0.738 for `Br->Cl` (80
counterfeits). `prop_shortcut_acc` and `fp_separability_auc` are diagnostic
scores from a random 70/30 split of at most 800 examples per class. They are
not the official parent-scaffold-split baselines in `baseline_results.json`.

`prevalence_analysis.json` uses the observed test-set TPR and FPR at threshold
0.5 to project precision at counterfeit:authentic prevalences of 1:10, 1:100
and 1:1000. It assumes those class-conditional error rates stay fixed and does
not retrain models or create physically imbalanced test sets. Because the saved
baseline metrics were rounded, the report gives the range across every
compatible integer confusion matrix.

`imbalance/` isolates scarcity during training: only evolved training
counterfeits are reduced to 1:10, 1:100, and 1:1000 against the unchanged
authentic class. Validation and test stay balanced with the original
parent-scaffold split.
`baseline_summary.json` compares fixed-threshold and balanced-validation-F1
threshold results for RF and XGBoost (seed 42), plus thresholds selected for
at most 1% false alarms on validation. `imbalance_effect.png` plots AUC,
false-negative rate, and precision; `fpr_budget_effect.png` shows the
false-alarm-budget results. At 1:1000 the RF F1-tuned
threshold flags every test row, so its zero FNR has only 0.500 precision on
this balanced test. See Section 6 of `WIKI.md` for the full interpretation.

`factor_analysis/` contains the fixed evolved RF's test probabilities and an
observational stratification of missed counterfeits by molecule size, edit
fraction, parent similarity, and scaffold-core overlap. Most E rows in that
analysis do not have an exact verified edit site, so mask-based strata remain
exploratory. `graph_size/` contains
fixed-count training subsets that select small or large counterfeits while
leaving authentic training rows and both evaluation splits intact. Selection
also changes chemical composition, so it is a stress test rather than a pure
causal size intervention.

`scaffold_diversity/` fixes the 3,700 authentic train rows and the validation
and test rows, and matches three heavy-atom size-bin counts in the 3,700
counterfeit train rows. Selecting from 10 versus 3,700 scaffolds increases RF
test AUC from 0.645 to 0.740 and XGBoost AUC from 0.638 to 0.738 over three
seeds. Threshold behavior and chemical-composition caveats are documented in
that directory's README and Section 6 of `WIKI.md`.

`gnn_comparison/Cl_to_F_summary.json` includes a three-seed GIN ablation with
explicit atom ring-size features (`gin_ring`) on the Cl→F subcategory. It is a
single-subcategory exploratory comparison, not a tuned model ranking.
`gnn_comparison/Ring_resize_summary.json` repeats the same ablation on a
ring-resize subcategory (160 test molecules), where the added features improved
GIN AUC and FNR in all three paired seeds; it also includes RF/XGBoost.
`gnn_comparison/Ring_hetero_swap_summary.json` uses a larger ring-heteroatom
swap split and shows essentially unchanged GIN AUC with those features.

## Guarantees

- **Balanced** 1:1 authentic:counterfeit in every dataset and in each
  train/val/test split of the regenerated snapshot. `split_label_counts` in
  `index.json` records the counts.
- **Property matching and diagnostics**: authentic and counterfeit are matched
  (DUD-E style) on molecular weight, heavy-atom count and SMILES length. Each
  `summary.json` reports Cohen's *d* per property and the accuracy of a trivial
  logistic-regression baseline on those three features. The target is ≈ 0.5,
  but some small subcategories retain measurable shortcuts (up to 0.783 in this
  snapshot); inspect that metric before interpreting a model score.
- **Parent-scaffold split**: the stored Bemis–Murcko scaffold groups are
  assigned to one split while both class quotas are balanced. A counterfeit
  stays with its parent scaffold family. Its own molecular scaffold can occur
  in another split; `child_scaffold_overlap_audit.json` measures that overlap
  and `strict_child_scaffold/` gives a stricter evolved test subset.

These guarantees apply **within each dataset**. The category datasets and
`evolved/` were split independently. Concatenating four M categories directly
would introduce 2,936 opposite-label SMILES and train/val scaffold overlap.
`cross_version/M_four_categories.csv` resolves this: it removes those conflicting
molecules, deduplicates, balances the classes and assigns one global scaffold
split (97,470 rows, exactly balanced within each split).

For an M→E experiment, train and tune on this derived M table, then score
`cross_version/evolved_unseen_from_M.csv`. It contains only E test rows whose
SMILES and scaffold occur in none of the four original M category train/val
sets. Of 9,250 E test rows, 628 pass this filter (286 counterfeit, 342
authentic); class balancing retains 572 (286 per class). Recreate both tables
with `python build_cross_version_m.py` and
`python cross_version_holdout.py --train category_bioisostere category_halogen-walk category_homologation category_scaffold-hop --test evolved --out benchmark/cross_version/evolved_unseen_from_M.csv --balance`.
`python audit_cross_version.py` verifies both derived files against their
sources. `cross_version/baseline_summary.json` aggregates three fixed-seed
fingerprint runs on this strict M→E protocol: Random Forest ROC-AUC
0.866 ± 0.002, F1 0.756 ± 0.006, FNR 0.287 ± 0.004; XGBoost ROC-AUC
0.810 ± 0.005, F1 0.680 ± 0.008, FNR 0.395 ± 0.010 (572 E rows).

The CSV snapshot was regenerated after the assembler began excluding generated
counterfeits from the authentic matching pool. All 90 datasets (781,692 rows)
pass `python audit_benchmark_integrity.py benchmark`: balanced labels, matching
index counts, and no SMILES or scaffold shared across splits within a dataset.
An independent rebuild in a temporary directory passed
`verify_reproduction.py`: all 90 CSV tables, summaries and `index.json`
matched exactly after applying `edit_mask_corrections.csv`, the verified
single-atom manifest and the final split assignment. The result
is in `reproduction_audit.json`. The raw MCS step produced 29 source-mask
variants on this rebuild; the versioned correction manifests normalized them.
Two counterfeit rows have an empty `edited_atoms` field because their edit
removed a separate salt component and changed no atom in the retained child;
exclude those rows from atom-level explanation scoring.
Some source edit masks were missing or used invalid indexing and were
reconstructed from parent/child maximum-common-substructure alignment. Symmetric
molecules can admit multiple valid alignments. For direct halogen substitutions,
`repair_halogen_substitution_masks.py` accepts a mask only when reversing one
specific child halogen exactly reconstructs the parent; `halogen_mask_audit.json`
records 6,160 verified table rows (2,707 unique pairs) and 3,251 unresolved
rows. The selected corrections are pinned in `edit_mask_corrections.csv`
(5,577 parent/child pairs, 9,615 rows across the datasets).
`edit_mask_corrections.py apply` reproduces them without repeating MCS searches.
`edit_mask_size_audit.json` shows that 18,712/52,359 pooled counterfeit masks
cover more than half the heavy atoms. This is a warning for local attribution
studies, not proof that every large mask is wrong; use chemically verified
subsets for quantitative explanation scores.

`verified_edits/manifest.csv` identifies 23,826 unique parent/child pairs
(49,741 table rows) whose substitution or addition can be reversed at exactly
one child atom to reconstruct the parent. Applying these masks corrected
40,298 rows that previously disagreed; the earlier counts are retained in
`verified_edits/precorrection_summary.json`. The remaining masks have no such
proof and should be treated as approximate. The manifest, method limits and
audit procedure are described in `verified_edits/README.md`.

`explanations/` contains a first atom-saliency check on the saved Cl→F GIN
checkpoint. It scores the 220 test counterfeits with a unique exact F→Cl
reversal and reports changed-atom and edit-region localization separately.
The evolved GIN check in `explanations/evolved_gin_seed42/` scores 292 verified
test counterfeits: the saliency often reaches the edit region, while exact-site
top-1 accuracy remains below a uniform-atom baseline.
`pair_ordering/` compares how often the saved E GIN and fixed fingerprint
baselines score an actual counterfeit child higher than its authentic parent,
with per-pair predictions and summaries across seeds 42-44. This is a
supplementary sensitivity measure for molecule-only classifiers, not an atom-
level explanation.

## The three "versions" (proposal C1)

| Version | Where | How counterfeits are made |
|---|---|---|
| **R** | (the original 50k dataset) | 21 hand-written medicinal-chemistry rules |
| **M** | `category_*`, `subcategory_*` | 3,257 rules **mined from real drugs** (MMP), split per category/subcategory |
| **E** | `evolved/` | **evolutionary search** (genetic algorithm + NSGA-II), no fixed rule list |

Training on R/M and testing on E measures whether a detector generalizes from
enumerated modifications to discovered ones.

## Loading example

```python
import pandas as pd
df = pd.read_csv("benchmark/category_scaffold-hop/dataset.csv")
train = df[df.split == "train"]
test  = df[df.split == "test"]
# labels: df.label ; edit masks: df.edited_atoms (check verification status)
```

## Regenerating

```bash
python mine_mmp_rules.py --from-pt <authentic>.pt   # mine rules  (Version M source)
python build_version_m.py                            # per-category counterfeits
python build_version_e.py --no-adversarial           # evolutionary counterfeits (E)
python assemble_36_benchmark.py --out benchmark_regen
python integrate_evolved_categories.py --out benchmark_regen
python edit_mask_corrections.py apply --root benchmark_regen
python apply_verified_single_atom_edits.py --root benchmark_regen
python audit_halogen_masks.py --root benchmark_regen --out benchmark_regen/halogen_mask_audit.json
python rebalance_scaffold_splits.py benchmark_regen --apply
python audit_benchmark_integrity.py benchmark_regen
python verify_reproduction.py --expected benchmark --actual benchmark_regen
python generate_conformers.py --workers 8             # central conformer library
python audit_conformers.py --library benchmark/conformers_full.sdf --missing-out benchmark/conformers_missing.smi
python refresh_sample_conformers.py --library benchmark/conformers_full.sdf
python archive_conformers.py benchmark/conformers_full.sdf --out benchmark
python build_cross_version_m.py
python cross_version_holdout.py --train category_bioisostere category_halogen-walk category_homologation category_scaffold-hop --test evolved --out benchmark/cross_version/evolved_unseen_from_M.csv --balance
python audit_cross_version.py
python build_imbalance_variants.py
python audit_imbalance_variants.py
python train_baseline.py --train-csv benchmark/evolved/dataset.csv --train-only --tune-threshold --target-fpr 0.01 --xgboost --name evolved_train_1_to_1 --out benchmark/imbalance/baseline_1_to_1.json
for ratio in 10 100 1000; do
  python train_baseline.py --train-csv "benchmark/imbalance/evolved_train_1_to_${ratio}.csv" --train-only --tune-threshold --target-fpr 0.01 --xgboost --name "evolved_train_1_to_${ratio}" --out "benchmark/imbalance/baseline_1_to_${ratio}.json"
done
python summarize_imbalance_baselines.py
python plot_imbalance_results.py
python plot_fpr_budget_results.py
```
