# Counterfeit-Detection Benchmark — Dataset Format Specification

A family of datasets for detecting counterfeit medicines from molecular structure
with Graph Neural Networks. Each dataset pairs authentic drugs (label `0`) with
counterfeit variants (label `1`) produced by a known structural modification, so
the discriminative region — the modified atoms — is known in advance.

This document specifies the **format**. For the *methodology* (how counterfeits
are generated and mined, the evolutionary search, the anti-shortcut and split
protocols), see [`../WIKI.md`](../WIKI.md).

## Directory layout

```
benchmark/
├── index.json                     # summary of every dataset (sizes, metrics)
├── conformers_full.sdf.gz         # central 3D conformer library (all molecules, gzipped)
├── conformers_report.json         # embedding coverage
├── realism_audit.json             # generation-quality / realism / bias metrics
├── category_<name>/               # one dataset per transformation category
│   ├── dataset.csv
│   └── summary.json               # (per-dataset conformers.sdf regenerable, not tracked)
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
| `edited_atoms` | str | Space-separated **atom indices in `smiles`** that the transformation changed — the explanation ground truth (empty for authentic) |
| `scaffold` | str | Bemis–Murcko scaffold used for splitting |
| `split` | str | `train` / `val` / `test` (scaffold-disjoint) |
| `mw` | float | Molecular weight |
| `logp` | float | Crippen logP |
| `qed` | float | Quantitative Estimate of Drug-likeness |
| `heavy_atoms` | int | Heavy-atom count |
| `smiles_length` | int | Length of the SMILES string |

Atom indices in `edited_atoms` index into the molecule parsed from `smiles`
(`Chem.MolFromSmiles(smiles)`), matching the conformer atom order **before**
`AddHs` (heavy atoms first, in SMILES order).

## 3D conformers

`conformers_full.sdf.gz` holds one low-energy 3D conformer per **unique**
molecule across the whole benchmark, generated with **ETKDGv3 + MMFF94** (gzipped;
~100% coverage). Each SDF record carries a `smiles` property = the canonical
SMILES, used to join back to the tables. Regenerate with
`python generate_conformers.py`.

```python
import gzip
from rdkit import Chem
supplier = Chem.ForwardSDMolSupplier(
    gzip.open("benchmark/conformers_full.sdf.gz"), removeHs=False)
conf = {m.GetProp("smiles"): m for m in supplier if m is not None}
mol3d = conf[row["smiles"]]           # 3D coords via mol3d.GetConformer()
```

## Guarantees

- **Balanced** 1:1 authentic:counterfeit in every dataset.
- **No trivial shortcut**: authentic and counterfeit are property-matched
  (DUD-E style) on molecular weight, heavy-atom count and SMILES length. Each
  `summary.json` reports Cohen's *d* per property and the accuracy of a trivial
  logistic-regression baseline on those three features — target ≈ 0.5.
- **Scaffold split**: whole Bemis–Murcko scaffold families are assigned to a
  single split, so a model is tested on cores it never saw, and a counterfeit
  never shares a split boundary with its parent family.

## The three "versions" (proposal C1)

| Version | Where | How counterfeits are made |
|---|---|---|
| **R** | (the original 50k dataset) | 21 hand-written medicinal-chemistry rules |
| **M** | `category_*`, `subcategory_*` | ~131 rules **mined from real drugs** (MMP), split per category/subcategory |
| **E** | `evolved/` | **evolutionary search** (genetic algorithm + NSGA-II), no fixed rule list |

Training on R/M and testing on E measures whether a detector generalizes from
enumerated modifications to discovered ones.

## Loading example

```python
import pandas as pd
df = pd.read_csv("benchmark/category_scaffold-hop/dataset.csv")
train = df[df.split == "train"]
test  = df[df.split == "test"]
# labels: df.label ; ground-truth edited atoms: df.edited_atoms
```

## Regenerating

```bash
python mine_mmp_rules.py --from-pt <authentic>.pt   # mine rules  (Version M source)
python build_version_m.py                            # per-category counterfeits
python build_version_e.py --no-adversarial           # evolutionary counterfeits (E)
python assemble_benchmark.py --conformers            # pair + match + split + 3D
python generate_conformers.py                        # central conformer library
```
