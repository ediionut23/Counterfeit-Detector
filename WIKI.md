# Benchmark Wiki — How the Counterfeit Dataset Is Built

This is the long-form, read-me-to-understand-it companion to the code. It
explains **what each component does, why, and the algorithms behind it** — with
particular attention to the evolutionary search, which is the least standard
piece. For the dataset *format*, see [`benchmark/README.md`](benchmark/README.md).
For the enlarged, modular counterfeit bank used to construct later training
sets, see [`final_dataset/README.md`](final_dataset/README.md). The bank keeps
one row per unique synthetic analogue and separate parent/edit relations; the
90 balanced CSVs below are earlier evaluation snapshots and their row counts
must not be interpreted as unique counterfeits.

> **Snapshot status (2026-09-26).** The current benchmark has 90 balanced CSV
> datasets (781,692 rows), generated from a 3,257-rule MMP catalogue. The
> historical baseline table in Section 6 was measured on earlier, smaller
> snapshots; the new table there uses the current CSVs. The archived
> central 3D library covers 249,803 of 249,975 molecules; details are in
> [`benchmark/README.md`](benchmark/README.md).

> **The one-paragraph summary.** A *counterfeit* is a real drug with a small,
> chemistry-aware edit. To build a dataset of them we (1) **learn** what edits
> real drugs undergo, by mining Matched Molecular Pairs from a large set of
> authentic molecules; (2) **apply** those edits, and also **discover new ones**
> with an evolutionary algorithm that searches molecule-space under several
> competing objectives; (3) **assemble** each set of counterfeits into a proper
> dataset — paired with authentic negatives, matched to reduce shortcuts,
> split by scaffold, and paired with a near-complete 3D conformer library and the ground-truth of
> which atoms were changed.

---

## 0. The pipeline at a glance

```
 authentic drugs (ChEMBL/PubChem)
        │
        ▼
 ┌───────────────────────┐   Phase A — LEARN the edits
 │ mine_mmp_rules.py     │   Matched Molecular Pairs (mmpdb)
 │  → mined_transformations.py (3257 rules, categorized, with support)
 └───────────────────────┘
        │                         │
        ▼                         ▼
 ┌───────────────────┐     ┌──────────────────────────┐   Phase B — MAKE counterfeits
 │ build_version_m.py│     │ evolutionary_generator.py│   GA + NSGA-II
 │  apply rules      │     │  discover/combine edits  │
 │  → version_m/     │     └──────────────────────────┘
 └───────────────────┘         │  used by build_version_e.py → version_e/
        │                        │
        └───────────┬────────────┘
                    ▼
        ┌───────────────────────────────┐   Phase C — ASSEMBLE datasets
        │ assemble_36_benchmark.py      │   pair + match + scaffold split
        │  → benchmark/                 │
        └───────────────────────────────┘
                    │  integrate_evolved_categories.py adds 6 evolved slices
                    │
                    ▼
        ┌────────────────────────┐
        │ generate_conformers.py │   central 3D library (ETKDGv3+MMFF)
        └────────────────────────┘

  evaluate_generalization.py — measures a detector's detection rate across R/M/E
```

---

## 1. Chemistry background you need (30 seconds)

- **SMILES** — a molecule written as a string, e.g. aspirin is
  `CC(=O)Oc1ccccc1C(=O)O`. RDKit parses these into atom-and-bond graphs.
- **Canonical SMILES** — the one, unique SMILES for a molecule, so we can
  deduplicate reliably.
- **Morgan fingerprint** — a bit-vector summarizing which circular atom
  neighborhoods a molecule contains. Two molecules' **Tanimoto similarity** is
  the overlap of their fingerprints (0 = nothing in common, 1 = identical).
- **Bemis–Murcko scaffold** — the ring-system "core" of a molecule with the
  side chains stripped. Molecules sharing a scaffold are structurally related.
- **QED** — Quantitative Estimate of Drug-likeness, in [0,1]; higher = more
  drug-like. **SA score** — synthetic accessibility (Ertl), ~1 (easy) to ~10
  (hard to make). Together they say "is this a plausible compound".

---

## 2. Phase A — Learning the edits (Matched Molecular Pairs)

**File:** [`mine_mmp_rules.py`](mine_mmp_rules.py) → produces
[`mined_transformations.py`](mined_transformations.py).

### The idea

Instead of hand-writing "swap a chlorine for a fluorine", we *learn* the edits
that occur among real drugs. A **Matched Molecular Pair (MMP)** is two molecules
that are identical except in one small region — e.g. the same drug once with
`-Cl` and once with `-F`. The *difference* between them is a transformation rule.

### The algorithm (Hussain–Rea, via `mmpdb`)

1. **Fragmentation.** For every molecule, cut one single, non-ring bond. This
   splits it into a **variable** part (a small R-group) and a **constant** part
   (the rest). The cut point is marked with an attachment atom `[*:1]`.
2. **Indexing.** Group all molecules by their *constant* part. Within a group,
   every pair of different variable parts is a matched pair, giving a rule
   `V1 → V2` (e.g. `[*:1]Cl → [*:1]F`). The number of molecule pairs exhibiting
   a rule is its **support** — a data-grounded measure of how common/realistic
   that edit is.
3. **Representation.** Because the fragments carry `[*:1]`, the string
   `V1 >> V2` is a **reaction SMARTS** — a pattern RDKit can *apply* to any new
   molecule that contains `V1`, rewriting it to `V2`. That is exactly what makes
   a mined rule reusable as a generator/mutation operator later.

### What our wrapper adds on top of mmpdb

- **Filtering** (`extract_rules`): keep single-cut rules, bound the fragment
  size, and drop **noise** (`is_clean_fragment`) — isotopes like `[18F]` and
  protonation states like `[O-]` are recording artifacts, not edits a
  counterfeiter makes.
- **Auto-categorization** (`categorize_rule`, `_subcategorize`): a rule is
  labeled by comparing the two fragments' profiles (ring count/size/aromaticity,
  in-ring heteroatoms, halogens, saturation, chain length):
  - **scaffold-hop** — the ring system changes (e.g. benzene→thiophene, or
    cyclohexyl→phenyl). Subcategories: `aromatize`, `add-ring`, `ring-resize`,
    `ring-heteroatom-swap`.
  - **halogen-walk** — only halogens move (F↔Cl, add-F, remove-Cl).
  - **homologation** — a saturated carbon chain grows/shrinks, no new
    heteroatom or unsaturation (methyl→ethyl, N-demethylation). Subcategories:
    `grow/shrink` × `single/short/long`.
  - **bioisostere** — heteroatom composition or a functional group changes
    (acylation, phosphorylation, ester↔amide). Subcategories by the heteroatom
    introduced (`add-O`, `add-N`, …) or `unsaturation-change`.
- **Export** in the exact 6-tuple format of the hand-written catalogue plus two
  side dictionaries `SUPPORT` and `SUBCATEGORY`, so downstream code consumes it
  with no changes.

> **Practical note — an RDKit 2026.03 bug.** Building environment fingerprints at
> radius > 0 crashes on some fragments (`neither end atom traversed`). We index
> with `--max-radius 0`, which still yields every rule and its support.

**Result:** the current catalogue was mined from 58,616 authentic molecules
fetched from public databases (ChEMBL + PubChem, see
`fetch_public_molecules.py`). It holds **3,257 categorized rules** with support
counts, rather than the original 21 hand-written rules. Earlier passes held
131 and then 1,216 rules; the current counts and category breakdown are in
`mined_transformations.json`.

---

## 3. Phase B — Making counterfeits

There are two generators. The rule-based one is simple; the evolutionary one is
the heart of the project, so it gets most of this section.

### 3.1 Rule-based (Version M)

**File:** [`build_version_m.py`](build_version_m.py). For each authentic parent,
pick a mined rule *of a given category* (weighted by support), apply it with
RDKit's reaction engine, and keep the product if it is valid, drug-like, novel,
and within a Tanimoto band of the parent (`0.4–0.98`). Emitted **per category**
so a detector can be scored per modification type. Each counterfeit records its
parent, rule, category, similarity, and **edited atoms**.

### 3.2 Evolutionary (Version E) — the genetic algorithm

**File:** [`evolutionary_generator.py`](evolutionary_generator.py), scaled by
[`build_version_e.py`](build_version_e.py).

#### Why evolutionary at all?

A fixed rule list can only produce edits someone wrote down, and only **one at a
time**. A real counterfeiter is not restricted like that: they can *stack*
several edits and land on molecules nobody enumerated. An **evolutionary
algorithm** explores that larger space by *breeding* molecules: it keeps a
population of candidate counterfeits, recombines and mutates them, and lets a
fitness function decide which survive — so useful, non-obvious edits emerge and
poor ones die out.

#### Genetic-algorithm vocabulary, mapped to molecules

| GA concept | Here |
|---|---|
| **Individual / genome** | one candidate molecule (a canonical SMILES) |
| **Population** | a set of candidate counterfeits of one parent drug |
| **Generation** | one round of selection + variation |
| **Fitness** | a *vector* of objective scores (see 3.3) |
| **Selection** | keep the best; here via NSGA-II (3.4) |
| **Crossover** | recombine two molecules |
| **Mutation** | apply a small edit to one molecule |

#### Initialization

The starting population is the parent drug with 1–3 random mined edits already
applied (`MolSampling`), so the search begins near the parent rather than from
nothing.

#### Mutation — reuse the mined rules

`RuleLibrary.mutate` picks one mined rule, **weighted by its support**, and
applies it. Because rules are chained across generations, one individual can
accumulate several edits — this is how the search **combines** modifications.

#### Crossover — Graph GA (Jensen, 2019)

Molecules are graphs, not fixed-length vectors, so ordinary "cut the string at
position k" crossover does not work. The graph-based recipe (`crossover`):

1. Cut each parent at a random single, non-ring bond → two fragments, each
   carrying an attachment dummy atom (`FragmentOnBonds`).
2. Take one fragment from parent A and one from parent B.
3. **Zip** them together at their dummy atoms with `Chem.molzip`, forming a new
   valid molecule.

So a child inherits a substructure from each parent — the molecular analogue of
recombination. If a particular cut yields an invalid molecule, the operator
falls back to a parent, keeping the search robust.

#### 3.3 The fitness — several objectives, deliberately in conflict

A counterfeit is "good" only if it is *simultaneously*:

| Objective (all **minimized**) | Definition | Why |
|---|---|---|
| `band_penalty` | distance of Tanimoto(parent, child) outside `[0.4, 0.9]` | too similar = trivial; too different = not a counterfeit *of that drug* |
| `1 − QED` | drug-likeness | must look like a real compound |
| `SA / 10` | synthetic accessibility | must be *makeable* |
| `P(counterfeit)` *(optional)* | a trained detector's score | low = it *fools* the detector (adversarial) |

These pull against each other (an adversarially-hard molecule may be less
drug-like, etc.). There is no single "best" — only trade-offs. This is why we use
**multi-objective optimization** rather than collapsing everything into one
weighted sum (a weighting would bake in an arbitrary preference).

#### 3.4 NSGA-II — selecting on multiple objectives

NSGA-II (Deb et al., 2002) is the standard multi-objective genetic algorithm.
Two ideas make it work:

**(a) Pareto dominance.** Solution *A* **dominates** *B* if *A* is at least as
good on every objective and strictly better on at least one. The solutions that
nobody dominates form the **Pareto front** — the set of best possible
trade-offs. There isn't one winner; there's a frontier.

**(b) Non-dominated sorting + crowding distance.** Each generation:
1. Sort the population into "fronts": front 1 = non-dominated solutions; front 2
   = those dominated only by front 1; and so on.
2. Fill the next generation front by front. When a front doesn't fit entirely,
   break ties by **crowding distance** — prefer solutions in *sparse* regions of
   objective space, which keeps the population diverse and spread along the
   front instead of clumping.

The result of a run is the **entire Pareto front** of counterfeits — a spectrum
from "very similar, very drug-like" to "more aggressive, more adversarial". That
front *is* the answer to "which candidates are the most promising to keep."

We drive NSGA-II with `pymoo`, using custom operators (`MolSampling`,
`MolCrossover`, `MolMutation`) because individuals are molecules, not numeric
vectors; `pymoo` supports this via object-typed populations.

#### 3.5 The adversarial objective (optional, model-in-the-loop)

`load_detector_scorer` loads the project's trained HGT detector and exposes
`smiles → P(counterfeit)`. Adding it as a 4th objective makes NSGA-II seek
counterfeits the *current model* misclassifies as authentic — a way to surface
its blind spots. **For the benchmark dataset this is turned off** (`--no-adversarial`),
because a benchmark must be model-independent and reproducible; the adversarial
mode is a research tool, not part of the released data.

> A subtlety we handle: the saved checkpoints' architecture had drifted from the
> code defaults, so the loader **infers** `hidden_channels` and the node-type set
> from the checkpoint's weights and picks the highest-F1 one, then loads strictly.

#### 3.6 Which atoms changed

`edited_atoms` indexes the stored child SMILES. The generator uses reaction
provenance where it survives canonicalization and falls back to a
parent/child **Maximum Common Substructure** (`rdFMCS`) alignment. The mask
includes the changed child atom and often its bonded boundary. MCS can choose
the wrong correspondence in symmetric molecules, so these masks are not all
equally reliable as explanation ground truth. For direct halogen substitutions,
`repair_halogen_substitution_masks.py` proves a unique edited site by changing
one child halogen back and checking whether it reconstructs the parent. That
verified subset supports a stricter explanation evaluation. The wider
`build_verified_single_atom_edits.py` pass accepts uniquely reversible
single-atom substitutions and additions across all datasets; its separate
manifest gives the exact changed atom and a local mask.

---

## 4. Phase C — Assembling real datasets

**Current files:** [`assemble_36_benchmark.py`](assemble_36_benchmark.py) and
[`integrate_evolved_categories.py`](integrate_evolved_categories.py).
[`assemble_benchmark.py`](assemble_benchmark.py) is the earlier, smaller
assembler. Counterfeit lists are not datasets; these scripts turn each slice
(a category, a subcategory, E, or the pool) into one.

### 4.1 Authentic negatives + property matching (DUD-E)

We pair counterfeits with authentic drugs, then **property-match** them
(`property_match`): bin both classes on molecular weight, heavy-atom count and
SMILES length (quantile bins), and keep an equal number from each class in every
bin. The two classes then share the same marginal distributions on those
properties, so **no classifier can cheat** by reading molecular weight off the
label. This mirrors DUD-E (Mysinger et al., 2012).

### 4.2 Shortcut diagnostics

`shortcut_metrics` reports **Cohen's *d*** (standardized mean difference; ~0 =
distributions overlap) for each matched property, and the test accuracy of a
**trivial logistic regression** on just those three features. Target ≈ 0.5 (a
coin flip) — if a three-feature linear model can separate the classes, the
dataset has a shortcut and any GNN result on it is suspect. In the current
snapshot the pooled dataset scores 0.518, while some small subcategories reach
0.783; matching reduces but does not eliminate every shortcut.

### 4.3 Scaffold split

`scaffold_split` groups molecules by **Bemis–Murcko scaffold** (using the
*parent's* scaffold for a counterfeit) and assigns whole scaffold families to
`train`/`val`/`test` (80/10/10), minimizing the deviation from both class
quotas in each split. Because families never straddle the split, a
model is evaluated on ring systems it has **not** seen — the harder, more honest
setting established by MoleculeNet and OGB, and it prevents a counterfeit from
leaking information about a parent that sits in another split.

### 4.4 3D conformers (ETKDG)

`write_conformers` / [`generate_conformers.py`](generate_conformers.py) embed a
3D structure for each molecule with **ETKDGv3** (Riniker & Landrum, 2015) — a
distance-geometry method biased by real torsion-angle preferences — then relax
it with the **MMFF94** force field when parameters are available. The central library
`benchmark/conformers_full.sdf` embeds each successfully processed unique
molecule once (in parallel), keyed by canonical SMILES. The archived five-shard
library covers 249,803/249,975 molecules (99.9312%). The 172 failures are in
`benchmark/conformers_missing.smi`; check membership before geometry-based
experiments in C1/C3.
The failures are somewhat heavier (421.51 vs 372.79 Da on average), and
39/172 contain disconnected components; the exact breakdown is in
`benchmark/conformers_missing_summary.json`.

The `edited_atoms` field is tied to child SMILES atom indices. When source edit
metadata was missing or misindexed, `repair_edited_atoms.py` inferred a child
mask from the parent/child maximum common substructure. Symmetry can make that
mapping ambiguous; atom-level XAI comparisons should report this limitation and
exclude the two rows whose edit removed only a disconnected salt component.
An additional exact reverse-substitution pass corrected 4,010 halogen rows in
seven tables and propagated the same verified masks to 26 repeated-pair rows
in evolved-category tables, replacing many implausibly large MCS masks. The chosen corrections are
pinned in `benchmark/edit_mask_corrections.csv` for exact dataset regeneration:
5,577 parent/child pairs affect 9,615 table rows. The independent
`audit_halogen_masks.py` verifies all 6,160 exactly reversible table rows
(2,707 unique parent/child pairs) with zero mismatches. The other 3,251
halogen-substitution rows do not admit a unique one-atom reversal and retain
their existing masks; they should not be treated as exact edit sites.
`benchmark/edit_mask_size_audit.json` gives a broader warning: 18,712 of the
52,359 pooled counterfeit masks cover more than half of the molecule's heavy
atoms, as do 24,128 of 40,469 homologation masks. Size alone does not prove an
annotation wrong, but whole-molecule masks cannot support a convincing local
explanation score. Atom-attribution results below therefore use a chemically
verified subset rather than all available masks.
The full reverse-edit pass examined 196,975 unique counterfeit pairs and
verified 23,826 pairs, appearing in 49,741 table rows. Exactly 9,443 of those
rows initially matched the local verified mask. The other 40,298 masks were
corrected in the CSV snapshot. `benchmark/verified_edits/` preserves both the
manifest and the prior agreement counts; `audit_verified_single_atom_edits.py`
checks membership and mask geometry throughout and independently repeats the
chemical reversal on 300 deterministic sample pairs. Unverified rows remain
approximate and should not be pooled into exact-site explanation scores.

| Dataset | Verified counterfeit rows | Coverage |
|---|---:|---:|
| Bioisostere | 3,484 / 32,773 | 10.6% |
| Halogen walk | 6,077 / 21,947 | 27.7% |
| Homologation | 8,265 / 40,469 | 20.4% |
| Other | 695 / 13,322 | 5.2% |
| Scaffold hop | 654 / 32,166 | 2.0% |
| Evolved | 3,578 / 46,251 | 7.7% |
| Pooled | 6,345 / 52,359 | 12.1% |

The same parent/child pair may appear in several datasets, so table rows in
this summary are not distinct chemical edits. Coverage is especially limited
for scaffold hops, whose changes usually require more than one atom.
In the independent rebuild, `verify_reproduction.py` found no differences in
any of the 90 CSVs, their summaries or the index; see
`benchmark/reproduction_audit.json`.
Nine additional parent/child corrections pin MCS alignments that varied on a
fresh rebuild; the manifest is applied before the exact comparison.
In this rebuild, 29 raw source masks differed from both the manifest's recorded
source mask and its final target, illustrating further MCS alignment variation.
Applying the two manifests still produced an exact 90-dataset match.

---

## 5. The three versions and the generalization question

| Version | Counterfeits made by | Distribution shift |
|---|---|---|
| **R** | 21 hand-written rules (original 50k set) | in-distribution baseline |
| **M** | 3257 mined rules, per category/subcategory | mild — new rule *types* |
| **E** | evolutionary search, no rule list | large — discovered, combined edits |

The benchmark's central question: **does a detector trained on the edits we
enumerated (R/M) still catch edits discovered another way (E)?**
[`evaluate_generalization.py`](evaluate_generalization.py) answers a fast version
of this with detection rate (fraction scored `P>0.5`) per version and per M
category. Honest caveats are printed with the numbers: R is in-distribution (an
upper bound), E built with the adversarial objective is white-box against the
scored model (also an upper bound), so **M is the cleanest generalization
signal**.

The fixed splits are disjoint on the **stored scaffold**, which is the parent
scaffold for a counterfeit row and its own scaffold for an authentic row.
The actual child molecule's scaffold can cross splits: this occurs in 55 of
90 datasets. In E, 1,623/9,250 test molecules (17.55%) have their own
scaffold in train/validation, including 1,333/4,625 counterfeits; the pooled
rate is 16.0%. `benchmark/child_scaffold_overlap_audit.json` records every
dataset. Thus the original splits test novel **parent** scaffolds, while the
supplementary `benchmark/strict_child_scaffold/` subset tests child-scaffold
novelty as well. Scaffold assignments also differ across datasets. For a
four-category M→E experiment,
train and select on the `train`/`val` rows of
`benchmark/cross_version/M_four_categories.csv`, then score
`benchmark/cross_version/evolved_unseen_from_M.csv`. The M table removes 2,936
conflicting-label SMILES, balances the classes and reassigns scaffolds globally.
The E table's 572 class-balanced test rows have no exact SMILES or stored
parent scaffold in any of the original M category train/val splits. This is a small
held-out subset of the 9,250-row E test split; report its size with the score.
An own-scaffold check against the derived M train/val table finds overlap in
68/572 E rows (62 counterfeits and 6 authentic molecules). A separate
child-scaffold-filtered cross-version test is therefore provided below.
`build_cross_version_m.py` and `cross_version_holdout.py` construct the tables.
`audit_cross_version.py` verifies the resulting rows against the original CSVs.
On this parent-scaffold-filtered slice, the fixed 300-tree fingerprint Random Forest trained on
the combined M train+val rows reaches mean ROC-AUC **0.866 ± 0.002**, F1
**0.756 ± 0.006** and false-negative rate **0.287 ± 0.004** over three seeds
(572 E test rows). Its property-only ROC-AUC is **0.576**. A 300-tree XGBoost
fingerprint baseline reaches AUC **0.810 ± 0.005**, F1 **0.680 ± 0.008** and
FNR **0.395 ± 0.010** on the same holdout. These are
cross-version results; the individual-dataset numbers in Section 6 answer a
different, within-dataset question.

For within-E robustness, `build_strict_child_scaffold_holdout.py` removes
test molecules whose *own* scaffold occurs in E train/validation (290
authentic and 1,333 counterfeit), then balances the eligible rows to
3,292 per class. `audit_strict_child_scaffold_holdout.py` confirms no such
overlap. Applying the **same fitted-model settings** to the 9,250-row original
and 6,584-row stricter tests gives:

| Model, seeds 42–44 | Original AUC / FNR | Child-scaffold-novel AUC / FNR |
|---|---:|---:|
| GIN | 0.884 / 0.183 | 0.887 / 0.188 |
| Fingerprint RF | 0.885 / 0.143 | 0.893 / 0.139 |
| Fingerprint XGBoost | 0.835 / 0.174 | 0.835 / 0.187 |

The original ranking scores do not collapse on this stricter subset. Filtering
also changes the chemical composition of the test, so these differences do
not measure the isolated causal effect of scaffold overlap. Full seed means
and SDs are in `benchmark/strict_child_scaffold/model_summary.json`.

`build_strict_cross_version_holdout.py` removes the 68 rows whose own
molecular scaffold is in derived M train/validation, then balances the
remaining test to 224 rows per class. `audit_strict_cross_version_holdout.py`
confirms the 448 rows are source-identical and have no actual molecule-scaffold
overlap. With the same M training protocol and seeds, RF AUC is **0.866 ±
0.002**, FNR **0.292 ± 0.007**; XGBoost AUC is **0.811 ± 0.008**, FNR
**0.409 ± 0.014** (`strict_own_scaffold_baseline_summary.json`). The AUCs are
essentially unchanged; test composition also changed, so this comparison
does not quantify a causal leakage advantage.

---

## 6. Is the task learnable? Baseline results

Before any elaborate model, the first question is whether authentic and
counterfeit molecules actually differ in a learnable way — or whether, after
property matching, they are effectively indistinguishable. Three scripts answer
this, always on each dataset's **parent-scaffold-disjoint test split** (the
stored scaffold is based on the parent for counterfeits), reporting the proposal's C4 metrics
(False-Negative Rate first, then Precision / Recall / F1, plus accuracy and
ROC-AUC):

- [`train_baseline.py`](train_baseline.py) — property-only logistic regression
  (MW / heavy atoms / logP), Morgan-fingerprint Random Forest and optional
  XGBoost (the C2 lower tier).
- [`train_gnn.py`](train_gnn.py) — small GIN, GCN and GAT models (the C2
  middle tier), kept CPU-friendly.
- [`learnability_nn.py`](learnability_nn.py) — adds "distance to nearest known
  drug" features, to test a relational signal.

### Current fingerprint baselines (regenerated snapshot, one seed)

`baseline_results.json` records these results on the fixed, class-balanced,
parent-scaffold-disjoint test splits. The fingerprint model is a 300-tree Random
Forest; the property-only model is logistic regression on MW, heavy atoms and
logP. The GNN has since been run on E; the other five large slices remain for
the broader architecture study.

| Dataset | test rows | property AUC | fingerprint AUC | fingerprint F1 | FNR |
|---|---:|---:|---:|---:|---:|
| evolved | 9,250 | 0.521 | 0.902 | 0.825 | 0.150 |
| pooled | 10,472 | 0.527 | 0.866 | 0.787 | 0.201 |
| bioisostere | 6,554 | 0.555 | 0.916 | 0.817 | 0.228 |
| scaffold-hop | 6,432 | 0.584 | 0.872 | 0.809 | 0.123 |
| halogen-walk | 4,390 | 0.592 | 0.952 | 0.885 | 0.114 |
| homologation | 8,094 | 0.540 | 0.945 | 0.873 | 0.150 |

The evolved tier is learnable at its current scale: its fingerprint AUC is
0.902. The earlier claim of a large M-versus-E gap does **not** carry over to
this snapshot. Property-only AUC reaches 0.592 in halogen-walk, so residual
property shortcuts still need attention in some categories.

`benchmark/gnn_comparison/Evolved_summary.json` gives a more directly matched
three-seed comparison on E: GIN trains on 74,002 train rows, selects an epoch
by validation AUC, and tests on the 9,250 fixed test rows. RF and XGBoost use
the **same train rows** with fixed settings and leave validation untouched.
The RF AUC 0.885 here is lower than the 0.902 above because the earlier
single-seed RF baseline fits train+validation (83,252 rows).

| Model (seeds 42–44) | Test AUC, mean ± SD | FNR, mean ± SD | F1, mean ± SD |
|---|---:|---:|---:|
| GIN | 0.884 ± 0.003 | 0.183 ± 0.014 | 0.801 ± 0.001 |
| Fingerprint RF | 0.885 ± 0.001 | 0.143 ± 0.001 | 0.813 ± 0.001 |
| Fingerprint XGBoost | 0.835 ± 0.001 | 0.174 ± 0.003 | 0.763 ± 0.001 |

GIN and RF have similar ranking AUC at this scale, while RF misses fewer
counterfeits at the default decision threshold. These fixed settings do not
provide an equal hyperparameter-search budget, and the test is disjoint by
stored parent scaffold rather than every analogue's own scaffold.

The refreshed `benchmark/realism_audit.json` covers all 90 datasets and checks
novelty against 58,616 authentic molecules. Every counterfeit in these tables
parses and is unique within its dataset. Novelty is 0.983 for E and 0.976 for
pooled, but drops to 0.738 for a small 80-counterfeit Br→Cl subcategory.
Its classifier diagnostics use random 70/30 splits with up to 800 examples per
class; they are distinct from the parent-scaffold-split baselines above. Very small
subcategories can give unstable and extreme diagnostic scores.

### Preliminary architecture check on Cl→F

`benchmark/gnn_comparison/` records a fixed-protocol comparison on
`subcategory_halogen-walk.Cl->F` (4,336 train / 542 val / 542 test rows). GIN,
GCN and GAT use the same atom features, graph readout, learning rate, 40-epoch
cap and raw validation-AUC early stopping, with seeds 42–44 and two fixed CPU
threads. The fingerprint models
also use three seeds and fit train+val because their settings are fixed; the
neural models use val for selection.

| Model | Test AUC | Test F1 | FNR |
|---|---:|---:|---:|
| GIN (3 seeds, mean ± SD) | 0.976 ± 0.002 | 0.930 ± 0.015 | 0.057 ± 0.048 |
| GIN + ring-size atom bits (3 seeds) | 0.967 ± 0.006 | 0.905 ± 0.010 | 0.105 ± 0.034 |
| GCN (3 seeds, mean ± SD) | 0.974 ± 0.002 | 0.913 ± 0.020 | 0.092 ± 0.045 |
| GAT (3 seeds, mean ± SD) | 0.977 (all 3 rounded) | 0.923 ± 0.004 | 0.063 ± 0.013 |
| Random Forest (3 seeds) | 0.984 ± 0.001 | 0.945 ± 0.002 | 0.026 ± 0.004 |
| XGBoost (3 seeds) | 0.983 ± 0.001 | 0.948 ± 0.001 | 0.017 ± 0.006 |

This is one subcategory without matched hyperparameter-search budgets. It shows
the fingerprint baselines remain competitive here; it does not establish a
ranking across the broader benchmark.
Means and sample SDs here use saved per-run test metrics rounded to three
decimals; a displayed zero spread can hide smaller differences.
The seed-42 GIN checkpoint in `benchmark/models/` was independently rerun with
the fixed two-thread CPU setting; all 34 saved tensors and its metrics matched
exactly (`benchmark/models/reproducibility_audit.json`).

The ring-size row is a first C3 representation ablation: it adds six binary
atom features for membership in rings of sizes 3–8 to the standard atom vector
and keeps the same GIN architecture, training settings, split and seeds. On
Cl→F, this addition did not improve mean test AUC or FNR. Three seeds on one
subcategory do not establish that ring-size features are generally harmful;
the wider representation study and equal tuning budgets are still needed.

The same fixed protocol on the 1,600-row ring-resize subcategory gives a
different result (`benchmark/gnn_comparison/Ring_resize_summary.json`):

| Model (seeds 42–44) | Test AUC, mean ± SD | FNR, mean ± SD | F1, mean ± SD |
|---|---:|---:|---:|
| GIN | 0.968 ± 0.004 | 0.117 ± 0.019 | 0.904 ± 0.015 |
| GIN + ring-size atom bits | 0.981 ± 0.006 | 0.075 ± 0.013 | 0.927 ± 0.010 |
| Fingerprint RF | 0.978 ± 0.002 | 0.045 ± 0.014 | 0.922 ± 0.006 |
| Fingerprint XGBoost | 0.986 ± 0.001 | 0.029 ± 0.007 | 0.934 ± 0.006 |

The ring-feature GIN improves AUC and FNR in each paired seed on this task,
where ring size is directly related to the modification. Each test split has
only 160 molecules, and these are fixed-settings comparisons without equal
hyperparameter search; the result is evidence of a task-specific effect, not a
general architecture ranking.

On ring-heteroatom swap (556 test rows), the same three-seed comparison is
nearly tied in AUC: GIN 0.979 ± 0.003 versus ring-feature GIN 0.979 ± 0.002;
FNR is 0.019 versus 0.023. See `Ring_hetero_swap_summary.json` in the same
directory. This supports the narrower interpretation that explicit ring size
helped the ring-resize task, not every transformation involving a ring.
`benchmark/gnn_comparison/ring_feature_ablation.png` plots the paired
seed-level AUC and FNR changes for all three tasks; dots are individual seeds
and error bars are mean ± sample SD.

### First atom-attribution audit on Cl→F

`explain_cl_to_f_gin.py` loads the saved seed-42 GIN checkpoint and computes
absolute input-gradient × feature saliency for each atom in the Cl→F test
counterfeits. It scores only the 220/271 test counterfeits with a **unique,
exactly reversible F→Cl edit**. For those molecules, the top-ranked atom lies
in the two-atom edit region (F plus its bonded neighbor) in **62.3%** of cases,
versus **7.7%** expected from choosing one atom uniformly. At least one of the
top two atoms hits the region in **76.4%**, versus **15.1%** random expectation.
Yet the F atom itself is never ranked first or second; the neighboring carbon
typically receives the attribution. These are region-localization results for
one simple saliency method and one checkpoint, not a claim of faithful causal
explanation. `benchmark/explanations/summary.json` and its per-molecule CSV
record both site and region metrics; the 51 ambiguous or multi-edit cases are
reported separately rather than scored against uncertain ground truth.
The comparison is plotted in `benchmark/explanations/localization.png`.

`explain_verified_gin.py` repeats the same localization procedure on the saved
seed-42 evolved GIN. Only 292/4,625 counterfeit test molecules have a unique,
exactly reversible one-atom edit in the verified manifest. On that subset,
top-1 edit-region accuracy is 0.250 versus a molecule-size-adjusted random
expectation of 0.095; top-2 region recall is 0.339 versus 0.183. Exact edited
atom accuracy is only 0.024 at top 1, below random expectation 0.044. The
verified subset is selective (179 substitutions and 113 additions), and the
saliency method can place signal on neighbors. These figures measure
localization, not whether masking those atoms changes the model's decision.
See `benchmark/explanations/evolved_gin_seed42/summary.json` for the detected
subset and edit-type breakdown.

Two follow-up checks probe model sensitivity. Zeroing the top saliency atom's
input features changes the counterfeit probability by 0.575 on average,
versus 0.459 for a uniformly chosen atom on the same molecules (paired mean
difference 0.115; bootstrap 95% interval 0.084–0.146). Zero vectors are not
valid atom features, and even a random atom has a large effect, so this is a
limited feature-masking fidelity check rather than chemical evidence; see
`benchmark/explanations/evolved_gin_seed42/fidelity_summary.json`.

A chemically valid paired check scores the actual parent and child molecules.
Across all 4,625 E test counterfeits, the seed-42 GIN gives the child a higher
counterfeit score than its parent in 88.1% of pairs. It assigns both sides the
intended classes at threshold 0.5 in only 49.6%. On the 292 exactly verified
one-atom edits, these values are 85.6% and 34.3%. Twenty-two test parents
appear as *counterfeit molecules* in E train or validation, reflecting the
relative meaning of the label; excluding those pairs barely changes the
ordering result. The pairwise audit is in
`benchmark/explanations/evolved_gin_seed42/all_pairs/parent_child_summary.json`.
Pair ordering is a useful supplementary benchmark metric but does not reveal
which internal features caused a prediction.

`benchmark/pair_ordering/summary.json` extends the same paired check to the
saved GIN and the fixed train-only fingerprint RF/XGBoost models at seeds
42–44. The fingerprint runs first reproduce their saved 9,250-row test metrics
exactly. Each seed scores the same 4,625 E test parent/child pairs; sample SD
therefore describes model-seed variation rather than uncertainty from new
test cohorts.

| Model | Child score higher, all 4,625 pairs | Both classes correct at 0.5 | Child score higher, 292 verified one-atom pairs |
|---|---:|---:|---:|
| GIN | 0.883 ± 0.002 | 0.509 ± 0.011 | 0.865 ± 0.009 |
| Fingerprint RF | 0.883 ± 0.003 | 0.513 ± 0.003 | 0.871 ± 0.004 |
| Fingerprint XGBoost | 0.834 ± 0.005 | 0.367 ± 0.004 | 0.835 ± 0.014 |

RF and GIN have essentially the same pair ordering under these fixed
settings. Their lower paired-class accuracy shows that ranking child above
parent is easier than putting both on the intended sides of the default
threshold. The verified subset is chemically selective and should not be
read as performance on all edit types.

### A first C4 prevalence projection

`prevalence_analysis.py` combines each recorded model's fixed-threshold test
TPR/FPR with assumed counterfeit prevalences of 1:10, 1:100 and 1:1000. For
the evolved Random Forest, precision is 0.802 on the balanced test set and
projects to about **0.288**, **0.039** and **0.004**, respectively. This is a
Bayes-rate calculation, not a model retrained on rare counterfeits or a new
imbalanced test set. It assumes the error rates remain stable as prevalence
changes; the complete ranges compatible with rounded saved metrics are in
`benchmark/prevalence_analysis.json`. The controlled class-imbalance datasets
below address a different part of C4: scarce counterfeits during training.

### C4: scarce counterfeits during training

`build_imbalance_variants.py` retains all 37,001 authentic evolved training
rows, downsamples only the counterfeit training rows, and leaves the original
balanced validation and parent-scaffold-split test rows unchanged (4,625 per class
in each). Counterfeit training subsets are nested; at 1:10, 1:100, and 1:1000,
they contain 3,700, 370, and 37 molecules. `audit_imbalance_variants.py`
verifies row identity and the fixed parent-scaffold splits. The fixed 300-tree
fingerprint RF and XGBoost settings use seed 42, train rows only, and two
decision rules: probability threshold 0.5, or a threshold maximizing F1 on the
unchanged balanced validation set. Results are in
`benchmark/imbalance/baseline_summary.json`; the three-panel figure is
`benchmark/imbalance/imbalance_effect.png`.

| Training counterfeit:authentic | RF AUC | RF FNR @ 0.5 | RF FNR / precision @ val threshold | XGB AUC | XGB FNR @ 0.5 | XGB FNR / precision @ val threshold |
|---|---:|---:|---:|---:|---:|---:|
| 1:1 | 0.884 | 0.142 | 0.116 / 0.751 | 0.834 | 0.175 | 0.111 / 0.674 |
| 1:10 | 0.804 | 0.995 | 0.107 / 0.646 | 0.809 | 0.974 | 0.121 / 0.649 |
| 1:100 | 0.660 | 1.000 | 0.127 / 0.555 | 0.706 | 1.000 | 0.140 / 0.579 |
| 1:1000 | 0.547 | 1.000 | 0.000 / 0.500 | 0.585 | 1.000 | 0.001 / 0.500 |

These are one-seed measurements on a **balanced test set**. At 1:1000 the
validation-F1 threshold makes the RF flag every test molecule, explaining its
zero FNR and 0.500 precision; it does not recover discrimination. The ranking
quality itself falls sharply as counterfeit training examples disappear. Test
precision here is not the expected precision at deployment prevalence; use the
separate prevalence projection above for that conditional calculation. A
deployment threshold needs a context-specific alarm budget and validation data
representative of the molecules it will screen.

We also set each model's threshold on validation so **no more than 1% of
authentic validation molecules are flagged**, then measure both error rates on
the untouched test. The threshold is selected separately for each training
ratio and model. The test FPR is allowed to differ from the validation budget;
`benchmark/imbalance/fpr_budget_effect.png` plots the tradeoff.

| Training ratio | RF test FNR / FPR | XGB test FNR / FPR |
|---|---:|---:|
| 1:1 | 0.803 / 0.0108 | 0.876 / 0.0082 |
| 1:10 | 0.909 / 0.0069 | 0.866 / 0.0078 |
| 1:100 | 0.960 / 0.0061 | 0.931 / 0.0093 |
| 1:1000 | 0.981 / 0.0071 | 0.974 / 0.0091 |

This strict false-alarm budget exposes the loss of sensitivity: at 1:1000,
the RF catches only about 1.9% of test counterfeits and XGBoost about 2.6%.
Their balanced-test precisions are 0.727 and 0.741, but if these measured
class-conditional rates held at an actual 1:1000 screening prevalence, Bayes'
rule projects precisions of only about **0.0027** and **0.0028**. These are
projections, not a test on 1,000 authentic molecules per counterfeit.

### C4: first error-factor audit on E

`analyze_evolved_factors.py` fits the same evolved RF used in the main baseline
(train+val, seed 42), then saves its 9,250 scaffold-test scores to
`benchmark/factor_analysis/evolved_rf_test_predictions.csv`. Its summary
reproduces AUC 0.902 and FNR 0.150 and groups the 4,625 counterfeit test rows
by molecule size, edited-atom fraction, parent fingerprint similarity and
whether an edited atom intersects the Bemis–Murcko core. The 95% Wilson
intervals describe uncertainty within each group.

Miss rates are 0.167 for 20–29-heavy-atom counterfeits (n=2,531) and 0.122
for those with at least 30 heavy atoms (n=1,489). With the corrected masks,
core-overlapping edits have FNR 0.171 (n=2,086), versus 0.133 for peripheral
edits (n=2,539). These are
**observational associations**: graph size, chemistry and edit type can change
together. Parent similarity must be interpreted cautiously because some
recorded parent SMILES include disconnected components, and most E edit masks
still lack exact one-atom verification. The controlled size-selection
stress test below probes the effect of training composition separately.

### C4: graph-size selection stress test

`build_graph_size_variants.py` keeps the same 37,001 authentic training rows and
3,700 counterfeit training rows as the random 1:10 experiment, but selects the
smallest or largest counterfeits by heavy-atom count. Median counterfeit size
is 16 (small), 35 (large), versus 24 for the random 1:10 subset. Validation
and test remain exactly the same, and `audit_graph_size_variants.py` verifies
the rows and scaffold split. `benchmark/graph_size/baseline_summary.json`
compares the fixed RF/XGBoost settings at seed 42:

| Counterfeit training selection | RF AUC | RF FNR / precision @ val threshold | XGB AUC | XGB FNR / precision @ val threshold |
|---|---:|---:|---:|---:|
| Random 1:10 | 0.804 | 0.107 / 0.646 | 0.809 | 0.121 / 0.649 |
| Smallest 3,700 | 0.704 | 0.071 / 0.561 | 0.615 | 0.022 / 0.511 |
| Largest 3,700 | 0.676 | 0.082 / 0.543 | 0.663 | 0.109 / 0.549 |

Both extreme selections lower ranking quality on the unchanged test set. Their
lower threshold-tuned FNRs arrive with lower precision, so the AUC and both
decision metrics need to be read together. This selection also changes
chemical composition, parent molecules and scaffold coverage; it cannot by
itself establish a causal graph-size effect.

### C4: scaffold-diversity training stress test

`build_scaffold_diversity_variants.py` selects 3,700 authentic and 3,700
counterfeit evolved training rows for each condition. The authentic train rows
are identical. One counterfeit sample draws from **10** scaffolds, the other
draws one molecule from each of **3,700** scaffolds. The counts of counterfeit
molecules with <20, 20–29 and ≥30 heavy atoms are identical in both conditions
(895/1,921/884). Validation and test are copied unchanged; the audit in
`benchmark/scaffold_diversity/audit.json` passes. RF and XGBoost use the same
fixed settings and seeds 42–44, fitting only the 7,400 training rows.

| Model | 10-scaffold AUC | 3,700-scaffold AUC | FNR at 0.5: low / high diversity | Validation-F1-tuned test F1: low / high |
|---|---:|---:|---:|---:|
| Fingerprint RF | 0.645 ± 0.001 | 0.740 ± 0.001 | 0.926 / 0.181 | 0.680 / 0.729 |
| Fingerprint XGBoost | 0.638 ± 0.002 | 0.738 ± 0.001 | 0.916 / 0.174 | 0.671 / 0.727 |

The more diverse training sample improves test ranking by about 0.10 AUC in
both models. The very large fixed-threshold FNR gap also reflects score
calibration: when the threshold is tuned for F1 on validation, low-diversity
models recover many more counterfeits but with lower precision (RF 0.538 vs
0.604; XGBoost 0.516 vs 0.589). Under a validation FPR budget of 1%, both
conditions still miss roughly 95–97% of test counterfeits. The property-only
AUCs remain near chance (0.523 and 0.513). This is a controlled training
selection stress test, but the two counterfeit subsets also differ in chemical
composition beyond scaffold count and coarse size bins. See
`benchmark/scaffold_diversity/baseline_summary.json` for all means and SDs.

### Historical headline results (earlier, smaller snapshots)

The following measurements are retained for context and are not current
benchmark baselines.

| Dataset | property-only | fingerprint RF | GNN (GIN) |
|---|---|---|---|
| **M** — halogen-walk | 0.65 | 0.95 | **0.97** |
| **M** — bioisostere | 0.58 | 0.87 | — |
| **M** — homologation | 0.56 | 0.89 | — |
| **M** — scaffold-hop | 0.60 | 0.80 | — |
| pooled (M + E) | 0.54 | 0.83 | — |
| **E** — evolved (small, 108 test) | 0.40 | 0.53 | 0.49 |
| **E** — evolved (2× data, 265 test) | 0.40 | **0.67** | **0.67** |

### Historical interpretation

1. **The task is real, not noise.** For rule-based counterfeits (M) a fingerprint
   model reaches AUC 0.80–0.95 on scaffolds it never trained on — the edit is a
   genuine, transferable structural signal.
2. **The signal is structural, not a shortcut.** The property-only baseline stays
   near 0.5 — property matching removed the gross-property shortcut, so any real
   signal must come from structure.
3. **M generalizes; the GNN confirms it.** On halogen-walk the GIN reaches 0.97
   on unseen scaffolds — the consistent edit pattern transfers across scaffolds.
4. **E is hard but learnable, and it scales with data.** With only ~1k evolved
   molecules the models sat near chance on unseen scaffolds (0.49–0.53); doubling
   the data lifted both to **0.67**. The earlier near-chance number was a
   small-sample artefact, not fundamental unlearnability — E is the *hard tier*,
   not an impossible one.
5. **A negative result, stated honestly.** Adding "distance to nearest known
   drug" features did **not** help on E (+0.016) — a naive relational signal does
   not crack the evolved set.

### Historical conclusion, superseded by the current snapshot

The earlier gap **M ≈ 0.95–0.97 vs E ≈ 0.67** was measured on much smaller
datasets. It motivated scaling the evolved tier but should not be cited as a
result of the regenerated benchmark. A scaffold split remains essential; the
new fixed splits balance each class within train, val and test.

---

## 7. File-by-file reference

| File | Role |
|---|---|
| [`fetch_public_molecules.py`](fetch_public_molecules.py) | Fetch real molecules from ChEMBL / PubChem (mining input) |
| [`mine_mmp_rules.py`](mine_mmp_rules.py) | Mine + categorize MMP rules from authentic drugs → `mined_transformations.py/.json` |
| [`mmp_transformations.py`](mmp_transformations.py) | The original 21 hand-written rules (Version R source) |
| [`build_version_m.py`](build_version_m.py) | Apply mined rules per category → `version_m/` |
| [`evolutionary_generator.py`](evolutionary_generator.py) | GA + NSGA-II counterfeit search (core engine) |
| [`build_version_e.py`](build_version_e.py) | Scale the evolutionary search over many parents → `version_e/` |
| [`assemble_benchmark.py`](assemble_benchmark.py) | Pair + property-match + scaffold-split + 3D → `benchmark/` |
| [`assemble_36_benchmark.py`](assemble_36_benchmark.py) | Current large-scale assembler for category, subcategory, evolved, and pooled slices |
| [`integrate_evolved_categories.py`](integrate_evolved_categories.py) | Add the six category-conditioned evolved datasets |
| [`audit_benchmark_integrity.py`](audit_benchmark_integrity.py) | Check labels, counts, and scaffold/SMILES split isolation |
| [`audit_child_scaffold_overlap.py`](audit_child_scaffold_overlap.py) | Measure actual molecule-scaffold overlap across fixed splits |
| [`build_strict_child_scaffold_holdout.py`](build_strict_child_scaffold_holdout.py) | Filter E test to unseen child scaffolds and rebalance |
| [`audit_strict_child_scaffold_holdout.py`](audit_strict_child_scaffold_holdout.py) | Check the supplementary E holdout |
| [`summarize_strict_child_scaffold.py`](summarize_strict_child_scaffold.py) | Compare the same E models on original and child-scaffold-novel tests |
| [`build_strict_cross_version_holdout.py`](build_strict_cross_version_holdout.py) | Filter M→E holdout to child scaffolds unseen in M |
| [`audit_strict_cross_version_holdout.py`](audit_strict_cross_version_holdout.py) | Audit the stricter M→E holdout |
| [`repair_halogen_substitution_masks.py`](repair_halogen_substitution_masks.py) | Correct masks where a unique reverse halogen swap proves the edit site |
| [`audit_halogen_masks.py`](audit_halogen_masks.py) | Check exact halogen masks across all tables |
| [`audit_edit_mask_sizes.py`](audit_edit_mask_sizes.py) | Flag unusually large edit masks by dataset |
| [`build_verified_single_atom_edits.py`](build_verified_single_atom_edits.py) | Find uniquely reversible one-atom edits across all tables |
| [`audit_verified_single_atom_edits.py`](audit_verified_single_atom_edits.py) | Audit the verified manifest and replay a fixed chemical sample |
| [`apply_verified_single_atom_edits.py`](apply_verified_single_atom_edits.py) | Apply proven local masks to benchmark CSVs |
| [`build_imbalance_variants.py`](build_imbalance_variants.py) | C4: nested train-only counterfeit downsampling |
| [`audit_imbalance_variants.py`](audit_imbalance_variants.py) | Check fixed validation/test rows and nested training subsets |
| [`analyze_evolved_factors.py`](analyze_evolved_factors.py) | Stratify evolved RF misses by graph size and edit features |
| [`build_graph_size_variants.py`](build_graph_size_variants.py) | Select equal-count small and large training counterfeits |
| [`audit_graph_size_variants.py`](audit_graph_size_variants.py) | Verify the size-selection stress datasets |
| [`build_scaffold_diversity_variants.py`](build_scaffold_diversity_variants.py) | Select matched-size low/high-scaffold training subsets |
| [`audit_scaffold_diversity_variants.py`](audit_scaffold_diversity_variants.py) | Check fixed rows and matched size bins |
| [`summarize_scaffold_diversity_baselines.py`](summarize_scaffold_diversity_baselines.py) | Aggregate scaffold-diversity runs over matched seeds |
| [`generate_conformers.py`](generate_conformers.py) | Central parallel 3D conformer library |
| [`audit_realism.py`](audit_realism.py) | Realism / bias audit (GuacaMol/MOSES + decoy-bias metrics) |
| [`train_baseline.py`](train_baseline.py) | Learnability: property-only + fingerprint-RF baselines (C2 lower tier) |
| [`train_gnn.py`](train_gnn.py) | Learnability: small GIN GNN (C2 middle tier) |
| [`summarize_gnn_comparison.py`](summarize_gnn_comparison.py) | Aggregate matched-seed model and representation runs |
| [`plot_ring_feature_ablation.py`](plot_ring_feature_ablation.py) | Plot paired ring-feature effects on three edit tasks |
| [`predict_gnn.py`](predict_gnn.py) | Load a saved GNN state dict and score benchmark rows |
| [`explain_cl_to_f_gin.py`](explain_cl_to_f_gin.py) | Compare GIN saliency with verified Cl→F edit sites |
| [`explain_verified_gin.py`](explain_verified_gin.py) | Score evolved GIN saliency on uniquely reversible edits |
| [`audit_explanation_fidelity.py`](audit_explanation_fidelity.py) | Probe saliency impact by zeroing one atom's input features |
| [`audit_parent_child_counterfactual.py`](audit_parent_child_counterfactual.py) | Score real parent/child molecules as a paired sensitivity check |
| [`plot_explanation_localization.py`](plot_explanation_localization.py) | Plot site and edit-region localization against random expectation |
| [`learnability_nn.py`](learnability_nn.py) | Distance-to-known-drug signal test |
| [`evaluate_generalization.py`](evaluate_generalization.py) | Detection-rate experiment across R/M/E |

---

## 8. Glossary & references (things to look up)

- **Matched Molecular Pair (MMP)** — Hussain & Rea, *J. Chem. Inf. Model.* 2010;
  tool: `mmpdb` (Dalke, Hert, Kramer 2018).
- **Reaction SMARTS** — RDKit chemical-reaction patterns.
- **Morgan / ECFP fingerprints; Tanimoto similarity** — Rogers & Hahn, 2010.
- **QED** — Bickerton et al., *Nat. Chem.* 2012. **SA score** — Ertl & Schuffenhauer, 2009.
- **Bemis–Murcko scaffold** — Bemis & Murcko, 1996.
- **Graph Genetic Algorithm (Graph-GA)** — Jensen, *Chem. Sci.* 2019.
- **NSGA-II** (non-dominated sorting, crowding distance) — Deb, Pratap, Agarwal,
  Meyarivan, *IEEE Trans. Evol. Comput.* 2002. Library: `pymoo` (Blank & Deb, 2020).
- **Pareto dominance / multi-objective optimization** — standard EA theory.
- **Maximum Common Substructure (MCS)** — RDKit `rdFMCS`.
- **DUD-E property matching** — Mysinger et al., *J. Med. Chem.* 2012.
- **Scaffold split** — MoleculeNet (Wu et al., 2018), OGB (Hu et al., 2020).
- **ETKDG (3D embedding)** — Riniker & Landrum, *J. Chem. Inf. Model.* 2015.
  **MMFF94 force field** — Halgren, 1996.

---

## 9. Reproduce from scratch

```bash
# Phase A — mine rules from an authentic-molecule set
python fetch_public_molecules.py --target 20000            # real molecules (ChEMBL/PubChem)
python mine_mmp_rules.py --smiles-file public_molecules.smi --min-support 5

# Phase B — make counterfeits
python build_version_m.py --per-category 1500                 # Version M
python build_version_e.py --no-adversarial --max-parents 200  # Version E

# Phase C — assemble datasets + 3D
python assemble_36_benchmark.py --out benchmark_regen
python integrate_evolved_categories.py --out benchmark_regen
python edit_mask_corrections.py apply --root benchmark_regen
python apply_verified_single_atom_edits.py --root benchmark_regen
python audit_halogen_masks.py --root benchmark_regen --out benchmark_regen/halogen_mask_audit.json
python audit_edit_mask_sizes.py
python rebalance_scaffold_splits.py benchmark_regen --apply
python audit_benchmark_integrity.py benchmark_regen
python verify_reproduction.py --expected benchmark --actual benchmark_regen
python generate_conformers.py --workers 8
python audit_conformers.py --library benchmark/conformers_full.sdf --missing-out benchmark/conformers_missing.smi
python refresh_sample_conformers.py --library benchmark/conformers_full.sdf
python archive_conformers.py benchmark/conformers_full.sdf --out benchmark
python build_cross_version_m.py
python cross_version_holdout.py --train category_bioisostere category_halogen-walk category_homologation category_scaffold-hop --test evolved --out benchmark/cross_version/evolved_unseen_from_M.csv --balance
python audit_cross_version.py
for seed in 42 43 44; do
  python train_baseline.py --train-csv benchmark/cross_version/M_four_categories.csv --test-csv benchmark/cross_version/evolved_unseen_from_M.csv --xgboost --seed "$seed" --out "benchmark/cross_version/baseline_seed${seed}.json"
done
python summarize_cross_version_baselines.py

# Audit realism/bias, and check learnability (baselines + GNN)
python audit_realism.py
python train_baseline.py --all              # property-only + fingerprint RF
python train_gnn.py --dataset evolved       # GIN on the hard tier
for model in gin gcn gat; do
  for seed in 42 43 44; do
    python train_gnn.py --dataset 'subcategory_halogen-walk.Cl->F' --model "$model" --epochs 40 --threads 2 --seed "$seed" --out "benchmark/gnn_comparison/Cl_to_F_${model}_seed${seed}.json"
  done
done
for seed in 42 43 44; do
  python train_gnn.py --dataset 'subcategory_halogen-walk.Cl->F' --model gin --ring-size-features --epochs 40 --threads 2 --seed "$seed" --out "benchmark/gnn_comparison/Cl_to_F_gin_ring_seed${seed}.json"
  python train_baseline.py --dataset 'subcategory_halogen-walk.Cl->F' --xgboost --seed "$seed" --out "benchmark/gnn_comparison/Cl_to_F_fingerprint_seed${seed}.json"
done
python summarize_gnn_comparison.py
python train_gnn.py --dataset 'subcategory_halogen-walk.Cl->F' --model gin --epochs 40 --seed 42 --threads 2 --out benchmark/gnn_comparison/Cl_to_F_gin_seed42.json --weights-out benchmark/models/Cl_to_F_gin_seed42.pt
python predict_gnn.py --weights benchmark/models/Cl_to_F_gin_seed42.pt --csv 'benchmark/subcategory_halogen-walk.Cl->F/dataset.csv' --out benchmark/models/Cl_to_F_gin_seed42_test_predictions.csv
python explain_cl_to_f_gin.py
python plot_explanation_localization.py
python prevalence_analysis.py
python build_imbalance_variants.py
python audit_imbalance_variants.py
python train_baseline.py --train-csv benchmark/evolved/dataset.csv --train-only --tune-threshold --target-fpr 0.01 --xgboost --name evolved_train_1_to_1 --out benchmark/imbalance/baseline_1_to_1.json
for ratio in 10 100 1000; do
  python train_baseline.py --train-csv "benchmark/imbalance/evolved_train_1_to_${ratio}.csv" --train-only --tune-threshold --target-fpr 0.01 --xgboost --name "evolved_train_1_to_${ratio}" --out "benchmark/imbalance/baseline_1_to_${ratio}.json"
done
python summarize_imbalance_baselines.py
python plot_imbalance_results.py
python plot_fpr_budget_results.py
python analyze_evolved_factors.py
python build_graph_size_variants.py
python audit_graph_size_variants.py
for size in small large; do
  python train_baseline.py --train-csv "benchmark/graph_size/evolved_train_size_${size}.csv" --train-only --tune-threshold --xgboost --name "evolved_train_size_${size}" --out "benchmark/graph_size/baseline_${size}.json"
done
python summarize_graph_size_baselines.py

# (optional) generalization probe on an existing detector
python evaluate_generalization.py --sample 250
```

The dataset rebuild is checked byte-for-byte by `verify_reproduction.py`.
Model results also depend on the pinned environment and CPU thread count;
`train_gnn.py` fixes two threads by default and records that choice.

---

## 10. Environment & large files on GitHub

**Python environment.** No `venv/` is committed (platform-specific compiled
binaries, ~1GB, wouldn't run on another machine anyway). Recreate it with:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
On macOS, the optional XGBoost baseline also needs the OpenMP runtime:
`brew install libomp`.
`benchmark/environment.json`, generated by `python record_environment.py`,
records Python 3.12.13, Torch 2.13.0, PyG 2.7.0, RDKit 2026.03.1 and the
CPU-only arm64 machine used for the current runs.

**Files too large for git, stored zstd-compressed instead.** GitHub hard-blocks
any file over 100MB. Three files exceed that; each is committed as a
`zstd -19` compressed sibling (`zstd -d` to restore):

| Original (not in git) | Compressed (in git) |
|---|---|
| `intelligent_pharma_50k_v2.pt` (209MB) | `intelligent_pharma_50k_v2.pt.zst` (12MB) |
| `mmp_mining_100k/mined.mmpdb` (311MB) | `mmp_mining_100k/mined.mmpdb.zst` (66MB) |
| `mmp_mining_100k/fragments.fragdb` (255MB) | `mmp_mining_100k/fragments.fragdb.zst` (38MB) |

```bash
zstd -d intelligent_pharma_50k_v2.pt.zst
zstd -d mmp_mining_100k/mined.mmpdb.zst
zstd -d mmp_mining_100k/fragments.fragdb.zst
```

The complete central 3D library is tracked as five independent gzip shards,
`benchmark/conformers_part_000.sdf.gz` through `_004.sdf.gz` (31–39MB each).
Restore the ignored 1GB SDF with:

```bash
for part in benchmark/conformers_part_*.sdf.gz; do gzip -dc "$part"; done > benchmark/conformers_full.sdf
```
