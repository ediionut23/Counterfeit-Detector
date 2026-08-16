# Benchmark Wiki — How the Counterfeit Dataset Is Built

This is the long-form, read-me-to-understand-it companion to the code. It
explains **what each component does, why, and the algorithms behind it** — with
particular attention to the evolutionary search, which is the least standard
piece. For the dataset *format*, see [`benchmark/README.md`](benchmark/README.md).

> **The one-paragraph summary.** A *counterfeit* is a real drug with a small,
> chemistry-aware edit. To build a dataset of them we (1) **learn** what edits
> real drugs undergo, by mining Matched Molecular Pairs from a large set of
> authentic molecules; (2) **apply** those edits, and also **discover new ones**
> with an evolutionary algorithm that searches molecule-space under several
> competing objectives; (3) **assemble** each set of counterfeits into a proper
> dataset — paired with authentic negatives, matched so no shortcut exists,
> split by scaffold, and shipped with 3D conformers and the ground-truth of
> which atoms were changed.

---

## 0. The pipeline at a glance

```
 authentic drugs (ChEMBL/PubChem)
        │
        ▼
 ┌───────────────────────┐   Phase A — LEARN the edits
 │ mine_mmp_rules.py     │   Matched Molecular Pairs (mmpdb)
 │  → mined_transformations.py (1216 rules, categorized, with support)
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
        ┌────────────────────────┐   Phase C — ASSEMBLE datasets
        │ assemble_benchmark.py  │   pair + property-match + scaffold-split + 3D
        │  → benchmark/          │
        └────────────────────────┘
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

**Result:** mined from ~16k real molecules fetched from public databases
(ChEMBL + PubChem, see `fetch_public_molecules.py`), the catalogue holds **1216
categorized rules** with support counts — median support 8 matched pairs, max
452 — vastly larger than the 21 hand-written ones, and it *discovered*
transformations that were never enumerated (CF₃→Cl, aromatizations, ring swaps,
demethylation). (The first pass, from ~7.7k molecules, gave 131 rules at median
support 4; mining from more real data raised both the count and the support.)

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

#### 3.6 Ground truth — which atoms changed (MCS)

For explanation benchmarking we must know *exactly which atoms* a transformation
changed. `edited_atoms` computes the **Maximum Common Substructure** between
parent and child (`rdFMCS`): the atoms of the child that fall *outside* the MCS
are the edited ones. This is the label against which attribution methods
(explanation accuracy, Fidelity, Sparsity) are scored — the whole reason the
task offers "free" explanation ground truth.

---

## 4. Phase C — Assembling real datasets

**File:** [`assemble_benchmark.py`](assemble_benchmark.py). Counterfeit lists are
not datasets; this turns each slice (a category, a subcategory, E, or the pool)
into one.

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
dataset has a shortcut and any GNN result on it is suspect. Ours land ~0.44–0.62,
pooled 0.498.

### 4.3 Scaffold split

`scaffold_split` groups molecules by **Bemis–Murcko scaffold** (using the
*parent's* scaffold for a counterfeit) and assigns whole scaffold families to
`train`/`val`/`test` (80/10/10). Because families never straddle the split, a
model is evaluated on ring systems it has **not** seen — the harder, more honest
setting established by MoleculeNet and OGB, and it prevents a counterfeit from
leaking information about a parent that sits in another split.

### 4.4 3D conformers (ETKDG)

`write_conformers` / [`generate_conformers.py`](generate_conformers.py) embed a
3D structure for each molecule with **ETKDGv3** (Riniker & Landrum, 2015) — a
distance-geometry method biased by real torsion-angle preferences — then relax
it with the **MMFF94** force field. The central library
`benchmark/conformers_full.sdf` embeds every unique molecule once (in parallel),
keyed by canonical SMILES. 3D coordinates enable the geometry-based
representation experiments in C1/C3.

---

## 5. The three versions and the generalization question

| Version | Counterfeits made by | Distribution shift |
|---|---|---|
| **R** | 21 hand-written rules (original 50k set) | in-distribution baseline |
| **M** | 1216 mined rules, per category/subcategory | mild — new rule *types* |
| **E** | evolutionary search, no rule list | large — discovered, combined edits |

The benchmark's central question: **does a detector trained on the edits we
enumerated (R/M) still catch edits discovered another way (E)?**
[`evaluate_generalization.py`](evaluate_generalization.py) answers a fast version
of this with detection rate (fraction scored `P>0.5`) per version and per M
category. Honest caveats are printed with the numbers: R is in-distribution (an
upper bound), E built with the adversarial objective is white-box against the
scored model (also an upper bound), so **M is the cleanest generalization
signal**.

---

## 6. Is the task learnable? Baseline results

Before any elaborate model, the first question is whether authentic and
counterfeit molecules actually differ in a learnable way — or whether, after
property matching, they are effectively indistinguishable. Three scripts answer
this, always on each dataset's **scaffold-disjoint test split** (train on seen
scaffolds, test on unseen ones), reporting the proposal's C4 metrics
(False-Negative Rate first, then Precision / Recall / F1, plus accuracy and
ROC-AUC):

- [`train_baseline.py`](train_baseline.py) — property-only logistic regression
  (MW / heavy atoms / logP) and a Morgan-fingerprint Random Forest (the C2 lower
  tier).
- [`train_gnn.py`](train_gnn.py) — a small Graph Isomorphism Network (the C2
  middle tier), kept CPU-friendly.
- [`learnability_nn.py`](learnability_nn.py) — adds "distance to nearest known
  drug" features, to test a relational signal.

### Headline results (ROC-AUC on unseen scaffolds)

| Dataset | property-only | fingerprint RF | GNN (GIN) |
|---|---|---|---|
| **M** — halogen-walk | 0.65 | 0.95 | **0.97** |
| **M** — bioisostere | 0.58 | 0.87 | — |
| **M** — homologation | 0.56 | 0.89 | — |
| **M** — scaffold-hop | 0.60 | 0.80 | — |
| pooled (M + E) | 0.54 | 0.83 | — |
| **E** — evolved (small, 108 test) | 0.40 | 0.53 | 0.49 |
| **E** — evolved (2× data, 265 test) | 0.40 | **0.67** | **0.67** |

### What it tells us

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

### The finding that matters for the thesis

The gap **M ≈ 0.95–0.97 vs E ≈ 0.67** on unseen scaffolds is the benchmark's
central claim, measured: **detectors handle the modifications we enumerated but
struggle with those discovered by evolutionary search.** And a scaffold split is
essential to see it — on E the GNN scored ~0.90 on a *same-scaffold* validation
split but only ~0.67 on unseen scaffolds, so a random split would have massively
overstated performance.

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
| [`generate_conformers.py`](generate_conformers.py) | Central parallel 3D conformer library |
| [`audit_realism.py`](audit_realism.py) | Realism / bias audit (GuacaMol/MOSES + decoy-bias metrics) |
| [`train_baseline.py`](train_baseline.py) | Learnability: property-only + fingerprint-RF baselines (C2 lower tier) |
| [`train_gnn.py`](train_gnn.py) | Learnability: small GIN GNN (C2 middle tier) |
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
python assemble_benchmark.py --conformers
python generate_conformers.py

# Audit realism/bias, and check learnability (baselines + GNN)
python audit_realism.py
python train_baseline.py --all              # property-only + fingerprint RF
python train_gnn.py --dataset evolved       # GNN on the hard tier

# (optional) generalization probe on an existing detector
python evaluate_generalization.py --sample 250
```

Every step is deterministic given `--seed` (default 42).
