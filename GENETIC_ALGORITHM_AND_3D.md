# Genetic analogue generation and 3D GNN data

## Scope and labels

`version_ga/` is a new, separate output. It does not modify Version M or the
existing final molecule bank. Each category run starts from authentic parent
SMILES and allows mutation only through mined rules assigned to that category.
Scaffold crossover combines members of that same category-conditioned
population. The `evolved_<category>` label therefore records the generation
protocol; it is not a claim that every final molecule has one uniquely
verified edit of that category. These are generated analogues, not
experimentally confirmed counterfeit medicines.

## Genetic-search changes

`evolutionary_generator.py` retains its multiobjective NSGA-II search and adds
an optional per-parent archive of every valid SMILES evaluated over all
generations. Candidates that left the final population can still be considered.
`evolve_category_pilot.py` enables this archive, then applies strict similarity,
QED, SA and alert filters, removes exact duplicates, and keeps at most six
outputs per parent with pairwise Morgan similarity no higher than 0.90. A
scaffold-balanced round-robin then spreads output across parent scaffolds.
Approximate MCS edit masks are computed in parallel only for the final
category-level selections, after candidate filtering.

The mutation library now weights rules by the square root of their observed
support. Common mined rules remain favored while less frequent rules receive
more sampling. It checks each candidate rule's reactant SMARTS before running
the reaction, avoiding reaction attempts for inapplicable rules. Selection
filters require radius-3 Morgan Tanimoto in the requested band (default
0.55–0.87, matching the previously audited conditioned-E range), QED at least
0.45, SA at most 4.2, and no PAINS/Brenk alerts.
Delivered rows must also be single-component, 150–800 Da, and contain 10–70
heavy atoms. Children matching either authentic input library or the existing
`final_dataset/molecules.csv` counterfeit bank are excluded as exact structures.
These are computational plausibility criteria; they do not establish
synthesizability, safety, pharmacology, or clinical equivalence.

## Reproduction

The generation run is CPU intensive. `run_version_ga.py` runs categories in
sequence with 7,000 parents and automatically retries a category at 12,000
parents if it misses its 10,000 unique-child target. It keeps a log per
category and seed:

```bash
venv/bin/python run_version_ga.py --parents public_molecules_100k.smi \
  --rules mined_transformations.py --max-parents 7000 \
  --retry-max-parents 12000 --target 10000 --pop 16 --generations 5 \
  --per-parent-keep 6 --workers 6 --seed 20261024 --out version_ga
```

Each category produces a CSV and `summary_<category>.json`; the driver updates
`summary_all_categories.json` as each category completes. Check
`target_reached` before downstream dataset construction.

If a category falls short because of parent/child-scaffold caps, a separate
category-conditioned top-up batch can exclude its existing CSV, then
`merge_ga_category_supplement.py` rebalances the combined pool.

After all five categories are present, assemble a labelled aggregate table,
unique molecule table, and parent-relation manifest:

```bash
venv/bin/python assemble_version_ga.py --root version_ga
```

The `label=1` column means “generated analogue” for this synthetic benchmark.
It does not identify a medicine confirmed counterfeit in commerce.
After the CSVs, audits, and 3D graph files are final, record byte sizes and
SHA-256 checksums with `venv/bin/python write_version_ga_manifest.py`.

## 3D conformers and graph tensors

`prepare_3d_gnn.py` generates an ETKDGv3 conformer ensemble per molecule,
minimizes each with MMFF94s (UFF fallback when parameters are unavailable),
and stores the lowest-energy converged geometry. Hydrogens are retained. The
SDF keeps chemistry and provenance tags; `graphs_3d.npz` packs atom features,
coordinates in Å, directed bond indices and features, bond lengths, noncovalent
radius neighbors and distances (default 5 Å), triplets, and valence angles in
radians with pointer arrays delimiting molecules. The
packed arrays include a `label=1` for generated analogues; authentic controls
must be added as a separate molecule set for balanced training.
`edge_index`, `radius_edge_index`, and `angle_index` use the usual GNN layouts
`[2, E]`, `[2, E_radius]`, and `[3, N_angles]`; pointer arrays use global
concatenated node offsets.
`conformers_3d_report.json` records coverage and method.

Example:

```bash
venv/bin/python prepare_3d_gnn.py \
  --input version_ga/counterfeits_evolved_bioisostere.csv \
  --out version_ga/3d/bioisostere --workers 6 --conformers 8
```

To prepare every completed category in sequence and resume categories whose
complete conformer report already exists:

```bash
venv/bin/python prepare_version_ga_3d.py --root version_ga \
  --workers 6 --conformers 8 --radius-cutoff 5.0
```

The 3D audit checks coordinate, edge, distance, angle, SDF, and pointer-array
consistency:

```bash
venv/bin/python audit_version_ga_3d.py --root version_ga
```

The 3D coordinates are force-field-minimized predictions, not experimental
crystal structures or solution ensembles. Keep all geometry generation
metadata, and do not feed parent, category, similarity, QED, SA, edit masks, or
force-field energy into a structure-only classifier. Use those only for
stratification, provenance, or quality audits. Before training, split by
parent/scaffold groups and build balanced class datasets from the category
tables.

## Check against experimental PDB ligand coordinates

`validate_ga_experimental_3d.py` checks exact generated-molecule InChIKeys
against the wwPDB Chemical Component Dictionary and compares compatible
structures using symmetry-aware heavy-atom RMSD. It uses CCD model coordinates
from deposited PDB ligand structures, never the CCD idealized coordinates.
Run it with network access to the public RCSB API:

```bash
venv/bin/python validate_ga_experimental_3d.py --root version_ga \
  --batch-size 100 --workers 4
```

The completed bank had 15 exact CCD InChIKey matches among 49,969 distinct
InChIKeys from 49,996 molecules with 3D coordinates. Eleven allowed an
atom-wise coordinate comparison (median RMSD 1.070 Å); three exact-key matches used a different tautomer graph, and one had
no source PDB ID for its CCD model coordinates. This small matched subset does
not establish that all generated conformers are experimentally accurate.
Outputs, cached CCD CIFs, exclusions, and per-molecule RMSDs are documented in
[`version_ga/REPORT.md`](version_ga/REPORT.md) and stored under
`version_ga/experimental_reference/`.

## Completed output

The final banks contain 50,000 unique generated analogues, 10,000 per category.
The category and 3D metrics, failed-embedding list, and audit results are in
[`version_ga/REPORT.md`](version_ga/REPORT.md). In total, 49,996 molecules
(99.992%) have 3D graph tensors and SDF conformers; four molecules could not be
embedded with the configured RDKit workflow and are listed in
`version_ga/3d/failed_embeddings.csv`. The produced graph/SDF archives pass
the internal consistency audit. The coordinates are minimized force-field
predictions, not experimental geometries.

Version M 3D outputs and the first balanced M/E datasets are documented in
[`BALANCED_GENERATOR_DATASETS.md`](BALANCED_GENERATOR_DATASETS.md). The expanded
500k+ bank and 200k+ balanced datasets are documented in
[`LARGE_DATASET_EXPANSION.md`](LARGE_DATASET_EXPANSION.md). The Version M graph
bank embeds 49,960 of its 50,000 molecules (99.92%); all five category archives
pass the graph and coordinate audit.
