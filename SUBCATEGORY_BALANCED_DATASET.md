# Fine-grained subcategory-balanced dataset

## Output

`balanced_datasets/subcategory_balanced/dataset.csv` is a scaffold-split view
of the current `final_dataset_400k` bank. It balances the **generated** class
at the full `generator/category/subcategory` level, then selects an equal
number of property-matched public reference molecules for label 0.

The build produced 12,070 unique rows: 6,035 generated analogues and 6,035
public reference molecules. It includes 85 subcategories with exactly 71
generated molecules each before the train/validation/test split. The split
column is assigned by parent Bemis–Murcko scaffold; as a result, individual
subcategories have different counts in each split. Use the train split for
training and inspect `summary.json` before using validation/test to report
fine-grained per-subcategory metrics.

The smallest eligible classes limit this dataset's size. Three subcategories
with fewer than 50 unique examples were excluded: `M/bioisostere/I->O` (6),
`M/bioisostere/remove-N/S` (2), and `M/bioisostere/remove-P` (33). Although
the builder caps a class at 2,000, property matching and the least-supported
included class set the final count at 71 per subcategory. This is why this
fine-grained view is much smaller than the three existing 200k-row balanced
datasets: balancing at every detailed transformation label discards most
examples from common subcategories. The 200k-row sets remain available for
large-scale binary training.

## Labels and provenance

- Label 1 means a generated analogue from M, E, or the category-conditioned
  genetic extension. It does not mean a laboratory-confirmed counterfeit.
- Label 0 means a quality-filtered public reference molecule, from the
  project's existing reference pool and the filtered ChEMBL controls. It does
  not mean an approved drug or a verified genuine medicine.
- The output contains no repeated SMILES. A child found in multiple generated
  subcategories is assigned once to its rarest source subcategory.
- The builder matches molecular weight, heavy-atom count, and SMILES length
  using six quantile bins. The shortcut diagnostics are recorded in
  `summary.json`.

ChEMBL reference compounds are distributed under CC BY-SA 3.0; attribute
ChEMBL and preserve the applicable license when redistributing adapted data.
See the provenance and source notes in
[`LARGE_DATASET_EXPANSION.md`](LARGE_DATASET_EXPANSION.md).

## Rebuild

```bash
venv/bin/python build_subcategory_balanced_dataset.py \
  --bank final_dataset_400k \
  --extension training_subcategory_expansion/counterfeits_mmp_by_subcategory.csv \
  --authentic-extra molecule_bank_500k/chembl_controls.csv \
  --max-per-subcategory 2000 --min-class-count 50 \
  --seed 20261031 --out balanced_datasets/subcategory_balanced
```

The exact build counts, per-split subclass counts, exclusions, and shortcut
diagnostics are in `balanced_datasets/subcategory_balanced/summary.json`.
