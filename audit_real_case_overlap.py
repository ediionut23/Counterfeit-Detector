#!/usr/bin/env python3
"""Audit chemical exposure of real-case compounds in the prior synthetic data.

Exact identity and scaffold overlap are separate from case-label leakage. The
old unary classifier never received parent_smiles; these are audited as future
declared/observed inputs, not falsely described as old model features.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_real_case_benchmark import standardized, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("balanced_datasets/training_200k/subcategory_weighted/dataset.csv"))
    parser.add_argument("--benchmark", type=Path, default=Path("real_world_falsification/v1"))
    args = parser.parse_args()
    with (args.benchmark / "molecules.csv").open(newline="") as stream:
        molecules = list(csv.DictReader(stream))
    target_identity = defaultdict(set)
    target_scaffold = defaultdict(set)
    for row in molecules:
        target_identity[row["connectivity_smiles"]].add(row["compound_id"])
        target_scaffold[row["scaffold_key"]].add(row["compound_id"])
    counts = Counter()
    source_counts = Counter()
    linked_rows, failures = [], []
    out = args.benchmark / "prior_model_overlap"
    out.mkdir(parents=True, exist_ok=True)
    with args.dataset.open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 1):
            split, label = row["split"], row["label"]
            source_counts[split] += 1
            for role, smiles in (("old_model_input", row["smiles"]),
                                  ("parent_context_not_old_model_input", row.get("parent_smiles", ""))):
                if not smiles:
                    continue
                try:
                    normalized, connectivity, scaffold = standardized(smiles)
                except Exception as error:
                    failures.append({"source_row_id": row["id"], "role": role,
                                     "smiles": smiles, "error": str(error)})
                    continue
                for compound_id in target_identity.get(connectivity, set()):
                    counts[compound_id, split, role, "identity_ignoring_stereo", label] += 1
                for compound_id in target_scaffold.get(scaffold, set()):
                    counts[compound_id, split, role, "scaffold", label] += 1
                if role.startswith("parent") and target_identity.get(connectivity):
                    linked_rows.append({
                        "source_row_id": row["id"], "source_relation_id": row.get("source_relation_id", ""),
                        "source_split": split, "generation_method": row["generation_method"],
                        "declared_reference_compound_ids": json.dumps(sorted(target_identity[connectivity])),
                        "declared_reference_smiles": normalized, "simulated_observed_smiles": row["smiles"],
                        "scenario": "hypothetical_structural_analogue_substitution",
                        "evidence_tier": "synthetic_generator_relation", "confirmed_real_case": 0,
                        "holdout_policy": "quarantine_from_future_training_for_this_case_challenge",
                    })
            if index % 25000 == 0:
                print(f"Audited {index:,} dataset rows", flush=True)

    overlap = [{"compound_id": k[0], "split": k[1], "role": k[2], "match_type": k[3],
                "source_label": k[4], "row_count": n} for k, n in sorted(counts.items())]
    write_csv(out / "compound_exposure.csv", overlap,
              ["compound_id", "split", "role", "match_type", "source_label", "row_count"])
    write_csv(out / "normalization_failures.csv", failures,
              ["source_row_id", "role", "smiles", "error"])
    write_csv(out / "synthetic_relations_linked_to_case_compounds.csv", linked_rows,
              ["source_row_id", "source_relation_id", "source_split", "generation_method",
               "declared_reference_compound_ids", "declared_reference_smiles", "simulated_observed_smiles",
               "scenario", "evidence_tier", "confirmed_real_case", "holdout_policy"])
    summary = {
        "source_dataset": str(args.dataset), "source_rows_by_split": dict(source_counts),
        "normalization_failures": len(failures),
        "normalization_policy": "Same active-moiety normalization as registry; exact identity exposure ignores stereo conservatively; no tautomer merging.",
        "training_input_exact_identity_exposure": sorted({k[0] for k in counts if k[1] == "train" and k[2] == "old_model_input" and k[3] == "identity_ignoring_stereo"}),
        "training_input_scaffold_exposure": sorted({k[0] for k in counts if k[1] == "train" and k[2] == "old_model_input" and k[3] == "scaffold"}),
        "synthetic_parent_relations_linked_to_case_compounds": len(linked_rows),
        "interpretation": "Chemical overlap is exposure, not proof that real product-case labels were in training. The prior model did not use parent context.",
        "future_holdout_policy": "Quarantine all case compounds and their scaffolds from both declared and detected sides of future synthetic training; keep all source-report clusters external.",
        "leakage_free_real_world_validation_established": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
