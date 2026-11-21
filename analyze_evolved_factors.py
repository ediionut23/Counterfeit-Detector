"""Stratify fixed-split evolved RF errors by graph and edit characteristics.

This is an observational test-set analysis, not a controlled intervention. The
model and threshold match the single-seed evolved RF baseline in
baseline_results.json (train+val, 300 trees, Morgan radius 2, seed 42).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier

from train_baseline import load_split, metrics, morgan_array


def wilson_interval(errors: int, count: int) -> list[float]:
    """95% Wilson interval for a binomial error fraction."""
    z = 1.959963984540054
    p = errors / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [round(float(center - half), 4), round(float(center + half), 4)]


def bucket(value: float, cuts: tuple[float, float], names: tuple[str, str, str]) -> str:
    return names[0] if value < cuts[0] else names[1] if value < cuts[1] else names[2]


def positive_features(row: dict, mol: Chem.Mol) -> dict:
    tokens = row["edited_atoms"].split()
    try:
        edited = {int(token) for token in tokens}
    except ValueError as exc:
        raise ValueError(f"Invalid edit mask for {row['id']}") from exc
    if any(index < 0 or index >= mol.GetNumAtoms() for index in edited):
        raise ValueError(f"Edit mask outside molecule for {row['id']}")

    parent = Chem.MolFromSmiles(row["parent_smiles"])
    if parent is None:
        raise ValueError(f"Invalid parent for {row['id']}")
    similarity = DataStructs.TanimotoSimilarity(
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048),
        rdMolDescriptors.GetMorganFingerprintAsBitVect(parent, 2, 2048),
    )
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(scaffold) if scaffold.GetNumAtoms() else ()
    in_scaffold = (bool(edited.intersection(match)) if edited and match else None)
    return {
        "n_edited": len(edited),
        "edited_fraction": len(edited) / mol.GetNumHeavyAtoms(),
        "parent_tanimoto": similarity,
        "edited_in_scaffold": in_scaffold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("benchmark/evolved/dataset.csv"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/factor_analysis"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    split = load_split(args.dataset)
    train, test = split["train"], split["test"]
    if not train or not test:
        raise ValueError("Expected nonempty train+val and test splits")
    print(f"Featurizing {len(train)} train+val and {len(test)} test molecules", flush=True)
    x_train = np.array([morgan_array(row["smiles"]) for row in train])
    x_test = np.array([morgan_array(row["smiles"]) for row in test])
    y_train = np.array([int(row["label"]) for row in train])
    y_test = np.array([int(row["label"]) for row in test])
    print("Fitting 300-tree Random Forest", flush=True)
    model = RandomForestClassifier(n_estimators=300, n_jobs=args.jobs,
                                   random_state=args.seed).fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    predictions = (scores > 0.5).astype(int)
    overall = metrics(y_test, predictions, scores)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["id", "label", "score", "predicted_label", "heavy_atoms", "n_edited",
              "edited_fraction", "parent_tanimoto", "edited_in_scaffold"]
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    missing_masks = 0
    with (args.out / "evolved_rf_test_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, score, predicted in zip(test, scores, predictions):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError(f"Invalid test molecule {row['id']}")
            record = {"id": row["id"], "label": row["label"],
                      "score": round(float(score), 8),
                      "predicted_label": int(predicted),
                      "heavy_atoms": mol.GetNumHeavyAtoms(), "n_edited": "",
                      "edited_fraction": "", "parent_tanimoto": "",
                      "edited_in_scaffold": ""}
            if row["label"] == "1":
                details = positive_features(row, mol)
                if not details["n_edited"]:
                    missing_masks += 1
                record.update({key: round(value, 6) if isinstance(value, float) else value
                               for key, value in details.items()})
                groups["heavy_atoms"][bucket(mol.GetNumHeavyAtoms(), (20, 30),
                      ("<20", "20-29", ">=30"))].append(int(predicted == 0))
                groups["edited_fraction"][bucket(details["edited_fraction"], (0.1, 0.2),
                      ("<0.10", "0.10-0.19", ">=0.20"))].append(int(predicted == 0))
                groups["parent_tanimoto"][bucket(details["parent_tanimoto"], (0.7, 0.85),
                      ("<0.70", "0.70-0.84", ">=0.85"))].append(int(predicted == 0))
                scaffold_group = ("unknown" if details["edited_in_scaffold"] is None else
                                  "core" if details["edited_in_scaffold"] else "periphery")
                groups["edited_in_scaffold"][scaffold_group].append(int(predicted == 0))
            writer.writerow(record)

    report = {
        "dataset": str(args.dataset), "seed": args.seed,
        "protocol": "300-tree Morgan RF, train+val, fixed 0.5 threshold, scaffold test; observational strata",
        "test_rows": len(test), "test_counterfeits": int(y_test.sum()),
        "overall": overall, "counterfeits_without_edit_mask": missing_masks,
        "strata": {
            dimension: {
                name: {"n": len(errors), "missed": sum(errors),
                       "false_negative_rate": round(sum(errors) / len(errors), 4),
                       "wilson_95": wilson_interval(sum(errors), len(errors))}
                for name, errors in sorted(bins.items())
            }
            for dimension, bins in groups.items()
        },
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
