"""Aggregate validation-calibrated low-alarm results across saved-model seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


NAMES = (("properties", "Three properties"), ("rf", "Morgan + RF"),
         ("xgboost", "Morgan + XGBoost"), ("gin", "GIN"),
         ("gcn", "GCN"), ("gat", "GAT"))
FIELDS = ("validation_fpr", "test_fpr", "test_fnr", "test_precision")


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": round(mean(values), 4),
            "sample_sd": round(stdev(values), 4) if len(values) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("training_runs/mixed_20k"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epoch-cap", type=int, default=80)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    source = []
    for seed in args.seeds:
        path = args.root / f"fpr_budget_seed{seed}_{args.epoch_cap}.json"
        row = json.loads(path.read_text())
        if (row["seed"] != seed or row["target_validation_fpr"] != 0.01
                or row["n_val"] != 3968 or row["n_test"] != 2948
                or set(row["models"]) != {name for name, _ in NAMES}):
            raise ValueError(f"Inconsistent FPR budget run: {path}")
        source.append(row)
    if len({row["dataset_sha256"] for row in source}) != 1:
        raise ValueError("Runs use different training datasets")
    models = {}
    for name, _ in NAMES:
        runs = [row["models"][name] for row in source]
        if any(r["validation_false_positives"] > 19 or
               r["validation_negatives"] != 1984 or
               r["test_negatives"] != 1474 or r["test_positives"] != 1474
               for r in runs):
            raise ValueError(f"Unexpected class counts or validation FPR: {name}")
        models[name] = {
            "metrics": {field: stats([float(r[field]) for r in runs])
                        for field in FIELDS},
            "positive_method_fnr": {
                method: stats([float(r["positive_method"][method]["fnr"])
                               for r in runs]) for method in ("E", "M")},
            "runs": [{"seed": seed, **row} for seed, row in zip(args.seeds, runs)],
        }
    result = {"dataset_sha256": source[0]["dataset_sha256"],
              "target_validation_fpr": 0.01, "seeds": args.seeds,
              "epoch_cap": args.epoch_cap, "models": models}
    stem = f"fpr_budget_summary_{args.epoch_cap}"
    output = args.root / f"{stem}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    def cell(name: str, field: str) -> str:
        row = models[name]["metrics"][field]
        return f"{row['mean']:.3f} ± {row['sample_sd']:.3f}"

    lines = ["# Results under a 1% validation false-alarm budget", "",
             "Thresholds were selected on validation for each fitted model;",
             "every model used the same 1,984 authentic validation rows and",
             "1,474 authentic plus 1,474 generated test rows. Values are",
             "means ± sample standard deviations over seeds " +
             ", ".join(map(str, args.seeds)) + ".", "",
             "| Model | Test FPR | Test FNR | Test precision |",
             "|---|---:|---:|---:|",]
    for name, label in NAMES:
        lines.append(f"| {label} | {cell(name, 'test_fpr')} | "
                     f"{cell(name, 'test_fnr')} | {cell(name, 'test_precision')} |")
    lines += ["", "The 1% target applies to validation. Test FPR can be above",
              "or below it. Under this strict threshold the models miss most",
              "generated analogues; these classifiers are not ready for a",
              "low-false-alarm screening claim. Test precision reflects a",
              "balanced test set and would change with real-world prevalence.",
              "This is a fixed-setting comparison on selected synthetic",
              "analogues, without equal model tuning or biological labels.", ""]
    (args.root / f"{stem}.md").write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
