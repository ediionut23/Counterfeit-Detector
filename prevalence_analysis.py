"""Project fixed-threshold precision to rarer counterfeit prevalences.

This changes only the assumed class prevalence. It does not retrain models or
claim to measure a real 1:1000 test set. TPR and FPR come from the recorded,
balanced, scaffold-disjoint test sets and are assumed stable under the shift.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def label_counts(path: Path) -> tuple[int, int]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and "split" in rows[0]:
        rows = [row for row in rows if row["split"] == "test"]
    return (sum(row["label"] == "1" for row in rows),
            sum(row["label"] == "0" for row in rows))


def infer_confusion(result: dict, positive: int, negative: int) -> list[dict]:
    candidates = []
    for fn in range(positive + 1):
        if round(fn / positive, 3) != result["false_negative_rate"]:
            continue
        tp = positive - fn
        if round(tp / positive, 3) != result["recall"]:
            continue
        for fp in range(negative + 1):
            tn = negative - fp
            precision = tp / (tp + fp) if tp + fp else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            if (round(precision, 3) == result["precision"]
                    and round(f1, 3) == result["f1"]
                    and round((tp + tn) / (positive + negative), 3)
                    == result["accuracy"]):
                candidates.append({"tp": tp, "fp": fp, "tn": tn, "fn": fn})
    if not candidates:
        raise ValueError("Rounded metrics are inconsistent with all confusion matrices")
    return candidates


def analyze(result: dict, positive: int, negative: int) -> dict:
    candidates = infer_confusion(result, positive, negative)
    tprs = [matrix["tp"] / positive for matrix in candidates]
    fprs = [matrix["fp"] / negative for matrix in candidates]
    projections = {}
    for ratio in (10, 100, 1000):
        prevalence = 1 / (ratio + 1)
        values = []
        for tpr, fpr in zip(tprs, fprs):
            numerator = tpr * prevalence
            denominator = numerator + fpr * (1 - prevalence)
            values.append(numerator / denominator)
        projections[f"1:{ratio}"] = {
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    return {
        "compatible_confusion_matrices": len(candidates),
        "confusion": candidates[0] if len(candidates) == 1 else None,
        "tpr_range": [round(min(tprs), 6), round(max(tprs), 6)],
        "fpr_range": [round(min(fprs), 6), round(max(fprs), 6)],
        "projected_precision_range": projections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    parser.add_argument("--baselines", type=Path, default=Path("baseline_results.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark/prevalence_analysis.json"))
    args = parser.parse_args()

    results = {}
    baseline = json.loads(args.baselines.read_text())
    for name, run in baseline.items():
        positive, negative = label_counts(args.benchmark / name / "dataset.csv")
        results[name] = {
            "n_positive_test": positive,
            "n_negative_test": negative,
            "models": {model: analyze(run[model], positive, negative)
                       for model in ("property_only", "fingerprint_rf")},
        }
    for path in sorted((args.benchmark / "cross_version").glob("baseline_seed*.json")):
        run = json.loads(path.read_text())["cross_version_M_to_E"]
        positive, negative = label_counts(Path(run["test_csv"]))
        name = f"cross_version_M_to_E_seed{run['seed']}"
        results[name] = {
            "n_positive_test": positive,
            "n_negative_test": negative,
            "models": {model: analyze(run[model], positive, negative)
                       for model in ("property_only", "fingerprint_rf",
                                     "fingerprint_xgboost")},
        }
    report = {
        "method": "Precision projected from fixed-threshold TPR/FPR; no retraining or resampling",
        "assumption": "Class-conditional error rates remain constant as prevalence changes",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Analyzed {len(results)} recorded test runs -> {args.out}")


if __name__ == "__main__":
    main()
