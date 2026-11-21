"""Collect train-only baseline results across evolved class ratios."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("benchmark/imbalance")
    variants = json.loads((root / "summary.json").read_text())
    report = {
        "source": variants["source"],
        "seed": variants["seed"],
        "protocol": "Train split only; validation and test fixed and balanced; report threshold 0.5, validation-F1 threshold, and validation FPR<=0.01 threshold",
        "results": {},
    }
    for ratio in (1, 10, 100, 1000):
        path = root / f"baseline_1_to_{ratio}.json"
        payload = json.loads(path.read_text())
        run = payload[f"evolved_train_1_to_{ratio}"]
        expected_train = (74002 if ratio == 1 else
                          variants["variants"][str(ratio)]["split_label_counts"]["train"]["0"] +
                          variants["variants"][str(ratio)]["split_label_counts"]["train"]["1"])
        if not run["train_only"] or run["seed"] != variants["seed"] or \
           run["n_train"] != expected_train or run["n_test"] != 9250 or \
           run.get("target_fpr") != 0.01:
            raise ValueError(f"Unexpected protocol in {path}")
        if any(value > run["target_fpr"] for value in
               run["validation_fpr_observed"].values()):
            raise ValueError(f"Validation FPR budget exceeded in {path}")
        report["results"][str(ratio)] = {
            "n_train": run["n_train"],
            "n_test": run["n_test"],
            "property_only": run["property_only"],
            "fingerprint_rf": run["fingerprint_rf"],
            "fingerprint_xgboost": run["fingerprint_xgboost"],
            "validation_thresholds": run.get("validation_thresholds"),
            "validation_tuned_test": run.get("validation_tuned_test"),
            "target_fpr": run.get("target_fpr"),
            "validation_fpr_thresholds": run.get("validation_fpr_thresholds"),
            "validation_fpr_observed": run.get("validation_fpr_observed"),
            "validation_fpr_test": run.get("validation_fpr_test"),
        }
    output = root / "baseline_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    for ratio, result in report["results"].items():
        print(f"1:{ratio} train={result['n_train']} test={result['n_test']}")
        for model in ("fingerprint_rf", "fingerprint_xgboost"):
            score = result[model]
            print(f"  {model}: AUC={score['roc_auc']} F1={score['f1']} "
                  f"FNR={score['false_negative_rate']}")
            if result["validation_tuned_test"] is not None:
                tuned = result["validation_tuned_test"][model]
                threshold = result["validation_thresholds"][model]
                print(f"    val threshold={threshold}: F1={tuned['f1']} "
                      f"FNR={tuned['false_negative_rate']} "
                      f"precision={tuned['precision']}")
            if result["validation_fpr_test"] is not None:
                bounded = result["validation_fpr_test"][model]
                print(f"    val FPR<={result['target_fpr']}: "
                      f"test FNR={bounded['false_negative_rate']} "
                      f"FPR={bounded['false_positive_rate']} "
                      f"precision={bounded['precision']}")


if __name__ == "__main__":
    main()
