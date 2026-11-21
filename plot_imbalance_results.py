"""Plot the effect of rare training counterfeits on fixed-test baselines."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path("benchmark/imbalance")
    report = json.loads((root / "baseline_summary.json").read_text())
    ratios = [1, 10, 100, 1000]
    models = (("fingerprint_rf", "Random Forest", "#1f77b4", "o"),
              ("fingerprint_xgboost", "XGBoost", "#d95f02", "s"))

    fig, (auc_ax, fnr_ax, precision_ax) = plt.subplots(
        1, 3, figsize=(14, 4.5), sharex=True)
    for model, label, color, marker in models:
        runs = [report["results"][str(ratio)] for ratio in ratios]
        auc_ax.plot(ratios, [run[model]["roc_auc"] for run in runs],
                    color=color, marker=marker, linewidth=2, label=label)
        fnr_ax.plot(ratios, [run[model]["false_negative_rate"] for run in runs],
                    color=color, marker=marker, linewidth=2, label=f"{label}, 0.5")
        fnr_ax.plot(ratios, [run["validation_tuned_test"][model]["false_negative_rate"]
                             for run in runs], color=color, marker=marker,
                    linestyle="--", linewidth=2, label=f"{label}, val threshold")
        precision_ax.plot(ratios, [run[model]["precision"] for run in runs],
                          color=color, marker=marker, linewidth=2)
        precision_ax.plot(ratios, [run["validation_tuned_test"][model]["precision"]
                                   for run in runs], color=color, marker=marker,
                          linestyle="--", linewidth=2)

    for ax in (auc_ax, fnr_ax, precision_ax):
        ax.set_xscale("log")
        ax.set_xticks(ratios, [f"1:{ratio}" for ratio in ratios])
        ax.set_xlabel("Counterfeit:authentic ratio in training")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1.02)
    auc_ax.set_title("Ranking: test ROC-AUC")
    auc_ax.set_ylabel("ROC-AUC")
    auc_ax.legend(loc="lower left", frameon=False)
    fnr_ax.set_title("Detection: test false-negative rate")
    fnr_ax.set_ylabel("FNR (lower is better)")
    fnr_ax.legend(loc="upper left", fontsize=8, frameon=False)
    precision_ax.set_title("Alarm quality: test precision")
    precision_ax.set_ylabel("Precision (higher is better)")
    precision_ax.axhline(0.5, color="gray", linewidth=1, alpha=0.6)
    fig.suptitle("Evolved counterfeits · fixed balanced scaffold test (n = 9,250) · seed 42")
    fig.text(0.5, 0.01, "Solid: decision threshold 0.5   |   Dashed: threshold maximizing F1 on balanced validation",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(root / "imbalance_effect.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "imbalance_effect.svg", bbox_inches="tight")
    print(root / "imbalance_effect.png")


if __name__ == "__main__":
    main()
