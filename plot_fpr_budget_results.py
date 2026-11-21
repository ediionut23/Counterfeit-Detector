"""Plot class-scarcity performance under a 1% validation false-alarm budget."""

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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    for model, label, color, marker in models:
        runs = [report["results"][str(ratio)] for ratio in ratios]
        metrics = [run["validation_fpr_test"][model] for run in runs]
        for ax, key in zip(axes, ("false_negative_rate", "false_positive_rate", "precision")):
            ax.plot(ratios, [item[key] for item in metrics], color=color,
                    marker=marker, linewidth=2, label=label)

    titles = ("Missed counterfeits: test FNR",
              "False alarms: test FPR",
              "Alarm quality: test precision")
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xticks(ratios, [f"1:{ratio}" for ratio in ratios])
        ax.set_xlabel("Counterfeit:authentic ratio in training")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("FNR (lower is better)")
    axes[1].set_ylabel("FPR (lower is better)")
    axes[1].set_ylim(0, 0.05)
    axes[1].axhline(0.01, color="gray", linewidth=1,
                    linestyle=":", label="1% validation target")
    axes[2].set_ylabel("Precision on balanced test")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Evolved counterfeits · threshold chosen for validation FPR ≤ 1% · seed 42")
    fig.text(0.5, 0.01, "Validation and scaffold test fixed at 4,625 molecules per class; test FPR can exceed the validation target.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(root / "fpr_budget_effect.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "fpr_budget_effect.svg", bbox_inches="tight")
    print(root / "fpr_budget_effect.png")


if __name__ == "__main__":
    main()
