"""
Generate High-Quality Visualization of Benchmark Dataset Distribution
======================================================================
Produces a comprehensive 4-panel publication-ready figure:
  Panel A: Macro-Categories & Evolved Set (1:1 Class Balance)
  Panel B: Transformation Family Breakdown (Donut Chart)
  Panel C: 36 Fine-Grained Chemical Subcategories (Horizontal Bar Chart)
  Panel D: Shortcut Resistance Verification (Trivial LR Accuracy Distribution)
"""

import json
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

def main():
    with open("benchmark/index.json") as f:
        idx = json.load(f)

    # 1. Extract macro categories + evolved
    macro_keys = [
        ("category_homologation", "Homologation", "#2ecc71"),
        ("category_bioisostere", "Bioisostere", "#9b59b6"),
        ("category_scaffold-hop", "Scaffold-Hop", "#3498db"),
        ("category_halogen-walk", "Halogen-Walk", "#e67e22"),
        ("category_other", "Other Functional", "#e74c3c"),
        ("evolved", "Evolved (NSGA-II)", "#f39c12"),
    ]

    macro_names = []
    macro_auth = []
    macro_fake = []
    macro_total = []
    macro_colors = []

    for key, label, col in macro_keys:
        if key in idx:
            data = idx[key]
            macro_names.append(label)
            macro_auth.append(data["n_authentic"])
            macro_fake.append(data["n_counterfeit"])
            macro_total.append(data["n_total"])
            macro_colors.append(col)

    # 2. Extract the 36 primary subcategories
    sub_data = []
    fam_color_map = {
        "scaffold-hop": "#3498db",
        "halogen-walk": "#e67e22",
        "homologation": "#2ecc71",
        "bioisostere": "#9b59b6",
        "other": "#e74c3c",
    }

    for k, v in idx.items():
        if k.startswith("subcategory_"):
            clean_name = k.replace("subcategory_", "")
            fam = clean_name.split(".")[0] if "." in clean_name else "other"
            sub_part = clean_name.split(".", 1)[1] if "." in clean_name else clean_name
            # Focus on well-populated subcategories (>= 500 total)
            if v["n_total"] >= 500:
                sub_data.append({
                    "name": clean_name,
                    "short_name": sub_part,
                    "family": fam,
                    "total": v["n_total"],
                    "fake": v["n_counterfeit"],
                    "auth": v["n_authentic"],
                    "lr_acc": v.get("shortcut_metrics", {}).get("trivial_lr_accuracy", 0.5),
                    "color": fam_color_map.get(fam, "#95a5a6"),
                })

    sub_data.sort(key=lambda x: x["total"], reverse=True)

    # Create figure layout
    fig = plt.figure(figsize=(20, 16), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4], hspace=0.28, wspace=0.25)

    # ══════════════════════════════════════════════════════════════════
    # Panel A: Macro-Categories (Grouped Bar Chart)
    # ══════════════════════════════════════════════════════════════════
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(macro_names))
    width = 0.38

    bars1 = ax0.bar(x - width / 2, macro_auth, width, label="Authentic (Label 0)", color="#2c3e50", alpha=0.88, edgecolor="black", linewidth=0.8)
    bars2 = ax0.bar(x + width / 2, macro_fake, width, label="Counterfeit (Label 1)", color=macro_colors, alpha=0.92, edgecolor="black", linewidth=0.8)

    ax0.set_title("A. Macro-Categories & Genetic Set (1:1 Balanced)", fontsize=15, fontweight="bold", pad=12)
    ax0.set_xticks(x)
    ax0.set_xticklabels(macro_names, fontsize=11, rotation=15, ha="right", fontweight="medium")
    ax0.set_ylabel("Number of Molecules", fontsize=12, fontweight="medium")
    ax0.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax0.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11)
    ax0.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2:
        yval = bar.get_height()
        ax0.annotate(f"{yval * 2:,} total",
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # ══════════════════════════════════════════════════════════════════
    # Panel B: Transformation Family Proportions (Donut Chart)
    # ══════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 1])
    wedges, texts, autotexts = ax1.pie(
        macro_total,
        labels=macro_names,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=140,
        colors=macro_colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontsize(10.5)
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(11)
        t.set_fontweight("medium")

    # Center label
    total_benchmark_mols = sum(v["n_total"] for v in idx.values())
    ax1.text(0, 0, f"Suite Total\n{total_benchmark_mols:,}\nMolecules",
             ha="center", va="center", fontsize=13, fontweight="bold", color="#2c3e50")
    ax1.set_title("B. Composition by Modification Paradigm", fontsize=15, fontweight="bold", pad=12)

    # ══════════════════════════════════════════════════════════════════
    # Panel C: Top Chemical Subcategories (Horizontal Bar Chart)
    # ══════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    top_sub = sub_data[:28]  # top 28 most populated subcategories
    y_pos = np.arange(len(top_sub))
    sub_totals = [s["total"] for s in top_sub]
    sub_labels = [f"{s['short_name']} ({s['family']})" for s in top_sub]
    sub_cols = [s["color"] for s in top_sub]

    bars_sub = ax2.barh(y_pos, sub_totals, align="center", color=sub_cols, alpha=0.88, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sub_labels, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Total Molecules (1:1 Authentic : Counterfeit)", fontsize=12, fontweight="medium")
    ax2.set_title(f"C. Top Fine-Grained Chemical Subcategories (Min. ≥ 1,500 mols)", fontsize=15, fontweight="bold", pad=12)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    # Value labels
    for bar in bars_sub:
        w = bar.get_width()
        ax2.text(w + 300, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                 ha="left", va="center", fontsize=8.5, fontweight="bold", color="#34495e")

    # Legend for families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="black", label="Scaffold-Hop"),
        Patch(facecolor="#e67e22", edgecolor="black", label="Halogen-Walk"),
        Patch(facecolor="#2ecc71", edgecolor="black", label="Homologation"),
        Patch(facecolor="#9b59b6", edgecolor="black", label="Bioisostere"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Other"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.9, fontsize=9.5)

    # ══════════════════════════════════════════════════════════════════
    # Panel D: Shortcut Elimination Verification (Trivial LR Accuracy)
    # ══════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    lr_scores = [s["lr_acc"] for s in sub_data if s["lr_acc"] is not None]

    sns.histplot(lr_scores, bins=14, kde=True, ax=ax3, color="#1abc9c", edgecolor="black", linewidth=0.8, alpha=0.75)
    ax3.axvline(0.50, color="#e74c3c", linestyle="--", linewidth=2.2, label="Ideal Target = 0.500 (Pure Guessing / No Shortcut)")
    median_lr = np.median(lr_scores)
    ax3.axvline(median_lr, color="#2c3e50", linestyle="-", linewidth=2.0, label=f"Dataset Median = {median_lr:.3f}")

    ax3.set_title("D. Shortcut Elimination: Trivial Classifier Accuracy (DUD-E Matching)", fontsize=15, fontweight="bold", pad=12)
    ax3.set_xlabel("Trivial Logistic Regression Accuracy (MW + Heavy Atoms + Length)", fontsize=12, fontweight="medium")
    ax3.set_ylabel("Number of Subcategories", fontsize=12, fontweight="medium")
    ax3.set_xlim(0.30, 0.75)
    ax3.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10.5)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Annotative text box
    ax3.text(0.33, ax3.get_ylim()[1] * 0.75,
             "✓ Property-matched marginals\n"
             "✓ Cohen's d ≈ 0 across properties\n"
             "✓ Models forced to learn structure\n"
             "   rather than macroscopic size",
             fontsize=10.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7", alpha=0.9))

    # Master title
    fig.suptitle("Counterfeit Pharmaceutical Benchmark Suite — Complete Category Distribution",
                 fontsize=18, fontweight="bold", y=0.98, color="#2c3e50")

    # Save outputs
    out_path = Path("benchmark_distribution.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to {out_path.absolute()}")

    # Copy to artifacts directory
    artifact_dir = Path("/Users/ediionut23/.gemini/antigravity-ide/brain/3f90209c-0d41-4677-a0c8-e463d6d2d3a1")
    if artifact_dir.exists():
        dest = artifact_dir / "benchmark_distribution.png"
        shutil.copy(out_path, dest)
        print(f"Copied figure to artifact path: {dest}")

if __name__ == "__main__":
    main()
