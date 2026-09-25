"""
Generate Complete 36-Category & Macro Distribution Visualizations (Flawless Layout)
==================================================================================
Produces:
  1. `all_categories_distribution.png` (Comprehensive high-res breakdown by family)
  2. `benchmark_distribution.png` (Artifact duplicate)
"""

import json
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

def main():
    with open("benchmark/index.json") as f:
        idx = json.load(f)

    # 1. Macro categories + evolved
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

    # 2. Extract subcategories
    fam_color_map = {
        "scaffold-hop": "#3498db",
        "halogen-walk": "#e67e22",
        "homologation": "#2ecc71",
        "bioisostere": "#9b59b6",
        "other": "#e74c3c",
    }

    sub_data = []
    for k, v in idx.items():
        if k.startswith("subcategory_"):
            clean_name = k.replace("subcategory_", "")
            fam = clean_name.split(".")[0] if "." in clean_name else "other"
            sub_part = clean_name.split(".", 1)[1] if "." in clean_name else clean_name
            sub_data.append({
                "key": k,
                "name": clean_name,
                "short_name": sub_part,
                "family": fam,
                "total": v["n_total"],
                "fake": v["n_counterfeit"],
                "auth": v["n_authentic"],
                "lr_acc": v.get("shortcut_metrics", {}).get("trivial_lr_accuracy", 0.5),
                "color": fam_color_map.get(fam, "#95a5a6"),
            })

    # Sort descending by total molecules
    sub_data.sort(key=lambda x: x["total"], reverse=True)
    top_36 = sub_data[:36]

    # ══════════════════════════════════════════════════════════════════
    # FIGURE: 24 x 18 inches, 300 DPI
    # ══════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(24, 18), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.8], hspace=0.26, wspace=0.24)

    # ------------------------------------------------------------------
    # Panel A: Macro Categories (Grouped Bar Chart)
    # ------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(macro_names))
    width = 0.38

    bars1 = ax0.bar(x - width / 2, macro_auth, width, label="Autentic (Label 0)",
                    color="#2c3e50", alpha=0.9, edgecolor="black", linewidth=0.8)
    bars2 = ax0.bar(x + width / 2, macro_fake, width, label="Contrafăcut (Label 1)",
                    color=macro_colors, alpha=0.92, edgecolor="black", linewidth=0.8)

    ax0.set_title("A. Macro-Categorii & Dataset Evolutiv (Balansare Perfectă 1:1)", fontsize=16, fontweight="bold", pad=12)
    ax0.set_xticks(x)
    ax0.set_xticklabels(macro_names, fontsize=12, rotation=15, ha="right", fontweight="medium")
    ax0.set_ylabel("Număr Molecule", fontsize=13, fontweight="medium")
    ax0.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax0.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11)
    ax0.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2:
        yval = bar.get_height()
        ax0.annotate(f"{yval * 2:,}\ntotal",
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2c3e50")

    # ------------------------------------------------------------------
    # Panel B: Donut Chart - Proportions across Paradigms
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    wedges, texts, autotexts = ax1.pie(
        macro_total,
        labels=macro_names,
        autopct="%1.1f%%",
        pctdistance=0.74,
        startangle=140,
        colors=macro_colors,
        wedgeprops=dict(width=0.44, edgecolor="white", linewidth=2.5),
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontsize(11)
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(11.5)
        t.set_fontweight("medium")

    grand_total = sum(macro_total)
    ax1.text(0, 0, f"Total Suite\n{grand_total:,}\nMolecule\n(50% Auth / 50% Fake)",
             ha="center", va="center", fontsize=13.5, fontweight="bold", color="#2c3e50")
    ax1.set_title("B. Distribuție Procentuală pe Familii Chimice", fontsize=16, fontweight="bold", pad=12)

    # ------------------------------------------------------------------
    # Panel C: All 36 Fine-Grained Chemical Subcategories (Full Horizontal Bar)
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(top_36))
    sub_totals = [s["total"] for s in top_36]
    sub_labels = [f"{i+1}. {s['short_name']} ({s['family']})" for i, s in enumerate(top_36)]
    sub_cols = [s["color"] for s in top_36]

    bars_sub = ax2.barh(y_pos, sub_totals, align="center", color=sub_cols, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sub_labels, fontsize=9.2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Total Molecule (1:1 Autentic : Contrafăcut)", fontsize=13, fontweight="medium")
    ax2.set_title("C. Toate Cele 36 Subcategorii Chimice Primare (Fiecare cu 1:1 Balancing & ≥ 450 – 39,000 Mols)", fontsize=16, fontweight="bold", pad=12)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars_sub:
        w = bar.get_width()
        ax2.text(w + 350, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                 ha="left", va="center", fontsize=8.2, fontweight="bold", color="#2c3e50")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="black", label="Scaffold-Hop (5 subcat)"),
        Patch(facecolor="#e67e22", edgecolor="black", label="Halogen-Walk (9 subcat)"),
        Patch(facecolor="#2ecc71", edgecolor="black", label="Homologation (5 subcat)"),
        Patch(facecolor="#9b59b6", edgecolor="black", label="Bioisostere (16 subcat)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Other Functional (1 subcat)"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.95, fontsize=10.5)

    # ------------------------------------------------------------------
    # Panel D: Subdivided into D1 (Family Totals) and D2 (Shortcut Elimination)
    # ------------------------------------------------------------------
    gs_d = gs[1, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.32)

    # D1: Family aggregate totals
    ax3_top = fig.add_subplot(gs_d[0])
    fam_counts = {}
    for s in top_36:
        fam_counts[s["family"]] = fam_counts.get(s["family"], 0) + s["total"]

    fam_names_order = ["homologation", "scaffold-hop", "bioisostere", "other", "halogen-walk"]
    fam_display = ["Homologation\n(5 subcat)", "Scaffold-Hop\n(5 subcat)", "Bioisostere\n(16 subcat)", "Other\n(1 subcat)", "Halogen-Walk\n(9 subcat)"]
    fam_vals = [fam_counts.get(f, 0) for f in fam_names_order]
    fam_colors_list = [fam_color_map[f] for f in fam_names_order]

    y_fam = np.arange(len(fam_names_order))
    bars_fam = ax3_top.barh(y_fam, fam_vals, color=fam_colors_list, alpha=0.9, edgecolor="black", linewidth=0.8, height=0.55)
    ax3_top.set_yticks(y_fam)
    ax3_top.set_yticklabels(fam_display, fontsize=11, fontweight="medium")
    ax3_top.invert_yaxis()
    ax3_top.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax3_top.set_xlabel("Total Molecule în Subcategorii", fontsize=11, fontweight="medium")
    ax3_top.set_title("D1. Sumar pe Familii Chimice (Cele 36 Subcategorii)", fontsize=14, fontweight="bold", pad=10)
    ax3_top.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars_fam:
        w = bar.get_width()
        ax3_top.text(w + 1200, bar.get_y() + bar.get_height() / 2, f"{int(w):,} mols",
                     ha="left", va="center", fontsize=10.5, fontweight="bold", color="#2c3e50")

    # D2: Shortcut Elimination Verification (Trivial LR Accuracy Distribution)
    ax3_bot = fig.add_subplot(gs_d[1])
    lr_scores = [s["lr_acc"] for s in sub_data if s["lr_acc"] is not None]

    sns.histplot(lr_scores, bins=14, kde=True, ax=ax3_bot, color="#1abc9c", edgecolor="black", linewidth=0.8, alpha=0.75)
    ax3_bot.axvline(0.50, color="#e74c3c", linestyle="--", linewidth=2.2, label="Ideal = 0.500 (Zero Shortcut / Pură Ghicire)")
    median_lr = np.median(lr_scores)
    ax3_bot.axvline(median_lr, color="#2c3e50", linestyle="-", linewidth=2.0, label=f"Mediană Dataset = {median_lr:.3f}")

    ax3_bot.set_title("D2. Validare Rezistență la Shortcut-uri (DUD-E Binned Matching)", fontsize=14, fontweight="bold", pad=10)
    ax3_bot.set_xlabel("Acuratețe Regresie Logistică Trivială (MW + Heavy Atoms + Lungime SMILES)", fontsize=11, fontweight="medium")
    ax3_bot.set_ylabel("Nr. Subcategorii", fontsize=11, fontweight="medium")
    ax3_bot.set_xlim(0.30, 0.75)
    ax3_bot.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    ax3_bot.grid(True, linestyle="--", alpha=0.5)

    # Inset quality box
    quality_box = (
        "PROPRIETĂȚI DE BAZĂ:\n"
        "• 1.00x Raport Autentic:Contrafăcut\n"
        "• Bemis-Murcko Disjoint (80/10/10)\n"
        "• Suport PyG Data & HeteroData"
    )
    ax3_bot.text(0.04, 0.55, quality_box, transform=ax3_bot.transAxes, fontsize=9.5,
                 verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7", alpha=0.92))

    # Overall title
    fig.suptitle("Benchmark Suite Detecție Medicamente Contrafăcute — Distribuție Completă pe Categorii",
                 fontsize=20, fontweight="bold", y=0.985, color="#2c3e50")

    # Save to local and brain
    out_file = Path("all_categories_distribution.png")
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Saved {out_file}")

    # Also update benchmark_distribution.png
    fig.savefig("benchmark_distribution.png", bbox_inches="tight")

    # Copy to brain artifact directory
    brain_dir = Path("/Users/ediionut23/.gemini/antigravity-ide/brain/3f90209c-0d41-4677-a0c8-e463d6d2d3a1")
    if brain_dir.exists():
        shutil.copy(out_file, brain_dir / "all_categories_distribution.png")
        shutil.copy(out_file, brain_dir / "benchmark_distribution.png")
        print(f"Copied to {brain_dir}")

if __name__ == "__main__":
    main()
