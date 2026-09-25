"""
Generate High-Quality Visualization for Category-Conditioned Evolved Datasets
=============================================================================
Produces `evolved_categories_comparison.png`:
  Panel A: Molecule Counts & 1:1 Class Balancing
  Panel B: Chemical Plausibility (QED vs SA Scores)
  Panel C: Transformation Complexity (MCS Atoms Edited Depth)
  Panel D: Shortcut Elimination (Trivial Classifier Accuracy ~0.50)
"""

import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

def main():
    with open("version_e_categories/audit_report.json") as f:
        rep = json.load(f)

    with open("benchmark/index.json") as f:
        idx = json.load(f)

    cats = ["bioisostere", "halogen-walk", "homologation", "scaffold-hop", "other"]
    cat_labels = ["Bioisostere", "Halogen-Walk", "Homologation", "Scaffold-Hop", "Other Functional"]
    colors = ["#9b59b6", "#e67e22", "#2ecc71", "#3498db", "#e74c3c"]

    counts_fake = [rep[c]["n_molecules"] for c in cats]
    counts_total = [idx[f"evolved_{c}"]["n_total"] for c in cats]
    qeds = [rep[c]["drug_likeness"]["qed_median"] for c in cats]
    sas = [rep[c]["drug_likeness"]["sa_median"] for c in cats]
    edits_mean = [rep[c]["multi_step_edits"]["mean_atoms_edited"] for c in cats]
    edits_max = [rep[c]["multi_step_edits"]["max_atoms_edited"] for c in cats]
    lr_accs = [idx[f"evolved_{c}"]["shortcut_metrics"]["trivial_lr_accuracy"] for c in cats]

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(18, 13), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # ------------------------------------------------------------------
    # Panel A: Total Molecules & Class Balancing
    # ------------------------------------------------------------------
    x = np.arange(len(cats))
    width = 0.55
    bars = ax0.bar(x, counts_total, width, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(cat_labels, fontsize=11, fontweight="medium")
    ax0.set_ylabel("Total Molecule (1:1 Autentic : Contrafăcut)", fontsize=12, fontweight="medium")
    ax0.set_title("A. Volum Dataseturi Evolutive pe Categorii (1:1 Balanced)", fontsize=14, fontweight="bold", pad=10)
    ax0.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax0.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax0.annotate(f"{int(yval):,} mols\n({int(yval//2):,} fake / {int(yval//2):,} auth)",
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2c3e50")

    # ------------------------------------------------------------------
    # Panel B: Drug-Likeness (QED) vs Synthesizability (SA)
    # ------------------------------------------------------------------
    width_b = 0.35
    b1 = ax1.bar(x - width_b/2, qeds, width_b, label="QED Median (Target ≥ 0.45)", color="#1abc9c", edgecolor="black", linewidth=0.8)
    b2 = ax1.bar(x + width_b/2, [s/10.0 for s in sas], width_b, label="SA Scor / 10 (Target ≤ 0.42)", color="#f39c12", edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontsize=11, fontweight="medium")
    ax1.set_ylabel("Scor Normalizat [0, 1]", fontsize=12, fontweight="medium")
    ax1.set_title("B. Calitate Farmaceutică: QED & Sintezabilitate SA", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for i, (q, s) in enumerate(zip(qeds, sas)):
        ax1.text(x[i] - width_b/2, q + 0.02, f"{q:.2f}", ha="center", fontsize=9.5, fontweight="bold", color="#16a085")
        ax1.text(x[i] + width_b/2, s/10.0 + 0.02, f"SA={s:.2f}", ha="center", fontsize=9.5, fontweight="bold", color="#d35400")

    # ------------------------------------------------------------------
    # Panel C: Multi-Step Edit Complexity (MCS Atoms Changed)
    # ------------------------------------------------------------------
    b3 = ax2.bar(x, edits_mean, width=0.5, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_labels, fontsize=11, fontweight="medium")
    ax2.set_ylabel("Atomi Modificați față de Părinte (MCS)", fontsize=12, fontweight="medium")
    ax2.set_title("C. Complexitate Multi-Etapă (Atomi Modificați Genetic)", fontsize=14, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, max(edits_mean) * 1.35)

    for i, bar in enumerate(b3):
        yval = bar.get_height()
        ax2.annotate(f"Mediu: {yval:.1f} atomi\n(Max: {edits_max[i]} atomi)",
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2c3e50")

    # ------------------------------------------------------------------
    # Panel D: Shortcut Elimination (Trivial Classifier Accuracy)
    # ------------------------------------------------------------------
    bars_d = ax3.bar(x, lr_accs, width=0.5, color="#34495e", alpha=0.85, edgecolor="black", linewidth=0.8)
    ax3.axhline(0.50, color="#e74c3c", linestyle="--", linewidth=2.0, label="Ideal Target = 0.500 (Pură Ghicire / Fără Scurtături)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(cat_labels, fontsize=11, fontweight="medium")
    ax3.set_ylabel("Acuratețe Regresie Logistică Trivială", fontsize=12, fontweight="medium")
    ax3.set_title("D. Rezistență la Scurtături (Cuplare DUD-E pe MW, Atomi, Lungime)", fontsize=14, fontweight="bold", pad=10)
    ax3.set_ylim(0.40, 0.62)
    ax3.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars_d:
        yval = bar.get_height()
        diff = abs(yval - 0.50)
        ax3.annotate(f"{yval:.3f}\n(Δ={diff:.3f})",
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2c3e50")

    fig.suptitle("Evaluarea Dataseturilor Evolutive Condiționate pe Categorii (NSGA-II Anti-Shortcut)",
                 fontsize=17, fontweight="bold", y=0.98, color="#2c3e50")

    out_file = Path("evolved_categories_comparison.png")
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Saved {out_file}")

    brain_dir = Path("/Users/ediionut23/.gemini/antigravity-ide/brain/3f90209c-0d41-4677-a0c8-e463d6d2d3a1")
    if brain_dir.exists():
        shutil.copy(out_file, brain_dir / "evolved_categories_comparison.png")
        print(f"Copied to {brain_dir}")

if __name__ == "__main__":
    main()
