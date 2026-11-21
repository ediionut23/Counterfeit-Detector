"""Audit whether matched generator-transfer pairs share the edited atom's first shell."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_mmp_hard_control_challenge import Components
from exact_edit_context import edit_site_and_shell


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def intervals(differences: np.ndarray, rows: list[dict],
              draws: int, seed: int) -> dict:
    random = np.random.default_rng(seed)
    sampled = random.integers(0, len(differences), size=(draws, len(differences)))
    pair_ci = np.quantile(differences[sampled].mean(axis=1), [0.025, 0.975])
    components = Components()
    for row in rows:
        pair_node = "pair:" + row["pair_id"]
        for scaffold in row["scaffolds"]:
            components.union(pair_node, "scaffold:" + scaffold)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[components.root("pair:" + row["pair_id"])].append(index)
    grouped = list(groups.values())
    size = np.array([len(indices) for indices in grouped])
    totals = np.array([differences[indices].sum() for indices in grouped])
    sampled_groups = random.integers(0, len(grouped), size=(draws, len(grouped)))
    cluster_values = (totals[sampled_groups].sum(axis=1)
                      / size[sampled_groups].sum(axis=1))
    cluster_ci = np.quantile(cluster_values, [0.025, 0.975])
    return {
        "pair_bootstrap_95pct_ci": [round(float(x), 4) for x in pair_ci],
        "scaffold_cluster_bootstrap_95pct_ci": [round(float(x), 4) for x in cluster_ci],
        "scaffold_clusters": len(grouped),
        "largest_scaffold_cluster": int(size.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("At least 100 bootstrap draws required")
    data = read_csv(args.root / "dataset.csv")
    test = [r for r in data if r["split"] == "test"]
    pairs = defaultdict(dict)
    for row in test:
        if row["label"] == "1":
            pairs[row["comparison_pair_id"]][row["generation_method"]] = row
    expected_n = json.loads((args.root / "summary.json").read_text())["matched_test_pairs"]
    if len(pairs) != expected_n or any(set(pair) != {"M", "E"} for pair in pairs.values()):
        raise ValueError("Test pairs do not match manifest")
    local_rows = []
    for pair_id, pair in pairs.items():
        m_site, m_shell = edit_site_and_shell(pair["M"])
        e_site, e_shell = edit_site_and_shell(pair["E"])
        local_rows.append({
            "pair_id": pair_id, "signature": pair["M"]["signature"],
            "same_first_shell_context": int(m_shell == e_shell),
            "M_site": m_site, "E_site": e_site,
            "M_shell": repr(m_shell), "E_shell": repr(e_shell),
            "scaffolds": {pair[method][field] for method in ("M", "E")
                          for field in ("scaffold", "own_scaffold")},
        })
    with (args.root / "local_context_pairs.csv").open("w", newline="") as stream:
        fields = ("pair_id", "signature", "same_first_shell_context", "M_site",
                  "E_site", "M_shell", "E_shell")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in local_rows)
    by_signature = {}
    for signature in sorted({row["signature"] for row in local_rows}):
        sig_rows = [row for row in local_rows if row["signature"] == signature]
        by_signature[signature] = {
            "same_first_shell": sum(row["same_first_shell_context"] for row in sig_rows),
            "pairs": len(sig_rows),
        }
    report = {
        "definition": "Same edited-atom aromaticity, ring status, degree, and sorted first-shell neighbor element/bond/aromaticity/ring descriptors after unique reverse substitution.",
        "verified_unique_reverse_substitutions_and_masks": len(local_rows) * 2,
        "pairs": len(local_rows),
        "same_first_shell_pairs": sum(row["same_first_shell_context"] for row in local_rows),
        "by_signature": by_signature,
        "models": {},
        "caution": "First-shell matching is a conditional sensitivity analysis, not a causal isolation of generator provenance; full molecular contexts still differ.",
    }
    for seed in args.seeds:
        predictions = read_csv(args.root / f"seed{seed}/predictions.csv")
        original = json.loads((args.root / f"seed{seed}/evaluation.json").read_text())
        report["models"][str(seed)] = {}
        for model in ("properties", "rf", "xgboost"):
            rows = [row for row in predictions if row["model"] == model]
            negatives = np.array([float(row["score"]) for row in rows if row["label"] == "0"])
            if len(negatives) != original["test_source"]:
                raise ValueError("Negative pool differs from main evaluation")
            positive = defaultdict(dict)
            for row in rows:
                if row["label"] == "1":
                    positive[row["comparison_pair_id"]][row["method"]] = row
            if set(positive) != set(pairs):
                raise ValueError("Predictions do not cover every test pair")
            sorted_negatives = np.sort(negatives)
            def rank(score: float) -> float:
                lo = np.searchsorted(sorted_negatives, score, side="left")
                hi = np.searchsorted(sorted_negatives, score, side="right")
                return float((lo + hi) / (2 * len(sorted_negatives)))
            all_diffs = np.array([
                rank(float(positive[row["pair_id"]]["E"]["score"]))
                - rank(float(positive[row["pair_id"]]["M"]["score"]))
                for row in local_rows])
            if abs(float(all_diffs.mean()) - original["models"][model]["paired_E_minus_M_auc"]) > 0.0001:
                raise ValueError("AUC decomposition differs from main evaluation")
            report["models"][str(seed)][model] = {}
            for group, selected in (
                ("same_first_shell", [row for row in local_rows if row["same_first_shell_context"]]),
                ("different_first_shell", [row for row in local_rows if not row["same_first_shell_context"]]),
            ):
                if not selected:
                    continue
                m_scores = [float(positive[row["pair_id"]]["M"]["score"]) for row in selected]
                e_scores = [float(positive[row["pair_id"]]["E"]["score"]) for row in selected]
                m_auc = float(np.mean([rank(score) for score in m_scores]))
                e_auc = float(np.mean([rank(score) for score in e_scores]))
                m_tpr = float(np.mean([int(positive[row["pair_id"]]["M"]["predicted"])
                                       for row in selected]))
                e_tpr = float(np.mean([int(positive[row["pair_id"]]["E"]["predicted"])
                                       for row in selected]))
                differences = np.array([rank(e) - rank(m)
                                        for m, e in zip(m_scores, e_scores)])
                report["models"][str(seed)][model][group] = {
                    "n_pairs": len(selected), "auc_M": round(m_auc, 4),
                    "auc_E": round(e_auc, 4),
                    "paired_E_minus_M_auc": round(float(differences.mean()), 4),
                    "tpr_M": round(m_tpr, 4), "tpr_E": round(e_tpr, 4),
                    **intervals(differences, selected, args.bootstrap,
                                20260927 + seed + (0 if group == "same_first_shell" else 1)),
                }
    (args.root / "local_context_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"pairs": report["pairs"],
                      "same_first_shell_pairs": report["same_first_shell_pairs"],
                      "root": str(args.root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
