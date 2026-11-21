"""
Learnability check: can a model tell counterfeit from authentic at all?
=======================================================================

Before any GNN, the first question is whether the task is learnable — i.e.
whether authentic and counterfeit molecules actually differ in a way a model can
pick up, or whether (after property matching) they are effectively
indistinguishable. This trains the C2 "lower-tier" baselines and answers that.

For each dataset it trains on the dataset's own scaffold split (normally
train+val, or train alone with --train-only) and tests on the scaffold-disjoint
test set:

  * property-only baseline  — logistic regression on MW / heavy atoms / logP.
    This SHOULD be near chance (property matching removed those shortcuts); if it
    is, good — it means any real signal must come from structure.
  * fingerprint baseline     — Random Forest on 2048-bit Morgan fingerprints.
    If this beats chance on unseen scaffolds, the local edit is a learnable
    structural signal — the task is real. This is also the C2 fingerprint
    baseline a GNN must beat to justify itself.
  * optional XGBoost         — the second C2 fingerprint baseline, with the
    same fixed split and metrics (enable with --xgboost).

Metrics (proposal C4 order): False Negative Rate first (a missed counterfeit is
the costly error), then Precision, Recall, F1, plus accuracy and ROC-AUC.

Usage
-----
    python train_baseline.py --dataset pooled
    python train_baseline.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def morgan_array(smiles: str) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros((2048,), dtype=np.int8)
    if m is not None:
        DataStructs.ConvertToNumpyArray(
            rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048), arr)
    return arr


def prop_row(smiles: str) -> List[float]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return [0.0, 0.0, 0.0]
    return [Descriptors.MolWt(m), float(m.GetNumHeavyAtoms()), Descriptors.MolLogP(m)]


def load_split(path: Path, train_only: bool = False) -> Dict[str, List[dict]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    train_splits = ("train",) if train_only else ("train", "val")
    train = [r for r in rows if r["split"] in train_splits]
    test = [r for r in rows if r["split"] == "test"]
    val = [r for r in rows if r["split"] == "val"]
    return {"train": train, "val": val, "test": test}


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    # counterfeit = positive class (label 1)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n_pos = int((y_true == 1).sum())
    fnr = fn / n_pos if n_pos else None  # missed counterfeits
    out = {
        "false_negative_rate": round(fnr, 3) if fnr is not None else None,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "accuracy": round(float((y_pred == y_true).mean()), 3),
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 3)
    except Exception:
        out["roc_auc"] = None
    return out


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose a fixed classification threshold using validation data only."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == f1.max())
    return float(thresholds[best[-1]])


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray,
                     target_fpr: float) -> float:
    """Lowest threshold with empirical validation FPR at or below the target."""
    negatives = np.asarray(scores)[np.asarray(y_true) == 0]
    if not len(negatives):
        raise ValueError("FPR threshold needs authentic validation examples")
    allowed = math.floor(target_fpr * len(negatives))
    if allowed >= len(negatives):
        return float("-inf")
    boundary = np.sort(negatives)[::-1][allowed]
    # Advance in the score array's own dtype. A float64 epsilon around a
    # float32 XGBoost score can round back to the tied boundary in comparisons.
    return float(np.nextafter(boundary, np.array(np.inf, dtype=negatives.dtype)))


def metrics_with_fpr(y_true: np.ndarray, y_pred: np.ndarray,
                     y_score: np.ndarray) -> Dict:
    out = metrics(y_true, y_pred, y_score)
    negatives = (y_true == 0)
    out["false_positive_rate"] = round(float((y_pred[negatives] == 1).mean()), 4)
    return out


def train_eval(path: Path, jobs: int = -1, test_path: Path | None = None,
               include_xgboost: bool = False, seed: int = 42,
               train_only: bool = False, tune_threshold: bool = False,
               target_fpr: float | None = None,
               predictions_dir: Path | None = None,
               models_dir: Path | None = None) -> Dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sp = load_split(path, train_only=train_only)
    if test_path is not None:
        with test_path.open(newline="") as stream:
            external = list(csv.DictReader(stream))
            sp["test"] = [row for row in external if row.get("split", "test") == "test"]
    if len(sp["test"]) < 20 or len(sp["train"]) < 40:
        return {}
    if (tune_threshold or target_fpr is not None) and (len(sp["val"]) < 20
            or len({row["label"] for row in sp["val"]}) < 2):
        raise ValueError("Threshold tuning needs both classes in validation")

    ytr = np.array([int(r["label"]) for r in sp["train"]])
    yte = np.array([int(r["label"]) for r in sp["test"]])
    yval = (np.array([int(r["label"]) for r in sp["val"]])
            if tune_threshold or target_fpr is not None else None)

    # --- property-only baseline (should be ~chance after matching) ---
    logger.info(f"  Computing property features for {len(sp['train'])} train and "
                f"{len(sp['test'])} test molecules")
    Xtr_p = np.array([prop_row(r["smiles"]) for r in sp["train"]], float)
    Xte_p = np.array([prop_row(r["smiles"]) for r in sp["test"]], float)
    sc = StandardScaler().fit(Xtr_p)
    lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr_p), ytr)
    p_score = lr.predict_proba(sc.transform(Xte_p))[:, 1]
    prop_m = metrics(yte, (p_score > 0.5).astype(int), p_score)

    # --- fingerprint baseline (the real learnability test) ---
    logger.info("  Computing Morgan fingerprints")
    Xtr_f = np.array([morgan_array(r["smiles"]) for r in sp["train"]])
    Xte_f = np.array([morgan_array(r["smiles"]) for r in sp["test"]])
    logger.info("  Fitting Random Forest")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=jobs, random_state=seed).fit(Xtr_f, ytr)
    f_score = rf.predict_proba(Xte_f)[:, 1]
    fp_m = metrics(yte, (f_score > 0.5).astype(int), f_score)

    result = {
        "n_train": len(sp["train"]), "n_test": len(sp["test"]),
        "seed": seed,
        "property_only": prop_m,
        "fingerprint_rf": fp_m,
    }
    if tune_threshold or target_fpr is not None:
        logger.info("  Choosing decision thresholds on validation")
        Xval_p = np.array([prop_row(r["smiles"]) for r in sp["val"]], float)
        Xval_f = np.array([morgan_array(r["smiles"]) for r in sp["val"]])
        p_val = lr.predict_proba(sc.transform(Xval_p))[:, 1]
        f_val = rf.predict_proba(Xval_f)[:, 1]
        if tune_threshold:
            p_cut = best_f1_threshold(yval, p_val)
            f_cut = best_f1_threshold(yval, f_val)
            result["validation_thresholds"] = {"property_only": round(p_cut, 6),
                                               "fingerprint_rf": round(f_cut, 6)}
            result["validation_tuned_test"] = {
                "property_only": metrics(yte, (p_score >= p_cut).astype(int), p_score),
                "fingerprint_rf": metrics(yte, (f_score >= f_cut).astype(int), f_score),
            }
        if target_fpr is not None:
            result["target_fpr"] = target_fpr
            result["validation_fpr_thresholds"] = {}
            result["validation_fpr_observed"] = {}
            result["validation_fpr_test"] = {}
            for name, val_score, test_score in (
                ("property_only", p_val, p_score),
                ("fingerprint_rf", f_val, f_score),
            ):
                cut = threshold_at_fpr(yval, val_score, target_fpr)
                result["validation_fpr_thresholds"][name] = cut
                result["validation_fpr_observed"][name] = round(
                    float((val_score[yval == 0] >= cut).mean()), 4)
                result["validation_fpr_test"][name] = metrics_with_fpr(
                    yte, (test_score >= cut).astype(int), test_score)
    if include_xgboost:
        from xgboost import XGBClassifier

        logger.info("  Fitting XGBoost")
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=jobs, random_state=seed,
                            eval_metric="logloss")
        xgb.fit(Xtr_f, ytr)
        x_score = xgb.predict_proba(Xte_f)[:, 1]
        result["fingerprint_xgboost"] = metrics(
            yte, (x_score > 0.5).astype(int), x_score)
        if tune_threshold:
            x_val = xgb.predict_proba(Xval_f)[:, 1]
            x_cut = best_f1_threshold(yval, x_val)
            result["validation_thresholds"]["fingerprint_xgboost"] = round(x_cut, 6)
            result["validation_tuned_test"]["fingerprint_xgboost"] = metrics(
                yte, (x_score >= x_cut).astype(int), x_score)
        if target_fpr is not None:
            if not tune_threshold:
                x_val = xgb.predict_proba(Xval_f)[:, 1]
            x_fpr_cut = threshold_at_fpr(yval, x_val, target_fpr)
            result["validation_fpr_thresholds"]["fingerprint_xgboost"] = x_fpr_cut
            result["validation_fpr_observed"]["fingerprint_xgboost"] = round(
                float((x_val[yval == 0] >= x_fpr_cut).mean()), 4)
            result["validation_fpr_test"]["fingerprint_xgboost"] = metrics_with_fpr(
                yte, (x_score >= x_fpr_cut).astype(int), x_score)
    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        score_sets = {"properties": p_score, "rf": f_score}
        if include_xgboost:
            score_sets["xgboost"] = x_score
        for model_name, scores in score_sets.items():
            with (predictions_dir / f"{model_name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "id", "split", "label", "counterfeit_score", "predicted_label"),
                    lineterminator="\n")
                writer.writeheader()
                for row, score in zip(sp["test"], scores):
                    writer.writerow({"id": row["id"], "split": row.get("split", "test"),
                                     "label": row["label"],
                                     "counterfeit_score": float(score),
                                     "predicted_label": int(score > 0.5)})
    if models_dir is not None:
        import joblib

        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": sc, "model": lr},
                    models_dir / "properties.joblib", compress=3)
        joblib.dump(rf, models_dir / "rf.joblib", compress=3)
        if include_xgboost:
            xgb.save_model(models_dir / "xgboost.json")
        (models_dir / "metadata.json").write_text(json.dumps({
            "source_csv": str(path), "external_test_csv": str(test_path) if test_path else None,
            "seed": seed, "train_only": train_only,
            "n_train": len(sp["train"]), "n_test": len(sp["test"]),
            "morgan_radius": 2, "morgan_bits": 2048,
            "property_features": ["molecular_weight", "heavy_atoms", "logp"],
            "models": ["properties", "rf"] + (["xgboost"] if include_xgboost else []),
        }, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="benchmark")
    ap.add_argument("--dataset", default=None, help="single dataset name")
    ap.add_argument("--all", action="store_true", help="run every dataset")
    ap.add_argument("--jobs", type=int, default=-1, help="Random Forest worker count")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--xgboost", action="store_true",
                    help="Also train a 300-tree XGBoost fingerprint baseline")
    ap.add_argument("--train-csv", type=Path, default=None,
                    help="Source CSV; use its test rows unless --test-csv is supplied")
    ap.add_argument("--test-csv", type=Path, default=None,
                    help="External test CSV; only rows marked test are scored")
    ap.add_argument("--train-only", action="store_true",
                    help="Fit on train rows only, leaving val untouched")
    ap.add_argument("--tune-threshold", action="store_true",
                    help="Choose a 0/1 threshold by F1 on the untouched val split")
    ap.add_argument("--target-fpr", type=float, default=None,
                    help="Choose a threshold on val with FPR at or below this target")
    ap.add_argument("--name", default=None,
                    help="Result key when using --train-csv/--test-csv")
    ap.add_argument("--out", default="baseline_results.json")
    ap.add_argument("--predictions-dir", type=Path, default=None,
                    help="Optional directory for per-model test scores; requires one dataset")
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="Optional directory for fitted baseline models; requires one dataset")
    args = ap.parse_args()

    if args.test_csv and not args.train_csv:
        ap.error("--test-csv requires --train-csv")
    if (args.tune_threshold or args.target_fpr is not None) and not args.train_only:
        ap.error("Validation threshold selection requires --train-only")
    if args.target_fpr is not None and not 0 <= args.target_fpr < 1:
        ap.error("--target-fpr must be between 0 (inclusive) and 1 (exclusive)")
    if args.predictions_dir is not None and not (args.train_csv or args.dataset):
        ap.error("--predictions-dir requires a single dataset")
    if args.models_dir is not None and not (args.train_csv or args.dataset):
        ap.error("--models-dir requires a single dataset")
    if args.train_csv:
        paths = [args.train_csv]
    elif args.dataset:
        paths = [Path(args.root) / args.dataset / "dataset.csv"]
    elif args.all:
        paths = sorted(Path(args.root).rglob("dataset.csv"))
    else:
        paths = [Path(args.root) / d / "dataset.csv"
                 for d in ("pooled", "evolved", "category_bioisostere",
                           "category_scaffold-hop", "category_halogen-walk",
                           "category_homologation")]

    results = {}
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Baseline dataset does not exist: {p}")
        if args.train_csv:
            name = args.name or ("cross_version_M_to_E" if args.test_csv else p.stem)
        else:
            name = str(p.parent.relative_to(args.root))
        logger.info(f"Training baselines on {name} ...")
        r = train_eval(p, jobs=args.jobs, test_path=args.test_csv,
                       include_xgboost=args.xgboost, seed=args.seed,
                       train_only=args.train_only,
                       tune_threshold=args.tune_threshold,
                       target_fpr=args.target_fpr,
                       predictions_dir=args.predictions_dir,
                       models_dir=args.models_dir)
        if r:
            if args.train_csv:
                r["train_csv"] = str(p)
                r["test_csv"] = str(args.test_csv or p)
                r["train_only"] = args.train_only
                r["threshold_tuned_on_val"] = args.tune_threshold
            results[name] = r
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    # report
    logger.info("=" * 96)
    logger.info("LEARNABILITY on scaffold-disjoint test set  (FNR = missed counterfeits, lower better)")
    logger.info(f"{'dataset':<30}{'ntest':>6} | {'PROPERTY-ONLY':^22} | {'FINGERPRINT RF':^28}")
    logger.info(f"{'':<30}{'':>6} | {'F1':>6}{'AUC':>7}{'acc':>7} | {'FNR':>6}{'F1':>7}{'AUC':>7}{'acc':>7}")
    logger.info("-" * 96)
    for name, r in results.items():
        p, f = r["property_only"], r["fingerprint_rf"]
        logger.info(f"{name:<30}{r['n_test']:>6} | "
                    f"{p['f1']:>6}{str(p['roc_auc']):>7}{p['accuracy']:>7} | "
                    f"{str(f['false_negative_rate']):>6}{f['f1']:>7}"
                    f"{str(f['roc_auc']):>7}{f['accuracy']:>7}")
    logger.info("=" * 96)
    if args.xgboost:
        for name, r in results.items():
            x = r["fingerprint_xgboost"]
            logger.info(f"XGBoost {name}: FNR={x['false_negative_rate']} "
                        f"F1={x['f1']} AUC={x['roc_auc']} accuracy={x['accuracy']}")
    logger.info("Read: property-only AUC near 0.5 suggests no large shortcut in the "
                "three measured properties. Fingerprint AUC above 0.5 on unseen "
                "scaffolds shows a learnable structural signal; use the realism "
                "audit before attributing that signal to the intended edits.")
    logger.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
