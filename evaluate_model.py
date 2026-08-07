"""
Evaluare metrici HGT pe setul de test (80/20 split, random_state=42).
Afiseaza: Accuracy, F1, Precision, Recall, ROC-AUC, Confusion Matrix.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

from hyp import RobustEnhancedHGTDetector

THRESHOLD = 0.692  # acelasi prag calibrat din hyp.py


def load_model(model_dir="./HGT_Enhanced_Results"):
    model_files = list(Path(model_dir).glob("best_model_f1_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"Niciun model gasit in {model_dir}")
    best = max(model_files, key=lambda p: float(p.stem.split("f1_")[1]))
    print(f"Model incarcat: {best.name}")
    device = torch.device("cpu")
    ckpt = torch.load(best, map_location=device, weights_only=False)
    model = RobustEnhancedHGTDetector(ckpt["feature_dims"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, device


def load_dataset():
    candidates = [
        "./enhanced_pharma_30k_20251001_165724.pt",
        "./enhanced_pharma_v2_20260401_191857.pt",
        "./intelligent_pharma_50k_v2.pt",
        "./intelligent_pharma_50k.pt",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            print(f"Dataset incarcat: {p.name}")
            data = torch.load(p, map_location="cpu", weights_only=False)
            return data[0] if isinstance(data, (list, tuple)) else data
    raise FileNotFoundError("Niciun dataset gasit.")


def run_evaluation():
    model, device = load_model()
    dataset = load_dataset()

    labels = [int(g.y.item()) for g in dataset]
    indices = list(range(len(dataset)))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

    test_data = [dataset[i] for i in test_idx]
    all_probs, all_preds, all_labels = [], [], []

    print(f"\nEvaluare pe {len(test_data)} exemple de test...")
    with torch.no_grad():
        for i, graph in enumerate(test_data):
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(test_data)}...")
            graph = graph.to(device)
            logits = model(graph.x_dict, graph.edge_index_dict, None, batch_size=1)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            pred = int(prob >= THRESHOLD)
            label = int(graph.y.item())
            all_probs.append(prob)
            all_preds.append(pred)
            all_labels.append(label)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds)
    prec   = precision_score(all_labels, all_preds)
    rec    = recall_score(all_labels, all_preds)
    auc    = roc_auc_score(all_labels, all_probs)
    cm     = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 45)
    print("         REZULTATE EVALUARE HGT")
    print("=" * 45)
    print(f"  Threshold clasificare : {THRESHOLD}")
    print(f"  Numar exemple test    : {len(all_labels)}")
    print("-" * 45)
    print(f"  Accuracy              : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score              : {f1:.4f}")
    print(f"  Precision             : {prec:.4f}")
    print(f"  Recall                : {rec:.4f}")
    print(f"  ROC-AUC               : {auc:.4f}")
    print("-" * 45)
    print("  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("=" * 45)
    print("\nRaport detaliat per clasa:")
    print(classification_report(all_labels, all_preds,
                                target_names=["Authentic", "Counterfeit"]))


if __name__ == "__main__":
    run_evaluation()
