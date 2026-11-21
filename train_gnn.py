"""
GNN detector — the C2 "middle tier" — trained on a benchmark dataset
====================================================================

A GNN sees the whole molecular graph and can, in principle, learn combinations
of local edits. This trains small GIN, GCN or GAT classifiers on a dataset's
fixed scaffold split and tests whether they beat the fingerprint baseline on
the same snapshot.

Kept deliberately small so it runs on CPU. Reports (proposal C4 order):
False-Negative Rate, Precision, Recall, F1, plus accuracy and ROC-AUC, on the
scaffold-disjoint test set.

Usage
-----
    python train_gnn.py --dataset evolved --epochs 80
    python train_gnn.py --dataset pooled  --epochs 60
    python train_gnn.py --dataset category_halogen-walk --model gat
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, global_add_pool, global_mean_pool

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "H", "other"]
_HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2]


def atom_features(atom, ring_size_features: bool = False) -> List[float]:
    sym = atom.GetSymbol()
    onehot = [float(sym == e) for e in _ELEMENTS[:-1]] + [float(sym not in _ELEMENTS[:-1])]
    hyb = [float(atom.GetHybridization() == h) for h in _HYB]
    features = onehot + hyb + [
        atom.GetDegree() / 4.0,
        (atom.GetFormalCharge() + 2) / 4.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]
    if ring_size_features:
        features += [float(atom.IsInRingSize(size)) for size in range(3, 9)]
    return features


def smiles_to_data(smiles: str, label: int, group_id: int = 0,
                   ring_size_features: bool = False,
                   sample_weight: float = 1.0) -> Optional[Data]:
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a, ring_size_features) for a in m.GetAtoms()],
                     dtype=torch.float)
    src, dst = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    if not src:  # single atom / no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long),
                group_id=torch.tensor([group_id], dtype=torch.long),
                sample_weight=torch.tensor([sample_weight], dtype=torch.float))


def load(path: Path, ring_size_features: bool = False):
    rows = list(csv.DictReader(open(path)))

    def group_name(row):
        if int(row["label"]) == 0:
            return "authentic"
        category = row.get("balancing_category") or row.get("category", "")
        subcategory = row.get("balancing_subcategory") or row.get("subcategory", "")
        return f"{row.get('generation_method', '')}/{category}/{subcategory}"

    group_names = sorted({
        group_name(r) for r in rows
    })
    group_ids = {name: i for i, name in enumerate(group_names)}

    def build(split_names):
        out = []
        for r in rows:
            if r["split"] in split_names:
                label = int(r["label"])
                d = smiles_to_data(r["smiles"], int(r["label"]),
                                   group_id=group_ids[group_name(r)],
                                   ring_size_features=ring_size_features,
                                   sample_weight=float(r.get("sample_weight", 1.0) or 1.0))
                if d is not None:
                    out.append(d)
        return out
    return build(("train",)), build(("val",)), build(("test",))


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            nn = Sequential(Linear(d, hidden), BatchNorm1d(hidden), ReLU(),
                            Linear(hidden, hidden), ReLU())
            self.convs.append(GINConv(nn))
            d = hidden
        self.head = Sequential(Linear(hidden * 2, hidden), ReLU(),
                               torch.nn.Dropout(0.3), Linear(hidden, 2))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
        h = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        return self.head(h)


class MessagePassingClassifier(torch.nn.Module):
    """GCN, GAT, or GATv2 with the same graph readout and head as GIN."""

    def __init__(self, architecture: str, in_dim: int, hidden: int = 64,
                 layers: int = 3):
        super().__init__()
        if architecture not in ("gcn", "gat", "gat2"):
            raise ValueError(f"Unknown architecture: {architecture}")
        if architecture in ("gat", "gat2") and hidden % 4:
            raise ValueError("GAT/GATv2 hidden size must be divisible by four")
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            if architecture == "gcn":
                conv = GCNConv(d, hidden)
            elif architecture == "gat":
                conv = GATConv(d, hidden // 4, heads=4, dropout=0.1)
            else:
                conv = GATv2Conv(d, hidden // 4, heads=4, dropout=0.1)
            self.convs.append(conv)
            self.norms.append(BatchNorm1d(hidden))
            d = hidden
        self.head = Sequential(Linear(hidden * 2, hidden), ReLU(),
                               torch.nn.Dropout(0.3), Linear(hidden, 2))

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
        h = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        return self.head(h)


def metrics(y, pred, score):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    y, pred, score = np.array(y), np.array(pred), np.array(score)
    fn = int(((pred == 0) & (y == 1)).sum()); npos = int((y == 1).sum())
    out = {
        "fnr": round(fn / npos, 3) if npos else None,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
        "accuracy": round(float((pred == y).mean()), 3),
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y, score)), 3)
    except Exception:
        out["roc_auc"] = None
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import roc_auc_score

    model.eval()
    ys, preds, scores = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = F.softmax(model(batch.x, batch.edge_index, batch.batch), dim=-1)
        scores += out[:, 1].cpu().tolist()
        preds += out.argmax(1).cpu().tolist()
        ys += batch.y.cpu().tolist()
    report = metrics(ys, preds, scores)
    raw_auc = float(roc_auc_score(ys, scores)) if len(set(ys)) > 1 else 0.0
    return report, raw_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evolved")
    ap.add_argument("--root", default="benchmark")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--model", choices=("gin", "gcn", "gat", "gat2"), default="gin",
                    help="gat2 selects PyG GATv2Conv (often called GATv2)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ring-size-features", action="store_true",
                    help="Append one-hot ring membership for sizes 3 through 8 to each atom")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2,
                    help="Fixed PyTorch CPU thread count for reproducible runs")
    ap.add_argument("--class-weighting", choices=("balanced", "none"),
                    default="balanced",
                    help="Use inverse-frequency training loss weights; has no effect on a 1:1 train split")
    ap.add_argument("--subcategory-sampling", choices=("balanced", "none"),
                    default="balanced",
                    help="Sample authentic and each counterfeit subcategory with equal total probability")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON file for test metrics and run settings")
    ap.add_argument("--weights-out", type=Path, default=None,
                    help="Save selected model weights and reconstruction settings")
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error("--epochs must be at least 1")
    if args.hidden < 4 or (args.model in ("gat", "gat2") and args.hidden % 4):
        ap.error("--hidden must be at least 4 and divisible by 4 for GAT/GATv2")
    if args.threads < 1:
        ap.error("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)

    tr_set, val_set, test_set = load(Path(args.root) / args.dataset / "dataset.csv",
                                   ring_size_features=args.ring_size_features)
    logger.info(f"{args.dataset}: train={len(tr_set)}, val={len(val_set)}, test={len(test_set)}")
    if len(test_set) < 20 or len(val_set) < 10:
        logger.error("Validation or test split too small.")
        return

    train_counts = torch.bincount(
        torch.cat([graph.y.reshape(-1) for graph in tr_set]), minlength=2
    ).to(dtype=torch.float)
    if torch.any(train_counts == 0):
        logger.error("Training split must contain both classes.")
        return
    class_weights = None
    if args.class_weighting == "balanced":
        # sklearn's balanced heuristic: n_samples / (n_classes * class_count).
        # For a balanced dataset both weights are exactly 1, preserving the
        # previous cross-entropy objective.
        class_weights = train_counts.sum() / (2.0 * train_counts)
        class_weights = class_weights.to(device)
    logger.info(
        "Train label counts: authentic=%d, counterfeit=%d; loss weights=%s",
        int(train_counts[0]), int(train_counts[1]),
        class_weights.detach().cpu().tolist() if class_weights is not None else "none",
    )

    group_ids = torch.cat([graph.group_id.reshape(-1) for graph in tr_set])
    train_labels = torch.cat([graph.y.reshape(-1) for graph in tr_set])
    has_sample_weights = any(
        not torch.allclose(graph.sample_weight, torch.ones_like(graph.sample_weight))
        for graph in tr_set
    )
    effective_subcategory_sampling = (
        "sample_weight_column" if has_sample_weights and args.subcategory_sampling == "balanced"
        else args.subcategory_sampling
    )
    fake_group_counts = Counter(
        int(group_id) for group_id, label in zip(group_ids.tolist(), train_labels.tolist())
        if label == 1
    )
    sampler = None
    if has_sample_weights and args.subcategory_sampling == "balanced":
        logger.info("Using per-row sample_weight values; disabling the balanced sampler to avoid double subcategory weighting.")
    elif args.subcategory_sampling == "balanced":
        authentic_count = int((train_labels == 0).sum())
        n_fake_groups = len(fake_group_counts)
        if authentic_count == 0 or n_fake_groups == 0:
            logger.error("Balanced subcategory sampling needs authentic and counterfeit groups.")
            return
        weights = []
        for group_id, label in zip(group_ids.tolist(), train_labels.tolist()):
            if label == 0:
                weights.append(0.5 / authentic_count)
            else:
                weights.append(0.5 / (n_fake_groups * fake_group_counts[group_id]))
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double), num_samples=len(tr_set),
            replacement=True, generator=torch.Generator().manual_seed(args.seed),
        )
        logger.info("Balanced sampler: authentic group plus %d counterfeit subcategories", n_fake_groups)

    in_dim = tr_set[0].x.size(1)
    model = (GIN(in_dim, hidden=args.hidden) if args.model == "gin" else
             MessagePassingClassifier(args.model, in_dim, hidden=args.hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    tr_loader = DataLoader(tr_set, batch_size=64,
                           sampler=sampler,
                           shuffle=(sampler is None))
    val_loader = DataLoader(val_set, batch_size=128)
    test_loader = DataLoader(test_set, batch_size=128)

    best_val, best_state, patience, bad = -1, None, 15, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            per_graph_loss = F.cross_entropy(out, batch.y, weight=class_weights, reduction="none")
            sample_weights = batch.sample_weight.reshape(-1)
            loss = (per_graph_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        _, vauc = evaluate(model, val_loader, device)
        if vauc > best_val:
            best_val, best_state, bad = vauc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        logger.info(f"  epoch {epoch:>3}: train_loss={mean_loss:.4f} val_auc={vauc:.3f} (best {best_val:.3f})")
        if bad >= patience:
            logger.info(f"  early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    test_m, _ = evaluate(model, test_loader, device)

    logger.info("=" * 60)
    logger.info(f"GNN ({args.model.upper()}) on '{args.dataset}' — scaffold test set")
    for k, v in test_m.items():
        logger.info(f"  {k:<16}: {v}")
    logger.info("=" * 60)
    logger.info("Compare roc_auc to the fingerprint baseline for this dataset.")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "dataset": args.dataset, "model": args.model, "seed": args.seed,
            "hidden": args.hidden, "learning_rate": args.lr,
            "ring_size_features": args.ring_size_features,
            "class_weighting": args.class_weighting,
            "subcategory_sampling": effective_subcategory_sampling,
            "sample_weight_column_used": has_sample_weights,
            "n_counterfeit_subcategories": len(fake_group_counts),
            "train_label_counts": train_counts.to(dtype=torch.long).tolist(),
            "class_weights": (class_weights.detach().cpu().tolist()
                              if class_weights is not None else None),
            "threads": args.threads,
            "epochs_run": epoch, "best_val_auc": best_val,
            "n_train": len(tr_set), "n_val": len(val_set),
            "n_test": len(test_set), "test_metrics": test_m,
        }, indent=2))
    if args.weights_out is not None:
        args.weights_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1,
            "dataset": args.dataset,
            "model": args.model,
            "in_dim": in_dim,
            "hidden": args.hidden,
            "layers": 3,
            "ring_size_features": args.ring_size_features,
            "class_weighting": args.class_weighting,
            "subcategory_sampling": effective_subcategory_sampling,
            "sample_weight_column_used": has_sample_weights,
            "n_counterfeit_subcategories": len(fake_group_counts),
            "train_label_counts": train_counts.to(dtype=torch.long).tolist(),
            "class_weights": (class_weights.detach().cpu().tolist()
                              if class_weights is not None else None),
            "threads": args.threads,
            "seed": args.seed,
            "best_val_auc": best_val,
            "epochs_run": epoch,
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        }, args.weights_out)
        logger.info(f"Saved model weights -> {args.weights_out}")


if __name__ == "__main__":
    main()
