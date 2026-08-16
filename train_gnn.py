"""
GNN detector — the C2 "middle tier" — trained on a benchmark dataset
====================================================================

The fingerprint baseline fails on Version E because a bag of local substructures
cannot capture "a plausible drug carrying an unusual combination of edits". A GNN
sees the whole molecular graph and can, in principle, learn that. This trains a
small Graph Isomorphism Network (GIN) on a dataset's own scaffold split and
reports whether it beats the fingerprint floor.

Kept deliberately small so it runs on CPU. Reports (proposal C4 order):
False-Negative Rate, Precision, Recall, F1, plus accuracy and ROC-AUC, on the
scaffold-disjoint test set.

Usage
-----
    python train_gnn.py --dataset evolved --epochs 80
    python train_gnn.py --dataset pooled  --epochs 60
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "H", "other"]
_HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2]


def atom_features(atom) -> List[float]:
    sym = atom.GetSymbol()
    onehot = [float(sym == e) for e in _ELEMENTS[:-1]] + [float(sym not in _ELEMENTS[:-1])]
    hyb = [float(atom.GetHybridization() == h) for h in _HYB]
    return onehot + hyb + [
        atom.GetDegree() / 4.0,
        (atom.GetFormalCharge() + 2) / 4.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]


def smiles_to_data(smiles: str, label: int) -> Optional[Data]:
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in m.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    if not src:  # single atom / no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    def build(split_names):
        out = []
        for r in rows:
            if r["split"] in split_names:
                d = smiles_to_data(r["smiles"], int(r["label"]))
                if d is not None:
                    out.append(d)
        return out
    return build(("train", "val")), build(("test",))


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
    model.eval()
    ys, preds, scores = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = F.softmax(model(batch.x, batch.edge_index, batch.batch), dim=-1)
        scores += out[:, 1].cpu().tolist()
        preds += out.argmax(1).cpu().tolist()
        ys += batch.y.cpu().tolist()
    return metrics(ys, preds, scores), ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evolved")
    ap.add_argument("--root", default="benchmark")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set = load(Path(args.root) / args.dataset / "dataset.csv")
    logger.info(f"{args.dataset}: train+val={len(train_set)}, test={len(test_set)}")
    if len(test_set) < 20:
        logger.error("Test split too small.")
        return

    # hold out 15% of train as val for early stopping
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_set))
    n_val = max(1, int(0.15 * len(train_set)))
    val_set = [train_set[i] for i in idx[:n_val]]
    tr_set = [train_set[i] for i in idx[n_val:]]

    in_dim = train_set[0].x.size(1)
    model = GIN(in_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    tr_loader = DataLoader(tr_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)
    test_loader = DataLoader(test_set, batch_size=128)

    best_val, best_state, patience, bad = -1, None, 15, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
        val_m, _ = evaluate(model, val_loader, device)
        vauc = val_m["roc_auc"] or 0
        if vauc > best_val:
            best_val, best_state, bad = vauc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == args.epochs:
            logger.info(f"  epoch {epoch:>3}: val_auc={vauc:.3f} (best {best_val:.3f})")
        if bad >= patience:
            logger.info(f"  early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    test_m, _ = evaluate(model, test_loader, device)

    logger.info("=" * 60)
    logger.info(f"GNN (GIN) on '{args.dataset}' — scaffold test set")
    for k, v in test_m.items():
        logger.info(f"  {k:<16}: {v}")
    logger.info("=" * 60)
    logger.info("Compare roc_auc to the fingerprint baseline for this dataset.")


if __name__ == "__main__":
    main()
