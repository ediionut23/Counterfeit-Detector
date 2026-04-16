"""
Homogeneous GNN Training Script - WITH DATA INTEGRITY CHECKS
Comparable baseline for HGT heterogeneous model
Author: For ICTAI paper comparison
Version: 2.0.0

Changes from v1:
  - Deduplicates SMILES before splitting
  - Removes conflicting labels (same SMILES, different label)
  - Family-aware train/test split (parent + all children stay together)
  - Reports data quality metrics before training
  - Logs artifact indicators (length bias, trivial classifier accuracy)
  - Drops node types not in ATOM_TYPES (Se, Si, Ca, etc. that leak class info)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

RESULTS_DIR = './Homo_GNN_Results_Clean'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Node types from HGT model — order matters for one-hot encoding
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
ATOM_TYPE_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)


# ═════════════════════════════════════════════
#  DATA INTEGRITY PIPELINE
# ═════════════════════════════════════════════

class DataIntegrityChecker:
    """
    Runs before any training to clean and validate the dataset.
    Produces a report and returns clean data with family-aware splits.
    """

    def __init__(self, graphs, labels):
        self.original_graphs = graphs
        self.original_labels = labels
        self.report_lines = []

    def _log(self, msg):
        print(msg)
        self.report_lines.append(msg)

    def run_full_pipeline(self):
        """
        Returns (clean_graphs, clean_labels, train_idx, test_idx)
        with all integrity issues resolved.
        """
        self._log("=" * 65)
        self._log("  DATA INTEGRITY CHECK")
        self._log("=" * 65)

        n_original = len(self.original_graphs)
        self._log(f"\nOriginal dataset: {n_original} graphs")

        # ── Step 1: Remove graphs without SMILES ──
        graphs, labels = self._filter_has_smiles()

        # ── Step 2: Remove conflicting labels ──
        graphs, labels = self._remove_conflicting_labels(graphs, labels)

        # ── Step 3: Deduplicate (keep first occurrence) ──
        graphs, labels = self._deduplicate(graphs, labels)

        # ── Step 4: Report class balance ──
        self._report_class_balance(labels)

        # ── Step 5: Artifact indicators ──
        self._report_artifact_indicators(graphs, labels)

        # ── Step 6: Family-aware split ──
        train_idx, test_idx = self._family_aware_split(graphs, labels)

        # ── Step 7: Verify no leakage in final split ──
        self._verify_split(graphs, labels, train_idx, test_idx)

        # ── Save report ──
        report_path = Path(RESULTS_DIR) / 'data_integrity_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report_lines))
        print(f"\nFull report saved: {report_path.absolute()}")

        return graphs, labels, train_idx, test_idx

    # ──────────────────────────────────────────

    def _filter_has_smiles(self):
        kept_g, kept_l = [], []
        dropped = 0
        for g, lbl in zip(self.original_graphs, self.original_labels):
            if hasattr(g, 'smiles') and g.smiles:
                kept_g.append(g)
                kept_l.append(int(lbl) if hasattr(lbl, 'item') else int(lbl))
            else:
                dropped += 1
        self._log(f"\n[Step 1] Filter missing SMILES: dropped {dropped}, kept {len(kept_g)}")
        return kept_g, kept_l

    def _remove_conflicting_labels(self, graphs, labels):
        """Remove SMILES that appear with BOTH label 0 and label 1."""
        smiles_to_labels = defaultdict(set)
        for g, lbl in zip(graphs, labels):
            smiles_to_labels[g.smiles].add(lbl)

        conflicting = {s for s, lbls in smiles_to_labels.items() if len(lbls) > 1}

        if conflicting:
            self._log(f"\n[Step 2] Conflicting labels: {len(conflicting)} SMILES have both 0 and 1")
            self._log(f"         Removing ALL instances of these SMILES")
            kept_g, kept_l = [], []
            for g, lbl in zip(graphs, labels):
                if g.smiles not in conflicting:
                    kept_g.append(g)
                    kept_l.append(lbl)
            self._log(f"         {len(graphs)} → {len(kept_g)} graphs")
            return kept_g, kept_l
        else:
            self._log(f"\n[Step 2] Conflicting labels: none found ✓")
            return graphs, labels

    def _deduplicate(self, graphs, labels):
        """Keep only the first occurrence of each SMILES."""
        seen = set()
        kept_g, kept_l = [], []
        dup_count = 0
        for g, lbl in zip(graphs, labels):
            if g.smiles not in seen:
                seen.add(g.smiles)
                kept_g.append(g)
                kept_l.append(lbl)
            else:
                dup_count += 1

        self._log(f"\n[Step 3] Deduplication: removed {dup_count} duplicates")
        self._log(f"         {len(graphs)} → {len(kept_g)} unique graphs")
        return kept_g, kept_l

    def _report_class_balance(self, labels):
        labels_np = np.array(labels)
        unique, counts = np.unique(labels_np, return_counts=True)
        self._log(f"\n[Step 4] Class balance after cleaning:")
        for lbl, cnt in zip(unique, counts):
            name = "Authentic" if lbl == 0 else "Counterfeit"
            self._log(f"         {name}: {cnt} ({cnt / len(labels_np) * 100:.1f}%)")
        ratio = counts.min() / counts.max() if len(counts) == 2 else 0
        self._log(f"         Balance ratio: {ratio:.3f}")

    def _report_artifact_indicators(self, graphs, labels):
        """Quick shortcut check so we know what to expect from training."""
        self._log(f"\n[Step 5] Artifact indicators:")

        # SMILES length
        auth_len = [len(g.smiles) for g, l in zip(graphs, labels) if l == 0]
        fake_len = [len(g.smiles) for g, l in zip(graphs, labels) if l == 1]
        if auth_len and fake_len:
            a_m, f_m = np.mean(auth_len), np.mean(fake_len)
            a_s, f_s = np.std(auth_len), np.std(fake_len)
            pooled = np.sqrt((a_s ** 2 + f_s ** 2) / 2)
            d = abs(a_m - f_m) / max(pooled, 1e-6)
            self._log(f"         SMILES length — auth: {a_m:.1f}±{a_s:.1f}, "
                       f"fake: {f_m:.1f}±{f_s:.1f}, Cohen's d: {d:.3f}")
            if d > 0.5:
                self._log(f"         ⚠  Length bias present (d>{0.5})")

        # Trivial logistic regression
        try:
            features, valid_labels = [], []
            for g, lbl in zip(graphs, labels):
                total_nodes = sum(
                    g[nt].x.size(0) for nt in g.node_types
                    if hasattr(g[nt], 'x') and g[nt].x.size(0) > 0
                )
                total_edges = sum(
                    g[et].edge_index.size(1) for et in g.edge_types
                    if hasattr(g[et], 'edge_index')
                )
                features.append([total_nodes, total_edges, len(g.smiles)])
                valid_labels.append(lbl)

            X = np.array(features)
            y = np.array(valid_labels)
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                                      stratify=y, random_state=99)
            sc = StandardScaler()
            lr = LogisticRegression(max_iter=500, random_state=99)
            lr.fit(sc.fit_transform(X[tr]), y[tr])
            trivial_acc = lr.score(sc.transform(X[te]), y[te])
            self._log(f"         Trivial LogReg (nodes, edges, len) accuracy: {trivial_acc:.3f}")
            if trivial_acc > 0.65:
                self._log(f"         ⚠  Shortcuts exist — expect inflated GNN scores")
        except Exception as e:
            self._log(f"         Trivial classifier skipped: {e}")

    def _family_aware_split(self, graphs, labels, test_size=0.2):
        """
        Group each authentic molecule with all counterfeits derived from it.
        Split by FAMILY so no parent-child pair spans train and test.
        """
        self._log(f"\n[Step 6] Family-aware train/test split:")

        # Build families: parent SMILES -> list of dataset indices
        families = defaultdict(list)
        for idx, g in enumerate(graphs):
            parent = getattr(g, 'original_smiles', None)
            if parent:
                families[parent].append(idx)
            else:
                # Authentic molecules or molecules without parent info
                # Use own SMILES as family key
                families[g.smiles].append(idx)

        family_keys = list(families.keys())
        self._log(f"         Total families: {len(family_keys)}")

        # Assign a majority label to each family for stratification
        family_labels = []
        for key in family_keys:
            member_labels = [labels[i] for i in families[key]]
            majority = 1 if sum(member_labels) > len(member_labels) / 2 else 0
            family_labels.append(majority)

        # Split families
        try:
            train_fam_idx, test_fam_idx = train_test_split(
                np.arange(len(family_keys)), test_size=test_size,
                stratify=family_labels, random_state=42
            )
        except ValueError:
            # If stratification fails (too few of one class), fall back
            self._log("         ⚠  Stratification failed, using non-stratified split")
            train_fam_idx, test_fam_idx = train_test_split(
                np.arange(len(family_keys)), test_size=test_size, random_state=42
            )

        # Expand family indices to molecule indices
        train_idx = []
        test_idx = []
        for fi in train_fam_idx:
            train_idx.extend(families[family_keys[fi]])
        for fi in test_fam_idx:
            test_idx.extend(families[family_keys[fi]])

        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        self._log(f"         Train: {len(train_idx)} graphs "
                   f"(auth={sum(1 for l in train_labels if l==0)}, "
                   f"fake={sum(1 for l in train_labels if l==1)})")
        self._log(f"         Test:  {len(test_idx)} graphs "
                   f"(auth={sum(1 for l in test_labels if l==0)}, "
                   f"fake={sum(1 for l in test_labels if l==1)})")

        return train_idx, test_idx

    def _verify_split(self, graphs, labels, train_idx, test_idx):
        """Final verification: zero SMILES overlap between train and test."""
        train_smiles = {graphs[i].smiles for i in train_idx}
        test_smiles = {graphs[i].smiles for i in test_idx}
        overlap = train_smiles & test_smiles

        self._log(f"\n[Step 7] Split verification:")
        self._log(f"         Train unique SMILES: {len(train_smiles)}")
        self._log(f"         Test unique SMILES:  {len(test_smiles)}")
        self._log(f"         Overlap:             {len(overlap)}")

        if overlap:
            self._log(f"         ⚠  LEAKAGE STILL PRESENT — {len(overlap)} shared SMILES")
        else:
            self._log(f"         ✓  Zero overlap — clean split confirmed")

        # Also check parent-child across split
        train_smiles_set = {graphs[i].smiles for i in train_idx}
        test_parent_set = set()
        for i in test_idx:
            parent = getattr(graphs[i], 'original_smiles', None)
            if parent:
                test_parent_set.add(parent)

        parent_leak = train_smiles_set & test_parent_set
        self._log(f"         Parent-child leakage: {len(parent_leak)}")
        if parent_leak:
            self._log(f"         ⚠  {len(parent_leak)} test counterfeits derived from train authentics")
        else:
            self._log(f"         ✓  No parent-child leakage")


# ═════════════════════════════════════════════
#  HETERO -> HOMO CONVERSION
# ═════════════════════════════════════════════

def hetero_to_homo(hetero_graph: HeteroData, feature_dims: dict) -> Data:
    """
    Convert a HeteroData graph to a homogeneous Data graph.
    Only includes node types in ATOM_TYPES — drops exotic types
    like Se, Si, Ca that were shown to leak class information.
    """
    base_dim = max(feature_dims.values()) if feature_dims else 26
    total_feature_dim = base_dim + NUM_ATOM_TYPES

    all_x = []
    node_offsets = {}
    current_offset = 0

    for atom_type in ATOM_TYPES:
        if atom_type in hetero_graph.node_types and \
                hasattr(hetero_graph[atom_type], 'x') and \
                hetero_graph[atom_type].x.size(0) > 0:

            x = hetero_graph[atom_type].x.float()
            n_nodes = x.size(0)
            feat_dim = x.size(1)

            if feat_dim < base_dim:
                x = F.pad(x, (0, base_dim - feat_dim))
            elif feat_dim > base_dim:
                x = x[:, :base_dim]

            one_hot = torch.zeros(n_nodes, NUM_ATOM_TYPES)
            idx = ATOM_TYPE_TO_IDX.get(atom_type, 0)
            one_hot[:, idx] = 1.0

            x_full = torch.cat([x, one_hot], dim=1)
            all_x.append(x_full)
            node_offsets[atom_type] = current_offset
            current_offset += n_nodes

    if not all_x:
        return None

    x_combined = torch.cat(all_x, dim=0)

    # Merge edges — only for node types we kept
    all_edges = []
    all_edge_attrs = []

    for edge_type in hetero_graph.edge_types:
        src_type, rel_type, dst_type = edge_type

        if src_type not in node_offsets or dst_type not in node_offsets:
            continue
        if not hasattr(hetero_graph[edge_type], 'edge_index'):
            continue

        edge_index = hetero_graph[edge_type].edge_index
        if edge_index.size(1) == 0:
            continue

        src_offset = node_offsets[src_type]
        dst_offset = node_offsets[dst_type]

        remapped = edge_index.clone()
        remapped[0] += src_offset
        remapped[1] += dst_offset

        all_edges.append(remapped)

        if hasattr(hetero_graph[edge_type], 'edge_attr') and \
                hetero_graph[edge_type].edge_attr.size(0) == edge_index.size(1):
            all_edge_attrs.append(hetero_graph[edge_type].edge_attr.float())

    if not all_edges:
        return None

    edge_index_combined = torch.cat(all_edges, dim=1)
    homo = Data(x=x_combined, edge_index=edge_index_combined)

    if len(all_edge_attrs) == len(all_edges) and all_edge_attrs:
        homo.edge_attr = torch.cat(all_edge_attrs, dim=0)

    for attr in ['y', 'smiles', 'source_id', 'source', 'counterfeit_type',
                 'difficulty', 'difficulty_level', 'original_smiles']:
        if hasattr(hetero_graph, attr):
            setattr(homo, attr, getattr(hetero_graph, attr))

    return homo


def convert_dataset(hetero_graphs, labels, feature_dims):
    homo_graphs = []
    homo_labels = []

    print("Converting heterogeneous → homogeneous graphs...")
    for i, (hg, label) in enumerate(tqdm(zip(hetero_graphs, labels),
                                          total=len(hetero_graphs))):
        try:
            hg_homo = hetero_to_homo(hg, feature_dims)
            if hg_homo is not None and hg_homo.x.size(0) > 0 and \
                    hg_homo.edge_index.size(1) > 0:
                homo_graphs.append(hg_homo)
                homo_labels.append(label)
        except Exception as e:
            if i < 5:
                print(f"  Skipping graph {i}: {e}")

    print(f"Converted: {len(homo_graphs)} valid homogeneous graphs")
    return homo_graphs, homo_labels


# ═════════════════════════════════════════════
#  FOCAL LOSS
# ═════════════════════════════════════════════

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ═════════════════════════════════════════════
#  HOMOGENEOUS GAT MODEL (unchanged architecture)
# ═════════════════════════════════════════════

class HomoGATDetector(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=192, out_channels=2,
                 num_heads=8, num_layers=3, dropout=0.15):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        self.input_norm = torch.nn.LayerNorm(hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=num_heads,
                        dropout=dropout, concat=False)
            )
            self.norms.append(torch.nn.LayerNorm(hidden_channels))

        clf_in = hidden_channels * 2
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(clf_in, hidden_channels * 2),
            torch.nn.BatchNorm1d(hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.7),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_norm(self.input_proj(x)))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        return self.classifier(h_graph)


# ═════════════════════════════════════════════
#  TRAINER (unchanged)
# ═════════════════════════════════════════════

class HomoTrainer:
    def __init__(self, model, device, save_path=RESULTS_DIR):
        self.model = model
        self.device = device
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)

        log_file = self.save_path / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.best_f1 = 0
        self.best_model_state = None
        self.train_losses = []
        self.train_f1_scores = []
        self.test_losses = []
        self.test_f1_scores = []
        self.test_accuracies = []

    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        predictions, true_labels = [], []

        for batch_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
            try:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.view(-1)
                if out.size(0) != labels.size(0):
                    s = min(out.size(0), labels.size(0))
                    out, labels = out[:s], labels[:s]
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                predictions.extend(out.argmax(dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            except Exception as e:
                self.logger.warning(f"Batch {batch_idx} error: {e}")
                continue

        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss, predictions, true_labels

    def evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                try:
                    batch = batch.to(self.device)
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    labels = batch.y.view(-1)
                    if out.size(0) != labels.size(0):
                        s = min(out.size(0), labels.size(0))
                        out, labels = out[:s], labels[:s]
                    loss = criterion(out, labels)
                    total_loss += loss.item()
                    predictions.extend(out.argmax(dim=1).cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                except:
                    continue

        if not predictions:
            return {'loss': float('inf'), 'accuracy': 0, 'f1': 0,
                    'precision': 0, 'recall': 0}, [], []

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0)

        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1,
                'precision': precision, 'recall': recall}, predictions, true_labels

    def train(self, train_loader, test_loader, criterion, optimizer,
              num_epochs, patience=20, scheduler=None):

        total_params = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad)
        print(f"Starting training — {total_params:,} parameters")
        self.logger.info(f"Training started: {num_epochs} epochs, patience={patience}")

        best_epoch = 0
        patience_counter = 0
        final_preds, final_labels = [], []

        for epoch in range(num_epochs):
            train_loss, train_preds, train_lbls = self.train_epoch(
                train_loader, optimizer, criterion)
            test_metrics, test_preds, test_lbls = self.evaluate(
                test_loader, criterion)

            train_f1 = 0.0
            if train_preds:
                _, _, train_f1, _ = precision_recall_fscore_support(
                    train_lbls, train_preds, average='binary', zero_division=0)

            self.train_losses.append(train_loss)
            self.train_f1_scores.append(train_f1)
            self.test_losses.append(test_metrics['loss'])
            self.test_f1_scores.append(test_metrics['f1'])
            self.test_accuracies.append(test_metrics['accuracy'])

            current_f1 = test_metrics['f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_model_state = {k: v.clone()
                                          for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
                final_preds = test_preds
                final_labels = test_lbls

                ckpt = self.save_path / f'best_model_f1_{current_f1:.4f}.pt'
                torch.save({'model_state_dict': self.best_model_state,
                            'f1_score': self.best_f1,
                            'epoch': epoch + 1}, ckpt)
                self.logger.info(f"New best model: F1={current_f1:.4f}")
            else:
                patience_counter += 1

            if scheduler:
                if isinstance(scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_f1)
                else:
                    scheduler.step()

            msg = (f"Epoch {epoch+1}/{num_epochs} | "
                   f"Train Loss={train_loss:.4f} F1={train_f1:.4f} | "
                   f"Test Loss={test_metrics['loss']:.4f} "
                   f"Acc={test_metrics['accuracy']:.4f} "
                   f"F1={current_f1:.4f} "
                   f"P={test_metrics['precision']:.4f} "
                   f"R={test_metrics['recall']:.4f}")
            print(msg)
            self.logger.info(msg)

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        self._create_plots(final_preds, final_labels)

        return {'best_f1': self.best_f1, 'best_epoch': best_epoch + 1,
                'final_preds': final_preds, 'final_labels': final_labels}

    def _create_plots(self, final_preds, final_labels):
        epochs = range(1, len(self.train_losses) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', lw=2)
        axes[0, 0].plot(epochs, self.test_losses, 'r-', label='Test', lw=2)
        axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, self.train_f1_scores, 'b-', label='Train F1', lw=2)
        axes[0, 1].plot(epochs, self.test_f1_scores, 'r-', label='Test F1', lw=2)
        axes[0, 1].axhline(y=self.best_f1, color='g', ls='--',
                           label=f'Best={self.best_f1:.3f}', lw=2)
        axes[0, 1].set_title('F1 Score'); axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(epochs, self.test_accuracies, 'g-', lw=2)
        axes[0, 2].set_title('Test Accuracy'); axes[0, 2].set_ylim([0, 1])
        axes[0, 2].grid(True, alpha=0.3)

        if final_preds and final_labels:
            cm = confusion_matrix(final_labels, final_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                        xticklabels=['Auth', 'CF'], yticklabels=['Auth', 'CF'])
            axes[1, 0].set_title('Confusion Matrix')
            axes[1, 0].set_ylabel('True'); axes[1, 0].set_xlabel('Predicted')

            cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1],
                        xticklabels=['Auth', 'CF'], yticklabels=['Auth', 'CF'])
            axes[1, 1].set_title('Normalized CM')
            axes[1, 1].set_ylabel('True'); axes[1, 1].set_xlabel('Predicted')

        axes[1, 2].plot(epochs,
                        [max(0, f - l) for f, l in
                         zip(self.test_f1_scores, self.test_losses)],
                        'm-', lw=2)
        axes[1, 2].set_title('F1 - Loss'); axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Homo GAT — CLEAN DATA', fontsize=16, fontweight='bold', y=1.01)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.save_path / f'training_results_{ts}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path.absolute()}")
        plt.show()


# ═════════════════════════════════════════════
#  DATASET LOADING
# ═════════════════════════════════════════════

def load_hetero_dataset(file_path):
    fp = Path(file_path)
    if not fp.exists():
        print(f"File not found: {fp.absolute()}")
        return None, None, None

    try:
        loaded = torch.load(fp, map_location='cpu')
    except Exception:
        loaded = torch.load(fp, weights_only=False, map_location='cpu')

    if isinstance(loaded, tuple) and len(loaded) >= 3:
        hetero_graphs, labels, metadata = loaded[0], loaded[1], loaded[2]
    elif isinstance(loaded, dict):
        hetero_graphs = loaded.get('hetero_graphs', loaded.get('graphs'))
        labels = loaded.get('labels')
        metadata = loaded.get('metadata')
    else:
        raise TypeError(f"Unknown format: {type(loaded)}")

    print(f"Loaded {len(hetero_graphs)} molecules")

    feature_dims = {}
    for graph in hetero_graphs[:50]:
        for nt in graph.node_types:
            if hasattr(graph[nt], 'x') and graph[nt].x.size(0) > 0:
                d = graph[nt].x.shape[-1]
                if nt not in feature_dims:
                    feature_dims[nt] = d
                else:
                    feature_dims[nt] = max(feature_dims[nt], d)

    for nt in ATOM_TYPES:
        if nt not in feature_dims:
            feature_dims[nt] = max(feature_dims.values()) if feature_dims else 26

    return hetero_graphs, labels, feature_dims


# ═════════════════════════════════════════════
#  MAIN TRAINING (with integrity pipeline)
# ═════════════════════════════════════════════

def run_homo_training(dataset_path,
                      hidden_channels=192,
                      num_heads=8,
                      num_layers=3,
                      dropout=0.15,
                      lr=0.0003,
                      weight_decay=0.01,
                      num_epochs=80,
                      patience=15,
                      batch_size=None):

    print("=" * 65)
    print("  HOMOGENEOUS GAT TRAINING — WITH DATA INTEGRITY CHECKS")
    print("=" * 65)

    hetero_graphs, labels, feature_dims = load_hetero_dataset(dataset_path)
    if hetero_graphs is None:
        print("Failed to load dataset.")
        return None

    # ── DATA INTEGRITY PIPELINE ──
    checker = DataIntegrityChecker(hetero_graphs, labels)
    clean_graphs, clean_labels, train_idx, test_idx = checker.run_full_pipeline()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Convert to homogeneous
    homo_graphs, homo_labels = convert_dataset(clean_graphs, clean_labels, feature_dims)

    # We need to remap train_idx/test_idx because some graphs may have been
    # dropped during hetero->homo conversion.
    # Build mapping: original clean index -> homo index
    # The convert_dataset function processes in order and skips failures,
    # so we track which survived.
    original_to_homo = {}
    homo_idx = 0
    for orig_idx in range(len(clean_graphs)):
        # Try converting this graph to check if it survived
        # (We already converted, so just check if homo_idx matches)
        pass  # We need a different approach

    # Simpler: re-derive split from SMILES after conversion
    print("\nRe-deriving family split on converted homogeneous graphs...")
    homo_families = defaultdict(list)
    for idx, g in enumerate(homo_graphs):
        parent = getattr(g, 'original_smiles', None)
        key = parent if parent else g.smiles
        homo_families[key].append(idx)

    family_keys = list(homo_families.keys())
    family_labels = []
    for key in family_keys:
        member_labels = [homo_labels[i] for i in homo_families[key]]
        family_labels.append(1 if sum(member_labels) > len(member_labels) / 2 else 0)

    try:
        train_fam, test_fam = train_test_split(
            np.arange(len(family_keys)), test_size=0.2,
            stratify=family_labels, random_state=42
        )
    except ValueError:
        train_fam, test_fam = train_test_split(
            np.arange(len(family_keys)), test_size=0.2, random_state=42
        )

    final_train_idx = []
    final_test_idx = []
    for fi in train_fam:
        final_train_idx.extend(homo_families[family_keys[fi]])
    for fi in test_fam:
        final_test_idx.extend(homo_families[family_keys[fi]])

    # Assign labels
    for g, lbl in zip(homo_graphs, homo_labels):
        g.y = torch.tensor([lbl], dtype=torch.long)

    train_graphs = [homo_graphs[i] for i in final_train_idx]
    test_graphs = [homo_graphs[i] for i in final_test_idx]

    train_labels_final = [homo_labels[i] for i in final_train_idx]
    test_labels_final = [homo_labels[i] for i in final_test_idx]

    # Verify no leakage in homo split
    train_sm = {homo_graphs[i].smiles for i in final_train_idx}
    test_sm = {homo_graphs[i].smiles for i in final_test_idx}
    overlap = train_sm & test_sm
    print(f"Final homo split — Train: {len(train_graphs)}, Test: {len(test_graphs)}, "
          f"Overlap: {len(overlap)}")

    print(f"Train class dist: auth={sum(1 for l in train_labels_final if l==0)}, "
          f"fake={sum(1 for l in train_labels_final if l==1)}")
    print(f"Test class dist:  auth={sum(1 for l in test_labels_final if l==0)}, "
          f"fake={sum(1 for l in test_labels_final if l==1)}")

    in_channels = homo_graphs[0].x.size(1)
    print(f"Node feature dimension: {in_channels}")

    if batch_size is None:
        batch_size = 32 if device.type == 'cuda' else 16

    train_loader = DataLoader(train_graphs, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    model = HomoGATDetector(
        in_channels=in_channels, hidden_channels=hidden_channels,
        out_channels=2, num_heads=num_heads, num_layers=num_layers,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=10, min_lr=1e-6)

    trainer = HomoTrainer(model, device, save_path=RESULTS_DIR)
    results = trainer.train(
        train_loader, test_loader, criterion, optimizer,
        num_epochs=num_epochs, patience=patience, scheduler=scheduler)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  HOMOGENEOUS GAT — CLEAN DATA RESULTS")
    print("=" * 65)
    print(f"Best F1 Score : {results['best_f1']:.4f}")
    print(f"Best Epoch    : {results['best_epoch']}")

    baseline_f1 = 0.711
    improvement = (results['best_f1'] - baseline_f1) / baseline_f1 * 100
    print(f"vs baseline   : {improvement:+.1f}%")

    old_f1 = 0.9198
    diff = results['best_f1'] - old_f1
    print(f"\nvs dirty data (F1=0.9198): {diff:+.4f} ({diff/old_f1*100:+.1f}%)")
    if diff < -0.05:
        print("  → Significant drop confirms previous result was inflated")
        print("    by data leakage and duplicate contamination.")

    summary_path = trainer.save_path / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Homogeneous GAT — Clean Data Training Summary\n")
        f.write("=" * 45 + "\n")
        f.write(f"Best F1 Score        : {results['best_f1']:.4f}\n")
        f.write(f"Best Epoch           : {results['best_epoch']}\n")
        f.write(f"vs baseline (0.711)  : {improvement:+.1f}%\n")
        f.write(f"vs dirty (0.9198)    : {diff:+.4f}\n")
        f.write(f"Model Parameters     : {total_params:,}\n")
        f.write(f"Node feature dim     : {in_channels}\n")
        f.write(f"Hidden channels      : {hidden_channels}\n")
        f.write(f"GAT heads            : {num_heads}\n")
        f.write(f"GAT layers           : {num_layers}\n")
        f.write(f"Dropout              : {dropout}\n")
        f.write(f"Device               : {device}\n")
        f.write(f"Dataset cleaned      : Yes (dedup + family split)\n")
        f.write(f"SMILES overlap       : {len(overlap)}\n")
        f.write(f"Training Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Summary saved: {summary_path.absolute()}")
    return results, trainer


# ═════════════════════════════════════════════
#  COMPARISON UTILITY
# ═════════════════════════════════════════════

def compare_models(hgt_results: dict, homo_results: dict):
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON — HGT vs Homogeneous GAT (Clean Data)")
    print("=" * 60)
    print(f"{'Metric':<25} {'HGT (Hetero)':>15} {'GAT (Homo)':>15}")
    print("-" * 55)

    for label, key in [('Best F1', 'best_f1'), ('Best Epoch', 'best_epoch')]:
        hv = hgt_results.get(key, 'N/A')
        gv = homo_results.get(key, 'N/A')
        if isinstance(hv, float):
            print(f"{label:<25} {hv:>15.4f} {gv:>15.4f}")
        else:
            print(f"{label:<25} {str(hv):>15} {str(gv):>15}")


def main():
    print("Homogeneous GAT — WITH DATA INTEGRITY CHECKS")
    print("=" * 65)

    patterns = ['*enhanced*.pt', '*pharma*.pt', '*hetero*.pt', '*.pt']
    found = []
    for pat in patterns:
        found.extend(Path('.').glob(pat))
    found = sorted(set(f for f in found if 'best_model' not in str(f)
                        and 'Homo_GNN' not in str(f)
                        and 'HGT_Enhanced' not in str(f)))

    if not found:
        print("No .pt dataset files found.")
        return

    print(f"Found {len(found)} dataset file(s):")
    for i, f in enumerate(found, 1):
        sz = f.stat().st_size / (1024 * 1024)
        print(f"  {i}. {f} ({sz:.1f} MB)")

    enhanced = [f for f in found if 'enhanced' in f.name.lower()]
    dataset_path = str(enhanced[0] if enhanced else found[0])
    print(f"\nUsing: {dataset_path}")

    run_homo_training(dataset_path)


if __name__ == "__main__":
    main()