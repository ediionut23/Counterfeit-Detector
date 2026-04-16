"""
Homogeneous GNN Training Script for VS Code
Comparable baseline for HGT heterogeneous model
Author: For ICTAI paper comparison
Version: 1.0.0

Architecture: GAT (Graph Attention Network) on homogeneous graphs
- Converts HeteroData -> Data by merging all atom types into one node type
- Atom type encoded as one-hot feature (appended to existing features)
- Fair comparison with HGT: same dataset, same train/test split, same metrics
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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

RESULTS_DIR = './Homo_GNN_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Node types from HGT model — order matters for one-hot encoding
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
ATOM_TYPE_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)


# ─────────────────────────────────────────────
#  HETERO -> HOMO CONVERSION
# ─────────────────────────────────────────────

def hetero_to_homo(hetero_graph: HeteroData, feature_dims: dict) -> Data:
    """
    Convert a HeteroData graph to a homogeneous Data graph.

    Strategy:
      - All atom node types are concatenated into a single node matrix.
      - A one-hot vector (length = NUM_ATOM_TYPES) is appended to each atom's
        features so the model can still distinguish atom types.
      - All edge types are merged into a single edge_index (node indices are
        remapped to the global ordering).
      - Edge attributes, if present, are concatenated as well.
    """
    base_dim = max(feature_dims.values()) if feature_dims else 26
    total_feature_dim = base_dim + NUM_ATOM_TYPES

    all_x = []
    node_offsets = {}   # atom_type -> start index in the combined node list
    current_offset = 0

    for atom_type in ATOM_TYPES:
        if atom_type in hetero_graph.node_types and \
                hasattr(hetero_graph[atom_type], 'x') and \
                hetero_graph[atom_type].x.size(0) > 0:

            x = hetero_graph[atom_type].x.float()
            n_nodes = x.size(0)
            feat_dim = x.size(1)

            # Pad / truncate to base_dim
            if feat_dim < base_dim:
                x = F.pad(x, (0, base_dim - feat_dim))
            elif feat_dim > base_dim:
                x = x[:, :base_dim]

            # One-hot for atom type
            one_hot = torch.zeros(n_nodes, NUM_ATOM_TYPES)
            idx = ATOM_TYPE_TO_IDX.get(atom_type, 0)
            one_hot[:, idx] = 1.0

            x_full = torch.cat([x, one_hot], dim=1)
            all_x.append(x_full)
            node_offsets[atom_type] = current_offset
            current_offset += n_nodes

    if not all_x:
        return None

    x_combined = torch.cat(all_x, dim=0)  # [total_nodes, base_dim + NUM_ATOM_TYPES]

    # Merge edges
    all_edges = []
    all_edge_attrs = []
    edge_attr_dim = None

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
            ea = hetero_graph[edge_type].edge_attr.float()
            all_edge_attrs.append(ea)
            edge_attr_dim = ea.size(1)

    if not all_edges:
        return None

    edge_index_combined = torch.cat(all_edges, dim=1)

    homo = Data(x=x_combined, edge_index=edge_index_combined)

    # Edge attributes: only attach if ALL edge groups had them
    if len(all_edge_attrs) == len(all_edges) and all_edge_attrs:
        homo.edge_attr = torch.cat(all_edge_attrs, dim=0)

    # Copy graph-level attributes
    for attr in ['y', 'smiles', 'source_id', 'source', 'counterfeit_type', 'difficulty']:
        if hasattr(hetero_graph, attr):
            setattr(homo, attr, getattr(hetero_graph, attr))

    return homo


def convert_dataset(hetero_graphs, labels, feature_dims):
    """Convert full list of HeteroData graphs to homogeneous Data graphs."""
    homo_graphs = []
    homo_labels = []

    print("Converting heterogeneous -> homogeneous graphs...")
    for i, (hg, label) in enumerate(tqdm(zip(hetero_graphs, labels), total=len(hetero_graphs))):
        try:
            hg_homo = hetero_to_homo(hg, feature_dims)
            if hg_homo is not None and hg_homo.x.size(0) > 0 and hg_homo.edge_index.size(1) > 0:
                homo_graphs.append(hg_homo)
                homo_labels.append(label)
        except Exception as e:
            if i < 10:
                print(f"Skipping graph {i}: {e}")

    print(f"Converted: {len(homo_graphs)} valid homogeneous graphs")
    return homo_graphs, homo_labels


# ─────────────────────────────────────────────
#  FOCAL LOSS  (same as HGT script)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  HOMOGENEOUS GAT MODEL
# ─────────────────────────────────────────────

class HomoGATDetector(torch.nn.Module):
    """
    GAT-based homogeneous graph classifier.

    Design choices for fair comparison with HGT:
      - Same hidden_channels (192), same depth (3 layers), same dropout (0.15)
      - Multi-head attention (8 heads) mirroring HGT's num_heads
      - Residual connections after each GAT layer
      - Combined mean+max global pooling (richer graph-level representation)
      - Identical 4-layer MLP classifier head
    """

    def __init__(self, in_channels, hidden_channels=192, out_channels=2,
                 num_heads=8, num_layers=3, dropout=0.15):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        self.input_norm = torch.nn.LayerNorm(hidden_channels)

        # GAT layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            # concat=False averages heads -> output is hidden_channels (not heads * hidden_channels)
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=num_heads,
                        dropout=dropout, concat=False)
            )
            self.norms.append(torch.nn.LayerNorm(hidden_channels))

        # Classifier head (input = mean_pool + max_pool = 2 * hidden_channels)
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
        # Input projection
        h = F.relu(self.input_norm(self.input_proj(x)))

        # GAT layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h  # residual

        # Global pooling: mean + max
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=1)

        return self.classifier(h_graph)


# ─────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────

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
        predictions = []
        true_labels = []

        for batch_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
            try:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                out = self.model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.view(-1)

                if out.size(0) != labels.size(0):
                    min_s = min(out.size(0), labels.size(0))
                    out, labels = out[:min_s], labels[:min_s]

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
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                try:
                    batch = batch.to(self.device)
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    labels = batch.y.view(-1)

                    if out.size(0) != labels.size(0):
                        min_s = min(out.size(0), labels.size(0))
                        out, labels = out[:min_s], labels[:min_s]

                    loss = criterion(out, labels)
                    total_loss += loss.item()
                    predictions.extend(out.argmax(dim=1).cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

                except Exception:
                    continue

        if not predictions:
            return {'loss': float('inf'), 'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}, [], []

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }, predictions, true_labels

    def train(self, train_loader, test_loader, criterion, optimizer,
              num_epochs, patience=20, scheduler=None):

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Starting training — {total_params:,} parameters")
        self.logger.info(f"Training started: {num_epochs} epochs, patience={patience}")

        best_epoch = 0
        patience_counter = 0
        final_preds = []
        final_labels = []

        for epoch in range(num_epochs):
            train_loss, train_preds, train_lbls = self.train_epoch(train_loader, optimizer, criterion)
            test_metrics, test_preds, test_lbls = self.evaluate(test_loader, criterion)

            if train_preds:
                _, _, train_f1, _ = precision_recall_fscore_support(
                    train_lbls, train_preds, average='binary', zero_division=0
                )
            else:
                train_f1 = 0.0

            self.train_losses.append(train_loss)
            self.train_f1_scores.append(train_f1)
            self.test_losses.append(test_metrics['loss'])
            self.test_f1_scores.append(test_metrics['f1'])
            self.test_accuracies.append(test_metrics['accuracy'])

            current_f1 = test_metrics['f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
                final_preds = test_preds
                final_labels = test_lbls

                ckpt_path = self.save_path / f'best_model_f1_{current_f1:.4f}.pt'
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'f1_score': self.best_f1,
                    'epoch': epoch + 1,
                }, ckpt_path)
                self.logger.info(f"New best model: F1={current_f1:.4f}")
            else:
                patience_counter += 1

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
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

        return {
            'best_f1': self.best_f1,
            'best_epoch': best_epoch + 1,
            'final_preds': final_preds,
            'final_labels': final_labels
        }

    def _create_plots(self, final_preds, final_labels):
        epochs = range(1, len(self.train_losses) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.test_losses, 'r-', label='Test', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1
        axes[0, 1].plot(epochs, self.train_f1_scores, 'b-', label='Train F1', linewidth=2)
        axes[0, 1].plot(epochs, self.test_f1_scores, 'r-', label='Test F1', linewidth=2)
        axes[0, 1].axhline(y=self.best_f1, color='g', linestyle='--',
                           label=f'Best={self.best_f1:.3f}', linewidth=2)
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 2].plot(epochs, self.test_accuracies, 'g-', linewidth=2)
        axes[0, 2].set_title('Test Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].grid(True, alpha=0.3)

        # Confusion matrix
        if final_preds and final_labels:
            cm = confusion_matrix(final_labels, final_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                        xticklabels=['Authentic', 'Counterfeit'],
                        yticklabels=['Authentic', 'Counterfeit'])
            axes[1, 0].set_title('Confusion Matrix (Best Epoch)')
            axes[1, 0].set_ylabel('True')
            axes[1, 0].set_xlabel('Predicted')

            # Normalized confusion matrix
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1],
                        xticklabels=['Authentic', 'Counterfeit'],
                        yticklabels=['Authentic', 'Counterfeit'])
            axes[1, 1].set_title('Confusion Matrix — Normalized')
            axes[1, 1].set_ylabel('True')
            axes[1, 1].set_xlabel('Predicted')

        # Learning dynamics
        axes[1, 2].plot(epochs,
                        [max(0, f1 - loss) for f1, loss in
                         zip(self.test_f1_scores, self.test_losses)],
                        'm-', linewidth=2)
        axes[1, 2].set_title('Performance Score (F1 - Loss)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Homogeneous GAT Results', fontsize=16, fontweight='bold', y=1.01)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.save_path / f'training_results_{ts}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path.absolute()}")
        plt.show()


# ─────────────────────────────────────────────
#  DATASET LOADING  (reuses HGT loading logic)
# ─────────────────────────────────────────────

def load_hetero_dataset(file_path):
    """Load the same .pt dataset used by the HGT model."""
    from pathlib import Path as _P
    fp = _P(file_path)
    if not fp.exists():
        print(f"File not found: {fp.absolute()}")
        return None, None, None

    try:
        loaded = torch.load(fp, map_location='cpu')
    except Exception as e:
        print(f"Trying weights_only=False due to: {e}")
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

    # Analyze feature dims (same logic as HGT script)
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


# ─────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────

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
    """
    Train the homogeneous GAT model.
    Default hyperparameters mirror those used in run_enhanced_training() for HGT
    so results are directly comparable.
    """
    print("=" * 60)
    print("HOMOGENEOUS GAT TRAINING")
    print("=" * 60)

    hetero_graphs, labels, feature_dims = load_hetero_dataset(dataset_path)
    if hetero_graphs is None:
        print("Failed to load dataset.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Convert to homogeneous graphs
    homo_graphs, homo_labels = convert_dataset(hetero_graphs, labels, feature_dims)

    labels_np = np.array(homo_labels)
    unique, counts = np.unique(labels_np, return_counts=True)
    print("Class distribution:")
    for lbl, cnt in zip(unique, counts):
        name = "Authentic" if lbl == 0 else "Counterfeit"
        print(f"  {name}: {cnt} ({cnt/len(labels_np)*100:.1f}%)")

    # Add y labels
    for g, lbl in zip(homo_graphs, homo_labels):
        g.y = torch.tensor([lbl], dtype=torch.long)

    # Train / test split — same random_state as HGT for reproducibility
    train_idx, test_idx = train_test_split(
        np.arange(len(homo_labels)), test_size=0.2,
        stratify=homo_labels, random_state=42
    )
    train_graphs = [homo_graphs[i] for i in train_idx]
    test_graphs = [homo_graphs[i] for i in test_idx]

    print(f"Train: {len(train_graphs)}, Test: {len(test_graphs)}")

    # Input feature dimension
    in_channels = homo_graphs[0].x.size(1)
    print(f"Node feature dimension: {in_channels}")

    if batch_size is None:
        batch_size = 32 if device.type == 'cuda' else 16

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=0)

    model = HomoGATDetector(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=2,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=10, min_lr=1e-6
    )

    trainer = HomoTrainer(model, device, save_path=RESULTS_DIR)
    results = trainer.train(
        train_loader, test_loader, criterion, optimizer,
        num_epochs=num_epochs, patience=patience, scheduler=scheduler
    )

    # Summary
    print("\n" + "=" * 60)
    print("HOMOGENEOUS GAT — RESULTS")
    print("=" * 60)
    print(f"Best F1 Score : {results['best_f1']:.4f}")
    print(f"Best Epoch    : {results['best_epoch']}")

    baseline_f1 = 0.711
    improvement = (results['best_f1'] - baseline_f1) / baseline_f1 * 100
    print(f"vs baseline   : {improvement:+.1f}%")

    # Save summary
    summary_path = trainer.save_path / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Homogeneous GAT Training Summary\n")
        f.write("=" * 32 + "\n")
        f.write(f"Best F1 Score      : {results['best_f1']:.4f}\n")
        f.write(f"Best Epoch         : {results['best_epoch']}\n")
        f.write(f"vs baseline (0.711): {improvement:+.1f}%\n")
        f.write(f"Model Parameters   : {total_params:,}\n")
        f.write(f"Node feature dim   : {in_channels}\n")
        f.write(f"Hidden channels    : {hidden_channels}\n")
        f.write(f"GAT heads          : {num_heads}\n")
        f.write(f"GAT layers         : {num_layers}\n")
        f.write(f"Dropout            : {dropout}\n")
        f.write(f"Device             : {device}\n")
        f.write(f"Training Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Summary saved: {summary_path.absolute()}")
    return results, trainer


# ─────────────────────────────────────────────
#  COMPARISON UTILITY
# ─────────────────────────────────────────────

def compare_models(hgt_results: dict, homo_results: dict):
    """
    Print a side-by-side comparison table for the paper.
    Pass in the results dicts returned by run_enhanced_training and run_homo_training.
    """
    print("\n" + "=" * 50)
    print("MODEL COMPARISON — HGT vs Homogeneous GAT")
    print("=" * 50)
    print(f"{'Metric':<20} {'HGT (Hetero)':>15} {'GAT (Homo)':>15}")
    print("-" * 50)

    metrics = [
        ('Best F1', 'best_f1'),
        ('Best Epoch', 'best_epoch'),
    ]
    for label, key in metrics:
        hgt_val = hgt_results.get(key, 'N/A')
        homo_val = homo_results.get(key, 'N/A')
        if isinstance(hgt_val, float):
            print(f"{label:<20} {hgt_val:>15.4f} {homo_val:>15.4f}")
        else:
            print(f"{label:<20} {str(hgt_val):>15} {str(homo_val):>15}")




def main():
    print("Homogeneous GAT Training Script")
    print("=" * 60)

    # Auto-discover dataset (same logic as HGT script)
    patterns = ['*enhanced*.pt', '*pharma*.pt', '*hetero*.pt', '*.pt']
    found = []
    for pat in patterns:
        found.extend(Path('.').glob(pat))
    found = sorted(set(found))

    if not found:
        print("No .pt dataset files found in current directory.")
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
