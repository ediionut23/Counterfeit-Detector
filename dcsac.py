import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, BatchNorm
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import math
import glob
from pathlib import Path

from torch.serialization import add_safe_globals
add_safe_globals([
    'torch_geometric.data.storage.BaseStorage',
    'torch_geometric.data.hetero_data.HeteroData',
    'torch_geometric.data.data.Data',
    'torch_geometric.data.batch.Batch'
])

warnings.filterwarnings('ignore')

RESULTS_DIR = './HGT_Enhanced_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_unique_filename(base_name, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def save_results(file_path, description="File"):
    abs_path = os.path.abspath(file_path)
    print(f"{description} saved: {abs_path}")
    
    if os.name == 'nt':  # Windows
        print(f"Open with: code \"{abs_path}\"")
    else:  
        print(f"Open with: code '{abs_path}'")
    
    return abs_path

try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = [12, 8]


def load_and_prepare_enhanced_dataset(file_path):
    try:
        file_path = Path(file_path)
        print(f"Loading enhanced dataset from: {file_path.absolute()}")

        if not file_path.exists():
            print(f"Error: File not found at {file_path.absolute()}")
            return None, None, None

        try:
            loaded_data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            print(f"Trying alternative loading method due to: {e}")
            try:
                loaded_data = torch.load(file_path, weights_only=False, map_location='cpu')
            except Exception as e2:
                print(f"Failed to load dataset: {e2}")
                return None, None, None

        if isinstance(loaded_data, tuple):
            if len(loaded_data) >= 3:
                hetero_graphs, labels, metadata = loaded_data[0], loaded_data[1], loaded_data[2]
            else:
                raise ValueError(f"Expected at least 3 components, got {len(loaded_data)}")
        elif isinstance(loaded_data, dict):
            hetero_graphs = loaded_data.get('hetero_graphs', loaded_data.get('graphs'))
            labels = loaded_data.get('labels')
            metadata = loaded_data.get('metadata')
        else:
            raise TypeError(f"Unexpected data format: {type(loaded_data)}")

        print(f"Raw dataset loaded: {len(hetero_graphs)} molecules")

        print("Analyzing feature dimensions...")
        feature_analysis = analyze_dataset_features(hetero_graphs[:50])

        print("Feature dimensions found:")
        for node_type, dim in feature_analysis.items():
            print(f"  {node_type}: {dim} features")

        print("Cleaning and validating graphs...")
        clean_graphs, clean_labels = clean_heterographs(hetero_graphs, labels, feature_analysis)

        print(f"Clean dataset: {len(clean_graphs)} valid molecules")

        unique_labels, counts = np.unique(clean_labels, return_counts=True)
        print(f"Class distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = "Authentic" if label == 0 else "Counterfeit"
            print(f"  {class_name}: {count} ({count/len(clean_labels)*100:.1f}%)")

        return clean_graphs, clean_labels, feature_analysis

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

ATOM_TYPES_ORDERED = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'B']
ATOM_TYPE_TO_IDX   = {a: i for i, a in enumerate(ATOM_TYPES_ORDERED)}

def add_atom_type_onehot(graphs, feature_analysis):
    n_types = len(ATOM_TYPES_ORDERED)
    for g in graphs:
        for node_type in g.node_types:
            if not hasattr(g[node_type], 'x') or g[node_type].x is None:
                continue
            n_nodes = g[node_type].x.size(0)
            idx = ATOM_TYPE_TO_IDX.get(node_type, n_types - 1)
            one_hot = torch.zeros(n_nodes, n_types)
            one_hot[:, idx] = 1.0
            g[node_type].x = torch.cat([g[node_type].x, one_hot], dim=1)
    updated = {k: v + n_types for k, v in feature_analysis.items()}
    return updated


def analyze_dataset_features(sample_graphs):
    feature_dims = {}

    for graph in sample_graphs:
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x') and graph[node_type].x.size(0) > 0:
                dim = graph[node_type].x.shape[-1]
                if node_type not in feature_dims:
                    feature_dims[node_type] = dim
                else:
                    if feature_dims[node_type] != dim:
                        print(f"Warning: Inconsistent dimensions for {node_type}: {feature_dims[node_type]} vs {dim}")
                        feature_dims[node_type] = max(feature_dims[node_type], dim)

    common_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
    for node_type in common_types:
        if node_type not in feature_dims:
            most_common_dim = max(feature_dims.values()) if feature_dims else 26
            feature_dims[node_type] = most_common_dim
            print(f"Adding missing node type {node_type} with dimension {most_common_dim}")

    return feature_dims

def clean_heterographs(hetero_graphs, labels, feature_dims):
    """Clean and validate heterographs with progress bar"""
    clean_graphs = []
    clean_labels = []

    print("Processing graphs...")
    for i, (graph, label) in enumerate(tqdm(zip(hetero_graphs, labels), total=len(hetero_graphs))):
        try:
            cleaned_graph = clean_single_graph(graph, feature_dims)
            if cleaned_graph is not None and has_valid_structure(cleaned_graph):
                clean_graphs.append(cleaned_graph)
                clean_labels.append(label)
        except Exception as e:
            if i < 10:  # Only print first 10 errors
                print(f"Skipping graph {i}: {e}")
            continue

    return clean_graphs, clean_labels

def clean_single_graph(graph, feature_dims):
    try:
        cleaned = HeteroData()

        valid_node_types = []
        for node_type in feature_dims.keys():
            if (node_type in graph.node_types and
                hasattr(graph[node_type], 'x') and
                graph[node_type].x.size(0) > 0):

                x = graph[node_type].x
                expected_dim = feature_dims[node_type]

                if x.shape[-1] != expected_dim:
                    if x.shape[-1] < expected_dim:
                        padding = torch.zeros(x.size(0), expected_dim - x.shape[-1])
                        x = torch.cat([x, padding], dim=1)
                    else:
                        x = x[:, :expected_dim]

                cleaned[node_type].x = x
                valid_node_types.append(node_type)

        for edge_type in graph.edge_types:
            src_type, rel_type, dst_type = edge_type

            if (src_type in valid_node_types and dst_type in valid_node_types and
                hasattr(graph[edge_type], 'edge_index') and
                graph[edge_type].edge_index.size(1) > 0):

                edge_index = graph[edge_type].edge_index
                src_max = cleaned[src_type].x.size(0)
                dst_max = cleaned[dst_type].x.size(0)

                valid_mask = (edge_index[0] < src_max) & (edge_index[1] < dst_max) & (edge_index[0] >= 0) & (edge_index[1] >= 0)

                if valid_mask.sum() > 0:
                    cleaned[edge_type].edge_index = edge_index[:, valid_mask]

                    if hasattr(graph[edge_type], 'edge_attr') and graph[edge_type].edge_attr.size(0) > 0:
                        edge_attr = graph[edge_type].edge_attr[valid_mask]
                        if edge_attr.size(0) > 0:
                            cleaned[edge_type].edge_attr = edge_attr

        for attr in ['y']:
            if hasattr(graph, attr):
                setattr(cleaned, attr, getattr(graph, attr))

        return cleaned

    except Exception:
        return None

def has_valid_structure(graph):
    """Check if graph has valid structure"""
    has_nodes = any(hasattr(graph[nt], 'x') and graph[nt].x.size(0) > 0
                   for nt in graph.node_types)
    has_edges = any(hasattr(graph[et], 'edge_index') and graph[et].edge_index.size(1) > 0
                   for et in graph.edge_types)
    return has_nodes and has_edges


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class RobustEnhancedHGTDetector(torch.nn.Module):
    def __init__(self, feature_dims, hidden_channels=192, out_channels=2,
                 num_heads=8, num_layers=3, dropout=0.15):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_dims = feature_dims

        self.node_embeddings = torch.nn.ModuleDict()
        self.node_norms = torch.nn.ModuleDict()

        for node_type, input_dim in feature_dims.items():
            self.node_embeddings[node_type] = torch.nn.Linear(input_dim, hidden_channels)
            self.node_norms[node_type] = torch.nn.LayerNorm(hidden_channels)

        node_types = list(feature_dims.keys())
        edge_types = []
        for src in node_types:
            for dst in node_types:
                edge_types.append((src, 'bond_to', dst))

        self.metadata = (node_types, edge_types)

        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, self.metadata, num_heads))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))

        self.node_attention = torch.nn.Parameter(torch.ones(len(node_types)))

        classifier_input_dim = hidden_channels * 2 * len(node_types)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_input_dim, hidden_channels * 2),
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

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x_dict, edge_index_dict, batch_dict=None, batch_size=None):
        device = next(self.parameters()).device

        h_dict = {}
        node_types = list(self.feature_dims.keys())

        for node_type in node_types:
            if node_type in x_dict and x_dict[node_type].size(0) > 0:
                x = x_dict[node_type]
                h = self.node_embeddings[node_type](x)
                h = self.node_norms[node_type](h)
                h_dict[node_type] = F.relu(h)
            else:
                h_dict[node_type] = torch.zeros((0, self.hidden_channels), device=device)

        for i, (conv, layer_norm) in enumerate(zip(self.convs, self.layer_norms)):
            try:
                h_dict_new = conv(h_dict, edge_index_dict)

                for node_type in h_dict_new.keys():
                    if h_dict_new[node_type].size(0) > 0:
                        h_normalized = layer_norm(h_dict_new[node_type])
                        h_activated = F.relu(h_normalized)
                        h_dropped = F.dropout(h_activated, p=self.dropout, training=self.training)

                        if (node_type in h_dict and
                            h_dict[node_type].shape == h_dropped.shape and
                            h_dict[node_type].size(0) > 0):
                            h_dict_new[node_type] = h_dropped + h_dict[node_type]
                        else:
                            h_dict_new[node_type] = h_dropped

                h_dict = h_dict_new

            except Exception as e:
                print(f"Warning: HGT layer {i} error: {e}")
                break

        batch_size = self._get_batch_size(batch_dict, h_dict, batch_size)
        pooled_features = self._global_pooling(h_dict, batch_dict, batch_size, node_types)

        if pooled_features.size(0) > 0:
            out = self.classifier(pooled_features)
        else:
            out = torch.zeros((batch_size, self.out_channels), device=device)

        return out

    def _get_batch_size(self, batch_dict, h_dict, provided_batch_size):
        if provided_batch_size is not None:
            return provided_batch_size

        if batch_dict:
            max_batch = 0
            for node_type, batch_tensor in batch_dict.items():
                if node_type in h_dict and batch_tensor.numel() > 0:
                    max_batch = max(max_batch, int(batch_tensor.max().item()) + 1)
            return max_batch if max_batch > 0 else 1

        return 1

    def _global_pooling(self, h_dict, batch_dict, batch_size, node_types):
        device = next(self.parameters()).device
        attention_weights = F.softmax(self.node_attention, dim=0)

        pooled_features = []

        for idx, node_type in enumerate(node_types):
            if node_type in h_dict and h_dict[node_type].size(0) > 0:
                node_features = h_dict[node_type]

                if batch_dict and node_type in batch_dict and batch_dict[node_type].numel() > 0:
                    pooled_mean = torch.zeros(batch_size, self.hidden_channels, device=device)
                    pooled_max = torch.zeros(batch_size, self.hidden_channels, device=device)

                    for i in range(batch_size):
                        mask = batch_dict[node_type] == i
                        if mask.any():
                            masked_features = node_features[mask]
                            pooled_mean[i] = masked_features.mean(dim=0)
                            pooled_max[i] = masked_features.max(dim=0).values
                else:
                    pooled_mean = node_features.mean(dim=0, keepdim=True)
                    pooled_max = node_features.max(dim=0, keepdim=True).values
                    if batch_size > 1:
                        pooled_mean = pooled_mean.expand(batch_size, -1)
                        pooled_max = pooled_max.expand(batch_size, -1)

                pooled = torch.cat([pooled_mean, pooled_max], dim=1) * attention_weights[idx]
                pooled_features.append(pooled)
            else:
                empty_pool = torch.zeros(batch_size, self.hidden_channels * 2, device=device)
                pooled_features.append(empty_pool)

        if pooled_features:
            return torch.cat(pooled_features, dim=1)
        else:
            return torch.zeros(batch_size, self.hidden_channels * 2 * len(node_types), device=device)


class VSCodeTrainer:
    
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

        progress_bar = tqdm(loader, desc="Training", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                batch_size = self._get_batch_size(batch)
                out = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch_size)

                labels = self._get_labels(batch, batch_size)
                out, labels = self._align_dimensions(out, labels)

                loss = criterion(out, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                predictions.extend(out.argmax(dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            except Exception as e:
                self.logger.warning(f"Error in batch {batch_idx}: {str(e)}")
                continue

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
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
                    batch_size = self._get_batch_size(batch)
                    out = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch_size)

                    labels = self._get_labels(batch, batch_size)
                    out, labels = self._align_dimensions(out, labels)

                    loss = criterion(out, labels)
                    total_loss += loss.item()
                    predictions.extend(out.argmax(dim=1).cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

                except Exception:
                    continue

        if len(predictions) == 0:
            return {'loss': float('inf'), 'accuracy': 0, 'f1': 0}, [], []

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(true_labels, predictions)
        _, _, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )

        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}, predictions, true_labels

    def train(self, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, patience=20, scheduler=None):
        print(f"Starting training with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} parameters")
        self.logger.info(f"Training started - {num_epochs} epochs, patience={patience}")

        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_preds, train_labels = self.train_epoch(train_loader, optimizer, criterion)
            val_metrics, _, _ = self.evaluate(val_loader, criterion)
            test_metrics, test_preds, test_labels = self.evaluate(test_loader, criterion)

            if len(train_preds) > 0 and len(train_labels) > 0:
                _, _, train_f1, _ = precision_recall_fscore_support(
                    train_labels, train_preds, average='binary', zero_division=0
                )
            else:
                train_f1 = 0.0

            self.train_losses.append(train_loss)
            self.train_f1_scores.append(train_f1)
            self.test_losses.append(test_metrics['loss'])
            self.test_f1_scores.append(test_metrics['f1'])
            self.test_accuracies.append(test_metrics['accuracy'])

            current_val_f1 = val_metrics['f1']
            if current_val_f1 > self.best_f1:
                self.best_f1 = current_val_f1
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0

                checkpoint_path = self.save_path / f'best_model_val_f1_{current_val_f1:.4f}.pt'
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'val_f1': self.best_f1,
                    'epoch': epoch + 1,
                    'feature_dims': self.model.feature_dims
                }, checkpoint_path)

                self.logger.info(f"New best model saved: Val F1={current_val_f1:.4f}")

            else:
                patience_counter += 1

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_val_f1)
                else:
                    scheduler.step()

            epoch_msg = (f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train F1={train_f1:.4f} - "
                        f"Val F1={current_val_f1:.4f} (best={self.best_f1:.4f}) - "
                        f"Test F1={test_metrics['f1']:.4f}")
            print(epoch_msg)
            self.logger.info(epoch_msg)

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.create_training_plots()

        return {
            'best_val_f1': self.best_f1,
            'best_epoch': best_epoch + 1
        }

    def create_training_plots(self):
        """Create and save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses) + 1)

        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.test_losses, 'r-', label='Test Loss', linewidth=2)
        axes[0, 0].set_title('Loss Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, self.train_f1_scores, 'b-', label='Train F1', linewidth=2)
        axes[0, 1].plot(epochs, self.test_f1_scores, 'r-', label='Test F1', linewidth=2)
        axes[0, 1].axhline(y=self.best_f1, color='g', linestyle='--',
                          label=f'Best F1: {self.best_f1:.3f}', linewidth=2)
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        axes[1, 0].plot(epochs, self.test_accuracies, 'g-', label='Test Accuracy', linewidth=2)
        axes[1, 0].set_title('Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])

        axes[1, 1].plot(epochs, [max(0, f1 - loss) for f1, loss in zip(self.test_f1_scores, self.test_losses)], 
                       'm-', label='Performance Score', linewidth=2)
        axes[1, 1].set_title('Performance Score (F1 - Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Enhanced HGT Training Results', fontsize=16, fontweight='bold', y=0.98)

        plot_filename = get_unique_filename('training_results', 'png')
        plot_path = self.save_path / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        save_results(plot_path, "Training plots")
        plt.show()

    def _get_batch_size(self, batch):
        if hasattr(batch, 'y') and batch.y is not None:
            return batch.y.size(0)
        elif batch.batch_dict:
            max_batch = 0
            for batch_tensor in batch.batch_dict.values():
                if batch_tensor.numel() > 0:
                    max_batch = max(max_batch, int(batch_tensor.max().item()) + 1)
            return max_batch if max_batch > 0 else 1
        return 1

    def _get_labels(self, batch, batch_size):
        if hasattr(batch, 'y') and batch.y is not None:
            return batch.y.view(-1)
        else:
            return torch.zeros(batch_size, dtype=torch.long, device=self.device)

    def _align_dimensions(self, out, labels):
        if labels.size(0) != out.size(0):
            min_size = min(labels.size(0), out.size(0))
            labels = labels[:min_size]
            out = out[:min_size]
        return out, labels


def run_enhanced_training(dataset_path):
    """
    Main training function optimized for VS Code
    """
    print("="*60)
    print("ENHANCED HGT TRAINING - VS CODE VERSION")
    print("="*60)

    logging.basicConfig(level=logging.INFO)

    hetero_graphs, labels, feature_analysis = load_and_prepare_enhanced_dataset(dataset_path)

    if hetero_graphs is None:
        print("❌ Failed to load dataset!")
        return None

    feature_analysis = add_atom_type_onehot(hetero_graphs, feature_analysis)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, stratify=train_val_labels, random_state=42
    )

    train_graphs = [hetero_graphs[i] for i in train_idx]
    val_graphs   = [hetero_graphs[i] for i in val_idx]
    test_graphs  = [hetero_graphs[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels   = [labels[i] for i in val_idx]
    test_labels  = [labels[i] for i in test_idx]

    for i, g in enumerate(train_graphs):
        g.y = torch.tensor([train_labels[i]], dtype=torch.long)
    for i, g in enumerate(val_graphs):
        g.y = torch.tensor([val_labels[i]], dtype=torch.long)
    for i, g in enumerate(test_graphs):
        g.y = torch.tensor([test_labels[i]], dtype=torch.long)

    print(f"📊 Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    batch_size = 32 if device.type == 'cuda' else 16  # Adjust for available memory
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = RobustEnhancedHGTDetector(
        feature_dims=feature_analysis,
        hidden_channels=96,
        num_heads=4,
        num_layers=3,
        dropout=0.20
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,}")

    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=10, min_lr=1e-6
    )

    trainer = VSCodeTrainer(model, device)
    print("🚀 Starting training...")
    
    results = trainer.train(
        train_loader, val_loader, test_loader, criterion, optimizer,
        num_epochs=80, patience=15, scheduler=scheduler
    )

    print("\n" + "="*60)
    print("🎯 TRAINING COMPLETED")
    print("="*60)
    print(f"✅ Best F1 Score: {results['best_f1']:.4f}")
    print(f"📈 Best Epoch: {results['best_epoch']}")

    baseline_f1 = 0.711
    improvement = (results['best_f1'] - baseline_f1) / baseline_f1 * 100
    print(f"📊 Improvement over baseline: {improvement:.1f}%")

    if results['best_f1'] > baseline_f1:
        print("🎉 Enhanced dataset achieved better performance!")
    
    results_summary = {
        'best_f1': results['best_f1'],
        'best_epoch': results['best_epoch'],
        'improvement': improvement,
        'total_params': total_params,
        'feature_dims': feature_analysis
    }
    
    summary_path = trainer.save_path / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Enhanced HGT Training Summary\n")
        f.write("="*30 + "\n")
        f.write(f"Best F1 Score: {results['best_f1']:.4f}\n")
        f.write(f"Best Epoch: {results['best_epoch']}\n")
        f.write(f"Improvement: {improvement:.1f}%\n")
        f.write(f"Model Parameters: {total_params:,}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    save_results(summary_path, "Training summary")

    return results, trainer

def run_adaptive_training(dataset_path, use_original_params=True, custom_params=None):

    hetero_graphs, labels, feature_analysis = load_and_prepare_enhanced_dataset(dataset_path)

    if hetero_graphs is None:
        print("❌ Failed to load dataset!")
        return None

    feature_analysis = add_atom_type_onehot(hetero_graphs, feature_analysis)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    if custom_params:
        params = custom_params
    elif use_original_params:
        params = {
            'hidden_channels': 96,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.20,
            'lr': 0.0001,
            'weight_decay': 0.001,
            'batch_size': 64 if device.type == 'cuda' else 32,
            'num_epochs': 100,
            'patience': 30
        }
    else:
        params = {
            'hidden_channels': 96,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.20,
            'lr': 0.0001,
            'weight_decay': 0.008,
            'batch_size': 64 if device.type == 'cuda' else 32,
            'num_epochs': 120,
            'patience': 20
        }
        print("⚡ Using enhanced parameters for complex datasets")

    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
    )
    train_val_labels_tmp = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, stratify=train_val_labels_tmp, random_state=42
    )

    train_graphs = [hetero_graphs[i] for i in train_idx]
    val_graphs   = [hetero_graphs[i] for i in val_idx]
    test_graphs  = [hetero_graphs[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels   = [labels[i] for i in val_idx]
    test_labels  = [labels[i] for i in test_idx]

    for i, g in enumerate(train_graphs):
        g.y = torch.tensor([train_labels[i]], dtype=torch.long)
    for i, g in enumerate(val_graphs):
        g.y = torch.tensor([val_labels[i]], dtype=torch.long)
    for i, g in enumerate(test_graphs):
        g.y = torch.tensor([test_labels[i]], dtype=torch.long)

    print(f"📊 Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    class_weights = total_samples / (len(unique_labels) * label_counts)
    class_weight_tensor = torch.FloatTensor(class_weights).to(device)

    train_loader = DataLoader(train_graphs, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_graphs,   batch_size=params['batch_size'], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_graphs,  batch_size=params['batch_size'], shuffle=False, num_workers=0)

    model = RobustEnhancedHGTDetector(
        feature_dims=feature_analysis,
        hidden_channels=params['hidden_channels'],
        out_channels=2,
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,}")

    criterion_focal = FocalLoss(alpha=0.75, gamma=2.0)
    criterion_weighted = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-7
    )

    trainer = VSCodeTrainer(model, device)
    print("🚀 Starting adaptive training...")

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion_weighted,
        optimizer=optimizer,
        num_epochs=params['num_epochs'],
        patience=params['patience'],
        scheduler=scheduler
    )

    print("\n" + "="*60)
    print("🎯 ADAPTIVE TRAINING COMPLETED")
    print("="*60)
    print(f"✅ Best F1 Score: {results['best_f1']:.4f}")
    print(f"📈 Best Epoch: {results['best_epoch']}")

    baseline_f1 = 0.711
    improvement = (results['best_f1'] - baseline_f1) / baseline_f1 * 100
    print(f"📊 Improvement over baseline: {improvement:.1f}%")

    return results, trainer

def find_dataset_files(directory="."):
    """Find dataset files in VS Code workspace"""
    search_dir = Path(directory)
    patterns = [
        '*enhanced*.pt',
        '*pharma*.pt',
        '*hetero*.pt',
        '*.pt'
    ]

    found_files = []
    for pattern in patterns:
        files = list(search_dir.glob(pattern))
        found_files.extend(files)

    # Remove duplicates and sort
    found_files = list(set(found_files))
    found_files.sort()

    return [str(f) for f in found_files]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced HGT Training Script")
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to .pt dataset file (e.g. intelligent_pharma_50k.pt)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive training instead of standard training')
    args = parser.parse_args()

    print("🧬 Enhanced HGT Training Script - VS Code Edition")
    print("="*60)

    if args.dataset:
        dataset_path = args.dataset
        if not Path(dataset_path).exists():
            print(f"❌ File not found: {dataset_path}")
            return None
        print(f"\n🎯 Using dataset: {Path(dataset_path).name}")
    else:
        dataset_files = find_dataset_files()

        if not dataset_files:
            print("❌ No dataset files found!")
            print("Please ensure you have a .pt file in your workspace.")
            print("Expected patterns: *enhanced*.pt, *pharma*.pt, *hetero*.pt")
            return None

        print(f"✅ Found {len(dataset_files)} dataset file(s):")
        for i, file in enumerate(dataset_files, 1):
            file_size = Path(file).stat().st_size / (1024*1024)
            print(f"   {i}. {file} ({file_size:.1f} MB)")

        enhanced_files = [f for f in dataset_files if 'enhanced' in f.lower()]
        if enhanced_files:
            dataset_path = enhanced_files[0]
            print(f"\n🎯 Using enhanced dataset: {Path(dataset_path).name}")
        else:
            dataset_path = dataset_files[0]
            print(f"\n🎯 Using dataset: {Path(dataset_path).name}")

    print(f"📁 Full path: {Path(dataset_path).absolute()}")
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)

    try:
        if args.adaptive:
            results, trainer = run_adaptive_training(dataset_path)
        else:
            results, trainer = run_enhanced_training(dataset_path)

        if results is not None:
            print("\n🎉 Training completed successfully!")
            print(f"   📊 Best F1 Score: {results['best_f1']:.4f}")
            print(f"   📈 Best Epoch: {results['best_epoch']}")
            print(f"   📁 Results saved in: {trainer.save_path.absolute()}")
            
            result_files = list(trainer.save_path.glob("*"))
            if result_files:
                print(f"\n📋 Generated files ({len(result_files)}):")
                for file in sorted(result_files):
                    print(f"   • {file.name}")
        else:
            print("❌ Training failed!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        error_log = Path(RESULTS_DIR) / 'error_log.txt'
        with open(error_log, 'w') as f:
            f.write(f"Error occurred at: {datetime.now()}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Error: {str(e)}\n")
            f.write("Full traceback:\n")
            traceback.print_exc(file=f)
        
        print(f"📝 Error details saved to: {error_log.absolute()}")

def quick_test():
    print("🧪 Running quick test...")
    
    try:
        import torch
        import torch_geometric
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("ℹ️ CUDA not available, using CPU")
    
    test_dir = Path(RESULTS_DIR)
    test_dir.mkdir(exist_ok=True)
    print(f"✅ Results directory: {test_dir.absolute()}")
    
    print("✅ All tests passed!")
    return True


if __name__ == "__main__":

    main()
    