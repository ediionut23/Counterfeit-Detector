"""
Enhanced HGT Training Script with Multiple Evaluation Options
Compatible with generated datasets from dataset_generator.py
File: training_script.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, BatchNorm
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
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

# Configure PyTorch for safe loading
from torch.serialization import add_safe_globals
add_safe_globals([
    'torch_geometric.data.storage.BaseStorage',
    'torch_geometric.data.hetero_data.HeteroData',
    'torch_geometric.data.data.Data',
    'torch_geometric.data.batch.Batch'
])

warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = './HGT_Enhanced_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_unique_filename(base_name, extension):
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def save_results(file_path, description="File"):
    """Save results and provide path information"""
    abs_path = os.path.abspath(file_path)
    print(f"{description} saved: {abs_path}")
    return abs_path

# Set plotting style
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


def load_and_prepare_dataset(file_path):
    """Load and prepare dataset with error handling"""
    try:
        file_path = Path(file_path)
        print(f"Loading dataset from: {file_path.absolute()}")

        if not file_path.exists():
            print(f"Error: File not found at {file_path.absolute()}")
            return None, None, None

        # Load data with enhanced error handling
        try:
            loaded_data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            print(f"Trying alternative loading method due to: {e}")
            try:
                loaded_data = torch.load(file_path, weights_only=False, map_location='cpu')
            except Exception as e2:
                print(f"Failed to load dataset: {e2}")
                return None, None, None

        # Extract components
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

        # Analyze and fix dataset
        print("Analyzing feature dimensions...")
        feature_analysis = analyze_dataset_features(hetero_graphs[:50])

        print("Feature dimensions found:")
        for node_type, dim in feature_analysis.items():
            print(f"  {node_type}: {dim} features")

        # Clean and validate graphs
        print("Cleaning and validating graphs...")
        clean_graphs, clean_labels = clean_heterographs(hetero_graphs, labels, feature_analysis)

        print(f"Clean dataset: {len(clean_graphs)} valid molecules")

        # Print statistics
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

def analyze_dataset_features(sample_graphs):
    """Analyze feature dimensions from sample graphs"""
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

    # Ensure we have all common node types
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
            if i < 10:
                print(f"Skipping graph {i}: {e}")
            continue

    return clean_graphs, clean_labels

def clean_single_graph(graph, feature_dims):
    """Clean a single heterograph"""
    try:
        cleaned = HeteroData()

        # Process node features
        valid_node_types = []
        for node_type in feature_dims.keys():
            if (node_type in graph.node_types and
                hasattr(graph[node_type], 'x') and
                graph[node_type].x.size(0) > 0):

                x = graph[node_type].x
                expected_dim = feature_dims[node_type]

                # Handle dimension mismatches
                if x.shape[-1] != expected_dim:
                    if x.shape[-1] < expected_dim:
                        padding = torch.zeros(x.size(0), expected_dim - x.shape[-1])
                        x = torch.cat([x, padding], dim=1)
                    else:
                        x = x[:, :expected_dim]

                cleaned[node_type].x = x
                valid_node_types.append(node_type)

        # Process edges with validation
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

        # Copy labels and other attributes
        for attr in ['y', 'smiles', 'source_id', 'source', 'counterfeit_type', 'difficulty']:
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
    """Focal Loss for handling class imbalance"""
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
    """Robust HGT model for counterfeit detection"""
    def __init__(self, feature_dims, hidden_channels=192, out_channels=2,
                 num_heads=8, num_layers=3, dropout=0.15):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_dims = feature_dims

        # Node embeddings
        self.node_embeddings = torch.nn.ModuleDict()
        self.node_norms = torch.nn.ModuleDict()

        for node_type, input_dim in feature_dims.items():
            self.node_embeddings[node_type] = torch.nn.Linear(input_dim, hidden_channels)
            self.node_norms[node_type] = torch.nn.LayerNorm(hidden_channels)

        # Create metadata
        node_types = list(feature_dims.keys())
        edge_types = []
        for src in node_types:
            for dst in node_types:
                edge_types.append((src, 'bond_to', dst))

        self.metadata = (node_types, edge_types)

        # HGT layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, self.metadata, num_heads))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))

        # Node attention
        self.node_attention = torch.nn.Parameter(torch.ones(len(node_types)))

        # Enhanced classifier
        classifier_input_dim = hidden_channels * len(node_types)
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

        # Process node types
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

        # Apply HGT layers
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
                            h_dict[node_type] = h_dropped + h_dict[node_type] * 0.1
                        else:
                            h_dict[node_type] = h_dropped

                h_dict = h_dict_new

            except Exception as e:
                print(f"Warning: HGT layer {i} error: {e}")
                break

        # Global pooling
        batch_size = self._get_batch_size(batch_dict, h_dict, batch_size)
        pooled_features = self._global_pooling(h_dict, batch_dict, batch_size, node_types)

        # Classification
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
                    pooled = torch.zeros(batch_size, self.hidden_channels, device=device)

                    for i in range(batch_size):
                        mask = batch_dict[node_type] == i
                        if mask.any():
                            masked_features = node_features[mask]
                            pooled[i] = masked_features.mean(dim=0)
                else:
                    pooled = node_features.mean(dim=0, keepdim=True)
                    if batch_size > 1:
                        pooled = pooled.expand(batch_size, -1)

                pooled = pooled * attention_weights[idx]
                pooled_features.append(pooled)
            else:
                empty_pool = torch.zeros(batch_size, self.hidden_channels, device=device)
                pooled_features.append(empty_pool)

        if pooled_features:
            return torch.cat(pooled_features, dim=1)
        else:
            return torch.zeros(batch_size, self.hidden_channels * len(node_types), device=device)


class TrainingManager:
    """Training manager with comprehensive evaluation options"""
    
    def __init__(self, model, device, save_path=RESULTS_DIR):
        self.model = model
        self.device = device
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Setup logging
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

        # Metrics tracking
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

    def train(self, train_loader, test_loader, criterion, optimizer, num_epochs, patience=20, scheduler=None):
        print(f"Starting training with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} parameters")
        self.logger.info(f"Training started - {num_epochs} epochs, patience={patience}")

        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            train_loss, train_preds, train_labels = self.train_epoch(train_loader, optimizer, criterion)
            test_metrics, test_preds, test_labels = self.evaluate(test_loader, criterion)

            # Calculate training F1
            if len(train_preds) > 0 and len(train_labels) > 0:
                _, _, train_f1, _ = precision_recall_fscore_support(
                    train_labels, train_preds, average='binary', zero_division=0
                )
            else:
                train_f1 = 0.0

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_f1_scores.append(train_f1)
            self.test_losses.append(test_metrics['loss'])
            self.test_f1_scores.append(test_metrics['f1'])
            self.test_accuracies.append(test_metrics['accuracy'])

            # Check for best model
            current_f1 = test_metrics['f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                checkpoint_path = self.save_path / f'best_model_f1_{current_f1:.4f}.pt'
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'f1_score': self.best_f1,
                    'epoch': epoch + 1,
                    'feature_dims': self.model.feature_dims
                }, checkpoint_path)

                self.logger.info(f"New best model saved: F1={current_f1:.4f}")

            else:
                patience_counter += 1

            # Scheduler step
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_f1)
                else:
                    scheduler.step()

            # Logging
            epoch_msg = (f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train: Loss={train_loss:.4f}, F1={train_f1:.4f} - "
                        f"Test: Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}, F1={current_f1:.4f}")
            print(epoch_msg)
            self.logger.info(epoch_msg)

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Create and save plots
        self.create_training_plots()

        return {
            'best_f1': self.best_f1,
            'best_epoch': best_epoch + 1
        }

    def create_training_plots(self):
        """Create and save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.test_losses, 'r-', label='Test Loss', linewidth=2)
        axes[0, 0].set_title('Loss Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1 Score plot
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

        # Accuracy plot
        axes[1, 0].plot(epochs, self.test_accuracies, 'g-', label='Test Accuracy', linewidth=2)
        axes[1, 0].set_title('Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])

        # Learning curve
        axes[1, 1].plot(epochs, [max(0, f1 - loss) for f1, loss in zip(self.test_f1_scores, self.test_losses)], 
                       'm-', label='Performance Score', linewidth=2)
        axes[1, 1].set_title('Performance Score (F1 - Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Enhanced HGT Training Results', fontsize=16, fontweight='bold', y=0.98)

        # Save plot
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


# === EVALUATION OPTIONS ===

def quick_evaluation(model, dataset_sample, device, num_samples=1000):
    """Quick evaluation on subset of new dataset"""
    print(f"\nQuick evaluation on {num_samples} molecules...")
    
    # Take subset
    subset_graphs = dataset_sample[:num_samples]
    subset_labels = [g.y.item() for g in subset_graphs]
    
    # Create loader
    test_loader = DataLoader(subset_graphs, batch_size=32, shuffle=False, num_workers=0)
    
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating subset"):
            try:
                batch = batch.to(device)
                batch_size = len([g for g in subset_graphs if hasattr(g, 'y')])
                out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch_size)
                
                labels = batch.y.view(-1) if hasattr(batch, 'y') else torch.zeros(batch_size, dtype=torch.long, device=device)
                
                # Align dimensions
                min_size = min(out.size(0), labels.size(0))
                out = out[:min_size]
                labels = labels[:min_size]
                
                predictions.extend(out.argmax(dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
    
    if len(predictions) > 0:
        accuracy = accuracy_score(true_labels, predictions)
        _, _, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
        
        print(f"\nQuick Evaluation Results:")
        print(f"   Samples evaluated: {len(predictions)}")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1 Score: {f1:.3f}")
        
        return f1
    else:
        print("No valid predictions made")
        return 0.0

def transfer_learning_training(model, dataset, device, epochs=20):
    """Transfer learning with pre-trained model"""
    print(f"\nTransfer learning training ({epochs} epochs)...")
    
    # Split dataset
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, 
        stratify=[g.y.item() for g in dataset], random_state=42
    )
    
    train_graphs = [dataset[i] for i in train_idx]
    test_graphs = [dataset[i] for i in test_idx]
    
    # Create loaders
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False, num_workers=0)
    
    # Lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
    
    # Short training
    trainer = TrainingManager(model, device)
    results = trainer.train(
        train_loader, test_loader, criterion, optimizer,
        num_epochs=epochs, patience=8, scheduler=scheduler
    )
    
    return results

def cross_validation_estimate(model, dataset, device, k=3):
    """Cross-validation estimate"""
    print(f"\nCross-validation estimate ({k} folds)...")
    
    # Convert to indices and labels
    indices = np.arange(len(dataset))
    labels = [g.y.item() for g in dataset]
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(indices)):
        print(f"Fold {fold+1}/{k}")
        
        # Take smaller test subset for speed
        test_subset_idx = test_idx[:min(200, len(test_idx))]
        test_subset = [dataset[i] for i in test_subset_idx]
        
        f1 = quick_evaluation(model, test_subset, device, len(test_subset))
        f1_scores.append(f1)
    
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"\nCross-Validation Results:")
    print(f"   Average F1: {avg_f1:.3f} ± {std_f1:.3f}")
    print(f"   Individual folds: {[f'{f1:.3f}' for f1 in f1_scores]}")
    
    return avg_f1

def load_pretrained_model(feature_dims, device):
    """Try to load a pre-trained model"""
    # Look for saved models
    model_files = list(Path(RESULTS_DIR).glob("best_model_*.pt"))
    
    if not model_files:
        print("No pre-trained model found. Creating new model...")
        return RobustEnhancedHGTDetector(feature_dims).to(device), None
    
    # Use the most recent model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading pre-trained model: {latest_model.name}")
    
    try:
        checkpoint = torch.load(latest_model, map_location=device)
        model = RobustEnhancedHGTDetector(feature_dims).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        baseline_f1 = checkpoint.get('f1_score', 0.0)
        print(f"Loaded model with F1 score: {baseline_f1:.3f}")
        
        return model, baseline_f1
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model...")
        return RobustEnhancedHGTDetector(feature_dims).to(device), None

def full_training(dataset_path):
    """Full training on dataset"""
    print("Full training on dataset...")
    
    # Load dataset
    hetero_graphs, labels, feature_analysis = load_and_prepare_dataset(dataset_path)
    
    if hetero_graphs is None:
        print("Failed to load dataset!")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split dataset
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
    )

    train_graphs = [hetero_graphs[i] for i in train_idx]
    test_graphs = [hetero_graphs[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    # Add labels to graphs
    for i, g in enumerate(train_graphs):
        g.y = torch.tensor([train_labels[i]], dtype=torch.long)
    for i, g in enumerate(test_graphs):
        g.y = torch.tensor([test_labels[i]], dtype=torch.long)

    print(f"Training: {len(train_graphs)}, Testing: {len(test_graphs)}")

    # Create data loaders
    batch_size = 32 if device.type == 'cuda' else 16
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = RobustEnhancedHGTDetector(
        feature_dims=feature_analysis,
        hidden_channels=192,
        num_heads=8,
        num_layers=3,
        dropout=0.17
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=10, min_lr=1e-6
    )

    # Training
    trainer = TrainingManager(model, device)
    print("Starting training...")
    
    results = trainer.train(
        train_loader, test_loader, criterion, optimizer,
        num_epochs=80, patience=15, scheduler=scheduler
    )

    return results

def find_dataset_files(directory="."):
    """Find dataset files in workspace"""
    search_dir = Path(directory)
    patterns = [
        '*enhanced*.pt',
        '*pharma*.pt',
        '*20k*.pt',
        '*.pt'
    ]

    found_files = []
    for pattern in patterns:
        files = list(search_dir.glob(pattern))
        found_files.extend(files)

    # Remove duplicates and sort
    found_files = list(set(found_files))
    found_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first

    return [str(f) for f in found_files]

def show_evaluation_menu():
    """Show evaluation options menu"""
    print("\n" + "="*60)
    print("EVALUATION/TRAINING OPTIONS")
    print("="*60)
    print("Choose how to evaluate the dataset:")
    print()
    print("1. Quick Evaluation (5 min)")
    print("   - Test on 1000 molecules subset")
    print("   - Fast estimate of model performance")
    print("   - No training, just evaluation")
    print()
    print("2. Transfer Learning (30 min)")
    print("   - Fine-tune pre-trained model")
    print("   - 20 epochs with lower learning rate")
    print("   - Best balance of time vs accuracy")
    print()
    print("3. Cross-Validation (15 min)")
    print("   - Statistical estimate with 3-fold CV")
    print("   - More reliable than single evaluation")
    print("   - No training, just evaluation")
    print()
    print("4. Full Re-training (2-3 hours)")
    print("   - Complete training on new dataset")
    print("   - Most accurate results")
    print("   - Recommended for final evaluation")
    print()
    print("0. Exit")
    print()

def main():
    """Main execution with interactive menu"""
    print("Enhanced HGT Training Script")
    print("="*60)
    print("Features:")
    print("• Multiple evaluation options")
    print("• Pre-trained model loading")
    print("• Interactive dataset selection")
    print("• Comprehensive logging")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")

    while True:
        # Find dataset files
        dataset_files = find_dataset_files()
        
        if not dataset_files:
            print("\nNo dataset files found!")
            print("Please run dataset_generator.py first to create a dataset.")
            return
        
        # Select dataset
        print(f"\nFound {len(dataset_files)} dataset file(s):")
        for i, file in enumerate(dataset_files, 1):
            file_size = Path(file).stat().st_size / (1024*1024)
            print(f"   {i}. {file} ({file_size:.1f} MB)")
        
        # Auto-select the newest dataset
        dataset_path = dataset_files[0]
        print(f"\nUsing: {Path(dataset_path).name}")

        try:
            # Load dataset
            print(f"\nLoading dataset from: {dataset_path}")
            hetero_graphs, labels, feature_analysis = load_and_prepare_dataset(dataset_path)
            
            if hetero_graphs is None:
                print("Failed to load dataset!")
                continue

            print(f"Dataset loaded: {len(hetero_graphs)} molecules")
            
            # Load pre-trained model
            model, baseline_f1 = load_pretrained_model(feature_analysis, device)
            
            # Show menu
            show_evaluation_menu()
            
            try:
                choice = input("Enter your choice (0-4): ").strip()
                
                if choice == '0':
                    print("Goodbye!")
                    return
                
                elif choice == '1':
                    # Quick evaluation
                    f1_score = quick_evaluation(model, hetero_graphs, device)
                    if baseline_f1:
                        improvement = ((f1_score - baseline_f1) / baseline_f1) * 100
                        print(f"vs Baseline: {improvement:+.1f}% change")
                    
                elif choice == '2':
                    # Transfer learning
                    if baseline_f1 is None:
                        print("No baseline model found. Running full training instead...")
                        results = full_training(dataset_path)
                    else:
                        results = transfer_learning_training(model, hetero_graphs, device)
                        if results:
                            improvement = ((results['best_f1'] - baseline_f1) / baseline_f1) * 100
                            print(f"\nTransfer Learning Results:")
                            print(f"   Baseline F1: {baseline_f1:.3f}")
                            print(f"   New F1: {results['best_f1']:.3f}")
                            print(f"   Improvement: {improvement:+.1f}%")
                
                elif choice == '3':
                    # Cross-validation
                    avg_f1 = cross_validation_estimate(model, hetero_graphs, device)
                    if baseline_f1:
                        improvement = ((avg_f1 - baseline_f1) / baseline_f1) * 100
                        print(f"vs Baseline ({baseline_f1:.3f}): {improvement:+.1f}% change")
                
                elif choice == '4':
                    # Full retraining
                    print("\nStarting full retraining...")
                    results = full_training(dataset_path)
                    if results and baseline_f1:
                        improvement = ((results['best_f1'] - baseline_f1) / baseline_f1) * 100
                        print(f"\nFull Training Results:")
                        print(f"   Baseline F1: {baseline_f1:.3f}")
                        print(f"   New F1: {results['best_f1']:.3f}")
                        print(f"   Improvement: {improvement:+.1f}%")
                
                else:
                    print("Invalid choice. Please enter 0-4.")
                    continue
                
                # Ask if user wants to continue
                print(f"\n{'='*60}")
                continue_choice = input("Would you like to try another option? (y/N): ").lower()
                if continue_choice not in ['y', 'yes']:
                    print("Thanks for using the HGT training script!")
                    return
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                continue
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue


if __name__ == "__main__":
    main()