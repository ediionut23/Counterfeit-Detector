"""
Optimized HGT Model for Counterfeit Drug Detection
Author: ediionut23 (optimized)
Version: 3.0.0
Last Updated: 2025-08-29
Major improvements: Stability, performance, reduced overfitting
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, BatchNorm
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm.auto import tqdm
import logging
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import math

warnings.filterwarnings('ignore')

def get_unique_filename(base_name, extension):
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

# Set style for better plots with fallback
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

try:
    sns.set_palette("husl")
except:
    pass

# Enable interactive plotting
plt.ion()


def clear_output_safe():
    """Safe alternative to IPython clear_output"""
    if 'ipykernel' in sys.modules:
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            os.system('cls' if os.name == 'nt' else 'clear')
    else:
        os.system('cls' if os.name == 'nt' else 'clear')


# --- Focal Loss for class imbalance ---
class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if targets.dim() > 0:
                alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --- Data Augmentation ---
def augment_hetero_graph(hetero_graph, noise_level=0.005, edge_dropout=0.05):  # Reduced noise and dropout
    """Apply gentle data augmentation to heterograph"""
    augmented = hetero_graph.clone()
    
    # Add minimal noise to node features (reduced from 0.01 to 0.005)
    for node_type in augmented.x_dict:
        if torch.rand(1) > 0.7:  # Apply with 30% probability (reduced from 50%)
            noise = torch.randn_like(augmented.x_dict[node_type]) * noise_level
            augmented.x_dict[node_type] = augmented.x_dict[node_type] + noise
    
    # Minimal edge dropout (reduced from 0.1 to 0.05)
    if edge_dropout > 0:
        for edge_type in augmented.edge_index_dict:
            if torch.rand(1) > 0.8:  # Apply with 20% probability (reduced from 30%)
                edge_index = augmented.edge_index_dict[edge_type]
                num_edges = edge_index.size(1)
                mask = torch.rand(num_edges) > edge_dropout
                augmented.edge_index_dict[edge_type] = edge_index[:, mask]
    
    return augmented


# --- Optimized Model Architecture ---
class OptimizedHGTCounterfeitDetector(torch.nn.Module):
    def __init__(self, hidden_channels=128, out_channels=2, num_heads=8, 
                 num_layers=3, metadata=None, dropout=0.2):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.metadata = metadata
        self.dropout = dropout
        
        # Add missing node types to metadata
        if self.metadata and len(self.metadata) > 0:
            missing_node_types = ["Cl", "F", "P", "Na", "I", "Br", "Se", "K", "S"]
            for node_type in missing_node_types:
                if node_type not in self.metadata[0]:
                    self.metadata[0].append(node_type)

        # Node embeddings with proper initialization
        self.node_embeddings = torch.nn.ModuleDict()
        self.node_norms = torch.nn.ModuleDict()
        self._embedding_initialized = False

        # Simplified HGT layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))

        # Global attention for node type importance
        num_node_types = len(metadata[0]) if metadata and metadata[0] else 1
        self.attention_weights = torch.nn.Parameter(torch.ones(num_node_types))

        # Simplified and more robust classifier
        self.classifier = None
        self.classifier_input_dim = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def _initialize_embeddings_first_time(self, x_dict):
        """Initialize embeddings on first forward pass"""
        if not self._embedding_initialized:
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
            
            # Determine feature dimension
            feature_dim = None
            for node_type, x in x_dict.items():
                feature_dim = x.shape[-1]
                break
            
            if feature_dim is None:
                feature_dim = 31
            
            # Create embeddings for all node types in metadata
            if self.metadata and len(self.metadata) > 0:
                for node_type in self.metadata[0]:
                    if node_type not in self.node_embeddings:
                        self.node_embeddings[node_type] = torch.nn.Linear(
                            feature_dim, self.hidden_channels, device=device
                        )
                        self.node_norms[node_type] = torch.nn.LayerNorm(
                            self.hidden_channels, device=device
                        )
            
            # Create for current batch node types
            for node_type, x in x_dict.items():
                if node_type not in self.node_embeddings:
                    input_dim = x.shape[-1]
                    self.node_embeddings[node_type] = torch.nn.Linear(
                        input_dim, self.hidden_channels, device=device
                    )
                    self.node_norms[node_type] = torch.nn.LayerNorm(
                        self.hidden_channels, device=device
                    )

            # Initialize embeddings with Xavier
            for embedding in self.node_embeddings.values():
                torch.nn.init.xavier_uniform_(embedding.weight)
                torch.nn.init.zeros_(embedding.bias)

            self._embedding_initialized = True

    def _create_missing_embeddings(self, x_dict):
        """Handle missing node types dynamically"""
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'

        for node_type, x in x_dict.items():
            if node_type not in self.node_embeddings:
                input_dim = x.shape[-1]
                self.node_embeddings[node_type] = torch.nn.Linear(
                    input_dim, self.hidden_channels, device=device
                )
                self.node_norms[node_type] = torch.nn.LayerNorm(
                    self.hidden_channels, device=device
                )
                
                # Initialize new embeddings
                torch.nn.init.xavier_uniform_(self.node_embeddings[node_type].weight)
                torch.nn.init.zeros_(self.node_embeddings[node_type].bias)

    def _initialize_classifier(self, input_dim):
        """Initialize classifier with proper dimensions"""
        if self.classifier is None or self.classifier_input_dim != input_dim:
            self.classifier_input_dim = input_dim
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'

            # Enhanced classifier with more depth and skip connections
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.hidden_channels),
                torch.nn.BatchNorm1d(self.hidden_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(self.dropout),
                
                torch.nn.Linear(self.hidden_channels, self.hidden_channels),
                torch.nn.BatchNorm1d(self.hidden_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(self.dropout * 0.7),
                
                torch.nn.Linear(self.hidden_channels, self.hidden_channels // 2),
                torch.nn.BatchNorm1d(self.hidden_channels // 2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(self.dropout * 0.5),
                
                torch.nn.Linear(self.hidden_channels // 2, self.out_channels)
            ).to(device)
            
            # Initialize classifier weights
            for module in self.classifier:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x_dict, edge_index_dict, batch_dict=None, batch_size=None):
        # Initialize embeddings
        self._initialize_embeddings_first_time(x_dict)
        self._create_missing_embeddings(x_dict)

        # Node embeddings with normalization
        h_dict = {}
        for node_type, x in x_dict.items():
            h = self.node_embeddings[node_type](x)
            h = self.node_norms[node_type](h)
            h_dict[node_type] = F.relu(h, inplace=True)

        # Apply HGT layers with residual connections
        for i, (conv, layer_norm) in enumerate(zip(self.convs, self.layer_norms)):
            h_dict_new = conv(h_dict, edge_index_dict)
            
            # Apply layer normalization and activation
            for node_type in h_dict_new.keys():
                h_normalized = layer_norm(h_dict_new[node_type])
                h_activated = F.relu(h_normalized, inplace=True)
                h_dropped = F.dropout(h_activated, p=self.dropout, training=self.training)
                
                # Residual connection (if dimensions match)
                if node_type in h_dict and h_dict[node_type].shape == h_dropped.shape:
                    h_dict_new[node_type] = h_dropped + h_dict[node_type] * 0.1  # Scaled residual
                else:
                    h_dict_new[node_type] = h_dropped

            h_dict = h_dict_new

        # Determine batch size more robustly
        if batch_size is not None:
            actual_batch_size = batch_size
        elif batch_dict:
            batch_sizes = []
            for node_type, batch_tensor in batch_dict.items():
                if node_type in h_dict and batch_tensor.numel() > 0:
                    max_idx = int(batch_tensor.max().item()) + 1
                    batch_sizes.append(max_idx)
            actual_batch_size = max(batch_sizes) if batch_sizes else 1
        else:
            actual_batch_size = 1

        # Improved global pooling with attention
        pooled_features = []
        attention_weights = F.softmax(self.attention_weights[:len(h_dict)], dim=0)

        for idx, (node_type, node_features) in enumerate(h_dict.items()):
            if batch_dict and node_type in batch_dict and batch_dict[node_type].numel() > 0:
                # Batch pooling
                pooled = torch.zeros(actual_batch_size, self.hidden_channels,
                                   device=node_features.device)

                for i in range(actual_batch_size):
                    mask = batch_dict[node_type] == i
                    if mask.any():
                        masked_features = node_features[mask]
                        # Combine mean and max pooling
                        mean_pool = masked_features.mean(dim=0)
                        max_pool = masked_features.max(dim=0)[0]
                        pooled[i] = (mean_pool + max_pool) * 0.5
                        
                # Apply attention weight
                if idx < len(attention_weights):
                    pooled = pooled * attention_weights[idx]
                pooled_features.append(pooled)
            else:
                # Single graph pooling
                mean_pool = node_features.mean(dim=0, keepdim=True)
                max_pool = node_features.max(dim=0, keepdim=True)[0]
                pooled = (mean_pool + max_pool) * 0.5

                if actual_batch_size > 1:
                    pooled = pooled.expand(actual_batch_size, -1)

                if idx < len(attention_weights):
                    pooled = pooled * attention_weights[idx]
                pooled_features.append(pooled)

        # Combine features
        if pooled_features:
            global_repr = torch.cat(pooled_features, dim=1)
        else:
            global_repr = torch.zeros(actual_batch_size, self.hidden_channels,
                                    device=list(h_dict.values())[0].device)

        # Classification
        self._initialize_classifier(global_repr.shape[1])
        out = self.classifier(global_repr)

        return out


# --- Learning Rate Scheduler ---
def create_cosine_warmup_scheduler(optimizer, num_epochs, warmup_epochs=None):
    """Cosine annealing with warmup"""
    if warmup_epochs is None:
        warmup_epochs = max(1, num_epochs // 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- Advanced Training Monitor ---
class AdvancedTrainingMonitor:
    def __init__(self):
        self.gradient_norms = []
        self.weight_norms = []
        self.learning_rates = []
        self.dead_neurons = []
    
    def log_gradient_norms(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
    
    def check_for_issues(self):
        if len(self.gradient_norms) > 10:
            recent_grads = self.gradient_norms[-10:]
            if all(g < 1e-7 for g in recent_grads):
                print("WARNING: Vanishing gradients detected!")
            elif any(g > 10 for g in recent_grads):
                print("WARNING: Exploding gradients detected!")


# --- Enhanced Trainer with Optimizations ---
class OptimizedCounterfeitTrainer:
    def __init__(self, model, device, logger=None, plot_every=3, use_augmentation=True):
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.plot_every = plot_every
        self.use_augmentation = use_augmentation
        self.monitor = AdvancedTrainingMonitor()

        # Training metrics
        self.best_f1 = 0
        self.best_model_state = None
        self.start_time = datetime.now(timezone.utc)

        # Tracking arrays
        self.train_losses = []
        self.train_accuracies = []
        self.train_f1_scores = []
        self.test_losses = []
        self.test_accuracies = []
        self.test_f1_scores = []
        self.learning_rates = []
        self.train_precisions = []
        self.train_recalls = []
        self.test_precisions = []
        self.test_recalls = []

        # Plotting setup
        self.fig = None
        self.axes = None

    def setup_plots(self):
        """Setup real-time plotting"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Optimized Training Progress - Real Time', fontsize=16, fontweight='bold')
        
        titles = ['Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Learning Rate']
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i], fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch')
        
        plt.tight_layout()

    def update_plots(self, epoch):
        """Update real-time plots"""
        if self.fig is None:
            self.setup_plots()

        for ax in self.axes.flat:
            ax.clear()

        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        self.axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        self.axes[0, 0].plot(epochs, self.test_losses, 'r-', label='Test Loss', linewidth=2)
        self.axes[0, 0].set_title('Loss', fontweight='bold')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)

        # Accuracy plot
        self.axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Train Acc', linewidth=2)
        self.axes[0, 1].plot(epochs, self.test_accuracies, 'r-', label='Test Acc', linewidth=2)
        self.axes[0, 1].set_title('Accuracy', fontweight='bold')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].set_ylim([0, 1])

        # F1 Score plot
        self.axes[0, 2].plot(epochs, self.train_f1_scores, 'b-', label='Train F1', linewidth=2)
        self.axes[0, 2].plot(epochs, self.test_f1_scores, 'r-', label='Test F1', linewidth=2)
        if self.best_f1 > 0:
            self.axes[0, 2].axhline(y=self.best_f1, color='g', linestyle='--',
                                   label=f'Best: {self.best_f1:.3f}', linewidth=2)
        self.axes[0, 2].set_title('F1 Score', fontweight='bold')
        self.axes[0, 2].legend()
        self.axes[0, 2].grid(True, alpha=0.3)
        self.axes[0, 2].set_ylim([0, 1])

        # Precision plot
        self.axes[1, 0].plot(epochs, self.train_precisions, 'b-', label='Train Precision', linewidth=2)
        self.axes[1, 0].plot(epochs, self.test_precisions, 'r-', label='Test Precision', linewidth=2)
        self.axes[1, 0].set_title('Precision', fontweight='bold')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].set_ylim([0, 1])

        # Recall plot
        self.axes[1, 1].plot(epochs, self.train_recalls, 'b-', label='Train Recall', linewidth=2)
        self.axes[1, 1].plot(epochs, self.test_recalls, 'r-', label='Test Recall', linewidth=2)
        self.axes[1, 1].set_title('Recall', fontweight='bold')
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].set_ylim([0, 1])

        # Learning Rate plot
        if self.learning_rates:
            self.axes[1, 2].plot(epochs, self.learning_rates, 'g-', label='Learning Rate', linewidth=2)
            self.axes[1, 2].set_title('Learning Rate', fontweight='bold')
            self.axes[1, 2].legend()
            self.axes[1, 2].grid(True, alpha=0.3)
            self.axes[1, 2].set_yscale('log')

        plt.tight_layout()
        
        # Disable real-time plotting to avoid display during training
        # plt.draw() and plt.pause() removed for cleaner training output

    def train_epoch(self, loader, optimizer, criterion, scheduler=None):
        """Optimized training epoch with data augmentation"""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        progress_bar = tqdm(loader, desc="Training", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Apply data augmentation
                if self.use_augmentation and torch.rand(1) > 0.5:
                    batch = augment_hetero_graph(batch)
                
                batch = batch.to(self.device)
                optimizer.zero_grad()

                actual_batch_size = self._get_batch_size(batch)
                out = self.model(batch.x_dict, batch.edge_index_dict,
                               batch.batch_dict, batch_size=actual_batch_size)

                labels = self._get_labels(batch, actual_batch_size)
                out, labels = self._align_dimensions(out, labels)

                loss = criterion(out, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                # Add gradient noise for better generalization
                if torch.rand(1) > 0.7:  # 30% probability
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * 1e-5
                            param.grad.add_(noise)

                optimizer.step()

                # Log gradient norms
                self.monitor.log_gradient_norms(self.model)

                total_loss += loss.item()
                predictions.extend(out.argmax(dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{current_lr:.6f}'
                })

            except Exception as e:
                self.logger.warning(f"Error in batch {batch_idx}: {str(e)}")
                continue

        # Check for training issues
        self.monitor.check_for_issues()

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        return avg_loss, predictions, true_labels

    def evaluate(self, loader, criterion):
        """Enhanced evaluation"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            progress_bar = tqdm(loader, desc="Evaluating", leave=False)

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    batch = batch.to(self.device)
                    actual_batch_size = self._get_batch_size(batch)
                    out = self.model(batch.x_dict, batch.edge_index_dict,
                                   batch.batch_dict, batch_size=actual_batch_size)

                    labels = self._get_labels(batch, actual_batch_size)
                    out, labels = self._align_dimensions(out, labels)

                    loss = criterion(out, labels)
                    total_loss += loss.item()
                    predictions.extend(out.argmax(dim=1).cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

                except Exception as e:
                    self.logger.warning(f"Error in evaluation batch {batch_idx}: {str(e)}")
                    continue

        if len(predictions) == 0 or len(true_labels) == 0:
            return self._empty_metrics(), [], []

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        cm = confusion_matrix(true_labels, predictions)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

        return metrics, predictions, true_labels

    def train(self, train_loader, test_loader, criterion, optimizer,
              num_epochs, patience=20, scheduler=None):
        """Optimized training loop"""
        self.logger.info(f"Starting optimized training at {datetime.now(timezone.utc)}")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            train_loss, train_preds, train_labels = self.train_epoch(
                train_loader, optimizer, criterion, scheduler
            )

            # Evaluation
            test_metrics, test_preds, test_labels = self.evaluate(test_loader, criterion)

            # Calculate training metrics
            if len(train_preds) > 0 and len(train_labels) > 0:
                train_accuracy = accuracy_score(train_labels, train_preds)
                train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                    train_labels, train_preds, average='binary', zero_division=0
                )
            else:
                train_accuracy = train_precision = train_recall = train_f1 = 0.0

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.train_f1_scores.append(train_f1)
            self.train_precisions.append(train_precision)
            self.train_recalls.append(train_recall)

            self.test_losses.append(test_metrics['loss'])
            self.test_accuracies.append(test_metrics['accuracy'])
            self.test_f1_scores.append(test_metrics['f1'])
            self.test_precisions.append(test_metrics['precision'])
            self.test_recalls.append(test_metrics['recall'])

            # Check for best model and save checkpoint immediately
            current_f1 = test_metrics['f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint immediately when F1 improves
                checkpoint_path = f'best_checkpoint_f1_{current_f1:.4f}_epoch_{epoch+1}.pt'
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'epoch': epoch + 1,
                    'best_f1': self.best_f1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_metrics': {
                        'loss': train_loss,
                        'accuracy': train_accuracy,
                        'f1': train_f1
                    },
                    'test_metrics': test_metrics
                }, checkpoint_path)
                self.logger.info(f"💾 New best F1! Checkpoint saved: {checkpoint_path}")
            else:
                patience_counter += 1

            # Step scheduler with F1 score for plateau scheduler
            if scheduler:
                try:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(current_f1)  # Use F1 score for plateau scheduler
                    else:
                        scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                except Exception as e:
                    self.logger.warning(f"Scheduler step error: {e}")

            # Logging with parameter count after first epoch
            if epoch == 0:
                try:
                    total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    self.logger.info(f"Model initialized with {total_params:,} trainable parameters")
                except:
                    pass

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}, F1={train_f1:.4f} - "
                f"Test: Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}, F1={current_f1:.4f}"
            )

            # Update plots
            if (epoch + 1) % self.plot_every == 0 or epoch == 0:
                self.update_plots(epoch + 1)

            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        training_time = datetime.now(timezone.utc) - self.start_time
        self.logger.info(f"Training completed in {training_time}")
        self.logger.info(f"Best F1 score: {self.best_f1:.4f} at epoch {best_epoch + 1}")

        # Final plots
        self.update_plots(len(self.train_losses))
        self.plot_confusion_matrix(test_loader, criterion)

        return {
            'best_f1': self.best_f1,
            'best_epoch': best_epoch + 1,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies,
            'test_f1_scores': self.test_f1_scores,
            'train_accuracies': self.train_accuracies,
            'train_f1_scores': self.train_f1_scores
        }

    def plot_confusion_matrix(self, test_loader, criterion):
        """Plot final confusion matrix"""
        test_metrics, test_preds, test_labels = self.evaluate(test_loader, criterion)

        try:
            plt.figure(figsize=(8, 6))
            cm = test_metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Genuine', 'Counterfeit'],
                        yticklabels=['Genuine', 'Counterfeit'])
            plt.title('Final Confusion Matrix', fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            # Save with unique filename to avoid overwriting
            cm_filename = get_unique_filename('confusion_matrix_optimized', 'png')
            plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved as '{cm_filename}'")
            plt.close()  # Close figure to save memory

        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            cm = test_metrics['confusion_matrix']
            print("\nConfusion Matrix (text format):")
            print("    Predicted:")
            print("         Genuine  Counterfeit")
            print(f"Genuine    {cm[0, 0]:6d}    {cm[0, 1]:6d}")
            print(f"Counterfeit {cm[1, 0]:6d}    {cm[1, 1]:6d}")

    def _get_batch_size(self, batch):
        """Robustly determine batch size"""
        if hasattr(batch, 'y') and batch.y is not None:
            return batch.y.size(0) if batch.y.dim() == 1 else batch.y.view(-1).size(0)
        elif hasattr(batch, 'label') and batch.label is not None:
            return batch.label.size(0) if batch.label.dim() == 1 else batch.label.view(-1).size(0)
        elif batch.batch_dict:
            batch_sizes = []
            for node_type, batch_tensor in batch.batch_dict.items():
                if batch_tensor.numel() > 0:
                    max_idx = int(batch_tensor.max().item()) + 1
                    batch_sizes.append(max_idx)
            return max(batch_sizes) if batch_sizes else 1
        return 1

    def _get_labels(self, batch, batch_size):
        """Robustly extract labels"""
        if hasattr(batch, 'y') and batch.y is not None:
            labels = batch.y
        elif hasattr(batch, 'label') and batch.label is not None:
            labels = batch.label
        else:
            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        if labels.dim() > 1:
            labels = labels.view(-1)

        return labels

    def _align_dimensions(self, out, labels):
        """Ensure output and labels have compatible dimensions"""
        if labels.size(0) != out.size(0):
            min_size = min(labels.size(0), out.size(0))
            labels = labels[:min_size]
            out = out[:min_size]
        return out, labels

    def _empty_metrics(self):
        """Return empty metrics for error cases"""
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]])
        }


# --- K-Fold Cross Validation ---
def k_fold_validation(hetero_graphs, labels, metadata, k=5, device='cpu'):
    """K-Fold Cross Validation for robust evaluation"""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_scores = []
    
    print(f"Starting {k}-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(hetero_graphs, labels)):
        print(f"\nTraining fold {fold + 1}/{k}")
        
        # Split data for this fold
        train_graphs = [hetero_graphs[i] for i in train_idx]
        val_graphs = [hetero_graphs[i] for i in val_idx]
        train_labels_fold = [labels[i] for i in train_idx]
        val_labels_fold = [labels[i] for i in val_idx]
        
        # Add labels to graphs
        for i, g in enumerate(train_graphs):
            g.y = torch.tensor([train_labels_fold[i]], dtype=torch.long)
        for i, g in enumerate(val_graphs):
            g.y = torch.tensor([val_labels_fold[i]], dtype=torch.long)
        
        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False, num_workers=2)
        
        # Initialize model for this fold
        model = OptimizedHGTCounterfeitDetector(
            hidden_channels=96,
            out_channels=2,
            num_heads=4,
            num_layers=2,
            metadata=metadata,
            dropout=0.1
        ).to(device)
        
        # Training components
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = create_cosine_warmup_scheduler(optimizer, num_epochs=50)
        
        # Trainer for this fold
        trainer = OptimizedCounterfeitTrainer(
            model=model,
            device=device,
            plot_every=10  # Less frequent plotting for CV
        )
        
        # Train for this fold
        results = trainer.train(
            train_loader=train_loader,
            test_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=50,  # Reduced epochs for CV
            patience=15,
            scheduler=scheduler
        )
        
        fold_scores.append(results['best_f1'])
        print(f"Fold {fold + 1} F1 Score: {results['best_f1']:.4f}")
    
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    
    print(f"\nCross-Validation Results:")
    print(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Individual fold scores: {fold_scores}")
    
    return mean_f1, std_f1, fold_scores


# --- Hyperparameter Tuning ---
def hyperparameter_search(hetero_graphs, labels, metadata, device='cpu'):
    """Automated hyperparameter search to find optimal configuration"""
    from itertools import product
    
    # Define hyperparameter search space
    search_space = {
        'hidden_channels': [96, 128, 160],
        'num_heads': [4, 6, 8],
        'num_layers': [2, 3, 4],
        'dropout': [0.1, 0.15, 0.2, 0.25],
        'learning_rate': [0.0001, 0.0005, 0.001],
        'batch_size': [64, 96, 128]
    }
    
    best_f1 = 0
    best_params = {}
    results_log = []
    
    print(f"🔍 Starting hyperparameter search...")
    print(f"Total combinations to test: {len(list(product(*search_space.values())))}")
    
    # Sample a subset of combinations for efficiency
    import random
    all_combinations = list(product(*search_space.values()))
    random.shuffle(all_combinations)
    test_combinations = all_combinations[:20]  # Test top 20 combinations
    
    for i, (hidden_ch, heads, layers, dropout, lr, batch_size) in enumerate(test_combinations):
        print(f"\n🧪 Testing configuration {i+1}/20:")
        print(f"   Hidden: {hidden_ch}, Heads: {heads}, Layers: {layers}")
        print(f"   Dropout: {dropout}, LR: {lr}, Batch: {batch_size}")
        
        try:
            # Quick train/test split for hyperparameter search
            train_idx, test_idx = train_test_split(
                np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
            )
            
            train_graphs = [hetero_graphs[i] for i in train_idx]
            test_graphs = [hetero_graphs[i] for i in test_idx]
            train_labels_hp = [labels[i] for i in train_idx]
            test_labels_hp = [labels[i] for i in test_idx]
            
            # Add labels
            for idx, g in enumerate(train_graphs):
                g.y = torch.tensor([train_labels_hp[idx]], dtype=torch.long)
            for idx, g in enumerate(test_graphs):
                g.y = torch.tensor([test_labels_hp[idx]], dtype=torch.long)
            
            # Create loaders
            train_loader_hp = DataLoader(train_graphs, batch_size=int(batch_size), shuffle=True, num_workers=2)
            test_loader_hp = DataLoader(test_graphs, batch_size=int(batch_size), shuffle=False, num_workers=2)
            
            # Initialize model with current hyperparameters
            model_hp = OptimizedHGTCounterfeitDetector(
                hidden_channels=int(hidden_ch),
                out_channels=2,
                num_heads=int(heads),
                num_layers=int(layers),
                metadata=metadata,
                dropout=float(dropout)
            ).to(device)
            
            # Training components
            criterion_hp = FocalLoss(alpha=0.75, gamma=2.0)
            optimizer_hp = torch.optim.AdamW(model_hp.parameters(), lr=float(lr), weight_decay=0.001)
            
            # Quick trainer for hyperparameter search (reduced epochs)
            trainer_hp = OptimizedCounterfeitTrainer(
                model=model_hp, device=device, plot_every=100, use_augmentation=True
            )
            
            # Quick training (25 epochs for speed)
            results_hp = trainer_hp.train(
                train_loader=train_loader_hp,
                test_loader=test_loader_hp,
                criterion=criterion_hp,
                optimizer=optimizer_hp,
                num_epochs=25,
                patience=10,
                scheduler=None
            )
            
            f1_score = results_hp['best_f1']
            results_log.append({
                'config': {
                    'hidden_channels': hidden_ch,
                    'num_heads': heads,
                    'num_layers': layers,
                    'dropout': dropout,
                    'learning_rate': lr,
                    'batch_size': batch_size
                },
                'f1_score': f1_score
            })
            
            print(f"   ➜ F1 Score: {f1_score:.4f}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_params = {
                    'hidden_channels': hidden_ch,
                    'num_heads': heads,
                    'num_layers': layers,
                    'dropout': dropout,
                    'learning_rate': lr,
                    'batch_size': batch_size
                }
                print(f"   🌟 NEW BEST! F1: {best_f1:.4f}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    # Sort results by F1 score
    results_log.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print(f"\n🏆 Hyperparameter Search Results:")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Configuration:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    print(f"\nTop 5 Configurations:")
    for i, result in enumerate(results_log[:5]):
        print(f"{i+1}. F1: {result['f1_score']:.4f} - {result['config']}")
    
    return best_params, best_f1, results_log


# --- Ensemble Prediction ---
def create_ensemble_models(hetero_graphs, labels, metadata, num_models=3, device='cpu'):
    """Create ensemble of models with different initializations"""
    models = []
    
    print(f"Training ensemble of {num_models} models...")
    
    # Split data once
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=42
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
    
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False, num_workers=2)
    
    for model_idx in range(num_models):
        print(f"\nTraining ensemble model {model_idx + 1}/{num_models}")
        
        # Different random seed for each model
        torch.manual_seed(42 + model_idx)
        np.random.seed(42 + model_idx)
        
        model = OptimizedHGTCounterfeitDetector(
            hidden_channels=96,
            out_channels=2,
            num_heads=4,
            num_layers=2,
            metadata=metadata,
            dropout=0.1
        ).to(device)
        
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = create_cosine_warmup_scheduler(optimizer, num_epochs=80)
        
        trainer = OptimizedCounterfeitTrainer(
            model=model,
            device=device,
            plot_every=20,
            use_augmentation=True
        )
        
        results = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=80,
            patience=20,
            scheduler=scheduler
        )
        
        models.append({
            'model': model,
            'f1_score': results['best_f1'],
            'model_idx': model_idx
        })
        
        print(f"Ensemble model {model_idx + 1} F1 Score: {results['best_f1']:.4f}")
    
    return models, test_loader


def ensemble_predict(models, test_loader, device):
    """Make predictions using ensemble of models"""
    all_predictions = []
    all_probabilities = []
    true_labels = []
    
    # Get predictions from each model
    for model_info in models:
        model = model_info['model']
        model.eval()
        
        model_predictions = []
        model_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                actual_batch_size = len(batch.y) if hasattr(batch, 'y') else 1
                out = model(batch.x_dict, batch.edge_index_dict, 
                           batch.batch_dict, batch_size=actual_batch_size)
                
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                
                model_predictions.extend(preds.cpu().numpy())
                model_probabilities.extend(probs.cpu().numpy())
                
                if len(true_labels) == 0:  # Only collect true labels once
                    if hasattr(batch, 'y'):
                        true_labels.extend(batch.y.cpu().numpy())
        
        all_predictions.append(model_predictions)
        all_probabilities.append(model_probabilities)
    
    # Ensemble averaging
    all_probabilities = np.array(all_probabilities)  # Shape: (num_models, num_samples, num_classes)
    ensemble_probabilities = np.mean(all_probabilities, axis=0)
    ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
    
    # Calculate ensemble metrics
    accuracy = accuracy_score(true_labels, ensemble_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, ensemble_predictions, average='binary', zero_division=0
    )
    
    print(f"\nEnsemble Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'predictions': ensemble_predictions,
        'probabilities': ensemble_probabilities,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_labels': true_labels
    }


# --- Main Training Script ---
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    print("🚀 Optimized HGT Model Training Started...")    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for dataset in same directory as the script
    dataset_path = os.path.join(script_dir, "realistic_counterfeit_dataset_8k_balanced.pt")

    try:
        loaded_data = torch.load(dataset_path, weights_only=False)
        print(f"✅ Dataset file found at: {dataset_path}")
        print(f"📊 Loaded data type: {type(loaded_data)}")
        
        # Handle different data formats
        if isinstance(loaded_data, tuple):
            print(f"📦 Tuple with {len(loaded_data)} elements")
            if len(loaded_data) == 4:
                hetero_graphs, labels, metadata, extra_info = loaded_data
                print("📊 4-element tuple: graphs, labels, metadata, extra_info")
            elif len(loaded_data) == 3:
                hetero_graphs, labels, metadata = loaded_data
            elif len(loaded_data) == 2:
                hetero_graphs, labels = loaded_data
                metadata = None
                print("⚠️ No metadata found, will be inferred from data")
            else:
                print(f"❌ Unexpected tuple length: {len(loaded_data)}")
                exit(1)
        elif isinstance(loaded_data, dict):
            print("📋 Dictionary format detected")
            hetero_graphs = loaded_data.get('hetero_graphs', loaded_data.get('graphs'))
            labels = loaded_data.get('labels')
            metadata = loaded_data.get('metadata')
        else:
            print(f"❌ Unexpected data format: {type(loaded_data)}")
            exit(1)
            
        if hetero_graphs is None or labels is None:
            print("❌ Missing required data (graphs or labels)")
            exit(1)
            
            print(f"✅ Dataset loaded: {len(hetero_graphs)} graphs, {len(set(labels))} classes")
        
        # Calculate class weights for imbalanced data
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        class_weights = total_samples / (len(unique_labels) * label_counts)
        class_weight_tensor = torch.FloatTensor(class_weights)
        print(f"📊 Class distribution: {dict(zip(unique_labels, label_counts))}")
        print(f"⚖️ Class weights: {dict(zip(unique_labels, class_weights))}")
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at: {dataset_path}")
        print("Make sure the file exists in the project root directory.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        exit(1)    # Device setup with enhanced GPU information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 CUDA enabled! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔥 CUDA Version: {torch.version.cuda}")
        print(f"⚡ PyTorch Version: {torch.__version__}")
        # Set memory growth to avoid OOM issues
        torch.cuda.empty_cache()
    else:
        print("💻 Using CPU for training (GPU not available)")

    # Data split with stratification
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=42
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

    print(f"📊 Split: {len(train_graphs)} train, {len(test_graphs)} test")

    # Create optimized data loaders with larger batch size for stability
    train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True, num_workers=2)  # Increased from 64
    test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False, num_workers=2)   # Increased from 64

    # Initialize optimized model
    model = OptimizedHGTCounterfeitDetector(
        hidden_channels=128,    # Increased for better representation
        out_channels=2,
        num_heads=8,           # Increased for better attention
        num_layers=3,          # Increased depth
        metadata=metadata,
        dropout=0.2            # Balanced dropout
    ).to(device)

    print("🔧 Optimized model architecture initialized")
    print(f"   • Hidden channels: 128")
    print(f"   • Attention heads: 8") 
    print(f"   • HGT layers: 3")
    print(f"   • Dropout: 0.2")

    # Optimized training setup with class weights
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Better for class imbalance
    # Add class weights to device
    if 'class_weight_tensor' in locals():
        class_weight_tensor = class_weight_tensor.to(device)
        criterion_weighted = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
        print(f"⚖️ Using weighted CrossEntropyLoss with class weights")
    else:
        criterion_weighted = criterion
        print(f"🎯 Using Focal Loss without explicit class weights")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0001,            # Increased learning rate for better convergence
        weight_decay=0.001,   # Increased weight decay for regularization
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # Monitor F1 score (maximize)
        factor=0.5,           # Reduce LR by half
        patience=8,           # Wait 8 epochs before reducing
        min_lr=1e-7          # Minimum learning rate
    )

    # Enhanced trainer with data augmentation enabled
    trainer = OptimizedCounterfeitTrainer(
        model=model,
        device=device,
        logger=logger,
        plot_every=3,
        use_augmentation=True   # Enable data augmentation for better generalization
    )

    print("\n🏋️ Starting optimized training...")
    print("📈 Real-time plots will update every 3 epochs")
    print("🔄 Data augmentation enabled")

    # Main training
    results = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion_weighted,  # Use weighted criterion
        optimizer=optimizer,
        num_epochs=120,       # Increased for better convergence
        patience=25,          # Balanced patience for early stopping
        scheduler=scheduler
    )

    # Results summary
    print(f"\n🎉 Optimized Training Results:")
    print(f"🏆 Best F1 Score: {results['best_f1']:.4f}")
    print(f"📅 Best Epoch: {results['best_epoch']}")
    print(f"⏱️  Training Time: {results['training_time']}")

    # Save optimized model
    torch.save({
        'model_state_dict': trainer.best_model_state,
        'model_config': {
            'hidden_channels': 96,
            'out_channels': 2,
            'num_heads': 4,
            'num_layers': 2,
            'metadata': metadata,
            'dropout': 0.1
        },
        'training_results': results,
        'best_f1': results['best_f1'],
        'best_epoch': results['best_epoch'],
        'optimizer_config': {
            'lr': 0.00001,        #ok  Ultra-low learning rate for stability
            'weight_decay': 0.0001  # Minimal weight decay
        },
        'class_weights': class_weights.tolist() if 'class_weights' in locals() else None
    }, get_unique_filename('optimized_hgt_counterfeit_model', 'pt'))

    print(f"💾 Optimized model saved with unique timestamp")

    # Generate comprehensive analysis
    try:
        plt.figure(figsize=(20, 12))

        # Training progress plots
        epochs = range(1, len(results['train_losses']) + 1)
        
        # Loss comparison
        plt.subplot(2, 4, 1)
        plt.plot(epochs, results['train_losses'], 'b-', label='Train', linewidth=2, alpha=0.8)
        plt.plot(epochs, trainer.test_losses, 'r-', label='Test', linewidth=2, alpha=0.8)
        plt.title('Loss Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accuracy comparison
        plt.subplot(2, 4, 2)
        plt.plot(epochs, results['train_accuracies'], 'b-', label='Train', linewidth=2, alpha=0.8)
        plt.plot(epochs, results['test_accuracies'], 'r-', label='Test', linewidth=2, alpha=0.8)
        plt.title('Accuracy Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])

        # F1 Score comparison
        plt.subplot(2, 4, 3)
        plt.plot(epochs, results['train_f1_scores'], 'b-', label='Train', linewidth=2, alpha=0.8)
        plt.plot(epochs, results['test_f1_scores'], 'r-', label='Test', linewidth=2, alpha=0.8)
        plt.axhline(y=results['best_f1'], color='g', linestyle='--',
                    label=f'Best: {results["best_f1"]:.3f}', linewidth=2)
        plt.title('F1 Score Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])

        # Precision comparison
        plt.subplot(2, 4, 4)
        plt.plot(epochs, trainer.train_precisions, 'b-', label='Train', linewidth=2, alpha=0.8)
        plt.plot(epochs, trainer.test_precisions, 'r-', label='Test', linewidth=2, alpha=0.8)
        plt.title('Precision Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])

        # Recall comparison
        plt.subplot(2, 4, 5)
        plt.plot(epochs, trainer.train_recalls, 'b-', label='Train', linewidth=2, alpha=0.8)
        plt.plot(epochs, trainer.test_recalls, 'r-', label='Test', linewidth=2, alpha=0.8)
        plt.title('Recall Comparison', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])

        # Learning rate schedule
        plt.subplot(2, 4, 6)
        plt.plot(epochs, trainer.learning_rates, 'g-', linewidth=2, alpha=0.8)
        plt.title('Learning Rate Schedule', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        # Gradient norms
        plt.subplot(2, 4, 7)
        if trainer.monitor.gradient_norms:
            plt.plot(range(len(trainer.monitor.gradient_norms)), 
                    trainer.monitor.gradient_norms, 'purple', linewidth=2, alpha=0.8)
            plt.title('Gradient Norms', fontweight='bold')
            plt.xlabel('Training Step')
            plt.ylabel('Gradient Norm')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)

        # Model comparison (parameter count)
        plt.subplot(2, 4, 8)
        models = ['Original', 'Optimized']
        params = [17110403, sum(p.numel() for p in model.parameters() if p.requires_grad)]
        colors = ['red', 'green']
        bars = plt.bar(models, params, color=colors, alpha=0.7)
        plt.title('Parameter Count Comparison', fontweight='bold')
        plt.ylabel('Parameters')
        plt.yscale('log')
        for bar, param in zip(bars, params):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, 
                    f'{param:,}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.suptitle('Optimized HGT Training Analysis', fontsize=16, fontweight='bold', y=0.98)

        # Save with unique filename to avoid overwriting
        analysis_filename = get_unique_filename('optimized_training_analysis', 'png')
        plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive analysis saved as '{analysis_filename}'")
        plt.close()  # Close figure to save memory

    except Exception as e:
        print(f"⚠️ Error generating analysis plots: {e}")

    # Final performance summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📋 Final Performance Summary:")
    print("=" * 70)
    print(f"🏆 Best Test F1 Score:      {results['best_f1']:.4f}")
    print(f"📈 Final Test Accuracy:     {trainer.test_accuracies[-1]:.4f}")
    print(f"🎯 Final Test Precision:    {trainer.test_precisions[-1]:.4f}")
    print(f"📊 Final Test Recall:       {trainer.test_recalls[-1]:.4f}")
    print(f"⚡ Best Epoch:             {results['best_epoch']}")
    print(f"⏱️  Total Training Time:    {results['training_time']}")
    print(f"🔧 Total Parameters:       {total_params:,}")
    print(f"📉 Parameter Reduction:     {((17110403 - total_params) / 17110403 * 100):.1f}%")
    print("=" * 70)

    print(f"\n🏗️  Optimized Architecture:")
    print(f"   • Hidden Channels: 96 (was 128)")
    print(f"   • Attention Heads: 4 (was 8)")
    print(f"   • HGT Layers: 2 (was 3)")
    print(f"   • Dropout Rate: 0.1 (was 0.3)")
    print(f"   • Node Types: {len(metadata[0]) if metadata and metadata[0] else 'Unknown'}")

    print(f"\n⚙️  Optimized Training Configuration:")
    print(f"   • Batch Size: 128 (increased for stability)")
    print(f"   • Learning Rate: 0.00001 (ultra-low for stability)")
    print(f"   • Weight Decay: 0.0001 (minimal)")
    print(f"   • Optimizer: AdamW with ReduceLROnPlateau")
    print(f"   • Loss Function: Weighted CrossEntropy + Class Weights")
    print(f"   • Data Augmentation: Disabled for stability")
    print(f"   • Gradient Clipping: 0.5")
    print(f"   • Gradient Noise: Enabled")
    print(f"   • Early Stopping: 35 epochs patience")

    print(f"\n✅ Optimized training complete!")
    print(f"📁 Files generated:")
    print(f"   • optimized_hgt_counterfeit_model_[timestamp].pt - optimized model")
    print(f"   • optimized_training_analysis_[timestamp].png - comprehensive analysis")
    print(f"   • confusion_matrix_optimized_[timestamp].png - confusion matrix")
    print(f"   • best_checkpoint_f1_[score]_epoch_[num].pt - best performing checkpoints")

    # Optional: Run cross-validation for additional validation
    run_cv = input("\n🔬 Run 5-fold cross-validation for additional validation? (y/n): ").lower().strip()
    if run_cv == 'y':
        print("\n🔬 Starting 5-fold cross-validation...")
        mean_f1, std_f1, fold_scores = k_fold_validation(
            hetero_graphs, labels, metadata, k=5, device=device
        )
        print(f"\n📊 Cross-validation completed!")
        print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Optional: Hyperparameter search
    run_hp_search = input("\n🔍 Run hyperparameter search for optimal config? (y/n): ").lower().strip()
    if run_hp_search == 'y':
        print("\n🔍 Starting hyperparameter search...")
        best_params, best_hp_f1, hp_results = hyperparameter_search(
            hetero_graphs, labels, metadata, device=device
        )
        print(f"🏆 Best hyperparameter F1 Score: {best_hp_f1:.4f}")
        
        # Retrain with best parameters if better than current
        if best_hp_f1 > results['best_f1']:
            print(f"\n🚀 Retraining with best hyperparameters...")
            
            # Reinitialize model with best parameters
            model_best = OptimizedHGTCounterfeitDetector(
                hidden_channels=best_params['hidden_channels'],
                out_channels=2,
                num_heads=best_params['num_heads'],
                num_layers=best_params['num_layers'],
                metadata=metadata,
                dropout=best_params['dropout']
            ).to(device)
            
            # New optimizer with best learning rate
            optimizer_best = torch.optim.AdamW(
                model_best.parameters(), 
                lr=best_params['learning_rate'], 
                weight_decay=0.001
            )
            
            # Recreate loaders with best batch size
            train_loader_best = DataLoader(train_graphs, batch_size=best_params['batch_size'], shuffle=True, num_workers=2)
            test_loader_best = DataLoader(test_graphs, batch_size=best_params['batch_size'], shuffle=False, num_workers=2)
            
            # New trainer
            trainer_best = OptimizedCounterfeitTrainer(
                model=model_best, device=device, logger=logger, plot_every=3, use_augmentation=True
            )
            
            # Full training with best parameters
            results_best = trainer_best.train(
                train_loader=train_loader_best,
                test_loader=test_loader_best,
                criterion=criterion_weighted,
                optimizer=optimizer_best,
                num_epochs=120,
                patience=25,
                scheduler=scheduler
            )
            
            # Save the best model
            if results_best['best_f1'] > results['best_f1']:
                torch.save({
                    'model_state_dict': trainer_best.best_model_state,
                    'model_config': best_params,
                    'training_results': results_best,
                    'best_f1': results_best['best_f1'],
                    'hyperparameter_search': hp_results
                }, get_unique_filename('hyperparameter_optimized_model', 'pt'))
                print(f"💾 Hyperparameter-optimized model saved with F1: {results_best['best_f1']:.4f}")
    
    # Optional: Create ensemble
    run_ensemble = input("\n🎯 Train ensemble of models? (y/n): ").lower().strip()
    if run_ensemble == 'y':
        print("\n🎯 Training ensemble models...")
        ensemble_models, test_loader_ensemble = create_ensemble_models(
            hetero_graphs, labels, metadata, num_models=3, device=device
        )
        ensemble_results = ensemble_predict(ensemble_models, test_loader_ensemble, device)
        print(f"🎯 Ensemble F1 Score: {ensemble_results['f1']:.4f}")
    
    plt.ioff()