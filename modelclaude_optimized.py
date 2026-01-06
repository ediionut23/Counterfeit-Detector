"""
GPU Memory Optimized HGT Model for Counterfeit Drug Detection
Optimized for GTX 1060 6GB VRAM
Author: ediionut23 (memory optimized)
Version: 3.1.0
Last Updated: 2025-09-05
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
import gc

warnings.filterwarnings('ignore')

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_optimal_batch_size():
    """Get optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 32
    
    # Get GPU memory info
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = gpu_memory / (1024**3)
    
    print(f"🖥️ GPU Memory: {gpu_memory_gb:.1f} GB")
    
    # Conservative batch sizes for different GPU memory sizes
    if gpu_memory_gb <= 4:
        return 16
    elif gpu_memory_gb <= 6:
        return 32  # Conservative for GTX 1060
    elif gpu_memory_gb <= 8:
        return 64
    else:
        return 128

def monitor_gpu_memory():
    """Monitor and print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enable interactive plotting with fallback
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

plt.ion()

class OptimizedHGTModel(torch.nn.Module):
    """Memory-optimized HGT model for counterfeit drug detection"""
    
    def __init__(self, metadata, hidden_channels=64, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Use smaller hidden channels to reduce memory usage
        self.node_embeddings = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in metadata[0]
        })
          # Reduced number of HGT layers for memory efficiency
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            norm = torch.nn.ModuleDict({
                node_type: BatchNorm(hidden_channels)
                for node_type in metadata[0]
            })
            self.convs.append(conv)
            self.norms.append(norm)
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels // 2, 2)
        )
    
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        # Initial embeddings
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_embeddings[node_type](x)
        
        # HGT layers with memory-efficient processing
        for conv, norm in zip(self.convs, self.norms):
            # Apply HGT convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply normalization and activation
            for node_type, x in x_dict.items():
                x_dict[node_type] = F.relu(norm[node_type](x))
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout_rate, training=self.training)
            
            # Clear intermediate tensors
            if self.training:
                clear_gpu_cache()        
        # Global pooling - use mean pooling for memory efficiency
        if batch_dict is not None:
            # Batch processing - handle different node type sizes properly
            batch_size = max(batch_dict[node_type].max().item() + 1 for node_type in batch_dict.keys())
            pooled_features = []
            
            for node_type, x in x_dict.items():
                if node_type in batch_dict:
                    # Pool nodes of this type per graph
                    pooled = torch.zeros(batch_size, self.hidden_channels, device=x.device)
                    pooled = pooled.scatter_add_(0, batch_dict[node_type].unsqueeze(1).expand(-1, self.hidden_channels), x)
                    # Average pooling across node types
                    pooled_features.append(pooled.mean(dim=0, keepdim=True))  # Shape: (1, hidden_channels)
            
            if pooled_features:
                # Now all tensors have the same shape (1, hidden_channels)
                graph_repr = torch.cat(pooled_features, dim=0).mean(dim=0, keepdim=True)  # Shape: (1, hidden_channels)
                # Expand to batch size
                graph_repr = graph_repr.expand(batch_size, -1)
            else:
                # Fallback
                graph_repr = torch.zeros(batch_size, self.hidden_channels, device=list(x_dict.values())[0].device)
        else:
            # Single graph processing - concatenate all node features
            all_features = []
            for node_type, x in x_dict.items():
                # Global mean pooling for each node type
                all_features.append(x.mean(dim=0, keepdim=True))  # Shape: (1, hidden_channels)
            
            if all_features:
                graph_repr = torch.cat(all_features, dim=0).mean(dim=0, keepdim=True)  # Shape: (1, hidden_channels)
            else:
                graph_repr = torch.zeros(1, self.hidden_channels, device=list(x_dict.values())[0].device)
        # Classification
        return self.classifier(graph_repr)

def train_memory_optimized(model, train_loader, val_loader, num_epochs=50, device='cpu'):
    """Memory-optimized training function"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 10
    
    train_losses = []
    val_accuracies = []
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(train_bar):
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                
                # Get labels
                if hasattr(batch, 'y'):
                    labels = batch.y
                else:
                    # Fallback label extraction
                    labels = torch.zeros(out.size(0), dtype=torch.long, device=device)
                    for i, graph in enumerate(batch.to_data_list()):
                        if hasattr(graph, 'y'):
                            labels[i] = graph.y
                
                loss = criterion(out, labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
                # Update progress bar
                train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
                # Clear cache every few batches
                if batch_idx % 5 == 0:
                    clear_gpu_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM in batch {batch_idx}, clearing cache and continuing...")
                    clear_gpu_cache()
                    continue
                else:
                    raise e
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                try:
                    batch = batch.to(device)
                    out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                    
                    # Get labels
                    if hasattr(batch, 'y'):
                        labels = batch.y
                    else:
                        labels = torch.zeros(out.size(0), dtype=torch.long, device=device)
                        for i, graph in enumerate(batch.to_data_list()):
                            if hasattr(graph, 'y'):
                                labels[i] = graph.y
                    
                    pred = out.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
                    
                    clear_gpu_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("OOM in validation, clearing cache and continuing...")
                        clear_gpu_cache()
                        continue
                    else:
                        raise e
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_optimized.pt')
        else:
            patience_counter += 1
        
        # Logging
        logger.info(f'Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}, Best={best_val_acc:.4f}')
        
        # Update plots every 5 epochs
        if epoch % 5 == 0:
            ax1.clear()
            ax1.plot(train_losses, label='Training Loss', color='blue')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            ax2.clear()
            ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.pause(0.01)
        
        # Monitor GPU memory
        if epoch % 5 == 0:
            monitor_gpu_memory()
        
        # Clear cache after each epoch
        clear_gpu_cache()
        
        if patience_counter >= max_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    plt.ioff()
    plt.show()
    
    return train_losses, val_accuracies, best_val_acc

def main():
    """Main training function with memory optimization"""
    
    print("🚀 Memory-Optimized HGT Model Training Started...")
    
    # Device selection with memory info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"💻 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        monitor_gpu_memory()
      # Load dataset
    dataset_path = os.path.join(os.getcwd(), 'realistic_counterfeit_dataset_8k_balanced.pt')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please run data_realistic.py first to generate the dataset")
        return
        return
    
    print(f"✅ Dataset file found at: {dataset_path}")
    
    try:
        # Fix for PyTorch 2.6+ security restrictions
        import torch_geometric.data.storage
        torch.serialization.add_safe_globals([torch_geometric.data.storage.BaseStorage])
        
        data = torch.load(dataset_path, map_location='cpu', weights_only=False)  # Load to CPU first
        print(f"📊 Loaded data type: {type(data)}")
        
        if isinstance(data, tuple) and len(data) >= 3:
            graphs, labels, metadata = data[0], data[1], data[2]
        else:
            print("❌ Invalid dataset format")
            return
        
        print(f"📦 Loaded {len(graphs)} graphs")
        print(f"📊 Labels shape: {len(labels) if isinstance(labels, list) else labels.shape}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Get optimal batch size for your GPU
    optimal_batch_size = get_optimal_batch_size()
    print(f"🎯 Using batch size: {optimal_batch_size}")
    
    # Split data
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        graphs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Split: {len(train_graphs)} train, {len(test_graphs)} test")
    
    # Create data loaders with optimal batch size
    train_loader = DataLoader(train_graphs, batch_size=optimal_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=optimal_batch_size, shuffle=False, num_workers=0)
    
    # Initialize model with reduced parameters for memory efficiency
    model = OptimizedHGTModel(
        metadata=metadata,
        hidden_channels=64,  # Reduced from 128
        num_heads=4,         # Reduced from 8
        num_layers=2,        # Reduced from 3
        dropout=0.3
    ).to(device)
    
    print("🔧 Memory-optimized model initialized")
    print(f"   • Hidden channels: 64")
    print(f"   • Attention heads: 4") 
    print(f"   • HGT layers: 2")
    print(f"   • Dropout: 0.3")
    
    # Clear cache before training
    clear_gpu_cache()
    monitor_gpu_memory()
    
    # Start training
    print("\n🏋️ Starting memory-optimized training...")
    
    try:
        train_losses, val_accuracies, best_acc = train_memory_optimized(
            model, train_loader, test_loader, 
            num_epochs=50, device=device
        )
        
        print(f"\n🎉 Training completed!")
        print(f"📊 Best validation accuracy: {best_acc:.4f}")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        clear_gpu_cache()
        raise

if __name__ == "__main__":
    main()
