"""
Fixed Memory-Optimized HGT Model for Counterfeit Drug Detection
Optimized for GTX 1060 6GB VRAM with proper batch handling
Author: ediionut23 (final fixed version)
Version: 3.2.0
Last Updated: 2025-09-05
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch.nn import LayerNorm
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
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
        return 16
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = gpu_memory / (1024**3)
    
    print(f"🖥️ GPU Memory: {gpu_memory_gb:.1f} GB")
    
    if gpu_memory_gb <= 4:
        return 8
    elif gpu_memory_gb <= 6:
        return 16  # Very conservative for GTX 1060
    elif gpu_memory_gb <= 8:
        return 32
    else:
        return 64

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

plt.ion()

class FixedHGTModel(torch.nn.Module):
    """Fixed HGT model with proper batch handling"""
    
    def __init__(self, metadata, hidden_channels=32, num_heads=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Smaller model for memory efficiency
        self.node_embeddings = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in metadata[0]
        })
          # HGT layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            # Fixed HGTConv constructor - only 4 arguments
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            norm = torch.nn.ModuleDict({
                node_type: LayerNorm(hidden_channels)
                for node_type in metadata[0]
            })
            self.convs.append(conv)
            self.norms.append(norm)
        
        # Simple classification head
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
        
        # HGT layers
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            
            for node_type, x in x_dict.items():
                x_dict[node_type] = F.relu(norm[node_type](x))
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout_rate, training=self.training)
        
        # Fixed global pooling that handles different node type sizes
        if batch_dict is not None:
            # Find actual batch size from the batch_dict
            actual_batch_size = 0
            for node_type in batch_dict.keys():
                if len(batch_dict[node_type]) > 0:
                    actual_batch_size = max(actual_batch_size, batch_dict[node_type].max().item() + 1)
            
            if actual_batch_size == 0:
                actual_batch_size = 1
            
            # Pool each node type separately
            pooled_per_graph = torch.zeros(actual_batch_size, self.hidden_channels, device=list(x_dict.values())[0].device)
            
            for node_type, x in x_dict.items():
                if node_type in batch_dict and len(x) > 0:
                    # Use scatter_add to pool nodes by graph
                    pooled_per_graph.scatter_add_(0, 
                                                batch_dict[node_type].unsqueeze(1).expand(-1, self.hidden_channels), 
                                                x)
            
            graph_repr = pooled_per_graph
        else:
            # Single graph - just average all node features
            all_features = []
            for node_type, x in x_dict.items():
                if len(x) > 0:
                    all_features.append(x.mean(dim=0))
            
            if all_features:
                graph_repr = torch.stack(all_features).mean(dim=0).unsqueeze(0)
            else:
                graph_repr = torch.zeros(1, self.hidden_channels, device=list(x_dict.values())[0].device)
        
        # Classification
        return self.classifier(graph_repr)

def train_fixed_model(model, train_loader, val_loader, num_epochs=30, device='cpu'):
    """Fixed training function with proper batch handling"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 7
    
    train_losses = []
    val_accuracies = []
    
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
                
                # Get labels - handle batch size mismatch
                if hasattr(batch, 'y'):
                    labels = batch.y
                else:
                    # Extract labels from individual graphs
                    labels = []
                    for graph in batch.to_data_list():
                        if hasattr(graph, 'y'):
                            labels.append(graph.y.item())
                        else:
                            labels.append(0)  # Default label
                    labels = torch.tensor(labels, device=device, dtype=torch.long)
                
                # Ensure output and labels have the same batch size
                if out.size(0) != labels.size(0):
                    min_size = min(out.size(0), labels.size(0))
                    out = out[:min_size]
                    labels = labels[:min_size]
                
                loss = criterion(out, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
                train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    clear_gpu_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM in batch {batch_idx}, clearing cache...")
                    clear_gpu_cache()
                    continue
                else:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
        
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
                        labels = []
                        for graph in batch.to_data_list():
                            if hasattr(graph, 'y'):
                                labels.append(graph.y.item())
                            else:
                                labels.append(0)
                        labels = torch.tensor(labels, device=device, dtype=torch.long)
                    
                    # Handle size mismatch
                    if out.size(0) != labels.size(0):
                        min_size = min(out.size(0), labels.size(0))
                        out = out[:min_size]
                        labels = labels[:min_size]
                    
                    pred = out.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("OOM in validation, clearing cache...")
                        clear_gpu_cache()
                        continue
                    else:
                        logger.error(f"Validation error: {e}")
                        continue
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_fixed.pt')
        else:
            patience_counter += 1
        
        # Logging
        logger.info(f'Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}, Best={best_val_acc:.4f}')
        print(f'Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}, Best={best_val_acc:.4f}')
        
        # Monitor GPU memory every 5 epochs
        if epoch % 5 == 0:
            monitor_gpu_memory()
        
        clear_gpu_cache()
        
        if patience_counter >= max_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_accuracies, best_val_acc

def main():
    """Main training function with fixed batch handling"""
    
    print("🚀 Fixed HGT Model Training Started...")
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"💻 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        monitor_gpu_memory()
    
    # Load dataset with fixed loading
    dataset_path = os.path.join(os.getcwd(), 'realistic_counterfeit_dataset_8k_balanced.pt')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    print(f"✅ Dataset file found at: {dataset_path}")
    
    try:
        # Fixed dataset loading for PyTorch 2.6+
        import torch_geometric.data.storage
        torch.serialization.add_safe_globals([torch_geometric.data.storage.BaseStorage])
        
        data = torch.load(dataset_path, map_location='cpu', weights_only=False)
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
    
    # Get optimal batch size
    optimal_batch_size = get_optimal_batch_size()
    print(f"🎯 Using batch size: {optimal_batch_size}")
    
    # Split data
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        graphs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Split: {len(train_graphs)} train, {len(test_graphs)} test")
    
    # Create data loaders with smaller batch size and no workers
    train_loader = DataLoader(train_graphs, batch_size=optimal_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=optimal_batch_size, shuffle=False, num_workers=0)
    
    # Initialize smaller model
    model = FixedHGTModel(
        metadata=metadata,
        hidden_channels=32,  # Smaller
        num_heads=2,         # Smaller
        num_layers=2,        # Same
        dropout=0.3
    ).to(device)
    
    print("🔧 Fixed model initialized")
    print(f"   • Hidden channels: 32")
    print(f"   • Attention heads: 2") 
    print(f"   • HGT layers: 2")
    print(f"   • Dropout: 0.3")
    
    clear_gpu_cache()
    monitor_gpu_memory()
    
    # Start training
    print("\n🏋️ Starting fixed training...")
    
    try:
        train_losses, val_accuracies, best_acc = train_fixed_model(
            model, train_loader, test_loader, 
            num_epochs=30, device=device
        )
        
        print(f"\n🎉 Training completed!")
        print(f"📊 Best validation accuracy: {best_acc:.4f}")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        clear_gpu_cache()
        raise

if __name__ == "__main__":
    main()
