"""
Script to load and use the enhanced_pharma_v2 dataset
"""
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json

# Load the dataset
dataset_path = "enhanced_pharma_v2_20260401_191857.pt"
stats_path = "enhanced_pharma_v2_20260401_191857_stats.json"

print("Loading dataset...")
data = torch.load(dataset_path, weights_only=False)

# Load stats
with open(stats_path, 'r') as f:
    stats = json.load(f)

print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"Total molecules: {stats['total']}")
print(f"Authentic: {stats['authentic']}")
print(f"Counterfeit: {stats['counterfeit']}")
print(f"Unique SMILES: {stats['unique_smiles']}")
print(f"Node types: {stats['node_types']}")
print(f"Edge types: {stats['edge_types_count']}")
print(f"Atom features: {stats['atom_features']}")
print(f"Bond features: {stats['bond_features']}")
print(f"Generator version: {stats['generator_version']}")
print("\nImprovements:")
for imp in stats['improvements']:
    print(f"  - {imp}")

print("\n" + "="*60)
print("DATASET STRUCTURE")
print("="*60)

# The dataset is saved as a tuple: (graphs_list, labels_list)
if isinstance(data, tuple):
    print(f"Dataset is a tuple with {len(data)} elements")
    
    if len(data) == 2:
        graphs, labels = data
        print(f"Number of graphs: {len(graphs)}")
        print(f"Number of labels: {len(labels)}")
        print(f"Labels: {labels}")
    else:
        print(f"Unexpected tuple length: {len(data)}")
        graphs = data[0] if len(data) > 0 else []
        labels = data[1] if len(data) > 1 else []
    
    if len(graphs) > 0:
        sample = graphs[0]
        print(f"\nSample graph structure:")
        print(f"  - Type: {type(sample)}")
        if hasattr(sample, 'keys'):
            print(f"  - Keys: {sample.keys}")
        if hasattr(sample, 'x'):
            print(f"  - Node features shape: {sample.x.shape}")
        if hasattr(sample, 'edge_index'):
            print(f"  - Edge index shape: {sample.edge_index.shape}")
        if hasattr(sample, 'edge_attr'):
            print(f"  - Edge attributes shape: {sample.edge_attr.shape}")
        if hasattr(sample, 'y'):
            print(f"  - Label: {sample.y}")
        if hasattr(sample, 'smiles'):
            print(f"  - SMILES: {sample.smiles}")
        if hasattr(sample, 'original_smiles'):
            print(f"  - Original SMILES: {sample.original_smiles}")
elif isinstance(data, list):
    print(f"Dataset contains {len(data)} graphs")
    graphs = data
    labels = None
    if len(data) > 0:
        sample = data[0]
        print(f"\nSample graph structure:")
        print(f"  - Keys: {sample.keys if hasattr(sample, 'keys') else 'N/A'}")
        if hasattr(sample, 'x'):
            print(f"  - Node features shape: {sample.x.shape}")
        if hasattr(sample, 'edge_index'):
            print(f"  - Edge index shape: {sample.edge_index.shape}")
        if hasattr(sample, 'edge_attr'):
            print(f"  - Edge attributes shape: {sample.edge_attr.shape}")
        if hasattr(sample, 'y'):
            print(f"  - Label: {sample.y}")
        if hasattr(sample, 'smiles'):
            print(f"  - SMILES: {sample.smiles}")
        if hasattr(sample, 'original_smiles'):
            print(f"  - Original SMILES: {sample.original_smiles}")
else:
    print("Dataset is a single object")
    print(f"Type: {type(data)}")
    graphs = data
    labels = None

print("\n" + "="*60)
print("READY TO USE")
print("="*60)
print("\nYou can now:")
print("1. Use this dataset with homo_gnn_clean.py")
print("2. Create a DataLoader: loader = DataLoader(data, batch_size=32, shuffle=True)")
print("3. Train your model with the loaded data")
print("\nExample:")
print("  from torch_geometric.loader import DataLoader")
print("  loader = DataLoader(data, batch_size=32, shuffle=True)")
print("  for batch in loader:")
print("      # Your training code here")
print("      pass")
