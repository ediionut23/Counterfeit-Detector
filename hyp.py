
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import HGTConv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import warnings
from typing import List, Dict, Tuple, Optional
import pandas as pd
import json
import copy
import random
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from PIL import Image
import io

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from attention_extractor import HGTAttentionExtractor

RESULTS_DIR = './PyG_Explainability_Results'
Path(RESULTS_DIR).mkdir(exist_ok=True)

CLASSIFICATION_THRESHOLD = 0.692


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
        classifier_input_dim = hidden_channels * 2 * len(node_types)  # mean+max pooling
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
                break
        batch_size = self._get_batch_size(batch_dict, h_dict, batch_size)
        pooled_features = self._global_pooling(h_dict, batch_dict, batch_size, node_types)
        if pooled_features.size(0) > 0:
            out = self.classifier(pooled_features)
        else:
            out = torch.zeros((batch_size, self.out_channels), device=device)
        return out

    def _get_batch_size(self, batch_dict, h_dict, provided_batch_size):
        if provided_batch_size is not None: return provided_batch_size
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


class HGTModelWrapper(torch.nn.Module):
    def __init__(self, hgt_model):
        super().__init__()
        self.hgt_model = hgt_model
    
    def forward(self, x_dict, edge_index_dict, batch_dict=None, batch_size=1, **kwargs):
        node_mask = kwargs.get('node_mask', None)
        if node_mask is not None:
            masked_x_dict = {}
            if isinstance(node_mask, dict):
                for node_type in x_dict.keys():
                    if node_type in node_mask and node_mask[node_type] is not None:
                        mask = node_mask[node_type].view(-1, 1)
                        masked_x_dict[node_type] = x_dict[node_type] * mask
                    else:
                        masked_x_dict[node_type] = x_dict[node_type]
            else:
                masked_x_dict = self._apply_homogeneous_mask(x_dict, node_mask)
            x_dict = masked_x_dict
        out = self.hgt_model(x_dict, edge_index_dict, batch_dict, batch_size)
        return out
    
    def _apply_homogeneous_mask(self, x_dict, mask):
        masked_x_dict = {}
        offset = 0
        for node_type in x_dict.keys():
            num_nodes = x_dict[node_type].size(0)
            if offset + num_nodes <= mask.size(0):
                node_mask = mask[offset:offset + num_nodes].view(-1, 1)
                masked_x_dict[node_type] = x_dict[node_type] * node_mask
            else:
                masked_x_dict[node_type] = x_dict[node_type]
            offset += num_nodes
        return masked_x_dict


class PyGNativeExplainer:
    def __init__(self, model, device, dataset_sample=None):
        self.original_model = model
        self.device = device
        self.model = HGTModelWrapper(model).to(device)
        self.model.eval()
        
        self.feature_means = {}
        if dataset_sample:
            try:
                node_types = dataset_sample[0].node_types
                for node_type in node_types:
                    all_feats = []
                    for data in dataset_sample[:min(len(dataset_sample), 200)]:
                        if node_type in data.x_dict:
                            all_feats.append(data.x_dict[node_type])
                    if all_feats:
                        self.feature_means[node_type] = torch.cat(all_feats, dim=0).mean(dim=0).to(device)
            except: pass

        self.atom_feature_names = [
            'Atomic_Number', 'Degree', 'Total_Degree', 'Formal_Charge',
            'Hybridization', 'Is_Aromatic', 'Total_Hs', 'Mass',
            'Is_Chiral', 'Chirality_Possible', 'Is_H_Donor', 'Is_H_Acceptor',
            'Is_Halogen', 'Is_Heteroatom', 'In_Ring', 'In_Ring_Size_5',
            'In_Ring_Size_6', 'In_Other_Ring', 'Radical_Electrons',
            'Aromatic_Neighbors_Ratio', 'Heteroatom_Neighbors_Ratio',
            'Halogen_Neighbors_Ratio', 'Charged_Neighbors', 'Is_SP',
            'Is_SP2', 'Is_SP3'
        ]
        
        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=150),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type=None,
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
        self.explainer_object = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=150),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type=None,   
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
        logger.info("PyTorch Geometric GNNExplainer initialized (attributes + object)")
    
    def explain_molecule(self, graph: HeteroData, mol_smiles: str,
                         target_class: Optional[int] = None,
                         top_k_substructures: int = 5) -> Dict:
        graph = graph.to(self.device)

        with torch.no_grad():
            batch_dict = self._safe_get_batch_dict(graph)
            logits = self.model(graph.x_dict, graph.edge_index_dict, batch_dict, 1)
            probs = F.softmax(logits, dim=1)
            pred_class = 1 if probs[0, 1].item() >= CLASSIFICATION_THRESHOLD else 0
            confidence = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class
        logger.info(f"Explaining: class={pred_class}, confidence={confidence:.3f}")

        try:
            gnn_explanation = self._generate_pyg_explanation(graph, target_class)
        except Exception as e:
            logger.warning(f"GNNExplainer (attributes) failed: {e}")
            gnn_explanation = None

        try:
            gnn_object_explanation = self._generate_pyg_explanation_object(graph, target_class)
        except Exception as e:
            logger.warning(f"GNNExplainer (object) failed: {e}")
            gnn_object_explanation = None

        try:
            grad_explanation = self._compute_gradient_importance(graph, target_class)
        except Exception as e:
            logger.warning(f"Gradient importance failed: {e}")
            grad_explanation = None

        try:
            edge_grad_explanation = self._compute_edge_gradient_importance(graph, target_class)
        except Exception as e:
            logger.warning(f"Edge gradient importance failed: {e}")
            edge_grad_explanation = None

        hgt_attention_result = None
        try:
            with HGTAttentionExtractor(self.model) as extractor:
                hgt_attention_result = extractor.extract(graph)
            logger.info(
                f"Captured native HGT attention: "
                f"{hgt_attention_result['num_layers']} layers, "
                f"{len(hgt_attention_result['edge_type_attention'])} edge types"
            )
        except Exception as e:
            logger.warning(f"HGT attention extraction failed: {e}")

        combined_importance = self._combine_explanations(
            gnn_explanation, grad_explanation, hgt_attention_result, graph.node_types
        )
        final_explanation = {'node_importance': combined_importance}

        substructures = self._identify_critical_substructures(
            graph, mol_smiles, final_explanation, top_k=top_k_substructures
        )

        counterfactuals = self._generate_counterfactual_insights(
            graph, mol_smiles, pred_class, substructures
        )

        feature_importance = self._analyze_feature_importance(graph, final_explanation)
        node_contributions = self._analyze_node_type_contributions(graph, final_explanation)

        mol = Chem.MolFromSmiles(mol_smiles)
        num_atoms = mol.GetNumAtoms() if mol else 1
        important_atoms_count = 0
        for substruct in substructures:
            if substruct['importance_score'] > 0.3:
                important_atoms_count += len(substruct['atom_indices'])
        sparsity_score = 1.0 - (min(important_atoms_count, num_atoms) / num_atoms) if num_atoms > 0 else 0
        fidelity_drop = confidence - counterfactuals.get('perturbed_prob', confidence)

        agreement = self._compute_agreement(
            gnn_explanation, grad_explanation, hgt_attention_result, graph.node_types
        )

        return {
            'prediction': {
                'class': 'Counterfeit' if pred_class == 1 else 'Authentic',
                'class_id': pred_class,
                'confidence': confidence,
                'probabilities': probs[0].cpu().numpy()
            },
            'explanation': final_explanation,
            'critical_substructures': substructures,
            'feature_importance': feature_importance,
            'node_contributions': node_contributions,
            'counterfactuals': counterfactuals,
            'smiles': mol_smiles,
            'metrics': {
                'fidelity_drop': fidelity_drop,
                'sparsity': sparsity_score,
                'perturbed_prob': counterfactuals.get('perturbed_prob', 0.0),
                'method_agreement': agreement,
            },
            'hgt_attention': hgt_attention_result,
            'gnn_explainer_raw': gnn_explanation,
            'gnn_object_raw': gnn_object_explanation,   
            'gradient_raw': grad_explanation,
            'edge_gradient_raw': edge_grad_explanation,
        }
    
    def _safe_get_batch_dict(self, graph: HeteroData):
        try:
            if hasattr(graph, 'batch_dict') and graph.batch_dict is not None:
                return graph.batch_dict
        except: pass
        return None
    
    def _generate_pyg_explanation(self, graph: HeteroData, target_class: int) -> Dict:
        batch_dict = self._safe_get_batch_dict(graph)
        explanation_result = self.explainer(
            x=graph.x_dict,
            edge_index=graph.edge_index_dict,
            batch_dict=batch_dict,
            batch_size=1,
            target=target_class
        )
        node_importance = {}
        if hasattr(explanation_result, 'node_types'):
            for node_type in explanation_result.node_types:
                node_store = explanation_result[node_type]
                mask = None
                if hasattr(node_store, 'node_mask') and node_store.node_mask is not None:
                    mask = node_store.node_mask
                elif hasattr(node_store, 'mask') and node_store.mask is not None:
                    mask = node_store.mask
                if mask is not None:
                    if mask.ndim == 2: mask = mask.mean(dim=1)
                    node_importance[node_type] = mask.detach().cpu().numpy()
        edge_importance = {}
        if hasattr(explanation_result, 'edge_types'):
            for edge_type in explanation_result.edge_types:
                edge_store = explanation_result[edge_type]
                emask = None
                if hasattr(edge_store, 'edge_mask') and edge_store.edge_mask is not None:
                    emask = edge_store.edge_mask
                if emask is not None:
                    edge_importance[edge_type] = emask.detach().cpu().numpy()
        return {'node_importance': node_importance, 'edge_importance': edge_importance}

    def _generate_pyg_explanation_object(self, graph: HeteroData, target_class: int) -> Dict:
        batch_dict = self._safe_get_batch_dict(graph)
        explanation_result = self.explainer_object(
            x=graph.x_dict,
            edge_index=graph.edge_index_dict,
            batch_dict=batch_dict,
            batch_size=1,
            target=target_class
        )
        node_importance_object = {}
        if hasattr(explanation_result, 'node_types'):
            for node_type in explanation_result.node_types:
                node_store = explanation_result[node_type]
                mask = None
                if hasattr(node_store, 'node_mask') and node_store.node_mask is not None:
                    mask = node_store.node_mask
                elif hasattr(node_store, 'mask') and node_store.mask is not None:
                    mask = node_store.mask
                if mask is not None:
                    if mask.ndim == 2: mask = mask.squeeze(-1)  # object mask is [N,1]
                    node_importance_object[node_type] = mask.detach().cpu().numpy()
        edge_importance_object = {}
        if hasattr(explanation_result, 'edge_types'):
            for edge_type in explanation_result.edge_types:
                edge_store = explanation_result[edge_type]
                emask = getattr(edge_store, 'edge_mask', None)
                if emask is not None:
                    edge_importance_object[edge_type] = emask.detach().cpu().numpy()
        return {
            'node_importance_object': node_importance_object,
            'edge_importance_object': edge_importance_object,
        }

    def _compute_edge_gradient_importance(self, graph: HeteroData, target_class: int) -> Dict:
        original_ea = {}
        for et in graph.edge_types:
            if hasattr(graph[et], 'edge_attr') and graph[et].edge_attr is not None:
                original_ea[et] = graph[et].edge_attr.clone()
                graph[et].edge_attr = graph[et].edge_attr.detach().requires_grad_(True)

        if not original_ea:
            return {'edge_importance': {}}

        batch_dict = self._safe_get_batch_dict(graph)
        logits = self.model(graph.x_dict, graph.edge_index_dict, batch_dict, 1)
        logits[0, target_class].backward()

        edge_importance = {}
        for et, orig in original_ea.items():
            ea = graph[et].edge_attr
            if ea.grad is not None:
                imp = (ea.grad * orig).abs().sum(dim=1)
                if imp.max() > 0:
                    imp = imp / imp.max()
                edge_importance[et] = imp.detach().cpu().numpy()
            else:
                edge_importance[et] = np.zeros(orig.size(0))
            graph[et].edge_attr = orig.detach()

        return {'edge_importance': edge_importance}

    def _compute_gradient_importance(self, graph: HeteroData, target_class: int) -> Dict:
        original_x_dict = {}
        for node_type in graph.x_dict.keys():
            original_x_dict[node_type] = graph[node_type].x.clone()
            graph.x_dict[node_type].requires_grad = True
        batch_dict = self._safe_get_batch_dict(graph)
        logits = self.model(graph.x_dict, graph.edge_index_dict, batch_dict, 1)
        target_logit = logits[0, target_class]
        target_logit.backward()
        node_importance = {}
        for node_type in graph.x_dict.keys():
            if graph.x_dict[node_type].grad is not None:
                grads = graph.x_dict[node_type].grad
                features = original_x_dict[node_type]
                importance = (grads * features).abs().sum(dim=1)
                if importance.max() > 0: importance = importance / importance.max()
                node_importance[node_type] = importance.detach().cpu().numpy()
            else:
                node_importance[node_type] = np.zeros(graph.x_dict[node_type].size(0))
        for node_type in graph.x_dict.keys():
            graph.x_dict[node_type] = original_x_dict[node_type].detach()
        return {'node_importance': node_importance}

    def _combine_explanations(self, gnn_exp, grad_exp, hgt_attn, node_types):
        combined = {}
        for node_type in node_types:
            sources = []

            def _take(d, key):
                if d is None or key not in d or node_type not in d[key]:
                    return None
                v = np.asarray(d[key][node_type], dtype=np.float64)
                if v.size == 0:
                    return None
                m = v.max()
                return v / m if m > 0 else v

            s1 = _take(gnn_exp, 'node_importance')
            s2 = _take(grad_exp, 'node_importance')
            s3 = _take(hgt_attn, 'node_attention') if hgt_attn else None

            for s in (s1, s2, s3):
                if s is not None:
                    sources.append(s)

            if not sources:
                combined[node_type] = np.array([])
                continue

            min_len = min(s.shape[0] for s in sources)
            sources = [s[:min_len] for s in sources]

            
            w_map = {1: [1.0], 2: [0.5, 0.5], 3: [0.4, 0.4, 0.2]}
            w = w_map[len(sources)]
            stacked = np.stack(sources, axis=0)
            combined[node_type] = np.average(stacked, weights=w, axis=0).astype(np.float32)

        return combined

    def _compute_agreement(self, gnn_exp, grad_exp, hgt_attn, node_types) -> float:
        def _extract(d, key):
            if d is None or key not in d:
                return None
            parts = []
            for nt in node_types:
                if nt in d[key]:
                    parts.append(np.asarray(d[key][nt], dtype=np.float64).ravel())
            if not parts:
                return None
            return np.concatenate(parts)

        srcs = []
        for d, k in [(gnn_exp, 'node_importance'),
                     (grad_exp, 'node_importance'),
                     (hgt_attn, 'node_attention') if hgt_attn else (None, None)]:
            v = _extract(d, k) if k else None
            if v is not None and v.size > 1 and v.std() > 1e-9:
                srcs.append(v)

        if len(srcs) < 2:
            return 0.0

        L = min(s.shape[0] for s in srcs)
        srcs = [s[:L] for s in srcs]

        corrs = []
        for i in range(len(srcs)):
            for j in range(i + 1, len(srcs)):
                c = float(np.corrcoef(srcs[i], srcs[j])[0, 1])
                if not np.isnan(c):
                    corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    def _identify_critical_substructures(self, graph: HeteroData, smiles: str, 
                                      explanation: Dict, top_k: int = 5) -> List[Dict]:
        if explanation is None: return []
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return []
        node_importance = explanation.get('node_importance', {})
        if not node_importance: return []
        all_node_scores = []
        node_idx = 0
        for node_type, importance in node_importance.items():
            for local_idx, score in enumerate(importance):
                if node_idx < mol.GetNumAtoms():
                    atom = mol.GetAtomWithIdx(node_idx)
                    if atom.GetSymbol() == node_type:
                        all_node_scores.append((node_idx, float(score)))
                node_idx += 1
        all_node_scores.sort(key=lambda x: x[1], reverse=True)
        substructures = []
        visited = set()
        for atom_idx, score in all_node_scores[:top_k * 2]:
            if atom_idx in visited or atom_idx >= mol.GetNumAtoms(): continue
            substructure_atoms = {atom_idx}
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                substructure_atoms.add(neighbor.GetIdx())
            if substructure_atoms not in [s['atom_indices'] for s in substructures]:
                aggregate_importance = sum(
                    s for idx, s in all_node_scores if idx in substructure_atoms
                ) / len(substructure_atoms) if len(substructure_atoms) > 0 else 0
                substructures.append({
                    'atom_indices': substructure_atoms,
                    'center_atom': atom_idx,
                    'center_atom_symbol': atom.GetSymbol(),
                    'importance_score': float(score),
                    'aggregate_importance': float(aggregate_importance),
                    'substructure_size': len(substructure_atoms),
                    'functional_group': self._identify_functional_group(mol, atom_idx)
                })
                visited.update(substructure_atoms)
                if len(substructures) >= top_k: break
        return substructures

    def _generate_counterfactual_insights(self, graph: HeteroData, smiles: str, 
                                        pred_class: int, substructures: List[Dict]) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not substructures:
            return {'text': 'Analysis not possible', 'metrics': {}}
        top_substruct = substructures[0]
        critical_atoms = list(top_substruct['atom_indices'])
        
        perturbed_graph = graph.clone()
        for node_type in perturbed_graph.node_types:
            if hasattr(perturbed_graph[node_type], 'x') and perturbed_graph[node_type].x is not None:
                perturbed_graph[node_type].x = perturbed_graph[node_type].x.detach().clone()
        
        atom_idx = 0
        mapped_indices = []
        for node_type in graph.node_types:
            num_nodes = graph[node_type].x.size(0)
            for local_idx in range(num_nodes):
                if atom_idx in critical_atoms:
                    mapped_indices.append((node_type, local_idx))
                atom_idx += 1
        
        for node_type, local_idx in mapped_indices:
            if node_type in self.feature_means:
                perturbed_graph[node_type].x[local_idx] = self.feature_means[node_type].clone()
            else:
                perturbed_graph[node_type].x[local_idx] = torch.zeros_like(perturbed_graph[node_type].x[local_idx])

        with torch.no_grad():
            batch_dict = self._safe_get_batch_dict(perturbed_graph)
            logits = self.model(perturbed_graph.x_dict, perturbed_graph.edge_index_dict, batch_dict, 1)
            probs = F.softmax(logits, dim=1)
            new_prob = probs[0, pred_class].item()
            new_class = logits.argmax(dim=1).item()
            
        prob_drop = (1.0 - new_prob) * 100 
        fg_name = top_substruct['functional_group'].replace('_', ' ')
        
        if new_class != pred_class:
            insight = f"STRONG EVIDENCE: Replacing the {fg_name} flips the prediction."
            impact = "High"
        elif prob_drop > 20:
            insight = f"MODERATE EVIDENCE: Replacing the {fg_name} reduces confidence by {prob_drop:.1f}%."
            impact = "Medium"
        else:
            insight = f"WEAK EVIDENCE: The {fg_name} contributes, but is not solely responsible."
            impact = "Low"
            
        return {
            'text': insight,
            'impact_level': impact,
            'perturbed_prob': new_prob,
            'critical_group': fg_name,
            'original_properties': {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logp': Descriptors.MolLogP(mol)
            }
        }

    def _identify_functional_group(self, mol, atom_idx: int) -> str:
        atom = mol.GetAtomWithIdx(atom_idx)
        patterns = {
            'Carboxylic_Acid': 'C(=O)O', 'Ester': 'C(=O)O[C,c]', 'Amide': 'C(=O)N',
            'Amine': 'N', 'Alcohol': '[C,c]O[H]', 'Ketone': 'C(=O)[C,c]',
            'Nitro': 'N(=O)=O', 'Sulfone': 'S(=O)(=O)', 'Phenyl': 'c1ccccc1',
            'Halogen': '[F,Cl,Br,I]'
        }
        for name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    if atom_idx in match: return name
        return f"{atom.GetSymbol()}_group"

    def _analyze_feature_importance(self, graph: HeteroData, explanation: Dict) -> Dict:
        if explanation is None: return {}
        feature_importance = {}
        node_importance = explanation.get('node_importance', {})
        for node_type in graph.node_types:
            if node_type not in graph.x_dict or node_type not in node_importance: continue
            features = graph[node_type].x.detach().cpu().numpy()
            node_imp = node_importance[node_type]
            if len(node_imp) == 0 or len(features) == 0: continue
            
            weighted_features = features.T * node_imp
            feature_scores = np.abs(weighted_features).mean(axis=1)
            top_indices = np.argsort(feature_scores)[-5:][::-1]
            top_features_list = [
                {
                    'name': self.atom_feature_names[idx] if idx < len(self.atom_feature_names) else f'Feat_{idx}',
                    'importance': float(feature_scores[idx]),
                    'avg_value': float(features[:, idx].mean())
                }
                for idx in top_indices if idx < feature_scores.shape[0]
            ]
            feature_importance[node_type] = {'top_features': top_features_list}
        return feature_importance

    def _analyze_node_type_contributions(self, graph: HeteroData, explanation: Dict) -> Dict:
        if explanation is None: return {}
        contributions = {}
        node_importance = explanation.get('node_importance', {})
        total = 0
        for node_type, importance in node_importance.items():
            val = float(importance.sum()) if len(importance) > 0 else 0
            contributions[node_type] = val
            total += val
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        return contributions



class ExplainabilityVisualizer:
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def create_comprehensive_report(self, explanation_dict: Dict, save_name: str):
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
        
        pred_class = explanation_dict['prediction']['class']
        confidence = explanation_dict['prediction']['confidence']
        fig.suptitle(
            f'Combined GNN + Gradient Explanation Report\n'
            f'Prediction: {pred_class} (Confidence: {confidence:.2%})',
            fontsize=16, fontweight='bold'
        )
        
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_molecule_with_importance(ax1, explanation_dict['smiles'], explanation_dict)
        
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_prediction_confidence(ax2, explanation_dict['prediction'])
        
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_critical_substructures(ax3, explanation_dict['critical_substructures'])
        
        ax4 = fig.add_subplot(gs[2, :2])
        ax4.text(0.5, 0.5, "Detailed Feature Importance\nSee Metrics Report (Image 2)", ha='center', fontsize=12)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_node_contributions(ax5, explanation_dict['node_contributions'])
        
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_counterfactual_impact(ax6, explanation_dict['counterfactuals'])
        
        save_path = self.results_dir / f'{save_name}_comprehensive.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive report saved: {save_path}")
        plt.close()
        return save_path

    def create_json_visualization(self, explanation_dict: Dict, save_name: str):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        fig.suptitle('Detailed Metrics & Feature Analysis', fontsize=16, fontweight='bold')

        ax1 = fig.add_subplot(gs[0, :])
        feat_imp = explanation_dict['feature_importance']
        all_features = []
        for ntype, data in feat_imp.items():
            for f in data['top_features']:
                all_features.append({'Name': f"{ntype}:{f['name']}", 'Value': f['importance']})
        if all_features:
            df_feat = pd.DataFrame(all_features).sort_values('Value', ascending=True).tail(15)
            ax1.barh(df_feat['Name'], df_feat['Value'], color='purple')
            ax1.set_title("Top 15 Weighted Chemical Features")
        else:
            ax1.text(0.5, 0.5, "No Feature Data", ha='center')

        ax2 = fig.add_subplot(gs[1, 0])
        metrics = explanation_dict.get('metrics', {})
        if metrics:
            keys = list(metrics.keys())
            vals = list(metrics.values())
            ax2.bar(keys, vals, color=['orange', 'blue', 'green'])
            ax2.set_ylim(0, 1)
            ax2.set_title("Explanation Quality Metrics")
            for i, v in enumerate(vals):
                ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')
        else:
            ax2.text(0.5, 0.5, "No Metrics", ha='center')

        ax3 = fig.add_subplot(gs[1, 1])
        probs = explanation_dict['prediction']['probabilities']
        ax3.pie(probs, labels=['Authentic', 'Counterfeit'], autopct='%1.1f%%', colors=['green', 'red'])
        ax3.set_title("Model Probability Split")

        save_path = self.results_dir / f'{save_name}_metrics_detail.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics report saved: {save_path}")
        plt.close()

    def _plot_molecule_with_importance(self, ax, smiles: str, explanation_dict: Dict):
        
        substructures = explanation_dict['critical_substructures']
        feature_imp_data = explanation_dict['feature_importance']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return
        
        # 1. Compute Importance & Champion
        atom_importance = np.zeros(mol.GetNumAtoms())
        max_score = 0.0
        champion_atom_idx = -1
        
        for substruct in substructures:
            score = substruct['importance_score']
            for atom_idx in substruct['atom_indices']:
                atom_importance[atom_idx] = max(atom_importance[atom_idx], score)
                if atom_importance[atom_idx] > max_score:
                    max_score = atom_importance[atom_idx]
                    champion_atom_idx = atom_idx

        norm_val = atom_importance.max() if atom_importance.max() > 0 else 1.0
        norm_importance = atom_importance / norm_val

        highlight_atoms = []
        highlight_bonds = []
        atom_colors = {}
        bond_colors = {}
        
        for idx, score in enumerate(norm_importance):
            if score > 0.2: 
                highlight_atoms.append(idx)
                if score > 0.6:
                    atom_colors[idx] = (1.0, 0.0, 0.0) 
                elif score > 0.3:
                    atom_colors[idx] = (1.0, 0.8, 0.0)
                else:
                    atom_colors[idx] = (0.8, 0.8, 0.8)

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u in highlight_atoms and v in highlight_atoms:
                highlight_bonds.append(bond.GetIdx())
                c_u = atom_colors[u]
                c_v = atom_colors[v]
                bond_colors[bond.GetIdx()] = ((c_u[0]+c_v[0])/2, (c_u[1]+c_v[1])/2, (c_u[2]+c_v[2])/2)

        try:
            AllChem.Compute2DCoords(mol)
            try: AllChem.GenerateDepictionMatching2DStructure(mol, mol)
            except: pass

            drawer = rdMolDraw2D.MolDraw2DCairo(900, 650)
            opts = drawer.drawOptions()
            opts.bondLineWidth = 3
            opts.highlightBondWidthMultiplier = 2
            opts.addAtomIndices = False
            opts.atomLabelFontSize = 16
            opts.padding = 0.15 

            drawer.DrawMolecule(mol, 
                                highlightAtoms=highlight_atoms,
                                highlightAtomColors=atom_colors,
                                highlightBonds=highlight_bonds,
                                highlightBondColors=bond_colors)
            drawer.FinishDrawing()
            
            png_data = drawer.GetDrawingText()
            bio = io.BytesIO(png_data)
            img = Image.open(bio)
            ax.imshow(img, extent=[0, 1, 0, 1]) 
            ax.axis('off')
            ax.set_title('Structural Importance & Primary Driver', fontsize=14, fontweight='bold', pad=15)

            if champion_atom_idx != -1 and max_score > 0.4:
                pos = mol.GetConformer().GetAtomPosition(champion_atom_idx)
                x_norm = 0.5 + (pos.x / 15.0) 
                y_norm = 0.5 - (pos.y / 15.0)

                atom_symbol = mol.GetAtomWithIdx(champion_atom_idx).GetSymbol()
                top_feature_text = "Unknown"
                if atom_symbol in feature_imp_data and feature_imp_data[atom_symbol]['top_features']:
                    top_feat = feature_imp_data[atom_symbol]['top_features'][0]
                    feat_name = top_feat['name'].replace('Is_', '').replace('_', ' ')
                    feat_val = top_feat['avg_value']
                    top_feature_text = f"Feature Driver:\n{feat_name}\n(Val: {feat_val:.2f})"

                text_x = max(0.05, min(0.95, x_norm - 0.25))
                text_y = max(0.05, min(0.95, y_norm + 0.25))

                ax.annotate(
                    top_feature_text,
                    xy=(x_norm, y_norm),
                    xytext=(text_x, text_y),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    fontsize=11, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="darkred", lw=2, alpha=0.9),
                    arrowprops=dict(facecolor='darkred', shrink=0.05, width=3, headwidth=10, connectionstyle="arc3,rad=-0.2")
                )

        except Exception as e:
            logger.error(f"Drawing error: {e}")
            ax.text(0.5, 0.5, f"Error drawing molecule: {e}", ha='center')
            ax.axis('off')

    def _plot_prediction_confidence(self, ax, prediction):
        probs = prediction['probabilities']
        classes = ['Authentic', 'Counterfeit']
        colors = ['green', 'red']
        ax.barh(classes, probs, color=colors, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_title('Confidence')

    def _plot_critical_substructures(self, ax, substructures):
        if not substructures: return
        data = []
        for s in substructures[:5]:
            data.append({
                'Group': s['functional_group'],
                'Center': s['center_atom_symbol'],
                'Score': s['importance_score']
            })
        df = pd.DataFrame(data)
        ax.barh(df.index, df['Score'], color='steelblue')
        ax.set_yticks(df.index)
        ax.set_yticklabels([f"{row['Group']} ({row['Center']})" for _, row in df.iterrows()])
        ax.set_title('Top Substructures Importance')

    def _plot_node_contributions(self, ax, contributions):
        if not contributions: return
        labels = list(contributions.keys())
        sizes = list(contributions.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Atom Type Contribution')

    def _plot_counterfactual_impact(self, ax, counterfactuals):
        ax.axis('off')
        text = "Counterfactual Analysis (Virtual Perturbation)\n"
        text += "=" * 50 + "\n\n"
        text += f"Impact Level: {counterfactuals.get('impact_level', 'Unknown')}\n"
        text += f"Critical Group: {counterfactuals.get('critical_group', 'Unknown')}\n\n"
        text += f"Insight: {counterfactuals.get('text', 'No data')}\n"

        color_map = {'High': '#ffcccc', 'Medium': '#fff4cc', 'Low': '#ccffcc'}
        color = color_map.get(counterfactuals.get('impact_level'), 'white')

        ax.text(0.05, 0.9, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        ax.set_title('Intuitive Counterfactuals', fontsize=12, fontweight='bold')


    def plot_layer_wise_trajectories(self, ax, hgt_attention: Dict, top_n_per_type: int = 3):
        if not hgt_attention or 'trajectories' not in hgt_attention:
            ax.text(0.5, 0.5, 'No HGT attention data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='#6b7280')
            ax.axis('off')
            return

        trajs = hgt_attention['trajectories']
        num_layers = hgt_attention.get('num_layers', 0)
        if num_layers == 0:
            ax.text(0.5, 0.5, 'No layers captured', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            return

        type_colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajs), 1)))
        layer_axis = np.arange(1, num_layers + 1)

        all_handles = []
        for type_idx, (nt, per_layer_list) in enumerate(trajs.items()):
            if not per_layer_list:
                continue
            stacked = np.stack(per_layer_list, axis=0)  # [L, N_nt]
            if stacked.shape[1] == 0:
                continue

            color = type_colors[type_idx]

            for atom_local_idx in range(stacked.shape[1]):
                ax.plot(layer_axis, stacked[:, atom_local_idx],
                        color=color, alpha=0.15, linewidth=0.8)

            final_scores = stacked[-1, :]
            top_idx = np.argsort(-final_scores)[:top_n_per_type]
            for rank, a_idx in enumerate(top_idx):
                handle, = ax.plot(
                    layer_axis, stacked[:, a_idx],
                    color=color,
                    linewidth=2.2 if rank == 0 else 1.5,
                    marker='o', markersize=6 if rank == 0 else 4,
                    label=f'{nt}[{a_idx}]' if rank == 0 else None,
                    alpha=0.95 if rank == 0 else 0.65,
                )
                if rank == 0:
                    all_handles.append(handle)

        ax.set_xticks(layer_axis)
        ax.set_xlabel('HGT Layer', fontsize=10)
        ax.set_ylabel('Normalized native attention', fontsize=10)
        ax.set_title(
            'Layer-wise HGT Attention Trajectories\n'
            '(per atom, grouped by element type — native softmax scores)',
            fontsize=11, fontweight='bold', pad=10,
        )
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.25, linestyle='--')
        if all_handles:
            ax.legend(handles=all_handles, fontsize=8, loc='upper left',
                      frameon=True, framealpha=0.9, title='Top atom per type',
                      title_fontsize=8)

    def plot_edge_type_attention(self, ax, hgt_attention: Dict):
        if not hgt_attention or 'edge_type_attention' not in hgt_attention:
            ax.text(0.5, 0.5, 'No edge-type attention', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='#6b7280')
            ax.axis('off')
            return

        eta = hgt_attention['edge_type_attention']
        items = [(et, v) for et, v in eta.items() if v > 0]
        if not items:
            ax.text(0.5, 0.5, 'No active edge types', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='#6b7280')
            ax.axis('off')
            return

        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:15]

        labels = [f'{et[0]} -> {et[2]}' for et, _ in items]
        values = [v for _, v in items]
        y_pos = np.arange(len(labels))

        cmap = LinearSegmentedColormap.from_list(
            'hgt', ['#fed976', '#fd8d3c', '#e31a1c', '#800026']
        )
        max_v = max(values) if values else 1.0
        colors = [cmap(v / max_v) for v in values]

        bars = ax.barh(y_pos, values, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean softmax attention', fontsize=10)
        ax.set_title(
            'Heterogeneous Edge-Type Attention\n'
            "(HGT's unique advantage — per-relation weighting)",
            fontsize=11, fontweight='bold', pad=10,
        )
        ax.grid(True, alpha=0.25, axis='x', linestyle='--')
        for bar, v in zip(bars, values):
            ax.text(v + max_v * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=8, color='#e2e8f0')

    def plot_method_agreement(self, ax, explanation_dict: Dict):
        sources = {
            'GNNExplainer': explanation_dict.get('gnn_explainer_raw'),
            'Gradient': explanation_dict.get('gradient_raw'),
            'HGT Native': explanation_dict.get('hgt_attention'),
        }

        node_types_seen = []
        for src_dict in sources.values():
            if src_dict is None:
                continue
            key = 'node_importance' if 'node_importance' in src_dict else 'node_attention'
            if key in src_dict:
                node_types_seen = list(src_dict[key].keys())
                break

        if not node_types_seen:
            ax.text(0.5, 0.5, 'No method data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            return

        rows = []
        method_labels = list(sources.keys())
        for nt in node_types_seen:
            n = 0
            for src_dict in sources.values():
                if src_dict is None:
                    continue
                key = 'node_importance' if 'node_importance' in src_dict else 'node_attention'
                if key in src_dict and nt in src_dict[key]:
                    n = max(n, len(src_dict[key][nt]))
            if n == 0:
                continue
            for local_i in range(n):
                row = []
                for src_dict in sources.values():
                    if src_dict is None:
                        row.append(0.0)
                        continue
                    key = 'node_importance' if 'node_importance' in src_dict else 'node_attention'
                    arr = src_dict.get(key, {}).get(nt)
                    if arr is None or local_i >= len(arr):
                        row.append(0.0)
                    else:
                        row.append(float(arr[local_i]))
                rows.append((f'{nt}[{local_i}]', row))

        if not rows:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            return

        labels = [r[0] for r in rows]
        matrix = np.array([r[1] for r in rows], dtype=np.float64)

        for c in range(matrix.shape[1]):
            col = matrix[:, c]
            m = col.max()
            if m > 0:
                matrix[:, c] = col / m

        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(method_labels)))
        ax.set_xticklabels(method_labels, fontsize=9)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(
            'Per-Atom Importance by Method\n(red rows = consensus critical atoms)',
            fontsize=11, fontweight='bold', pad=10,
        )
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='Normalized score')

    def create_attention_report(self, explanation_dict: Dict, save_name: str):
        fig = plt.figure(figsize=(18, 13))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3,
                               height_ratios=[1.0, 1.1])

        pred = explanation_dict['prediction']
        agreement = explanation_dict.get('metrics', {}).get('method_agreement', 0.0)

        fig.suptitle(
            f"Native HGT Attention Analysis · {pred['class']} "
            f"({pred['confidence']:.1%})\n"
            f"3-method agreement score: {agreement:+.3f}  "
            f"(GNNExplainer <-> Gradient <-> HGT native)",
            fontsize=14, fontweight='bold',
        )

        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_layer_wise_trajectories(ax1, explanation_dict.get('hgt_attention'))

        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_edge_type_attention(ax2, explanation_dict.get('hgt_attention'))

        ax3 = fig.add_subplot(gs[1, :])
        self.plot_method_agreement(ax3, explanation_dict)

        save_path = self.results_dir / f'{save_name}_hgt_attention.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"HGT attention report saved: {save_path}")
        return save_path


class PyGExplainabilityPipeline:
    def __init__(self, model, device, dataset=None):
        self.model = model
        self.device = device
        self.explainer = PyGNativeExplainer(model, device, dataset_sample=dataset)
        self.visualizer = ExplainabilityVisualizer()
    
    def explain_dataset(self, dataset, num_samples=5):
        import random
        logger.info(f"Explaining {num_samples} random molecules...")
        
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        
        count_auth, count_fake = 0, 0
        selected_graphs, selected_indices = [], []
        target = (num_samples // 2) + (num_samples % 2)

        for idx in all_indices:
            g = dataset[idx]
            is_fake = (g.y.item() == 1)
            if is_fake and count_fake < target:
                selected_graphs.append(g)
                selected_indices.append(idx)
                count_fake += 1
            elif not is_fake and count_auth < target:
                selected_graphs.append(g)
                selected_indices.append(idx)
                count_auth += 1
            if len(selected_graphs) >= num_samples: break
        
        for i, graph in enumerate(selected_graphs):
            original_idx = selected_indices[i]
            smiles = graph.smiles if hasattr(graph, 'smiles') else f"mol_{original_idx}"
            explanation = self.explainer.explain_molecule(graph, smiles)
            save_name = f"mol_{original_idx}_{explanation['prediction']['class']}"

            self.visualizer.create_comprehensive_report(explanation, save_name)
            self.visualizer.create_json_visualization(explanation, save_name)
            self.visualizer.create_attention_report(explanation, save_name)


def load_model_and_dataset():
    model_files = list(Path('./HGT_Enhanced_Results').glob("best_model_*.pt"))
    if not model_files:
        print("No model found. Run training_script.py first.")
        return None, None, None
    latest = max(model_files, key=lambda x: x.stat().st_mtime)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(latest, map_location=device, weights_only=False)
    model = RobustEnhancedHGTDetector(feature_dims=checkpoint['feature_dims']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    data_files = list(Path('.').glob("*enhanced*.pt"))
    if not data_files:
        data_files = list(Path('.').glob("*.pt"))
        data_files = [f for f in data_files if "best_model" not in str(f)]
        
    if not data_files:
        print("No dataset found.")
        return model, None, device
        
    dataset = torch.load(data_files[0], map_location='cpu', weights_only=False)[0] 
    return model, dataset, device

def main():
    model, dataset, device = load_model_and_dataset()
    if not model or not dataset: return
    pipeline = PyGExplainabilityPipeline(model, device, dataset=dataset)
    pipeline.explain_dataset(dataset, num_samples=5)

if __name__ == "__main__":
    main()