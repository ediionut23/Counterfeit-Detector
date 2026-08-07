
from __future__ import annotations

import math
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import HGTConv
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)




class _LayerCapture:
    """Stores what a single HGTConv layer produced on a single forward pass."""
    __slots__ = ("alpha", "edge_index", "src_offset", "dst_offset",
                 "edge_type_counts", "node_type_counts")

    def __init__(self):
        self.alpha: Optional[Tensor] = None           # [E_total, heads]
        self.edge_index: Optional[Tensor] = None      # [2, E_total] bipartite
        self.src_offset: Dict = {}                    # edge_type -> int
        self.dst_offset: Dict = {}                    # node_type -> int
        self.edge_type_counts: Dict = {}              # edge_type -> num_edges
        self.node_type_counts: Dict = {}              # node_type -> num_nodes


class HGTAttentionExtractor:
    def __init__(self, wrapped_model):
        self.wrapped = wrapped_model
        self.hgt = wrapped_model.hgt_model if hasattr(wrapped_model, "hgt_model") \
                   else wrapped_model
        self._layer_captures: List[_LayerCapture] = []
        self._original_message = {}   # id(layer) -> (original_message, original_propagate)
        self._node_types: List[str] = list(self.hgt.feature_dims.keys())
        self._patch_all_layers()


    def _patch_all_layers(self):
        for layer_idx, conv in enumerate(self.hgt.convs):
            if not isinstance(conv, HGTConv):
                continue
            self._patch_one(conv, layer_idx)

    def _patch_one(self, conv: HGTConv, layer_idx: int):
        original_message = conv.message
        original_propagate = conv.propagate
        self._original_message[id(conv)] = (original_message, original_propagate)

        extractor = self

        def patched_propagate(edge_index, **kwargs):
            while len(extractor._layer_captures) <= layer_idx:
                extractor._layer_captures.append(_LayerCapture())
            cap = extractor._layer_captures[layer_idx]
            cap.edge_index = (edge_index.detach().cpu()
                              if torch.is_tensor(edge_index) else None)
            return original_propagate(edge_index, **kwargs)

        def patched_message(k_j: Tensor, q_i: Tensor, v_j: Tensor,
                            edge_attr: Tensor, index: Tensor,
                            ptr, size_i):
            out = original_message(k_j=k_j, q_i=q_i, v_j=v_j,
                                   edge_attr=edge_attr, index=index,
                                   ptr=ptr, size_i=size_i)

            with torch.no_grad():
                alpha = (q_i * k_j).sum(dim=-1) * edge_attr
                alpha = alpha / math.sqrt(q_i.size(-1))
                alpha = pyg_softmax(alpha, index, ptr, size_i)
                extractor._layer_captures[layer_idx].alpha = alpha.detach().cpu()

            return out  

        conv.message = patched_message
        conv.propagate = patched_propagate


    @torch.no_grad()
    def extract(self, graph: HeteroData) -> Dict:
        self._layer_captures = []
        self.wrapped.eval()

        node_counts = {nt: graph[nt].x.size(0) if nt in graph.x_dict else 0
                       for nt in self._node_types}
        dst_offset = {}
        cumsum = 0
        for nt in self._node_types:
            dst_offset[nt] = cumsum
            cumsum += node_counts[nt]

        src_offset = {}
        src_cumsum = 0
        edge_type_counts = {}
        for et, ei in graph.edge_index_dict.items():
            src_offset[et] = src_cumsum
            src_cumsum += node_counts[et[0]]
            edge_type_counts[et] = int(ei.size(1))

        _ = self.wrapped(graph.x_dict, graph.edge_index_dict, None, 1)

        for cap in self._layer_captures:
            cap.src_offset = dict(src_offset)
            cap.dst_offset = dict(dst_offset)
            cap.node_type_counts = dict(node_counts)
            cap.edge_type_counts = dict(edge_type_counts)

        return self._aggregate(graph)

    def _aggregate(self, graph: HeteroData) -> Dict:
        node_types = self._node_types
        node_counts = {nt: graph[nt].x.size(0) if nt in graph.x_dict else 0
                       for nt in node_types}

        trajectories: Dict[str, List[np.ndarray]] = {nt: [] for nt in node_types}
        raw_alphas_per_layer: List[Dict] = []
        edge_type_attention_accum: Dict[Tuple, List[float]] = defaultdict(list)

        for cap in self._layer_captures:
            if cap.alpha is None:
                continue

            per_type_alpha = {}
            cursor = 0
            dst_to_local_scores: Dict[str, np.ndarray] = {
                nt: np.zeros(node_counts[nt], dtype=np.float64)
                for nt in node_types
            }
            dst_to_edge_counts: Dict[str, np.ndarray] = {
                nt: np.zeros(node_counts[nt], dtype=np.int64)
                for nt in node_types
            }

            for et, ei in graph.edge_index_dict.items():
                num_e = int(ei.size(1))
                if num_e == 0:
                    per_type_alpha[et] = np.zeros((0, cap.alpha.size(1)))
                    continue

                chunk = cap.alpha[cursor:cursor + num_e]   # [num_e, heads]
                cursor += num_e

                per_type_alpha[et] = chunk.numpy()
                per_edge_scalar = chunk.mean(dim=1).numpy()   # [num_e]

                edge_type_attention_accum[et].extend(per_edge_scalar.tolist())

                dst_type = et[2]
                dst_local = ei[1].cpu().numpy()
                scores_arr = dst_to_local_scores[dst_type]
                counts_arr = dst_to_edge_counts[dst_type]
                np.add.at(scores_arr, dst_local, per_edge_scalar)
                np.add.at(counts_arr, dst_local, 1)

            for nt in node_types:
                counts = dst_to_edge_counts[nt]
                scores = dst_to_local_scores[nt]
                with np.errstate(invalid="ignore", divide="ignore"):
                    layer_score = np.where(counts > 0, scores / np.maximum(counts, 1), 0.0)
                m = float(layer_score.max()) if layer_score.size > 0 else 0.0
                if m > 0:
                    layer_score = layer_score / m
                trajectories[nt].append(layer_score.astype(np.float32))

            raw_alphas_per_layer.append(per_type_alpha)

        num_layers = len(self._layer_captures)
        if num_layers == 0:
            final_attention = {nt: np.zeros(node_counts[nt], dtype=np.float32)
                               for nt in node_types}
        else:
            w = np.arange(1, num_layers + 1, dtype=np.float64)
            w = w / w.sum()
            final_attention = {}
            for nt in node_types:
                layers_arr = trajectories[nt]
                if not layers_arr or node_counts[nt] == 0:
                    final_attention[nt] = np.zeros(node_counts[nt], dtype=np.float32)
                    continue
                stacked = np.stack(layers_arr, axis=0)   # [L, N_nt]
                weighted = (stacked * w[:, None]).sum(axis=0)
                m = float(weighted.max())
                if m > 0:
                    weighted = weighted / m
                final_attention[nt] = weighted.astype(np.float32)

        edge_type_attention = {
            et: float(np.mean(vals)) if vals else 0.0
            for et, vals in edge_type_attention_accum.items()
        }

        return {
            "node_attention": final_attention,
            "trajectories": trajectories,
            "edge_type_attention": edge_type_attention,
            "raw_alphas": raw_alphas_per_layer,
            "num_layers": num_layers,
        }


    def close(self):
        for conv in self.hgt.convs:
            saved = self._original_message.pop(id(conv), None)
            if saved is not None:
                original_message, original_propagate = saved
                conv.message = original_message
                conv.propagate = original_propagate
        self._layer_captures = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
