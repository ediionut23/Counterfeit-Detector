"""Versioned geometric classifier with explicit chemical and spatial messages.

This module never changes the historical geometric/GAT implementations. Inputs
use the same 21 atom channels and 14 covalent-bond channels as the recent GATv2
featurizer. Covalent edges and spatial neighbors are separate relations:
nonbonded proximity is never encoded as an "other chemical bond".

``geometry_variant`` selects a bond-only control (``none``), distance context,
or distance plus the existing invariant angular summaries. These variants share
the chemical branch and graph readout; adding geometry also adds learned
spatial modules, so this is not a parameter-count-matched causal experiment.

Expected attributes of a PyG Data/Batch:
    z [N], x [N,21], bond_edge_index [2,B], bond_edge_attr [B,14],
    pos [N,3], spatial_edge_index [2,E], batch [N] (optional for one graph).
``none`` does not read positions or spatial edges. Atom/coordinate alignment
must be verified by the versioned cache builder, not guessed by this model.

Supervision remains generated-analogue versus public-reference origin. No
physical force, conformer optimization or experimental falsification inference
is performed here.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from geometric_angular_model import angular_features, smooth_cutoff


MODEL_VERSION = "shared_gat_chemical_spatial_v2"
NODE_FEATURE_DIM = 21
BOND_FEATURE_DIM = 14
GEOMETRY_VARIANTS = ("none", "distance", "angular")


def _required_tensor(data, name):
    value = getattr(data, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Missing tensor input: {name}")
    return value


def _check_edges(edges, name, n_atoms, batch, device):
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError(f"{name} must have shape [2,E] and dtype torch.long")
    if edges.device != device:
        raise ValueError(f"{name} is on a different device from atom inputs")
    if edges.numel():
        if int(edges.min()) < 0 or int(edges.max()) >= n_atoms:
            raise ValueError(f"{name} contains an atom index outside the batch")
        src, dst = edges
        if bool(torch.any(src == dst)):
            raise ValueError(f"{name} contains a self edge")
        if bool(torch.any(batch[src] != batch[dst])):
            raise ValueError(f"{name} contains a cross-molecule edge")


def validate_graph_inputs(data, geometry_variant="angular"):
    """Validate graph tensors and return molecule membership plus graph count.

    Numerical checks use only the current batch. The dataset must additionally
    validate source identities, chemical schema and atom-coordinate provenance.
    Labels, sample weights and metadata are not predictive model inputs.
    """
    if geometry_variant not in GEOMETRY_VARIANTS:
        raise ValueError("Unknown geometry variant")
    z, x = _required_tensor(data, "z"), _required_tensor(data, "x")
    if z.ndim != 1 or z.dtype != torch.long or not z.numel():
        raise ValueError("z must be a nonempty [N] torch.long tensor")
    n_atoms = len(z)
    if int(z.min()) < 1 or int(z.max()) > 118:
        raise ValueError("Atomic number must lie in [1,118]")
    if x.shape != (n_atoms, NODE_FEATURE_DIM) or not x.is_floating_point():
        raise ValueError("x must have shape [N,21] and floating dtype")
    if x.device != z.device or not bool(torch.isfinite(x).all()):
        raise ValueError("Atom features have incompatible device or nonfinite values")
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(n_atoms, device=z.device, dtype=torch.long)
    if not isinstance(batch, torch.Tensor) or batch.shape != z.shape or batch.dtype != torch.long:
        raise ValueError("batch must have shape [N] and dtype torch.long")
    if batch.device != z.device or int(batch.min()) < 0:
        raise ValueError("batch has incompatible device or negative molecule IDs")
    n_graphs = int(batch.max()) + 1
    if n_graphs > n_atoms:
        raise ValueError("batch molecule IDs exceed the number of atoms")
    counts = torch.bincount(batch, minlength=n_graphs)
    if bool(torch.any(counts == 0)):
        raise ValueError("batch molecule IDs must be contiguous and nonempty")
    declared_graphs = getattr(data, "num_graphs", None)
    if declared_graphs is not None and int(declared_graphs) != n_graphs:
        raise ValueError("Declared graph count disagrees with atom membership")
    bonds = _required_tensor(data, "bond_edge_index")
    bond_attr = _required_tensor(data, "bond_edge_attr")
    _check_edges(bonds, "bond_edge_index", n_atoms, batch, z.device)
    if bond_attr.shape != (bonds.shape[1], BOND_FEATURE_DIM) or not bond_attr.is_floating_point():
        raise ValueError("bond_edge_attr must have shape [B,14] and floating dtype")
    if bond_attr.device != z.device or not bool(torch.isfinite(bond_attr).all()):
        raise ValueError("Bond features have incompatible device or nonfinite values")
    if geometry_variant != "none":
        pos = _required_tensor(data, "pos")
        if pos.shape != (n_atoms, 3) or not pos.is_floating_point():
            raise ValueError("pos must have shape [N,3] and floating dtype")
        if pos.device != z.device or not bool(torch.isfinite(pos).all()):
            raise ValueError("Coordinates have incompatible device or nonfinite values")
        spatial = _required_tensor(data, "spatial_edge_index")
        _check_edges(spatial, "spatial_edge_index", n_atoms, batch, z.device)
        if spatial.numel():
            distances = torch.linalg.vector_norm(pos[spatial[0]] - pos[spatial[1]], dim=-1)
            if bool(torch.any(distances <= 1e-8)):
                raise ValueError("Spatial edges must connect distinct coordinate positions")
    return batch, n_graphs


def _aggregate(messages, destinations, template):
    aggregate = torch.zeros_like(template)
    aggregate.index_add_(0, destinations, messages)
    counts = template.new_zeros(len(template))
    counts.index_add_(0, destinations, torch.ones_like(destinations, dtype=template.dtype))
    return aggregate / counts.clamp_min(1).sqrt()[:, None]


class BondGeometryInteraction(nn.Module):
    """Residual update from distinct covalent and spatial message channels."""

    def __init__(self, hidden, radial_basis, angular_basis, dropout, geometry_variant):
        super().__init__()
        self.geometry_variant = geometry_variant
        self.bond_source = nn.Linear(hidden, hidden, bias=False)
        self.bond_filter = nn.Sequential(
            nn.Linear(BOND_FEATURE_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        if geometry_variant != "none":
            spatial_dim = radial_basis + (2 * angular_basis if geometry_variant == "angular" else 0)
            self.spatial_source = nn.Linear(hidden, hidden, bias=False)
            self.spatial_filter = nn.Sequential(
                nn.Linear(spatial_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        # A fixed two-channel update interface makes the bond-only control
        # explicit: the second input block is zero, never fabricated bond data.
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, features, bond_edges, bond_attr, spatial_edges=None,
                radial=None, angular=None, cutoff_values=None):
        src, dst = bond_edges
        chemical_messages = self.bond_source(features)[src] * self.bond_filter(bond_attr)
        chemical = _aggregate(chemical_messages, dst, features)
        spatial = torch.zeros_like(features)
        if self.geometry_variant != "none":
            src, dst = spatial_edges
            context = torch.cat((radial, angular), dim=-1) if self.geometry_variant == "angular" else radial
            spatial_messages = self.spatial_source(features)[src] * self.spatial_filter(context)
            spatial_messages = spatial_messages * cutoff_values[:, None]
            spatial = _aggregate(spatial_messages, dst, features)
        combined = torch.cat((chemical, spatial), dim=-1)
        return self.norm(features + self.update(combined))


class BondGeometryGNN(nn.Module):
    """Chemical-edge-aware invariant classifier and explicit geometry ablations.

    Every learned covalent message consumes all 14 bond-feature columns through
    an MLP. This establishes a computational and gradient path; predictive
    benefit must still be measured through controlled training comparisons.
    """

    def __init__(self, hidden=64, interactions=3, radial_basis=16, angular_basis=6,
                 cutoff=4.5, dropout=0.1, geometry_variant="angular", validate_inputs=True):
        super().__init__()
        if geometry_variant not in GEOMETRY_VARIANTS:
            raise ValueError(f"geometry_variant must be one of {GEOMETRY_VARIANTS}")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (hidden, interactions, radial_basis, angular_basis)):
            raise ValueError("Model dimensions and block count must be positive integers")
        if not math.isfinite(float(cutoff)) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        if not math.isfinite(float(dropout)) or not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.geometry_variant = geometry_variant
        self.cutoff = float(cutoff)
        self.validate_inputs = bool(validate_inputs)
        self.embedding = nn.Embedding(119, hidden)
        self.node_projection = nn.Linear(NODE_FEATURE_DIM, hidden)
        self.register_buffer("centres", torch.linspace(0, self.cutoff, radial_basis))
        self.gamma = float((radial_basis / self.cutoff) ** 2)
        if geometry_variant == "angular":
            self.register_buffer("angular_centres", torch.linspace(0, self.cutoff, angular_basis))
            self.angular_gamma = float((angular_basis / self.cutoff) ** 2)
            self.angular_projection = nn.Linear(3 * angular_basis, hidden)
            self.angular_gate = nn.Parameter(torch.tensor(0.0))
        self.interactions = nn.ModuleList([
            BondGeometryInteraction(hidden, radial_basis, angular_basis, dropout, geometry_variant)
            for _ in range(interactions)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, data):
        if self.validate_inputs:
            batch, n_graphs = validate_graph_inputs(data, self.geometry_variant)
        else:
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(len(data.z), dtype=torch.long, device=data.z.device)
            n_graphs = int(batch.max()) + 1
        features = self.embedding(data.z) + self.node_projection(data.x)
        spatial_edges = radial = angular = cutoff_values = None
        if self.geometry_variant != "none":
            spatial_edges = data.spatial_edge_index
            src, dst = spatial_edges
            distances = torch.linalg.vector_norm(data.pos[src] - data.pos[dst], dim=-1)
            radial = torch.exp(-self.gamma * (distances[:, None] - self.centres[None, :]) ** 2)
            cutoff_values = smooth_cutoff(distances, self.cutoff)
            if self.geometry_variant == "angular":
                angular, node_context = angular_features(data.pos, spatial_edges,
                    self.angular_centres, self.angular_gamma, self.cutoff)
                features = features + self.angular_gate.sigmoid() * self.angular_projection(node_context)
        for interaction in self.interactions:
            features = interaction(features, data.bond_edge_index, data.bond_edge_attr,
                spatial_edges, radial, angular, cutoff_values)
        summed = features.new_zeros((n_graphs, features.shape[1]))
        summed.index_add_(0, batch, features)
        counts = features.new_zeros(n_graphs)
        counts.index_add_(0, batch, torch.ones(len(features), device=features.device, dtype=features.dtype))
        pooled = torch.cat((summed / counts.clamp_min(1)[:, None],
                            summed / counts.clamp_min(1).sqrt()[:, None]), dim=-1)
        return self.head(pooled).reshape(-1)
