"""Distance and angular continuous-filter GNNs for existing conformer graphs.

Angular features use weighted first/second directional moments. They are exact
weighted averages of cos(theta) and P2(cos(theta)) over distinct neighbors,
computed in linear edge count rather than by materializing neighbor triplets.
Coordinates, radial neighbors and chemical atom properties come from the same
immutable source graphs used by the distance-only model.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from train_geometric_gnn import DistanceGNN as FrozenDistanceGNN


def smooth_cutoff(distance, cutoff):
    values = 0.5 * (torch.cos(math.pi * distance.clamp(max=cutoff) / cutoff) + 1)
    return values * (distance < cutoff).to(distance.dtype)


def angular_features(positions, edges, centres, gamma, cutoff):
    """Return directed-edge and central-atom angular summaries.

    For edge j->i and each radial channel b, edge features average cos(theta_jik)
    and P2(cos(theta_jik)) over other incoming neighbors k!=j, weighted by that
    neighbor's radial basis/cutoff value. Node features average these quantities
    over all distinct ordered pairs, weighted by both neighbors, plus mean
    radial mass. Self-pairs are explicitly subtracted. Empty neighborhoods and
    channels with negligible pair mass receive zero angular context.

    Vector/tensor moments transform covariantly internally; all returned values
    are scalar dot products/contractions and therefore rigid-motion invariant.
    The calculation remains differentiable with respect to actual coordinates.
    """
    src, dst = edges
    n_atoms, n_basis = positions.shape[0], centres.numel()
    vector = positions[src] - positions[dst]
    distances = torch.linalg.vector_norm(vector, dim=-1)
    direction = vector / distances.clamp_min(1e-8)[:, None]
    weight = torch.exp(-gamma * (distances[:, None] - centres[None, :]) ** 2)
    weight = weight * smooth_cutoff(distances, cutoff)[:, None]
    mass = positions.new_zeros((n_atoms, n_basis))
    moment1 = positions.new_zeros((n_atoms, n_basis, 3))
    moment2 = positions.new_zeros((n_atoms, n_basis, 6))
    squared_mass = positions.new_zeros((n_atoms, n_basis))
    count = positions.new_zeros(n_atoms)
    ux, uy, uz = direction.unbind(dim=-1)
    outer = torch.stack((ux * ux, uy * uy, uz * uz, ux * uy, ux * uz, uy * uz), dim=-1)
    mass.index_add_(0, dst, weight)
    squared_mass.index_add_(0, dst, weight.square())
    moment1.index_add_(0, dst, weight[:, :, None] * direction[:, None, :])
    moment2.index_add_(0, dst, weight[:, :, None] * outer[:, None, :])
    count.index_add_(0, dst, torch.ones_like(distances))

    def safe_average(numerator, denominator, low, high):
        value = numerator / denominator.clamp_min(1e-6)
        return torch.where(denominator > 1e-6, value.clamp(low, high), torch.zeros_like(value))

    pair_mass = mass.square() - squared_mass
    first_pairs = moment1.square().sum(dim=-1) - squared_mass
    tensor_norm = moment2[:, :, :3].square().sum(dim=-1) + 2 * moment2[:, :, 3:].square().sum(dim=-1)
    second_pairs = 1.5 * tensor_norm - 0.5 * mass.square() - squared_mass
    node_p1 = safe_average(first_pairs, pair_mass, -1.0, 1.0)
    node_p2 = safe_average(second_pairs, pair_mass, -0.5, 1.0)
    node_mass = mass / count.clamp_min(1)[:, None]
    node_context = torch.cat((node_p1, node_p2, node_mass), dim=-1)

    remaining = mass[dst] - weight
    first_edge = (moment1[dst] * direction[:, None, :]).sum(dim=-1) - weight
    # Off-diagonal tensor elements occur twice in u^T M u.
    contraction = outer * outer.new_tensor([1, 1, 1, 2, 2, 2])
    second_edge = 1.5 * (moment2[dst] * contraction[:, None, :]).sum(dim=-1) - 0.5 * mass[dst] - weight
    edge_p1 = safe_average(first_edge, remaining, -1.0, 1.0)
    edge_p2 = safe_average(second_edge, remaining, -0.5, 1.0)
    return torch.cat((edge_p1, edge_p2), dim=-1), node_context


class AngularInteraction(nn.Module):
    """Chemical messages gated jointly by radial distances and angular context."""
    def __init__(self, hidden, radial_basis, angular_basis, dropout):
        super().__init__()
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.filter = nn.Sequential(nn.Linear(radial_basis + 2 * angular_basis, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, features, radial, angular, cutoff_values, edges):
        src, dst = edges
        filters = self.filter(torch.cat((radial, angular), dim=-1))
        messages = self.source(features)[src] * filters * cutoff_values[:, None]
        aggregate = torch.zeros_like(features)
        aggregate.index_add_(0, dst, messages)
        counts = features.new_zeros(len(features))
        counts.index_add_(0, dst, torch.ones_like(dst, dtype=features.dtype))
        aggregate = aggregate / counts.clamp_min(1).sqrt()[:, None]
        return self.norm(features + self.update(aggregate))


class GeometryVariantGNN(FrozenDistanceGNN):
    """Compatible distance baseline or angular-context invariant GNN.

    ``distance`` reproduces the frozen distance model architecture. ``angular``
    adds direction-moment features to atom inputs and every spatial message
    filter. This is an invariant classifier, not a coordinate-equivariant model
    or a complete representation of all angular/chiral configurations.
    """
    def __init__(self, hidden=64, interactions=3, radial_basis=16, cutoff=4.5,
                 dropout=0.1, geometry_variant="angular", angular_basis=6):
        if geometry_variant not in {"distance", "angular"} or angular_basis < 1:
            raise ValueError("Unknown geometry variant or nonpositive angular basis")
        super().__init__(hidden, interactions, radial_basis, cutoff, dropout)
        self.geometry_variant = geometry_variant
        if geometry_variant == "angular":
            self.register_buffer("angular_centres", torch.linspace(0, cutoff, angular_basis))
            self.angular_gamma = float((angular_basis / cutoff) ** 2)
            self.angular_projection = nn.Linear(3 * angular_basis, hidden)
            self.angular_gate = nn.Parameter(torch.tensor(0.0))
            self.interactions = nn.ModuleList([AngularInteraction(hidden, radial_basis, angular_basis, dropout)
                                              for _ in range(interactions)])

    def forward(self, data):
        if self.geometry_variant == "distance":
            return super().forward(data)
        features = self.embedding(data.z) + self.scalar_projection(data.scalar_features)
        angular, node_context = angular_features(data.pos, data.edge_index, self.angular_centres,
                                                 self.angular_gamma, self.cutoff)
        features = features + self.angular_gate.sigmoid() * self.angular_projection(node_context)
        src, dst = data.edge_index
        distances = torch.linalg.vector_norm(data.pos[src] - data.pos[dst], dim=-1)
        radial = torch.exp(-self.gamma * (distances[:, None] - self.centres[None, :]) ** 2)
        cutoff_values = smooth_cutoff(distances, self.cutoff)
        for interaction in self.interactions:
            features = interaction(features, radial, angular, cutoff_values, data.edge_index)
        batch = data.batch if getattr(data, "batch", None) is not None else torch.zeros(len(features), device=features.device, dtype=torch.long)
        n_graphs = data.num_graphs if hasattr(data, "num_graphs") else 1
        summed = features.new_zeros((n_graphs, features.shape[1]))
        summed.index_add_(0, batch, features)
        counts = features.new_zeros(n_graphs)
        counts.index_add_(0, batch, torch.ones(len(features), device=features.device, dtype=features.dtype))
        pooled = torch.cat((summed / counts.clamp_min(1)[:, None], summed / counts.clamp_min(1).sqrt()[:, None]), dim=-1)
        return self.head(pooled).reshape(-1)
