"""
BenchmarkDatasetDict — Composable Benchmark Dataset Suite for Counterfeit Detection
===================================================================================

Provides a modular, dictionary-like Python interface to the complete collection of
benchmark datasets (36 fine-grained chemical subcategories, 5 macro-categories,
evolved GA/NSGA-II set, and authentic drug pool).

Enables flexible category combination, customizable class balancing, scaffold-aware
splitting, and seamless export to PyTorch Geometric (Data / HeteroData) DataLoaders,
standard PyTorch .pt files, and pandas DataFrames.

Usage:
------
    from benchmark_dataset_dict import BenchmarkDatasetDict

    # 1. Load benchmark suite
    benchmarks = BenchmarkDatasetDict.load("benchmark")
    print(benchmarks.summary())

    # 2. Dictionary-like access:
    ds_aromatize = benchmarks["aromatize"]             # by short alias
    ds_bio = benchmarks["category_bioisostere"]        # by macro-category
    ds_clf = benchmarks["halogen-walk.Cl->F"]          # by full name
    ds_evolved = benchmarks["evolved"]                 # evolutionary set

    # 3. Flexible training combination:
    train_loader, val_loader, test_loader = benchmarks.combine(
        categories=["aromatize", "Cl->F", "grow-single", "add-O"],
        authentic_ratio=1.0,  # 50% counterfeit, 50% authentic
        batch_size=64,
        graph_type="hetero",  # or "homo"
    )

    # 4. Out-of-Distribution (OOD) Transferability Experiment:
    train_loader, ood_test_loaders = benchmarks.get_ood_experiment(
        train_categories=["grow-single", "add-O", "Cl->F"],
        test_categories=["aromatize", "evolved"],
        batch_size=64,
        graph_type="hetero",
    )
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("BenchmarkDatasetDict")

# PyTorch & PyG imports with graceful fallback
try:
    import torch
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.loader import DataLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None
    HeteroData = None
    DataLoader = None


# ═════════════════════════════════════════════════════════════════════
#  Graph Featurization Utilities (Homo & Hetero)
# ═════════════════════════════════════════════════════════════════════

_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "other"]
_HYB = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]

def smiles_to_homo_graph(smiles: str, label: int, metadata: Optional[Dict] = None) -> Optional[Any]:
    """Converts SMILES to homogeneous PyG Data(x, edge_index, y, ...)."""
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric is required for graph conversion.")
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return None

    node_feats = []
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        onehot = [float(sym == e) for e in _ELEMENTS[:-1]] + [float(sym not in _ELEMENTS[:-1])]
        hyb = [float(a.GetHybridization() == h) for h in _HYB]
        atom_f = onehot + hyb + [
            a.GetDegree() / 4.0,
            (a.GetFormalCharge() + 2) / 4.0,
            a.GetTotalNumHs() / 4.0,
            float(a.GetIsAromatic()),
            float(a.IsInRing()),
        ]
        node_feats.append(atom_f)

    x = torch.tensor(node_feats, dtype=torch.float)
    src, dst = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    if not src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        smiles=smiles,
    )
    if metadata:
        for k, v in metadata.items():
            setattr(data, k, v)
    return data


def smiles_to_hetero_graph(smiles: str, label: int, metadata: Optional[Dict] = None) -> Optional[Any]:
    """Converts SMILES to heterogeneous PyG HeteroData matching repository HGT specs."""
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric is required for graph conversion.")
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return None

    hetero = HeteroData()
    atom_types = defaultdict(list)
    atom_features = defaultdict(list)
    atom_to_local = {}

    ring_info = m.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    for idx, atom in enumerate(m.GetAtoms()):
        sym = atom.GetSymbol()
        local_idx = len(atom_types[sym])
        atom_types[sym].append(idx)
        atom_to_local[idx] = (sym, local_idx)

        # 26-dim advanced feature vector matching data.py
        feat = [
            atom.GetDegree() / 4.0,
            (atom.GetFormalCharge() + 2) / 4.0,
            float(atom.GetHybridization()),
            float(atom.GetIsAromatic()),
            atom.GetTotalNumHs() / 4.0,
            float(atom.IsInRing()),
            atom.GetMass() / 100.0,
            float(atom.GetChiralTag()),
            float(atom.GetTotalValence()) / 4.0,
            float(atom.GetNumRadicalElectrons()),
            float(atom.GetExplicitValence()) / 4.0,
            float(len(atom.GetNeighbors())) / 4.0,
            float(atom.GetImplicitValence()) / 4.0,
            float(any(len(r) == 3 for r in atom_rings if idx in r)),
            float(any(len(r) == 4 for r in atom_rings if idx in r)),
            float(any(len(r) == 5 for r in atom_rings if idx in r)),
            float(any(len(r) == 6 for r in atom_rings if idx in r)),
            float(any(len(r) == 7 for r in atom_rings if idx in r)),
            float(atom.GetNumImplicitHs()) / 4.0,
            float(atom.GetNumExplicitHs()) / 4.0,
        ]
        atom_features[sym].append(feat)

    # Set node features
    for sym, feats in atom_features.items():
        hetero[sym].x = torch.tensor(feats, dtype=torch.float)
        hetero[sym].num_nodes = len(feats)

    # Set edges
    edge_indices = defaultdict(list)
    edge_feats = defaultdict(list)

    for bond in m.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sym_i, loc_i = atom_to_local[i]
        sym_j, loc_j = atom_to_local[j]
        btype = str(bond.GetBondType())

        b_feat = [
            float(bond.GetIsAromatic()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]

        # Bidirectional edges
        for (src_sym, dst_sym, u, v) in [(sym_i, sym_j, loc_i, loc_j), (sym_j, sym_i, loc_j, loc_i)]:
            edge_key = (src_sym, f"{src_sym}_{btype}_{dst_sym}", dst_sym)
            edge_indices[edge_key].append([u, v])
            edge_feats[edge_key].append(b_feat)

    for edge_key, indices in edge_indices.items():
        hetero[edge_key].edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
        hetero[edge_key].edge_attr = torch.tensor(edge_feats[edge_key], dtype=torch.float)

    hetero.y = torch.tensor([label], dtype=torch.long)
    hetero.smiles = smiles
    if metadata:
        for k, v in metadata.items():
            setattr(hetero, k, v)
    return hetero


# ═════════════════════════════════════════════════════════════════════
#  SubDataset Class (Represents a single category dataset)
# ═════════════════════════════════════════════════════════════════════

class SubDataset:
    """Wrapper around a single category benchmark dataset."""

    def __init__(self, name: str, df: pd.DataFrame, shortcut_metrics: Optional[Dict] = None):
        self.name = name
        self.df = df.copy().reset_index(drop=True)
        self.metrics = shortcut_metrics or {}

    @property
    def n_total(self) -> int:
        return len(self.df)

    @property
    def n_authentic(self) -> int:
        return int((self.df["label"] == 0).sum())

    @property
    def n_counterfeit(self) -> int:
        return int((self.df["label"] == 1).sum())

    @property
    def balance_ratio(self) -> float:
        return self.n_counterfeit / max(1, self.n_authentic)

    def filter_split(self, split: str) -> pd.DataFrame:
        if "split" not in self.df.columns or split == "all":
            return self.df
        return self.df[self.df["split"] == split].reset_index(drop=True)

    def to_dataframe(self, split: str = "all") -> pd.DataFrame:
        return self.filter_split(split)

    def to_pyg_graphs(self, split: str = "all", graph_type: str = "hetero") -> List[Any]:
        df = self.filter_split(split)
        graphs = []
        converter = smiles_to_hetero_graph if graph_type == "hetero" else smiles_to_homo_graph
        for _, row in df.iterrows():
            meta = {
                "category": row.get("category", ""),
                "subcategory": row.get("subcategory", ""),
                "original_smiles": row.get("parent_smiles", row["smiles"]),
            }
            g = converter(row["smiles"], int(row["label"]), meta)
            if g is not None:
                graphs.append(g)
        return graphs

    def get_dataloader(
        self,
        split: str = "all",
        batch_size: int = 64,
        shuffle: bool = True,
        graph_type: str = "hetero",
    ) -> Any:
        graphs = self.to_pyg_graphs(split=split, graph_type=graph_type)
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for DataLoader.")
        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)

    def to_ecfp(self, split: str = "all", radius: int = 2, n_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        df = self.filter_split(split)
        fps, labels = [], []
        for _, row in df.iterrows():
            m = Chem.MolFromSmiles(row["smiles"])
            if m is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, n_bits)
                arr = np.zeros((n_bits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
                labels.append(int(row["label"]))
        return np.array(fps), np.array(labels)

    def save_csv(self, path: Union[str, Path]) -> None:
        self.df.to_csv(path, index=False)

    def save_pt(self, path: Union[str, Path], graph_type: str = "hetero") -> None:
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required to save .pt files.")
        graphs = self.to_pyg_graphs(graph_type=graph_type)
        labels = [int(g.y.item()) for g in graphs]
        node_types = set()
        edge_types = set()
        if graphs and graph_type == "hetero":
            for g in graphs:
                node_types.update(g.node_types)
                edge_types.update(g.edge_types)
        metadata = (sorted(list(node_types)), sorted(list(edge_types)))
        torch.save((graphs, labels, metadata), path)
        logger.info(f"Saved {len(graphs)} graphs to {path}")

    def __len__(self) -> int:
        return self.n_total

    def __repr__(self) -> str:
        return (
            f"SubDataset(name='{self.name}', total={self.n_total:,}, "
            f"auth={self.n_authentic:,}, fake={self.n_counterfeit:,})"
        )


# ═════════════════════════════════════════════════════════════════════
#  BenchmarkDatasetDict (Central Dictionary Class)
# ═════════════════════════════════════════════════════════════════════

class BenchmarkDatasetDict:
    """
    Central dictionary of benchmark datasets.
    Provides indexed access, aliases, combiners, and OOD benchmark generators.
    """

    def __init__(self, datasets: Dict[str, SubDataset], authentic_pool: Optional[pd.DataFrame] = None):
        self._datasets: Dict[str, SubDataset] = datasets
        self._alias_map: Dict[str, str] = {}
        self.authentic_pool: Optional[pd.DataFrame] = authentic_pool
        self._build_alias_map()

    def _build_alias_map(self) -> None:
        """Builds flexible short aliases for all categories and subcategories."""
        self._alias_map.clear()
        for canonical_name in self._datasets.keys():
            self._alias_map[canonical_name.lower()] = canonical_name
            # Without category_ or subcategory_ prefixes
            clean = canonical_name.replace("category_", "").replace("subcategory_", "")
            self._alias_map[clean.lower()] = canonical_name

            # If it's a dotted subcategory e.g. scaffold-hop.aromatize
            if "." in clean:
                sub_part = clean.split(".", 1)[1]
                self._alias_map[sub_part.lower()] = canonical_name

    def _resolve_name(self, key: str) -> str:
        norm = key.strip().lower()
        if norm in self._alias_map:
            return self._alias_map[norm]
        # Partial match fallback
        for alias, canonical in self._alias_map.items():
            if norm == alias or norm in alias:
                return canonical
        raise KeyError(
            f"Category '{key}' not found in BenchmarkDatasetDict. "
            f"Available categories: {sorted(list(self.list_categories()))}"
        )

    # ── Dict-like Protocol ──

    def __getitem__(self, key: str) -> SubDataset:
        canonical = self._resolve_name(key)
        return self._datasets[canonical]

    def __contains__(self, key: str) -> bool:
        try:
            self._resolve_name(key)
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self._datasets)

    def __iter__(self) -> Iterator[str]:
        return iter(self._datasets.keys())

    def keys(self):
        return self._datasets.keys()

    def values(self):
        return self._datasets.values()

    def items(self):
        return self._datasets.items()

    def list_categories(self, subcategories_only: bool = False) -> List[str]:
        if subcategories_only:
            return sorted([k for k in self._datasets.keys() if "subcategory_" in k or "." in k])
        return sorted(list(self._datasets.keys()))

    # ── Combiner Method ──

    def combine(
        self,
        categories: Union[str, List[str]] = "all",
        split: str = "train",
        authentic_ratio: float = 1.0,
        max_samples_per_category: Optional[int] = None,
        category_weights: Optional[Dict[str, float]] = None,
        batch_size: Optional[int] = None,
        graph_type: str = "hetero",
        seed: int = 42,
    ) -> Union[SubDataset, Any]:
        """
        Combines any arbitrary subset of categories into a unified dataset.

        Args:
            categories: List of category names/aliases, or 'all'.
            split: 'train', 'val', 'test', or 'all'.
            authentic_ratio: Ratio of authentic to counterfeit molecules (1.0 = 50/50 balance).
            max_samples_per_category: Optional cap per category.
            category_weights: Optional sampling weights dict.
            batch_size: If specified, directly returns a PyG DataLoader.
            graph_type: 'hetero' or 'homo'.
            seed: Random seed.
        """
        rng = random.Random(seed)

        if categories == "all":
            selected_names = [k for k in self._datasets.keys() if k not in ("pooled", "authentic")]
        elif isinstance(categories, (list, tuple, set)):
            selected_names = [self._resolve_name(c) for c in categories]
        else:
            selected_names = [self._resolve_name(categories)]

        combined_fakes = []
        combined_auths = []

        for name in selected_names:
            ds = self._datasets[name]
            df_split = ds.filter_split(split)

            fakes = df_split[df_split["label"] == 1].to_dict("records")
            auths = df_split[df_split["label"] == 0].to_dict("records")

            if category_weights and name in category_weights:
                weight = category_weights[name]
                sample_k = int(len(fakes) * weight)
                rng.shuffle(fakes)
                fakes = fakes[:sample_k]

            if max_samples_per_category is not None:
                rng.shuffle(fakes)
                fakes = fakes[:max_samples_per_category]

            combined_fakes.extend(fakes)
            combined_auths.extend(auths)

        # Apply authentic ratio
        rng.shuffle(combined_fakes)
        rng.shuffle(combined_auths)

        if authentic_ratio <= 0.0:
            final_rows = combined_fakes
        else:
            n_auth_needed = int(len(combined_fakes) * authentic_ratio)
            # If subdatasets don't have enough auth, draw from pool if available
            if len(combined_auths) < n_auth_needed and self.authentic_pool is not None:
                needed = n_auth_needed - len(combined_auths)
                pool_records = self.authentic_pool.to_dict("records")
                rng.shuffle(pool_records)
                combined_auths.extend(pool_records[:needed])

            final_rows = combined_fakes + combined_auths[:n_auth_needed]

        rng.shuffle(final_rows)
        combined_df = pd.DataFrame(final_rows).drop_duplicates(subset=["smiles"]).reset_index(drop=True)

        dataset_name = f"combined_{split}_{len(selected_names)}cats"
        sub_ds = SubDataset(dataset_name, combined_df)

        if batch_size is not None:
            return sub_ds.get_dataloader(split="all", batch_size=batch_size, graph_type=graph_type)
        return sub_ds

    # ── Out-of-Distribution (OOD) Benchmark Experiment ──

    def get_ood_experiment(
        self,
        train_categories: List[str],
        test_categories: List[str],
        batch_size: int = 64,
        authentic_ratio: float = 1.0,
        graph_type: str = "hetero",
        seed: int = 42,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Creates an Out-Of-Distribution (OOD) / Generalization experiment:
        Trains strictly on `train_categories`, and provides separate test DataLoaders
        for each out-of-distribution category in `test_categories`.
        """
        logger.info(f"Setting up OOD Experiment:")
        logger.info(f"  • Training on: {train_categories}")
        logger.info(f"  • Testing OOD on: {test_categories}")

        train_ds = self.combine(
            categories=train_categories,
            split="train",
            authentic_ratio=authentic_ratio,
            seed=seed,
        )
        val_ds = self.combine(
            categories=train_categories,
            split="val",
            authentic_ratio=authentic_ratio,
            seed=seed,
        )

        train_loader = train_ds.get_dataloader(batch_size=batch_size, shuffle=True, graph_type=graph_type)
        val_loader = val_ds.get_dataloader(batch_size=batch_size, shuffle=False, graph_type=graph_type)

        ood_test_loaders = {}
        for test_cat in test_categories:
            canonical = self._resolve_name(test_cat)
            test_ds = self._datasets[canonical]
            # Use test split or all if split missing
            df_test = test_ds.filter_split("test")
            if len(df_test) < 20:
                df_test = test_ds.to_dataframe()
            sub = SubDataset(f"ood_{test_cat}", df_test)
            ood_test_loaders[test_cat] = sub.get_dataloader(
                batch_size=batch_size, shuffle=False, graph_type=graph_type
            )

        return train_loader, val_loader, ood_test_loaders

    # ── Summary & Reporting ──

    def summary(self) -> str:
        """Generates a comprehensive summary table of all loaded benchmark datasets."""
        lines = []
        lines.append("=" * 95)
        lines.append(f"{'BENCHMARK DATASET DICTIONARY SUMMARY':^95}")
        lines.append("=" * 95)
        lines.append(f"{'Category Name':<42s} | {'Total':>7s} | {'Auth':>6s} | {'Fake':>6s} | {'Balance':>7s} | {'Shortcuts (LR)':>14s}")
        lines.append("-" * 95)

        for name, ds in sorted(self._datasets.items()):
            lr_acc = ds.metrics.get("trivial_lr_accuracy")
            lr_str = f"{lr_acc:.3f}" if lr_acc is not None else "N/A"
            clean_name = name.replace("subcategory_", "sub: ").replace("category_", "cat: ")
            lines.append(
                f"{clean_name:<42s} | {ds.n_total:>7,d} | {ds.n_authentic:>6,d} | "
                f"{ds.n_counterfeit:>6,d} | {ds.balance_ratio:>6.2f}x | {lr_str:>14s}"
            )

        lines.append("-" * 95)
        total_mols = sum(d.n_total for d in self._datasets.values())
        total_fake = sum(d.n_counterfeit for d in self._datasets.values())
        total_auth = sum(d.n_authentic for d in self._datasets.values())
        lines.append(
            f"{'TOTAL ACROSS BENCHMARK SUITE':<42s} | {total_mols:>7,d} | {total_auth:>6,d} | "
            f"{total_fake:>6,d} | {total_fake/max(1, total_auth):>6.2f}x |"
        )
        lines.append("=" * 95)
        return "\n".join(lines)

    # ── Factory Loader ──

    @classmethod
    def load(cls, benchmark_dir: Union[str, Path] = "benchmark") -> BenchmarkDatasetDict:
        """Loads all dataset CSVs and summary JSONs from a benchmark directory."""
        benchmark_dir = Path(benchmark_dir)
        if not benchmark_dir.exists():
            raise FileNotFoundError(f"Benchmark directory '{benchmark_dir}' not found.")

        datasets = {}
        logger.info(f"Scanning benchmark directory: {benchmark_dir.absolute()}...")

        for sub_dir in sorted(benchmark_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            csv_path = sub_dir / "dataset.csv"
            if not csv_path.exists():
                continue

            summary_path = sub_dir / "summary.json"
            metrics = {}
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        data = json.load(f)
                        metrics = data.get("shortcut_metrics", {})
                except Exception:
                    pass

            df = pd.read_csv(csv_path, low_memory=False)
            ds_name = sub_dir.name
            datasets[ds_name] = SubDataset(ds_name, df, metrics)

        logger.info(f"Loaded {len(datasets)} benchmark datasets.")
        return cls(datasets)
