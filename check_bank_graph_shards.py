#!/usr/bin/env python3
"""Check pinned graph shards, offsets, bond attributes and lazy-loader behavior."""

import argparse
import csv
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from bank_graph_shards import (BankGraphDataset, BankShardBatchSampler, EDGE_DIM,
                               NODE_DIM, _read_only, file_sha256, molecule_arrays, prepare)


def validate(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["state"] != "complete":
        raise ValueError("Cannot validate a complete dataset while preparation is incomplete")
    if file_sha256(root / "graph_index.csv") != manifest["graph_index_csv_sha256"]:
        raise ValueError("Graph CSV index hash mismatch")
    if file_sha256(root / "graph_index.sqlite") != manifest["graph_index_sqlite_sha256"]:
        raise ValueError("Graph SQLite index hash mismatch")
    graph_count = node_count = edge_count = 0
    with closing(_read_only(root / "graph_index.sqlite")) as connection:
        for record in manifest["shards"]:
            path = root / record["graph_file"]
            assert file_sha256(path) == record["sha256"]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            assert not {"y", "label", "labels"} & arrays.keys()
            rows = list(connection.execute("SELECT * FROM items WHERE shard_no=? ORDER BY graph_index", (record["shard_no"],)))
            n = len(rows)
            assert len(arrays["molecule_ids"]) == len(arrays["smiles"]) == n == record["graph_count"]
            assert arrays["x"].shape == (int(arrays["node_ptr"][-1]), NODE_DIM)
            assert arrays["edge_attr"].shape == (int(arrays["edge_ptr"][-1]), EDGE_DIM)
            assert arrays["edge_index"].shape == (2, int(arrays["edge_ptr"][-1]))
            assert np.isfinite(arrays["x"]).all() and np.isfinite(arrays["edge_attr"]).all()
            assert len(arrays["node_ptr"]) == len(arrays["edge_ptr"]) == n + 1
            assert arrays["node_ptr"][0] == arrays["edge_ptr"][0] == 0
            assert (np.diff(arrays["node_ptr"]) > 0).all() and (np.diff(arrays["edge_ptr"]) >= 0).all()
            assert arrays["edge_index"].shape[1] % 2 == 0
            assert np.array_equal(arrays["edge_index"][:, 0::2], arrays["edge_index"][::-1, 1::2])
            assert np.array_equal(arrays["edge_attr"][0::2], arrays["edge_attr"][1::2])
            if len(arrays["edge_attr"]):
                assert (arrays["edge_attr"][:, :5].sum(axis=1) == 1).all()
                assert np.isin(arrays["edge_attr"], [0, 1]).all()
            for i, row in enumerate(rows):
                first, last = map(int, arrays["node_ptr"][i:i+2])
                begin, end = map(int, arrays["edge_ptr"][i:i+2])
                assert str(arrays["molecule_ids"][i]) == row["molecule_id"]
                assert str(arrays["smiles"][i]) == row["smiles"]
                assert row["num_nodes"] == last - first and row["num_edges"] == end - begin
                localized = arrays["edge_index"][:, begin:end] - first
                assert not localized.size or (localized.min() >= 0 and localized.max() < last - first)
            graph_count += n
            node_count += arrays["x"].shape[0]
            edge_count += arrays["edge_index"].shape[1]
    with (root / "graph_index.csv").open(newline="") as stream:
        csv_count = sum(1 for _ in csv.DictReader(stream))
    assert graph_count == csv_count == manifest["snapshot"]["selected_molecules"]
    return {"checks_passed": True, "graphs": graph_count, "shards": len(manifest["shards"]),
            "nodes": node_count, "directed_bond_edges": edge_count,
            "node_features": NODE_DIM, "bond_features": EDGE_DIM,
            "intrinsic_counterfeit_labels": False}


def self_check():
    with tempfile.TemporaryDirectory(prefix="bank_graph_check_") as temporary:
        base = Path(temporary)
        db, out = base / "catalogue.sqlite", base / "graphs"
        records = [("mol_a", "C", 1), ("mol_b", "CCO", 1), ("mol_c", "c1ccccc1", 1),
                   ("mol_d", "CC(=O)O", 0), ("mol_e", "C[C@H](O)C(=O)O", 1), ("mol_f", "Cl", 1)]
        connection = sqlite3.connect(db)
        connection.execute("CREATE TABLE molecules(molecule_id TEXT PRIMARY KEY,smiles TEXT,training_eligible INTEGER)")
        connection.executemany("INSERT INTO molecules VALUES(?,?,?)", records)
        connection.commit()
        # Interrupt at a committed shard boundary, then resume the same snapshot.
        partial = prepare(db, out, shard_size=2, max_shards=1)
        assert partial["state"] == "in_progress" and partial["prepared_graphs"] == 2
        first_hash = partial["shards"][0]["sha256"]
        try:
            BankGraphDataset(out)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete dataset was silently accepted")
        complete = prepare(db, out, shard_size=2)
        assert complete["state"] == "complete" and complete["shards"][0]["sha256"] == first_hash
        check = validate(out)
        dataset = BankGraphDataset(out, cache_shards=1)
        assert len(dataset) == 5
        assert dataset[0].edge_index.shape == (2, 0)  # single-atom graph
        for i in (0, 2, 4, 1, 3):
            graph = dataset[i]
            direct = molecule_arrays(graph.smiles)
            assert torch.equal(graph.x, torch.from_numpy(direct[0]))
            assert torch.equal(graph.edge_index, torch.from_numpy(direct[1]))
            assert torch.equal(graph.edge_attr, torch.from_numpy(direct[2]))
            assert "y" not in graph and len(dataset._cache) <= 1
        assert dataset[-1].molecule_id == "mol_f"
        assert dataset[3].smiles == "C[C@H](O)C(=O)O"
        sampler = BankShardBatchSampler(dataset, batch_size=2, shuffle=True, seed=72)
        first = list(iter(sampler))
        assert sorted(index for batch in first for index in batch) == list(range(5))
        assert first == list(iter(sampler))
        sampler.set_epoch(1)
        assert sorted(index for batch in sampler for index in batch) == list(range(5))
        batches = list(DataLoader(dataset, batch_sampler=sampler, num_workers=0))
        assert sum(batch.num_graphs for batch in batches) == 5
        assert all("y" not in batch and batch.edge_attr.shape[1] == EDGE_DIM for batch in batches)
        dataset.close()
        # Even an appended ineligible row changes the full pinned catalogue.
        connection.execute("INSERT INTO molecules VALUES('mol_new','CO',0)")
        connection.commit()
        try:
            prepare(db, out, shard_size=2)
        except ValueError as error:
            assert "changed" in str(error)
        else:
            raise AssertionError("Changed catalogue snapshot was silently resumed")
        connection.close()
        return {**check, "checks": ["atomic staged resume", "holdout eligibility filtering", "pointer localization",
            "single-atom graph", "node/bond feature correspondence", "no labels", "LRU bound",
            "shard-aware sampler coverage", "PyG batching", "changed snapshot refusal"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("molecule_bank_1m/graphs_2d"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    result = self_check() if args.self_check else validate(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
