#!/usr/bin/env python3
"""Prepare and lazily load unlabelled 2D molecular graph shards.

The selected catalogue is streamed from SQLite and pinned by molecule count,
content digest and training-eligibility flags. Preparation can resume only when
that catalogue snapshot and the featurizer configuration still match. A shard
is atomically written before its index transaction is committed.

Node features are the existing train_gnn.atom_features (22 values). Directed
bond attributes are five bond-type indicators (single/double/triple/aromatic/
other), conjugation and ring membership. These features do not fully encode
stereochemistry; the original isomeric SMILES remains in the index. There is no
intrinsic authentic/counterfeit label attached to a chemical graph.

Use BankShardBatchSampler for training: unrestricted random molecule access
would repeatedly decompress different shards despite a small LRU cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import sqlite3
from collections import OrderedDict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data

import train_gnn


FORMAT_VERSION = 1
NODE_DIM = 22
EDGE_DIM = 7
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
EDGE_FEATURE_NAMES = ["single", "double", "triple", "aromatic", "other",
                      "conjugated", "in_ring"]
INDEX_FIELDS = ["dataset_index", "molecule_id", "graph_file", "graph_index", "num_nodes", "num_edges"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_only(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _fingerprint_row(row):
    return json.dumps([row["molecule_id"], row["smiles"], int(row["training_eligible"])],
                      separators=(",", ":"), ensure_ascii=True).encode() + b"\n"


def featurizer_spec():
    code = inspect.getsource(train_gnn.atom_features)
    code += repr(train_gnn._ELEMENTS) + repr([str(h) for h in train_gnn._HYB])
    return {
        "format_version": FORMAT_VERSION,
        "node_feature_source": "train_gnn.atom_features(ring_size_features=False)",
        "node_feature_source_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "bond_featurizer_source_sha256": hashlib.sha256(inspect.getsource(molecule_arrays).encode()).hexdigest(),
        "node_features": NODE_DIM, "edge_features": EDGE_DIM,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "edge_storage": "directed bond pairs with shard-global node indices; localize using node_ptr",
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "labels": "none; chemical graphs do not have an intrinsic product-falsification label",
        "stereochemistry": "Isomeric source SMILES preserved; existing22 node features and7 bond features do not encode full stereo.",
    }


def catalogue_fingerprint(connection, limit=None, include_ineligible=False):
    columns = {row[1] for row in connection.execute("PRAGMA table_info(molecules)")}
    required = {"molecule_id", "smiles", "training_eligible"}
    if not required <= columns:
        raise ValueError(f"Catalogue lacks columns {sorted(required - columns)}")
    all_digest, selected_digest = hashlib.sha256(), hashlib.sha256()
    total = eligible = selected = 0
    for row in connection.execute("SELECT molecule_id,smiles,training_eligible FROM molecules ORDER BY molecule_id"):
        if int(row["training_eligible"]) not in {0, 1}:
            raise ValueError(f"Invalid training_eligible flag for {row['molecule_id']}")
        payload = _fingerprint_row(row)
        all_digest.update(payload)
        total += 1
        eligible += int(row["training_eligible"])
        if (include_ineligible or row["training_eligible"]) and (limit is None or selected < limit):
            selected_digest.update(payload)
            selected += 1
    metadata_table = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()
    normalization = None
    if metadata_table:
        row = connection.execute("SELECT value FROM metadata WHERE key='normalization_version'").fetchone()
        normalization = row[0] if row else None
    return {"catalogue_molecules": total, "catalogue_training_eligible": eligible,
            "catalogue_sha256": all_digest.hexdigest(), "selected_molecules": selected,
            "selected_sha256": selected_digest.hexdigest(), "normalization_version": normalization}


def _selected_rows(connection, limit, include_ineligible):
    query = "SELECT molecule_id,smiles,training_eligible FROM molecules"
    if not include_ineligible:
        query += " WHERE training_eligible=1"
    query += " ORDER BY molecule_id"
    if limit is not None:
        query += " LIMIT ?"
        return connection.execute(query, (limit,))
    return connection.execute(query)


def _create_index(out, source_connection, pinned):
    path = out / "graph_index.sqlite"
    temporary = out / "graph_index.sqlite.tmp"
    if temporary.exists():
        temporary.unlink()
    index = sqlite3.connect(temporary)
    try:
        index.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE items(dataset_index INTEGER PRIMARY KEY,molecule_id TEXT UNIQUE NOT NULL,
          smiles TEXT NOT NULL,training_eligible INTEGER NOT NULL,shard_no INTEGER NOT NULL,
          graph_index INTEGER NOT NULL,graph_file TEXT NOT NULL,num_nodes INTEGER,num_edges INTEGER,
          state TEXT NOT NULL DEFAULT 'pending');
        CREATE INDEX item_shard ON items(shard_no,graph_index);
        CREATE TABLE shards(shard_no INTEGER PRIMARY KEY,graph_file TEXT UNIQUE NOT NULL,
          graph_count INTEGER NOT NULL,node_count INTEGER NOT NULL,edge_count INTEGER NOT NULL,
          sha256 TEXT NOT NULL,bytes INTEGER NOT NULL);
        """)
        index.execute("INSERT INTO metadata VALUES('pinned',?)", (json.dumps(pinned, sort_keys=True),))
        index.execute("INSERT INTO metadata VALUES('state','in_progress')")
        bucket, count = [], 0
        for count, row in enumerate(_selected_rows(source_connection, pinned["limit"], pinned["include_ineligible"]), 1):
            item_index = count - 1
            shard_no, graph_index = divmod(item_index, pinned["shard_size"])
            name = f"shard_{shard_no:06d}.npz"
            bucket.append((item_index, row["molecule_id"], row["smiles"], int(row["training_eligible"]),
                           shard_no, graph_index, name))
            if len(bucket) >= 2048:
                index.executemany("INSERT INTO items(dataset_index,molecule_id,smiles,training_eligible,shard_no,graph_index,graph_file) VALUES(?,?,?,?,?,?,?)", bucket)
                bucket = []
        if bucket:
            index.executemany("INSERT INTO items(dataset_index,molecule_id,smiles,training_eligible,shard_no,graph_index,graph_file) VALUES(?,?,?,?,?,?,?)", bucket)
        if count != pinned["snapshot"]["selected_molecules"]:
            raise ValueError("Selected catalogue count changed while its snapshot index was built")
        index.commit()
    finally:
        index.close()
    os.replace(temporary, path)


def molecule_arrays(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError(f"Invalid catalogue molecule: {smiles}")
    x = np.asarray([train_gnn.atom_features(atom) for atom in molecule.GetAtoms()], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != NODE_DIM:
        raise ValueError("Existing node featurizer changed dimensionality; use a new format/version")
    edges, attrs = [], []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        features = [float(bond_type == known) for known in BOND_TYPES]
        features += [float(bond_type not in BOND_TYPES), float(bond.GetIsConjugated()), float(bond.IsInRing())]
        edges.extend(((begin, end), (end, begin)))
        attrs.extend((features, features))
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float32) if attrs else np.empty((0, EDGE_DIM), dtype=np.float32)
    return x, edge_index, edge_attr


def _pack_shard(rows):
    xs, edges, attrs = [], [], []
    node_ptr, edge_ptr = [0], [0]
    for row in rows:
        try:
            x, edge_index, edge_attr = molecule_arrays(row["smiles"])
        except Exception as error:
            raise ValueError(f"Cannot prepare {row['molecule_id']}: {error}") from error
        xs.append(x)
        edges.append(edge_index + node_ptr[-1])
        attrs.append(edge_attr)
        node_ptr.append(node_ptr[-1] + x.shape[0])
        edge_ptr.append(edge_ptr[-1] + edge_index.shape[1])
    return {
        "format_version": np.asarray([FORMAT_VERSION], dtype=np.int16),
        "x": np.concatenate(xs, axis=0), "edge_index": np.concatenate(edges, axis=1),
        "edge_attr": np.concatenate(attrs, axis=0),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64), "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "molecule_ids": np.asarray([row["molecule_id"] for row in rows]),
        "smiles": np.asarray([row["smiles"] for row in rows]),
    }


def _atomic_npz(path, arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _export_index_csv(index, out):
    output = out / "graph_index.csv"
    temporary = out / "graph_index.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in index.execute("SELECT dataset_index,molecule_id,graph_file,graph_index,num_nodes,num_edges FROM items WHERE state='ready' ORDER BY dataset_index"):
            writer.writerow(dict(row))
    os.replace(temporary, output)
    return file_sha256(output)


def prepare(db, out, shard_size=2048, limit=None, include_ineligible=False, max_shards=None):
    """Prepare or resume shards, refusing a changed catalogue or feature setup.

    ``max_shards`` is a bounded-run option useful for staged preprocessing;
    the dataset remains incomplete until every selected shard is committed.
    """
    db, out = Path(db).resolve(), Path(out)
    if shard_size < 1 or (limit is not None and limit < 1) or (max_shards is not None and max_shards < 1):
        raise ValueError("shard_size, optional limit and optional max_shards must be positive")
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "graph_index.sqlite"
    if not index_path.exists() and any(p.name not in {"graph_index.sqlite.tmp"} for p in out.iterdir()):
        raise ValueError("Output is nonempty without a known graph index; use a new output directory")
    with closing(_read_only(db)) as source:
        source.execute("BEGIN")
        snapshot = catalogue_fingerprint(source, limit, include_ineligible)
        if not snapshot["selected_molecules"]:
            raise ValueError("No selected catalogue molecules to prepare")
        pinned = {"source_db": str(db), "snapshot": snapshot, "featurizer": featurizer_spec(),
                  "shard_size": shard_size, "limit": limit, "include_ineligible": include_ineligible}
        if not index_path.exists():
            _create_index(out, source, pinned)
        else:
            with closing(_read_only(index_path)) as prior_index:
                previous = json.loads(prior_index.execute("SELECT value FROM metadata WHERE key='pinned'").fetchone()[0])
            # Copying the same frozen DB to a new pathname is permitted. Its
            # content, eligibility, normalization and selection must match.
            comparable = {key: value for key, value in pinned.items() if key != "source_db"}
            previous_comparable = {key: value for key, value in previous.items() if key != "source_db"}
            if comparable != previous_comparable:
                raise ValueError("Catalogue snapshot or featurizer/config changed; refusing mixed-snapshot resume. Use a new output directory.")
            pinned = previous
        source.rollback()

    index = sqlite3.connect(index_path, timeout=120)
    index.row_factory = sqlite3.Row
    prepared_this_call = 0
    try:
        # Recheck committed shard digests once on resume. An interrupted write
        # before its index transaction can safely be overwritten deterministically.
        for shard in index.execute("SELECT * FROM shards ORDER BY shard_no"):
            path = out / shard["graph_file"]
            if not path.exists() or file_sha256(path) != shard["sha256"]:
                raise ValueError(f"Committed shard is missing or changed: {path}")
        shard_numbers = [row[0] for row in index.execute("SELECT DISTINCT shard_no FROM items WHERE state='pending' ORDER BY shard_no")]
        for shard_no in shard_numbers:
            rows = list(index.execute("SELECT * FROM items WHERE shard_no=? ORDER BY graph_index", (shard_no,)))
            if any(row["state"] != "pending" for row in rows):
                raise ValueError("A shard has mixed committed/pending items; index is inconsistent")
            arrays = _pack_shard(rows)
            path = out / rows[0]["graph_file"]
            _atomic_npz(path, arrays)
            checksum = file_sha256(path)
            with index:
                index.execute("INSERT INTO shards VALUES(?,?,?,?,?,?,?)", (shard_no, path.name, len(rows),
                    int(arrays["node_ptr"][-1]), int(arrays["edge_ptr"][-1]), checksum, path.stat().st_size))
                index.executemany("UPDATE items SET state='ready',num_nodes=?,num_edges=? WHERE dataset_index=?",
                    [(int(arrays["node_ptr"][i+1] - arrays["node_ptr"][i]),
                      int(arrays["edge_ptr"][i+1] - arrays["edge_ptr"][i]), row["dataset_index"])
                     for i, row in enumerate(rows)])
            prepared_this_call += 1
            complete_count = index.execute("SELECT COUNT(*) FROM items WHERE state='ready'").fetchone()[0]
            print(f"Prepared {complete_count:,}/{snapshot['selected_molecules']:,} graphs; {path.name}", flush=True)
            if max_shards is not None and prepared_this_call >= max_shards:
                break
        # This catches writers changing even unselected molecules or holdout
        # flags while preparation was in progress; no complete manifest follows.
        with closing(_read_only(db)) as current:
            current.execute("BEGIN")
            current_snapshot = catalogue_fingerprint(current, limit, include_ineligible)
            current.rollback()
        if current_snapshot != snapshot:
            raise ValueError("Source catalogue changed during preparation; pinned graph output remains incomplete")
        pending_count = index.execute("SELECT COUNT(*) FROM items WHERE state!='ready'").fetchone()[0]
        state = "complete" if pending_count == 0 else "in_progress"
        with index:
            index.execute("UPDATE metadata SET value=? WHERE key='state'", (state,))
        index_csv_sha = _export_index_csv(index, out)
        shards = [dict(row) for row in index.execute("SELECT * FROM shards ORDER BY shard_no")]
        manifest = {**pinned, "state": state, "written_utc": datetime.now(timezone.utc).isoformat(),
                    "prepared_graphs": snapshot["selected_molecules"] - pending_count,
                    "pending_graphs": pending_count, "prepared_shards": len(shards),
                    "prepared_this_call": prepared_this_call, "graph_index_csv_sha256": index_csv_sha,
                    "graph_index_sqlite_sha256": file_sha256(index_path),
                    "shards": shards,
                    "loader": "BankGraphDataset + BankShardBatchSampler; at most1–2 decompressed shards per worker by default"}
        _atomic_json(out / "manifest.json", manifest)
        (out / "README.md").write_text(f"""# Unlabelled 2D molecular graph shards

Source snapshot: `{snapshot['catalogue_sha256']}` ({snapshot['catalogue_molecules']:,} catalogue molecules).
Selection: {snapshot['selected_molecules']:,} molecules; training-ineligible rows are {'included explicitly' if include_ineligible else 'excluded'}.
State: `{state}`. Node features: 22 values from `train_gnn.atom_features`; bond attributes: single/double/triple/aromatic/other, conjugation and ring membership (7 values). Edges are stored globally within a shard and localized on loading. Source SMILES preserve explicit stereochemistry, but these features do not encode complete stereo.

There is no molecule-level counterfeit/authenticity label. These are chemical inputs for pretraining or for joining to an explicitly labelled claim/observation task. The index pins source structures and eligibility as of the saved snapshot. Resume rejects changes in catalogue content, selection or featurizer. Descriptors/job metadata unrelated to the molecule graph do not enter the snapshot digest.

Use the shard-aware sampler to avoid decompressing a different shard for every random sample:

```python
from torch_geometric.loader import DataLoader
from bank_graph_shards import BankGraphDataset, BankShardBatchSampler

dataset = BankGraphDataset({str(out)!r}, cache_shards=2)
sampler = BankShardBatchSampler(dataset, batch_size=64, shuffle=True, seed=42)
loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
for epoch in range(10):
    sampler.set_epoch(epoch)
    for batch in loader:
        # batch.x, batch.edge_index, batch.edge_attr; no intrinsic batch.y
        pass
```

Only the requested shard is loaded; each worker maintains its own LRU cache. Keep the pinned output immutable while training. A changed source catalogue requires a new output directory for preparation.
""")
        return manifest
    finally:
        index.close()


class BankGraphDataset(Dataset):
    """Map-style PyG graph access backed by a pinned SQLite/NPZ snapshot."""

    def __init__(self, root, cache_shards=2, verify_shards=True):
        self.root = Path(root)
        if cache_shards not in {1, 2}:
            raise ValueError("cache_shards must be1 or2 to bound worker memory")
        self.cache_shards = cache_shards
        self.verify_shards = verify_shards
        self._connection = None
        self._pid = None
        self._cache = OrderedDict()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["state"] != "complete":
            raise ValueError("Graph preparation is incomplete; resume prepare before training")
        if file_sha256(self.root / "graph_index.sqlite") != self.manifest["graph_index_sqlite_sha256"]:
            raise ValueError("Pinned graph SQLite index changed")
        self._length = int(self.manifest["snapshot"]["selected_molecules"])
        self._digests = {shard["graph_file"]: shard["sha256"] for shard in self.manifest["shards"]}

    def __len__(self):
        return self._length

    def _index(self):
        pid = os.getpid()
        if self._connection is None or self._pid != pid:
            if self._connection is not None:
                self._connection.close()
            self._connection = _read_only(self.root / "graph_index.sqlite")
            self._pid = pid
            self._cache = OrderedDict()
        return self._connection

    def _shard(self, name):
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        path = self.root / name
        if self.verify_shards and file_sha256(path) != self._digests[name]:
            raise ValueError(f"Graph shard changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        self._cache[name] = arrays
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            item = int(item.item())
        if not isinstance(item, int):
            raise TypeError("Dataset index must be an integer")
        if item < 0:
            item += self._length
        if not 0 <= item < self._length:
            raise IndexError(item)
        row = self._index().execute("SELECT * FROM items WHERE dataset_index=?", (item,)).fetchone()
        if row is None or row["state"] != "ready":
            raise ValueError(f"Missing ready graph index row {item}")
        arrays = self._shard(row["graph_file"])
        local = row["graph_index"]
        node_begin, node_end = map(int, arrays["node_ptr"][local:local+2])
        edge_begin, edge_end = map(int, arrays["edge_ptr"][local:local+2])
        if str(arrays["molecule_ids"][local]) != row["molecule_id"]:
            raise ValueError("Shard and SQLite molecular identities disagree")
        return Data(x=torch.from_numpy(arrays["x"][node_begin:node_end].copy()),
                    edge_index=torch.from_numpy((arrays["edge_index"][:, edge_begin:edge_end] - node_begin).copy()),
                    edge_attr=torch.from_numpy(arrays["edge_attr"][edge_begin:edge_end].copy()),
                    molecule_id=row["molecule_id"], smiles=row["smiles"],
                    catalogue_row_id=torch.tensor([item], dtype=torch.long))

    def shard_ranges(self):
        return [(row["shard_no"], row["first_index"], row["last_index"] + 1)
                for row in self._index().execute("SELECT shard_no,MIN(dataset_index) AS first_index,MAX(dataset_index) AS last_index FROM items GROUP BY shard_no ORDER BY shard_no")]

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_connection=None, _pid=None, _cache=OrderedDict())
        return state

    def close(self):
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._cache.clear()


class BankShardBatchSampler(Sampler):
    """Shuffle shard order and indices within one shard, using bounded memory."""

    def __init__(self, dataset, batch_size=64, shuffle=True, seed=42, drop_last=False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.ranges = dataset.shard_ranges()
        self.batch_size, self.shuffle, self.seed, self.drop_last = batch_size, shuffle, seed, drop_last
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return sum((end - begin) // self.batch_size if self.drop_last else math.ceil((end - begin) / self.batch_size)
                   for _, begin, end in self.ranges)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        ranges = list(self.ranges)
        if self.shuffle:
            rng.shuffle(ranges)
        for _, begin, end in ranges:
            indexes = list(range(begin, end))
            if self.shuffle:
                rng.shuffle(indexes)
            for offset in range(0, len(indexes), self.batch_size):
                batch = indexes[offset:offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--db", type=Path, default=Path("molecule_bank_1m/bank.sqlite"))
    command.add_argument("--out", type=Path, default=Path("molecule_bank_1m/graphs_2d"))
    command.add_argument("--shard-size", type=int, default=2048)
    command.add_argument("--limit", type=int)
    command.add_argument("--include-ineligible", action="store_true")
    command.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    result = prepare(args.db, args.out, args.shard_size, args.limit, args.include_ineligible, args.max_shards)
    print(json.dumps({"out": str(args.out), "state": result["state"],
                      "selected_graphs": result["snapshot"]["selected_molecules"],
                      "prepared_graphs": result["prepared_graphs"], "prepared_shards": result["prepared_shards"]}, indent=2))


if __name__ == "__main__":
    main()
