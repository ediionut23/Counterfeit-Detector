"""Persistent, resumable chemical catalogue and claim/observation relation store.

Chemical identities and product observations are separate. Neither a generated
origin nor a reference origin intrinsically labels a molecule as falsified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

NORMALIZATION_VERSION = "active_moiety_cleanup_fragment_uncharger_stereo_v1"
_BLOCKED_IDENTITIES = set()
_BLOCKED_SCAFFOLDS = set()


def stable_hash(text, length=24):
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chemistry_init(holdout=None):
    global _BLOCKED_IDENTITIES, _BLOCKED_SCAFFOLDS
    RDLogger.DisableLog("rdApp.*")
    _BLOCKED_IDENTITIES = set((holdout or {}).get("blocked_connectivity_smiles", []))
    _BLOCKED_SCAFFOLDS = set((holdout or {}).get("blocked_scaffold_keys", []))


@lru_cache(maxsize=15000)
def normalize_identity(raw):
    mol = Chem.MolFromSmiles(raw)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("invalid_smiles")
    raw_canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol, skipStandardize=True)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, isomericSmiles=False)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False) or "ACYCLIC:" + connectivity
    return raw_canonical, smiles, connectivity, scaffold


def numeric(value):
    try:
        parsed=float(value) if value not in (None, "", "None") else None
        return parsed if parsed is not None and math.isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=15000)
def basic_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return {"mw": float(Descriptors.MolWt(molecule)), "logp": float(Descriptors.MolLogP(molecule)),
            "qed": float(Descriptors.qed(molecule)), "heavy_atoms": molecule.GetNumHeavyAtoms()}


def prepare_catalogue_row(row):
    raw = row.get("counterfeit_smiles") or row.get("smiles") or ""
    try:
        raw_canonical, smiles, connectivity, scaffold = normalize_identity(raw)
        # Existing audited descriptors can be reused only when normalization
        # leaves the molecular structure unchanged. Missing values are computed.
        props = {key: numeric(row.get(key)) for key in ("mw", "logp", "qed", "sa_score", "heavy_atoms")}
        if smiles != raw_canonical or any(props[key] is None for key in ("mw", "logp", "qed", "heavy_atoms")):
            props.update(basic_properties(smiles))
        if smiles != raw_canonical:
            props["sa_score"]=None  # original pharmaceutical form's SA is not the normalized moiety's SA
        generated = row.get("label", "1") != "0"
        method = row.get("generation_method") or row.get("method") or ""
        methods = sorted({letter for letter in method if letter in {"M", "E"}}) if generated else []
        return {"molecule_id": "mol_" + stable_hash(smiles), "smiles": smiles,
                "raw_smiles": raw, "raw_canonical_smiles": raw_canonical,
                "connectivity_smiles": connectivity, "scaffold_key": scaffold, **props,
                "is_generated": int(generated), "is_reference": int(not generated),
                "methods": methods, "source_id": row.get("molecule_id") or row.get("source_id", ""),
                "source_type": row.get("molecule_source") or row.get("generation_protocol") or "legacy_catalogue",
                "training_eligible": int(connectivity not in _BLOCKED_IDENTITIES and scaffold not in _BLOCKED_SCAFFOLDS),
                "source_metadata": {k: row.get(k, "") for k in ("category", "subcategory", "parent_smiles", "generator_variant")}}
    except Exception as error:
        return {"error": str(error), "raw_smiles": raw, "source_id": row.get("molecule_id", "")}


def prepare_chunk(rows):
    return [prepare_catalogue_row(row) for row in rows]


class BankStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=120)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA foreign_keys=ON;
        PRAGMA cache_size=-65536;
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS molecules(
          molecule_id TEXT PRIMARY KEY,smiles TEXT NOT NULL UNIQUE,connectivity_smiles TEXT NOT NULL,
          scaffold_key TEXT NOT NULL,mw REAL,logp REAL,qed REAL,sa_score REAL,heavy_atoms INTEGER,
          is_generated INTEGER NOT NULL,is_reference INTEGER NOT NULL,training_eligible INTEGER NOT NULL,
          first_source TEXT NOT NULL,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE INDEX IF NOT EXISTS molecule_scaffold ON molecules(scaffold_key);
        CREATE TABLE IF NOT EXISTS molecule_sources(
          molecule_id TEXT NOT NULL REFERENCES molecules(molecule_id),source_path TEXT NOT NULL,
          source_id TEXT NOT NULL,raw_smiles TEXT NOT NULL,source_type TEXT NOT NULL,metadata_json TEXT,
          PRIMARY KEY(molecule_id,source_path,source_id,raw_smiles));
        CREATE INDEX IF NOT EXISTS source_raw_smiles ON molecule_sources(raw_smiles);
        CREATE TABLE IF NOT EXISTS molecule_methods(molecule_id TEXT REFERENCES molecules(molecule_id),
          method TEXT NOT NULL,PRIMARY KEY(molecule_id,method));
        CREATE TABLE IF NOT EXISTS descriptor_cache(smiles TEXT PRIMARY KEY,normalization_version TEXT,
          rdkit_version TEXT,properties_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS references_catalogue(reference_id TEXT PRIMARY KEY,name TEXT,smiles TEXT,
          molecule_id TEXT REFERENCES molecules(molecule_id),source TEXT,source_id TEXT,status TEXT,
          generation_eligible INTEGER,metadata_json TEXT);
        CREATE TABLE IF NOT EXISTS relations(
          relation_id TEXT PRIMARY KEY,molecule_id TEXT NOT NULL REFERENCES molecules(molecule_id),
          reference_id TEXT REFERENCES references_catalogue(reference_id),parent_smiles TEXT NOT NULL,
          method TEXT NOT NULL,category TEXT,subcategory TEXT,scenario TEXT,difficulty TEXT,
          protocol TEXT,operator_history_json TEXT,verification_status TEXT,evidence_tier TEXT NOT NULL,
          similarity REAL,objective_json TEXT,source_id TEXT,job_id TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE INDEX IF NOT EXISTS relation_coverage ON relations(method,category,difficulty);
        CREATE INDEX IF NOT EXISTS relation_reference ON relations(reference_id);
        CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY,method TEXT,reference_id TEXT,
          category TEXT,difficulty TEXT,round INTEGER,seed INTEGER,config_json TEXT,
          state TEXT DEFAULT 'pending',attempts INTEGER DEFAULT 0,produced INTEGER DEFAULT 0,
          new_molecules INTEGER DEFAULT 0,relations_added INTEGER DEFAULT 0,rejections_json TEXT,
          started_at TEXT,finished_at TEXT,error TEXT);
        CREATE TABLE IF NOT EXISTS import_sources(source_id TEXT PRIMARY KEY,path TEXT,sha256 TEXT,
          state TEXT,rows_processed INTEGER DEFAULT 0,new_molecules INTEGER DEFAULT 0,
          errors INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS import_chunks(source_id TEXT,chunk_no INTEGER,rows_processed INTEGER,
          PRIMARY KEY(source_id,chunk_no));
        CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY,declared_reference_id TEXT,
          observed_components_json TEXT,quantities_json TEXT,scenario TEXT,evidence_tier TEXT,
          composition_mismatch INTEGER,metadata_json TEXT);
        CREATE TABLE IF NOT EXISTS events(event_id INTEGER PRIMARY KEY,kind TEXT,payload_json TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)
        version = self.connection.execute("SELECT value FROM metadata WHERE key='normalization_version'").fetchone()
        if version and version[0] != NORMALIZATION_VERSION:
            raise ValueError("Cannot resume a bank with a different identity-normalization policy")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('normalization_version',?)", (NORMALIZATION_VERSION,))
        self.connection.commit()

    def count(self):
        return self.connection.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]

    def insert_molecule(self, row, source_path):
        before = self.connection.total_changes
        self.connection.execute("""INSERT OR IGNORE INTO molecules
          (molecule_id,smiles,connectivity_smiles,scaffold_key,mw,logp,qed,sa_score,heavy_atoms,
           is_generated,is_reference,training_eligible,first_source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (row["molecule_id"],row["smiles"],row["connectivity_smiles"],row["scaffold_key"],
           row.get("mw"),row.get("logp"),row.get("qed"),row.get("sa_score"),row.get("heavy_atoms"),
           row["is_generated"],row["is_reference"],row["training_eligible"],source_path))
        is_new = self.connection.total_changes > before
        prior = self.connection.execute("SELECT smiles FROM molecules WHERE molecule_id=?", (row["molecule_id"],)).fetchone()
        if not prior or prior[0] != row["smiles"]:
            raise ValueError("Molecular ID collision or inconsistent identity")
        if not is_new:
            self.connection.execute("""UPDATE molecules SET
              is_generated=MAX(is_generated,?),is_reference=MAX(is_reference,?),
              training_eligible=MIN(training_eligible,?),sa_score=COALESCE(sa_score,?) WHERE molecule_id=?""",
              (row["is_generated"],row["is_reference"],row["training_eligible"],row.get("sa_score"),row["molecule_id"]))
        self.connection.execute("INSERT OR IGNORE INTO molecule_sources VALUES(?,?,?,?,?,?)",
            (row["molecule_id"],source_path,row.get("source_id", ""),row.get("raw_smiles",row["smiles"]),
             row.get("source_type","generated"),json.dumps(row.get("source_metadata",{}),sort_keys=True)))
        self.connection.executemany("INSERT OR IGNORE INTO molecule_methods VALUES(?,?)",
            [(row["molecule_id"],method) for method in row.get("methods",[])])
        self.connection.execute("INSERT OR IGNORE INTO descriptor_cache VALUES(?,?,?,?)",
            (row["smiles"],NORMALIZATION_VERSION,Chem.rdBase.rdkitVersion,
             json.dumps({key: row.get(key) for key in ("mw","logp","qed","sa_score","heavy_atoms","scaffold_key")},sort_keys=True)))
        return int(is_new)

    def import_catalogue(self, path, workers=4, chunk_size=1000, holdout=None, method_override=None):
        path = Path(path)
        checksum = file_sha256(path)
        source_id = stable_hash(str(path.resolve()) + "|" + checksum)
        existing = self.connection.execute("SELECT * FROM import_sources WHERE path=?", (str(path),)).fetchone()
        if existing and existing["sha256"] != checksum:
            raise ValueError(f"Imported source changed; use an explicit new source version: {path}")
        if existing and existing["state"] == "complete":
            print(f"Import cached: {path}",flush=True)
            return
        self.connection.execute("INSERT OR IGNORE INTO import_sources(source_id,path,sha256,state) VALUES(?,?,?,'running')",
                                (source_id,str(path),checksum))
        self.connection.commit()
        completed = {r[0] for r in self.connection.execute("SELECT chunk_no FROM import_chunks WHERE source_id=?",(source_id,))}
        errors_path = self.path.parent / "import_errors.jsonl"

        def chunks():
            with path.open(newline="") as stream:
                bucket, chunk_no = [], 0
                for row in csv.DictReader(stream):
                    if method_override:
                        row.update(label="1",generation_method=method_override)
                    bucket.append(row)
                    if len(bucket) == chunk_size:
                        if chunk_no not in completed:
                            yield chunk_no,bucket
                        chunk_no += 1
                        bucket = []
                if bucket and chunk_no not in completed:
                    yield chunk_no,bucket

        iterator = iter(chunks())
        with ProcessPoolExecutor(max_workers=workers,initializer=chemistry_init,initargs=(holdout,)) as pool:
            pending = {}
            def refill():
                while len(pending) < workers * 2:
                    try:
                        chunk_no, rows = next(iterator)
                    except StopIteration:
                        return
                    pending[pool.submit(prepare_chunk,rows)] = (chunk_no,len(rows))
            refill()
            finished = 0
            while pending:
                ready,_ = wait(pending,return_when=FIRST_COMPLETED)
                for future in ready:
                    chunk_no,n_rows = pending.pop(future)
                    prepared = future.result()
                    added,failures = 0,[]
                    with self.connection:
                        for row in prepared:
                            if "error" in row:
                                failures.append(row)
                            else:
                                added += self.insert_molecule(row,str(path))
                        self.connection.execute("INSERT INTO import_chunks VALUES(?,?,?)", (source_id,chunk_no,n_rows))
                        self.connection.execute("UPDATE import_sources SET rows_processed=rows_processed+?,new_molecules=new_molecules+?,errors=errors+? WHERE source_id=?",
                                                (n_rows,added,len(failures),source_id))
                    if failures:
                        with errors_path.open("a") as stream:
                            for row in failures:
                                stream.write(json.dumps({"source":str(path),**row})+"\n")
                    finished += 1
                    if finished % 10 == 0:
                        progress = self.connection.execute("SELECT rows_processed,new_molecules FROM import_sources WHERE source_id=?",(source_id,)).fetchone()
                        print(f"Import {path.name}: rows={progress[0]:,} new={progress[1]:,} bank={self.count():,}",flush=True)
                refill()
        self.connection.execute("UPDATE import_sources SET state='complete' WHERE source_id=?",(source_id,))
        self.connection.commit()
        print(f"Import complete {path}: bank={self.count():,}",flush=True)

    def summary(self):
        total = self.connection.execute("""SELECT COUNT(*),SUM(is_generated),SUM(is_reference),SUM(training_eligible),
                 SUM(is_generated AND is_reference) FROM molecules""").fetchone()
        return {"unique_molecules":total[0],"generated_origin_molecules":total[1],
                "public_reference_origin_molecules":total[2],"training_eligible_molecules":total[3],
                "molecules_with_both_origins":total[4],
                "method_origin_counts":dict(self.connection.execute("SELECT method,COUNT(*) FROM molecule_methods GROUP BY method")),
                "relations":self.connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
                "named_references":self.connection.execute("SELECT COUNT(*) FROM references_catalogue WHERE name<>''").fetchone()[0],
                "normalization_version":NORMALIZATION_VERSION}
