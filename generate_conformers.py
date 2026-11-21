"""
Central 3D conformer library for the benchmark (proposal C1)
============================================================

This builds one central conformer library keyed by canonical SMILES, attempting
every unique molecule across all datasets in parallel over CPU cores. Molecules
that cannot be embedded are reported as missing by audit_conformers.py.

Method: RDKit ETKDGv3 (Riniker & Landrum, 2015) — distance-geometry embedding
biased by experimental torsion preferences — followed by an MMFF94 force-field
relaxation when MMFF parameters are available. One conformer per molecule.

Output: benchmark/conformers_full.sdf (each record tagged with prop `smiles` =
canonical SMILES) plus a coverage report.

Usage
-----
    python generate_conformers.py                 # all unique molecules
    python generate_conformers.py --limit 2000    # quick subset
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def embed_one(smiles: str) -> Optional[Tuple[str, str]]:
    """Return (smiles, molblock-with-3D) or None if embedding failed."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(m, params) != 0:
        # retry with random coordinates as a fallback
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=400)
    except Exception:
        pass
    return smiles, Chem.MolToMolBlock(m)


def collect_unique_smiles(limit: Optional[int], dataset_root: Path = Path("benchmark")) -> List[str]:
    seen = set()
    for f in dataset_root.rglob("dataset.csv"):
        with f.open(newline="") as stream:
            for r in csv.DictReader(stream):
                seen.add(r["smiles"])
    smis = sorted(seen)
    if limit:
        smis = smis[:limit]
    return smis


def trim_incomplete_record(path: Path) -> None:
    """Keep only complete SDF records after an interrupted write."""
    with path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if not size:
            return
        window = min(size, 4 * 1024 * 1024)
        stream.seek(size - window)
        tail = stream.read(window)
        last = tail.rfind(b"$$$$\n")
        if last < 0:
            raise ValueError(f"No complete SDF record near the end of {path}")
        complete_size = size - window + last + len(b"$$$$\n")
        if complete_size != size:
            logger.warning(f"Trimming {size - complete_size} incomplete bytes from {path}")
            stream.truncate(complete_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/conformers_full.sdf")
    ap.add_argument("--dataset-root", type=Path, default=Path("benchmark"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Append missing molecules to a tagged partial SDF")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    smis = collect_unique_smiles(args.limit, args.dataset_root)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and output.exists():
        trim_incomplete_record(output)
        for mol in Chem.SDMolSupplier(str(output), removeHs=False):
            if mol is None or not mol.HasProp("smiles"):
                raise ValueError(f"Cannot resume from untagged or invalid SDF: {output}")
            completed.add(mol.GetProp("smiles"))
    pending = [s for s in smis if s not in completed]
    logger.info(f"Embedding {len(pending)} of {len(smis)} unique molecules "
                f"on {args.workers} workers ({len(completed)} already present) ...")

    n_ok = len(completed.intersection(smis))
    with output.open("a" if args.resume and output.exists() else "w") as stream:
        writer = Chem.SDWriter(stream)
        try:
            with mp.Pool(args.workers) as pool:
                for i, res in enumerate(pool.imap_unordered(embed_one, pending, chunksize=16)):
                    if res is not None:
                        smiles, molblock = res
                        m = Chem.MolFromMolBlock(molblock, removeHs=False)
                        if m is not None:
                            m.SetProp("smiles", smiles)
                            writer.write(m)
                            n_ok += 1
                    if (i + 1) % 1000 == 0:
                        logger.info(f"  {i+1}/{len(pending)} processed, {n_ok} embedded")
        finally:
            writer.close()

    report = {
        "n_unique_molecules": len(smis),
        "n_embedded": n_ok,
        "coverage": round(n_ok / max(len(smis), 1), 6),
        "method": "ETKDGv3; MMFF94 optimization attempted",
        "output": args.out,
    }
    (Path(args.out).parent / "conformers_report.json").write_text(
        json.dumps(report, indent=2))
    logger.info(f"Done: {n_ok}/{len(smis)} embedded "
                f"({report['coverage']*100:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
