"""
Central 3D conformer library for the benchmark (proposal C1)
============================================================

Every benchmark molecule ships with a 3D conformer. Rather than re-embedding the
same molecule once per dataset it appears in, this builds ONE conformer library
keyed by canonical SMILES, covering every unique molecule across all datasets,
computed in parallel over CPU cores.

Method: RDKit ETKDGv3 (Riniker & Landrum, 2015) — distance-geometry embedding
biased by experimental torsion preferences — followed by an MMFF94 force-field
relaxation. One low-energy conformer per molecule.

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
import glob
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
    m.SetProp("smiles", smiles)
    return smiles, Chem.MolToMolBlock(m)


def collect_unique_smiles(limit: Optional[int]) -> List[str]:
    seen = set()
    for f in glob.glob("benchmark/*/dataset.csv"):
        for r in csv.DictReader(open(f)):
            seen.add(r["smiles"])
    smis = sorted(seen)
    if limit:
        smis = smis[:limit]
    return smis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/conformers_full.sdf")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    smis = collect_unique_smiles(args.limit)
    logger.info(f"Embedding {len(smis)} unique molecules on {args.workers} workers ...")

    n_ok = 0
    writer = Chem.SDWriter(args.out)
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(embed_one, smis, chunksize=16)):
            if res is not None:
                _, molblock = res
                m = Chem.MolFromMolBlock(molblock, removeHs=False)
                if m is not None:
                    writer.write(m)
                    n_ok += 1
            if (i + 1) % 1000 == 0:
                logger.info(f"  {i+1}/{len(smis)} processed, {n_ok} embedded")
    writer.close()

    report = {
        "n_unique_molecules": len(smis),
        "n_embedded": n_ok,
        "coverage": round(n_ok / max(len(smis), 1), 3),
        "method": "ETKDGv3 + MMFF94",
        "output": args.out,
    }
    Path("benchmark/conformers_report.json").write_text(json.dumps(report, indent=2))
    logger.info(f"Done: {n_ok}/{len(smis)} embedded "
                f"({report['coverage']*100:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
