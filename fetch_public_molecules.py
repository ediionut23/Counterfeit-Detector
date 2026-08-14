"""
Fetch a large real-molecule set from public databases, for MMP rule mining
==========================================================================

The mined transformation rules are only as good as the molecules they are mined
from. The first catalogue was mined from ~7.7k molecules, giving a median
support of just 4 matched pairs per rule. Mining from a much larger, analog-rich
public set (ChEMBL, PubChem) yields more rules with real support — transformations
that many real drug pairs actually exhibit, which is the empirical grounding we
want for "realistic" counterfeit edits.

Key design point for MMP mining: we deliberately keep MANY molecules per scaffold
(analog series), because a Matched Molecular Pair only exists between two
molecules that share a constant part. Over-diversifying (one molecule per
scaffold) would destroy the very pairs we mine.

Reuses the project's proven AuthenticFetcher (ChEMBL REST + PubChem), applies its
drug-likeness filter, deduplicates on canonical SMILES, and writes a plain .smi
ready for:  python mine_mmp_rules.py --smiles-file <out> --min-support 8

Usage
-----
    python fetch_public_molecules.py --target 30000 --out public_molecules.smi
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=30000,
                    help="Approx. number of molecules to fetch")
    ap.add_argument("--per-scaffold", type=int, default=1000,
                    help="Max molecules per scaffold (HIGH on purpose, to keep "
                         "analog series so MMP pairs exist)")
    ap.add_argument("--sources", default="chembl,pubchem",
                    help="Comma list: chembl,pubchem")
    ap.add_argument("--out", default="public_molecules.smi")
    args = ap.parse_args()

    # Reuse the tested fetcher from the existing pipeline.
    from intelligent_dataset_generator_v3 import AuthenticFetcher

    fetcher = AuthenticFetcher(max_per_scaffold=args.per_scaffold)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    rows = []
    if "chembl" in sources:
        rows += fetcher.fetch_chembl_rest(int(args.target * 0.7))
    if "pubchem" in sources:
        rows += fetcher.fetch_pubchem(int(args.target * 0.5))

    # canonical dedup
    seen, out = set(), []
    for r in rows:
        smi = r.get("smiles")
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if c in seen:
            continue
        seen.add(c)
        out.append(c)

    path = Path(args.out)
    with open(path, "w") as fh:
        for i, smi in enumerate(out):
            fh.write(f"{smi}\tmol{i}\n")

    logger.info(f"Wrote {len(out)} unique molecules to {path}")
    logger.info(f"Fetcher stats: {dict(fetcher.stats)}")
    logger.info(f"Next:  python mine_mmp_rules.py --smiles-file {path} "
                f"--min-support 8 --out mined_transformations_big")


if __name__ == "__main__":
    main()
