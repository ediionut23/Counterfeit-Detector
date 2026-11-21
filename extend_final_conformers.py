"""Embed final-bank molecules absent from the existing benchmark 3D shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from generate_conformers import embed_one


def smiles_in_sdf_gz(path: Path):
    """Yield tags from an already audited RDKit SDF shard without parsing 3D."""
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith(">  <smiles>"):
                yield next(stream).strip()


def wanted_smiles(root: Path) -> set[str]:
    wanted = set()
    for name in ("molecules.csv", "authentic_eligible.csv"):
        with (root / name).open(newline="") as stream:
            wanted.update(row["smiles"] for row in csv.DictReader(stream))
    return wanted


def embed_retry(smiles: str):
    """Use independent distance-geometry seeds for difficult fused/ring systems."""
    for method, seed in (("ETDG", 104729), ("ETDG", 2027),
                         ("ETKDGv3", 43), ("ETKDGv3", 101),
                         ("ETKDGv3", 2026)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3() if method == "ETKDGv3" else AllChem.ETDG()
        params.randomSeed = seed
        params.useRandomCoords = True
        params.useSmallRingTorsions = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
        except Exception:
            pass
        return smiles, Chem.MolToMolBlock(mol)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    parser.add_argument("--base", type=Path, default=Path("benchmark"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retry-only", action="store_true",
                        help="Use the robust fallback directly for the currently missing structures")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    wanted = wanted_smiles(args.bank)
    base_shards = sorted(args.base.glob("conformers_part_*.sdf.gz"))
    if not base_shards:
        raise FileNotFoundError(f"No base conformer shards in {args.base}")
    covered = set()
    duplicate_tags = 0
    for path in base_shards:
        print(f"Indexing {path}", flush=True)
        for smiles in smiles_in_sdf_gz(path):
            if smiles in covered:
                duplicate_tags += 1
            covered.add(smiles)
    old_covered = len(wanted & covered)
    extra_shards = sorted(args.bank.glob("conformers_extra*.sdf.gz"))
    for extra in extra_shards:
        for smiles in smiles_in_sdf_gz(extra):
            if smiles in covered:
                duplicate_tags += 1
            covered.add(smiles)
    pending = sorted(wanted - covered)
    print(f"Wanted={len(wanted):,}; base coverage={old_covered:,}; pending={len(pending):,}", flush=True)
    if pending:
        extra = (args.bank / "conformers_extra.sdf.gz" if not extra_shards else
                 args.bank / f"conformers_extra_{len(extra_shards):03d}.sdf.gz")
        if extra.exists():
            raise FileExistsError(extra)
        partial = args.bank / "conformers_extra.partial.sdf"
        with partial.open("w") as stream:
            writer = Chem.SDWriter(stream)
            try:
                with mp.Pool(args.workers) as pool:
                    task = embed_retry if args.retry_only else embed_one
                    for i, result in enumerate(pool.imap_unordered(task, pending, chunksize=8), 1):
                        if result is not None:
                            smiles, block = result
                            mol = Chem.MolFromMolBlock(block, removeHs=False)
                            if mol is not None and mol.GetNumConformers() and mol.GetConformer().Is3D():
                                mol.SetProp("smiles", smiles)
                                writer.write(mol)
                                covered.add(smiles)
                        if i % 1000 == 0:
                            print(f"  {i:,}/{len(pending):,} processed; {len(wanted & covered):,} covered", flush=True)
                retry = sorted(wanted - covered)
                if retry and not args.retry_only:
                    print(f"Retrying {len(retry):,} failed embeddings with independent seeds", flush=True)
                    with mp.Pool(args.workers) as pool:
                        for result in pool.imap_unordered(embed_retry, retry, chunksize=1):
                            if result is None:
                                continue
                            smiles, block = result
                            mol = Chem.MolFromMolBlock(block, removeHs=False)
                            if mol is not None and mol.GetNumConformers() and mol.GetConformer().Is3D():
                                mol.SetProp("smiles", smiles)
                                writer.write(mol)
                                covered.add(smiles)
            finally:
                writer.close()
        with partial.open("rb") as source, gzip.open(extra, "wb", compresslevel=6) as target:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
        partial.unlink()
        extra_shards.append(extra)
    missing = sorted(wanted - covered)
    (args.bank / "conformers_missing.smi").write_text("".join(s + "\n" for s in missing))
    report = {
        "wanted_unique_smiles": len(wanted),
        "covered_by_existing_benchmark_shards": old_covered,
        "covered_total": len(wanted & covered),
        "missing": len(missing),
        "coverage": round(len(wanted & covered) / max(len(wanted), 1), 6),
        "duplicate_tags_across_shards": duplicate_tags,
        "base_shards": [str(path) for path in base_shards],
        "extra_shards": [str(path) for path in extra_shards],
        "method": "Existing ETKDGv3/MMFF94; new ETKDGv3 with ETDG fallback, MMFF94 attempted",
    }
    (args.bank / "conformers_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
