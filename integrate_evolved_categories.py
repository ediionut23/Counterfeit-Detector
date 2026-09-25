"""
Integrate Category-Conditioned Evolved Datasets into Benchmark Suite
====================================================================
Assembles the newly generated category-conditioned evolved counterfeits into
the central `benchmark/` suite with:
  1. DUD-E property-matched authentic drugs (1:1 balance)
  2. Bemis-Murcko scaffold disjoint splits (80/10/10)
  3. Shortcut resistance validation (Cohen's d + trivial LR accuracy)
  4. Updated `benchmark/index.json`
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import assemble_36_benchmark as asm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("integrate_evolved_categories")


def load_counterfeits_csv(csv_path: Path, cat_tag: str) -> List[Dict]:
    rows = []
    seen = set()
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            smi = r["counterfeit_smiles"]
            if smi in seen:
                continue
            seen.add(smi)
            d = asm.descriptors(smi)
            if d is None:
                continue
            rows.append({
                "smiles": smi,
                "label": 1,
                "category": f"evolved_{cat_tag}",
                "subcategory": f"evolved_{cat_tag}",
                "parent_smiles": r.get("parent_smiles", ""),
                "edited_atoms": r.get("edited_atoms", ""),
                **d
            })
    return rows


def main():
    bench_dir = Path("benchmark")
    in_dir = Path("version_e_categories")
    auth_path = "public_molecules_100k.smi"
    if not Path(auth_path).exists():
        auth_path = "public_molecules.smi"

    logger.info("Loading authentic library...")
    auth_pool = asm.load_authentic_library(auth_path)

    # Load index.json
    idx_path = bench_dir / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}

    cat_files = {
        "evolved_bioisostere": in_dir / "counterfeits_evolved_bioisostere.csv",
        "evolved_halogen-walk": in_dir / "counterfeits_evolved_halogen-walk.csv",
        "evolved_homologation": in_dir / "counterfeits_evolved_homologation.csv",
        "evolved_scaffold-hop": in_dir / "counterfeits_evolved_scaffold-hop.csv",
        "evolved_other": in_dir / "counterfeits_evolved_other.csv",
    }

    all_evolved_fakes = []
    for name, cf_path in cat_files.items():
        if not cf_path.exists():
            logger.warning(f"File {cf_path} not found, skipping.")
            continue
        cat_tag = name.replace("evolved_", "")
        fakes = load_counterfeits_csv(cf_path, cat_tag)
        logger.info(f"Loaded {len(fakes)} unique fakes for {name}")
        all_evolved_fakes.extend(fakes)

        # Assemble individual evolved category slice
        summary = asm.assemble_slice(name, fakes, auth_pool, bench_dir, seed=42)
        if summary:
            index[name] = summary

    # Also assemble the combined evolved categories set
    if all_evolved_fakes:
        summary_all = asm.assemble_slice("evolved_category_conditioned_all", all_evolved_fakes, auth_pool, bench_dir, seed=42)
        if summary_all:
            index["evolved_category_conditioned_all"] = summary_all

    # Save updated index.json
    idx_path.write_text(json.dumps(index, indent=2))
    logger.info(f"Successfully updated {idx_path} with {len(index)} total benchmark datasets!")


if __name__ == "__main__":
    main()
