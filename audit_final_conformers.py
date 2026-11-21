"""Read every new SDF record and verify its tag, 3D coordinates and coverage."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

from extend_final_conformers import wanted_smiles


RDLogger.DisableLog("rdApp.*")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("final_dataset"))
    args = parser.parse_args()
    root = args.bank
    report = json.loads((root / "conformers_report.json").read_text())
    wanted = wanted_smiles(root)
    problems = Counter()
    tags = set()
    records = 0
    shards = sorted(root.glob("conformers_extra*.sdf.gz"))
    for shard in shards:
        print(f"Auditing {shard}", flush=True)
        with gzip.open(shard, "rb") as stream:
            supplier = Chem.ForwardSDMolSupplier(stream, removeHs=False)
            for mol in supplier:
                records += 1
                if mol is None:
                    problems["invalid_sdf_record"] += 1
                    continue
                if not mol.HasProp("smiles"):
                    problems["missing_smiles_tag"] += 1
                    continue
                smiles = mol.GetProp("smiles")
                if smiles in tags:
                    problems["duplicate_extra_smiles"] += 1
                tags.add(smiles)
                if smiles not in wanted:
                    problems["extra_smiles_outside_bank"] += 1
                if mol.GetNumConformers() != 1 or not mol.GetConformer().Is3D():
                    problems["missing_3d_coordinates"] += 1
                without_h = Chem.RemoveHs(mol)
                tagged_mol = Chem.MolFromSmiles(smiles)
                if (tagged_mol is None or
                    Chem.MolToSmiles(without_h, isomericSmiles=False) !=
                        Chem.MolToSmiles(tagged_mol, isomericSmiles=False) or
                    not without_h.HasSubstructMatch(tagged_mol, useChirality=True)):
                    problems["structure_tag_mismatch"] += 1
    if report["wanted_unique_smiles"] != len(wanted):
        problems["wanted_count_mismatch"] += 1
    if len(tags) + report["covered_by_existing_benchmark_shards"] != report["covered_total"]:
        problems["coverage_count_mismatch"] += 1
    result = {
        "new_shards": len(shards), "new_sdf_records": records,
        "new_unique_smiles": len(tags), "wanted_unique_smiles": len(wanted),
        "reported_total_coverage": report["covered_total"],
        "problems": dict(problems), "passed": not problems,
    }
    (root / "conformers_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
