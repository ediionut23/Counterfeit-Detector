"""Count catalogue MMP controls that can be paired with generated parents.

Usage: venv/bin/python audit_near_analogue_controls_feasibility.py
This is a feasibility audit, not a claim that catalogue molecules are approved
drugs or experimentally verified negative controls.
"""

import csv
import json
import sqlite3
from pathlib import Path

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.warning")


ROOT = Path(__file__).resolve().parent


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule else None


def smiles_column(path, column):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = canonical(row[column])
            if value:
                yield value


def main():
    source = set(smiles_column(ROOT / "final_dataset/authentic.csv", "smiles"))
    parents = set(smiles_column(ROOT / "final_dataset/relations.csv", "parent_smiles"))
    generated = set(smiles_column(ROOT / "final_dataset/molecules.csv", "smiles"))
    db = sqlite3.connect(ROOT / "mmp_mining_big/mined.mmpdb")
    compounds = {
        identifier: canonical(smiles)
        for identifier, smiles in db.execute("SELECT id, clean_smiles FROM compound")
    }
    source_ids = {identifier for identifier, smiles in compounds.items() if smiles in source}
    shared_parents = parents & {compounds[identifier] for identifier in source_ids}

    pair_rows = 0
    source_pair_rows = 0
    parent_pair_rows = 0
    eligible_pair_rows = 0
    source_pairs = set()
    eligible_pairs = set()
    control_smiles = set()
    matched_parents = set()
    for first_id, second_id in db.execute("SELECT compound1_id, compound2_id FROM pair"):
        pair_rows += 1
        if first_id not in source_ids or second_id not in source_ids:
            continue
        source_pair_rows += 1
        first, second = compounds[first_id], compounds[second_id]
        source_pairs.add(tuple(sorted((first, second))))
        if first not in shared_parents and second not in shared_parents:
            continue
        parent_pair_rows += 1
        # A generated child may also be catalogued; it cannot serve as a
        # clean negative without a separate label-conflict audit.
        candidates = [(first, second), (second, first)]
        usable = [(parent, control) for parent, control in candidates
                  if parent in shared_parents and control not in generated and parent != control]
        if not usable:
            continue
        eligible_pair_rows += 1
        eligible_pairs.add(tuple(sorted((first, second))))
        matched_parents.update(parent for parent, _ in usable)
        control_smiles.update(control for _, control in usable)

    result = {
        "source_molecules": len(source),
        "generated_parent_molecules": len(parents),
        "generated_child_molecules": len(generated),
        "mmp_compounds": len(compounds),
        "mmp_compounds_in_source": len(source_ids),
        "mmp_pair_rows": pair_rows,
        "mmp_pair_rows_both_in_source": source_pair_rows,
        "distinct_mmp_pairs_both_in_source": len(source_pairs),
        "generated_parents_in_mmp_source": len(shared_parents),
        "mmp_pair_rows_with_generated_parent": parent_pair_rows,
        "candidate_control_pair_rows_excluding_generated_children": eligible_pair_rows,
        "distinct_candidate_control_pairs": len(eligible_pairs),
        "generated_parents_with_candidate_control": len(matched_parents),
        "distinct_candidate_control_molecules": len(control_smiles),
        "interpretation": "Catalogue-source near-analogue feasibility only; no biological or product-authenticity labels.",
    }
    db.close()
    destination = ROOT / "reports/near_analogue_controls_feasibility.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
