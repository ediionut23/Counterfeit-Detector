"""Reconstruct and describe a verified single-atom substitution."""

from __future__ import annotations

from rdkit import Chem

from audit_verified_single_atom_edits import reverse_matches


def edit_site_and_shell(row: dict) -> tuple[int, tuple]:
    child = Chem.MolFromSmiles(row["smiles"])
    row_id = row.get("id", row.get("relation_id", "unknown"))
    if child is None:
        raise ValueError(f"Invalid child SMILES: {row_id}")
    sites = reverse_matches(row["parent_smiles"], child, "substitution")
    if len(sites) != 1:
        raise ValueError(f"Edit does not reverse uniquely: {row_id}")
    atom = child.GetAtomWithIdx(sites[0])
    expected_mask = sorted({sites[0], *(neighbor.GetIdx() for neighbor in atom.GetNeighbors())})
    stored_mask = [int(index) for index in row["edited_atoms"].split()]
    if stored_mask != expected_mask:
        raise ValueError(f"Exact mask disagrees with reconstructed edit: {row_id}")
    shell = (
        atom.GetIsAromatic(), atom.IsInRing(), atom.GetDegree(),
        tuple(sorted((bond.GetBondTypeAsDouble(),
                      bond.GetOtherAtom(atom).GetSymbol(),
                      bond.GetOtherAtom(atom).GetIsAromatic(),
                      bond.GetOtherAtom(atom).IsInRing())
                     for bond in atom.GetBonds())),
    )
    return sites[0], shell
