"""NumPy chemistry shared by GAT-compatible and geometric-v2 graphs.

The canonical encoder reproduces ``train_gatv2_search.molecular_features``.
Alignment never attaches canonical-SMILES atom rows to conformer coordinates
without an explicit, stereochemistry-aware graph mapping.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, rdBase


FEATURE_VERSION = "gat21_bond14_shared_chemistry_v2"
GAT_REFERENCE_FEATURE_VERSION = "atomic_number_chirality_rings_bond_order_stereo_v1"
ALIGNMENT_VERSION = "canonical_gat_to_verified_sdf_heavy_order_v1"
COORDINATE_ATOL_ANGSTROM = 2e-4
NODE_DIM = 21
BOND_DIM = 14
RDKIT_VERSION = rdBase.rdkitVersion

NODE_COLUMNS = (
    "degree_div6", "formal_charge_div4", "total_hydrogens_div4",
    "aromatic", "in_ring", "hybrid_sp", "hybrid_sp2", "hybrid_sp3",
    "hybrid_sp3d", "hybrid_sp3d2", "chiral_unspecified", "chiral_cw",
    "chiral_ccw", "chiral_other", "ring3", "ring4", "ring5", "ring6",
    "ring7", "ring8", "mass_div200",
)
BOND_COLUMNS = (
    "single", "double", "triple", "aromatic_type", "other_type",
    "conjugated", "in_ring", "stereo_none", "stereo_any", "stereo_z",
    "stereo_e", "stereo_cis", "stereo_trans", "stereo_other",
)


class AlignmentError(ValueError):
    """Chemistry and immutable conformer provenance cannot be reconciled."""


def canonical_reference(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise AlignmentError("Invalid or empty canonical-reference SMILES")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise AlignmentError("Canonical SMILES round-trip failed")
    return mol


def encode_reference_mol(mol):
    """Encode an already canonical-round-tripped molecule, preserving its order."""
    hybrids = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
               Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
               Chem.HybridizationType.SP3D2]
    chirals = [Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_OTHER]
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append([atom.GetDegree()/6, atom.GetFormalCharge()/4,
                      atom.GetTotalNumHs()/4, float(atom.GetIsAromatic()),
                      float(atom.IsInRing())]
                     + [float(atom.GetHybridization() == item) for item in hybrids]
                     + [float(atom.GetChiralTag() == item) for item in chirals]
                     + [float(atom.IsInRingSize(size)) for size in range(3, 9)]
                     + [atom.GetMass()/200])
    types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
             Chem.BondType.AROMATIC]
    stereos = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY,
               Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE,
               Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS]
    edges, attributes = [], []
    for bond in mol.GetBonds():
        bond_type, stereo = bond.GetBondType(), bond.GetStereo()
        values = [float(bond_type == item) for item in types]
        values += [float(bond_type not in types), float(bond.GetIsConjugated()),
                   float(bond.IsInRing())]
        values += [float(stereo == item) for item in stereos]
        values += [float(stereo not in stereos)]
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((begin, end), (end, begin)))
        attributes.extend((values, values))
    return {
        "z": np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64),
        "x": np.asarray(atoms, dtype=np.float32).reshape(-1, NODE_DIM),
        "edge_index": np.asarray(edges, dtype=np.int64).reshape(-1, 2).T.copy(),
        "edge_attr": np.asarray(attributes, dtype=np.float32).reshape(-1, BOND_DIM),
    }


def encode_smiles(smiles: str):
    """Return canonical GAT-equivalent z/x/edge_index/edge_attr NumPy arrays."""
    return encode_reference_mol(canonical_reference(smiles))


def _source_edges(edges, n_atoms):
    values = np.asarray(edges, dtype=np.int64)
    if values.ndim != 2:
        raise AlignmentError("Source covalent edge array must be a matrix")
    if values.shape[0] != 2 and values.shape[1] == 2:
        values = values.T
    if values.shape[0] != 2:
        raise AlignmentError("Source covalent edge array must have shape [2,E]")
    if values.size and (values.min() < 0 or values.max() >= n_atoms):
        raise AlignmentError("Source covalent edge crosses molecule boundary")
    return values


def aligned_features(smiles, sdf_mol, source_atom_features, source_positions,
                     source_edges=None, source_bond_features=None):
    """Map canonical GAT chemistry to verified original heavy-coordinate order.

    ``sdf_mol`` must retain explicit hydrogens and source atom ordering. Full
    coordinate and element checks establish that it belongs to this NPZ row.
    CW/CCW features are copied from the canonical reference, never recomputed
    in SDF order. No coordinates are generated or altered.
    """
    if sdf_mol is None or sdf_mol.GetNumConformers() != 1:
        raise AlignmentError("Expected one readable source SDF conformer")
    raw = np.asarray(source_atom_features)
    positions = np.asarray(source_positions)
    n_atoms = sdf_mol.GetNumAtoms()
    if raw.shape != (n_atoms, 8) or positions.shape != (n_atoms, 3):
        raise AlignmentError("SDF and NPZ atom shapes disagree")
    if not np.isfinite(positions).all():
        raise AlignmentError("NPZ contains nonfinite coordinates")
    sdf_z = np.asarray([atom.GetAtomicNum() for atom in sdf_mol.GetAtoms()], dtype=np.int64)
    if not np.array_equal(sdf_z, raw[:, 0]):
        raise AlignmentError("SDF and NPZ original atom identities/order disagree")
    for index, atom in enumerate(sdf_mol.GetAtoms()):
        observed = [atom.GetFormalCharge(), atom.GetDegree(), int(atom.GetIsAromatic()), int(atom.IsInRing())]
        if not np.array_equal(raw[index, [1, 2, 3, 7]], observed):
            raise AlignmentError("SDF and NPZ original atom charge/degree/aromatic/ring properties disagree")
    sdf_positions = np.asarray(sdf_mol.GetConformer().GetPositions(), dtype=np.float64)
    coordinate_error = float(np.max(np.abs(sdf_positions - positions))) if n_atoms else 0.0
    if not np.isfinite(sdf_positions).all() or coordinate_error > COORDINATE_ATOL_ANGSTROM:
        raise AlignmentError(f"SDF/NPZ coordinate mismatch: max error {coordinate_error:g} Angstrom")
    if source_edges is not None:
        original_edges = _source_edges(source_edges, n_atoms)
        observed = {(int(a), int(b)) for a, b in original_edges.T}
        expected = set()
        for bond in sdf_mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            expected.update(((a, b), (b, a)))
        if observed != expected or len(observed) != original_edges.shape[1]:
            raise AlignmentError("SDF and NPZ covalent connectivity disagree or contain duplicate edges")
        if source_bond_features is not None:
            original_attributes = np.asarray(source_bond_features)
            if original_attributes.shape != (original_edges.shape[1], 5):
                raise AlignmentError("Original source bond features must have shape [E,5]")
            for index, (a, b) in enumerate(original_edges.T):
                bond = sdf_mol.GetBondBetweenAtoms(int(a), int(b))
                values = [int(bond.GetBondType()), int(bond.GetIsAromatic()),
                          int(bond.GetIsConjugated()), int(bond.IsInRing())]
                if not np.array_equal(original_attributes[index, :4], values):
                    raise AlignmentError("SDF and NPZ covalent bond chemistry disagree")

    reference = canonical_reference(smiles)
    encoded = encode_reference_mol(reference)
    if np.any(encoded["z"] <= 1):
        raise AlignmentError("Canonical GAT atom set contains explicit hydrogen/dummy; heavy-only parity requires an explicit policy")
    if sdf_mol.HasProp("smiles"):
        property_reference = canonical_reference(sdf_mol.GetProp("smiles"))
        if Chem.MolToSmiles(property_reference, isomericSmiles=True) != Chem.MolToSmiles(reference, isomericSmiles=True):
            raise AlignmentError("SDF identity property disagrees with immutable NPZ SMILES")
    heavy_ids = np.flatnonzero(raw[:, 0] > 1)
    target = Chem.RemoveHs(Chem.Mol(sdf_mol), sanitize=True)
    if target.GetNumAtoms() != len(heavy_ids) or reference.GetNumAtoms() != len(heavy_ids):
        raise AlignmentError("Canonical and source heavy-atom sets differ")
    heavy_z = np.asarray([atom.GetAtomicNum() for atom in target.GetAtoms()], dtype=np.int64)
    if not np.array_equal(heavy_z, raw[heavy_ids, 0]):
        raise AlignmentError("Removing hydrogens changed heavy-atom ordering")
    if not np.allclose(np.asarray(target.GetConformer().GetPositions()), positions[heavy_ids],
                       rtol=0, atol=COORDINATE_ATOL_ANGSTROM):
        raise AlignmentError("Removing hydrogens changed coordinate-to-heavy-atom ordering")
    if target.GetNumBonds() != reference.GetNumBonds():
        raise AlignmentError("Canonical and source heavy-bond counts disagree")
    # target.GetSubstructMatch(query) maps query/canonical indices to target.
    mapping = np.asarray(target.GetSubstructMatch(reference, useChirality=True), dtype=np.int64)
    if mapping.shape != (len(heavy_ids),) or len(np.unique(mapping)) != len(heavy_ids):
        raise AlignmentError("No complete stereochemistry-preserving canonical/source bijection")
    if not np.array_equal(encoded["z"], heavy_z[mapping]):
        raise AlignmentError("Mapped canonical atom identities disagree")
    for atom in reference.GetAtoms():
        target_atom = target.GetAtomWithIdx(int(mapping[atom.GetIdx()]))
        if (atom.GetFormalCharge(), atom.GetIsotope(), atom.GetIsAromatic()) != (
                target_atom.GetFormalCharge(), target_atom.GetIsotope(), target_atom.GetIsAromatic()):
            raise AlignmentError("Mapped canonical atom charge/isotope/aromaticity disagrees")
    # Explicitly check every mapped bond, rather than relying only on count.
    for bond in reference.GetBonds():
        target_bond = target.GetBondBetweenAtoms(int(mapping[bond.GetBeginAtomIdx()]),
                                                int(mapping[bond.GetEndAtomIdx()]))
        if target_bond is None or target_bond.GetBondType() != bond.GetBondType():
            raise AlignmentError("Mapped canonical bond connectivity/type disagrees")
    x = np.empty_like(encoded["x"])
    z = np.empty_like(encoded["z"])
    x[mapping] = encoded["x"]
    z[mapping] = encoded["z"]
    return {
        "z": z, "x": x,
        "bond_edge_index": mapping[encoded["edge_index"]],
        "bond_edge_attr": encoded["edge_attr"].copy(),
        "canonical_to_heavy": mapping,
        "audit": {"atoms": len(heavy_ids), "directed_bonds": encoded["edge_index"].shape[1],
                  "coordinate_max_abs_error_angstrom": coordinate_error,
                  "stereochemistry_mapping": True},
    }
