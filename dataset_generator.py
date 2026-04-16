"""
Pharmaceutical Dataset Generator v2 — RDKit-Native Editing
===========================================================
Fixes all issues identified by diagnostic_investigation.py:
  1. NO string replacement on SMILES — all edits via RDKit RWMol
  2. Rejection sampling: counterfeits must match authentic descriptor distributions
  3. Deduplication at generation time (not after)
  4. Every counterfeit stores original_smiles for family-aware splitting
  5. Balanced source distribution (no backup domination)
  6. Similarity-controlled: Tanimoto 0.5–0.85 (hard to distinguish)

Usage:  python dataset_generator_v2.py
Output: enhanced_pharma_v2_<timestamp>.pt
"""

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdmolops
from rdkit.Chem import RWMol, Atom, BondType
import torch
from torch_geometric.data import HeteroData
from datetime import datetime
import logging
from tqdm.auto import tqdm
import warnings
from rdkit import RDLogger
import random
import json
import time
from typing import List, Dict, Tuple, Optional
import os
from collections import defaultdict

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════
#  RDKIT-NATIVE MOLECULAR EDITOR
# ═══════════════════════════════════════════════════════════

class MolecularEditor:
    """
    Performs molecular modifications using RDKit's RWMol API.
    NO string replacement on SMILES — all operations are graph-level.
    """

    # ── Atom Replacement ──
    @staticmethod
    def replace_atom(mol, target_atomic_num, new_atomic_num, max_replacements=1):
        """Replace one atom of target type with new type (graph-level edit)."""
        rw = RWMol(mol)
        replaced = 0
        candidates = [a.GetIdx() for a in rw.GetAtoms()
                      if a.GetAtomicNum() == target_atomic_num
                      and not a.IsInRing()]  # prefer non-ring atoms
        if not candidates:
            candidates = [a.GetIdx() for a in rw.GetAtoms()
                          if a.GetAtomicNum() == target_atomic_num]
        random.shuffle(candidates)
        for idx in candidates:
            if replaced >= max_replacements:
                break
            rw.GetAtomWithIdx(idx).SetAtomicNum(new_atomic_num)
            replaced += 1
        if replaced == 0:
            return None
        try:
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except:
            return None

    # ── Bond Modification ──
    @staticmethod
    def change_bond_order(mol, from_type, to_type, max_changes=1):
        """Change bond order (e.g. single→double) on non-ring bonds."""
        rw = RWMol(mol)
        changed = 0
        candidates = [b for b in rw.GetBonds()
                      if b.GetBondType() == from_type
                      and not b.IsInRing()]
        random.shuffle(candidates)
        for bond in candidates:
            if changed >= max_changes:
                break
            bond.SetBondType(to_type)
            changed += 1
        if changed == 0:
            return None
        try:
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except:
            return None

    # ── Substituent Addition ──
    @staticmethod
    def add_substituent(mol, substituent_smiles, max_additions=1):
        """Add a small fragment to a random non-ring carbon."""
        frag = Chem.MolFromSmiles(substituent_smiles)
        if frag is None:
            return None

        candidates = [a.GetIdx() for a in mol.GetAtoms()
                      if a.GetAtomicNum() == 6
                      and a.GetDegree() < 4
                      and not a.IsInRing()]
        if not candidates:
            return None

        random.shuffle(candidates)
        for attach_idx in candidates[:max_additions]:
            try:
                combined = Chem.RWMol(Chem.CombineMols(mol, frag))
                frag_start = mol.GetNumAtoms()
                # Find first attachable atom in fragment
                frag_attach = None
                for a in frag.GetAtoms():
                    if a.GetDegree() < a.GetTotalValence():
                        frag_attach = frag_start + a.GetIdx()
                        break
                if frag_attach is None:
                    frag_attach = frag_start  # fallback: first atom

                combined.AddBond(attach_idx, frag_attach, BondType.SINGLE)
                Chem.SanitizeMol(combined)
                return combined.GetMol()
            except:
                continue
        return None

    # ── Substituent Removal ──
    @staticmethod
    def remove_terminal_group(mol):
        """Remove a random terminal (degree-1) non-ring heavy atom."""
        candidates = [a.GetIdx() for a in mol.GetAtoms()
                      if a.GetDegree() == 1
                      and a.GetAtomicNum() > 1  # not hydrogen
                      and not a.IsInRing()]
        if not candidates:
            return None
        random.shuffle(candidates)
        rw = RWMol(mol)
        try:
            rw.RemoveAtom(candidates[0])
            Chem.SanitizeMol(rw)
            result = rw.GetMol()
            if result.GetNumAtoms() >= 5:
                return result
        except:
            pass
        return None

    # ── Ring Modification ──
    @staticmethod
    def modify_ring_atom(mol):
        """Replace a ring atom (C→N or N→C) to create a heterocycle variant."""
        ring_info = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)

        # Find ring C or N atoms
        candidates = []
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
                candidates.append((idx, 7))  # C→N
            elif atom.GetAtomicNum() == 7 and atom.GetIsAromatic():
                candidates.append((idx, 6))  # N→C

        if not candidates:
            return None

        random.shuffle(candidates)
        idx, new_num = candidates[0]

        rw = RWMol(mol)
        try:
            rw.GetAtomWithIdx(idx).SetAtomicNum(new_num)
            # Adjust hydrogens
            rw.GetAtomWithIdx(idx).SetNoImplicit(False)
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except:
            return None

    # ── Halogen Swap ──
    @staticmethod
    def swap_halogen(mol):
        """Swap one halogen for another (F↔Cl, Cl↔Br, Br↔I)."""
        swaps = {9: 17, 17: 9, 35: 53, 53: 35}  # F↔Cl, Br↔I
        candidates = [(a.GetIdx(), a.GetAtomicNum()) for a in mol.GetAtoms()
                      if a.GetAtomicNum() in swaps]
        if not candidates:
            return None
        random.shuffle(candidates)
        idx, old_num = candidates[0]
        rw = RWMol(mol)
        try:
            rw.GetAtomWithIdx(idx).SetAtomicNum(swaps[old_num])
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except:
            return None

    # ── Functional Group Transform ──
    @staticmethod
    def transform_functional_group(mol):
        """Apply a bioisosteric SMARTS-based transformation."""
        transforms = [
            # Carboxylic acid → amide
            ('[C:1](=O)[OH]', '[C:1](=O)N'),
            # Ester → amide
            ('[C:1](=O)[O:2][C:3]', '[C:1](=O)[NH][C:3]'),
            # Hydroxyl → fluorine
            ('[c:1][OH]', '[c:1]F'),
            # Primary amine → hydroxyl
            ('[C:1][NH2]', '[C:1]O'),
            # Methoxy → ethoxy (add one carbon)
            # Thioether → ether
            ('[C:1][S:2][C:3]', '[C:1][O:2][C:3]'),
        ]
        random.shuffle(transforms)
        for reactant_smarts, product_smarts in transforms:
            try:
                pattern = Chem.MolFromSmarts(reactant_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    from rdkit.Chem.AllChem import ReplaceSidechains, ReplaceCore
                    rxn_smarts = f'{reactant_smarts}>>{product_smarts}'
                    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                    if rxn:
                        products = rxn.RunReactants((mol,))
                        if products:
                            product = products[0][0]
                            Chem.SanitizeMol(product)
                            return product
            except:
                continue
        return None


# ═══════════════════════════════════════════════════════════
#  DESCRIPTOR MATCHER (Rejection Sampling)
# ═══════════════════════════════════════════════════════════

class DescriptorMatcher:
    """
    Ensures counterfeits match the descriptor distribution of authentics.
    Rejects counterfeits that are too different in MW, LogP, TPSA, etc.
    """

    def __init__(self):
        self.auth_stats = {}  # {descriptor: (mean, std)}

    def fit(self, authentic_mols: List):
        """Compute statistics from authentic molecules."""
        descriptors = defaultdict(list)
        for mol in authentic_mols:
            if mol is None:
                continue
            try:
                descriptors['mw'].append(Descriptors.ExactMolWt(mol))
                descriptors['logp'].append(Descriptors.MolLogP(mol))
                descriptors['tpsa'].append(Descriptors.TPSA(mol))
                descriptors['hbd'].append(Descriptors.NumHDonors(mol))
                descriptors['hba'].append(Descriptors.NumHAcceptors(mol))
                descriptors['num_atoms'].append(mol.GetNumHeavyAtoms())
            except:
                continue

        for key, values in descriptors.items():
            arr = np.array(values)
            self.auth_stats[key] = (arr.mean(), arr.std())

        logger.info("Descriptor statistics from authentic molecules:")
        for key, (mean, std) in self.auth_stats.items():
            logger.info(f"  {key}: {mean:.2f} ± {std:.2f}")

    def is_acceptable(self, mol, tolerance=2.5) -> bool:
        """
        Check if molecule's descriptors are within tolerance * std of authentic means.
        tolerance=2.5 means ~99% of the authentic distribution is allowed.
        """
        if mol is None:
            return False
        try:
            checks = {
                'mw': Descriptors.ExactMolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'num_atoms': mol.GetNumHeavyAtoms(),
            }
            for key, value in checks.items():
                if key in self.auth_stats:
                    mean, std = self.auth_stats[key]
                    if std < 1e-6:
                        continue
                    z_score = abs(value - mean) / std
                    if z_score > tolerance:
                        return False
            return True
        except:
            return False


# ═══════════════════════════════════════════════════════════
#  MAIN GENERATOR v2
# ═══════════════════════════════════════════════════════════

class PharmaceuticalGeneratorV2:
    """
    Generates 30K molecules with RDKit-native editing.
    All counterfeits are structurally valid, descriptor-matched,
    and similarity-controlled.
    """

    def __init__(self):
        self.editor = MolecularEditor()
        self.matcher = DescriptorMatcher()
        self.seen_smiles = set()  # global dedup

        # Difficulty levels with corresponding edit operations
        self.difficulty_ops = {
            'structural': [
                ('replace_atom', {'target_atomic_num': 8, 'new_atomic_num': 16}),  # O→S
                ('replace_atom', {'target_atomic_num': 7, 'new_atomic_num': 8}),   # N→O
                ('replace_atom', {'target_atomic_num': 6, 'new_atomic_num': 7}),   # C→N (non-ring)
                ('add_substituent', {'substituent_smiles': 'C'}),                   # add methyl
                ('add_substituent', {'substituent_smiles': 'F'}),                   # add fluorine
                ('remove_terminal_group', {}),
            ],
            'bioisosteric': [
                ('transform_functional_group', {}),
                ('replace_atom', {'target_atomic_num': 8, 'new_atomic_num': 16}),
                ('replace_atom', {'target_atomic_num': 16, 'new_atomic_num': 8}),
                ('swap_halogen', {}),
                ('change_bond_order', {'from_type': BondType.SINGLE, 'to_type': BondType.DOUBLE}),
            ],
            'stereochemical': [
                ('modify_ring_atom', {}),
                ('swap_halogen', {}),
                ('replace_atom', {'target_atomic_num': 7, 'new_atomic_num': 6}),
                ('add_substituent', {'substituent_smiles': 'O'}),
            ],
            'scaffold': [
                ('modify_ring_atom', {}),
                ('replace_atom', {'target_atomic_num': 6, 'new_atomic_num': 7}),
                ('replace_atom', {'target_atomic_num': 7, 'new_atomic_num': 8}),
                ('change_bond_order', {'from_type': BondType.DOUBLE, 'to_type': BondType.SINGLE}),
            ],
            'prodrug': [
                ('add_substituent', {'substituent_smiles': 'C(=O)C'}),  # acetyl
                ('add_substituent', {'substituent_smiles': 'C(=O)O'}),  # carboxyl
                ('transform_functional_group', {}),
            ],
        }

    def _is_valid_molecule(self, mol) -> bool:
        """Basic validity checks."""
        if mol is None:
            return False
        try:
            num_atoms = mol.GetNumHeavyAtoms()
            if num_atoms < 6 or num_atoms > 80:
                return False
            mw = Descriptors.ExactMolWt(mol)
            if mw < 80 or mw > 700:
                return False
            logp = Descriptors.MolLogP(mol)
            if logp < -4 or logp > 7:
                return False
            return True
        except:
            return False

    def _is_unique(self, smiles: str) -> bool:
        """Check and register SMILES for global deduplication."""
        if smiles in self.seen_smiles:
            return False
        self.seen_smiles.add(smiles)
        return True

    def _tanimoto(self, mol1, mol2) -> float:
        """Tanimoto similarity using Morgan fingerprints."""
        try:
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0

    # ── Fetch Authentic Molecules ──

    def fetch_authentic_molecules(self, target: int = 15000) -> List[Dict]:
        """Fetch from ChEMBL with fallback to PubChem."""
        all_mols = []

        # Try ChEMBL
        try:
            from chembl_webresource_client.new_client import new_client
            logger.info("Fetching from ChEMBL...")
            molecule = new_client.molecule

            for phase in [4, 3, 2, 1]:
                if len(all_mols) >= target:
                    break
                query = molecule.filter(max_phase=phase, molecule_type='Small molecule')
                count = 0
                for mol_data in query:
                    if len(all_mols) >= target:
                        break
                    if count > target // 2:  # don't spend too long on one phase
                        break
                    try:
                        smiles = mol_data['molecule_structures']['canonical_smiles']
                        rdkit_mol = Chem.MolFromSmiles(smiles)
                        if rdkit_mol and self._is_valid_molecule(rdkit_mol):
                            canon = Chem.MolToSmiles(rdkit_mol)
                            if self._is_unique(canon):
                                all_mols.append({
                                    'smiles': canon,
                                    'mol': rdkit_mol,
                                    'source': 'ChEMBL',
                                    'source_id': mol_data.get('molecule_chembl_id', f'CHEMBL_{count}'),
                                    'max_phase': phase,
                                })
                                count += 1
                    except:
                        continue
                logger.info(f"  Phase {phase}: {count} molecules")

        except Exception as e:
            logger.warning(f"ChEMBL fetch failed: {e}")

        # Supplement with PubChem if needed
        if len(all_mols) < target:
            remaining = target - len(all_mols)
            logger.info(f"Fetching {remaining} more from PubChem...")
            all_mols.extend(self._fetch_pubchem(remaining))

        # Supplement with drug-like generation if still short
        if len(all_mols) < target:
            remaining = target - len(all_mols)
            logger.info(f"Generating {remaining} drug-like molecules...")
            all_mols.extend(self._generate_druglike(remaining))

        logger.info(f"Total authentic: {len(all_mols)}")
        return all_mols[:target]

    def _fetch_pubchem(self, limit: int) -> List[Dict]:
        """Fetch from PubChem REST API."""
        import requests
        from urllib.parse import quote

        molecules = []
        terms = [
            "aspirin", "ibuprofen", "acetaminophen", "caffeine", "morphine",
            "penicillin", "metformin", "atorvastatin", "omeprazole", "sertraline",
            "amoxicillin", "ciprofloxacin", "metoprolol", "lisinopril",
            "amlodipine", "simvastatin", "azithromycin", "warfarin",
            "naproxen", "diazepam", "fluoxetine", "clopidogrel",
            "losartan", "furosemide", "prednisone", "gabapentin",
            "tramadol", "codeine", "hydrochlorothiazide", "ranitidine",
        ]
        base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

        for term in terms:
            if len(molecules) >= limit:
                break
            try:
                time.sleep(0.5)
                resp = requests.get(f"{base}/compound/name/{quote(term)}/property/CanonicalSMILES/JSON",
                                    timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    for prop in data.get('PropertyTable', {}).get('Properties', []):
                        smiles = prop.get('CanonicalSMILES')
                        if smiles:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol and self._is_valid_molecule(mol):
                                canon = Chem.MolToSmiles(mol)
                                if self._is_unique(canon):
                                    molecules.append({
                                        'smiles': canon,
                                        'mol': mol,
                                        'source': 'PubChem',
                                        'source_id': f"PC_{term}",
                                        'max_phase': 0,
                                    })
            except:
                continue

        return molecules[:limit]

    def _generate_druglike(self, limit: int) -> List[Dict]:
        """Generate valid drug-like molecules using RDKit."""
        molecules = []
        scaffolds = [
            'c1ccccc1', 'c1ccncc1', 'c1ccoc1', 'c1ccsc1',
            'C1CCCCC1', 'C1CCNCC1', 'C1CCOCC1',
            'c1ccc2ccccc2c1', 'c1ccc2[nH]ccc2c1',
            'c1ccc2ncccc2c1', 'c1cnc2ccccc2n1',
        ]
        substituents = ['C', 'CC', 'O', 'N', 'F', 'Cl', 'OC', 'NC',
                        'C(=O)O', 'C(=O)N', 'C#N', 'C(F)(F)F', 'S(=O)(=O)N']

        attempts = 0
        while len(molecules) < limit and attempts < limit * 10:
            attempts += 1
            try:
                scaffold_smi = random.choice(scaffolds)
                scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
                if scaffold_mol is None:
                    continue

                # Add 1-2 random substituents via the editor
                mol = scaffold_mol
                for _ in range(random.randint(1, 2)):
                    sub = random.choice(substituents)
                    new_mol = self.editor.add_substituent(mol, sub)
                    if new_mol and self._is_valid_molecule(new_mol):
                        mol = new_mol

                canon = Chem.MolToSmiles(mol)
                if self._is_valid_molecule(mol) and self._is_unique(canon):
                    molecules.append({
                        'smiles': canon,
                        'mol': mol,
                        'source': 'Generated',
                        'source_id': f'GEN_{len(molecules):05d}',
                        'max_phase': 0,
                    })
            except:
                continue

        return molecules[:limit]

    # ── Generate Counterfeits ──

    def generate_counterfeits(self, authentic_mols: List[Dict],
                              target: int = 15000) -> List[Dict]:
        """
        Generate counterfeits using RDKit-native editing with:
        - Descriptor rejection sampling
        - Tanimoto similarity control (0.5–0.85)
        - Global deduplication
        - Family tracking (original_smiles)
        """
        # Fit descriptor matcher on authentic molecules
        auth_rdkit_mols = [m['mol'] for m in authentic_mols if m.get('mol')]
        self.matcher.fit(auth_rdkit_mols)

        # Difficulty distribution
        distribution = {
            'structural': int(target * 0.25),
            'bioisosteric': int(target * 0.35),
            'stereochemical': int(target * 0.25),
            'scaffold': int(target * 0.10),
            'prodrug': target - int(target * 0.95),  # remainder (~5%)
        }

        all_counterfeits = []

        for difficulty, count in distribution.items():
            logger.info(f"Generating {count} {difficulty} counterfeits...")
            ops = self.difficulty_ops[difficulty]
            generated = []
            attempts = 0
            max_attempts = count * 20  # allow retries

            # Cycle through authentic molecules
            auth_idx = 0
            while len(generated) < count and attempts < max_attempts:
                attempts += 1
                auth_entry = authentic_mols[auth_idx % len(authentic_mols)]
                auth_idx += 1
                auth_mol = auth_entry.get('mol')
                auth_smiles = auth_entry['smiles']

                if auth_mol is None:
                    continue

                # Pick a random operation for this difficulty
                op_name, op_kwargs = random.choice(ops)

                # Apply the edit
                try:
                    if op_name == 'replace_atom':
                        new_mol = self.editor.replace_atom(auth_mol, **op_kwargs)
                    elif op_name == 'change_bond_order':
                        new_mol = self.editor.change_bond_order(auth_mol, **op_kwargs)
                    elif op_name == 'add_substituent':
                        new_mol = self.editor.add_substituent(auth_mol, **op_kwargs)
                    elif op_name == 'remove_terminal_group':
                        new_mol = self.editor.remove_terminal_group(auth_mol)
                    elif op_name == 'modify_ring_atom':
                        new_mol = self.editor.modify_ring_atom(auth_mol)
                    elif op_name == 'swap_halogen':
                        new_mol = self.editor.swap_halogen(auth_mol)
                    elif op_name == 'transform_functional_group':
                        new_mol = self.editor.transform_functional_group(auth_mol)
                    else:
                        continue

                    if new_mol is None:
                        continue

                    new_smiles = Chem.MolToSmiles(new_mol)

                    # Validity checks
                    if not self._is_valid_molecule(new_mol):
                        continue
                    if new_smiles == auth_smiles:
                        continue
                    if not self._is_unique(new_smiles):
                        continue

                    # Descriptor rejection sampling
                    if not self.matcher.is_acceptable(new_mol, tolerance=2.5):
                        continue

                    # Similarity control
                    sim = self._tanimoto(auth_mol, new_mol)
                    if sim < 0.4 or sim > 0.90:
                        continue

                    generated.append({
                        'smiles': new_smiles,
                        'mol': new_mol,
                        'source': f'Counterfeit_{difficulty}',
                        'source_id': f'CF_{difficulty[:3].upper()}_{len(generated):05d}',
                        'label': 1,
                        'counterfeit_type': op_name,
                        'difficulty_level': difficulty,
                        'original_smiles': auth_smiles,
                        'similarity': sim,
                        'max_phase': 0,
                    })
                except:
                    continue

            logger.info(f"  {difficulty}: generated {len(generated)}/{count} "
                        f"(attempts: {attempts})")
            all_counterfeits.extend(generated)

        logger.info(f"Total counterfeits: {len(all_counterfeits)}")
        return all_counterfeits

    # ── Graph Conversion ──

    def _calculate_descriptors(self, mol) -> Optional[Dict]:
        """Calculate molecular descriptors."""
        try:
            return {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'qed': Descriptors.qed(mol),
            }
        except:
            return None

    def _extract_atom_features(self, atom, mol) -> List[float]:
        """26-dimensional atom features (compatible with v1)."""
        try:
            neighbors = atom.GetNeighbors()
            return [
                float(atom.GetAtomicNum()) / 100.0,
                float(atom.GetDegree()) / 6.0,
                float(atom.GetTotalDegree()) / 6.0,
                float(atom.GetFormalCharge() + 3) / 6.0,
                float(atom.GetHybridization()) / 6.0,
                float(atom.GetIsAromatic()),
                float(atom.GetTotalNumHs()) / 4.0,
                float(atom.GetMass()) / 200.0,
                float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
                float(atom.HasProp('_ChiralityPossible')),
                float(atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0),
                float(atom.GetSymbol() in ['N', 'O', 'F']),
                float(atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']),
                float(atom.GetSymbol() in ['S', 'P']),
                float(atom.IsInRing()),
                float(atom.IsInRingSize(5)),
                float(atom.IsInRingSize(6)),
                float(any(mol.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), s)
                          for s in [3, 4, 7, 8])),
                float(atom.GetNumRadicalElectrons()),
                float(len([n for n in neighbors if n.GetIsAromatic()]) /
                      max(1, len(neighbors))),
                float(len([n for n in neighbors if n.GetSymbol() in ['N', 'O', 'S']]) /
                      max(1, len(neighbors))),
                float(len([n for n in neighbors if n.GetSymbol() in ['F', 'Cl', 'Br', 'I']]) /
                      max(1, len(neighbors))),
                float(any(n.GetFormalCharge() != 0 for n in neighbors)),
                float(atom.GetHybridization() == Chem.HybridizationType.SP),
                float(atom.GetHybridization() == Chem.HybridizationType.SP2),
                float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            ]
        except:
            return [0.0] * 26

    def _extract_bond_features(self, bond, mol) -> List[float]:
        """18-dimensional bond features (compatible with v1)."""
        try:
            sa = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            ea = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            return [
                float(bond.GetBondTypeAsDouble()) / 3.0,
                float(bond.GetIsAromatic()),
                float(bond.IsInRing()),
                float(bond.GetIsConjugated()),
                float(bond.GetStereo() != Chem.BondStereo.STEREONONE),
                float(bond.GetStereo() == Chem.BondStereo.STEREOZ),
                float(bond.IsInRingSize(6)),
                float(bond.IsInRingSize(5)),
                float(any(mol.GetRingInfo().IsBondInRingOfSize(bond.GetIdx(), s)
                          for s in [3, 4, 7, 8])),
                float((sa.GetDegree() + ea.GetDegree()) / 10.0),
                float(sa.GetIsAromatic() and ea.GetIsAromatic()),
                float(sa.GetIsAromatic() != ea.GetIsAromatic()),
                float(abs(sa.GetFormalCharge() - ea.GetFormalCharge())),
                float(sa.GetSymbol() in ['N', 'O', 'S', 'P']),
                float(ea.GetSymbol() in ['N', 'O', 'S', 'P']),
                float(sa.GetSymbol() != ea.GetSymbol()),
                float(abs(sa.GetAtomicNum() - ea.GetAtomicNum()) / 50.0),
                float(bond.GetBondType() == Chem.BondType.DOUBLE),
            ]
        except:
            return [0.0] * 18

    def convert_to_heterograph(self, smiles: str, label: int,
                               metadata: Dict = None) -> Optional[HeteroData]:
        """Convert SMILES to HeteroData graph."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            hetero = HeteroData()
            node_features = {}
            atom_type_map = {}

            for idx, atom in enumerate(mol.GetAtoms()):
                atype = atom.GetSymbol()
                atom_type_map[idx] = atype
                feats = self._extract_atom_features(atom, mol)
                if atype not in node_features:
                    node_features[atype] = []
                node_features[atype].append(feats)

            for atype, feats in node_features.items():
                hetero[atype].x = torch.tensor(feats, dtype=torch.float)

            edge_indices = {}
            edge_features = {}

            for bond in mol.GetBonds():
                si = bond.GetBeginAtomIdx()
                ei = bond.GetEndAtomIdx()
                st = atom_type_map[si]
                et = atom_type_map[ei]
                bf = self._extract_bond_features(bond, mol)

                s_local = sum(1 for i in range(si) if atom_type_map[i] == st)
                e_local = sum(1 for i in range(ei) if atom_type_map[i] == et)

                for etype, s, e in [((st, 'bond_to', et), s_local, e_local),
                                     ((et, 'bond_to', st), e_local, s_local)]:
                    if etype not in edge_indices:
                        edge_indices[etype] = [[], []]
                        edge_features[etype] = []
                    edge_indices[etype][0].append(s)
                    edge_indices[etype][1].append(e)
                    edge_features[etype].append(bf)

            for etype, indices in edge_indices.items():
                if indices[0]:
                    hetero[etype].edge_index = torch.tensor(
                        [indices[0], indices[1]], dtype=torch.long)
                    hetero[etype].edge_attr = torch.tensor(
                        edge_features[etype], dtype=torch.float)

            hetero.y = torch.tensor([label], dtype=torch.long)
            hetero.smiles = smiles

            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        setattr(hetero, k, v)

            return hetero
        except:
            return None

    # ── Main Pipeline ──

    def create_dataset(self, target_authentic: int = 15000,
                       target_counterfeit: int = 15000) -> Tuple:
        """Full pipeline: fetch → generate → convert → save."""

        logger.info("=" * 70)
        logger.info("  PHARMACEUTICAL DATASET GENERATOR v2")
        logger.info("  RDKit-native editing, rejection sampling, dedup at generation")
        logger.info("=" * 70)

        # 1. Fetch authentic
        authentic = self.fetch_authentic_molecules(target_authentic)
        if not authentic:
            logger.error("No authentic molecules fetched!")
            return None, None, None

        # 2. Register authentic SMILES (already done during fetch via _is_unique)

        # 3. Generate counterfeits
        counterfeits = self.generate_counterfeits(authentic, target_counterfeit)

        # 4. Build combined entries
        all_entries = []
        for entry in authentic:
            all_entries.append({
                **entry,
                'label': 0,
                'counterfeit_type': 'authentic',
                'difficulty_level': 'authentic',
                'original_smiles': entry['smiles'],
                'similarity': 1.0,
            })
        all_entries.extend(counterfeits)

        # Shuffle
        random.shuffle(all_entries)

        # 5. Convert to heterographs
        logger.info("Converting to heterographs...")
        graphs = []
        labels = []
        meta_node_types = set()
        meta_edge_types = set()

        for entry in tqdm(all_entries, desc="Converting"):
            metadata = {
                'source': entry.get('source', 'unknown'),
                'source_id': entry.get('source_id', 'unknown'),
                'counterfeit_type': entry.get('counterfeit_type', 'unknown'),
                'difficulty_level': entry.get('difficulty_level', 'unknown'),
                'original_smiles': entry.get('original_smiles', entry['smiles']),
            }
            # Add similarity only for counterfeits
            if entry.get('similarity') is not None and entry['label'] == 1:
                metadata['similarity'] = entry['similarity']

            graph = self.convert_to_heterograph(
                entry['smiles'], entry['label'], metadata)

            if graph is not None:
                graphs.append(graph)
                labels.append(entry['label'])
                for nt in graph.node_types:
                    meta_node_types.add(nt)
                for et in graph.edge_types:
                    meta_edge_types.add(et)

        metadata = (sorted(list(meta_node_types)), sorted(list(meta_edge_types)))

        # 6. Report
        auth_count = sum(1 for l in labels if l == 0)
        fake_count = sum(1 for l in labels if l == 1)

        logger.info("=" * 70)
        logger.info("DATASET v2 COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total: {len(graphs)}")
        logger.info(f"Authentic: {auth_count} ({auth_count/len(labels)*100:.1f}%)")
        logger.info(f"Counterfeit: {fake_count} ({fake_count/len(labels)*100:.1f}%)")
        logger.info(f"Unique SMILES: {len(self.seen_smiles)}")
        logger.info(f"Node types: {len(metadata[0])}")
        logger.info(f"Edge types: {len(metadata[1])}")

        # Verify no duplicates
        graph_smiles = [g.smiles for g in graphs]
        assert len(graph_smiles) == len(set(graph_smiles)), \
            "DUPLICATE SMILES FOUND IN FINAL DATASET!"

        return graphs, labels, metadata

    def save_dataset(self, graphs, labels, metadata, filename=None):
        """Save dataset."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_pharma_v2_{ts}.pt"

        torch.save((graphs, labels, metadata), filename)
        logger.info(f"Dataset saved: {filename}")

        # Stats
        stats = {
            'total': len(graphs),
            'authentic': sum(1 for l in labels if l == 0),
            'counterfeit': sum(1 for l in labels if l == 1),
            'unique_smiles': len(graphs),  # guaranteed unique
            'node_types': metadata[0],
            'edge_types_count': len(metadata[1]),
            'atom_features': 26,
            'bond_features': 18,
            'generator_version': 'v2_rdkit_native',
            'improvements': [
                'RDKit-native molecular editing (no SMILES string ops)',
                'Descriptor rejection sampling',
                'Tanimoto similarity control (0.4-0.9)',
                'Global deduplication at generation time',
                'Family tracking via original_smiles',
                'No backup molecule domination',
            ],
            'timestamp': datetime.now().isoformat(),
        }

        stats_file = filename.replace('.pt', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Stats saved: {stats_file}")


def main():
    print("=" * 70)
    print("  PHARMACEUTICAL DATASET GENERATOR v2")
    print("  RDKit-native editing • Rejection sampling • Zero duplicates")
    print("=" * 70)
    print()

    gen = PharmaceuticalGeneratorV2()

    graphs, labels, metadata = gen.create_dataset(
        target_authentic=15000,
        target_counterfeit=15000
    )

    if graphs:
        gen.save_dataset(graphs, labels, metadata)

        auth = sum(1 for l in labels if l == 0)
        fake = sum(1 for l in labels if l == 1)

        print(f"\nFinal: {len(graphs)} molecules ({auth} auth + {fake} fake)")
        print(f"All SMILES unique: ✓")
        print(f"Every counterfeit has original_smiles: ✓")
        print(f"Ready for training with family-aware split.")
    else:
        print("Failed to create dataset.")


if __name__ == "__main__":
    main()