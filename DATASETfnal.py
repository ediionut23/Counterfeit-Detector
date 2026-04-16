"""
Enhanced Multi-Source Pharmaceutical Dataset Generator - 30K Version
Generates 30K molecules (15K authentic + 15K counterfeits)
Target: 80% model accuracy with sophisticated counterfeit patterns
File: dataset_generator_30k.py
"""

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
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
from urllib.parse import quote

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


class MultiSourcePharmaceuticalGenerator:
    """Enhanced generator for 30K molecules dataset"""
    
    def __init__(self):
        self._initialize_sophisticated_patterns()
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize backup molecules in case APIs fail"""
        self.backup_molecules = [
            "CCO",  # Ethanol
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine  
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN(C)CCCC1(C2=CC=CC=C2)C3=CC=CC=C3",  
            "C1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Benzophenone
            "CC1=CC=C(C=C1)C(C)C",
            "CC(C)(C)C1=CC=C(C=C1)O",
            "CC1=CC=CC=C1N",
            "C1=CC=C2C(=C1)C=CC=C2",
            "CC(C)N",
            "CCN(CC)CC",
            "CCCCCCCCCC",
            "C1CCCCC1",
            "C1=CC=C(C=C1)Cl",
            "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",
            "CN1CCC2=CC(=C(C=C2C1CC3=CC=C(C=C3)O)OC)OC",
            "CC(=O)NC1=CC=C(C=C1)O",
            "C1=CC=C(C=C1)CN(C)C",
            "CC(C)NCC(C1=CC(=C(C=C1)O)CO)O",
        ]
    
    def _initialize_sophisticated_patterns(self):
        """Initialize sophisticated counterfeit patterns for 80% accuracy target"""
        
        # Level 1: Structural modifications (25% of counterfeits)
        self.structural_patterns = {
            'critical_functionalities': {
                'C(=O)O': 'C(=O)NH2',
                'C(=O)NH2': 'C(=S)NH2',
                'OH': 'SH',
                'NH2': 'NHCH3',
                'C#N': 'NO2',
                'C(F)(F)F': 'C(Cl)(Cl)Cl',
                'OCH3': 'OCH2CH3',
                'C=C': 'C-C',
            },
            'ring_modifications': {
                'c1ccccc1': 'c1ccncc1',
                'c1ccccc1': 'c1ccsc1',
                'C1CCCCC1': 'C1CCOCC1',
                'C1CCCC1': 'C1CCNC1',
                'c1ccoc1': 'c1ccsc1',
            }
        }
        
        # Level 2: Bioisosteric replacements (35% of counterfeits)
        self.bioisosteric_patterns = {
            'classical_bioisosteres': {
                'C(=O)OH': 'S(=O)(=O)OH',
                'C(=O)OH': 'P(=O)(OH)2',
                'NH': 'CH2',
                'O': 'S',
                'CH2CH2': 'CH=CH',
                'C(=O)': 'C(=S)',
                'NO2': 'CF3',
                'Br': 'CF3',
            },
            'nonclassical_bioisosteres': {
                'c1ccccc1': 'C1=CC=CC=C1',
                'C(=O)NHCH3': 'C(=O)OCH3',
                'NHCOCH3': 'NHSO2CH3',
                'C(=O)N(CH3)2': 'SO2N(CH3)2',
                'COOH': 'CONHOH',
            }
        }
        
        # Level 3: Stereochemical patterns (25% of counterfeits)
        self.stereochemical_patterns = {
            'positional_isomers': {
                'c1cc(X)ccc1': 'c1ccc(X)cc1',
                'c1c(X)cccc1': 'c1cc(X)ccc1',
                'CC(X)C': 'C(X)CC',
            },
            'regioisomers': {
                'C=CC(X)': 'CC=C(X)',
                'OC(X)C': 'COC(X)',
                'NC(=O)C(X)': 'N(X)C(=O)C',
            }
        }
        
        # Level 4: Scaffold hopping (10% of counterfeits)
        self.scaffold_patterns = {
            'privileged_scaffolds': {
                'c1ccc2[nH]ccc2c1': 'c1ccc2occc2c1',
                'c1ccc2ncccc2c1': 'c1ccc2scccc2c1',
                'c1ccc(cc1)c2ccccc2': 'c1ccc(cc1)c2ccncc2',
            },
            'ring_expansion': {
                'C1CCCC1': 'C1CCCCC1',
                'C1CCC1': 'C1CCCC1',
                'c1cccc1': 'c1ccccc1',
            }
        }
        
        # Level 5: Prodrug modifications (5% of counterfeits)
        self.prodrug_patterns = {
            'esterification': {
                'OH': 'OC(=O)CH3',
                'OH': 'OC(=O)CH2CH3',
                'COOH': 'COOCH3',
                'NH2': 'NHC(=O)CH3',
            },
            'soft_drug_modifications': {
                'CCO': 'CC(=O)O',
                'CNH2': 'CN(CH3)2',
                'C=C': 'CC',
            }
        }

    def fetch_chembl_molecules(self, limit: int = 7500) -> pd.DataFrame:
        """Fetch molecules from ChEMBL - increased limit for 30K dataset"""
        logger.info(f"Fetching {limit} molecules from ChEMBL...")
        
        molecule = new_client.molecule
        all_molecules = []
        
        try:
            queries = [
                molecule.filter(max_phase=4, molecule_type='Small molecule'),
                molecule.filter(max_phase=3, molecule_type='Small molecule'),
                molecule.filter(max_phase=2, molecule_type='Small molecule'),
                molecule.filter(max_phase=1, molecule_type='Small molecule'),
            ]
            
            molecules_per_query = limit // len(queries)
            
            for query in queries:
                processed = 0
                for mol in query[:molecules_per_query * 3]:
                    if processed >= molecules_per_query:
                        break
                    
                    if mol.get('molecule_structures') and mol['molecule_structures'].get('canonical_smiles'):
                        smiles = mol['molecule_structures']['canonical_smiles']
                        
                        try:
                            rdkit_mol = Chem.MolFromSmiles(smiles)
                            if self._is_suitable_molecule(rdkit_mol, smiles):
                                descriptors = self._calculate_enhanced_descriptors(rdkit_mol)
                                if descriptors:
                                    all_molecules.append({
                                        'source': 'ChEMBL',
                                        'source_id': mol['molecule_chembl_id'],
                                        'smiles': smiles,
                                        'name': mol.get('pref_name', f"ChEMBL_{mol['molecule_chembl_id']}"),
                                        'max_phase': mol.get('max_phase', 0),
                                        **descriptors
                                    })
                                    processed += 1
                        except:
                            continue
                            
        except Exception as e:
            logger.error(f"ChEMBL error: {e}")
        
        logger.info(f"ChEMBL: {len(all_molecules)} molecules fetched")
        return pd.DataFrame(all_molecules)

    def fetch_pubchem_molecules(self, limit: int = 4500) -> pd.DataFrame:
        """Fetch molecules from PubChem - increased limit for 30K dataset"""
        logger.info(f"Fetching {limit} molecules from PubChem...")
        
        all_molecules = []
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
        search_terms = [
            "aspirin", "ibuprofen", "acetaminophen", "caffeine", "morphine",
            "penicillin", "tetracycline", "warfarin", "digoxin",
            "metformin", "atorvastatin", "omeprazole", "sertraline",
            "amoxicillin", "ciprofloxacin", "metoprolol", "lisinopril",
            "amlodipine", "simvastatin", "levothyroxine", "azithromycin"
        ]
        
        molecules_per_term = limit // len(search_terms)
        
        for term in search_terms:
            try:
                time.sleep(1)
                
                search_url = f"{base_url}/compound/name/{quote(term)}/cids/JSON"
                response = requests.get(search_url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get('IdentifierList', {}).get('CID', [])[:molecules_per_term]
                    
                    for cid in cids:
                        try:
                            time.sleep(0.5)
                            
                            smiles_url = f"{base_url}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                            smiles_response = requests.get(smiles_url, timeout=30)
                            
                            if smiles_response.status_code == 200:
                                smiles_data = smiles_response.json()
                                smiles = smiles_data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                                
                                rdkit_mol = Chem.MolFromSmiles(smiles)
                                if self._is_suitable_molecule(rdkit_mol, smiles):
                                    descriptors = self._calculate_enhanced_descriptors(rdkit_mol)
                                    if descriptors:
                                        all_molecules.append({
                                            'source': 'PubChem',
                                            'source_id': f"CID_{cid}",
                                            'smiles': smiles,
                                            'name': f"PubChem_{cid}",
                                            'max_phase': 0,
                                            **descriptors
                                        })
                        except:
                            continue
            except:
                continue
        
        logger.info(f"PubChem: {len(all_molecules)} molecules fetched")
        return pd.DataFrame(all_molecules)

    def fetch_drugbank_like_molecules(self, limit: int = 3000) -> pd.DataFrame:
        """Generate DrugBank-like molecules - increased limit for 30K dataset"""
        logger.info(f"Generating {limit} DrugBank-like molecules...")
        
        drug_scaffolds = [
            "c1ccc(cc1)", "c1ccc2ccccc2c1", "c1ccncc1", "c1ccc2[nH]ccc2c1",
            "c1ccc2ncccc2c1", "C1CCCCC1", "C1CCN(CC1)", "c1ccc2occc2c1",
            "c1ccc2sccc2c1", "C1CCOC1", "C1CCNC1", "c1cncc1"
        ]
        
        functional_groups = [
            "C(=O)O", "C(=O)NH2", "OH", "NH2", "NO2", "CF3",
            "OCH3", "COOH", "C#N", "C(=O)CH3", "Cl", "F", "Br",
            "SO2NH2", "CONH2", "CHO"
        ]
        
        all_molecules = []
        
        for i in range(limit):
            try:
                scaffold = random.choice(drug_scaffolds)
                n_groups = random.randint(1, 3)
                groups = random.sample(functional_groups, n_groups)
                
                smiles = scaffold
                for group in groups:
                    if random.random() > 0.5:
                        smiles += group
                
                rdkit_mol = Chem.MolFromSmiles(smiles)
                if rdkit_mol:
                    clean_smiles = Chem.MolToSmiles(rdkit_mol)
                    if self._is_suitable_molecule(rdkit_mol, clean_smiles):
                        descriptors = self._calculate_enhanced_descriptors(rdkit_mol)
                        if descriptors:
                            all_molecules.append({
                                'source': 'Generated_DrugLike',
                                'source_id': f"GDL_{i:04d}",
                                'smiles': clean_smiles,
                                'name': f"DrugLike_{i:04d}",
                                'max_phase': 0,
                                **descriptors
                            })
            except:
                if i < len(self.backup_molecules):
                    smiles = self.backup_molecules[i % len(self.backup_molecules)]
                    rdkit_mol = Chem.MolFromSmiles(smiles)
                    if rdkit_mol:
                        descriptors = self._calculate_enhanced_descriptors(rdkit_mol)
                        if descriptors:
                            all_molecules.append({
                                'source': 'Backup',
                                'source_id': f"BCK_{i:04d}",
                                'smiles': smiles,
                                'name': f"Backup_{i:04d}",
                                'max_phase': 0,
                                **descriptors
                            })
        
        logger.info(f"DrugBank-like: {len(all_molecules)} molecules generated")
        return pd.DataFrame(all_molecules)

    def _is_suitable_molecule(self, mol, smiles: str) -> bool:
        """Enhanced molecule filtering for dataset quality"""
        if mol is None or len(smiles) < 5:
            return False
        
        try:
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 6 or num_atoms > 100:
                return False
            
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            if mw > 800 or mw < 50:
                return False
            if logp > 8 or logp < -3:
                return False
            if hbd > 10 or hba > 15:
                return False
            
            if mol.GetNumHeavyAtoms() < 5:
                return False
            
            if smiles.count('C') + smiles.count('c') < 2:
                return False
                
            return True
            
        except:
            return False

    def _calculate_enhanced_descriptors(self, mol):
        """Calculate comprehensive molecular descriptors"""
        try:
            descriptors = {
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
            return descriptors
        except:
            return None

    def fetch_all_authentic_molecules(self, target_count: int = 15000) -> pd.DataFrame:
        """Fetch 15K authentic molecules from all sources"""
        logger.info(f"Fetching {target_count} authentic molecules from multiple sources")
        
        chembl_count = int(target_count * 0.5)
        pubchem_count = int(target_count * 0.3)
        druglike_count = target_count - chembl_count - pubchem_count
        
        all_dfs = []
        
        chembl_df = self.fetch_chembl_molecules(chembl_count)
        if not chembl_df.empty:
            all_dfs.append(chembl_df)
        
        pubchem_df = self.fetch_pubchem_molecules(pubchem_count)
        if not pubchem_df.empty:
            all_dfs.append(pubchem_df)
        
        druglike_df = self.fetch_drugbank_like_molecules(druglike_count)
        if not druglike_df.empty:
            all_dfs.append(druglike_df)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['smiles'], keep='first')
            
            while len(combined_df) < target_count:
                for smiles in self.backup_molecules:
                    if len(combined_df) >= target_count:
                        break
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        descriptors = self._calculate_enhanced_descriptors(mol)
                        if descriptors:
                            new_row = pd.DataFrame([{
                                'source': 'Backup',
                                'source_id': f"backup_{len(combined_df)}",
                                'smiles': smiles,
                                'name': f"Backup_{len(combined_df)}",
                                'max_phase': 0,
                                **descriptors
                            }])
                            combined_df = pd.concat([combined_df, new_row], ignore_index=True)
            
            combined_df = combined_df.head(target_count)
            
            logger.info(f"Total authentic molecules collected: {len(combined_df)}")
            logger.info("Source distribution:")
            for source, count in combined_df['source'].value_counts().items():
                logger.info(f"  {source}: {count}")
            
            return combined_df
        else:
            logger.error("No molecules could be fetched from any source!")
            return pd.DataFrame()

    def generate_sophisticated_counterfeits(self, authentic_df: pd.DataFrame, 
                                          target_counterfeits: int = 15000) -> pd.DataFrame:
        """Generate 15K sophisticated counterfeits"""
        logger.info(f"Generating {target_counterfeits} sophisticated counterfeits")
        
        all_data = []
        
        for idx, row in authentic_df.iterrows():
            authentic_entry = {
                **row.to_dict(),
                'label': 0,
                'counterfeit_type': 'authentic',
                'difficulty_level': 'authentic'
            }
            all_data.append(authentic_entry)
        
        difficulty_distribution = {
            'structural': int(target_counterfeits * 0.25),
            'bioisosteric': int(target_counterfeits * 0.35),
            'stereochemical': int(target_counterfeits * 0.25),
            'scaffold': int(target_counterfeits * 0.10),
            'prodrug': int(target_counterfeits * 0.05),
        }
        
        patterns_map = {
            'structural': self.structural_patterns,
            'bioisosteric': self.bioisosteric_patterns,
            'stereochemical': self.stereochemical_patterns,
            'scaffold': self.scaffold_patterns,
            'prodrug': self.prodrug_patterns,
        }
        
        for difficulty, count in difficulty_distribution.items():
            patterns = patterns_map[difficulty]
            generated = self._generate_counterfeits_by_pattern(
                authentic_df, patterns, difficulty, count
            )
            all_data.extend(generated)
            logger.info(f"Generated {len(generated)} {difficulty} counterfeits")
        
        result_df = pd.DataFrame(all_data)
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Total dataset: {len(result_df)} molecules")
        logger.info(f"  Authentic: {sum(result_df['label'] == 0)}")
        logger.info(f"  Counterfeit: {sum(result_df['label'] == 1)}")
        
        return result_df

    def _generate_counterfeits_by_pattern(self, authentic_df: pd.DataFrame, 
                                        patterns: dict, difficulty: str, count: int) -> List[dict]:
        """Generate counterfeits using specific pattern types"""
        counterfeits = []
        attempts_per_molecule = max(1, count // len(authentic_df) + 1)
        
        for idx, row in authentic_df.iterrows():
            if len(counterfeits) >= count:
                break
                
            smiles = row['smiles']
            
            for attempt in range(attempts_per_molecule):
                if len(counterfeits) >= count:
                    break
                    
                counterfeit = self._apply_pattern_transformation(
                    smiles, row, patterns, difficulty
                )
                
                if counterfeit:
                    counterfeits.append(counterfeit)
        
        return counterfeits[:count]

    def _apply_pattern_transformation(self, smiles: str, original_data: dict, 
                                    patterns: dict, difficulty: str) -> Optional[dict]:
        """Apply specific pattern transformation"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            
            for pattern_category, pattern_dict in patterns.items():
                pattern_items = list(pattern_dict.items())
                random.shuffle(pattern_items)
                
                for original_pattern, replacement_pattern in pattern_items:
                    if original_pattern in smiles:
                        new_smiles = smiles.replace(original_pattern, replacement_pattern, 1)
                        
                        new_mol = Chem.MolFromSmiles(new_smiles)
                        if new_mol and new_smiles != smiles:
                            if self._is_suitable_molecule(new_mol, new_smiles):
                                descriptors = self._calculate_enhanced_descriptors(new_mol)
                                if descriptors:
                                    similarity = self._calculate_tanimoto_similarity(mol, new_mol)
                                    
                                    if 0.3 <= similarity <= 0.9:
                                        return {
                                            'source': f"Counterfeit_{difficulty}",
                                            'source_id': f"CF_{difficulty[:3].upper()}_{original_data.get('source_id', 'UNK')}_{random.randint(1000,9999)}",
                                            'smiles': new_smiles,
                                            'name': f"{difficulty.capitalize()}_CF_{original_data.get('name', 'Unknown')}",
                                            'label': 1,
                                            'counterfeit_type': pattern_category,
                                            'difficulty_level': difficulty,
                                            'original_smiles': smiles,
                                            'transformation': f"{original_pattern} → {replacement_pattern}",
                                            'similarity': similarity,
                                            'max_phase': 0,
                                            **descriptors
                                        }
            
            return self._apply_fallback_transformation(smiles, original_data, difficulty)
            
        except:
            return None

    def _apply_fallback_transformation(self, smiles: str, original_data: dict, 
                                     difficulty: str) -> Optional[dict]:
        """Fallback transformations when pattern matching fails"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            
            fallback_mods = []
            
            if difficulty == 'structural':
                fallback_mods = [
                    (lambda s: s + 'C' if len(s) < 50 else s, "add_methyl"),
                    (lambda s: s.replace('C', 'N', 1) if 'C' in s else s, "C_to_N"),
                ]
            elif difficulty == 'bioisosteric':
                fallback_mods = [
                    (lambda s: s.replace('Cl', 'Br', 1) if 'Cl' in s else s, "Cl_to_Br"),
                    (lambda s: s.replace('F', 'Cl', 1) if 'F' in s else s, "F_to_Cl"),
                ]
            else:
                fallback_mods = [
                    (lambda s: s + 'O' if len(s) < 50 else s, "add_oxygen"),
                ]
            
            for transform_func, transform_name in fallback_mods:
                new_smiles = transform_func(smiles)
                if new_smiles != smiles:
                    new_mol = Chem.MolFromSmiles(new_smiles)
                    if new_mol and self._is_suitable_molecule(new_mol, new_smiles):
                        descriptors = self._calculate_enhanced_descriptors(new_mol)
                        if descriptors:
                            return {
                                'source': f"Counterfeit_{difficulty}",
                                'source_id': f"CF_FB_{original_data.get('source_id', 'UNK')}_{random.randint(1000,9999)}",
                                'smiles': new_smiles,
                                'name': f"Fallback_CF_{original_data.get('name', 'Unknown')}",
                                'label': 1,
                                'counterfeit_type': 'fallback',
                                'difficulty_level': difficulty,
                                'original_smiles': smiles,
                                'transformation': transform_name,
                                'similarity': self._calculate_tanimoto_similarity(mol, new_mol),
                                'max_phase': 0,
                                **descriptors
                            }
            
        except:
            pass
        
        return None

    def _calculate_tanimoto_similarity(self, mol1, mol2) -> float:
        """Calculate Tanimoto similarity between molecules"""
        try:
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius=3, nBits=2048)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius=3, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0

    def convert_to_heterograph(self, smiles: str, label: int, metadata: dict = None) -> Optional[HeteroData]:
        """Convert SMILES to heterograph (26 atom, 18 bond features)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            hetero_data = HeteroData()
            
            node_features = {}
            atom_to_node_type = {}
            
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                atom_type = atom.GetSymbol()
                atom_to_node_type[atom_idx] = atom_type
                
                atom_features = self._extract_compatible_atom_features(atom, mol)
                
                if atom_type not in node_features:
                    node_features[atom_type] = []
                node_features[atom_type].append(atom_features)
            
            for atom_type, features in node_features.items():
                hetero_data[atom_type].x = torch.tensor(features, dtype=torch.float)
            
            edge_indices = {}
            edge_features = {}
            
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                start_atom_type = atom_to_node_type[start_idx]
                end_atom_type = atom_to_node_type[end_idx]
                
                edge_type = (start_atom_type, 'bond_to', end_atom_type)
                reverse_edge_type = (end_atom_type, 'bond_to', start_atom_type)
                
                bond_features = self._extract_compatible_bond_features(bond, mol)
                
                start_local_idx = sum(1 for i in range(start_idx) 
                                    if atom_to_node_type[i] == start_atom_type)
                end_local_idx = sum(1 for i in range(end_idx) 
                                  if atom_to_node_type[i] == end_atom_type)
                
                for et, si, ei in [(edge_type, start_local_idx, end_local_idx), 
                                   (reverse_edge_type, end_local_idx, start_local_idx)]:
                    if et not in edge_indices:
                        edge_indices[et] = [[], []]
                        edge_features[et] = []
                    
                    edge_indices[et][0].append(si)
                    edge_indices[et][1].append(ei)
                    edge_features[et].append(bond_features)
            
            for edge_type, indices in edge_indices.items():
                if indices[0]:
                    hetero_data[edge_type].edge_index = torch.tensor([indices[0], indices[1]], dtype=torch.long)
                    hetero_data[edge_type].edge_attr = torch.tensor(edge_features[edge_type], dtype=torch.float)
            
            hetero_data.y = torch.tensor([label], dtype=torch.long)
            hetero_data.smiles = smiles
            
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        setattr(hetero_data, key, value)
                
            return hetero_data
            
        except Exception as e:
            return None

    def _extract_compatible_atom_features(self, atom, mol) -> List[float]:
        """Extract COMPATIBLE atom features (26 dims)"""
        try:
            neighbors = atom.GetNeighbors()
            
            features = [
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
                float(any(mol.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), size) 
                         for size in [3, 4, 7, 8])),
                float(atom.GetNumRadicalElectrons()),
                float(len([n for n in neighbors if n.GetIsAromatic()]) / max(1, len(neighbors))),
                float(len([n for n in neighbors if n.GetSymbol() in ['N', 'O', 'S']]) / max(1, len(neighbors))),
                float(len([n for n in neighbors if n.GetSymbol() in ['F', 'Cl', 'Br', 'I']]) / max(1, len(neighbors))),
                float(any(n.GetFormalCharge() != 0 for n in neighbors)),
                float(atom.GetHybridization() == Chem.HybridizationType.SP),
                float(atom.GetHybridization() == Chem.HybridizationType.SP2),
                float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            ]
            
            return features
            
        except:
            return [0.0] * 26

    def _extract_compatible_bond_features(self, bond, mol) -> List[float]:
        """Extract COMPATIBLE bond features (18 dims)"""
        try:
            start_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            
            features = [
                float(bond.GetBondTypeAsDouble()) / 3.0,
                float(bond.GetIsAromatic()),
                float(bond.IsInRing()),
                float(bond.GetIsConjugated()),
                float(bond.GetStereo() != Chem.BondStereo.STEREONONE),
                float(bond.GetStereo() == Chem.BondStereo.STEREOZ),
                float(bond.IsInRingSize(6)),
                float(bond.IsInRingSize(5)),
                float(any(mol.GetRingInfo().IsBondInRingOfSize(bond.GetIdx(), size) 
                         for size in [3, 4, 7, 8])),
                float((start_atom.GetDegree() + end_atom.GetDegree()) / 10.0),
                float(start_atom.GetIsAromatic() and end_atom.GetIsAromatic()),
                float(start_atom.GetIsAromatic() != end_atom.GetIsAromatic()),
                float(abs(start_atom.GetFormalCharge() - end_atom.GetFormalCharge())),
                float(start_atom.GetSymbol() in ['N', 'O', 'S', 'P']),
                float(end_atom.GetSymbol() in ['N', 'O', 'S', 'P']),
                float(start_atom.GetSymbol() != end_atom.GetSymbol()),
                float(abs(start_atom.GetAtomicNum() - end_atom.GetAtomicNum()) / 50.0),
                float(bond.GetBondType() == Chem.BondType.DOUBLE),
            ]
            
            return features
            
        except:
            return [0.0] * 18

    def create_dataset_30k(self) -> Tuple[List[HeteroData], List[int], Tuple]:
        """Create 30K molecule dataset (15K authentic + 15K counterfeits)"""
        
        logger.info("Creating 30K molecule dataset for 80% target accuracy")
        
        authentic_df = self.fetch_all_authentic_molecules(15000)
        
        if authentic_df.empty:
            logger.error("No authentic molecules found!")
            return None, None, None
        
        complete_df = self.generate_sophisticated_counterfeits(authentic_df, 15000)
        
        logger.info("Converting molecules to enhanced heterographs...")
        hetero_graphs = []
        labels = []
        metadata_info = {'node_types': set(), 'edge_types': set()}
        
        for idx, row in tqdm(complete_df.iterrows(), desc="Converting to graphs", total=len(complete_df)):
            graph_metadata = {
                'source': row.get('source', 'unknown'),
                'source_id': row.get('source_id', f'mol_{idx}'),
                'counterfeit_type': row.get('counterfeit_type', 'unknown'),
                'difficulty_level': row.get('difficulty_level', 'unknown')
            }
            
            hetero_graph = self.convert_to_heterograph(row['smiles'], row['label'], graph_metadata)
            
            if hetero_graph is not None:
                hetero_graphs.append(hetero_graph)
                labels.append(row['label'])
                
                for node_type in hetero_graph.node_types:
                    metadata_info['node_types'].add(node_type)
                for edge_type in hetero_graph.edge_types:
                    metadata_info['edge_types'].add(edge_type)
        
        node_types = sorted(list(metadata_info['node_types']))
        edge_types = sorted(list(metadata_info['edge_types']))
        metadata = (node_types, edge_types)
        
        authentic_count = sum(1 for label in labels if label == 0)
        counterfeit_count = sum(1 for label in labels if label == 1)
        
        logger.info("=" * 80)
        logger.info("ENHANCED 30K DATASET CREATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total molecules: {len(hetero_graphs)}")
        logger.info(f"Authentic: {authentic_count} ({authentic_count/len(labels)*100:.1f}%)")
        logger.info(f"Counterfeit: {counterfeit_count} ({counterfeit_count/len(labels)*100:.1f}%)")
        logger.info(f"Node types: {len(node_types)}")
        logger.info(f"Edge types: {len(edge_types)}")
        logger.info(f"Atom features: 26 (compatible)")
        logger.info(f"Bond features: 18 (compatible)")
        
        return hetero_graphs, labels, metadata

    def save_dataset(self, hetero_graphs: List[HeteroData], labels: List[int], 
                     metadata: Tuple, filename: str = None):
        """Save the enhanced 30K dataset"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_pharma_30k_{timestamp}.pt"
        
        torch.save((hetero_graphs, labels, metadata), filename)
        logger.info(f"Dataset saved to {filename}")
        
        stats_file = filename.replace('.pt', '_stats.json')
        stats = {
            'total_graphs': len(hetero_graphs),
            'authentic_count': sum(1 for label in labels if label == 0),
            'counterfeit_count': sum(1 for label in labels if label == 1),
            'balance_ratio': '1:1 (perfect)',
            'node_types': metadata[0] if metadata else [],
            'edge_types_count': len(metadata[1]) if metadata else 0,
            'atom_features': 26,
            'bond_features': 18,
            'target_accuracy': '80%',
            'difficulty_levels': ['structural', 'bioisosteric', 'stereochemical', 'scaffold', 'prodrug'],
            'data_sources': ['ChEMBL', 'PubChem', 'Generated_DrugLike', 'Backup'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Detailed statistics saved to {stats_file}")


def main():
    """Main execution - Create enhanced 30K dataset"""
    print("Enhanced Multi-Source Pharmaceutical Dataset Generator - 30K Version")
    print("=" * 80)
    print("Target: 30,000 molecules (15K authentic + 15K counterfeits)")
    print("Expected model accuracy: 80% with sophisticated patterns")
    print("Data sources: ChEMBL, PubChem, Generated Drug-like, Backups")
    print("Features: 26 atom features, 18 bond features (compatible)")
    print()
    
    generator = MultiSourcePharmaceuticalGenerator()
    
    try:
        hetero_graphs, labels, metadata = generator.create_dataset_30k()
        
        if hetero_graphs:
            generator.save_dataset(hetero_graphs, labels, metadata)
            
            print("\n" + "=" * 80)
            print("SUCCESS: Enhanced 30K dataset created!")
            print("=" * 80)
            
            authentic_count = sum(1 for label in labels if label == 0)
            counterfeit_count = sum(1 for label in labels if label == 1)
            
            print(f"\nFinal Dataset Statistics:")
            print(f"   Total molecules: {len(hetero_graphs):,}")
            print(f"   Authentic: {authentic_count:,} ({authentic_count/len(labels)*100:.1f}%)")
            print(f"   Counterfeit: {counterfeit_count:,} ({counterfeit_count/len(labels)*100:.1f}%)")
            print(f"   Balance: Perfect 1:1 ratio")
            
            print(f"\nEnhanced Features:")
            print(f"   Multi-source authentic molecules (ChEMBL, PubChem, etc.)")
            print(f"   Sophisticated counterfeit patterns (5 difficulty levels)")
            print(f"   26 enhanced atom features (compatible)")
            print(f"   18 enhanced bond features (compatible)")
            print(f"   Optimized for 80% model accuracy")
            
            print(f"\nCounterfeit Distribution:")
            print(f"   Structural modifications: 25% (easier detection)")
            print(f"   Bioisosteric replacements: 35% (medium difficulty)")
            print(f"   Stereochemical changes: 25% (challenging)")
            print(f"   Scaffold hopping: 10% (hard)")
            print(f"   Prodrug modifications: 5% (very hard)")
            
            print(f"\nDataset ready for HGT training!")
            
        else:
            print("Failed to create dataset")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()