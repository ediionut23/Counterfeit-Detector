"""
Realistic ChEMBL Dataset Generator for Counterfeit Detection
Folosește strategii realiste de contrafăcere bazate pe date din industria farmaceutică
Author: Claude + ediionut23
Version: 7.0.0 - Ultra-Realistic Counterfeits
"""

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import torch
from torch_geometric.data import HeteroData
from datetime import datetime, timezone
import logging
from tqdm.auto import tqdm
import warnings
from rdkit import RDLogger
from sklearn.utils.class_weight import compute_class_weight
import random

# Import realistic generator
from realistic_counterfeit_generator import RealisticCounterfeitGenerator

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

class RealisticChEMBLDataLoader:
    def __init__(self):
        self.molecule = new_client.molecule
        self.activity = new_client.activity
        self.logger = logging.getLogger(__name__)
        self.session_start = datetime.now(timezone.utc)
        self.counterfeit_generator = RealisticCounterfeitGenerator()
        self.logger.info(f"Realistic dataset generation session started at {self.session_start}")

    def calculate_molecular_descriptors(self, mol):
        """Calculează descriptorii moleculari avansați"""
        try:
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42345)
            
            descriptors = {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'alogp': Descriptors.MolLogP(mol),
                'psa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'rings': rdMolDescriptors.CalcNumRings(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'fraction_sp3': Descriptors.FractionCSP3(mol),
                'qed': Descriptors.qed(mol),
                'bertz': Descriptors.BertzCT(mol),
                'balaban_j': Descriptors.BalabanJ(mol),
                'chi0v': Descriptors.Chi0v(mol),
                'chi1v': Descriptors.Chi1v(mol),
                'chi2v': Descriptors.Chi2v(mol),
                'chi3v': Descriptors.Chi3v(mol),
                'chi4v': Descriptors.Chi4v(mol),
                'hall_kier_alpha': Descriptors.HallKierAlpha(mol),
                'kappa1': Descriptors.Kappa1(mol),
                'kappa2': Descriptors.Kappa2(mol),
                'kappa3': Descriptors.Kappa3(mol),
                # Noi descriptori pentru detectarea contrafăcerilor
                'formal_charge': sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
                'num_chiral_centers': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
                'num_heterocycles': rdMolDescriptors.CalcNumHeterocycles(mol),
                'num_aliphatic_carbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
                'num_saturated_carbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
                'num_aromatic_carbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
                'largest_ring_size': max([len(ring) for ring in mol.GetRingInfo().AtomRings()]) if mol.GetRingInfo().AtomRings() else 0
            }
            
            # Calculează volumul molecular dacă există conformații
            try:
                if mol.GetNumConformers() > 0:
                    descriptors['molecular_volume'] = AllChem.ComputeMolVolume(mol)
                else:
                    descriptors['molecular_volume'] = 0.0
            except:
                descriptors['molecular_volume'] = 0.0
                
            return descriptors
            
        except Exception as e:
            self.logger.warning(f"Error calculating descriptors: {e}")
            return None

    def extract_advanced_atom_features(self, atom, mol):
        """Extrage feature-uri avansate pentru atomi cu focus pe detectarea contrafăcerilor"""
        try:
            # Feature-uri de bază
            basic_features = [
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetMass(),
                int(atom.GetChiralTag()),
                atom.GetTotalValence(),
                atom.GetNumRadicalElectrons()
            ]
            
            # Feature-uri topologice
            topology_features = [
                atom.GetExplicitValence(),
                len(atom.GetNeighbors()),
                atom.GetImplicitValence(),
                int(any(len(ring) == 3 for ring in mol.GetRingInfo().AtomRings() if atom.GetIdx() in ring)),
                int(any(len(ring) == 4 for ring in mol.GetRingInfo().AtomRings() if atom.GetIdx() in ring)),
                int(any(len(ring) == 5 for ring in mol.GetRingInfo().AtomRings() if atom.GetIdx() in ring)),
                int(any(len(ring) == 6 for ring in mol.GetRingInfo().AtomRings() if atom.GetIdx() in ring)),
                int(any(len(ring) == 7 for ring in mol.GetRingInfo().AtomRings() if atom.GetIdx() in ring)),
                atom.GetNumImplicitHs(),
                atom.GetNumExplicitHs()
            ]
            
            # Feature-uri pentru detectarea contrafăcerilor
            counterfeit_detection_features = [
                int(atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']),  # Halogeni (des schimbați)
                int(atom.GetSymbol() in ['N', 'O', 'S', 'P']),    # Heteroatomi (ținte frecvente)
                int(atom.IsInRing() and atom.GetIsAromatic()),    # Aromatic și în inel
                sum([n.GetAtomicNum() for n in atom.GetNeighbors()]),  # Suma numerelor atomice vecini
                len([n for n in atom.GetNeighbors() if n.GetIsAromatic()]),  # Vecini aromatici
                int(atom.HasProp('_ChiralityPossible')),          # Poate fi chiral
                int(any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds())),  # Are legături duble
                int(any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in atom.GetBonds())), # Are legături triple
                atom.GetTotalDegree(),                            # Gradul total
                int(atom.HasQuery())                              # Are query
            ]
            
            # Feature-uri moleculare globale normalizate
            try:
                global_features = [
                    rdMolDescriptors.CalcCrippenDescriptors(mol)[0] / mol.GetNumAtoms(),  # LogP per atom
                    rdMolDescriptors.CalcTPSA(mol) / mol.GetNumAtoms(),                   # PSA per atom
                    float(atom.GetAtomicNum()) / 118.0,                                   # Numărul atomic normalizat
                    float(len([a for a in mol.GetAtoms() if a.GetSymbol() == atom.GetSymbol()])) / mol.GetNumAtoms(),  # Frecvența tipului de atom
                ]
            except:
                global_features = [0.0, 0.0, 0.0, 0.0]
            
            return basic_features + topology_features + counterfeit_detection_features + global_features
            
        except Exception as e:
            self.logger.warning(f"Error extracting atom features: {e}")
            return [0] * 34  # Updated count

    def _extract_advanced_bond_features(self, bond, mol):
        """Extrage feature-uri avansate pentru legături cu focus pe contrafăceri"""
        try:
            start_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            
            # Feature-uri de bază
            basic_features = [
                float(bond.GetBondTypeAsDouble()),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
                int(bond.GetStereo())
            ]
            
            # Feature-uri pentru inele
            topology_features = [
                int(bond.IsInRingSize(3)),
                int(bond.IsInRingSize(4)),
                int(bond.IsInRingSize(5)),
                int(bond.IsInRingSize(6)),
                int(bond.IsInRingSize(7))
            ]
            
            # Context atomic
            atom_context_features = [
                start_atom.GetDegree(),
                end_atom.GetDegree(),
                start_atom.GetTotalValence(),
                end_atom.GetTotalValence(),
                int(start_atom.GetIsAromatic()),
                int(end_atom.GetIsAromatic()),
                int(start_atom.GetSymbol() != end_atom.GetSymbol()),  # Atomii sunt diferiți
                abs(start_atom.GetFormalCharge() - end_atom.GetFormalCharge())  # Diferența de sarcină
            ]
            
            # Feature-uri pentru detectarea contrafăcerilor
            counterfeit_features = [
                float(bond.GetBondDir()),
                int(bond.HasQuery()),
                int(start_atom.IsInRing() and end_atom.IsInRing()),
                int((start_atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']) or (end_atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'])),  # Conține halogen
                int((start_atom.GetSymbol() in ['N', 'O', 'S', 'P']) or (end_atom.GetSymbol() in ['N', 'O', 'S', 'P']))     # Conține heteroatom
            ]
            
            # Lungimea legăturii (dacă există conformații)
            try:
                if mol.GetNumConformers() > 0:
                    bond_length = AllChem.GetBondLength(mol.GetConformer(),
                                                        bond.GetBeginAtomIdx(),
                                                        bond.GetEndAtomIdx())
                else:
                    bond_length = 0.0
            except:
                bond_length = 0.0
            
            return basic_features + topology_features + atom_context_features + counterfeit_features + [bond_length]
            
        except Exception as e:
            self.logger.warning(f"Error extracting bond features: {e}")
            return [0] * 24  # Updated count

    def fetch_drug_molecules(self, max_phase=4, limit=1000):
        """Descarcă molecule de medicamente de la ChEMBL cu filtrare balansată"""
        self.logger.info(f"Fetching {limit} high-quality drugs from ChEMBL (Phase {max_phase})...")
        
        try:
            # Balanced filtering - stricter than 0.1 QED but more permissive than original
            molecules = self.molecule.filter(
                max_phase=max_phase,
                molecule_type='Small molecule',
                chirality__in=[0, 1, 2]  # Reintroduce chirality but include all types
            )[:limit * 3]  # Fetch 3x more molecules to account for filtering
            
            data = []
            processed_smiles = set()  # Evită duplicatele
            
            for mol in tqdm(molecules, desc="Processing molecules"):
                # Remove early break - process all available molecules
                    
                if (mol['molecule_structures'] and 
                    mol['molecule_structures']['canonical_smiles']):
                    
                    smiles = mol['molecule_structures']['canonical_smiles']
                    
                    # Evită duplicatele
                    if smiles in processed_smiles:
                        continue
                        
                    try:
                        rdkit_mol = Chem.MolFromSmiles(smiles)
                        if rdkit_mol is None:
                            continue
                            
                        # Balanced filtering - between strict and permissive
                        if (rdkit_mol.GetNumAtoms() < 5 or   # Back to 5 atoms minimum
                            rdkit_mol.GetNumAtoms() > 120 or  # Reduced from 150 to 120
                            len(smiles) < 5):
                            continue
                            
                        descriptors = self.calculate_molecular_descriptors(rdkit_mol)
                        if descriptors is None:
                            continue
                            
                        # Balanced QED filter - between strict 0.3 and permissive 0.1
                        if descriptors.get('qed', 0) < 0.2:  # Compromise: 0.2 instead of 0.1 or 0.3
                            continue
                            
                        data.append({
                            'chembl_id': mol['molecule_chembl_id'],
                            'smiles': smiles,
                            'pref_name': mol.get('pref_name', 'Unknown'),
                            'max_phase': mol.get('max_phase', 0),
                            **descriptors,
                            'label': 'authentic'
                        })
                        processed_smiles.add(smiles)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing molecule {mol['molecule_chembl_id']}: {e}")
                        continue
                        
            self.logger.info(f"Successfully processed {len(data)} high-quality molecules")
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error fetching molecules: {e}")
            return pd.DataFrame()

    def generate_realistic_counterfeits(self, authentic_df, counterfeits_per_authentic=1):
        """
        Generează contrafăceri realiste folosind strategiile industriei farmaceutice
        """
        self.logger.info(f"Generating {counterfeits_per_authentic} realistic counterfeit(s) per authentic...")
        
        # Statistici pentru strategii
        strategy_stats = {
            'mild': 0,
            'stereochemistry': 0,
            'side_chain': 0,
            'salt_form': 0,
            'failed': 0
        }
        
        authentic_data = []
        counterfeit_data = []
        
        # Procesează moleculele autentice
        for idx, row in tqdm(authentic_df.iterrows(), desc="Creating realistic dataset"):
            # Adaugă molecula autentică
            authentic_entry = {
                'chembl_id': row['chembl_id'],
                'smiles': row['smiles'],
                'label': 0,  # 0 = authentic
                'pref_name': row.get('pref_name', 'Unknown'),
                'type': 'authentic',
                'original_idx': idx
            }
            
            # Copiază descriptorii
            for col in row.index:
                if col not in ['label', 'chembl_id', 'smiles', 'pref_name']:
                    authentic_entry[col] = row[col]
                    
            authentic_data.append(authentic_entry)
            
            # Generează contrafăceri pentru această moleculă
            smiles = row['smiles']
            counterfeits_created = 0
            attempts = 0
            max_attempts = 100  # Increased for better success rate
            
            while counterfeits_created < counterfeits_per_authentic and attempts < max_attempts:
                attempts += 1
                
                # Folosește generatorul realist
                counterfeit_mol, strategy = self._generate_single_realistic_counterfeit(smiles)
                
                # If realistic generation fails after many attempts, try simpler approach
                if counterfeit_mol is None and attempts > 70:  # Increased threshold for quality
                    counterfeit_mol, strategy = self._generate_simple_counterfeit(smiles)
                
                if counterfeit_mol:
                    counterfeit_smiles = Chem.MolToSmiles(counterfeit_mol)
                    
                    # Verifică că este diferită de originală
                    if counterfeit_smiles != smiles:
                        # Calculează descriptorii pentru contrafăcere
                        counterfeit_descriptors = self.calculate_molecular_descriptors(counterfeit_mol)
                        
                        if counterfeit_descriptors:
                            counterfeit_entry = {
                                'chembl_id': f"FAKE_{row['chembl_id']}_{counterfeits_created}",
                                'smiles': counterfeit_smiles,
                                'label': 1,  # 1 = counterfeit
                                'pref_name': f"Counterfeit_{row.get('pref_name', 'Unknown')}",
                                'type': 'counterfeit',
                                'original_idx': idx,
                                'original_smiles': smiles,
                                'counterfeit_strategy': strategy,
                                **counterfeit_descriptors
                            }
                            counterfeit_data.append(counterfeit_entry)
                            counterfeits_created += 1
                            strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1
                else:
                    strategy_stats['failed'] += 1
            
            if counterfeits_created == 0:
                self.logger.warning(f"Could not generate counterfeit for {row['chembl_id']}")
        
        # Log statisticile strategiilor
        self.logger.info("Counterfeit generation statistics:")
        for strategy, count in strategy_stats.items():
            self.logger.info(f"  {strategy}: {count}")
        
        self.logger.info(f"Generated {len(authentic_data)} authentic and {len(counterfeit_data)} counterfeit molecules")
        return pd.DataFrame(authentic_data + counterfeit_data)

    def _generate_single_realistic_counterfeit(self, original_smiles):
        """Generează o singură contrafăcere realistă"""
        strategies = ['mild', 'stereochemistry', 'side_chain', 'salt_form']
        strategy_weights = [0.4, 0.3, 0.2, 0.1]  # Probabilități bazate pe realitate
        
        # Alege strategia
        strategy = np.random.choice(strategies, p=strategy_weights)
        
        # Parametri pentru fiecare strategie
        if strategy == 'mild':
            mutation_rate = np.random.uniform(0.02, 0.06)  # 2-6% mutații
        elif strategy == 'stereochemistry':
            mutation_rate = 0.05
        elif strategy == 'side_chain':
            mutation_rate = np.random.uniform(0.03, 0.08)  # 3-8% mutații
        else:  # salt_form
            mutation_rate = 0.05
        
        try:
            counterfeit_mol = self.counterfeit_generator.generate_realistic_counterfeit(
                original_smiles, strategy, mutation_rate
            )
            
            if counterfeit_mol:
                # Validarea finală
                original_mol = Chem.MolFromSmiles(original_smiles)
                if self.counterfeit_generator.validate_counterfeit_quality(original_mol, counterfeit_mol):
                    return counterfeit_mol, strategy
                    
        except Exception as e:
            self.logger.warning(f"Error in realistic counterfeit generation: {e}")
            
        return None, 'failed'
    
    def _generate_simple_counterfeit(self, original_smiles):
        """
        Fallback method: generate simpler counterfeits when realistic methods fail
        """
        try:
            mol = Chem.MolFromSmiles(original_smiles)
            if mol is None:
                return None, 'failed'
            
            # Simple halogen substitution (most reliable) - but validate quality
            editable_mol = Chem.EditableMol(mol)
            halogens = ['F', 'Cl', 'Br']
            
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() in halogens and random.random() < 0.2:  # Reduced probability for quality
                    # Replace with different halogen
                    new_halogen = random.choice([h for h in halogens if h != atom.GetSymbol()])
                    new_atom = Chem.Atom(new_halogen)
                    editable_mol.ReplaceAtom(atom_idx, new_atom)
                    
                    result_mol = editable_mol.GetMol()
                    if result_mol:
                        try:
                            Chem.SanitizeMol(result_mol)
                            # Validate quality even for simple counterfeits
                            original_fp = AllChem.GetMorganFingerprint(mol, 2)
                            counterfeit_fp = AllChem.GetMorganFingerprint(result_mol, 2)
                            similarity = Chem.DataStructs.TanimotoSimilarity(original_fp, counterfeit_fp)
                            
                            if 0.7 <= similarity <= 0.95:  # Still require reasonable similarity
                                return result_mol, 'simple_halogen'
                        except:
                            continue
            
            # If no halogens, try simple atom substitution (more conservative)
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                # Only substitute terminal carbons to avoid breaking core structure
                if (atom.GetSymbol() == 'C' and not atom.GetIsAromatic() and 
                    atom.GetDegree() <= 2 and random.random() < 0.05):  # Very conservative
                    
                    # Replace C with N occasionally
                    new_atom = Chem.Atom('N')
                    editable_mol.ReplaceAtom(atom_idx, new_atom)
                    
                    result_mol = editable_mol.GetMol()
                    if result_mol:
                        try:
                            Chem.SanitizeMol(result_mol)
                            # Validate quality for C->N substitution too
                            original_fp = AllChem.GetMorganFingerprint(mol, 2)
                            counterfeit_fp = AllChem.GetMorganFingerprint(result_mol, 2)
                            similarity = Chem.DataStructs.TanimotoSimilarity(original_fp, counterfeit_fp)
                            
                            if 0.75 <= similarity <= 0.95:  # Stricter for C->N
                                return result_mol, 'simple_substitution'
                        except:
                            continue
                        
            return None, 'failed'
            
        except Exception as e:
            self.logger.warning(f"Error in simple counterfeit generation: {e}")
            return None, 'failed'

    def convert_to_heterograph(self, smiles, label, chembl_id=None):
        """Convertește SMILES în heterograph cu feature-uri îmbunătățite"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # Creează heterograph
            hetero_data = HeteroData()
            
            # Node features și edge features cu metoda existentă dar îmbunătățită
            node_features = {}
            edge_indices = {}
            edge_features = {}
            
            # Extrage feature-urile pentru fiecare tip de atom
            for atom in mol.GetAtoms():
                atom_type = atom.GetSymbol()
                atom_features = self.extract_advanced_atom_features(atom, mol)
                
                if atom_type not in node_features:
                    node_features[atom_type] = []
                node_features[atom_type].append(atom_features)
            
            # Convertește la tensori PyTorch
            for atom_type, features in node_features.items():
                hetero_data[atom_type].x = torch.tensor(features, dtype=torch.float)
            
            # Extrage edge features pentru fiecare tip de legătură
            edge_types_dict = {}
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                start_atom_type = mol.GetAtomWithIdx(start_idx).GetSymbol()
                end_atom_type = mol.GetAtomWithIdx(end_idx).GetSymbol()
                
                # Creează numele tipului de legătură
                edge_type = (start_atom_type, 'bond_to', end_atom_type)
                reverse_edge_type = (end_atom_type, 'bond_to', start_atom_type)
                
                bond_features = self._extract_advanced_bond_features(bond, mol)
                
                # Mapare index-uri locale pentru tipul de atom
                if start_atom_type not in edge_types_dict:
                    edge_types_dict[start_atom_type] = {}
                if end_atom_type not in edge_types_dict:
                    edge_types_dict[end_atom_type] = {}
                    
                # Calculează index-urile locale
                start_local_idx = sum(1 for i, a in enumerate(mol.GetAtoms()) if i < start_idx and a.GetSymbol() == start_atom_type)
                end_local_idx = sum(1 for i, a in enumerate(mol.GetAtoms()) if i < end_idx and a.GetSymbol() == end_atom_type)
                
                # Adaugă edge-urile (bidireccionale)
                if edge_type not in edge_indices:
                    edge_indices[edge_type] = [[], []]
                    edge_features[edge_type] = []
                    
                if reverse_edge_type not in edge_indices:
                    edge_indices[reverse_edge_type] = [[], []]
                    edge_features[reverse_edge_type] = []
                
                edge_indices[edge_type][0].append(start_local_idx)
                edge_indices[edge_type][1].append(end_local_idx)
                edge_features[edge_type].append(bond_features)
                
                edge_indices[reverse_edge_type][0].append(end_local_idx)
                edge_indices[reverse_edge_type][1].append(start_local_idx)
                edge_features[reverse_edge_type].append(bond_features)
            
            # Convertește la tensori
            for edge_type, indices in edge_indices.items():
                if indices[0]:  # Dacă există edge-uri
                    hetero_data[edge_type].edge_index = torch.tensor([indices[0], indices[1]], dtype=torch.long)
                    hetero_data[edge_type].edge_attr = torch.tensor(edge_features[edge_type], dtype=torch.float)
            
            # Adaugă label-ul și metadata
            hetero_data.y = torch.tensor([label], dtype=torch.long)
            hetero_data.smiles = smiles
            if chembl_id:
                hetero_data.chembl_id = chembl_id
                
            return hetero_data
            
        except Exception as e:
            self.logger.warning(f"Error converting to heterograph: {e}")
            return None

    def create_realistic_dataset(self, limit=1000, counterfeits_per_authentic=1, test_ratio=0.2):
        """
        Creează un dataset complet cu contrafăceri realiste
        """
        self.logger.info("🚀 Starting realistic counterfeit dataset creation...")
        
        # 1. Descarcă molecule autentice
        authentic_df = self.fetch_drug_molecules(max_phase=4, limit=limit)
        if authentic_df.empty:
            self.logger.error("No authentic molecules found!")
            return None, None, None
            
        # 2. Generează contrafăceri realiste
        balanced_df = self.generate_realistic_counterfeits(authentic_df, counterfeits_per_authentic)
        
        # 3. Convertește în heterograph-uri
        self.logger.info("Converting molecules to heterographs...")
        hetero_graphs = []
        labels = []
        metadata_info = {'node_types': set(), 'edge_types': set()}
        
        for idx, row in tqdm(balanced_df.iterrows(), desc="Converting to graphs"):
            hetero_graph = self.convert_to_heterograph(
                row['smiles'], 
                row['label'], 
                row.get('chembl_id', f'mol_{idx}')
            )
            
            if hetero_graph is not None:
                hetero_graphs.append(hetero_graph)
                labels.append(row['label'])
                
                # Collect metadata
                for node_type in hetero_graph.node_types:
                    metadata_info['node_types'].add(node_type)
                for edge_type in hetero_graph.edge_types:
                    metadata_info['edge_types'].add(edge_type)
        
        # 4. Creează metadata
        node_types = sorted(list(metadata_info['node_types']))
        edge_types = sorted(list(metadata_info['edge_types']))
        metadata = (node_types, edge_types)
        
        # 5. Statistici finale
        authentic_count = sum(1 for label in labels if label == 0)
        counterfeit_count = sum(1 for label in labels if label == 1)
        
        self.logger.info(f"✅ Dataset created successfully!")
        self.logger.info(f"   📊 Total molecules: {len(hetero_graphs)}")
        self.logger.info(f"   ✅ Authentic: {authentic_count}")
        self.logger.info(f"   🚨 Counterfeit: {counterfeit_count}")
        if authentic_count > 0:
            self.logger.info(f"   📈 Balance ratio: {counterfeit_count/authentic_count:.2f}")
        else:
            self.logger.info(f"   📈 Balance ratio: N/A (no authentic molecules)")
        self.logger.info(f"   🧬 Node types: {len(node_types)}")
        self.logger.info(f"   🔗 Edge types: {len(edge_types)}")
        
        return hetero_graphs, labels, metadata

# Exemplu de utilizare
if __name__ == "__main__":
    # Creează dataset-ul realist
    loader = RealisticChEMBLDataLoader()
    
    print("Creating realistic counterfeit detection dataset...")
    
    hetero_graphs, labels, metadata = loader.create_realistic_dataset(
        limit=8000,  # Increased to 8000 molecules
        counterfeits_per_authentic=1,  # 1 counterfeit per authentic for balance
        test_ratio=0.2
    )
    
    if hetero_graphs:
        # Salvează dataset-ul
        dataset_path = "realistic_counterfeit_dataset_8k_balanced.pt"
        torch.save((hetero_graphs, labels, metadata), dataset_path)
        print(f"Realistic dataset saved to: {dataset_path}")
        
        # Afișează statistici
        print("\nDataset Statistics:")
        print(f"Total graphs: {len(hetero_graphs)}")
        print(f"Labels distribution: {np.bincount(labels)}")
        print(f"Node types: {metadata[0]}")
        print(f"Edge types count: {len(metadata[1])}")
        
    else:
        print("❌ Failed to create dataset")