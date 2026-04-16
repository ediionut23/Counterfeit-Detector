"""
ChEMBL Heterogeneous Graph Loader pentru Counterfeit Detection - IMPROVED VERSION
Author: ediionut23
Version: 6.0.0 - Balanced Dataset with Better Counterfeits
Last Updated: 2025-08-29 
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

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

class ImprovedChEMBLDataLoader:
    def __init__(self):
        self.molecule = new_client.molecule
        self.activity = new_client.activity
        self.logger = logging.getLogger(__name__)
        self.session_start = datetime.now(timezone.utc)
        self.logger.info(f"Session started at {self.session_start}")

    def calculate_molecular_descriptors(self, mol):
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
                'sasa': AllChem.ComputeMolVolume(mol) if mol.GetNumConformers() > 0 else 0.0,
                'chi0v': Descriptors.Chi0v(mol),
                'chi1v': Descriptors.Chi1v(mol),
                'chi2v': Descriptors.Chi2v(mol),
                'chi3v': Descriptors.Chi3v(mol),
                'chi4v': Descriptors.Chi4v(mol),
                'hall_kier_alpha': Descriptors.HallKierAlpha(mol),
                'kappa1': Descriptors.Kappa1(mol),
                'kappa2': Descriptors.Kappa2(mol),
                'kappa3': Descriptors.Kappa3(mol)
            }
            return descriptors
        except Exception as e:
            self.logger.warning(f"Error calculating descriptors: {e}")
            return None

    def extract_advanced_atom_features(self, atom, mol):
        try:
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
            connectivity_features = [
                int(atom.IsInRing() and atom.GetIsAromatic()),
                atom.GetTotalDegree(),
                int(atom.HasQuery()),
                sum([n.GetAtomicNum() for n in atom.GetNeighbors()]),
                len([n for n in atom.GetNeighbors() if n.GetIsAromatic()]),
                int(atom.HasProp('_ChiralityPossible'))
            ]
            advanced_features = [
                rdMolDescriptors.CalcCrippenDescriptors(mol)[0] / mol.GetNumAtoms(),
                rdMolDescriptors.CalcTPSA(mol) / mol.GetNumAtoms(),
                int(atom.GetSymbol() in ['N', 'O', 'F', 'Cl', 'Br', 'I']),
                int(any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds())),
                int(any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in atom.GetBonds()))
            ]
            return basic_features + topology_features + connectivity_features + advanced_features
        except Exception as e:
            self.logger.warning(f"Error extracting atom features: {e}")
            return [0] * 31

    def _extract_advanced_bond_features(self, bond, mol):
        try:
            start_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            basic_features = [
                float(bond.GetBondTypeAsDouble()),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
                int(bond.GetStereo())
            ]
            topology_features = [
                int(bond.IsInRingSize(3)),
                int(bond.IsInRingSize(4)),
                int(bond.IsInRingSize(5)),
                int(bond.IsInRingSize(6)),
                int(bond.IsInRingSize(7))
            ]
            atom_context_features = [
                start_atom.GetDegree(),
                end_atom.GetDegree(),
                start_atom.GetTotalValence(),
                end_atom.GetTotalValence(),
                int(start_atom.GetIsAromatic()),
                int(end_atom.GetIsAromatic())
            ]
            advanced_features = [
                float(bond.GetBondDir()),
                int(bond.HasQuery()),
                int(start_atom.IsInRing() and end_atom.IsInRing()),
                abs(start_atom.GetFormalCharge() - end_atom.GetFormalCharge())
            ]
            if mol.GetNumConformers() > 0:
                bond_length = AllChem.GetBondLength(mol.GetConformer(),
                                                    bond.GetBeginAtomIdx(),
                                                    bond.GetEndAtomIdx())
            else:
                bond_length = 0.0
            return basic_features + topology_features + atom_context_features + \
                   advanced_features + [bond_length]
        except Exception as e:
            self.logger.warning(f"Error extracting bond features: {e}")
            return [0] * 21

    def validate_counterfeit_quality(self, original_mol, fake_mol, min_difference=0.02, max_difference=0.5):
        """
        Validează că molecula contrafăcută este suficient de diferită de originală
        """
        try:
            orig_desc = self.calculate_molecular_descriptors(original_mol)
            fake_desc = self.calculate_molecular_descriptors(fake_mol)
            
            if not orig_desc or not fake_desc:
                return False
            
            # Calculează diferențele relative pentru descriptorii cheie
            key_descriptors = ['molecular_weight', 'alogp', 'psa', 'hbd', 'hba', 'qed']
            differences = []
            
            for key in key_descriptors:
                if key in orig_desc and key in fake_desc:
                    orig_val = orig_desc[key]
                    fake_val = fake_desc[key]
                    
                    if abs(orig_val) > 1e-6:  # Evită împărțirea la 0
                        diff = abs(orig_val - fake_val) / abs(orig_val)
                        differences.append(diff)
            
            if not differences:
                return False
                
            avg_difference = np.mean(differences)
            max_difference = np.max(differences)
            
            # Criterii mai realiste: diferență moderată (5-30%)
            return min_difference <= avg_difference <= max_difference
            
        except Exception as e:
            self.logger.warning(f"Error validating counterfeit quality: {e}")
            return False

    def _apply_aggressive_mutations(self, mol, strategy, rate):
        """
        Aplică mutații mai agresive pentru a crea contrafăceri mai distincte
        """
        try:
            editable_mol = Chem.EditableMol(mol)
            atoms_to_change = []
            
            # Strategii îmbunătățite cu mutații mai agresive
            if strategy == 'aggressive_substitution':
                substitutions = {
                    'C': ['N', 'O', 'S', 'P'],
                    'N': ['C', 'O', 'S', 'P'], 
                    'O': ['N', 'S', 'C'],
                    'S': ['O', 'N', 'P'],
                    'P': ['N', 'S', 'C']
                }
                mutation_rate = rate * 2.0  # Rate mai mari
                
            elif strategy == 'halogen_swap':
                substitutions = {
                    'F': ['Cl', 'Br', 'I'],
                    'Cl': ['F', 'Br', 'I'],
                    'Br': ['F', 'Cl', 'I'],
                    'I': ['F', 'Cl', 'Br'],
                    'H': ['F', 'Cl']  # Adaugă halogeni în locul H
                }
                mutation_rate = rate * 1.5
                
            elif strategy == 'ring_disruption':
                # Modifică atomii din inele aromate (mai riscant dar mai distinct)
                substitutions = {
                    'C': ['N', 'O'],
                    'N': ['C', 'O']
                }
                mutation_rate = rate * 1.2
                
            elif strategy == 'functional_transformation':
                # Schimbări în grupele funcționale
                substitutions = {
                    'O': ['N', 'S'],
                    'N': ['O', 'S'],
                    'S': ['O', 'N']
                }
                mutation_rate = rate * 1.8
                
            else:
                # Default fallback
                substitutions = {
                    'C': ['N', 'O'],
                    'N': ['C', 'O'],
                    'O': ['N', 'S']
                }
                mutation_rate = rate
            
            # Aplică mutațiile
            for atom_idx in range(mol.GetNumAtoms()):
                if np.random.random() < mutation_rate:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    symbol = atom.GetSymbol()
                    
                    # Pentru ring_disruption, țintește doar atomii din inele
                    if strategy == 'ring_disruption' and not atom.IsInRing():
                        continue
                    
                    if symbol in substitutions:
                        possible_substitutes = substitutions[symbol]
                        
                        # Filtrare contextuală
                        if atom.IsInRing() and strategy != 'ring_disruption':
                            possible_substitutes = [s for s in possible_substitutes if s in ['C', 'N', 'O']]
                        
                        if possible_substitutes:
                            new_symbol = np.random.choice(possible_substitutes)
                            atoms_to_change.append((atom_idx, new_symbol))
            
            # Aplică schimbările
            for atom_idx, new_symbol in atoms_to_change:
                new_atom = Chem.Atom(new_symbol)
                old_atom = mol.GetAtomWithIdx(atom_idx)
                
                # Păstrează unele proprietăți pentru stabilitate
                new_atom.SetFormalCharge(0)  # Reset charge pentru siguranță
                editable_mol.ReplaceAtom(atom_idx, new_atom)
            
            try:
                mutated_mol = editable_mol.GetMol()
                Chem.SanitizeMol(mutated_mol)
                return mutated_mol
            except:
                return None
                
        except Exception as e:
            self.logger.warning(f"Error in aggressive mutation: {e}")
            return None

    def fetch_drug_molecules(self, max_phase=4, limit=1000):
        self.logger.info(f"Fetching {limit} approved drugs from ChEMBL...")
        try:
            molecules = self.molecule.filter(
                max_phase=max_phase,
                molecule_type='Small molecule'
            )[:limit]
            data = []
            for mol in tqdm(molecules, desc="Processing molecules"):
                if mol['molecule_structures'] and mol['molecule_structures']['canonical_smiles']:
                    try:
                        rdkit_mol = Chem.MolFromSmiles(mol['molecule_structures']['canonical_smiles'])
                        if rdkit_mol is None:
                            continue
                        descriptors = self.calculate_molecular_descriptors(rdkit_mol)
                        if descriptors is None:
                            continue
                        data.append({
                            'chembl_id': mol['molecule_chembl_id'],
                            'smiles': mol['molecule_structures']['canonical_smiles'],
                            **descriptors,
                            'label': 'authentic'
                        })
                    except Exception as e:
                        self.logger.warning(f"Error processing molecule {mol['molecule_chembl_id']}: {e}")
                        continue
            self.logger.info(f"Successfully processed {len(data)} molecules")
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Error fetching molecules: {e}")
            return pd.DataFrame()

    def generate_balanced_counterfeits(self, authentic_df, counterfeits_per_authentic=1, 
                                     max_attempts_per_molecule=10):
        """
        Generează contrafăcute cu ratio echilibrat și validare calitate
        """
        self.logger.info(f"Generating {counterfeits_per_authentic} counterfeit(s) per authentic molecule...")
        
        counterfeit_data = []
        strategies = ['aggressive_substitution', 'halogen_swap', 'ring_disruption', 'functional_transformation']
        
        for idx, row in tqdm(authentic_df.iterrows(), desc="Generating balanced counterfeits"):
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                continue
            
            counterfeits_created = 0
            attempts = 0
            
            while counterfeits_created < counterfeits_per_authentic and attempts < max_attempts_per_molecule:
                attempts += 1
                
                # Alege strategie și rate random pentru diversitate
                strategy = np.random.choice(strategies)
                mutation_rate = np.random.uniform(0.02, 0.08)  # Rate realiste: 2-8%
                
                try:
                    fake_mol = self._apply_aggressive_mutations(mol, strategy, mutation_rate)
                    
                    if fake_mol and Chem.MolToSmiles(fake_mol) != row['smiles']:
                        # Validează calitatea contrafăcerii
                        if self.validate_counterfeit_quality(mol, fake_mol):
                            descriptors = self.calculate_molecular_descriptors(fake_mol)
                            if descriptors:
                                counterfeit_data.append({
                                    'chembl_id': f"FAKE_{strategy}_{counterfeits_created}_{row['chembl_id']}",
                                    'smiles': Chem.MolToSmiles(fake_mol),
                                    'original_chembl_id': row['chembl_id'],
                                    'strategy': strategy,
                                    'mutation_rate': mutation_rate,
                                    **descriptors,
                                    'label': 'counterfeit'
                                })
                                counterfeits_created += 1
                                
                except Exception as e:
                    continue
            
            if counterfeits_created == 0:
                self.logger.warning(f"Could not generate valid counterfeit for {row['chembl_id']}")
        
        self.logger.info(f"Generated {len(counterfeit_data)} high-quality counterfeit variants")
        return pd.DataFrame(counterfeit_data)

    def smiles_to_heterogeneous_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        hetero_data = HeteroData()
        atom_types = {}
        atom_features = {}
        atom_to_idx = {}
        
        # Process atoms
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in atom_types:
                atom_types[atom_symbol] = []
                atom_features[atom_symbol] = []
            
            local_idx = len(atom_types[atom_symbol])
            atom_types[atom_symbol].append(atom_idx)
            atom_to_idx[atom_idx] = (atom_symbol, local_idx)
            
            features = self.extract_advanced_atom_features(atom, mol)
            atom_features[atom_symbol].append(features)
        
        # Set node features
        for atom_type, features in atom_features.items():
            if features:
                hetero_data[atom_type].x = torch.tensor(features, dtype=torch.float)
                hetero_data[atom_type].num_nodes = len(features)
        
        # Process edges
        edge_indices = {}
        edge_features = {}
        
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            start_symbol, start_local = atom_to_idx[start_idx]
            end_symbol, end_local = atom_to_idx[end_idx]
            
            bond_features = self._extract_advanced_bond_features(bond, mol)
            
            for (src, bond_type, dst) in [(start_symbol, str(bond.GetBondType()), end_symbol),
                                          (end_symbol, str(bond.GetBondType()), start_symbol)]:
                edge_key = (src, f"{src}_{bond_type}_{dst}", dst)
                if edge_key not in edge_indices:
                    edge_indices[edge_key] = []
                    edge_features[edge_key] = []
                
                edge_indices[edge_key].append([
                    atom_to_idx[start_idx][1] if src == start_symbol else atom_to_idx[end_idx][1],
                    atom_to_idx[end_idx][1] if src == start_symbol else atom_to_idx[start_idx][1]
                ])
                edge_features[edge_key].append(bond_features)
        
        # Set edge features
        for edge_key, indices in edge_indices.items():
            if indices:
                hetero_data[edge_key].edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
                hetero_data[edge_key].edge_attr = torch.tensor(edge_features[edge_key], dtype=torch.float)
        
        return hetero_data

    def prepare_balanced_heterogeneous_dataset(self, authentic_limit=1000, counterfeits_per_authentic=1):
        """
        Pregătește un dataset echilibrat cu contrafăcute de calitate
        """
        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting BALANCED dataset preparation at {start_time}")
        
        # Fetch authentic molecules
        authentic_df = self.fetch_drug_molecules(limit=authentic_limit)
        self.logger.info(f"Fetched {len(authentic_df)} authentic molecules")
        
        # Generate balanced counterfeits
        counterfeit_df = self.generate_balanced_counterfeits(
            authentic_df, 
            counterfeits_per_authentic=counterfeits_per_authentic
        )
        self.logger.info(f"Generated {len(counterfeit_df)} counterfeit variants")
        
        # Combine datasets
        full_dataset = pd.concat([authentic_df, counterfeit_df], ignore_index=True)
        
        # Convert to graphs
        hetero_graphs = []
        labels = []
        valid_indices = []
        
        for idx, row in tqdm(full_dataset.iterrows(), desc="Converting to graphs"):
            try:
                hetero_graph = self.smiles_to_heterogeneous_graph(row['smiles'])
                if hetero_graph is not None and len(hetero_graph.node_types) > 0:
                    hetero_graphs.append(hetero_graph)
                    labels.append(1 if row['label'] == 'counterfeit' else 0)
                    valid_indices.append(idx)
            except Exception as e:
                self.logger.warning(f"Error processing molecule {idx}: {e}")
                continue
        
        # Calculate class weights for balanced training
        if len(set(labels)) > 1:
            class_weights = compute_class_weight('balanced', 
                                               classes=np.unique(labels), 
                                               y=labels)
            class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        else:
            class_weights_dict = {0: 1.0, 1: 1.0}
        
        # Collect metadata
        if hetero_graphs:
            all_node_types = set()
            all_edge_types = set()
            
            for graph in hetero_graphs:
                all_node_types.update(graph.node_types)
                all_edge_types.update(graph.edge_types)
            
            node_types = sorted(list(all_node_types))
            edge_types = [(edge_type[0], edge_type[1], edge_type[2]) 
                         for edge_type in sorted(all_edge_types)]
            
            metadata = (node_types, edge_types)
            
            self.logger.info(f"Complete node types found: {node_types}")
            self.logger.info(f"Total edge types found: {len(edge_types)}")
        else:
            metadata = ([], [])
        
        end_time = datetime.now(timezone.utc)
        self.logger.info(f"Dataset preparation completed at {end_time}")
        self.logger.info(f"Total processing time: {end_time - start_time}")
        
        # Dataset statistics
        authentic_count = labels.count(0)
        counterfeit_count = labels.count(1)
        
        self.logger.info(f"Created {len(hetero_graphs)} valid graphs")
        self.logger.info(f"Authentic samples: {authentic_count}")
        self.logger.info(f"Counterfeit samples: {counterfeit_count}")
        self.logger.info(f"Balance ratio: 1:{counterfeit_count/max(authentic_count, 1):.2f}")
        self.logger.info(f"Class weights: {class_weights_dict}")
        
        return hetero_graphs, labels, metadata, class_weights_dict

if __name__ == "__main__":
    print(f"🔬 Improved ChEMBL Heterogeneous Graph Loader started at {datetime.now(timezone.utc)}")
    loader = ImprovedChEMBLDataLoader()
    
    # Generează dataset echilibrat
    hetero_graphs, labels, metadata, class_weights = loader.prepare_balanced_heterogeneous_dataset(
        authentic_limit=1000,
        counterfeits_per_authentic=1  # 1:1 ratio pentru început
    )
    
    print("\n=== BALANCED Dataset Statistics ===")
    print(f"Total graphs: {len(hetero_graphs)}")
    print(f"Authentic samples: {labels.count(0)}")
    print(f"Counterfeit samples: {labels.count(1)}")
    print(f"Balance ratio: {labels.count(1)/max(labels.count(0), 1):.2f}")
    print(f"Class weights: {class_weights}")
    
    if hetero_graphs:
        sample = hetero_graphs[0]
        print("\n=== Sample Graph ===")
        print(f"Node types: {list(sample.node_types)}")
        print(f"Edge types: {len(sample.edge_types)}")
        for node_type in sample.node_types:
            if hasattr(sample[node_type], 'x'):
                print(f"{node_type}: {sample[node_type].x.shape[0]} nodes, {sample[node_type].x.shape[1]} features")
    
    print("✅ IMPROVED Data loading complete! Ready for balanced training...")
    
    # Save improved dataset
    torch.save((hetero_graphs, labels, metadata, class_weights), 
               "improved_dataset_chembl_counterfeit_balanced.pt")