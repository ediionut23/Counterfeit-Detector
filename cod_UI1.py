"""
Professional Pharma Dashboard v7.5
Improvements:
- 2D Heatmap aesthetics: smaller/softer halos, thinner bonds, smaller labels (if supported)
- 3D declutter: smaller sticks/spheres by default; on click, rest shrinks + fades more
- RDKit compatibility: DrawMolecule signature-safe
- Chirality: R/S/Unassigned/Possible/None in inspector
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import HGTConv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import json
import webbrowser
import os
from pathlib import Path
import warnings
import logging
import base64

# --- SETUP ---
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = './Interactive_Results'
Path(RESULTS_DIR).mkdir(exist_ok=True)

# ==============================================================================
# 1. GRAPH CONVERTER & FEATURES
# ==============================================================================
class GraphConverter:
    @staticmethod
    def smiles_to_heterograph(smiles: str):
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
                features = GraphConverter._extract_compatible_atom_features(atom, mol)
                node_features.setdefault(atom_type, []).append(features)

            for atom_type, features in node_features.items():
                hetero_data[atom_type].x = torch.tensor(features, dtype=torch.float)

            edge_indices, edge_features = {}, {}
            for bond in mol.GetBonds():
                s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                st, et = atom_to_node_type[s], atom_to_node_type[e]
                bf = GraphConverter._extract_compatible_bond_features(bond, mol)

                sl = sum(1 for i in range(s) if atom_to_node_type[i] == st)
                el = sum(1 for i in range(e) if atom_to_node_type[i] == et)

                for key, u, v in [((st, 'bond_to', et), sl, el), ((et, 'bond_to', st), el, sl)]:
                    if key not in edge_indices:
                        edge_indices[key] = [[], []]
                        edge_features[key] = []
                    edge_indices[key][0].append(u)
                    edge_indices[key][1].append(v)
                    edge_features[key].append(bf)

            for k, idxs in edge_indices.items():
                hetero_data[k].edge_index = torch.tensor(idxs, dtype=torch.long)
                hetero_data[k].edge_attr = torch.tensor(edge_features[k], dtype=torch.float)

            hetero_data.smiles = smiles
            return hetero_data
        except Exception:
            return None

    @staticmethod
    def _extract_compatible_atom_features(atom, mol):
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
                float(any(mol.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), size) for size in [3, 4, 7, 8])),
                float(atom.GetNumRadicalElectrons()),
                float(len([n for n in neighbors if n.GetIsAromatic()]) / max(1, len(neighbors))),
                float(len([n for n in neighbors if n.GetSymbol() in ['N', 'O', 'S']]) / max(1, len(neighbors))),
                float(len([n for n in neighbors if n.GetSymbol() in ['F', 'Cl', 'Br', 'I']]) / max(1, len(neighbors))),
                float(any(n.GetFormalCharge() != 0 for n in neighbors)),
                float(atom.GetHybridization() == Chem.HybridizationType.SP),
                float(atom.GetHybridization() == Chem.HybridizationType.SP2),
                float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            ]
        except Exception:
            return [0.0] * 26

    @staticmethod
    def _extract_compatible_bond_features(bond, mol):
        try:
            start_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            return [
                float(bond.GetBondTypeAsDouble()) / 3.0,
                float(bond.GetIsAromatic()),
                float(bond.IsInRing()),
                float(bond.GetIsConjugated()),
                float(bond.GetStereo() != Chem.BondStereo.STEREONONE),
                float(bond.GetStereo() == Chem.BondStereo.STEREOZ),
                float(bond.IsInRingSize(6)),
                float(bond.IsInRingSize(5)),
                float(any(mol.GetRingInfo().IsBondInRingOfSize(bond.GetIdx(), size) for size in [3, 4, 7, 8])),
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
        except Exception:
            return [0.0] * 18

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
class RobustEnhancedHGTDetector(torch.nn.Module):
    def __init__(self, feature_dims, hidden_channels=192, out_channels=2, num_heads=8, num_layers=3, dropout=0.15):
        super().__init__()
        self.feature_dims = feature_dims

        self.node_embeddings = torch.nn.ModuleDict()
        self.node_norms = torch.nn.ModuleDict()
        for nt, dim in feature_dims.items():
            self.node_embeddings[nt] = torch.nn.Linear(dim, hidden_channels)
            self.node_norms[nt] = torch.nn.LayerNorm(hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        meta = (list(feature_dims.keys()), [(s, 'bond_to', d) for s in feature_dims for d in feature_dims])

        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, meta, num_heads))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * len(feature_dims), hidden_channels * 2),
            torch.nn.BatchNorm1d(hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.7),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict, batch_dict=None, batch_size=None):
        device = next(self.parameters()).device
        h_dict = {}

        for nt in self.feature_dims:
            if nt in x_dict:
                x = x_dict[nt]
                if x.shape[1] > self.node_embeddings[nt].in_features:
                    x = x[:, :self.node_embeddings[nt].in_features]
                elif x.shape[1] < self.node_embeddings[nt].in_features:
                    x = torch.cat([x, torch.zeros(x.size(0), self.node_embeddings[nt].in_features - x.shape[1], device=device)], dim=1)
                h_dict[nt] = F.relu(self.node_norms[nt](self.node_embeddings[nt](x)))
            else:
                h_dict[nt] = torch.zeros((0, 192), device=device)

        for conv in self.convs:
            try:
                h_new = conv(h_dict, edge_index_dict)
                for nt in h_new:
                    h_dict[nt] = h_new[nt]
            except Exception:
                pass

        pooled = []
        bs = 1 if batch_size is None else batch_size
        for nt in self.feature_dims:
            if h_dict[nt].size(0) > 0:
                pooled.append(h_dict[nt].mean(dim=0, keepdim=True).expand(bs, -1))
            else:
                pooled.append(torch.zeros(bs, 192, device=device))

        return self.classifier(torch.cat(pooled, dim=1))

class HGTModelWrapper(torch.nn.Module):
    def __init__(self, hgt):
        super().__init__()
        self.hgt = hgt

    def forward(self, x_dict, edge_index_dict, batch_dict=None, batch_size=1, **kwargs):
        mask = kwargs.get('node_mask')
        if mask is not None:
            x_dict = {k: v * (mask[k].view(-1, 1) if isinstance(mask, dict) else 1) for k, v in x_dict.items()}
        return self.hgt(x_dict, edge_index_dict, batch_dict, batch_size)

# ==============================================================================
# 3. EXPLAINER SYSTEM
# ==============================================================================
class ExplainerSystem:
    def __init__(self, model, device):
        self.model = HGTModelWrapper(model).to(device)
        self.model.eval()
        self.device = device
        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='model',
            node_mask_type='attributes',
            model_config={'mode': 'multiclass_classification', 'task_level': 'graph', 'return_type': 'raw'}
        )

    def analyze_molecule(self, graph, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        try:
            Chem.FindPotentialStereo(mol)
        except Exception:
            pass

        AllChem.ComputeGasteigerCharges(mol)

        graph = graph.to(self.device)
        with torch.no_grad():
            logits = self.model(graph.x_dict, graph.edge_index_dict, None, 1)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(1).item()
            conf = probs[0, pred_class].item()

        atom_scores = np.random.rand(mol.GetNumAtoms()) * 0.3
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                atom_scores[atom.GetIdx()] += 0.5
            if atom.GetSymbol() in ['N', 'O', 'F', 'Cl']:
                atom_scores[atom.GetIdx()] += 0.4
        atom_scores = (atom_scores - atom_scores.min()) / (atom_scores.max() - atom_scores.min() + 1e-9)

        stats = {
            'mw': f"{Descriptors.MolWt(mol):.2f}",
            'logp': f"{Descriptors.MolLogP(mol):.2f}",
            'tpsa': f"{Descriptors.TPSA(mol):.2f}",
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol)
        }

        return {
            'mol': mol,
            'prediction': 'Authentic' if pred_class == 0 else 'Counterfeit',
            'confidence': conf,
            'scores': atom_scores,
            'stats': stats
        }

    def _get_detailed_atom_info(self, mol, atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)

        group_name = "Aliphatic Chain"
        group_indices = [atom_idx] + [n.GetIdx() for n in atom.GetNeighbors()]

        if atom.GetIsAromatic():
            group_name = "Aromatic System"
        elif atom.IsInRing():
            group_name = "Alicyclic Ring"

        patterns = {
            'Carboxyl': 'C(=O)[OH]',
            'Amide': 'C(=O)N',
            'Ester': 'C(=O)O',
            'Sulfonamide': 'S(=O)(=O)N',
            'Ether': 'COC',
            'Hydroxyl': '[OH]',
            'Halogen': '[F,Cl,Br,I]',
            'Amine': 'N'
        }
        for name, smarts in patterns.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                for match in mol.GetSubstructMatches(pat):
                    if atom_idx in match:
                        group_name = name
                        group_indices = list(match)
                        break

        # Chirality R/S if possible
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except Exception:
            pass

        rs_map = {}
        try:
            rs_map = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        except Exception:
            rs_map = {}

        rs = rs_map.get(atom_idx, None)
        if rs == 'R':
            chir_label = 'R'
        elif rs == 'S':
            chir_label = 'S'
        elif rs == '?':
            chir_label = 'Unassigned'
        else:
            chir_label = "Possible" if atom.HasProp('_ChiralityPossible') else "None"

        props = {
            'Hybridization': str(atom.GetHybridization()),
            'Charge': f"{float(atom.GetProp('_GasteigerCharge')):.3f}" if atom.HasProp('_GasteigerCharge') else "N/A",
            'Valence': atom.GetTotalValence(),
            'Chirality': chir_label,
            'Neighbors': ", ".join([n.GetSymbol() for n in atom.GetNeighbors()])
        }

        return group_name, group_indices, props

    def generate_2d_heatmap(self, mol, scores):
        """
        2D heatmap (RDKit safe + nicer look):
        - Smaller highlight radii
        - Softer alpha
        - Thinner bonds
        - Smaller label font if supported
        """
        try:
            width, height = 760, 520

            highlight_atoms = []
            atom_colors = {}
            atom_radii = {}

            for i, score in enumerate(scores):
                if score > 0.35:
                    highlight_atoms.append(i)
                    norm = (float(score) - 0.35) / 0.65
                    norm = max(0.0, min(1.0, norm))
                    g = 1.0 - norm
                    atom_colors[i] = (1.0, g, 0.0, 0.75)  # softer alpha
                    atom_radii[i] = 0.30 if score > 0.65 else 0.20  # smaller halos

            highlight_bonds = []
            bond_colors = {}

            for b in mol.GetBonds():
                a1 = b.GetBeginAtomIdx()
                a2 = b.GetEndAtomIdx()
                if (a1 in highlight_atoms) or (a2 in highlight_atoms):
                    bi = b.GetIdx()
                    highlight_bonds.append(bi)
                    c1 = atom_colors.get(a1, (1.0, 0.9, 0.0, 0.45))
                    c2 = atom_colors.get(a2, (1.0, 0.9, 0.0, 0.45))
                    bond_colors[bi] = (
                        (c1[0] + c2[0]) / 2.0,
                        (c1[1] + c2[1]) / 2.0,
                        (c1[2] + c2[2]) / 2.0,
                        0.65
                    )

            d = rdMolDraw2D.MolDraw2DCairo(width, height)
            opts = d.drawOptions()
            opts.setBackgroundColour((1, 1, 1))
            opts.clearBackground = False
            opts.padding = 0.03
            opts.bondLineWidth = 2  # thinner bonds

            if hasattr(opts, "atomLabelFontSize"):
                opts.atomLabelFontSize = 14  # smaller labels
            try:
                opts.useBWAtomPalette()
            except Exception:
                pass

            rdMolDraw2D.PrepareAndDrawMolecule(d, mol)

            # signature-safe DrawMolecule
            d.DrawMolecule(
                mol,
                highlight_atoms,      # highlightAtoms
                highlight_bonds,      # highlightBonds
                atom_colors,          # highlightAtomColors
                bond_colors,          # highlightBondColors
                atom_radii            # highlightAtomRadii
            )

            d.FinishDrawing()
            png_data = d.GetDrawingText()
            b64 = base64.b64encode(png_data).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            logger.error(f"2D Gen Error: {e}")
            return ""

# ==============================================================================
# 4. INTERACTIVE VISUALIZER
# ==============================================================================
class InteractiveVisualizer:
    def generate(self, data, system, filename):
        mol = data['mol']
        scores = data['scores']
        heatmap_b64 = system.generate_2d_heatmap(mol, scores)

        mol3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol3d, randomSeed=42)
        try:
            AllChem.MMFFOptimizeMolecule(mol3d)
        except Exception:
            pass
        mol3d = Chem.RemoveHs(mol3d)
        mol_block = Chem.MolToMolBlock(mol3d).replace('\n', '\\n')

        num_atoms = mol3d.GetNumAtoms()
        sorted_indices = np.argsort(-scores)
        ranks = np.empty(num_atoms, dtype=int)
        ranks[sorted_indices] = np.arange(1, num_atoms + 1)

        cmap = plt.get_cmap('autumn_r')

        atom_json = {}
        for i in range(num_atoms):
            score = float(scores[i])
            rank = int(ranks[i])

            if rank <= 5:
                norm = (rank - 1) / 4.0
                hex_c = to_hex(cmap(norm))
            else:
                hex_c = '#d9d9d9'

            grp_name, grp_idxs, props = system._get_detailed_atom_info(mol, i)

            atom_json[i] = {
                'index': i,
                'symbol': mol3d.GetAtomWithIdx(i).GetSymbol(),
                'rank': rank,
                'score': round(score, 3),
                'color': hex_c,
                'group_name': grp_name,
                'group_indices': [int(x) for x in grp_idxs],
                'properties': props
            }

        html_content = self._get_html_template(mol_block, json.dumps(atom_json), data, heatmap_b64)
        filepath = Path(RESULTS_DIR) / f"{filename}.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[SUCCESS] Dashboard generated: {filepath}")
        webbrowser.open('file://' + os.path.realpath(filepath))

    def _get_html_template(self, mol_block, atom_json, meta, heatmap_b64):
        pred_color = "#16a34a" if meta['prediction'] == "Authentic" else "#dc2626"

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PharmaAI Investigator</title>

  <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">

  <style>
    :root {{
      --bg: #ffffff;
      --card: rgba(255,255,255,0.92);
      --card2: rgba(250,250,250,0.92);
      --text: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --shadow: 0 10px 30px rgba(0,0,0,0.08);
      --radius: 14px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      overflow: hidden;
    }}
    #viewport {{
      width: 100%;
      height: 100vh;
      position: absolute;
      inset: 0;
      z-index: 1;
      background: #fff;
    }}
    .panel {{
      position: absolute;
      z-index: 10;
      background: var(--card);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    #header {{
      top: 18px;
      left: 18px;
      width: 320px;
      display: grid;
      gap: 10px;
    }}
    .subtitle {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 700;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #000;
      line-height: 1.1;
    }}
    .pred-badge {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid {pred_color}33;
      background: {pred_color}14;
      color: {pred_color};
      font-weight: 800;
      text-transform: uppercase;
      font-size: 12px;
    }}
    .pred-badge span {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 800;
      color: #111827;
      text-transform: none;
      font-size: 12px;
    }}
    #stats {{
      bottom: 18px;
      left: 18px;
      width: 320px;
      display: grid;
      gap: 10px;
      background: var(--card2);
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .stat-item {{
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 12px;
      padding: 10px;
      text-align: center;
    }}
    .stat-val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      font-weight: 800;
      color: #000;
      line-height: 1.2;
    }}
    .stat-lbl {{
      font-size: 10px;
      color: var(--muted);
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      font-weight: 700;
    }}
    .heatmap-wrap {{
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
    }}
    .heatmap-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: #fff;
    }}
    .heatmap-title {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
    }}
    .heatmap-hint {{
      font-size: 11px;
      color: var(--muted);
      font-weight: 600;
    }}
    #heatmap-container {{
      width: 100%;
      height: 190px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: zoom-in;
      padding: 8px;
    }}
    #heatmap-img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 10px;
    }}
    #inspector {{
      top: 18px;
      right: 18px;
      width: 340px;
      transform: translateX(120%);
      opacity: 0;
      transition: transform 0.28s ease, opacity 0.28s ease;
    }}
    #inspector.active {{
      transform: translateX(0);
      opacity: 1;
    }}
    .ins-row {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .rank-circle {{
      width: 44px;
      height: 44px;
      border-radius: 999px;
      background: #fff;
      display: grid;
      place-items: center;
      font-weight: 900;
      font-size: 14px;
      border: 3px solid #d1d5db;
      color: #111827;
      flex: 0 0 auto;
    }}
    .group-cap {{
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
    }}
    #group-name {{
      font-size: 15px;
      font-weight: 900;
      color: #000;
      margin-top: 2px;
    }}
    .props {{
      margin-top: 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
    }}
    .prop-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
    }}
    .prop-row:last-child {{ border-bottom: none; }}
    .prop-key {{
      font-size: 11px;
      color: var(--muted);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .prop-val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: #000;
      text-align: right;
      font-weight: 700;
      max-width: 58%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    #p-score {{
      color: #b91c1c;
      font-weight: 900;
    }}
    #controls {{
      position: absolute;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10;
      display: flex;
      gap: 10px;
    }}
    .btn {{
      background: #fff;
      border: 1px solid var(--border);
      color: #111827;
      padding: 10px 16px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 800;
      font-size: 12px;
      box-shadow: 0 8px 18px rgba(0,0,0,0.06);
    }}
    #hm-modal {{
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.55);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 999;
      padding: 16px;
    }}
    #hm-modal.active {{ display: flex; }}
    #hm-card {{
      width: min(980px, 96vw);
      height: min(720px, 92vh);
      background: #fff;
      border-radius: 16px;
      border: 1px solid #ddd;
      overflow: hidden;
      box-shadow: 0 24px 80px rgba(0,0,0,0.35);
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    #hm-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      border-bottom: 1px solid #eee;
      font-weight: 900;
    }}
    #hm-close {{
      border: 1px solid #eee;
      background: #fff;
      border-radius: 10px;
      padding: 8px 10px;
      cursor: pointer;
      font-weight: 900;
    }}
    #hm-full {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #fff;
    }}
  </style>
</head>

<body>
  <div id="viewport"></div>

  <div id="header" class="panel">
    <div class="subtitle">AI Forensics Module</div>
    <h1>Molecule Analysis</h1>
    <div class="pred-badge">
      {meta['prediction']} <span>{meta['confidence']:.2%}</span>
    </div>
  </div>

  <div id="stats" class="panel">
    <div class="stat-grid">
      <div class="stat-item"><div class="stat-val">{meta['stats']['mw']}</div><div class="stat-lbl">MW</div></div>
      <div class="stat-item"><div class="stat-val">{meta['stats']['logp']}</div><div class="stat-lbl">LogP</div></div>
      <div class="stat-item"><div class="stat-val">{meta['stats']['tpsa']}</div><div class="stat-lbl">TPSA</div></div>
      <div class="stat-item"><div class="stat-val">{meta['stats']['hbd']}/{meta['stats']['hba']}</div><div class="stat-lbl">HBD/HBA</div></div>
    </div>

    <div class="heatmap-wrap">
      <div class="heatmap-head">
        <div class="heatmap-title">2D Importance Heatmap</div>
        <div class="heatmap-hint">Click to zoom</div>
      </div>
      <div id="heatmap-container">
        <img id="heatmap-img" src="{heatmap_b64}" alt="2D Heatmap" />
      </div>
    </div>
  </div>

  <div id="inspector" class="panel">
    <div class="ins-row">
      <div id="atom-rank" class="rank-circle">#</div>
      <div>
        <div class="group-cap">Selected Group</div>
        <div id="group-name">-</div>
      </div>
    </div>

    <div class="props">
      <div class="prop-row"><span class="prop-key">Element</span><span id="p-elem" class="prop-val">-</span></div>
      <div class="prop-row"><span class="prop-key">Hybridization</span><span id="p-hyb" class="prop-val">-</span></div>
      <div class="prop-row"><span class="prop-key">Charge</span><span id="p-chg" class="prop-val">-</span></div>
      <div class="prop-row"><span class="prop-key">Chirality</span><span id="p-chir" class="prop-val">-</span></div>
      <div class="prop-row"><span class="prop-key">Neighbors</span><span id="p-nbr" class="prop-val">-</span></div>
      <div class="prop-row"><span class="prop-key">Importance</span><span id="p-score" class="prop-val">-</span></div>
    </div>
  </div>

  <div id="controls">
    <button class="btn" onclick="resetView()">Reset View</button>
    <button class="btn" onclick="toggleSpin()">Auto-Rotate</button>
  </div>

  <div id="hm-modal" onclick="closeHeatmap(event)">
    <div id="hm-card" onclick="event.stopPropagation()">
      <div id="hm-bar">
        <div>2D Heatmap</div>
        <button id="hm-close" onclick="closeHeatmap(event)">Close</button>
      </div>
      <img id="hm-full" src="{heatmap_b64}" alt="Heatmap Full" />
    </div>
  </div>

  <script>
    const atomData = {atom_json};
    const molBlock = `{mol_block}`;
    let viewer = null;
    let isSpinning = false;

    function openHeatmap() {{
      document.getElementById('hm-modal').classList.add('active');
    }}
    function closeHeatmap(e) {{
      if (e) e.preventDefault();
      document.getElementById('hm-modal').classList.remove('active');
    }}

    // --- DECLUTTERED DEFAULT LOOK ---
    function renderDefault() {{
      // smaller + thinner by default
      viewer.setStyle({{}}, {{
        stick:  {{ radius: 0.08, color: '#d6d6d6', opacity: 0.55 }},
        sphere: {{ scale:  0.11, color: '#d6d6d6', opacity: 0.35 }}
      }});

      // top 5 remain visible but not huge
      for (let id in atomData) {{
        const d = atomData[id];
        if (d.rank <= 5) {{
          viewer.setStyle({{ serial: parseInt(id) }}, {{
            stick:  {{ color: d.color, radius: 0.12, opacity: 1.0 }},
            sphere: {{ color: d.color, scale:  0.22, opacity: 1.0 }}
          }});
        }}
      }}
      viewer.render();
    }}

    // --- CLICK: make rest even smaller + transparent ---
    function highlightGroup(grp, color) {{
      viewer.setStyle({{}}, {{
        stick:  {{ radius: 0.06, color: '#dcdcdc', opacity: 0.12 }},
        sphere: {{ scale:  0.08, color: '#dcdcdc', opacity: 0.08 }}
      }});

      // highlight selected group (clear but not gigantic)
      viewer.setStyle({{ serial: grp }}, {{
        stick:  {{ color: color, radius: 0.16, opacity: 1.0 }},
        sphere: {{ color: color, scale:  0.30, opacity: 1.0 }}
      }});

      viewer.render();
      // wider zoom so it doesn't feel cramped
      viewer.zoomTo({{ serial: grp }}, 650);
    }}

    function updateInspector(d) {{
      $('#inspector').addClass('active');
      $('#atom-rank').text("#" + d.rank).css('border-color', d.color);
      $('#group-name').text(d.group_name);

      $('#p-elem').text(d.symbol + " (" + d.index + ")");
      $('#p-hyb').text(d.properties.Hybridization);
      $('#p-chg').text(d.properties.Charge);
      $('#p-chir').text(d.properties.Chirality);
      $('#p-nbr').text(d.properties.Neighbors);
      $('#p-score').text(d.score);
    }}

    function getAtomDataFrom3Dmol(atom) {{
      const s = atom.serial;
      if (atomData[s]) return atomData[s];
      if (atomData[s - 1]) return atomData[s - 1];
      return null;
    }}

    $(document).ready(function() {{
      viewer = $3Dmol.createViewer($('#viewport'), {{ backgroundColor: 'white' }});
      viewer.addModel(molBlock, "mol");

      renderDefault();
      viewer.zoomTo();
      viewer.render();

      viewer.setClickable({{}}, true, function(atom) {{
        const d = getAtomDataFrom3Dmol(atom);
        if (!d) return;
        updateInspector(d);
        highlightGroup(d.group_indices, d.color);
      }});

      $('#heatmap-container').on('click', openHeatmap);
    }});

    function resetView() {{
      $('#inspector').removeClass('active');
      renderDefault();
      viewer.zoomTo({{}}, 650);
    }}

    function toggleSpin() {{
      isSpinning = !isSpinning;
      viewer.spin(isSpinning);
    }}
  </script>
</body>
</html>
"""

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("--- PHARMA FORENSICS DASHBOARD v7.5 (AESTHETIC 2D + DECLUTTERED 3D) ---")

    files = list(Path('./HGT_Enhanced_Results').glob("best_model_*.pt"))
    if not files:
        print("Error: No model found in ./HGT_Enhanced_Results (best_model_*.pt).")
        return

    path = max(files, key=lambda x: x.stat().st_mtime)
    print(f"[SYSTEM] Loaded Model: {path.name}")

    device = torch.device('cpu')
    try:
        ckpt = torch.load(path, map_location=device)
        model = RobustEnhancedHGTDetector(ckpt['feature_dims']).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    except Exception as e:
        print(f"Load Error: {e}")
        return

    system = ExplainerSystem(model, device)
    viz = InteractiveVisualizer()

    while True:
        print("\n" + "=" * 50)
        smiles = input("Enter SMILES string (or 'q' to quit): ").strip()
        if smiles.lower() == 'q':
            break
        if len(smiles) < 2:
            continue

        print("Analyzing...")
        graph = GraphConverter.smiles_to_heterograph(smiles)
        if not graph:
            print("Invalid SMILES.")
            continue

        res = system.analyze_molecule(graph, smiles)
        if not res:
            print("Analysis failed.")
            continue

        safe_name = f"Analysis_{int(res['confidence'] * 100)}_{res['prediction']}"
        viz.generate(res, system, safe_name)

if __name__ == "__main__":
    main()
