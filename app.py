import os
import sys
import io
import base64
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit.Chem import AllChem, rdDepictor, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from hyp import RobustEnhancedHGTDetector, PyGNativeExplainer
from explicablity import GraphConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app_state: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(__file__).parent / "HGT_Enhanced_Results"
    target_model = model_dir / "best_model_f1_0.8143.pt"
    if not target_model.exists():
        model_files = list(model_dir.glob("best_model_f1_*.pt"))
        target_model = max(model_files, key=os.path.getmtime) if model_files else None
    if not target_model:
        raise RuntimeError("No model found in HGT_Enhanced_Results/")
    logger.info(f"Loading model: {target_model.name}")
    device = torch.device("cpu")
    ckpt = torch.load(target_model, map_location=device, weights_only=False)
    model = RobustEnhancedHGTDetector(ckpt["feature_dims"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    app_state["explainer"] = PyGNativeExplainer(model, device)
    app_state["model"] = model
    logger.info("GNNExplainer engine ready.")
    yield
    app_state.clear()

app = FastAPI(title="PharmaAI Investigator Pro", lifespan=lifespan)

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    smiles: str

class AnalyzeResponse(BaseModel):
    html: str
    heatmap_svg: str
    prediction: str
    confidence: float
    charts_features: str
    charts_metrics: str
    charts_probability: str
    substructures: list
    mol_stats: dict
    counterfactual: dict

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_FG_PATTERNS = {
    'Carboxylic Acid': 'C(=O)O', 'Ester': 'C(=O)O[C,c]', 'Amide': 'C(=O)N',
    'Amine': 'N', 'Hydroxyl': '[OH]', 'Ketone': 'C(=O)[C,c]',
    'Nitro': 'N(=O)=O', 'Sulfone': 'S(=O)(=O)', 'Phenyl': 'c1ccccc1',
    'Halogen': '[F,Cl,Br,I]', 'Aromatic': '[a]',
}
_FG_CACHE: Dict[str, Any] = {}
for _k, _v in _FG_PATTERNS.items():
    try:
        _FG_CACHE[_k] = Chem.MolFromSmarts(_v)
    except Exception:
        _FG_CACHE[_k] = None


def identify_fg(mol: Chem.Mol, atom_idx: int) -> str:
    for name, pat in _FG_CACHE.items():
        if pat is None:
            continue
        if mol.HasSubstructMatch(pat):
            for match in mol.GetSubstructMatches(pat):
                if atom_idx in match:
                    return name
    return f"{mol.GetAtomWithIdx(atom_idx).GetSymbol()} Group"


def score_to_hex(score: float) -> str:
    """Map 0..1 importance score to YlOrRd: light yellow → orange → dark red."""
    s = max(0.0, min(1.0, float(score)))
    # Control points: (t, (R, G, B))
    stops = [
        (0.00, (255, 255, 204)),  # #ffffcc – very light yellow
        (0.25, (254, 217, 118)),  # #fed976 – yellow
        (0.50, (253, 141,  60)),  # #fd8d3c – orange
        (0.75, (227,  26,  28)),  # #e31a1c – red
        (1.00, (128,   0,  38)),  # #800026 – dark red / maroon
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if s <= t1 or i == len(stops) - 2:
            t = (s - t0) / (t1 - t0 + 1e-9)
            t = max(0.0, min(1.0, t))
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#800026"


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-9)


def generate_2d_svg(mol: Chem.Mol, scores: np.ndarray) -> str:
    """2D substructure-style heatmap: red blobs for high importance, fading to nothing."""
    mol2d = Chem.RWMol(mol)
    try:
        rdDepictor.Compute2DCoords(mol2d)
    except Exception:
        pass

    norm = normalize_scores(scores)
    highlight_atoms = []
    atom_colors: dict = {}
    atom_radii: dict = {}
    bond_colors: dict = {}
    highlight_bonds = []

    # Color: rank-based red → pink (only atoms above threshold)
    sorted_idx = np.argsort(-norm)
    ranks = np.empty(len(norm), dtype=int)
    ranks[sorted_idx] = np.arange(1, len(norm) + 1)

    for i, s in enumerate(norm):
        s = float(s)
        if s < 0.1:
            continue
        highlight_atoms.append(i)
        r = int(ranks[i])
        if r == 1:
            atom_colors[i] = (1.0, 0.0, 0.0)          # pure red
            atom_radii[i] = 0.60
        elif r <= 3:
            atom_colors[i] = (1.0, 0.15, 0.1)
            atom_radii[i] = 0.50
        elif r <= 6:
            t = (s - 0.35) / 0.65
            atom_colors[i] = (1.0, 0.4 + 0.35 * (1 - t), 0.35 + 0.35 * (1 - t))
            atom_radii[i] = 0.35
        else:
            atom_colors[i] = (1.0, 0.75, 0.72)         # light pink
            atom_radii[i] = 0.25

    # Highlight bonds between important atoms
    for bond in mol2d.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in atom_colors and v in atom_colors:
            highlight_bonds.append(bond.GetIdx())
            cu = atom_colors[u]
            cv = atom_colors[v]
            bond_colors[bond.GetIdx()] = (
                (cu[0] + cv[0]) / 2,
                (cu[1] + cv[1]) / 2,
                (cu[2] + cv[2]) / 2,
            )

    drawer = rdMolDraw2D.MolDraw2DSVG(420, 300)
    opts = drawer.drawOptions()
    opts.padding = 0.12
    opts.bondLineWidth = 1.8
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol2d,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _compute_property_importance(mol: Chem.Mol, atom_scores_norm: np.ndarray) -> list:
    """
    Compute weighted feature importance from real atom properties × GNNExplainer scores.
    For each (atom_type, property) pair: mean(property_value * score) across all atoms of that type.
    """
    from rdkit.Chem import rdchem
    HT = rdchem.HybridizationType
    ri = mol.GetRingInfo()

    accum: Dict[str, float] = {}
    count: Dict[str, int] = {}

    for atom_idx, score in enumerate(atom_scores_norm):
        atom = mol.GetAtomWithIdx(atom_idx)
        atype = atom.GetSymbol()
        nbrs = atom.GetNeighbors()
        total_nbrs = max(len(nbrs), 1)
        hyb = atom.GetHybridization()

        in5 = any(len(r) == 5 for r in ri.AtomRings() if atom_idx in r)
        in6 = any(len(r) == 6 for r in ri.AtomRings() if atom_idx in r)
        arom_nbrs = sum(1 for n in nbrs if n.GetIsAromatic()) / total_nbrs
        hetero_nbrs = sum(1 for n in nbrs if n.GetSymbol() not in ['C', 'H']) / total_nbrs
        halogen_nbrs = sum(1 for n in nbrs if n.GetSymbol() in ['F', 'Cl', 'Br', 'I']) / total_nbrs
        charged_nbrs = float(sum(1 for n in nbrs if n.GetFormalCharge() != 0))

        props = {
            'Formal_Charge':            float(abs(atom.GetFormalCharge())),
            'Is_H_Acceptor':            float(atom.GetSymbol() in ['N', 'O', 'F']),
            'Is_H_Donor':               float(atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0),
            'Is_Aromatic':              float(atom.GetIsAromatic()),
            'In_Ring':                  float(atom.IsInRing()),
            'In_Ring_Size_5':           float(in5),
            'In_Ring_Size_6':           float(in6),
            'Is_Halogen':               float(atom.GetSymbol() in ['F', 'Cl', 'Br', 'I']),
            'Is_Heteroatom':            float(atom.GetSymbol() not in ['C', 'H']),
            'Is_SP':                    float(hyb == HT.SP),
            'Is_SP2':                   float(hyb == HT.SP2),
            'Is_SP3':                   float(hyb == HT.SP3),
            'Aromatic_Neighbors_Ratio': arom_nbrs,
            'Heteroatom_Neighbors_Ratio': hetero_nbrs,
            'Halogen_Neighbors_Ratio':  halogen_nbrs,
            'Charged_Neighbors':        charged_nbrs,
        }

        for feat_name, val in props.items():
            key = f"{atype}:{feat_name}"
            accum[key] = accum.get(key, 0.0) + val * float(score)
            count[key] = count.get(key, 0) + 1

    rows = [
        {'Name': k, 'Value': accum[k] / count[k]}
        for k in accum
        if count[k] > 0 and accum[k] > 0
    ]
    rows.sort(key=lambda x: x['Value'], reverse=True)
    return rows[:15]


_SS_PATTERNS = [
    ('Sulfonamide',      'S(=O)(=O)N'),
    ('Sulfonyl',         'S(=O)(=O)'),
    ('Piperazinyl',      'N1CCNCC1'),
    ('Carboxylic Acid',  'C(=O)[OH]'),
    ('Ester',            'C(=O)O[C,c]'),
    ('Amide',            'C(=O)N'),
    ('Ketone',           'C(=O)[C,c]'),
    ('Nitro',            'N(=O)=O'),
    ('Pyrimidine',       'c1ncncn1'),
    ('Imidazole',        'c1cnc[nH]1'),
    ('Pyrazole',         'c1cc[nH]n1'),
    ('Triazole',         'c1cn[nH]n1'),
    ('Phenyl',           'c1ccccc1'),
    ('Pyridine',         'c1ccncc1'),
    ('Hydroxyl',         '[OH]'),
    ('Primary Amine',    '[NH2]'),
    ('Secondary Amine',  '[NH;!$(NC=O)]'),
    ('Tertiary Amine',   '[NX3;!$(NC=O);!$([NH])]'),
    ('Halogen',          '[F,Cl,Br,I]'),
    ('Alkyl Chain',      '[CX4H2][CX4H2]'),
    ('Aromatic System',  '[a]'),
]
_SS_CACHE = []
for _name, _sma in _SS_PATTERNS:
    try:
        pat = Chem.MolFromSmarts(_sma)
        if pat:
            _SS_CACHE.append((_name, pat))
    except Exception:
        pass


def _compute_critical_substructures(mol: Chem.Mol, atom_scores_norm: np.ndarray) -> list:
    """Identify functional groups with high GNNExplainer importance using RDKit SMARTS."""
    seen: set = set()
    results = []
    for name, pat in _SS_CACHE:
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            key = (name, tuple(sorted(match)))
            if key in seen:
                continue
            seen.add(key)
            avg_score = float(np.mean([atom_scores_norm[i] for i in match]))
            results.append({
                'functional_group': name,
                'importance_score': round(avg_score, 4),
                'center_atom': mol.GetAtomWithIdx(match[0]).GetSymbol(),
                'size': len(match),
            })
    results.sort(key=lambda x: x['importance_score'], reverse=True)
    return results[:8]


def charts_to_b64(raw_explanation: dict, mol: Chem.Mol = None,
                  atom_scores_norm: np.ndarray = None) -> dict:
    """Generate all three analysis charts (dark theme) as base64 PNG."""
    charts: Dict[str, str] = {}

    def _style(ax, fig, title: str):
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.set_title(title, color='#f1f5f9', fontsize=12, pad=12, fontweight='bold')
        ax.tick_params(colors='#6b7280', labelsize=9)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#1f2937')

    def _save(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor='#0d1117', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # Chart 1 – Feature Importance (from real atom properties × GNNExplainer scores)
    try:
        rows = []
        if mol is not None and atom_scores_norm is not None:
            rows = _compute_property_importance(mol, atom_scores_norm)
        # Fallback to raw explanation feature_importance if available
        if not rows:
            feat_imp = raw_explanation.get('feature_importance', {})
            for ntype, data in feat_imp.items():
                for f in data.get('top_features', []):
                    v = float(f['importance'])
                    if v > 1e-6:
                        rows.append({'Name': f"{ntype}:{f['name']}", 'Value': v})
            rows.sort(key=lambda x: x['Value'], reverse=True)
            rows = rows[:15]

        if rows:
            df = pd.DataFrame(rows).sort_values('Value', ascending=True)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            _style(ax, fig, "Top Weighted Chemical Features (GNNExplainer × Properties)")
            # YlOrRd gradient for bars based on value rank
            n_bars = len(df)
            bar_colors = [score_to_hex(i / max(n_bars - 1, 1)) for i in range(n_bars)]
            bars = ax.barh(df['Name'], df['Value'], color=bar_colors, alpha=0.9, height=0.65)
            ax.set_xlabel('Weighted Importance', color='#6b7280', fontsize=9)
            for b in bars:
                ax.text(b.get_width() + 0.0005, b.get_y() + b.get_height() / 2,
                        f"{b.get_width():.4f}", va='center', color='#cbd5e1', fontsize=8)
            plt.tight_layout(pad=1.5)
            charts['features'] = _save(fig)
    except Exception as e:
        logger.warning(f"features chart: {e}")

    # Chart 2 – Explanation Quality Metrics
    try:
        metrics = raw_explanation.get('metrics', {})
        if metrics:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            _style(ax, fig, "Explanation Quality Metrics")
            keys = list(metrics.keys())
            vals = [float(v) for v in metrics.values()]
            palette = ['#f59e0b', '#3b82f6', '#10b981']
            bars = ax.bar(keys, vals, color=palette[:len(keys)], alpha=0.88, width=0.5)
            ax.set_ylim(0, 1.15)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03,
                        f"{v:.2f}", ha='center', color='white', fontsize=11, fontweight='bold')
            plt.tight_layout(pad=1.5)
            charts['metrics'] = _save(fig)
    except Exception as e:
        logger.warning(f"metrics chart: {e}")

    # Chart 3 – Probability Pie
    try:
        probs = raw_explanation['prediction']['probabilities']
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor('#0d1117')
        wedges, texts, autotexts = ax.pie(
            probs, labels=['Authentic', 'Counterfeit'],
            autopct='%1.1f%%', colors=['#10b981', '#ef4444'],
            startangle=90, pctdistance=0.78,
            wedgeprops=dict(linewidth=2.5, edgecolor='#0d1117'),
        )
        for t in texts:
            t.set_color('#9ca3af')
            t.set_fontsize(10)
        for a in autotexts:
            a.set_color('white')
            a.set_fontweight('bold')
            a.set_fontsize(10)
        ax.set_title("Model Probability Split", color='#f1f5f9', fontsize=12,
                     pad=12, fontweight='bold')
        plt.tight_layout(pad=1.5)
        charts['probability'] = _save(fig)
    except Exception as e:
        logger.warning(f"probability chart: {e}")

    return charts


def build_3d_viewer_html(mol: Chem.Mol, atom_scores: np.ndarray,
                         prediction: str, confidence: float, stats: dict) -> str:
    """Build the 3Dmol.js viewer HTML with colors derived from real GNNExplainer scores."""
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except Exception:
        pass
    mol_3d = Chem.RemoveHs(mol_3d)
    mol_block = Chem.MolToMolBlock(mol_3d).replace("\n", "\\n")

    num_atoms = mol.GetNumAtoms()
    norm = normalize_scores(atom_scores)
    sorted_idx = np.argsort(-atom_scores)
    ranks = np.empty(num_atoms, dtype=int)
    ranks[sorted_idx] = np.arange(1, num_atoms + 1)

    AllChem.ComputeGasteigerCharges(mol)

    # IMPORTANT: 3Dmol.js uses 1-based serial numbers for atom identification.
    # We key atom_data by (i+1) so AD[atom.serial] works in the click handler.
    atom_data: Dict[int, dict] = {}
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        charge_str = "N/A"
        if atom.HasProp('_GasteigerCharge'):
            try:
                cv = float(atom.GetProp('_GasteigerCharge'))
                if not (cv != cv):  # isnan check
                    charge_str = f"{cv:.3f}"
            except Exception:
                pass
        hyb = str(atom.GetHybridization()).split('.')[-1]
        serial = i + 1  # 3Dmol.js serial is 1-based
        atom_data[serial] = {
            "serial": serial,   # 1-based – used for 3Dmol setStyle/zoomTo
            "index": i,         # 0-based – used for display only
            "symbol": atom.GetSymbol(),
            "rank": int(ranks[i]),
            "score": round(float(norm[i]), 4),
            "color": score_to_hex(float(norm[i])),
            "group_name": identify_fg(mol, i),
            "properties": {
                "Hybridization": hyb,
                "Charge": charge_str,
                "Is_Aromatic": str(atom.GetIsAromatic()),
                "In_Ring": str(atom.IsInRing()),
                "Neighbors": ", ".join([n.GetSymbol() for n in atom.GetNeighbors()]),
            },
        }

    pred_color = "#10b981" if prediction == "Authentic" else "#ef4444"
    atom_json_str = json.dumps(atom_data)

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:radial-gradient(ellipse at 50% 40%,#0f1322 0%,#080a0f 100%);color:#e2e8f0;overflow:hidden;height:100vh}}
#vp{{width:100%;height:100vh;position:absolute;top:0;left:0;z-index:1}}
.card{{position:absolute;z-index:10;background:rgba(10,12,20,0.88);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;box-shadow:0 8px 40px rgba(0,0,0,0.7);transition:all .3s ease}}
#hdr{{top:16px;left:16px;width:260px}}
.sub{{font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px}}
.hdr-h1{{font-size:15px;font-weight:800;color:#f1f5f9;margin-bottom:12px}}
.badge{{display:flex;align-items:center;justify-content:center;padding:9px 0;border-radius:8px;font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.5px;background:{pred_color}18;color:{pred_color};border:1px solid {pred_color}35;gap:6px}}
#stats{{bottom:16px;left:16px;width:260px;display:grid;grid-template-columns:1fr 1fr;gap:6px}}
.si{{background:rgba(255,255,255,0.03);padding:9px;border-radius:8px;text-align:center;border:1px solid rgba(255,255,255,0.05)}}
.sv{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#f1f5f9}}
.sl{{font-size:9px;color:#4b5563;margin-top:2px;text-transform:uppercase;letter-spacing:.3px}}
#insp{{top:16px;right:16px;width:285px;transform:translateX(130%);opacity:0}}
#insp.on{{transform:translateX(0);opacity:1}}
.rank-ring{{display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:50%;font-weight:800;font-size:14px;border:2px solid #333;color:#fff;flex-shrink:0;transition:border-color .3s,color .3s}}
.pr{{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05)}}
.pk{{font-size:11px;color:#6b7280}}.pv{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#e2e8f0;text-align:right;max-width:165px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
#ctrls{{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);z-index:10;display:flex;gap:8px}}
.btn{{background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.09);color:#cbd5e1;padding:8px 16px;border-radius:30px;cursor:pointer;font-weight:600;font-size:11px;transition:all .2s}}
.btn:hover{{background:rgba(255,255,255,0.14);color:#fff;border-color:rgba(255,255,255,0.2)}}
#cbar{{position:absolute;bottom:60px;left:50%;transform:translateX(-50%);z-index:10;background:rgba(10,12,20,0.85);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:7px 14px;display:flex;align-items:center;gap:10px;font-size:10px;color:#6b7280;pointer-events:none}}
.grad{{width:100px;height:8px;border-radius:4px;background:linear-gradient(to right,#ffffcc,#fed976,#fd8d3c,#e31a1c,#800026)}}
</style></head>
<body>
<div id="vp"></div>
<div id="hdr" class="card">
  <div class="sub">AI Forensics · GNNExplainer</div>
  <div class="hdr-h1">Structure Analysis</div>
  <div class="badge">{prediction}<span style="opacity:.4">·</span>{confidence:.1%}</div>
</div>
<div id="stats" class="card">
  <div class="si"><div class="sv">{stats['mw']}</div><div class="sl">Mol. Weight</div></div>
  <div class="si"><div class="sv">{stats['logp']}</div><div class="sl">LogP</div></div>
  <div class="si"><div class="sv">{stats['tpsa']}</div><div class="sl">TPSA</div></div>
  <div class="si"><div class="sv">{stats['hbd']}/{stats['hba']}</div><div class="sl">H-Bond D/A</div></div>
</div>
<div id="insp" class="card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
    <div id="i-rank" class="rank-ring">#1</div>
    <div>
      <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.5px">Selected</div>
      <div id="i-group" style="font-size:14px;font-weight:700;color:#f1f5f9">Group</div>
    </div>
  </div>
  <div id="i-props"></div>
</div>
<div id="cbar"><span>Low</span><div class="grad"></div><span>High Importance</span></div>
<div id="ctrls">
  <button class="btn" onclick="resetV()">&#8635; Reset</button>
  <button class="btn" onclick="toggleSpin()">&#8634; Rotate</button>
  <button class="btn" onclick="nextStyle()">&#11042; Style</button>
</div>
<script>
const AD = {atom_json_str};
const MB = `{mol_block}`;
let vw, spinning = false, styleIdx = 0;

$(function() {{
  vw = $3Dmol.createViewer($('#vp'), {{ backgroundColor: '#00000000' }});
  vw.addModel(MB, 'mol');
  drawDefault();
  vw.zoomTo();
  vw.render();
  vw.setClickable({{}}, true, function(a) {{
    const d = AD[a.serial];  // a.serial is 1-based, matches our keys
    if (!d) return;
    $('#insp').addClass('on');
    $('#i-rank').text('#' + d.rank).css({{ borderColor: d.color, color: d.color }});
    $('#i-group').text(d.group_name);
    const P = d.properties;
    const rows = [
      ['Element', d.symbol + ' (idx ' + d.index + ')'],
      ['Hybridization', P.Hybridization],
      ['Charge', P.Charge],
      ['Aromatic', P.Is_Aromatic],
      ['In Ring', P.In_Ring],
      ['Neighbors', P.Neighbors],
      ['GNN Score', d.score + '  (rank #' + d.rank + ')'],
    ];
    $('#i-props').html(rows.map(([k,v]) =>
      `<div class="pr"><span class="pk">${{k}}</span><span class="pv">${{v}}</span></div>`
    ).join(''));
    vw.setStyle({{}}, {{ line: {{ color: '#1a1f30', linewidth: 1, opacity: 0.4 }} }});
    vw.setStyle({{ serial: d.serial }}, {{
      sphere: {{ color: d.color, scale: 0.55 }},
      stick:  {{ color: d.color, radius: 0.22 }},
    }});
    vw.zoomTo({{ serial: d.serial }}, 400);
    vw.render();
    window.parent.postMessage({{ type: 'atomClick', ...d }}, '*');
  }});
}});

function drawDefault() {{
  // Base grey sticks for all atoms
  vw.setStyle({{}}, {{ stick: {{ radius: 0.07, color: '#1e2540' }} }});
  // Then override per-atom with GNNExplainer color (use 1-based serial key)
  Object.entries(AD).forEach(([serial, d]) => {{
    const s = d.score;
    vw.setStyle({{ serial: parseInt(serial) }}, {{
      sphere: {{ color: d.color, scale: 0.17 + s * 0.36 }},
      stick:  {{ color: d.color, radius: 0.06 + s * 0.1 }},
    }});
  }});
  vw.render();
}}

function resetV() {{ $('#insp').removeClass('on'); drawDefault(); vw.zoomTo({{}}, 500); }}
function toggleSpin() {{ spinning = !spinning; vw.spin(spinning); }}
function nextStyle() {{
  styleIdx = (styleIdx + 1) % 3;
  if (styleIdx === 0) drawDefault();
  else if (styleIdx === 1) {{ vw.setStyle({{}}, {{ stick: {{ radius: 0.15, colorscheme: 'Jmol' }} }}); vw.render(); }}
  else {{ vw.setStyle({{}}, {{ sphere: {{ scale: 0.35, colorscheme: 'Jmol' }} }}); vw.render(); }}
}}
</script>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    smiles = request.smiles.strip()
    explainer: PyGNativeExplainer = app_state["explainer"]

    graph = GraphConverter.smiles_to_heterograph(smiles)
    if not graph:
        raise HTTPException(422, "SMILES Invalid")

    raw = explainer.explain_molecule(graph, smiles)
    node_imp = raw['explanation']['node_importance']

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(422, "Could not parse molecule")

    # Map node-type local indices → global RDKit atom indices.
    # smiles_to_heterograph groups atoms by symbol in RDKit iteration order,
    # so local_idx k of type T = the k-th atom (0-based) with symbol T in mol.GetAtoms().
    node_type_to_globals: Dict[str, list] = {}
    for global_idx in range(mol.GetNumAtoms()):
        sym = mol.GetAtomWithIdx(global_idx).GetSymbol()
        node_type_to_globals.setdefault(sym, []).append(global_idx)

    all_scores: Dict[int, float] = {}
    for node_type, global_indices in node_type_to_globals.items():
        if node_type in node_imp:
            importance = node_imp[node_type]
            for local_idx, global_idx in enumerate(global_indices):
                if local_idx < len(importance):
                    all_scores[global_idx] = float(importance[local_idx])

    atom_scores = np.array([all_scores.get(i, 0.0) for i in range(mol.GetNumAtoms())])

    stats = {
        "mw": f"{Descriptors.MolWt(mol):.2f}",
        "logp": f"{Descriptors.MolLogP(mol):.2f}",
        "tpsa": f"{Descriptors.TPSA(mol):.2f}",
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
    }
    prediction = raw["prediction"]["class"]
    confidence = raw["prediction"]["confidence"]

    atom_scores_norm = normalize_scores(atom_scores)
    charts = charts_to_b64(raw, mol=mol, atom_scores_norm=atom_scores_norm)
    substructures = _compute_critical_substructures(mol, atom_scores_norm)

    cf = raw.get("counterfactuals", {})
    counterfactual = {
        "text": cf.get("text", ""),
        "impact": cf.get("impact_level", ""),
        "group": cf.get("critical_group", ""),
    }

    return AnalyzeResponse(
        html=build_3d_viewer_html(mol, atom_scores, prediction, confidence, stats),
        heatmap_svg=generate_2d_svg(mol, atom_scores),
        prediction=prediction,
        confidence=confidence,
        charts_features=charts.get("features", ""),
        charts_metrics=charts.get("metrics", ""),
        charts_probability=charts.get("probability", ""),
        substructures=substructures,
        mol_stats=stats,
        counterfactual=counterfactual,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND SPA
# ─────────────────────────────────────────────────────────────────────────────
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PharmaAI · Molecule Analyzer Pro</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
/* ── Reset & Variables ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:      #080a0f;
  --surface: #0d0f18;
  --card:    rgba(255,255,255,0.04);
  --border:  rgba(255,255,255,0.07);
  --accent:  #3b82f6;
  --success: #10b981;
  --danger:  #ef4444;
  --text:    #e2e8f0;
  --muted:   #6b7280;
  --sidebar: 264px;
  --results: 370px;
}
body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ── Loading Overlay ──────────────────────────────────────── */
#loading {
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(8,10,15,0.97);
  backdrop-filter: blur(8px);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 28px;
  transition: opacity .5s ease;
}
#loading.hidden { opacity: 0; pointer-events: none; }
#loading h2 { font-size: 22px; font-weight: 700; color: #f1f5f9; }
#loading p  { color: var(--muted); font-size: 13px; }

/* Atom animation */
.atom {
  position: relative;
  width: 190px; height: 190px;
  display: flex; align-items: center; justify-content: center;
}
.nucleus {
  width: 30px; height: 30px; border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #93c5fd, #1d4ed8);
  box-shadow: 0 0 0 5px rgba(59,130,246,.12), 0 0 24px rgba(59,130,246,.7), 0 0 48px rgba(59,130,246,.25);
  z-index: 2; position: relative;
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 0 5px rgba(59,130,246,.12), 0 0 24px rgba(59,130,246,.7); }
  50%      { box-shadow: 0 0 0 8px rgba(59,130,246,.18), 0 0 40px rgba(59,130,246,.9), 0 0 60px rgba(59,130,246,.3); }
}
.orbit {
  position: absolute;
  border: 1.5px solid rgba(96,165,250,.35);
  border-radius: 50%;
  width: 170px; height: 58px;
  top: 50%; left: 50%;
  margin: -29px 0 0 -85px;
}
.orbit::after {
  content: '';
  position: absolute;
  width: 10px; height: 10px;
  background: #60a5fa;
  border-radius: 50%;
  top: -5px; left: 50%; margin-left: -5px;
  box-shadow: 0 0 8px #60a5fa, 0 0 16px rgba(96,165,250,.5);
}
.orbit-1 { animation: o1 3s linear infinite; }
.orbit-2 { animation: o2 2.4s linear infinite; }
.orbit-3 { animation: o3 3.8s linear infinite; }
@keyframes o1 { from { transform: rotate(0deg);   } to { transform: rotate(360deg);   } }
@keyframes o2 { from { transform: rotate(60deg);  } to { transform: rotate(420deg);  } }
@keyframes o3 { from { transform: rotate(-60deg); } to { transform: rotate(300deg);  } }

.loading-steps {
  display: flex; flex-direction: column; gap: 6px; align-items: center;
  font-size: 12px; color: var(--muted);
}
.step { display: flex; align-items: center; gap: 8px; }
.step-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--muted);
  transition: background .4s;
}
.step-dot.active { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.step-dot.done   { background: var(--success); }

/* ── Header ──────────────────────────────────────────────── */
header {
  display: flex; align-items: center; gap: 14px;
  padding: 0 18px; height: 56px;
  background: rgba(13,15,24,0.98);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0; z-index: 100;
}
.logo {
  font-size: 17px; font-weight: 800; color: #fff;
  letter-spacing: -.3px; flex-shrink: 0;
}
.logo span {
  font-size: 10px; font-weight: 700; color: var(--accent);
  background: rgba(59,130,246,.15); border: 1px solid rgba(59,130,246,.3);
  padding: 1px 6px; border-radius: 4px; margin-left: 5px; vertical-align: middle;
}
#smiles-wrap {
  flex: 1; display: flex; align-items: center; gap: 10px;
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
  border-radius: 9px; padding: 0 14px; height: 36px;
  transition: border-color .2s;
}
#smiles-wrap:focus-within { border-color: rgba(59,130,246,.5); }
#smiles-in {
  flex: 1; background: none; border: none; outline: none;
  color: #f1f5f9; font-family: 'JetBrains Mono', monospace; font-size: 12px;
}
#smiles-in::placeholder { color: var(--muted); }
#analyze-btn {
  padding: 0 20px; height: 36px; border-radius: 8px;
  border: none; background: var(--accent); color: #fff;
  font-weight: 700; font-size: 13px; cursor: pointer;
  flex-shrink: 0; transition: all .2s;
  box-shadow: 0 0 16px rgba(59,130,246,.35);
}
#analyze-btn:hover { background: #2563eb; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(59,130,246,.5); }
#analyze-btn:active { transform: translateY(0); }
#analyze-btn.loading { background: #1e3a8a; pointer-events: none; }
.header-right { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.status-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--success);
  box-shadow: 0 0 8px var(--success); animation: blink 3s ease-in-out infinite;
}
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: .5; } }
.status-text { font-size: 11px; color: var(--muted); }

/* ── Workspace ───────────────────────────────────────────── */
.workspace {
  display: flex; flex: 1; overflow: hidden;
}

/* ── Sidebar ──────────────────────────────────────────────── */
#sidebar {
  width: var(--sidebar);
  background: rgba(13,15,24,0.98);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  overflow: hidden; flex-shrink: 0;
}
.sidebar-hdr {
  padding: 14px 16px 10px;
  font-size: 10px; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: .8px;
  border-bottom: 1px solid var(--border);
}
.catalog-scroll { overflow-y: auto; flex: 1; padding: 8px 0; }
.catalog-scroll::-webkit-scrollbar { width: 4px; }
.catalog-scroll::-webkit-scrollbar-track { background: transparent; }
.catalog-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,.1); border-radius: 2px; }

.cat-group-label {
  padding: 10px 16px 4px;
  font-size: 9px; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: .6px;
}
.mol-item {
  padding: 9px 16px; cursor: pointer;
  border-left: 2px solid transparent;
  transition: all .15s;
}
.mol-item:hover {
  background: rgba(59,130,246,.06);
  border-left-color: var(--accent);
}
.mol-item.active {
  background: rgba(59,130,246,.1);
  border-left-color: var(--accent);
}
.mol-name { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 2px; }
.mol-smiles {
  font-size: 9px; color: var(--muted);
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 220px;
}

/* ── Main Viewer ─────────────────────────────────────────── */
#viewer-area {
  flex: 1; position: relative;
  background: radial-gradient(ellipse at 50% 50%, #0d1020 0%, #080a0f 100%);
  overflow: hidden;
}
#mol-iframe {
  width: 100%; height: 100%; border: none; display: none;
}
#placeholder {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 22px; text-align: center;
  pointer-events: none;
}
#placeholder .atom { width: 140px; height: 140px; opacity: .7; }
#placeholder .atom .orbit { width: 125px; height: 42px; margin: -21px 0 0 -62.5px; }
#placeholder h2 { font-size: 22px; font-weight: 700; color: rgba(241,245,249,.6); }
#placeholder p  { font-size: 13px; color: var(--muted); max-width: 280px; line-height: 1.6; }

/* ── Results Panel ──────────────────────────────────────── */
#results {
  width: var(--results);
  background: rgba(10,12,20,0.97);
  border-left: 1px solid var(--border);
  display: flex; flex-direction: column;
  overflow: hidden; flex-shrink: 0;
  transform: translateX(100%);
  transition: transform .35s cubic-bezier(.4,0,.2,1);
}
#results.open { transform: translateX(0); }

.results-scroll { overflow-y: auto; flex: 1; padding: 14px; gap: 12px; display: flex; flex-direction: column; }
.results-scroll::-webkit-scrollbar { width: 4px; }
.results-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,.08); border-radius: 2px; }

.r-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 14px;
}
.r-card-title {
  font-size: 10px; font-weight: 700; color: var(--muted);
  text-transform: uppercase; letter-spacing: .6px;
  margin-bottom: 12px;
}

/* Prediction card */
.pred-badge {
  display: flex; align-items: center; justify-content: center;
  padding: 12px; border-radius: 9px; gap: 8px;
  font-weight: 700; font-size: 15px; text-transform: uppercase; letter-spacing: .5px;
  margin-bottom: 10px;
}
.pred-badge.auth { background: rgba(16,185,129,.12); color: var(--success); border: 1px solid rgba(16,185,129,.3); }
.pred-badge.fake { background: rgba(239,68,68,.12);  color: var(--danger);  border: 1px solid rgba(239,68,68,.3);  }
.conf-row { display: flex; justify-content: space-between; align-items: center; }
.conf-label { font-size: 11px; color: var(--muted); }
.conf-bar-wrap { flex: 1; mx: 10px; margin: 0 10px; height: 4px; background: rgba(255,255,255,.08); border-radius: 2px; }
.conf-bar { height: 100%; border-radius: 2px; transition: width .5s ease; }
.conf-val { font-size: 11px; font-family: 'JetBrains Mono', monospace; }

/* 2D Heatmap */
#heatmap-container { display: flex; justify-content: center; }
#heatmap-container svg { max-width: 100%; height: auto; border-radius: 6px; background: #fff; }

/* Chart tabs */
.chart-tabs { display: flex; gap: 6px; margin-bottom: 10px; }
.tab-btn {
  flex: 1; padding: 6px; border-radius: 7px; border: 1px solid var(--border);
  background: none; color: var(--muted); font-size: 10px; font-weight: 600;
  cursor: pointer; transition: all .2s; text-transform: uppercase; letter-spacing: .4px;
}
.tab-btn.active { background: rgba(59,130,246,.15); color: var(--accent); border-color: rgba(59,130,246,.35); }
.tab-btn:hover:not(.active) { background: rgba(255,255,255,.05); color: var(--text); }
#chart-img { max-width: 100%; border-radius: 8px; }

/* Substructures */
.ss-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,.04);
}
.ss-item:last-child { border-bottom: none; }
.ss-fg  { font-size: 12px; color: var(--text); font-weight: 600; }
.ss-ctr { font-size: 10px; color: var(--muted); margin-top: 1px; }
.ss-score {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--accent); background: rgba(59,130,246,.1);
  padding: 2px 7px; border-radius: 5px;
}

/* Counterfactual */
.cf-badge {
  display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px;
  border-radius: 5px; font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .4px; margin-bottom: 8px;
}
.cf-badge.High   { background: rgba(239,68,68,.15);   color: #f87171; border: 1px solid rgba(239,68,68,.3); }
.cf-badge.Medium { background: rgba(245,158,11,.15);  color: #fbbf24; border: 1px solid rgba(245,158,11,.3); }
.cf-badge.Low    { background: rgba(16,185,129,.15);  color: #34d399; border: 1px solid rgba(16,185,129,.3); }
.cf-text { font-size: 12px; color: var(--muted); line-height: 1.6; }

/* Results header */
.results-hdr {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  flex-shrink: 0;
}
.results-hdr-title { font-size: 12px; font-weight: 700; color: var(--text); }
.results-close {
  background: none; border: none; color: var(--muted);
  cursor: pointer; font-size: 16px; line-height: 1;
  padding: 4px; border-radius: 5px; transition: color .2s;
}
.results-close:hover { color: var(--text); }

/* Stats grid */
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.stat-item {
  background: rgba(255,255,255,.03); padding: 9px;
  border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,.05);
}
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #f1f5f9; }
.stat-lbl { font-size: 9px; color: var(--muted); margin-top: 2px; text-transform: uppercase; letter-spacing: .3px; }

/* Atom inspector (floating) */
#atom-panel {
  position: fixed; bottom: 20px; right: 20px; z-index: 200;
  background: rgba(10,12,20,.95); backdrop-filter: blur(20px);
  border: 1px solid var(--border); border-radius: 12px;
  padding: 14px; width: 260px;
  box-shadow: 0 8px 32px rgba(0,0,0,.7);
  display: none;
}
.ap-title { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 10px; }
.ap-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,.04); font-size: 11px; }
.ap-k { color: var(--muted); }
.ap-v { font-family: 'JetBrains Mono', monospace; color: #e2e8f0; text-align: right; }
.ap-close { position: absolute; top: 10px; right: 12px; background: none; border: none; color: var(--muted); cursor: pointer; font-size: 14px; }
</style>
</head>
<body>

<!-- Loading Overlay -->
<div id="loading" style="display:none">
  <div class="atom">
    <div class="nucleus"></div>
    <div class="orbit orbit-1"></div>
    <div class="orbit orbit-2"></div>
    <div class="orbit orbit-3"></div>
  </div>
  <div>
    <h2>Analyzing molecule...</h2>
    <p style="text-align:center;margin-top:6px">Running GNNExplainer &middot; 150 epochs</p>
  </div>
  <div class="loading-steps">
    <div class="step"><div id="st1" class="step-dot"></div><span>Converting SMILES to heterograph</span></div>
    <div class="step"><div id="st2" class="step-dot"></div><span>Running GNNExplainer (PyG)</span></div>
    <div class="step"><div id="st3" class="step-dot"></div><span>Computing gradient importance</span></div>
    <div class="step"><div id="st4" class="step-dot"></div><span>Generating visualizations</span></div>
  </div>
</div>

<!-- Header -->
<header>
  <div class="logo">PharmaAI <span>PRO</span></div>
  <div id="smiles-wrap">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" style="flex-shrink:0">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
    <input type="text" id="smiles-in" placeholder="Enter SMILES or select from catalog...">
    <button onclick="clearSmiles()" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:14px;line-height:1;padding:2px" title="Clear">&#x2715;</button>
  </div>
  <button id="analyze-btn" onclick="doAnalyze()">Analyze</button>
  <div class="header-right">
    <div class="status-dot"></div>
    <span class="status-text">Model ready &middot; F1: 0.8143</span>
  </div>
</header>

<!-- Workspace -->
<div class="workspace">

  <!-- Sidebar catalog -->
  <aside id="sidebar">
    <div class="sidebar-hdr">Molecule Catalog</div>
    <div class="catalog-scroll" id="catalog"></div>
  </aside>

  <!-- 3D Viewer -->
  <div id="viewer-area">
    <div id="placeholder">
      <div class="atom">
        <div class="nucleus"></div>
        <div class="orbit orbit-1"></div>
        <div class="orbit orbit-2"></div>
        <div class="orbit orbit-3"></div>
      </div>
      <h2>GNN Forensic Analysis</h2>
      <p>Select a molecule from the catalog or enter a SMILES string, then click Analyze.</p>
    </div>
    <iframe id="mol-iframe"></iframe>
  </div>

  <!-- Results Panel -->
  <aside id="results">
    <div class="results-hdr">
      <span class="results-hdr-title">Analysis Results</span>
      <button class="results-close" onclick="closeResults()" title="Close">&#x2715;</button>
    </div>
    <div class="results-scroll">

      <!-- Prediction -->
      <div class="r-card">
        <div class="r-card-title">Prediction</div>
        <div id="pred-badge" class="pred-badge auth">Authentic</div>
        <div class="conf-row">
          <span class="conf-label">Confidence</span>
          <div class="conf-bar-wrap"><div id="conf-bar" class="conf-bar"></div></div>
          <span id="conf-val" class="conf-val">0%</span>
        </div>
      </div>

      <!-- Mol Stats -->
      <div class="r-card">
        <div class="r-card-title">Molecular Properties</div>
        <div class="stats-grid">
          <div class="stat-item"><div id="s-mw" class="stat-val">—</div><div class="stat-lbl">Mol. Weight</div></div>
          <div class="stat-item"><div id="s-logp" class="stat-val">—</div><div class="stat-lbl">LogP</div></div>
          <div class="stat-item"><div id="s-tpsa" class="stat-val">—</div><div class="stat-lbl">TPSA</div></div>
          <div class="stat-item"><div id="s-hb" class="stat-val">—</div><div class="stat-lbl">H-Bond D/A</div></div>
        </div>
      </div>

      <!-- 2D Heatmap -->
      <div class="r-card">
        <div class="r-card-title">2D Importance Heatmap</div>
        <div id="heatmap-container"></div>
      </div>

      <!-- Charts -->
      <div class="r-card">
        <div class="r-card-title">GNNExplainer Analysis</div>
        <div class="chart-tabs">
          <button class="tab-btn active" onclick="showChart('features')" id="tab-features">Features</button>
          <button class="tab-btn" onclick="showChart('metrics')" id="tab-metrics">Metrics</button>
          <button class="tab-btn" onclick="showChart('probability')" id="tab-probability">Probability</button>
        </div>
        <img id="chart-img" src="" alt="Chart" style="max-width:100%;border-radius:8px;">
      </div>

      <!-- Substructures -->
      <div class="r-card">
        <div class="r-card-title">Critical Substructures</div>
        <div id="substruct-list"></div>
      </div>

      <!-- Counterfactual -->
      <div class="r-card" id="cf-card">
        <div class="r-card-title">Counterfactual Analysis</div>
        <div id="cf-badge-el"></div>
        <p id="cf-text" class="cf-text"></p>
      </div>

    </div>
  </aside>
</div>

<!-- Floating atom inspector -->
<div id="atom-panel">
  <button class="ap-close" onclick="document.getElementById('atom-panel').style.display='none'">&#x2715;</button>
  <div class="ap-title">Atom Inspector</div>
  <div id="ap-rows"></div>
</div>

<script>
// ── Catalog data ────────────────────────────────────────────
const CATALOG = [
  { group: 'Common Drugs', items: [
    { name: 'Aspirin',      smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
    { name: 'Paracetamol',  smiles: 'CC(=O)NC1=CC=C(O)C=C1' },
    { name: 'Ibuprofen',    smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O' },
    { name: 'Metformin',    smiles: 'CN(C)C(=N)NC(=N)N' },
  ]},
  { group: 'Cardiovascular', items: [
    { name: 'Atorvastatin', smiles: 'CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccccc3)c(c4ccc(F)cc4)n1CC(O)CC(O)CC(=O)O' },
    { name: 'Sildenafil',   smiles: 'CCCC1=NN(C)C2=C1NC(=NC2=O)c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC' },
    { name: 'Ramipril',     smiles: 'CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC2CCCCC12' },
    { name: 'Amlodipine',   smiles: 'CCOC(=O)C1=C(COCCN)NC(C)=C(C1c2ccccc2Cl)C(=O)OCC' },
  ]},
  { group: 'Antibiotics', items: [
    { name: 'Amoxicillin',     smiles: 'CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O' },
    { name: 'Ciprofloxacin',   smiles: 'OC(=O)c1cn(C2CC2)c3cc(N4CCNCC4)c(F)cc3c1=O' },
    { name: 'Azithromycin',    smiles: 'CCC1OC(=O)C(CC(CC(C(C(C(C(C1OC2CC(CC(O2)C)N(C)C)C)O)C)OC3C(C(CC(O3)C)N(C)C)O)C)=O)C' },
  ]},
  { group: 'Psychotropic', items: [
    { name: 'Diazepam',        smiles: 'CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13' },
    { name: 'Sertraline',      smiles: 'CNC1CCC(c2ccc(Cl)c(Cl)c2)c3ccccc13' },
    { name: 'Alprazolam',      smiles: 'Cc1nnc2CN=C(c3ccccc3)c4cc(Cl)ccc4-n12' },
  ]},
  { group: 'Oncology', items: [
    { name: 'Imatinib',        smiles: 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(n4)-c5cccnc5' },
    { name: 'Tamoxifen',       smiles: 'CCC(=C(c1ccccc1)c2ccc(OCCN(C)C)cc2)c3ccccc3' },
  ]},
];

// ── Build catalog UI ────────────────────────────────────────
(function buildCatalog() {
  const container = document.getElementById('catalog');
  CATALOG.forEach(group => {
    const lbl = document.createElement('div');
    lbl.className = 'cat-group-label';
    lbl.textContent = group.group;
    container.appendChild(lbl);
    group.items.forEach(mol => {
      const el = document.createElement('div');
      el.className = 'mol-item';
      el.dataset.smiles = mol.smiles;
      el.innerHTML = `<div class="mol-name">${mol.name}</div><div class="mol-smiles">${mol.smiles}</div>`;
      el.onclick = () => selectMolecule(mol.smiles, el);
      container.appendChild(el);
    });
  });
})();

// ── State ────────────────────────────────────────────────────
let currentCharts = {};
let activeChart = 'features';

// ── Loading helpers ──────────────────────────────────────────
function setStep(n) {
  [1,2,3,4].forEach(i => {
    const dot = document.getElementById('st' + i);
    if (i < n) { dot.className = 'step-dot done'; }
    else if (i === n) { dot.className = 'step-dot active'; }
    else { dot.className = 'step-dot'; }
  });
}

async function showLoading() {
  const ov = document.getElementById('loading');
  ov.classList.remove('hidden');
  ov.style.display = 'flex';
  [1,2,3,4].forEach(i => document.getElementById('st' + i).className = 'step-dot');
  await delay(120); setStep(1);
  await delay(300); setStep(2);
}

function hideLoading() {
  setStep(5);
  const ov = document.getElementById('loading');
  ov.classList.add('hidden');
  setTimeout(() => { ov.style.display = 'none'; }, 500);
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Actions ──────────────────────────────────────────────────
function clearSmiles() { document.getElementById('smiles-in').value = ''; }

function selectMolecule(smiles, el) {
  document.querySelectorAll('.mol-item').forEach(e => e.classList.remove('active'));
  if (el) el.classList.add('active');
  document.getElementById('smiles-in').value = smiles;
  document.getElementById('smiles-in').focus();
}

async function doAnalyze() {
  const smiles = document.getElementById('smiles-in').value.trim();
  if (!smiles) return;

  const btn = document.getElementById('analyze-btn');
  btn.classList.add('loading');
  btn.textContent = 'Analyzing...';

  await showLoading();

  try {
    await delay(400); setStep(3);
    const resp = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles }),
    });

    await delay(300); setStep(4);

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Analysis failed');
    }

    const data = await resp.json();
    await delay(200);
    showResults(data);
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    hideLoading();
    btn.classList.remove('loading');
    btn.textContent = 'Analyze';
  }
}

function showResults(data) {
  // 3D viewer
  document.getElementById('placeholder').style.display = 'none';
  const iframe = document.getElementById('mol-iframe');
  iframe.srcdoc = data.html;
  iframe.style.display = 'block';

  // Prediction badge
  const badge = document.getElementById('pred-badge');
  const isAuth = data.prediction === 'Authentic';
  badge.className = 'pred-badge ' + (isAuth ? 'auth' : 'fake');
  badge.innerHTML = (isAuth
    ? '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>'
    : '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
  ) + data.prediction;

  const pct = Math.round(data.confidence * 100);
  document.getElementById('conf-val').textContent = pct + '%';
  const bar = document.getElementById('conf-bar');
  bar.style.width = pct + '%';
  bar.style.background = isAuth ? 'var(--success)' : 'var(--danger)';

  // Mol stats
  const s = data.mol_stats;
  document.getElementById('s-mw').textContent   = s.mw;
  document.getElementById('s-logp').textContent  = s.logp;
  document.getElementById('s-tpsa').textContent  = s.tpsa;
  document.getElementById('s-hb').textContent    = s.hbd + '/' + s.hba;

  // 2D Heatmap
  document.getElementById('heatmap-container').innerHTML = data.heatmap_svg;

  // Charts
  currentCharts = {
    features:    data.charts_features,
    metrics:     data.charts_metrics,
    probability: data.charts_probability,
  };
  showChart(activeChart);

  // Substructures
  const ssList = document.getElementById('substruct-list');
  if (data.substructures && data.substructures.length > 0) {
    ssList.innerHTML = data.substructures.map(s =>
      `<div class="ss-item">
        <div>
          <div class="ss-fg">${s.functional_group.replace(/_/g,' ')}</div>
          <div class="ss-ctr">${s.center_atom} · ${s.size} atoms</div>
        </div>
        <span class="ss-score">${s.importance_score.toFixed(3)}</span>
      </div>`
    ).join('');
  } else {
    ssList.innerHTML = '<p style="color:var(--muted);font-size:12px">No critical substructures identified.</p>';
  }

  // Counterfactual
  const cf = data.counterfactual;
  const cfBadge = document.getElementById('cf-badge-el');
  const cfText  = document.getElementById('cf-text');
  if (cf && cf.text) {
    cfBadge.innerHTML = cf.impact
      ? `<div class="cf-badge ${cf.impact}">${cf.impact} Impact &middot; ${cf.group || ''}</div>`
      : '';
    cfText.textContent = cf.text;
  } else {
    cfBadge.innerHTML = '';
    cfText.textContent = 'No counterfactual data available.';
  }

  // Open results panel
  document.getElementById('results').classList.add('open');
}

function showChart(type) {
  activeChart = type;
  ['features','metrics','probability'].forEach(t => {
    const btn = document.getElementById('tab-' + t);
    btn.className = 'tab-btn' + (t === type ? ' active' : '');
  });
  const img = document.getElementById('chart-img');
  const src = currentCharts[type];
  img.src = src ? ('data:image/png;base64,' + src) : '';
  img.style.display = src ? 'block' : 'none';
}

function closeResults() {
  document.getElementById('results').classList.remove('open');
}

// ── Atom click messages from 3D iframe ──────────────────────
window.addEventListener('message', e => {
  if (e.data && e.data.type === 'atomClick') {
    const d = e.data;
    const panel = document.getElementById('atom-panel');
    const P = d.properties || {};
    document.getElementById('ap-rows').innerHTML = [
      ['Element',       d.symbol + ' (idx ' + d.index + ')'],
      ['GNN Score',     d.score + ' (rank #' + d.rank + ')'],
      ['Group',         d.group_name],
      ['Hybridization', P.Hybridization],
      ['Charge',        P.Charge],
      ['Aromatic',      P.Is_Aromatic],
      ['In Ring',       P.In_Ring],
      ['Neighbors',     P.Neighbors],
    ].map(([k, v]) =>
      `<div class="ap-row"><span class="ap-k">${k}</span><span class="ap-v">${v}</span></div>`
    ).join('');
    panel.style.display = 'block';
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
