"""
Benchmark script for two tables:
  Table 1 — Per-category F1 and FPR (HGT vs GAT)
  Table 2 — Explainability hit rate (80 bootstrap runs, top-3 substructures)

Run from the Counterfeit-Detector directory:
    python benchmark_tables.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from rdkit import Chem
from rdkit.Chem import rdFMCS
from torch_geometric.data import HeteroData

warnings.filterwarnings("ignore")

# ─── category mapping (dataset counterfeit_type → high-level category) ────────
CATEGORY_MAP = {
    # bioisostere
    'methyl_to_ethyl_on_aryl':   'bioisostere',
    'OMe_to_OEt':                'bioisostere',
    'COOH_to_tetrazole':         'bioisostere',
    'ester_to_amide':            'bioisostere',
    'amide_to_sulfonamide':      'bioisostere',
    'NH2_to_NHMe':               'bioisostere',
    'Cl_to_CF3_aryl':            'bioisostere',
    'ketone_to_sulfoxide':       'bioisostere',
    'piperidine_to_morpholine':  'bioisostere',
    # low-effort substitution
    'aryl_CH_to_N_para':         'low-effort',
    'aryl_C_to_N':               'low-effort',
    'F_to_Cl_aryl':              'low-effort',
    'Br_to_Cl_aryl':             'low-effort',
    # scaffold hop
    'alkene_to_alkane':          'scaffold-hop',
    'cyclopentane_to_cyclohexane': 'scaffold-hop',
}
CATEGORY_LABELS = ['bioisostere', 'low-effort', 'scaffold-hop']

HGT_THRESHOLD = 0.692
GAT_THRESHOLD = 0.50
HIT_SAMPLE    = 250   # max counterfeits for hit-rate evaluation (speed)
BOOTSTRAP_N   = 80    # bootstrap runs for mean±std

# ─── dataset ──────────────────────────────────────────────────────────────────
def load_dataset():
    for p in ['./intelligent_pharma_50k_v2.pt', './intelligent_pharma_50k.pt',
              './enhanced_pharma_v2_20260401_191857.pt']:
        if Path(p).exists():
            data = torch.load(p, map_location='cpu', weights_only=False)
            ds = data[0] if isinstance(data, (list, tuple)) else data
            print(f"Dataset: {p}  ({len(ds)} graphs)")
            return ds
    raise FileNotFoundError("No dataset found")

def split_dataset(ds):
    labels = [int(g.y.item()) for g in ds]
    idx = list(range(len(ds)))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    return [ds[i] for i in test_idx]

# ─── HGT model ────────────────────────────────────────────────────────────────
def load_hgt(model_dir='./HGT_Enhanced_Results'):
    from hyp import RobustEnhancedHGTDetector
    files = sorted(Path(model_dir).glob('best_model_f1_*.pt'),
                   key=lambda p: float(p.stem.split('f1_')[1]))
    ckpt = torch.load(files[-1], map_location='cpu', weights_only=False)
    model = RobustEnhancedHGTDetector(ckpt['feature_dims']).eval()
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"HGT: {files[-1].name}")
    return model

# ─── GAT model ────────────────────────────────────────────────────────────────
def load_gat(model_dir='./Homo_GNN_Results_Clean'):
    from homo_gnn_clean import HomoGATDetector, hetero_to_homo, ATOM_TYPES
    files = sorted(Path(model_dir).glob('best_model_f1_*.pt'),
                   key=lambda p: float(p.stem.split('f1_')[1]))
    ckpt = torch.load(files[-1], map_location='cpu', weights_only=False)
    # infer in_channels from state dict
    in_ch = ckpt['model_state_dict']['input_proj.weight'].shape[1]
    model = HomoGATDetector(in_channels=in_ch).eval()
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"GAT: {files[-1].name}  (in_channels={in_ch})")
    return model, hetero_to_homo, ATOM_TYPES

# ─── inference helpers ─────────────────────────────────────────────────────────
def hgt_predict(model, graph, threshold=HGT_THRESHOLD):
    with torch.no_grad():
        logits = model(graph.x_dict, graph.edge_index_dict, None, batch_size=1)
        prob = F.softmax(logits, dim=1)[0, 1].item()
    return int(prob >= threshold), prob

def gat_predict(model, homo_fn, graph, threshold=GAT_THRESHOLD):
    from homo_gnn_clean import ATOM_TYPES
    fd = {nt: graph[nt].x.shape[1] for nt in graph.node_types
          if hasattr(graph[nt], 'x')}
    homo = homo_fn(graph, fd)
    if homo is None:
        return 0, 0.5
    homo.batch = torch.zeros(homo.x.size(0), dtype=torch.long)
    with torch.no_grad():
        logits = model(homo.x, homo.edge_index, homo.batch)
        prob = F.softmax(logits, dim=1)[0, 1].item()
    return int(prob >= threshold), prob

# ─── per-category metrics ─────────────────────────────────────────────────────
def per_category_metrics(test_data, hgt_model, gat_model, gat_homo_fn):
    """Returns dict: category → {hgt_f1, gat_f1, hgt_fpr, gat_fpr}"""
    results = {cat: {'yt': [], 'hgt': [], 'gat': []} for cat in CATEGORY_LABELS}

    # authentics — same set reused per category
    auth = [g for g in test_data if int(g.y.item()) == 0]
    print(f"  {len(auth)} authentic test molecules")

    for cat in CATEGORY_LABELS:
        cfs = [g for g in test_data
               if int(g.y.item()) == 1
               and CATEGORY_MAP.get(getattr(g, 'counterfeit_type', ''), '') == cat]
        print(f"  {cat}: {len(cfs)} counterfeits", end='', flush=True)

        rng = np.random.RandomState(42)
        # balanced: sample authentics to match counterfeit count
        n = min(len(cfs), len(auth))
        auth_sub = [auth[i] for i in rng.choice(len(auth), n, replace=False)]
        subset = auth_sub + cfs

        for g in subset:
            label = int(g.y.item())
            hp, _ = hgt_predict(hgt_model, g)
            gp, _ = gat_predict(gat_model, gat_homo_fn, g)
            results[cat]['yt'].append(label)
            results[cat]['hgt'].append(hp)
            results[cat]['gat'].append(gp)
        print(' ✓')

    out = {}
    for cat, r in results.items():
        yt, h, ga = np.array(r['yt']), np.array(r['hgt']), np.array(r['gat'])
        tn_h, fp_h, fn_h, tp_h = confusion_matrix(yt, h, labels=[0,1]).ravel()
        tn_g, fp_g, fn_g, tp_g = confusion_matrix(yt, ga, labels=[0,1]).ravel()
        out[cat] = {
            'hgt_f1':  f1_score(yt, h, zero_division=0),
            'gat_f1':  f1_score(yt, ga, zero_division=0),
            'hgt_fpr': fp_h / (fp_h + tn_h + 1e-9),
            'gat_fpr': fp_g / (fp_g + tn_g + 1e-9),
        }
    return out

# ─── explainability hit rate ──────────────────────────────────────────────────
def find_modified_atoms(orig_smiles, mod_smiles):
    m1 = Chem.MolFromSmiles(orig_smiles)
    m2 = Chem.MolFromSmiles(mod_smiles)
    if m1 is None or m2 is None:
        return set()
    res = rdFMCS.FindMCS(
        [m1, m2], timeout=3,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False, ringMatchesRingOnly=False,
    )
    if res.numAtoms == 0:
        return set(range(m2.GetNumAtoms()))
    pat = Chem.MolFromSmarts(res.smartsString)
    match = m2.GetSubstructMatch(pat)
    return set(range(m2.GetNumAtoms())) - set(match)


def top3_atoms(explanation_ni, smiles, explainer_obj):
    """Get top-3 substructure atom sets using a single node_importance source."""
    subs = explainer_obj._identify_critical_substructures(
        None,  # graph not used — we pass smiles directly
        smiles,
        {'node_importance': explanation_ni},
        top_k=3,
    )
    return [s['atom_indices'] for s in subs]


def _node_imp_to_atom_scores(ni_dict, mol):
    """Convert node_importance (per type) to array indexed by RDKit atom idx."""
    atom_to_type = {a.GetIdx(): a.GetSymbol() for a in mol.GetAtoms()}
    type_counters = {}
    scores = {}
    for idx in range(mol.GetNumAtoms()):
        sym = atom_to_type[idx]
        pos = type_counters.get(sym, 0)
        if sym in ni_dict and pos < len(ni_dict[sym]):
            scores[idx] = float(ni_dict[sym][pos])
        type_counters[sym] = pos + 1
    return scores


def top3_from_scores(scores_dict, mol, top_k=3):
    """Return top-k atom index sets (atom + neighbours), sorted by score."""
    if not scores_dict:
        return []
    ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    subs = []
    visited = set()
    for atom_idx, score in ranked[:top_k * 2]:
        if atom_idx in visited:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        group = {atom_idx} | {n.GetIdx() for n in atom.GetNeighbors()}
        subs.append(group)
        visited.update(group)
        if len(subs) >= top_k:
            break
    return subs


def hits_overlap(top3_sets, modified_atoms):
    return any(bool(s & modified_atoms) for s in top3_sets)


def hit_rate_eval(test_data, hgt_model, hgt_dir='./HGT_Enhanced_Results',
                  n_sample=HIT_SAMPLE):
    from hyp import PyGNativeExplainer, HGTAttentionExtractor
    from explicablity import GraphConverter

    device = torch.device('cpu')
    explainer = PyGNativeExplainer(hgt_model, device)

    # Collect correctly-classified counterfeits with original_smiles
    cands = []
    for g in test_data:
        if int(g.y.item()) != 1:
            continue
        orig = getattr(g, 'original_smiles', None)
        if orig is None or (isinstance(orig, float)):
            continue
        pred, _ = hgt_predict(hgt_model, g)
        if pred == 1:
            cands.append(g)
    print(f"  Correctly classified counterfeits with ground truth: {len(cands)}")

    rng = np.random.RandomState(42)
    if len(cands) > n_sample:
        cands = [cands[i] for i in rng.choice(len(cands), n_sample, replace=False)]
    print(f"  Using {len(cands)} for hit-rate eval...")

    per_mol = []  # list of {gnn, grad, hgt, fusion}
    for k, g in enumerate(cands):
        if (k + 1) % 25 == 0:
            print(f"    {k+1}/{len(cands)}", flush=True)
        smiles = g.smiles
        orig   = g.original_smiles
        modified = find_modified_atoms(orig, smiles)
        if not modified:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        graph = GraphConverter.smiles_to_heterograph(smiles)
        if graph is None:
            continue

        try:
            raw = explainer.explain_molecule(graph, smiles)
        except Exception:
            continue

        def top3(ni_dict):
            scores = _node_imp_to_atom_scores(ni_dict, mol)
            return top3_from_scores(scores, mol, top_k=3)

        gnn_ni   = (raw.get('gnn_explainer_raw') or {}).get('node_importance', {})
        grad_ni  = (raw.get('gradient_raw') or {}).get('node_importance', {})
        comb_ni  = raw['explanation']['node_importance']

        # HGT attention → convert node_attention to node_importance format
        hgt_attn = raw.get('hgt_attention') or {}
        hgt_ni   = hgt_attn.get('node_attention', {})  # same key structure

        per_mol.append({
            'gnn':    hits_overlap(top3(gnn_ni), modified),
            'grad':   hits_overlap(top3(grad_ni), modified),
            'hgt':    hits_overlap(top3(hgt_ni), modified),
            'fusion': hits_overlap(top3(comb_ni), modified),
        })

    # 80 bootstrap samples
    n = len(per_mol)
    print(f"  Valid molecules for bootstrap: {n}")
    keys = ['gnn', 'grad', 'hgt', 'fusion']
    boot = {k: [] for k in keys}
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(n, n, replace=True)
        for k in keys:
            hits = sum(per_mol[i][k] for i in sample)
            boot[k].append(hits / n)

    return {k: (float(np.mean(boot[k])), float(np.std(boot[k]))) for k in keys}


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Loading data...")
    ds = load_dataset()
    test_data = split_dataset(ds)
    print(f"Test set: {len(test_data)} graphs")

    print("\nLoading models...")
    hgt_model = load_hgt()
    gat_model, gat_homo_fn, _ = load_gat()

    # ── Table 1 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 1: Per-Category F1 and False-Positive Rate")
    print("=" * 60)
    cat_results = per_category_metrics(test_data, hgt_model, gat_model, gat_homo_fn)

    total_cfs = sum(
        1 for g in ds if int(g.y.item()) == 1
        and CATEGORY_MAP.get(getattr(g, 'counterfeit_type', ''), '') != ''
    )

    header = f"{'Category':<28} {'HGT F1':>8} {'GAT F1':>8} {'HGT FPR':>9} {'GAT FPR':>9}"
    print(header)
    print("-" * len(header))
    for cat in CATEGORY_LABELS:
        n_cat = sum(
            1 for g in ds if int(g.y.item()) == 1
            and CATEGORY_MAP.get(getattr(g, 'counterfeit_type', ''), '') == cat
        )
        pct = 100 * n_cat / total_cfs if total_cfs else 0
        r = cat_results[cat]
        label = f"{cat} ({pct:.1f}%)"
        print(f"{label:<28} {r['hgt_f1']:>8.3f} {r['gat_f1']:>8.3f} "
              f"{r['hgt_fpr']:>9.3f} {r['gat_fpr']:>9.3f}")

    # ── Table 2 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"TABLE 2: Explainability Hit Rate (top-3, {BOOTSTRAP_N} bootstrap runs)")
    print("=" * 60)
    hr = hit_rate_eval(test_data, hgt_model)

    rows = [
        ('GNNExplainer alone',           hr['gnn']),
        ('Gradient saliency alone',       hr['grad']),
        ('Native HGT attention alone',    hr['hgt']),
        ('Geometric-mean fusion',         hr['fusion']),
    ]
    print(f"{'Method':<35} {'Hit rate':>20}")
    print("-" * 57)
    for name, (mean, std) in rows:
        print(f"{name:<35} {mean:.2f} ± {std:.2f}")

    # ── LaTeX output ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LaTeX Table 1:")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Category} & \textbf{HGT F1} & \textbf{GAT F1} & "
          r"\textbf{HGT FPR} & \textbf{GAT FPR} \\")
    print(r"\midrule")
    for cat in CATEGORY_LABELS:
        n_cat = sum(1 for g in ds if int(g.y.item()) == 1
                    and CATEGORY_MAP.get(getattr(g, 'counterfeit_type', ''), '') == cat)
        pct = 100 * n_cat / total_cfs if total_cfs else 0
        r = cat_results[cat]
        name_map = {'bioisostere': 'Bioisosteres', 'low-effort': 'Low-effort',
                    'scaffold-hop': 'Scaffold hops'}
        print(f"{name_map[cat]} ({pct:.1f}\\%) & "
              f"{r['hgt_f1']:.3f} & {r['gat_f1']:.3f} & "
              f"{r['hgt_fpr']:.3f} & {r['gat_fpr']:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\nLaTeX Table 2:")
    print(r"\begin{tabular}{lc}")
    print(r"\toprule")
    print(r"\textbf{Attribution source} & \textbf{Hit rate} \\")
    print(r"\midrule")
    latex_names = [
        ('GNNExplainer alone',               'gnn'),
        ('Gradient saliency alone',           'grad'),
        ('Native HGT attention alone',        'hgt'),
        (r'Geometric-mean fusion (this work)','fusion'),
    ]
    for name, key in latex_names:
        mean, std = hr[key]
        bold_open  = r'\mathbf{' if key == 'fusion' else ''
        bold_close = r'}' if key == 'fusion' else ''
        print(f"{name} & ${bold_open}{mean:.2f} \\pm {std:.2f}{bold_close}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
