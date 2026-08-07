import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

import torch
print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda}")

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} | CC: {cap}")
    
    if cap[0] < 7:
        print("\n" + "!"*60)
        print("EROARE DE COMPATIBILITATE: P100 (CC 6.0) nu suportă PyTorch 2.10+")
        print("SOLUȚIE: Schimbă Acceleratorul din 'P100' în 'T4' în setările Kaggle.")
        print("T4 este mai rapid și compatibil.")
        print("Încerc downgrade la PyTorch 2.4.1 pentru suport P100...")
        print("!"*60 + "\n")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.4.1', 'torch_geometric', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            print("Downgrade reușit. TE ROG DĂ 'RESTART & RUN ALL'.")
            sys.exit(0) 
        except Exception as e:
            print(f"Downgrade eșuat: {e}")
    else:
        try:
            _t = torch.randn(1, 1).to('cuda')
            _ = torch.nn.functional.layer_norm(_t, (1,))
            print("GPU sanity: OK")
        except Exception as e:
            print(f"GPU sanity: FAILED ({e})")

print("Installing PyG...")
pip('torch_geometric')
print("Done.")

# ── Imports ───────────────────────────────────────────────────────────────────
import os, json, random, warnings
import numpy as np
from pathlib import Path

import torch.nn.functional as F
from torch_geometric.nn import HGTConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy import stats

warnings.filterwarnings('ignore')

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path('/kaggle/working')
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
N_SEEDS    = 5
N_EPOCHS   = 60
PATIENCE   = 10
BATCH_SIZE = 128
LR, WD     = 3e-4, 0.01
ALPHA_LOSS, GAMMA_LOSS = 0.75, 2.0

ATOM_TYPES_11 = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'B']
ATOM_TYPES_9  = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
ATOM_IDX_11   = {a: i for i, a in enumerate(ATOM_TYPES_11)}
ATOM_IDX_9    = {a: i for i, a in enumerate(ATOM_TYPES_9)}

CATEGORY_MAP = {
    'methyl_to_ethyl_on_aryl': 'bioisostere', 'OMe_to_OEt': 'bioisostere',
    'COOH_to_tetrazole': 'bioisostere',        'ester_to_amide': 'bioisostere',
    'amide_to_sulfonamide': 'bioisostere',     'NH2_to_NHMe': 'bioisostere',
    'Cl_to_CF3_aryl': 'bioisostere',           'ketone_to_sulfoxide': 'bioisostere',
    'piperidine_to_morpholine': 'bioisostere', 'aryl_CH_to_N_para': 'low-effort',
    'aryl_C_to_N': 'low-effort',               'F_to_Cl_aryl': 'low-effort',
    'Br_to_Cl_aryl': 'low-effort',             'alkene_to_alkane': 'scaffold-hop',
    'cyclopentane_to_cyclohexane': 'scaffold-hop',
    # v3 dataset names
    'ArH_to_ArMe': 'low-effort',     'ArH_to_ArF': 'low-effort',
    'ArCl_to_ArF': 'low-effort',     'ArBr_to_ArCl': 'low-effort',
    'ArCH3_to_ArCF3': 'bioisostere', 'COOH_to_SO2NH2': 'bioisostere',
    'NH2_to_OH': 'bioisostere',      'CH2_to_O': 'bioisostere',
    'phenyl_to_pyridyl': 'scaffold-hop',
}
CATEGORIES = ['bioisostere', 'low-effort', 'scaffold-hop']

# ── Dataset ───────────────────────────────────────────────────────────────────
def find_dataset():
    for p in Path('/kaggle/input').rglob('*.pt'):
        if 'best_model' not in str(p):
            return str(p)
    raise FileNotFoundError("No .pt dataset found in /kaggle/input/")

def load_dataset(path):
    print(f"Loading: {path}")
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        data = torch.load(path, map_location='cpu')
    if isinstance(data, (list, tuple)):
        graphs, labels = data[0], data[1]
    else:
        graphs = data.get('hetero_graphs') or data.get('graphs')
        labels = data.get('labels')
    labels = [int(l) for l in labels]
    print(f"  {len(graphs)} graphs, auth={sum(1 for l in labels if l==0)}, "
          f"fake={sum(1 for l in labels if l==1)}")
    return graphs, labels

def extract_ctypes(graphs):
    """Extract counterfeit_type strings into a plain list (one per graph)."""
    ctypes = []
    for g in graphs:
        ct = ''
        for attr in ['counterfeit_type']:
            try:
                val = getattr(g, attr, None)
                if isinstance(val, str) and val:
                    ct = val; break
            except Exception:
                pass
        ctypes.append(ct)
    return ctypes

def add_onehot_11(graphs, feature_dims):
    """Append 11-bit one-hot atom-type: 26 → 37 dim, in-place."""
    n = len(ATOM_TYPES_11)
    for g in graphs:
        for nt in g.node_types:
            if not hasattr(g[nt], 'x') or g[nt].x is None:
                continue
            n_nodes = g[nt].x.size(0)
            oh = torch.zeros(n_nodes, n)
            oh[:, ATOM_IDX_11.get(nt, n-1)] = 1.0
            g[nt].x = torch.cat([g[nt].x, oh], dim=1)
    return {k: v + n for k, v in feature_dims.items()}

def hetero_to_homo(g, base_dim=26):
    n9 = len(ATOM_TYPES_9)
    all_x, offsets, offset = [], {}, 0
    for at in ATOM_TYPES_9:
        if at not in g.node_types: continue
        if not hasattr(g[at], 'x') or g[at].x.size(0) == 0: continue
        x = g[at].x.float()
        fd = x.size(1)
        if fd < base_dim:   x = F.pad(x, (0, base_dim - fd))
        elif fd > base_dim: x = x[:, :base_dim]
        oh = torch.zeros(x.size(0), n9)
        oh[:, ATOM_IDX_9.get(at, 0)] = 1.0
        all_x.append(torch.cat([x, oh], dim=1))
        offsets[at] = offset
        offset += x.size(0)
    if not all_x: return None
    x_all = torch.cat(all_x, dim=0)
    all_edges = []
    for et in g.edge_types:
        st, _, dt = et
        if st not in offsets or dt not in offsets: continue
        if not hasattr(g[et], 'edge_index') or g[et].edge_index.size(1) == 0: continue
        ei = g[et].edge_index.clone()
        ei[0] += offsets[st]; ei[1] += offsets[dt]
        all_edges.append(ei)
    if not all_edges: return None
    return Data(x=x_all, edge_index=torch.cat(all_edges, dim=1), y=g.y)

# ── Models ────────────────────────────────────────────────────────────────────
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=ALPHA_LOSS, gamma=GAMMA_LOSS):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inp, tgt):
        ce = F.cross_entropy(inp, tgt, reduction='none')
        pt = torch.exp(-ce)
        at = torch.where(tgt == 1, self.alpha, 1 - self.alpha)
        return (at * (1 - pt) ** self.gamma * ce).mean()

class HGTModel(torch.nn.Module):
    def __init__(self, feature_dims, edge_types, hidden=128, heads=4, layers=3, dropout=0.15):
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden = hidden
        self.dropout = dropout
        nts = list(feature_dims.keys())
        ets = edge_types
        self.proj   = torch.nn.ModuleDict({nt: torch.nn.Linear(dim, hidden) for nt, dim in feature_dims.items()})
        self.pnorm  = torch.nn.ModuleDict({nt: torch.nn.LayerNorm(hidden) for nt in feature_dims})
        self.convs  = torch.nn.ModuleList([HGTConv(hidden, hidden, (nts, ets), heads) for _ in range(layers)])
        self.lnorms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(layers)])
        self.node_attn = torch.nn.Parameter(torch.ones(len(nts)))
        clf_in = hidden * 2 * len(nts)
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(clf_in, hidden*2), torch.nn.BatchNorm1d(hidden*2), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden*2, hidden),  torch.nn.BatchNorm1d(hidden),  torch.nn.ReLU(), torch.nn.Dropout(dropout*0.7),
            torch.nn.Linear(hidden, hidden//2), torch.nn.BatchNorm1d(hidden//2),torch.nn.ReLU(), torch.nn.Dropout(dropout*0.5),
            torch.nn.Linear(hidden//2, 2),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: torch.nn.init.zeros_(m.bias)

    def forward(self, x_dict, edge_index_dict, batch_dict=None, batch_size=1):
        dev = next(self.parameters()).device
        nts = list(self.feature_dims.keys())
        h = {nt: F.relu(self.pnorm[nt](self.proj[nt](x_dict[nt])))
             if nt in x_dict and x_dict[nt].size(0) > 0
             else torch.zeros(0, self.hidden, device=dev) for nt in nts}
        for conv, ln in zip(self.convs, self.lnorms):
            try:
                hn = conv(h, edge_index_dict)
                for nt in hn:
                    if hn[nt].size(0) > 0:
                        r = F.dropout(F.relu(ln(hn[nt])), self.dropout, self.training)
                        hn[nt] = r + h[nt] if h.get(nt, torch.empty(0)).shape == r.shape else r
                h = hn
            except Exception as e:
                print(f"    [HGTConv warning] {e}")
                break
        if batch_dict:
            bs = max((int(b.max())+1) for b in batch_dict.values() if b.numel() > 0)
        else:
            bs = batch_size
        wa = F.softmax(self.node_attn, dim=0)
        pools = []
        for i, nt in enumerate(nts):
            if nt in h and h[nt].size(0) > 0:
                nf = h[nt]
                if batch_dict and nt in batch_dict and batch_dict[nt].numel() > 0:
                    pm = torch.zeros(bs, self.hidden, device=dev)
                    px = torch.zeros(bs, self.hidden, device=dev)
                    for bi in range(bs):
                        m = batch_dict[nt] == bi
                        if m.any(): pm[bi] = nf[m].mean(0); px[bi] = nf[m].max(0).values
                else:
                    pm = nf.mean(0, keepdim=True).expand(bs, -1)
                    px = nf.max(0, keepdim=True).values.expand(bs, -1)
                pools.append(torch.cat([pm, px], 1) * wa[i])
            else:
                pools.append(torch.zeros(bs, self.hidden*2, device=dev))
        return self.clf(torch.cat(pools, 1))

class GATModel(torch.nn.Module):
    def __init__(self, in_ch, hidden=128, heads=4, layers=3, dropout=0.15):
        super().__init__()
        self.proj  = torch.nn.Linear(in_ch, hidden)
        self.pnorm = torch.nn.LayerNorm(hidden)
        self.convs = torch.nn.ModuleList([GATConv(hidden, hidden, heads=heads, dropout=dropout, concat=False) for _ in range(layers)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = dropout
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(hidden*2, hidden*2), torch.nn.BatchNorm1d(hidden*2), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden*2, hidden),   torch.nn.BatchNorm1d(hidden),   torch.nn.ReLU(), torch.nn.Dropout(dropout*0.7),
            torch.nn.Linear(hidden, hidden//2),  torch.nn.BatchNorm1d(hidden//2),torch.nn.ReLU(), torch.nn.Dropout(dropout*0.5),
            torch.nn.Linear(hidden//2, 2),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: torch.nn.init.zeros_(m.bias)
    def forward(self, x, edge_index, batch):
        h = F.relu(self.pnorm(self.proj(x)))
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.relu(norm(conv(h, edge_index))), self.dropout, self.training) + h
        return self.clf(torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], 1))

# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def compute_metrics(preds, labels, ctypes):
    if not preds or not labels:
        return {'f1': 0.0, 'acc': 0.0, 'fnr': 0.0,
                'fnr_cat': {c: float('nan') for c in CATEGORIES},
                'f1_cat':  {c: float('nan') for c in CATEGORIES}}
    preds  = np.array(preds)
    labels = np.array(labels)
    f1  = float(f1_score(labels, preds, zero_division=0))
    acc = float(accuracy_score(labels, preds))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    fnr = float(fn / (fn + tp + 1e-9))
    fnr_cat, f1_cat = {}, {}
    for cat in CATEGORIES:
        fake_mask = [i for i, (lbl, ct) in enumerate(zip(labels, ctypes))
                     if lbl == 1 and CATEGORY_MAP.get(ct, '') == cat]
        fnr_cat[cat] = float(np.mean(preds[fake_mask] == 0)) if fake_mask else float('nan')
        # F1 on subset: authentic + fakes of this category
        auth_mask = [i for i, lbl in enumerate(labels) if lbl == 0]
        sub_idx   = auth_mask + fake_mask
        if len(fake_mask) > 0:
            f1_cat[cat] = float(f1_score(labels[sub_idx], preds[sub_idx], zero_division=0))
        else:
            f1_cat[cat] = float('nan')
    return {'f1': f1, 'acc': acc, 'fnr': fnr, 'fnr_cat': fnr_cat, 'f1_cat': f1_cat}

# ── Training ──────────────────────────────────────────────────────────────────
def run_hgt_seed(train_g, val_g, test_g, test_ctypes, feat_dims_37, edge_types, seed, device=None):
    dev = device or DEVICE
    set_seed(seed)
    train_l = DataLoader(train_g, BATCH_SIZE, shuffle=True,  num_workers=2)
    val_l   = DataLoader(val_g,   BATCH_SIZE, shuffle=False, num_workers=2)
    test_l  = DataLoader(test_g,  BATCH_SIZE, shuffle=False, num_workers=2)

    model  = HGTModel(feat_dims_37, edge_types).to(dev)
    base   = model
    crit   = FocalLoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 5, min_lr=1e-6)

    best_val, best_state, pat = -1.0, None, 0
    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_l:
            batch = batch.to(dev)
            opt.zero_grad()
            out  = base(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch.num_graphs)
            loss = crit(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_l:
                batch = batch.to(dev)
                out = base(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch.num_graphs)
                vp.extend(out.argmax(1).cpu().tolist())
                vl.extend(batch.y.view(-1).cpu().tolist())
        val_f1 = f1_score(vl, vp, zero_division=0) if vl else 0.0
        sched.step(val_f1)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.clone() for k, v in base.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            print(f"  HGT early stop epoch={epoch+1}, best_val_f1={best_val:.4f}")
            break

    base.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_l:
            batch = batch.to(dev)
            out = base(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch.num_graphs)
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(batch.y.view(-1).cpu().tolist())
    return compute_metrics(preds, labels, test_ctypes)


def run_gat_seed(train_g, val_g, test_g, test_ctypes, base_dim, seed, device=None):
    def conv(gs):
        return [h for g in gs for h in [hetero_to_homo(g, base_dim)] if h is not None]

    set_seed(seed)
    train_h, val_h, test_h = conv(train_g), conv(val_g), conv(test_g)
    if not train_h: raise RuntimeError("GAT: empty train set")

    in_ch   = train_h[0].x.size(1)
    train_l = DataLoader(train_h, BATCH_SIZE, shuffle=True,  num_workers=2)
    val_l   = DataLoader(val_h,   BATCH_SIZE, shuffle=False, num_workers=2)
    test_l  = DataLoader(test_h,  BATCH_SIZE, shuffle=False, num_workers=2)

    dev    = device or DEVICE
    model  = GATModel(in_ch).to(dev)
    base   = model
    crit   = FocalLoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 5, min_lr=1e-6)

    best_val, best_state, pat = -1.0, None, 0
    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_l:
            batch = batch.to(dev)
            opt.zero_grad()
            out  = model(batch.x, batch.edge_index, batch.batch)
            loss = crit(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_l:
                batch = batch.to(dev)
                out = model(batch.x, batch.edge_index, batch.batch)
                vp.extend(out.argmax(1).cpu().tolist())
                vl.extend(batch.y.view(-1).cpu().tolist())
        val_f1 = f1_score(vl, vp, zero_division=0) if vl else 0.0
        sched.step(val_f1)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.clone() for k, v in base.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            print(f"  GAT early stop epoch={epoch+1}, best_val_f1={best_val:.4f}")
            break

    base.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_l:
            batch = batch.to(dev)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(batch.y.view(-1).cpu().tolist())
    return compute_metrics(preds, labels, test_ctypes)

# ── Main ──────────────────────────────────────────────────────────────────────
ds_path = find_dataset()
graphs, labels = load_dataset(ds_path)

# Extract counterfeit types BEFORE adding one-hot (strings still present)
print("Extracting counterfeit types...")
all_ctypes = extract_ctypes(graphs)
print(f"  Unique types: {set(CATEGORY_MAP.get(ct,'?') for ct in all_ctypes if ct)}")

# Feature dims
feat_dims_26 = {}
for g in graphs[:100]:
    for nt in g.node_types:
        if hasattr(g[nt], 'x') and g[nt].x.size(0) > 0:
            if nt not in feat_dims_26:
                feat_dims_26[nt] = g[nt].x.size(1)
for nt in ATOM_TYPES_11:
    if nt not in feat_dims_26:
        feat_dims_26[nt] = 26

print("Adding atom-type one-hot (26 → 37)...")
feat_dims_37 = add_onehot_11(graphs, dict(feat_dims_26))
print(f"  HGT dim={next(iter(feat_dims_37.values()))}, GAT dim={next(iter(feat_dims_26.values()))+9}")

def clean_graph(g):
    """Rebuild HeteroData keeping only tensors — no string attrs."""
    ng = HeteroData()
    for nt in g.node_types:
        try:
            if hasattr(g[nt], 'x') and g[nt].x is not None:
                ng[nt].x = g[nt].x
        except Exception: pass
    for et in g.edge_types:
        try:
            if hasattr(g[et], 'edge_index') and g[et].edge_index is not None:
                ng[et].edge_index = g[et].edge_index
            if hasattr(g[et], 'edge_attr') and g[et].edge_attr is not None:
                ng[et].edge_attr = g[et].edge_attr
        except Exception: pass
    if hasattr(g, 'y') and g.y is not None:
        ng.y = g.y
    return ng

print("Cleaning graphs (removing string attrs)...")
graphs = [clean_graph(g) for g in graphs]
print(f"  Done — {len(graphs)} clean graphs.")

# Fixed split
idx = np.arange(len(labels))
lbl = np.array(labels)
itv, its = train_test_split(idx, test_size=0.15, stratify=lbl, random_state=42)
it, iv   = train_test_split(itv, test_size=0.15/0.85, stratify=lbl[itv], random_state=42)

def sub(idxs):
    gs = [graphs[i] for i in idxs]
    for g, lb in zip(gs, lbl[idxs]):
        g.y = torch.tensor([int(lb)], dtype=torch.long)
    return gs

train_g, val_g, test_g = sub(it), sub(iv), sub(its)
test_ctypes = [all_ctypes[i] for i in its]
print(f"Split: train={len(train_g)}, val={len(val_g)}, test={len(test_g)}")

# Extract actual edge types from data (avoids 11x11=121 phantom edge types)
print("Extracting real edge types...")
real_edge_types = set()
for g in graphs:
    for et in g.edge_types:
        if hasattr(g[et], 'edge_index') and g[et].edge_index.size(1) > 0:
            real_edge_types.add(et)
real_edge_types = list(real_edge_types)
print(f"  Real edge types: {len(real_edge_types)} (vs {len(ATOM_TYPES_11)**2} all-combos)")

# Quick sanity check
print("\nSanity check — one HGT batch...")
_b = next(iter(DataLoader(train_g[:4], 4, num_workers=0))).to(DEVICE)
_m = HGTModel(feat_dims_37, real_edge_types).to(DEVICE)
with torch.no_grad():
    _o = _m(_b.x_dict, _b.edge_index_dict, _b.batch_dict, _b.num_graphs)
print(f"  HGT OK: output={_o.shape}, sample={_o[:2].cpu().tolist()}")
del _b, _m, _o

# Multi-seed loop — HGT pe cuda:0, GAT pe cuda:1 in paralel
import threading

N_GPUS = torch.cuda.device_count()
DEV_HGT = torch.device('cuda:0') if N_GPUS >= 1 else DEVICE
DEV_GAT = torch.device('cuda:1') if N_GPUS >= 2 else DEVICE
print(f"GPU setup: HGT→{DEV_HGT}  GAT→{DEV_GAT}  (parallel={N_GPUS>=2})")

hgt_results, gat_results = [], []

for seed in range(N_SEEDS):
    print(f"\n{'='*55}\nSEED {seed+1}/{N_SEEDS}\n{'='*55}")
    results = {}

    def _hgt(s=seed):
        try:
            r = run_hgt_seed(train_g, val_g, test_g, test_ctypes, feat_dims_37, real_edge_types, s, device=DEV_HGT)
            results['hgt'] = r
            print(f"  [HGT] F1={r['f1']:.4f}  Acc={r['acc']:.4f}  FNR={r['fnr']:.4f}")
            for cat in CATEGORIES:
                v = r['fnr_cat'].get(cat, float('nan'))
                print(f"    FNR {cat}: {v:.4f}" if not np.isnan(v) else f"    FNR {cat}: N/A")
        except Exception as e:
            print(f"  [HGT] FAILED: {e}")
            import traceback; traceback.print_exc()

    def _gat(s=seed):
        try:
            r = run_gat_seed(train_g, val_g, test_g, test_ctypes, 26, s, device=DEV_GAT)
            results['gat'] = r
            print(f"  [GAT] F1={r['f1']:.4f}  Acc={r['acc']:.4f}  FNR={r['fnr']:.4f}")
            for cat in CATEGORIES:
                v = r['fnr_cat'].get(cat, float('nan'))
                print(f"    FNR {cat}: {v:.4f}" if not np.isnan(v) else f"    FNR {cat}: N/A")
        except Exception as e:
            print(f"  [GAT] FAILED: {e}")
            import traceback; traceback.print_exc()

    t1 = threading.Thread(target=_hgt)
    t2 = threading.Thread(target=_gat)
    t1.start(); t2.start()
    t1.join();  t2.join()

    if 'hgt' in results: hgt_results.append(results['hgt'])
    if 'gat' in results: gat_results.append(results['gat'])

    interim = {'hgt': hgt_results, 'gat': gat_results, 'seeds_done': seed+1}
    with open(OUT_DIR / 'interim_results.json', 'w') as f:
        json.dump(interim, f, indent=2)
    print(f"  Interim saved ({seed+1}/{N_SEEDS} seeds done)")

# ── Welch's t-test ─────────────────────────────────────────────────────────────
FNR_METRICS = {'fnr', 'fnr_bioisostere', 'fnr_low_effort', 'fnr_scaffold_hop'}

def clean(arr):
    return [float(x) for x in arr if x is not None and not np.isnan(float(x))]

def welch(a, b):
    a, b = clean(a), clean(b)
    if len(a) < 2 or len(b) < 2: return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    s1, s2, n1, n2 = np.std(a,ddof=1), np.std(b,ddof=1), len(a), len(b)
    df = (s1**2/n1+s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1)+(s2**2/n2)**2/(n2-1)+1e-12)
    return dict(t=t, df=df, p=p, mean_hgt=np.mean(a), std_hgt=s1,
                mean_gat=np.mean(b), std_gat=s2, n=n1)

def extract(results, metric):
    if metric in ('f1','acc','fnr'): return [r[metric] for r in results]
    cat = metric.replace('fnr_','').replace('_','-')
    return [r['fnr_cat'].get(cat, float('nan')) for r in results]

METRICS = {'f1':'F1','acc':'Accuracy','fnr':'FNR (overall)',
           'fnr_bioisostere':'FNR bioisostere',
           'fnr_low_effort':'FNR low-effort',
           'fnr_scaffold_hop':'FNR scaffold-hop'}

n_tests = len(METRICS); alpha = 0.05; alpha_c = alpha / n_tests
print(f"\n{'='*70}")
print(f"WELCH T-TEST (Bonferroni α={alpha}/{n_tests}={alpha_c:.4f})")
print(f"{'Metric':<22} {'HGT':>18} {'GAT':>18} {'t':>7} {'df':>5} {'p':>10}  sig?")
print('-'*70)

summary, paper_lines = {}, []
for metric, label in METRICS.items():
    hv, gv = extract(hgt_results, metric), extract(gat_results, metric)
    hc, gc = clean(hv), clean(gv)
    hs = f"{np.mean(hc):.3f}±{np.std(hc,ddof=1):.3f}" if hc else 'N/A'
    gs = f"{np.mean(gc):.3f}±{np.std(gc,ddof=1):.3f}" if gc else 'N/A'
    r = welch(hv, gv)
    summary[metric] = {'HGT':{'values':hv,'mean':np.mean(hc) if hc else None,'std':np.std(hc,ddof=1) if hc else None},
                        'GAT':{'values':gv,'mean':np.mean(gc) if gc else None,'std':np.std(gc,ddof=1) if gc else None}}
    if r is None:
        print(f"{label:<22} {hs:>18} {gs:>18}  N/A"); continue
    sig = r['p'] < alpha_c
    win = ('HGT' if r['mean_hgt'] < r['mean_gat'] else 'GAT') if metric in FNR_METRICS \
          else ('HGT' if r['mean_hgt'] > r['mean_gat'] else 'GAT')
    print(f"{label:<22} {hs:>18} {gs:>18}  {r['t']:>7.3f} {r['df']:>5.1f} {r['p']:>10.2e}  {'**' if sig else ''}")
    hx = f"{r['mean_hgt']:.3f} \\pm {r['std_hgt']:.3f}"
    gx = f"{r['mean_gat']:.3f} \\pm {r['std_gat']:.3f}"
    ts = f"t={r['t']:.2f}, df={r['df']:.1f}, p={r['p']:.2e}"
    if sig:
        if metric in FNR_METRICS:
            sent = f"HGT achieves a significantly lower {label} than GAT (${hx}$ vs.\\ ${gx}$; Welch's $t$-test: ${ts}$)."
        elif win == 'GAT':
            sent = f"GAT achieves a significantly higher {label} than HGT (${gx}$ vs.\\ ${hx}$; Welch's $t$-test: ${ts}$)."
        else:
            sent = f"HGT achieves a significantly higher {label} than GAT (${hx}$ vs.\\ ${gx}$; Welch's $t$-test: ${ts}$)."
    else:
        sent = f"The difference in {label} between HGT (${hx}$) and GAT (${gx}$) is not statistically significant (Welch's $t$-test: ${ts}$)."
    paper_lines.append((label, sent))
    summary[metric]['test'] = {**r, 'significant': sig, 'winner': win}

print(f"\n{'='*70}\nPAPER TEXT:")
for label, sent in paper_lines:
    print(f"\n[{label}]\n{sent}")

# ── Save ──────────────────────────────────────────────────────────────────────
final = {
    'meta': {'seeds':N_SEEDS,'epochs':N_EPOCHS,'patience':PATIENCE,
             'batch_size':BATCH_SIZE,'dataset':ds_path,
             'hgt_features':'37 (26+11 one-hot)','gat_features':'35 (26+9 one-hot)',
             'device':str(DEVICE)},
    'per_seed': {'hgt': hgt_results, 'gat': gat_results},
    'summary':  summary,
    'paper_sentences': {label: sent for label, sent in paper_lines},
}
out_path = OUT_DIR / 'multi_seed_results.json'
with open(out_path, 'w') as f:
    json.dump(final, f, indent=2)
print(f"\nSaved: {out_path}")
print("Download: Output tab → multi_seed_results.json")
print("Local: python3 welch_analysis.py --file multi_seed_results.json")
