"""
Multi-seed evaluation: HGT (37-feat, with atom-type one-hot) vs GAT.
Runs N_SEEDS independent training runs (fixed data split, varying model init)
and applies Welch's t-test on F1, Accuracy, and FNR per category.

Usage:
    python multi_seed_eval.py --dataset intelligent_pharma_50k_v2.pt --seeds 3
"""

import argparse, copy, json, random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch_geometric.loader import DataLoader
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

CATEGORY_MAP = {
    'methyl_to_ethyl_on_aryl':    'bioisostere',
    'OMe_to_OEt':                 'bioisostere',
    'COOH_to_tetrazole':          'bioisostere',
    'ester_to_amide':             'bioisostere',
    'amide_to_sulfonamide':       'bioisostere',
    'NH2_to_NHMe':                'bioisostere',
    'Cl_to_CF3_aryl':             'bioisostere',
    'ketone_to_sulfoxide':        'bioisostere',
    'piperidine_to_morpholine':   'bioisostere',
    'aryl_CH_to_N_para':          'low-effort',
    'aryl_C_to_N':                'low-effort',
    'F_to_Cl_aryl':               'low-effort',
    'Br_to_Cl_aryl':              'low-effort',
    'alkene_to_alkane':           'scaffold-hop',
    'cyclopentane_to_cyclohexane':'scaffold-hop',
}
CATEGORIES = ['bioisostere', 'low-effort', 'scaffold-hop']


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: str):
    from dcsac import load_and_prepare_enhanced_dataset
    graphs, labels, feature_dims = load_and_prepare_enhanced_dataset(path)
    if graphs is None:
        raise RuntimeError(f"Could not load dataset from {path}")
    return graphs, labels, feature_dims


def fixed_split(graphs, labels, random_state=42):
    """70/15/15 split fixed across all seeds."""
    idx = np.arange(len(labels))
    lbl = np.array(labels)
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, stratify=lbl,
                                        random_state=random_state)
    lbl_tv = lbl[idx_tv]
    val_frac = 0.15 / 0.85
    idx_train, idx_val = train_test_split(idx_tv, test_size=val_frac,
                                          stratify=lbl_tv,
                                          random_state=random_state)

    def subset(idxs):
        gs = [graphs[i] for i in idxs]
        for g, lb in zip(gs, lbl[idxs]):
            g.y = torch.tensor([int(lb)], dtype=torch.long)
        return gs

    return subset(idx_train), subset(idx_val), subset(idx_test)


def compute_metrics(preds, labels, test_graphs):
    preds  = np.array(preds)
    labels = np.array(labels)
    f1  = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    cm  = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp + 1e-9)

    fnr_cat = {}
    for cat in CATEGORIES:
        mask = [i for i, g in enumerate(test_graphs)
                if int(g.y.item()) == 1
                and CATEGORY_MAP.get(getattr(g, 'counterfeit_type', ''), '') == cat]
        fnr_cat[cat] = float(np.mean(preds[mask] == 0)) if mask else float('nan')

    return {'f1': f1, 'acc': acc, 'fnr': fnr, 'fnr_cat': fnr_cat}


# ── HGT (37-dim features with atom-type one-hot) ────────────────────────────

def train_hgt(train_graphs, val_graphs, test_graphs, feature_dims_37,
              device, seed, num_epochs=50, patience=10):
    from dcsac import RobustEnhancedHGTDetector, VSCodeTrainer, FocalLoss

    set_seed(seed)
    batch_size = 32 if device.type == 'cuda' else 16

    train_loader = DataLoader(train_graphs, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    model = RobustEnhancedHGTDetector(
        feature_dims=feature_dims_37,
        hidden_channels=192, num_heads=8, num_layers=3, dropout=0.15,
    ).to(device)

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    save_dir = Path(f'./multi_seed_tmp/hgt_seed{seed}')
    trainer  = VSCodeTrainer(model, device, save_path=save_dir)
    trainer.train(train_loader, val_loader, test_loader, criterion, optimizer,
                  num_epochs=num_epochs, patience=patience, scheduler=scheduler)

    model.load_state_dict(trainer.best_model_state)
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            try:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict,
                            batch.batch_dict, batch.num_graphs)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                true_labels.extend(batch.y.cpu().numpy().ravel())
            except Exception:
                continue

    return compute_metrics(preds, true_labels, test_graphs)


# ── GAT (26-dim base + 9-bit one-hot via hetero_to_homo) ────────────────────

def train_gat(train_graphs, val_graphs, test_graphs, feature_dims_26,
              device, seed, num_epochs=50, patience=10):
    from homo_gnn_clean import HomoGATDetector, HomoTrainer, FocalLoss, hetero_to_homo

    set_seed(seed)

    def convert(gs):
        out = []
        for g in gs:
            # feature_dims_26 → base_dim=26, hetero_to_homo trims one-hot we added
            h = hetero_to_homo(g, feature_dims_26)
            if h is not None:
                h.y = g.y
                if hasattr(g, 'counterfeit_type'):
                    h.counterfeit_type = g.counterfeit_type
                out.append(h)
        return out

    train_h = convert(train_graphs)
    val_h   = convert(val_graphs)
    test_h  = convert(test_graphs)
    if not train_h:
        raise RuntimeError("GAT conversion produced empty train set")

    in_channels = train_h[0].x.size(1)
    batch_size  = 32 if device.type == 'cuda' else 16

    train_loader = DataLoader(train_h, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_h,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_h,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    model = HomoGATDetector(
        in_channels=in_channels, hidden_channels=192, out_channels=2,
        num_heads=8, num_layers=3, dropout=0.15,
    ).to(device)

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    save_dir = Path(f'./multi_seed_tmp/gat_seed{seed}')
    trainer  = HomoTrainer(model, device, save_path=save_dir)
    # HomoTrainer.train(train, test, ...) — pass val as "test" for model selection
    trainer.train(train_loader, val_loader, criterion, optimizer,
                  num_epochs=num_epochs, patience=patience, scheduler=scheduler)

    model.load_state_dict(trainer.best_model_state)
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            try:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                true_labels.extend(batch.y.cpu().numpy().ravel())
            except Exception:
                continue

    return compute_metrics(preds, true_labels, test_h)


# ── Welch's t-test ────────────────────────────────────────────────────────────

def welch(a, b):
    """Returns (t, df, p) or None if not enough data."""
    a = [x for x in a if not np.isnan(x)]
    b = [x for x in b if not np.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    s1, s2, n1, n2 = np.std(a, ddof=1), np.std(b, ddof=1), len(a), len(b)
    num = (s1**2/n1 + s2**2/n2)**2
    den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    df  = num/den if den > 0 else float('inf')
    return t, df, p


def print_results_and_paper_text(hgt_results, gat_results, n_seeds, alpha=0.05):
    """Print table + ready-to-use LaTeX sentences for the paper."""

    def stats_str(vals):
        v = [x for x in vals if not np.isnan(x)]
        if not v:
            return 'N/A'
        return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"

    metrics = [
        ('F1',           [r['f1']  for r in hgt_results], [r['f1']  for r in gat_results]),
        ('Accuracy',     [r['acc'] for r in hgt_results], [r['acc'] for r in gat_results]),
        ('FNR (overall)',[r['fnr'] for r in hgt_results], [r['fnr'] for r in gat_results]),
    ]
    for cat in CATEGORIES:
        metrics.append((
            f'FNR {cat}',
            [r['fnr_cat'].get(cat, float('nan')) for r in hgt_results],
            [r['fnr_cat'].get(cat, float('nan')) for r in gat_results],
        ))

    # Bonferroni correction
    n_tests    = len(metrics)
    alpha_corr = alpha / n_tests

    print(f"\n{'='*70}")
    print(f"RESULTS  (mean ± std over {n_seeds} seeds)")
    print(f"Bonferroni-corrected α = {alpha:.2f}/{n_tests} = {alpha_corr:.4f}")
    print(f"{'Metric':<22} {'HGT':>18} {'GAT':>18}  {'t':>7} {'df':>6} {'p':>10}  sig?")
    print('-'*70)

    paper_lines = []

    for label, hgt_vals, gat_vals in metrics:
        res = welch(hgt_vals, gat_vals)
        hgt_str = stats_str(hgt_vals)
        gat_str = stats_str(gat_vals)

        if res is None:
            print(f"{label:<22} {hgt_str:>18} {gat_str:>18}  {'N/A':>7}")
            continue

        t, df, p = res
        sig = p < alpha_corr
        hgt_mean = np.nanmean(hgt_vals)
        gat_mean = np.nanmean(gat_vals)
        winner   = 'HGT' if hgt_mean > gat_mean else 'GAT'
        # For FNR: lower is better
        if 'FNR' in label:
            winner = 'HGT' if hgt_mean < gat_mean else 'GAT'

        flag = '**' if sig else ''
        print(f"{label:<22} {hgt_str:>18} {gat_str:>18}  "
              f"{t:>7.3f} {df:>6.1f} {p:>10.2e}  {flag}")

        # Generate paper sentence
        metric_name = label.replace('FNR ', 'FNR for ').replace('FNR (overall)', 'overall FNR')
        hgt_v = f"{np.nanmean(hgt_vals):.3f} \\pm {np.nanstd(hgt_vals, ddof=1):.3f}"
        gat_v = f"{np.nanmean(gat_vals):.3f} \\pm {np.nanstd(gat_vals, ddof=1):.3f}"

        if sig:
            if 'FNR' in label:
                sentence = (
                    f"HGT achieves a significantly lower {metric_name} than GAT "
                    f"(${hgt_v}$ vs ${gat_v}$; "
                    f"Welch's $t$-test: $t={t:.2f}$, $df={df:.1f}$, $p={p:.2e}$)."
                )
            elif winner == 'GAT':
                sentence = (
                    f"GAT achieves a significantly higher {metric_name} than HGT "
                    f"(${gat_v}$ vs ${hgt_v}$; "
                    f"Welch's $t$-test: $t={t:.2f}$, $df={df:.1f}$, $p={p:.2e}$)."
                )
            else:
                sentence = (
                    f"HGT achieves a significantly higher {metric_name} than GAT "
                    f"(${hgt_v}$ vs ${gat_v}$; "
                    f"Welch's $t$-test: $t={t:.2f}$, $df={df:.1f}$, $p={p:.2e}$)."
                )
        else:
            sentence = (
                f"The difference in {metric_name} between HGT (${hgt_v}$) and "
                f"GAT (${gat_v}$) is not statistically significant "
                f"($t={t:.2f}$, $df={df:.1f}$, $p={p:.2e}$)."
            )
        paper_lines.append((label, sentence))

    print(f"\n{'='*70}")
    print("PAPER TEXT (copy-paste ready, LaTeX math mode):")
    print('='*70)
    for label, sentence in paper_lines:
        print(f"\n[{label}]")
        print(sentence)

    return paper_lines


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='intelligent_pharma_50k_v2.pt')
    parser.add_argument('--seeds',   type=int, default=3)
    parser.add_argument('--epochs',  type=int, default=50)
    parser.add_argument('--subset',  type=int, default=None,
                        help='Use only N molecules (balanced). E.g. --subset 15000')
    parser.add_argument('--out',     default='multi_seed_results.json')
    args = parser.parse_args()

    Path('./multi_seed_tmp').mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading dataset: {args.dataset}")
    graphs, labels, feature_dims_26 = load_dataset(args.dataset)
    print(f"Dataset: {len(graphs)} graphs, base feature dim: "
          f"{next(iter(feature_dims_26.values()))}")

    # Apply one-hot in-place ONCE → graphs become 37-dim
    # GAT's hetero_to_homo will trim back to base_dim=26 and add its own 9-bit one-hot
    from dcsac import add_atom_type_onehot
    # Optional subset — balanced (equal auth/counterfeit)
    if args.subset and args.subset < len(graphs):
        lbl_arr = np.array(labels)
        auth_idx = np.where(lbl_arr == 0)[0]
        fake_idx = np.where(lbl_arr == 1)[0]
        n_each   = args.subset // 2
        rng      = np.random.RandomState(42)
        keep_idx = np.concatenate([
            rng.choice(auth_idx, min(n_each, len(auth_idx)), replace=False),
            rng.choice(fake_idx, min(n_each, len(fake_idx)), replace=False),
        ])
        graphs = [graphs[i] for i in keep_idx]
        labels = [labels[i] for i in keep_idx]
        print(f"Subset: using {len(graphs)} molecules ({n_each} auth + {n_each} fake)")

    print("Adding atom-type one-hot to HGT graphs (26 → 37)...")
    feature_dims_37 = add_atom_type_onehot(graphs, dict(feature_dims_26))
    print(f"  HGT feature dim: {next(iter(feature_dims_37.values()))}")
    print(f"  GAT will trim back to {next(iter(feature_dims_26.values()))} + 9 one-hot")

    print("\nSplitting (fixed random_state=42)...")
    train_g, val_g, test_g = fixed_split(graphs, labels)
    print(f"Train={len(train_g)}, Val={len(val_g)}, Test={len(test_g)}")

    hgt_results, gat_results = [], []

    for seed in range(args.seeds):
        print(f"\n{'='*55}")
        print(f"SEED {seed+1}/{args.seeds}")
        print('='*55)

        print(f"\n[HGT] seed={seed} — 37 features")
        try:
            hm = train_hgt(train_g, val_g, test_g, feature_dims_37,
                           device, seed, args.epochs)
            hgt_results.append(hm)
            print(f"  F1={hm['f1']:.4f}  Acc={hm['acc']:.4f}  FNR={hm['fnr']:.4f}")
            for cat in CATEGORIES:
                v = hm['fnr_cat'].get(cat, float('nan'))
                print(f"    FNR {cat}: {v:.4f}" if not np.isnan(v) else f"    FNR {cat}: N/A")
        except Exception as e:
            print(f"  HGT seed {seed} FAILED: {e}")
            import traceback; traceback.print_exc()

        print(f"\n[GAT] seed={seed} — 26+9 features")
        try:
            gm = train_gat(train_g, val_g, test_g, feature_dims_26,
                           device, seed, args.epochs)
            gat_results.append(gm)
            print(f"  F1={gm['f1']:.4f}  Acc={gm['acc']:.4f}  FNR={gm['fnr']:.4f}")
            for cat in CATEGORIES:
                v = gm['fnr_cat'].get(cat, float('nan'))
                print(f"    FNR {cat}: {v:.4f}" if not np.isnan(v) else f"    FNR {cat}: N/A")
        except Exception as e:
            print(f"  GAT seed {seed} FAILED: {e}")
            import traceback; traceback.print_exc()

    # ── Statistical analysis + paper text ────────────────────────────────────
    paper_lines = print_results_and_paper_text(
        hgt_results, gat_results, args.seeds
    )

    # ── Build per-metric arrays for easy re-analysis ────────────────────────
    def extract_arrays(results):
        return {
            'f1':              [r['f1']  for r in results],
            'acc':             [r['acc'] for r in results],
            'fnr':             [r['fnr'] for r in results],
            'fnr_bioisostere': [r['fnr_cat'].get('bioisostere', None) for r in results],
            'fnr_low_effort':  [r['fnr_cat'].get('low-effort',  None) for r in results],
            'fnr_scaffold_hop':[r['fnr_cat'].get('scaffold-hop',None) for r in results],
        }

    hgt_arrays = extract_arrays(hgt_results)
    gat_arrays = extract_arrays(gat_results)

    def summarize(arr):
        v = [x for x in arr if x is not None and not np.isnan(x)]
        if not v:
            return {'mean': None, 'std': None, 'values': arr}
        return {
            'mean':   round(float(np.mean(v)), 4),
            'std':    round(float(np.std(v, ddof=1)), 4),
            'values': arr,
        }

    summary = {
        metric: {
            'HGT': summarize(hgt_arrays[metric]),
            'GAT': summarize(gat_arrays[metric]),
        }
        for metric in hgt_arrays
    }

    out = {
        'meta': {
            'seeds': args.seeds,
            'epochs': args.epochs,
            'dataset': args.dataset,
            'hgt_features': '37 (26 base + 11 atom-type one-hot)',
            'gat_features': '35 (26 base + 9 atom-type one-hot via hetero_to_homo)',
        },
        'per_seed': {
            'hgt': hgt_results,
            'gat': gat_results,
        },
        'summary': summary,
        'paper_sentences': {label: sent for label, sent in paper_lines},
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved to: {args.out}")
    print("\nSummary (mean ± std):")
    for metric, vals in summary.items():
        h = vals['HGT']
        g = vals['GAT']
        hstr = f"{h['mean']:.4f} ± {h['std']:.4f}" if h['mean'] is not None else 'N/A'
        gstr = f"{g['mean']:.4f} ± {g['std']:.4f}" if g['mean'] is not None else 'N/A'
        print(f"  {metric:<22} HGT={hstr}   GAT={gstr}")


if __name__ == '__main__':
    main()
