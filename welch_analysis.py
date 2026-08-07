"""
Welch's t-test analysis on saved multi_seed_results.json.
Rulezi independent, fara re-antrenare.

Usage:
    python welch_analysis.py                          # default: multi_seed_results.json
    python welch_analysis.py --file rezultate.json
    python welch_analysis.py --alpha 0.01             # nivel incredere 99%
    python welch_analysis.py --no-bonferroni          # fara corectie multipla
"""

import argparse
import json
import numpy as np
from scipy import stats


METRIC_LABELS = {
    'f1':               'F1',
    'acc':              'Accuracy',
    'fnr':              'FNR (overall)',
    'fnr_bioisostere':  'FNR bioisostere',
    'fnr_low_effort':   'FNR low-effort',
    'fnr_scaffold_hop': 'FNR scaffold-hop',
}

# Pentru FNR: mai mic = mai bun (HGT castiga daca e mai mic)
FNR_METRICS = {'fnr', 'fnr_bioisostere', 'fnr_low_effort', 'fnr_scaffold_hop'}


def clean(arr):
    return [x for x in arr if x is not None and not np.isnan(float(x))]


def welch(a, b):
    a, b = clean(a), clean(b)
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    s1, s2, n1, n2 = np.std(a, ddof=1), np.std(b, ddof=1), len(a), len(b)
    num = (s1**2/n1 + s2**2/n2)**2
    den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    df = num/den if den > 0 else float('inf')
    return {'t': t, 'df': df, 'p': p,
            'mean_hgt': np.mean(a), 'std_hgt': np.std(a, ddof=1),
            'mean_gat': np.mean(b), 'std_gat': np.std(b, ddof=1),
            'n_hgt': n1, 'n_gat': n2}


def direction(metric, r):
    """Who wins? For FNR lower is better, for others higher is better."""
    if metric in FNR_METRICS:
        return 'HGT' if r['mean_hgt'] < r['mean_gat'] else 'GAT'
    return 'HGT' if r['mean_hgt'] > r['mean_gat'] else 'GAT'


def run_analysis(data, alpha=0.05, bonferroni=True):
    summary = data.get('summary', {})
    if not summary:
        # Reconstruct from per_seed if summary missing
        hgt_all = data['per_seed']['hgt']
        gat_all = data['per_seed']['gat']
        summary = {}
        for metric in METRIC_LABELS:
            hgt_vals = []
            gat_vals = []
            for r in hgt_all:
                if metric in ('fnr_bioisostere', 'fnr_low_effort', 'fnr_scaffold_hop'):
                    cat = metric.replace('fnr_', '').replace('_', '-')
                    hgt_vals.append(r.get('fnr_cat', {}).get(cat))
                elif metric == 'fnr':
                    hgt_vals.append(r.get('fnr'))
                else:
                    hgt_vals.append(r.get(metric))
            for r in gat_all:
                if metric in ('fnr_bioisostere', 'fnr_low_effort', 'fnr_scaffold_hop'):
                    cat = metric.replace('fnr_', '').replace('_', '-')
                    gat_vals.append(r.get('fnr_cat', {}).get(cat))
                elif metric == 'fnr':
                    gat_vals.append(r.get('fnr'))
                else:
                    gat_vals.append(r.get(metric))
            summary[metric] = {
                'HGT': {'values': hgt_vals},
                'GAT': {'values': gat_vals},
            }

    n_tests = len(METRIC_LABELS)
    alpha_eff = alpha / n_tests if bonferroni else alpha

    meta = data.get('meta', {})
    print(f"\n{'='*72}")
    print("WELCH'S T-TEST ANALYSIS")
    print(f"  Dataset : {meta.get('dataset', '?')}")
    print(f"  Seeds   : {meta.get('seeds', '?')}")
    print(f"  HGT feat: {meta.get('hgt_features', '?')}")
    print(f"  GAT feat: {meta.get('gat_features', '?')}")
    print(f"  α = {alpha}  |  Bonferroni: {bonferroni}  |  α_eff = {alpha_eff:.4f}")
    print(f"{'='*72}")
    print(f"{'Metric':<22} {'HGT mean±std':>20} {'GAT mean±std':>20} "
          f"{'t':>7} {'df':>6} {'p':>10}  {'winner':<5}  sig?")
    print('-'*72)

    results_out = {}
    paper_lines = []

    for metric, label in METRIC_LABELS.items():
        hgt_vals = summary[metric]['HGT']['values']
        gat_vals = summary[metric]['GAT']['values']
        r = welch(hgt_vals, gat_vals)

        hgt_c, gat_c = clean(hgt_vals), clean(gat_vals)
        hgt_str = f"{np.mean(hgt_c):.3f}±{np.std(hgt_c,ddof=1):.3f}" if hgt_c else 'N/A'
        gat_str = f"{np.mean(gat_c):.3f}±{np.std(gat_c,ddof=1):.3f}" if gat_c else 'N/A'

        if r is None:
            print(f"{label:<22} {hgt_str:>20} {gat_str:>20}  {'N/A':>7}  (insufficient data)")
            continue

        sig = r['p'] < alpha_eff
        win = direction(metric, r)
        flag = '**' if sig else '  '
        print(f"{label:<22} {hgt_str:>20} {gat_str:>20}  "
              f"{r['t']:>7.3f} {r['df']:>6.1f} {r['p']:>10.3e}  {win:<5}  {flag}")

        results_out[metric] = {**r, 'significant': sig, 'winner': win,
                               'alpha_eff': alpha_eff}

        # Paper sentence
        hv = f"{r['mean_hgt']:.3f} \\pm {r['std_hgt']:.3f}"
        gv = f"{r['mean_gat']:.3f} \\pm {r['std_gat']:.3f}"
        t_str = f"t={r['t']:.2f}, df={r['df']:.1f}, p={r['p']:.2e}"

        if sig:
            if metric in FNR_METRICS:
                sent = (f"HGT achieves a significantly lower {label} than GAT "
                        f"(${hv}$ vs.\ ${gv}$; Welch's $t$-test: ${t_str}$).")
            elif win == 'GAT':
                sent = (f"GAT achieves a significantly higher {label} than HGT "
                        f"(${gv}$ vs.\ ${hv}$; Welch's $t$-test: ${t_str}$).")
            else:
                sent = (f"HGT achieves a significantly higher {label} than GAT "
                        f"(${hv}$ vs.\ ${gv}$; Welch's $t$-test: ${t_str}$).")
        else:
            sent = (f"The difference in {label} between HGT (${hv}$) and "
                    f"GAT (${gv}$) is not statistically significant "
                    f"(Welch's $t$-test: ${t_str}$).")
        paper_lines.append((label, sent))

    print(f"\n{'='*72}")
    print("PAPER TEXT:")
    print('='*72)
    for label, sent in paper_lines:
        print(f"\n[{label}]")
        print(sent)

    # LaTeX table rows for Table I and Table III
    print(f"\n{'='*72}")
    print("LaTeX TABLE ROWS (paste into tabular):")
    print('='*72)
    for metric in ['f1', 'acc', 'fnr']:
        hgt_c = clean(summary[metric]['HGT']['values'])
        gat_c = clean(summary[metric]['GAT']['values'])
        if not hgt_c or not gat_c:
            continue
        hm, hs = np.mean(hgt_c), np.std(hgt_c, ddof=1)
        gm, gs = np.mean(gat_c), np.std(gat_c, ddof=1)
        print(f"{METRIC_LABELS[metric]:<15} & ${hm:.3f} \\pm {hs:.3f}$ "
              f"& ${gm:.3f} \\pm {gs:.3f}$ \\\\")

    print(f"\nPer-category FNR (mean ± std):")
    for metric in ['fnr_bioisostere', 'fnr_low_effort', 'fnr_scaffold_hop']:
        hgt_c = clean(summary[metric]['HGT']['values'])
        gat_c = clean(summary[metric]['GAT']['values'])
        if not hgt_c or not gat_c:
            continue
        hm, hs = np.mean(hgt_c), np.std(hgt_c, ddof=1)
        gm, gs = np.mean(gat_c), np.std(gat_c, ddof=1)
        cat = metric.replace('fnr_', '').replace('_', '-')
        print(f"{cat:<20} HGT: {hm:.3f}±{hs:.3f}   GAT: {gm:.3f}±{gs:.3f}")

    return results_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',         default='multi_seed_results.json')
    parser.add_argument('--alpha',        type=float, default=0.05)
    parser.add_argument('--no-bonferroni',action='store_true')
    args = parser.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    run_analysis(data, alpha=args.alpha, bonferroni=not args.no_bonferroni)


if __name__ == '__main__':
    main()
