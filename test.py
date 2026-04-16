"""
Diagnostic Investigation: Why does Homogeneous GAT outperform HGT?
=================================================================
Systematic verification of:
  1. Data Leakage (train/test contamination)
  2. SMILES artifact memorization
  3. Feature shortcut detection
  4. Generalization gap analysis
  5. Counterfeit difficulty breakdown
  6. Tanimoto similarity distribution analysis

Run: python diagnostic_investigation.py <dataset_path>
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import warnings
import sys
import json

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  SECTION 1: DATA LEAKAGE DETECTION
# ─────────────────────────────────────────────────────────────

class DataLeakageDetector:
    """Detects multiple forms of data leakage in the dataset."""

    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        self.n = len(graphs)

    def run_all_checks(self):
        print("\n" + "=" * 70)
        print("  TEST 1: DATA LEAKAGE DETECTION")
        print("=" * 70)

        results = {}

        # 1a. SMILES duplication check
        results['smiles_duplication'] = self._check_smiles_duplication()

        # 1b. Train/test contamination with same random_state=42
        results['train_test_contamination'] = self._check_train_test_contamination()

        # 1c. Parent-child leakage (counterfeit derived from authentic in other split)
        results['parent_child_leakage'] = self._check_parent_child_leakage()

        # 1d. Near-duplicate detection
        results['near_duplicates'] = self._check_near_duplicates()

        return results

    def _check_smiles_duplication(self):
        print("\n[1a] Checking SMILES duplication...")
        smiles_list = []
        for g in self.graphs:
            s = getattr(g, 'smiles', None)
            if s:
                smiles_list.append(s)

        total = len(smiles_list)
        unique = len(set(smiles_list))
        duplicates = total - unique

        # Check duplicates with different labels
        smiles_to_labels = defaultdict(set)
        for g, lbl in zip(self.graphs, self.labels):
            s = getattr(g, 'smiles', None)
            if s:
                smiles_to_labels[s].add(int(lbl) if isinstance(lbl, (int, np.integer)) else lbl)

        conflicting = sum(1 for s, lbls in smiles_to_labels.items() if len(lbls) > 1)

        print(f"  Total molecules:     {total}")
        print(f"  Unique SMILES:       {unique}")
        print(f"  Duplicates:          {duplicates} ({duplicates/max(total,1)*100:.1f}%)")
        print(f"  Conflicting labels:  {conflicting}")

        if conflicting > 0:
            print(f"  ⚠️  WARNING: {conflicting} SMILES have BOTH authentic and counterfeit labels!")
            print(f"     This is a CRITICAL data quality issue.")
        if duplicates > total * 0.01:
            print(f"  ⚠️  WARNING: >1% duplicates detected")
        else:
            print(f"  ✅  Duplication level acceptable")

        return {
            'total': total, 'unique': unique, 'duplicates': duplicates,
            'conflicting_labels': conflicting,
            'severity': 'CRITICAL' if conflicting > 0 else ('WARNING' if duplicates > total * 0.01 else 'OK')
        }

    def _check_train_test_contamination(self):
        print("\n[1b] Checking train/test contamination (random_state=42)...")

        indices = np.arange(self.n)
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=self.labels, random_state=42
        )

        train_smiles = set()
        test_smiles = set()
        for i in train_idx:
            s = getattr(self.graphs[i], 'smiles', None)
            if s:
                train_smiles.add(s)
        for i in test_idx:
            s = getattr(self.graphs[i], 'smiles', None)
            if s:
                test_smiles.add(s)

        overlap = train_smiles & test_smiles
        print(f"  Train SMILES:        {len(train_smiles)}")
        print(f"  Test SMILES:         {len(test_smiles)}")
        print(f"  Overlapping SMILES:  {len(overlap)} ({len(overlap)/max(len(test_smiles),1)*100:.1f}% of test)")

        if overlap:
            print(f"  ⚠️  WARNING: Identical molecules appear in both train and test!")
            # Show a few examples
            for s in list(overlap)[:5]:
                print(f"     Example: {s[:60]}...")
        else:
            print(f"  ✅  No SMILES overlap between train and test")

        return {
            'train_size': len(train_smiles),
            'test_size': len(test_smiles),
            'overlap': len(overlap),
            'overlap_pct': len(overlap) / max(len(test_smiles), 1) * 100,
            'severity': 'CRITICAL' if len(overlap) > len(test_smiles) * 0.05 else (
                'WARNING' if len(overlap) > 0 else 'OK')
        }

    def _check_parent_child_leakage(self):
        print("\n[1c] Checking parent-child leakage...")

        # Collect original_smiles from counterfeits
        indices = np.arange(self.n)
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=self.labels, random_state=42
        )

        train_authentic_smiles = set()
        test_counterfeit_parents = set()

        for i in train_idx:
            g = self.graphs[i]
            lbl = int(self.labels[i]) if isinstance(self.labels[i], (int, np.integer)) else self.labels[i]
            if lbl == 0:  # authentic
                s = getattr(g, 'smiles', None)
                if s:
                    train_authentic_smiles.add(s)

        for i in test_idx:
            g = self.graphs[i]
            lbl = int(self.labels[i]) if isinstance(self.labels[i], (int, np.integer)) else self.labels[i]
            if lbl == 1:  # counterfeit
                parent = getattr(g, 'original_smiles', None)
                if parent:
                    test_counterfeit_parents.add(parent)

        # How many test counterfeits were derived from train authentics?
        leaked_parents = train_authentic_smiles & test_counterfeit_parents
        
        # Also check reverse: test authentics whose children are in train
        test_authentic_smiles = set()
        train_counterfeit_parents = set()
        
        for i in test_idx:
            g = self.graphs[i]
            lbl = int(self.labels[i]) if isinstance(self.labels[i], (int, np.integer)) else self.labels[i]
            if lbl == 0:
                s = getattr(g, 'smiles', None)
                if s:
                    test_authentic_smiles.add(s)
        
        for i in train_idx:
            g = self.graphs[i]
            lbl = int(self.labels[i]) if isinstance(self.labels[i], (int, np.integer)) else self.labels[i]
            if lbl == 1:
                parent = getattr(g, 'original_smiles', None)
                if parent:
                    train_counterfeit_parents.add(parent)

        reverse_leaked = test_authentic_smiles & train_counterfeit_parents

        print(f"  Train authentics whose counterfeits are in test:  {len(leaked_parents)}")
        print(f"  Test authentics whose counterfeits are in train:  {len(reverse_leaked)}")

        total_leaked = len(leaked_parents) + len(reverse_leaked)
        if total_leaked > 0:
            print(f"  ⚠️  WARNING: Parent-child pairs span train/test split!")
            print(f"     The model can learn the 'transformation signature' in train")
            print(f"     and recognize it in test → inflated metrics.")
            print(f"     This is the MOST LIKELY cause of GAT's 92%.")
        else:
            print(f"  ✅  No parent-child leakage detected")
            print(f"     (Note: only checked if original_smiles attribute exists)")

        return {
            'leaked_parents': len(leaked_parents),
            'reverse_leaked': len(reverse_leaked),
            'total_leaked': total_leaked,
            'severity': 'CRITICAL' if total_leaked > 100 else (
                'WARNING' if total_leaked > 0 else 'OK')
        }

    def _check_near_duplicates(self):
        print("\n[1d] Checking near-duplicates (graph structure fingerprints)...")

        # Quick structural fingerprint: (num_nodes_per_type, num_edges)
        fingerprints = []
        for g in self.graphs:
            fp = []
            for nt in sorted(g.node_types):
                if hasattr(g[nt], 'x'):
                    fp.append((nt, g[nt].x.size(0)))
            total_edges = sum(
                g[et].edge_index.size(1)
                for et in g.edge_types
                if hasattr(g[et], 'edge_index')
            )
            fp.append(('edges', total_edges))
            fingerprints.append(tuple(fp))

        fp_counter = Counter(fingerprints)
        num_duplicate_fps = sum(count for fp, count in fp_counter.items() if count > 1)
        max_dup_group = max(fp_counter.values())

        print(f"  Unique structure fingerprints:  {len(fp_counter)}")
        print(f"  Molecules with duplicate FPs:   {num_duplicate_fps}")
        print(f"  Largest duplicate group:         {max_dup_group}")

        if max_dup_group > 50:
            print(f"  ⚠️  Large groups of structurally identical graphs exist")
        else:
            print(f"  ✅  Structural diversity seems reasonable")

        return {
            'unique_fps': len(fp_counter),
            'duplicate_fps': num_duplicate_fps,
            'max_group': max_dup_group,
            'severity': 'WARNING' if max_dup_group > 50 else 'OK'
        }


# ─────────────────────────────────────────────────────────────
#  SECTION 2: SMILES ARTIFACT DETECTION
# ─────────────────────────────────────────────────────────────

class ArtifactDetector:
    """Detects whether models exploit generation artifacts rather than real chemistry."""

    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def run_all_checks(self):
        print("\n" + "=" * 70)
        print("  TEST 2: SMILES ARTIFACT & SHORTCUT DETECTION")
        print("=" * 70)

        results = {}
        results['length_bias'] = self._check_smiles_length_bias()
        results['atom_count_bias'] = self._check_atom_count_bias()
        results['feature_distribution_shift'] = self._check_feature_distribution_shift()
        results['trivial_classifier'] = self._test_trivial_classifiers()
        return results

    def _check_smiles_length_bias(self):
        print("\n[2a] Checking SMILES length bias...")

        auth_lengths = []
        fake_lengths = []
        for g, lbl in zip(self.graphs, self.labels):
            s = getattr(g, 'smiles', None)
            lbl_int = int(lbl) if isinstance(lbl, (int, np.integer)) else lbl
            if s:
                if lbl_int == 0:
                    auth_lengths.append(len(s))
                else:
                    fake_lengths.append(len(s))

        if not auth_lengths or not fake_lengths:
            print("  ⚠️  Cannot check — missing SMILES data")
            return {'severity': 'UNKNOWN'}

        auth_mean = np.mean(auth_lengths)
        fake_mean = np.mean(fake_lengths)
        auth_std = np.std(auth_lengths)
        fake_std = np.std(fake_lengths)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((auth_std**2 + fake_std**2) / 2)
        cohens_d = abs(auth_mean - fake_mean) / max(pooled_std, 1e-6)

        print(f"  Authentic SMILES length: {auth_mean:.1f} ± {auth_std:.1f}")
        print(f"  Counterfeit SMILES length: {fake_mean:.1f} ± {fake_std:.1f}")
        print(f"  Cohen's d (effect size):   {cohens_d:.3f}")

        if cohens_d > 0.8:
            print(f"  ⚠️  LARGE effect size — SMILES length alone could be a strong predictor!")
        elif cohens_d > 0.5:
            print(f"  ⚠️  MEDIUM effect size — some length bias present")
        else:
            print(f"  ✅  Small/no length bias")

        return {
            'auth_mean': auth_mean, 'fake_mean': fake_mean,
            'cohens_d': cohens_d,
            'severity': 'CRITICAL' if cohens_d > 0.8 else ('WARNING' if cohens_d > 0.5 else 'OK')
        }

    def _check_atom_count_bias(self):
        print("\n[2b] Checking atom count bias per class...")

        auth_counts = defaultdict(list)
        fake_counts = defaultdict(list)

        for g, lbl in zip(self.graphs, self.labels):
            lbl_int = int(lbl) if isinstance(lbl, (int, np.integer)) else lbl
            target = auth_counts if lbl_int == 0 else fake_counts
            for nt in g.node_types:
                if hasattr(g[nt], 'x'):
                    target[nt].append(g[nt].x.size(0))

        print(f"  {'Atom Type':<10} {'Auth Mean':>10} {'Fake Mean':>10} {'Diff %':>10} {'Signal?':>10}")
        print(f"  {'-'*50}")

        biased_types = []
        for nt in sorted(set(list(auth_counts.keys()) + list(fake_counts.keys()))):
            a = np.mean(auth_counts[nt]) if auth_counts[nt] else 0
            f = np.mean(fake_counts[nt]) if fake_counts[nt] else 0
            diff_pct = abs(a - f) / max(a, f, 1e-6) * 100
            signal = "⚠️ YES" if diff_pct > 15 else "✅ no"
            if diff_pct > 15:
                biased_types.append(nt)
            print(f"  {nt:<10} {a:>10.2f} {f:>10.2f} {diff_pct:>9.1f}% {signal:>10}")

        if biased_types:
            print(f"\n  ⚠️  Atom count differs significantly for: {', '.join(biased_types)}")
            print(f"     A simple model could exploit atom counts as a shortcut.")
        else:
            print(f"\n  ✅  No major atom count bias")

        return {'biased_types': biased_types, 'severity': 'WARNING' if biased_types else 'OK'}

    def _check_feature_distribution_shift(self):
        print("\n[2c] Checking feature distribution shift between classes...")

        # Sample features from first common node type
        auth_features = []
        fake_features = []

        sample_size = min(2000, len(self.graphs))
        indices = np.random.choice(len(self.graphs), sample_size, replace=False)

        for i in indices:
            g = self.graphs[i]
            lbl = int(self.labels[i]) if isinstance(self.labels[i], (int, np.integer)) else self.labels[i]
            # Get features from first node type with data
            for nt in g.node_types:
                if hasattr(g[nt], 'x') and g[nt].x.size(0) > 0:
                    # Global mean of node features as graph-level representation
                    feat = g[nt].x.mean(dim=0).numpy()
                    if lbl == 0:
                        auth_features.append(feat)
                    else:
                        fake_features.append(feat)
                    break

        if not auth_features or not fake_features:
            print("  ⚠️  Cannot check — insufficient feature data")
            return {'severity': 'UNKNOWN'}

        auth_arr = np.array(auth_features)
        fake_arr = np.array(fake_features)

        # Per-feature Kolmogorov-Smirnov-like check (simplified)
        n_features = min(auth_arr.shape[1], fake_arr.shape[1])
        significant_features = 0
        feature_names = [
            'Atomic_Num', 'Degree', 'Total_Deg', 'Formal_Chg', 'Hybrid',
            'Aromatic', 'Total_Hs', 'Mass', 'Chiral', 'Chiral_Possible',
            'H_Donor', 'H_Acceptor', 'Halogen', 'Heteroatom', 'In_Ring',
            'Ring5', 'Ring6', 'Other_Ring', 'Radical_E', 'Arom_Nbr_Ratio',
            'Het_Nbr_Ratio', 'Hal_Nbr_Ratio', 'Chg_Nbr', 'SP', 'SP2', 'SP3'
        ]

        top_shifted = []
        for f_idx in range(n_features):
            a_mean = auth_arr[:, f_idx].mean()
            f_mean = fake_arr[:, f_idx].mean()
            a_std = auth_arr[:, f_idx].std()
            f_std = fake_arr[:, f_idx].std()
            pooled = np.sqrt((a_std**2 + f_std**2) / 2)
            d = abs(a_mean - f_mean) / max(pooled, 1e-6)

            fname = feature_names[f_idx] if f_idx < len(feature_names) else f'feat_{f_idx}'
            if d > 0.5:
                significant_features += 1
                top_shifted.append((fname, d, a_mean, f_mean))

        top_shifted.sort(key=lambda x: x[1], reverse=True)

        print(f"  Features with significant distribution shift (Cohen's d > 0.5):")
        print(f"  {significant_features} / {n_features} features")
        for fname, d, a_mean, f_mean in top_shifted[:10]:
            direction = "auth > fake" if a_mean > f_mean else "fake > auth"
            print(f"    {fname:<22} d={d:.3f}  ({direction})")

        if significant_features > n_features * 0.3:
            print(f"\n  ⚠️  >30% of features show significant shift!")
            print(f"     The generation process creates systematic feature differences.")
            print(f"     Models can exploit these as shortcuts.")
        else:
            print(f"\n  ✅  Feature distributions reasonably overlapping")

        return {
            'significant_features': significant_features,
            'total_features': n_features,
            'top_shifted': top_shifted[:10],
            'severity': 'CRITICAL' if significant_features > n_features * 0.3 else (
                'WARNING' if significant_features > n_features * 0.15 else 'OK')
        }

    def _test_trivial_classifiers(self):
        print("\n[2d] Testing trivial classifiers (shortcuts)...")

        # Extract simple features
        simple_features = []
        valid_labels = []

        for g, lbl in zip(self.graphs, self.labels):
            lbl_int = int(lbl) if isinstance(lbl, (int, np.integer)) else lbl
            try:
                total_nodes = sum(
                    g[nt].x.size(0) for nt in g.node_types if hasattr(g[nt], 'x')
                )
                total_edges = sum(
                    g[et].edge_index.size(1) for et in g.edge_types
                    if hasattr(g[et], 'edge_index')
                )
                num_node_types = len([
                    nt for nt in g.node_types
                    if hasattr(g[nt], 'x') and g[nt].x.size(0) > 0
                ])

                # Mean features from all nodes
                all_feats = []
                for nt in g.node_types:
                    if hasattr(g[nt], 'x') and g[nt].x.size(0) > 0:
                        all_feats.append(g[nt].x.mean(dim=0).numpy())

                if all_feats:
                    mean_feat = np.mean(all_feats, axis=0)
                else:
                    mean_feat = np.zeros(26)

                smiles_len = len(getattr(g, 'smiles', ''))

                feature_vec = [total_nodes, total_edges, num_node_types, smiles_len]
                feature_vec.extend(mean_feat[:10].tolist())  # First 10 chem features

                simple_features.append(feature_vec)
                valid_labels.append(lbl_int)
            except:
                continue

        if len(simple_features) < 100:
            print("  ⚠️  Insufficient data for trivial classifier test")
            return {'severity': 'UNKNOWN'}

        X = np.array(simple_features)
        y = np.array(valid_labels)

        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
        )

        # Test 1: Decision stump on each feature
        print(f"\n  Decision stump results (best single-feature classifier):")
        feature_labels = ['total_nodes', 'total_edges', 'num_types', 'smiles_len',
                          'feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4',
                          'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9']

        best_acc = 0
        best_feature = ""
        for f_idx in range(X.shape[1]):
            threshold = np.median(X[train_idx, f_idx])
            preds = (X[test_idx, f_idx] > threshold).astype(int)
            acc = accuracy_score(y[test_idx], preds)
            acc = max(acc, 1 - acc)  # Try both directions

            fname = feature_labels[f_idx] if f_idx < len(feature_labels) else f'feat_{f_idx}'
            if acc > best_acc:
                best_acc = acc
                best_feature = fname

            if acc > 0.6:
                print(f"    {fname:<20} accuracy: {acc:.3f}")

        print(f"\n  Best single-feature accuracy: {best_acc:.3f} ({best_feature})")

        # Test 2: Logistic regression on simple features
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y[train_idx])
            lr_acc = lr.score(X_test, y[test_idx])
            lr_preds = lr.predict(X_test)
            _, _, lr_f1, _ = precision_recall_fscore_support(
                y[test_idx], lr_preds, average='binary', zero_division=0
            )

            print(f"\n  Logistic Regression on simple features:")
            print(f"    Accuracy: {lr_acc:.3f}")
            print(f"    F1 Score: {lr_f1:.3f}")

            if lr_f1 > 0.75:
                print(f"  ⚠️  CRITICAL: Simple features achieve F1={lr_f1:.3f}!")
                print(f"     The GNN may be exploiting the same shortcuts.")
                print(f"     The dataset's generation process likely creates")
                print(f"     systematic differences detectable without graph structure.")
            elif lr_f1 > 0.6:
                print(f"  ⚠️  WARNING: Simple features achieve F1={lr_f1:.3f}")
            else:
                print(f"  ✅  Simple features insufficient (F1={lr_f1:.3f})")
                print(f"     Graph structure is likely needed → GNN advantage is real")

        except ImportError:
            lr_f1 = 0
            print("  sklearn not available for logistic regression test")

        return {
            'best_single_feature_acc': best_acc,
            'best_feature': best_feature,
            'logistic_regression_f1': lr_f1 if 'lr_f1' in dir() else None,
            'severity': 'CRITICAL' if best_acc > 0.7 or (lr_f1 and lr_f1 > 0.75) else (
                'WARNING' if best_acc > 0.6 else 'OK')
        }


# ─────────────────────────────────────────────────────────────
#  SECTION 3: DIFFICULTY BREAKDOWN ANALYSIS
# ─────────────────────────────────────────────────────────────

class DifficultyAnalyzer:
    """Analyzes performance across counterfeit difficulty levels."""

    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def run_analysis(self):
        print("\n" + "=" * 70)
        print("  TEST 3: COUNTERFEIT DIFFICULTY BREAKDOWN")
        print("=" * 70)

        # Collect metadata
        difficulty_counts = defaultdict(lambda: {'total': 0, 'auth': 0, 'fake': 0})
        source_counts = defaultdict(int)
        similarity_by_difficulty = defaultdict(list)

        for g, lbl in zip(self.graphs, self.labels):
            lbl_int = int(lbl) if isinstance(lbl, (int, np.integer)) else lbl

            diff = getattr(g, 'difficulty_level', getattr(g, 'difficulty', 'unknown'))
            ctype = getattr(g, 'counterfeit_type', 'unknown')
            source = getattr(g, 'source', 'unknown')
            similarity = getattr(g, 'similarity', None)

            key = f"{diff}" if diff != 'unknown' else ctype
            difficulty_counts[key]['total'] += 1
            if lbl_int == 0:
                difficulty_counts[key]['auth'] += 1
            else:
                difficulty_counts[key]['fake'] += 1

            source_counts[source] += 1

            if similarity is not None:
                similarity_by_difficulty[key].append(float(similarity))

        print("\n  Difficulty level distribution:")
        print(f"  {'Level':<25} {'Total':>8} {'Auth':>8} {'Fake':>8}")
        print(f"  {'-'*49}")
        for level, counts in sorted(difficulty_counts.items()):
            print(f"  {level:<25} {counts['total']:>8} {counts['auth']:>8} {counts['fake']:>8}")

        print(f"\n  Source distribution:")
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"    {source:<30} {count:>6}")

        if similarity_by_difficulty:
            print(f"\n  Tanimoto similarity to parent (counterfeits only):")
            print(f"  {'Difficulty':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
            print(f"  {'-'*57}")
            for level, sims in sorted(similarity_by_difficulty.items()):
                if sims:
                    arr = np.array(sims)
                    print(f"  {level:<25} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f}")

            # Check if high-similarity counterfeits dominate
            all_sims = []
            for sims in similarity_by_difficulty.values():
                all_sims.extend(sims)
            if all_sims:
                overall_mean = np.mean(all_sims)
                high_sim = sum(1 for s in all_sims if s > 0.8)
                low_sim = sum(1 for s in all_sims if s < 0.5)
                print(f"\n  Overall similarity: {overall_mean:.3f}")
                print(f"  High similarity (>0.8): {high_sim} ({high_sim/len(all_sims)*100:.1f}%)")
                print(f"  Low similarity  (<0.5): {low_sim} ({low_sim/len(all_sims)*100:.1f}%)")

                if overall_mean < 0.5:
                    print(f"  ⚠️  Low mean similarity → counterfeits are too different from originals")
                    print(f"     Easy to distinguish → inflated metrics")

        return difficulty_counts, source_counts


# ─────────────────────────────────────────────────────────────
#  SECTION 4: HGT vs GAT STRUCTURAL ADVANTAGE TEST
# ─────────────────────────────────────────────────────────────

class HeterogeneityAnalyzer:
    """Tests whether heterogeneous structure actually matters."""

    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def run_analysis(self):
        print("\n" + "=" * 70)
        print("  TEST 4: HETEROGENEOUS STRUCTURE VALUE ANALYSIS")
        print("=" * 70)

        # How many edge types are actually used per graph?
        edge_type_usage = []
        node_type_usage = []
        cross_type_edges = []

        for g in self.graphs[:5000]:
            active_node_types = [
                nt for nt in g.node_types
                if hasattr(g[nt], 'x') and g[nt].x.size(0) > 0
            ]
            node_type_usage.append(len(active_node_types))

            active_edge_types = 0
            cross_type = 0
            total_edges = 0
            for et in g.edge_types:
                if hasattr(g[et], 'edge_index') and g[et].edge_index.size(1) > 0:
                    active_edge_types += 1
                    n_edges = g[et].edge_index.size(1)
                    total_edges += n_edges
                    src_type, _, dst_type = et
                    if src_type != dst_type:
                        cross_type += n_edges

            edge_type_usage.append(active_edge_types)
            if total_edges > 0:
                cross_type_edges.append(cross_type / total_edges)

        print(f"\n  Active node types per graph: {np.mean(node_type_usage):.1f} ± {np.std(node_type_usage):.1f}")
        print(f"  Active edge types per graph: {np.mean(edge_type_usage):.1f} ± {np.std(edge_type_usage):.1f}")

        if cross_type_edges:
            print(f"  Cross-type edge ratio:       {np.mean(cross_type_edges):.3f} ± {np.std(cross_type_edges):.3f}")

        # If most graphs only have 1-2 node types, heterogeneity adds overhead with little benefit
        mostly_single_type = sum(1 for n in node_type_usage if n <= 2)
        print(f"\n  Graphs with ≤2 node types: {mostly_single_type} ({mostly_single_type/len(node_type_usage)*100:.1f}%)")

        if mostly_single_type > len(node_type_usage) * 0.5:
            print(f"  ⚠️  Most graphs have few node types — heterogeneous structure may add")
            print(f"     overhead without proportional benefit. GAT's simpler architecture")
            print(f"     avoids this overhead.")
        else:
            print(f"  ✅  Graphs use diverse node types — HGT should benefit")

        # Check if hetero information is discriminative
        print(f"\n  Node type composition difference between classes:")
        auth_composition = defaultdict(list)
        fake_composition = defaultdict(list)

        for g, lbl in zip(self.graphs[:5000], self.labels[:5000]):
            lbl_int = int(lbl) if isinstance(lbl, (int, np.integer)) else lbl
            total_nodes = sum(
                g[nt].x.size(0) for nt in g.node_types if hasattr(g[nt], 'x')
            )
            if total_nodes == 0:
                continue
            target = auth_composition if lbl_int == 0 else fake_composition
            for nt in g.node_types:
                if hasattr(g[nt], 'x'):
                    ratio = g[nt].x.size(0) / total_nodes
                    target[nt].append(ratio)

        print(f"  {'Type':<10} {'Auth Ratio':>12} {'Fake Ratio':>12} {'Diff':>10}")
        print(f"  {'-'*44}")
        for nt in sorted(set(list(auth_composition.keys()) + list(fake_composition.keys()))):
            a_ratio = np.mean(auth_composition[nt]) if auth_composition[nt] else 0
            f_ratio = np.mean(fake_composition[nt]) if fake_composition[nt] else 0
            diff = abs(a_ratio - f_ratio)
            marker = " ⚠️" if diff > 0.05 else ""
            print(f"  {nt:<10} {a_ratio:>12.4f} {f_ratio:>12.4f} {diff:>10.4f}{marker}")


# ─────────────────────────────────────────────────────────────
#  SECTION 5: RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────────────────────

def generate_recommendations(all_results):
    print("\n" + "=" * 70)
    print("  DIAGNOSIS SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    issues = []
    for section, results in all_results.items():
        if isinstance(results, dict):
            for check, result in results.items():
                if isinstance(result, dict) and result.get('severity') in ('CRITICAL', 'WARNING'):
                    issues.append((section, check, result['severity']))

    if not issues:
        print("\n  ✅  No major issues detected!")
        print("  GAT's higher performance may reflect genuine architectural advantage")
        print("  for this specific dataset.")
        return

    critical = [i for i in issues if i[2] == 'CRITICAL']
    warnings = [i for i in issues if i[2] == 'WARNING']

    if critical:
        print(f"\n  🔴 CRITICAL ISSUES ({len(critical)}):")
        for section, check, _ in critical:
            print(f"     • [{section}] {check}")

    if warnings:
        print(f"\n  🟡 WARNINGS ({len(warnings)}):")
        for section, check, _ in warnings:
            print(f"     • [{section}] {check}")

    print(f"\n  RECOMMENDED ACTIONS:")
    print(f"  {'─'*50}")

    action_num = 1

    # Parent-child leakage fix
    if any('parent_child' in c[1] for c in critical + warnings):
        print(f"\n  {action_num}. FIX PARENT-CHILD LEAKAGE (highest priority)")
        print(f"     Split by molecule FAMILY, not individual molecules.")
        print(f"     Group each authentic + all its counterfeits together,")
        print(f"     then split families into train/test.")
        print(f"     Code:")
        print(f"       from collections import defaultdict")
        print(f"       families = defaultdict(list)")
        print(f"       for i, g in enumerate(graphs):")
        print(f"           parent = getattr(g, 'original_smiles', g.smiles)")
        print(f"           families[parent].append(i)")
        print(f"       family_keys = list(families.keys())")
        print(f"       train_fam, test_fam = train_test_split(family_keys, ...)")
        action_num += 1

    # Artifact/shortcut fix
    if any('trivial' in c[1] or 'feature_distribution' in c[1] for c in critical + warnings):
        print(f"\n  {action_num}. IMPROVE COUNTERFEIT GENERATION")
        print(f"     Current string-replacement creates systematic artifacts.")
        print(f"     Options:")
        print(f"       a) Use RDKit substructure editing instead of SMILES string ops")
        print(f"       b) Ensure counterfeits match authentic descriptor distributions")
        print(f"       c) Add descriptor-based rejection sampling")
        print(f"       d) Use matched molecular pairs from ChEMBL (real transformations)")
        action_num += 1

    # Generalization test
    print(f"\n  {action_num}. RUN GENERALIZATION TEST")
    print(f"     Train on easy+medium counterfeits, test on hard+scaffold.")
    print(f"     If GAT drops more than HGT → HGT generalizes better → paper story.")
    action_num += 1

    print(f"\n  {action_num}. FOR THE ICTAI PAPER")
    print(f"     Include this diagnostic analysis as a section.")
    print(f"     Show that after fixing leakage/artifacts:")
    print(f"       - Both models' metrics may decrease")
    print(f"       - HGT's advantage in generalization becomes clearer")
    print(f"       - Explainability (your ex.py) works natively with HGT")
    print(f"     This turns a 'problem' into a 'contribution' about")
    print(f"     evaluation pitfalls in molecular GNN benchmarks.")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def load_dataset(path):
    """Load dataset with fallbacks."""
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path.absolute()}")
        return None, None

    try:
        loaded = torch.load(path, map_location='cpu')
    except:
        loaded = torch.load(path, weights_only=False, map_location='cpu')

    if isinstance(loaded, tuple) and len(loaded) >= 3:
        graphs, labels, metadata = loaded[0], loaded[1], loaded[2]
    elif isinstance(loaded, dict):
        graphs = loaded.get('hetero_graphs', loaded.get('graphs'))
        labels = loaded.get('labels')
    else:
        raise TypeError(f"Unknown format: {type(loaded)}")

    print(f"Loaded {len(graphs)} graphs")
    return graphs, labels


def main():
    print("=" * 70)
    print("  DIAGNOSTIC INVESTIGATION")
    print("  Why does Homogeneous GAT (92%) outperform HGT (82%)?")
    print("=" * 70)

    # Find dataset
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        patterns = ['*enhanced*.pt', '*pharma*.pt', '*hetero*.pt', '*.pt']
        found = []
        for pat in patterns:
            found.extend(Path('.').glob(pat))
        found = [f for f in sorted(set(found)) if 'best_model' not in str(f)]

        if not found:
            print("No dataset found. Usage: python diagnostic_investigation.py <dataset.pt>")
            return
        dataset_path = str(found[0])

    print(f"\nDataset: {dataset_path}")
    graphs, labels = load_dataset(dataset_path)
    if graphs is None:
        return

    all_results = {}

    # Run all diagnostics
    leakage = DataLeakageDetector(graphs, labels)
    all_results['data_leakage'] = leakage.run_all_checks()

    artifacts = ArtifactDetector(graphs, labels)
    all_results['artifacts'] = artifacts.run_all_checks()

    difficulty = DifficultyAnalyzer(graphs, labels)
    difficulty.run_analysis()

    heterogeneity = HeterogeneityAnalyzer(graphs, labels)
    heterogeneity.run_analysis()

    # Generate recommendations
    generate_recommendations(all_results)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"diagnostic_report_{timestamp}.txt"

    print(f"\n{'='*70}")
    print(f"  Report complete. Review findings above.")
    print(f"  Run with your dataset: python diagnostic_investigation.py <path.pt>")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()