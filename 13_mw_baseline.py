"""
13_mw_baseline.py
=================
Molecular Weight Confounding Control Analysis.

Motivation (Reviewer 1 Major #3, Reviewer 3 minor):
  Active compounds have significantly higher mean MW (522.9 Da) than
  inactives (391.2 Da). The top SHAP descriptors (QED, SMR_VSA, SlogP_VSA)
  are correlated with MW. The reviewer argues the RF model may be
  exploiting MW as a size shortcut rather than learning genuine
  pharmacophoric constraints.

This script tests that hypothesis directly:
  1. Build a trivial MW-only logistic regression baseline (single feature).
  2. Compare its 5-fold CV performance to RF + RDKit2D on identical folds.
  3. If RF >> MW-only, the model is learning beyond size.
  4. Additionally test RF on RDKit2D descriptors with MW removed,
     to confirm the descriptor advantage is not MW-driven.

Output:
  results/mw_baseline_results.json
  figures/mw_baseline_comparison.png
"""

import os, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              matthews_corrcoef, make_scorer)
from rdkit import Chem
from rdkit.Chem import Descriptors
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 65)
print("Molecular Weight Confounding Control Analysis")
print("=" * 65)

with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)
with open('results/best_combo.json') as f:
    best = json.load(f)

y       = data['y']
X_rdkit = data['rdkit']     # full RDKit2D (201 descriptors)
smiles  = data['smiles']

print(f"  Compounds  : {len(y):,}")
print(f"  Active     : {int(y.sum()):,}  ({y.mean():.1%})")
print(f"  RDKit2D    : {X_rdkit.shape[1]} descriptors")

# ── Compute Molecular Weight for every compound ───────────────────────────
print("\nComputing Molecular Weight from SMILES...")
mw_vals = []
for smi in smiles:
    mol = Chem.MolFromSmiles(str(smi))
    mw_vals.append(Descriptors.ExactMolWt(mol) if mol else np.nan)
mw_vals = np.array(mw_vals)

# Fill any NaN with median (should be none)
mw_vals = np.where(np.isnan(mw_vals), np.nanmedian(mw_vals), mw_vals)
X_mw = mw_vals.reshape(-1, 1)

# Identify the MW column index in RDKit2D for drop experiment
# RDKit2D descriptor list (same order as featurize.py)
from rdkit.Chem import Descriptors as D
desc_names = [name for name, _ in D.descList]
# Find columns that are MW-related
mw_related = [i for i, name in enumerate(desc_names) if
              any(kw in name.lower() for kw in ['mw', 'molwt', 'weight', 'exactmolwt'])]
print(f"\nMW-related descriptor indices in RDKit2D: {mw_related}")
print(f"  Names: {[desc_names[i] for i in mw_related]}")

# Build RDKit2D without MW-related columns
keep_cols = [i for i in range(X_rdkit.shape[1]) if i not in mw_related]
X_rdkit_nomw = X_rdkit[:, keep_cols]
print(f"  RDKit2D with MW removed: {X_rdkit_nomw.shape[1]} descriptors (dropped {len(mw_related)})")

# ── Cross-validation setup ────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorers = {
    'AUC_ROC': make_scorer(roc_auc_score,          response_method='predict_proba'),
    'AUC_PR':  make_scorer(average_precision_score, response_method='predict_proba'),
    'MCC':     make_scorer(matthews_corrcoef),
}

results = {}

# ── Model 1: MW-only logistic regression (trivial baseline) ───────────────
print("\n" + "─" * 65)
print("Model 1: MW-only Logistic Regression (trivial size baseline)")
mw_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(class_weight='balanced', max_iter=1000,
                                   random_state=42))
])
scores_mw = cross_validate(mw_pipe, X_mw, y, cv=cv, scoring=scorers)
mw_auc  = scores_mw['test_AUC_ROC'].mean()
mw_aupr = scores_mw['test_AUC_PR'].mean()
mw_mcc  = scores_mw['test_MCC'].mean()
mw_auc_std = scores_mw['test_AUC_ROC'].std()
print(f"  AUC-ROC : {mw_auc:.4f} ± {mw_auc_std:.4f}")
print(f"  AUC-PR  : {mw_aupr:.4f}")
print(f"  MCC     : {mw_mcc:.4f}")
results['mw_only'] = {
    'label':       'MW-only LR',
    'auc_roc':     round(mw_auc, 4),
    'auc_roc_std': round(mw_auc_std, 4),
    'auc_pr':      round(mw_aupr, 4),
    'mcc':         round(mw_mcc, 4),
}

# ── Model 2: RF + MW-only (non-linear MW baseline) ────────────────────────
print("\n" + "─" * 65)
print("Model 2: RF + MW-only (non-linear size baseline)")
rf_mw = RandomForestClassifier(
    n_estimators=300, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
scores_rf_mw = cross_validate(rf_mw, X_mw, y, cv=cv, scoring=scorers)
rf_mw_auc    = scores_rf_mw['test_AUC_ROC'].mean()
rf_mw_aupr   = scores_rf_mw['test_AUC_PR'].mean()
rf_mw_mcc    = scores_rf_mw['test_MCC'].mean()
rf_mw_std    = scores_rf_mw['test_AUC_ROC'].std()
print(f"  AUC-ROC : {rf_mw_auc:.4f} ± {rf_mw_std:.4f}")
print(f"  AUC-PR  : {rf_mw_aupr:.4f}")
print(f"  MCC     : {rf_mw_mcc:.4f}")
results['rf_mw_only'] = {
    'label':       'RF + MW-only',
    'auc_roc':     round(rf_mw_auc, 4),
    'auc_roc_std': round(rf_mw_std, 4),
    'auc_pr':      round(rf_mw_aupr, 4),
    'mcc':         round(rf_mw_mcc, 4),
}

# ── Model 3: RF + full RDKit2D (the primary model) ────────────────────────
print("\n" + "─" * 65)
print("Model 3: RF + full RDKit2D (primary benchmark model)")
rf_full = RandomForestClassifier(
    n_estimators=300, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
scores_full = cross_validate(rf_full, X_rdkit, y, cv=cv, scoring=scorers)
full_auc    = scores_full['test_AUC_ROC'].mean()
full_aupr   = scores_full['test_AUC_PR'].mean()
full_mcc    = scores_full['test_MCC'].mean()
full_std    = scores_full['test_AUC_ROC'].std()
print(f"  AUC-ROC : {full_auc:.4f} ± {full_std:.4f}")
print(f"  AUC-PR  : {full_aupr:.4f}")
print(f"  MCC     : {full_mcc:.4f}")
results['rf_rdkit_full'] = {
    'label':       'RF + RDKit2D (full)',
    'auc_roc':     round(full_auc, 4),
    'auc_roc_std': round(full_std, 4),
    'auc_pr':      round(full_aupr, 4),
    'mcc':         round(full_mcc, 4),
}

# ── Model 4: RF + RDKit2D with MW descriptors removed ────────────────────
print("\n" + "─" * 65)
print("Model 4: RF + RDKit2D (MW-removed) — ablation control")
rf_nomw = RandomForestClassifier(
    n_estimators=300, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
scores_nomw = cross_validate(rf_nomw, X_rdkit_nomw, y, cv=cv, scoring=scorers)
nomw_auc    = scores_nomw['test_AUC_ROC'].mean()
nomw_aupr   = scores_nomw['test_AUC_PR'].mean()
nomw_mcc    = scores_nomw['test_MCC'].mean()
nomw_std    = scores_nomw['test_AUC_ROC'].std()
print(f"  AUC-ROC : {nomw_auc:.4f} ± {nomw_std:.4f}")
print(f"  AUC-PR  : {nomw_aupr:.4f}")
print(f"  MCC     : {nomw_mcc:.4f}")
results['rf_rdkit_nomw'] = {
    'label':       f'RF + RDKit2D (MW removed, {X_rdkit_nomw.shape[1]} desc)',
    'auc_roc':     round(nomw_auc, 4),
    'auc_roc_std': round(nomw_std, 4),
    'auc_pr':      round(nomw_aupr, 4),
    'mcc':         round(nomw_mcc, 4),
}

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY — MW CONFOUNDING CONTROL")
print(f"{'='*65}")
print(f"\n  {'Model':<40} {'AUC-ROC':>10} {'AUC-PR':>8} {'MCC':>8}")
print(f"  {'-'*68}")
for key, r in results.items():
    marker = " ◄ PRIMARY" if key == 'rf_rdkit_full' else ""
    print(f"  {r['label']:<40} {r['auc_roc']:.4f}±{r['auc_roc_std']:.4f}  {r['auc_pr']:.4f}  {r['mcc']:.4f}{marker}")

delta_vs_mw_lr  = full_auc - mw_auc
delta_vs_rf_mw  = full_auc - rf_mw_auc
delta_nomw      = full_auc - nomw_auc

print(f"\n  RF + RDKit2D improvement over MW-only LR   : +{delta_vs_mw_lr:.4f} AUC-ROC")
print(f"  RF + RDKit2D improvement over RF + MW-only : +{delta_vs_rf_mw:.4f} AUC-ROC")
print(f"  RF + RDKit2D vs RF + RDKit2D (MW-removed)  : {delta_nomw:+.4f} AUC-ROC")

# Interpretation
print(f"\n  Interpretation:")
if delta_vs_rf_mw > 0.05:
    print(f"  ✓ RF + RDKit2D substantially outperforms RF + MW-only (+{delta_vs_rf_mw:.4f}).")
    print(f"    The model is NOT simply exploiting molecular size.")
elif delta_vs_rf_mw > 0.01:
    print(f"  ✓ RF + RDKit2D meaningfully outperforms RF + MW-only (+{delta_vs_rf_mw:.4f}).")
    print(f"    MW contributes but is NOT the sole driver of performance.")
else:
    print(f"  ⚠ RF + RDKit2D offers only marginal gain over RF + MW-only ({delta_vs_rf_mw:+.4f}).")
    print(f"    MW confounding may be significant — discuss as limitation.")

if abs(delta_nomw) < 0.01:
    print(f"  ✓ Removing MW descriptors has negligible effect ({delta_nomw:+.4f}).")
    print(f"    Model performance is not MW-dependent.")
else:
    print(f"  Note: Removing MW descriptors changes AUC-ROC by {delta_nomw:+.4f}.")

# Save
json_out = {
    'models': results,
    'delta_rf_rdkit_vs_mw_lr':   round(delta_vs_mw_lr, 4),
    'delta_rf_rdkit_vs_rf_mw':   round(delta_vs_rf_mw, 4),
    'delta_rf_rdkit_vs_nomw':    round(delta_nomw, 4),
    'mw_removed_cols':           mw_related,
    'mw_removed_names':          [desc_names[i] for i in mw_related],
}
with open('results/mw_baseline_results.json', 'w') as f:
    json.dump(json_out, f, indent=2)
print("\n  ✓ results/mw_baseline_results.json")

# ── Figure ────────────────────────────────────────────────────────────────
print("\nGenerating figure: mw_baseline_comparison.png ...")

labels     = [r['label'] for r in results.values()]
auc_vals   = [r['auc_roc'] for r in results.values()]
auc_errs   = [r['auc_roc_std'] for r in results.values()]
aupr_vals  = [r['auc_pr'] for r in results.values()]
mcc_vals   = [r['mcc'] for r in results.values()]

colors = ['#e05c5c', '#f39c12', '#2ecc71', '#3d7ebf']

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.suptitle('Molecular Weight Confounding Control\n'
             'RF + RDKit2D vs. MW-only baselines — 5-fold stratified CV',
             fontsize=11, fontweight='bold')

x = np.arange(len(labels))
short_labels = ['MW-only\nLR', 'RF +\nMW-only', 'RF +\nRDKit2D\n(full)', 'RF +\nRDKit2D\n(no MW)']

for ax, (vals, errs, title, ylabel) in zip(axes, [
    (auc_vals, auc_errs, 'AUC-ROC', 'AUC-ROC'),
    (aupr_vals, [0]*4,   'AUC-PR',  'AUC-PR'),
    (mcc_vals,  [0]*4,   'MCC',     'MCC'),
]):
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white', width=0.6)
    if any(e > 0 for e in errs):
        ax.errorbar(x, vals, yerr=errs, fmt='none', color='black',
                    capsize=5, linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, min(1.12, max(vals) + 0.15))
    ax.axhline(0.5, color='grey', linewidth=1, linestyle='--', alpha=0.5,
               label='Random (AUC=0.5)')
    ax.grid(axis='y', alpha=0.3)
    # Highlight primary model
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(2)

axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/mw_baseline_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/mw_baseline_comparison.png")
print()
print("✓ MW confounding control analysis complete.")
