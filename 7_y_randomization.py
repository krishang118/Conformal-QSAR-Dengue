"""
7_y_randomization.py  (v2 — revised after peer review)
=======================================================
Y-randomization (permutation) test — required for QSAR publication.

Key changes vs. v1:
  - N_PERM increased from 20 → 1,000 (R3 Major #2 demand).
  - p-value reported as empirical proportion p = k/N where k is the
    number of permutations ≥ real model score. With 1,000 permutations
    the minimum reportable p-value is 0.001 (1/1000); we report the
    exact count rather than claiming p < 0.0001 from a raw proportion.
  - Uses n_estimators=100 (instead of 300) for speed across 1,000 runs;
    this is standard practice for permutation tests and the reduced tree
    count has negligible effect on the permuted AUC distribution.
  - Progress printed every 50 permutations to track long run.

Output:
  results/y_randomization.json   — structured results
  figures/y_randomization.png    — updated figure (1,000-permutation histogram)
"""

import os, pickle, warnings, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                             average_precision_score, make_scorer)
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Y-Randomization Test  (v2 — 1,000 permutations)")
print("=" * 60)

with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)
with open('results/best_combo.json') as f:
    best = json.load(f)

y = data['y']
X = data[best['best_fp_key']]

print(f"  Feature matrix  : {X.shape}")
print(f"  Best combo      : {best['best_model_name']} + {best['best_fp_label']}")
print(f"  Canonical AUC   : {best['auc_roc']:.4f} ± {best['auc_roc_std']:.4f}  (from Stage 3)")
print(f"  N permutations  : 1,000")

# ── Model and CV setup ────────────────────────────────────────────────────
# n_estimators=100 for permutation speed (standard practice).
# The permuted-label AUC distribution is insensitive to tree count
# because shuffled labels yield near-random performance regardless.
model = RandomForestClassifier(
    n_estimators=100, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorers = {
    'AUC_ROC': make_scorer(roc_auc_score,          response_method='predict_proba'),
    'AUC_PR':  make_scorer(average_precision_score, response_method='predict_proba'),
    'MCC':     make_scorer(matthews_corrcoef),
}

# ── Real model score (independent 5-fold CV) ──────────────────────────────
print("\nFitting real model (independent 5-fold CV, n_estimators=100)...")
real_scores  = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
real_auc     = float(real_scores['test_AUC_ROC'].mean())
real_auc_std = float(real_scores['test_AUC_ROC'].std())
real_mcc     = float(real_scores['test_MCC'].mean())
real_aupr    = float(real_scores['test_AUC_PR'].mean())

print(f"  Real AUC-ROC : {real_auc:.4f} ± {real_auc_std:.4f}")
print(f"  Real MCC     : {real_mcc:.4f}")
print(f"  Real AUC-PR  : {real_aupr:.4f}")

# ── 1,000-permutation test ────────────────────────────────────────────────
N_PERM = 1000
print(f"\nRunning {N_PERM:,} permutations (shuffled labels)...")
print("  (Progress printed every 50 permutations)")

perm_aucs  = []
perm_mccs  = []
perm_auprs = []

rng = np.random.default_rng(42)
for i in range(N_PERM):
    y_shuf = rng.permutation(y)
    scores = cross_validate(model, X, y_shuf, cv=cv, scoring=scorers, n_jobs=-1)
    perm_aucs.append(float(scores['test_AUC_ROC'].mean()))
    perm_mccs.append(float(scores['test_MCC'].mean()))
    perm_auprs.append(float(scores['test_AUC_PR'].mean()))
    if (i + 1) % 50 == 0:
        print(f"  {i+1:4d}/{N_PERM} | "
              f"perm AUC mean so far: {np.mean(perm_aucs):.4f} ± {np.std(perm_aucs):.4f}")

perm_aucs  = np.array(perm_aucs)
perm_mccs  = np.array(perm_mccs)
perm_auprs = np.array(perm_auprs)

# ── p-values (exact empirical proportions) ────────────────────────────────
# p = (number of permutations with score >= real model) / N_PERM
# With N=1000, minimum reportable p = 0.001 (= 1/1000).
# We report the exact count k and proportion k/N; we do NOT claim
# p < 0.001 if k = 0 — we report p < 0.001 (i.e., 0/1000).
k_auc       = int((perm_aucs >= real_auc).sum())
k_mcc       = int((perm_mccs >= real_mcc).sum())
p_value_auc = k_auc / N_PERM
p_value_mcc = k_mcc / N_PERM

# Minimum bound: if k=0 out of 1000, report as p < 0.001
p_auc_str = f"< 0.001 ({k_auc}/{N_PERM})" if k_auc == 0 else f"= {p_value_auc:.4f} ({k_auc}/{N_PERM})"
p_mcc_str = f"< 0.001 ({k_mcc}/{N_PERM})" if k_mcc == 0 else f"= {p_value_mcc:.4f} ({k_mcc}/{N_PERM})"

print(f"\n{'='*60}")
print(f"Y-RANDOMIZATION RESULTS  (N = {N_PERM:,} permutations)")
print(f"{'='*60}")
print(f"  Real model    AUC-ROC : {real_auc:.4f} ± {real_auc_std:.4f}")
print(f"  Permuted mean AUC-ROC : {perm_aucs.mean():.4f} ± {perm_aucs.std():.4f}")
print(f"  Permuted max  AUC-ROC : {perm_aucs.max():.4f}")
print(f"  p-value (AUC-ROC)     : p {p_auc_str}  {'✓ SIGNIFICANT' if p_value_auc < 0.05 else '✗ NOT SIGNIFICANT'}")
print(f"")
print(f"  Real model    MCC     : {real_mcc:.4f}")
print(f"  Permuted mean MCC     : {perm_mccs.mean():.4f} ± {perm_mccs.std():.4f}")
print(f"  p-value (MCC)         : p {p_mcc_str}  {'✓ SIGNIFICANT' if p_value_mcc < 0.05 else '✗ NOT SIGNIFICANT'}")

conclusion = (
    "VALID: Model learns genuine SAR (p < 0.001, N=1,000 permutations)"
    if k_auc == 0 else
    f"VALID: Model learns genuine SAR (p = {p_value_auc:.4f})"
    if p_value_auc < 0.05 else
    "INVALID: Cannot distinguish from noise"
)
print(f"\n  Conclusion: {conclusion}")

# ── Save results ──────────────────────────────────────────────────────────
results_out = {
    'n_permutations':       N_PERM,
    'real_auc_roc':         round(real_auc, 4),
    'real_auc_roc_std':     round(real_auc_std, 4),
    'real_mcc':             round(real_mcc, 4),
    'real_auc_pr':          round(real_aupr, 4),
    'perm_auc_mean':        round(float(perm_aucs.mean()), 4),
    'perm_auc_std':         round(float(perm_aucs.std()),  4),
    'perm_auc_max':         round(float(perm_aucs.max()),  4),
    'perm_mcc_mean':        round(float(perm_mccs.mean()), 4),
    'perm_mcc_std':         round(float(perm_mccs.std()),  4),
    'k_auc_ge_real':        k_auc,
    'k_mcc_ge_real':        k_mcc,
    'p_value_auc':          p_value_auc,
    'p_value_mcc':          p_value_mcc,
    'p_value_auc_str':      f"p {p_auc_str}",
    'p_value_mcc_str':      f"p {p_mcc_str}",
    'conclusion':           conclusion,
    'all_perm_aucs':        [round(float(v), 4) for v in perm_aucs],
    'all_perm_mccs':        [round(float(v), 4) for v in perm_mccs],
}
with open('results/y_randomization.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print("\n  ✓ results/y_randomization.json")

# ── Figure: Y-randomization histogram ────────────────────────────────────
print("\nGenerating figure: y_randomization.png ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f'Y-Randomization Test — Dengue NS2B-NS3 QSAR\n'
    f'Random Forest + {best["best_fp_label"]} | N = {N_PERM:,} permutations',
    fontsize=11, fontweight='bold'
)

for ax, (vals, real_val, label, color, p_str) in zip(axes, [
    (perm_aucs, real_auc, 'AUC-ROC', '#3d7ebf', f"p {p_auc_str}"),
    (perm_mccs, real_mcc, 'MCC',     '#e05c5c', f"p {p_mcc_str}"),
]):
    ax.hist(vals, bins=40, color=color, alpha=0.70, edgecolor='white',
            label=f'Permuted labels\nmean={vals.mean():.3f} ± {vals.std():.3f}\nmax={vals.max():.3f}')
    ax.axvline(real_val, color='black', linewidth=2.5, linestyle='-',
               label=f'Real model ({label} = {real_val:.4f})')
    ax.axvline(vals.mean(), color=color, linewidth=1.5, linestyle='--', alpha=0.7,
               label='Permuted mean')

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{label}: Real vs. Permuted Labels\n({p_str})',
                 fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/y_randomization.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/y_randomization.png")
print()
print("✓ Y-randomization (v2, 1,000 permutations) complete.")
