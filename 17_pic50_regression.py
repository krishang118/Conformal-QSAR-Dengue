"""
17_pic50_regression.py
=======================
pIC50 regression model + conformal prediction intervals.

If scaffold-split R² >= 0.5 → include in paper.
If R² < 0.5 → skip (too noisy to be credible).

Output:
  results/pic50_regression.json
  figures/pic50_regression.png   (if R² sufficient)
"""

import pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit.Chem import DataStructs, AllChem

warnings.filterwarnings('ignore')

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
FIGURES_DIR = SCRIPT_DIR / 'figures'

print("=" * 60)
print("pIC50 Regression + Conformal Prediction Intervals")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────
with open(SCRIPT_DIR / 'data' / 'features.pkl', 'rb') as f:
    data = pickle.load(f)
X      = data['rdkit']
y_cls  = data['y']
smiles = list(data['smiles'])
df     = pd.read_csv(SCRIPT_DIR / 'data' / 'chembl_dengue_clean.csv')
y_reg  = df['pIC50'].values   # continuous label

print(f"Dataset: {len(y_cls):,} compounds")
print(f"pIC50 range: {y_reg.min():.2f} – {y_reg.max():.2f}  mean={y_reg.mean():.2f}")
print(f"Actives:  pIC50 range {y_reg[y_cls==1].min():.2f}–{y_reg[y_cls==1].max():.2f}")
print(f"Inactives: pIC50 range {y_reg[y_cls==0].min():.2f}–{y_reg[y_cls==0].max():.2f}")

# ── Stratified 5-fold CV (all compounds) ───────────────────────
print("\n[1] 5-fold CV regression metrics (all compounds)...")
rf_reg = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                                random_state=42, n_jobs=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(rf_reg, X, y_reg, cv=kf)

r2_cv   = r2_score(y_reg, y_pred_cv)
mae_cv  = mean_absolute_error(y_reg, y_pred_cv)
rmse_cv = np.sqrt(mean_squared_error(y_reg, y_pred_cv))
print(f"  R²   : {r2_cv:.4f}")
print(f"  MAE  : {mae_cv:.4f}")
print(f"  RMSE : {rmse_cv:.4f}")

# ── Scaffold split regression ───────────────────────────────────
print("\n[2] Scaffold split regression metrics...")

# Build scaffold split (same Butina logic as script 8)
mols = [Chem.MolFromSmiles(s) for s in smiles]
fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols if m]
valid_mask = [m is not None for m in mols]

dists = []
nfps = len(fps)
for i in range(nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - s for s in sims])

clusters = Butina.ClusterData(dists, nfps, 0.35, isDistData=True)

# 80/20 scaffold split
train_idx_sc = []; test_idx_sc = []
n_train_target = int(0.8 * len(smiles))
valid_indices = [i for i, m in enumerate(mols) if m is not None]
cluster_order = sorted(range(len(clusters)), key=lambda c: -len(clusters[c]))

assigned = set()
for ci in cluster_order:
    cluster_members = [valid_indices[j] for j in clusters[ci] if j < len(valid_indices)]
    if len(train_idx_sc) < n_train_target:
        train_idx_sc.extend(cluster_members)
    else:
        test_idx_sc.extend(cluster_members)
    assigned.update(cluster_members)

train_idx_sc = np.array(train_idx_sc)
test_idx_sc  = np.array(test_idx_sc)

# Scaffold split regression
rf_sc = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                               random_state=42, n_jobs=-1)
rf_sc.fit(X[train_idx_sc], y_reg[train_idx_sc])
y_pred_sc = rf_sc.predict(X[test_idx_sc])
y_true_sc = y_reg[test_idx_sc]
y_cls_sc  = y_cls[test_idx_sc]

r2_sc   = r2_score(y_true_sc, y_pred_sc)
mae_sc  = mean_absolute_error(y_true_sc, y_pred_sc)
rmse_sc = np.sqrt(mean_squared_error(y_true_sc, y_pred_sc))

print(f"  Scaffold split: train={len(train_idx_sc)}, test={len(test_idx_sc)}")
print(f"  R²   : {r2_sc:.4f}")
print(f"  MAE  : {mae_sc:.4f}")
print(f"  RMSE : {rmse_sc:.4f}")

# Active-only R² (the meaningful part)
act_mask = y_cls_sc == 1
r2_act = r2_score(y_true_sc[act_mask], y_pred_sc[act_mask]) if act_mask.sum() > 5 else np.nan
print(f"  R² (actives only, n={act_mask.sum()}): {r2_act:.4f}")

# Verdict
THRESHOLD = 0.50
if r2_sc >= THRESHOLD:
    print(f"\n  VERDICT: R²={r2_sc:.3f} >= {THRESHOLD} — INCLUDE IN PAPER ✓")
    include = True
else:
    print(f"\n  VERDICT: R²={r2_sc:.3f} < {THRESHOLD} — TOO NOISY, SKIP ✗")
    include = False

# ── Conformal Prediction Intervals (if R² sufficient) ──────────
cp_results = {}
if include:
    print("\n[3] Conformal Prediction Intervals (scaffold split)...")

    # Split training into proper train + calibration for CP
    # Use 80% of train as model train, 20% as calibration
    n_train = len(train_idx_sc)
    cal_size = int(0.2 * n_train)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_train)
    cal_idx  = train_idx_sc[perm[:cal_size]]
    prop_idx = train_idx_sc[perm[cal_size:]]

    # Train on proper train
    rf_cp = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                                   random_state=42, n_jobs=-1)
    rf_cp.fit(X[prop_idx], y_reg[prop_idx])

    # Calibration residuals (absolute)
    cal_pred = rf_cp.predict(X[cal_idx])
    residuals = np.abs(y_reg[cal_idx] - cal_pred)

    # Test predictions
    test_pred = rf_cp.predict(X[test_idx_sc])

    for conf in [0.80, 0.90, 0.95]:
        alpha = 1 - conf
        q = np.quantile(residuals, 1 - alpha)  # conformal quantile
        intervals = [(p - q, p + q) for p in test_pred]
        covered = sum(lo <= true <= hi
                      for (lo, hi), true in zip(intervals, y_true_sc))
        efficiency = np.mean([hi - lo for lo, hi in intervals])
        cp_results[f'{conf:.0%}'] = {
            'quantile': round(float(q), 4),
            'coverage': round(covered / len(y_true_sc), 4),
            'interval_width': round(float(efficiency), 4),
        }
        print(f"  {conf:.0%} confidence: q={q:.3f}  coverage={covered/len(y_true_sc):.3f}"
              f"  width={efficiency:.3f} pIC50 units")

    # ── Figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('pIC50 Regression — RF + RDKit2D | DENV NS2B-NS3\n'
                 f'Scaffold Split: R²={r2_sc:.3f}, MAE={mae_sc:.3f}, RMSE={rmse_sc:.3f}',
                 fontsize=11, fontweight='bold')

    # Panel A: Predicted vs Actual (scaffold split)
    ax = axes[0]
    colors = ['#e05c5c' if c == 1 else '#a8c8e8' for c in y_cls_sc]
    ax.scatter(y_true_sc, y_pred_sc, c=colors, s=15, alpha=0.6)
    lims = [min(y_true_sc.min(), y_pred_sc.min()) - 0.1,
            max(y_true_sc.max(), y_pred_sc.max()) + 0.1]
    ax.plot(lims, lims, 'k--', lw=1.2, label='Perfect')
    ax.set_xlabel('Observed pIC50', fontsize=10)
    ax.set_ylabel('Predicted pIC50', fontsize=10)
    ax.set_title(f'(A)  Predicted vs Observed\nScaffold split (test n={len(test_idx_sc)})',
                 fontsize=10, fontweight='bold')
    patches = [mpatches.Patch(color='#e05c5c', label='Active'),
               mpatches.Patch(color='#a8c8e8', label='Inactive')]
    ax.legend(handles=patches, fontsize=8)
    ax.text(0.05, 0.92, f'R²={r2_sc:.3f}\nRMSE={rmse_sc:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(alpha=0.3)

    # Panel B: Residuals distribution
    ax2 = axes[1]
    residuals_test = y_true_sc - y_pred_sc
    ax2.hist(residuals_test, bins=30, color='#3d7ebf', alpha=0.75, edgecolor='white')
    ax2.axvline(0, color='red', lw=1.5, ls='--', label='Zero residual')
    ax2.axvline(np.percentile(np.abs(residuals), 0.90), color='orange',
                lw=1.2, ls='--', label=f'90% CP q={np.quantile(residuals, 0.90):.2f}')
    ax2.set_xlabel('Residual (observed − predicted pIC50)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('(B)  Residual Distribution\n(test set)', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Panel C: CP interval width vs confidence
    ax3 = axes[2]
    conf_levels = [float(k.strip('%'))/100 for k in cp_results]
    widths      = [v['interval_width'] for v in cp_results.values()]
    coverages   = [v['coverage'] for v in cp_results.values()]
    ax3.bar([f'{c:.0%}' for c in conf_levels], widths,
            color='#2ecc71', alpha=0.8)
    for i, (w, cov, cl) in enumerate(zip(widths, coverages, conf_levels)):
        ax3.text(i, w + 0.01, f'cov={cov:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax3.set_xlabel('Confidence level', fontsize=10)
    ax3.set_ylabel('Prediction interval width (pIC50 units)', fontsize=10)
    ax3.set_title('(C)  CP Interval Width vs Confidence\n(regression, scaffold split)',
                  fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pic50_regression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ figures/pic50_regression.png")

# ── Save results ───────────────────────────────────────────────
results = {
    'include_in_paper': include,
    'cv_r2': round(float(r2_cv), 4),
    'cv_mae': round(float(mae_cv), 4),
    'cv_rmse': round(float(rmse_cv), 4),
    'scaffold_r2': round(float(r2_sc), 4),
    'scaffold_mae': round(float(mae_sc), 4),
    'scaffold_rmse': round(float(rmse_sc), 4),
    'scaffold_r2_actives_only': round(float(r2_act), 4) if not np.isnan(r2_act) else None,
    'n_train': int(len(train_idx_sc)),
    'n_test': int(len(test_idx_sc)),
    'cp_intervals': cp_results,
}
with open(RESULTS_DIR / 'pic50_regression.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ results/pic50_regression.json")

print("\n" + "=" * 60)
print(f"VERDICT: R² (scaffold) = {r2_sc:.4f} — {'INCLUDE' if include else 'SKIP'}")
print("=" * 60)
