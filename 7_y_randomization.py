import os, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, make_scorer
warnings.filterwarnings('ignore')

os.makedirs('results',  exist_ok=True)
os.makedirs('figures',  exist_ok=True)

print("=" * 60)
print("Y-Randomization Test")
print("=" * 60)
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)
with open('results/best_combo.json') as f:
    best = json.load(f)

y = data['y']
X = data[best['best_fp_key']]

print(f"  Feature matrix     : {X.shape}")
print(f"  Best combo         : {best['best_model_name']} + {best['best_fp_label']}")
print(f"  Real AUC-ROC       : {best['auc_roc']:.4f} ± {best['auc_roc_std']:.4f}")
print(f"  N permutations     : 20")

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

print("\nFitting real model (5-fold CV)...")
real_scores = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
real_auc    = real_scores['test_AUC_ROC'].mean()
real_auc_std = real_scores['test_AUC_ROC'].std()
real_mcc    = real_scores['test_MCC'].mean()
real_aupr   = real_scores['test_AUC_PR'].mean()
print(f"  Real AUC-ROC : {real_auc:.4f} ± {real_auc_std:.4f}")
print(f"  Real MCC     : {real_mcc:.4f}")
print(f"  Real AUC-PR  : {real_aupr:.4f}")

print("\nRunning 20 permutations (shuffled labels)...")
N_PERM = 20
perm_aucs = []
perm_mccs = []
perm_auprs = []

rng = np.random.default_rng(42)
for i in range(N_PERM):
    y_shuf = rng.permutation(y)
    scores = cross_validate(model, X, y_shuf, cv=cv, scoring=scorers, n_jobs=-1)
    perm_aucs.append(scores['test_AUC_ROC'].mean())
    perm_mccs.append(scores['test_MCC'].mean())
    perm_auprs.append(scores['test_AUC_PR'].mean())
    print(f"  Perm {i+1:02d}/20: AUC-ROC={perm_aucs[-1]:.4f}  MCC={perm_mccs[-1]:.4f}")

perm_aucs  = np.array(perm_aucs)
perm_mccs  = np.array(perm_mccs)
perm_auprs = np.array(perm_auprs)

p_value_auc = (perm_aucs >= real_auc).mean()
p_value_mcc = (perm_mccs >= real_mcc).mean()

print(f"\n{'='*60}")
print(f"Y-RANDOMIZATION RESULTS")
print(f"{'='*60}")
print(f"  Real model    AUC-ROC : {real_auc:.4f} ± {real_auc_std:.4f}")
print(f"  Permuted mean AUC-ROC : {perm_aucs.mean():.4f} ± {perm_aucs.std():.4f}")
print(f"  p-value (AUC-ROC)     : {p_value_auc:.4f}  {'✓ SIGNIFICANT' if p_value_auc < 0.05 else '✗ NOT SIGNIFICANT'}")
print(f"  Real model    MCC     : {real_mcc:.4f}")
print(f"  Permuted mean MCC     : {perm_mccs.mean():.4f} ± {perm_mccs.std():.4f}")
print(f"  p-value (MCC)         : {p_value_mcc:.4f}  {'✓ SIGNIFICANT' if p_value_mcc < 0.05 else '✗ NOT SIGNIFICANT'}")

conclusion = (
    "VALID: Model learns genuine SAR" if p_value_auc < 0.05
    else "INVALID: Cannot distinguish from noise"
)
print(f"\n  Conclusion: {conclusion}")

results_out = {
    'real_auc_roc':     float(round(real_auc, 4)),
    'real_auc_roc_std': float(round(real_auc_std, 4)),
    'real_mcc':         float(round(real_mcc, 4)),
    'real_auc_pr':      float(round(real_aupr, 4)),
    'perm_auc_mean':    float(round(perm_aucs.mean(), 4)),
    'perm_auc_std':     float(round(perm_aucs.std(), 4)),
    'perm_mcc_mean':    float(round(perm_mccs.mean(), 4)),
    'p_value_auc':      float(p_value_auc),
    'p_value_mcc':      float(p_value_mcc),
    'n_permutations':   N_PERM,
    'all_perm_aucs':    [float(round(v, 4)) for v in perm_aucs],
    'all_perm_mccs':    [float(round(v, 4)) for v in perm_mccs],
    'conclusion':       conclusion,
}
with open('results/y_randomization.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print("  ✓ results/y_randomization.json")

print("\nGenerating Figure 7: Y-randomization plot...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('Y-Randomization Test — Dengue NS2B-NS3 QSAR\n'
             f'Random Forest + {best["best_fp_label"]} ({N_PERM} permutations)',
             fontsize=11, fontweight='bold')

for ax, (vals, real_val, label, color, pval) in zip(axes, [
    (perm_aucs, real_auc, 'AUC-ROC', '#3d7ebf', p_value_auc),
    (perm_mccs, real_mcc, 'MCC',     '#e05c5c', p_value_mcc),
]):
    ax.hist(vals, bins=10, color=color, alpha=0.70, edgecolor='white',
            label=f'Permuted labels\n(mean={vals.mean():.3f}±{vals.std():.3f})')
    ax.axvline(real_val, color='black', linewidth=2.5, linestyle='-',
               label=f'Real model ({label}={real_val:.3f})')
    ax.axvline(vals.mean(), color=color, linewidth=1.5, linestyle='--', alpha=0.8)

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{label}: Real vs Permuted\n(p = {pval:.3f}{"*" if pval<0.05 else " n.s."})',
                 fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/y_randomization.png', dpi=300, bbox_inches='tight')
plt.close()
