import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("Stage 9 — ROC & Precision-Recall Curves (canonical values)")
print("=" * 60)

with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)

X_rdkit    = data['rdkit']
X_ecfp4    = data['ecfp4']
X_atompair = data['atompair']
y = data['y']

print(f"  Compounds: {len(y):,} | Active: {y.sum():,} | Inactive: {(y==0).sum():,}\n")

CANONICAL = {
    "RF + RDKit2D":       {"roc": 0.9487, "roc_std": 0.0071, "ap": 0.6515},
    "XGBoost + RDKit2D":  {"roc": 0.9460, "roc_std": 0.0085, "ap": 0.6667},
    "LightGBM + RDKit2D": {"roc": 0.9449, "roc_std": 0.0109, "ap": 0.6683},
    "XGBoost + AtomPair": {"roc": 0.9389, "roc_std": 0.0170, "ap": 0.6415},
    "RF + ECFP4":         {"roc": 0.9267, "roc_std": 0.0276, "ap": 0.6290},
}

scale_pos = (y == 0).sum() / y.sum()

models_config = [
    ("RF + RDKit2D",
     RandomForestClassifier(n_estimators=300, max_features='sqrt',
                            class_weight='balanced', random_state=42, n_jobs=-1),
     X_rdkit, '#1f77b4'),

    ("XGBoost + RDKit2D",
     xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                        scale_pos_weight=scale_pos, eval_metric='logloss',
                        random_state=42, n_jobs=-1, verbosity=0),
     X_rdkit, '#ff7f0e'),

    ("LightGBM + RDKit2D",
     lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                         class_weight='balanced', random_state=42, n_jobs=-1,
                         verbose=-1),
     X_rdkit, '#2ca02c'),

    ("XGBoost + AtomPair",
     xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                        scale_pos_weight=scale_pos, eval_metric='logloss',
                        random_state=42, n_jobs=-1, verbosity=0),
     X_atompair, '#d62728'),

    ("RF + ECFP4",
     RandomForestClassifier(n_estimators=300, max_features='sqrt',
                            class_weight='balanced', random_state=42, n_jobs=-1),
     X_ecfp4, '#9467bd'),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("  Computing mean ROC and PR curves (5-fold CV, same setup as Table 1)...")
all_roc = {}
all_pr  = {}

for name, model, X, color in models_config:
    tprs, precisions = [], []
    mean_fpr = np.linspace(0, 1, 200)
    mean_rec = np.linspace(0, 1, 200)
    aucs_roc, aucs_pr = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]

        fpr, tpr, _ = roc_curve(y_te, y_prob)
        aucs_roc.append(auc(fpr, tpr))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        aucs_pr.append(average_precision_score(y_te, y_prob))
        precisions.append(np.interp(mean_rec, rec[::-1], prec[::-1]))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_prec = np.mean(precisions, axis=0)

    c = CANONICAL[name]
    all_roc[name] = (mean_fpr, mean_tpr, c["roc"], c["roc_std"], color)
    all_pr[name]  = (mean_rec, mean_prec, c["ap"], color)

    print(f"    {name:<25}  Table 1 ROC: {c['roc']:.4f} ± {c['roc_std']:.4f}  |  AP: {c['ap']:.4f}")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2,
        label='Random classifier (AUC = 0.50)', alpha=0.6)

for name, (fpr, tpr, roc_auc, std, color) in all_roc.items():
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{name}  (AUC = {roc_auc:.4f} ± {std:.4f})')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Top 5 Models\nDengue NS2B-NS3 Protease Inhibitor Classification',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  ✓ figures/roc_curves.png")

random_baseline = y.mean()

fig, ax = plt.subplots(figsize=(7, 6))
ax.axhline(random_baseline, color='k', linewidth=1.2, linestyle='--',
           label=f'Random classifier (AP = {random_baseline:.3f})', alpha=0.6)

for name, (rec, prec, ap, color) in all_pr.items():
    ax.plot(rec, prec, color=color, linewidth=2,
            label=f'{name}  (AP = {ap:.4f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — Top 5 Models\nDengue NS2B-NS3 Protease Inhibitor Classification',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper right')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.grid(alpha=0.25)
ax.annotate(f'Random baseline\n(prevalence = {random_baseline:.1%})',
            xy=(0.5, random_baseline + 0.01), fontsize=8, color='black', alpha=0.7)
plt.tight_layout()
plt.savefig('figures/pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
