import os, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results',  exist_ok=True)
os.makedirs('figures',  exist_ok=True)

print("=" * 60)
print("Loading feature matrices...")
print("=" * 60)
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)

y = data['y']
print(f"  Compounds : {len(y):,}  |  Active: {y.sum():,}  |  Inactive: {(1-y).sum():,}")

print("\n" + "=" * 60)
print("Model definitions")
print("=" * 60)

MODELS = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=(1-y).sum() / y.sum(),
        eval_metric='logloss', random_state=42, n_jobs=-1,
        verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        class_weight='balanced', random_state=42, n_jobs=-1,
        verbose=-1
    ),
    'SVM': SVC(
        kernel='rbf', probability=True,
        class_weight='balanced', C=10, gamma='scale', random_state=42
    ),
}

FINGERPRINTS = ['ecfp4', 'maccs', 'rdkit', 'atompair']
FP_LABELS    = {'ecfp4': 'ECFP4', 'maccs': 'MACCS', 'rdkit': 'RDKit2D', 'atompair': 'AtomPair'}

for name, model in MODELS.items():
    print(f"  {name}")

print("\n" + "=" * 60)
print("Cross-validation: 5-fold stratified")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scorers = {
    'AUC_ROC': make_scorer(roc_auc_score,          response_method='predict_proba'),
    'AUC_PR':  make_scorer(average_precision_score, response_method='predict_proba'),
    'MCC':     make_scorer(matthews_corrcoef),
}

print("\n" + "=" * 60)
print("Training 16 combinations (4 fingerprints × 4 models)...")
print("=" * 60)

results = []
for fp_name in FINGERPRINTS:
    X = data[fp_name]
    print(f"\n  [{FP_LABELS[fp_name]}]  shape={X.shape}")
    for model_name, model in MODELS.items():
        print(f"    ├─ {model_name}...", end=' ', flush=True)
        cv_scores = cross_validate(
            model, X, y, cv=cv,
            scoring=scorers,
            n_jobs=-1,
            error_score='raise'
        )
        auc_roc_mean = cv_scores['test_AUC_ROC'].mean()
        auc_roc_std  = cv_scores['test_AUC_ROC'].std()
        auc_pr_mean  = cv_scores['test_AUC_PR'].mean()
        mcc_mean     = cv_scores['test_MCC'].mean()

        print(f"AUC-ROC={auc_roc_mean:.3f}±{auc_roc_std:.3f}  AUC-PR={auc_pr_mean:.3f}  MCC={mcc_mean:.3f}")

        results.append({
            'Fingerprint': FP_LABELS[fp_name],
            'Model':       model_name,
            'AUC_ROC':     round(auc_roc_mean, 4),
            'AUC_ROC_std': round(auc_roc_std,  4),
            'AUC_PR':      round(auc_pr_mean,  4),
            'MCC':         round(mcc_mean,     4),
            '_fp_key':     fp_name,
        })

results_df = pd.DataFrame(results).sort_values('AUC_ROC', ascending=False).reset_index(drop=True)
results_df['Rank'] = results_df.index + 1

print("\n" + "=" * 60)
print("MAIN RESULTS TABLE (Table 2)")
print("=" * 60)
display_cols = ['Rank', 'Fingerprint', 'Model', 'AUC_ROC', 'AUC_ROC_std', 'AUC_PR', 'MCC']
print(results_df[display_cols].to_string(index=False))

results_df.to_csv('results/results_table.csv', index=False)
print(f"\n  ✓ Saved to results/results_table.csv")

best = results_df.iloc[0]
print(f"\n  🏆 BEST: {best['Model']} + {best['Fingerprint']}")
print(f"     AUC-ROC = {best['AUC_ROC']:.4f} ± {best['AUC_ROC_std']:.4f}")
print(f"     AUC-PR  = {best['AUC_PR']:.4f}")
print(f"     MCC     = {best['MCC']:.4f}")

print("\n" + "=" * 60)
print("Training best model on full dataset (for SHAP + conformal)...")
print("=" * 60)
best_fp_key    = best['_fp_key']
best_model_name = best['Model']
X_best = data[best_fp_key]
best_model = MODELS[best_model_name]
best_model.fit(X_best, y)
print(f"  ✓ {best_model_name} on {best['Fingerprint']} fitted on full data")

xgb_ecfp4 = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    scale_pos_weight=(1-y).sum() / y.sum(),
    eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
)
xgb_ecfp4.fit(data['ecfp4'], y)

xgb_rdkit = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    scale_pos_weight=(1-y).sum() / y.sum(),
    eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
)
xgb_rdkit.fit(data['rdkit'], y)

model_bundle = {
    'best_model':      best_model,
    'best_fp_key':     best_fp_key,
    'best_model_name': best_model_name,
    'xgb_ecfp4':       xgb_ecfp4,
    'xgb_rdkit':       xgb_rdkit,
    'results_df':      results_df,
}
with open('results/best_model.pkl', 'wb') as f:
    pickle.dump(model_bundle, f, protocol=4)
print(f"  ✓ Saved to results/best_model.pkl")

print("\n" + "=" * 60)
print("Generating benchmark heatmap...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('16-Combination Benchmark: 4 Fingerprints × 4 Models',
             fontsize=13, fontweight='bold', y=1.02)

metrics_to_plot = [
    ('AUC_ROC', 'AUC-ROC', 'Blues'),
    ('AUC_PR',  'AUC-PR',  'Greens'),
    ('MCC',     'MCC',     'Purples'),
]

model_order = ['Random Forest', 'XGBoost', 'LightGBM', 'SVM']
fp_order    = ['ECFP4', 'MACCS', 'RDKit2D', 'AtomPair']

for ax, (col, label, cmap) in zip(axes, metrics_to_plot):
    pivot = results_df.pivot(index='Model', columns='Fingerprint', values=col)
    pivot = pivot.reindex(index=model_order, columns=fp_order)
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt='.3f',
        cmap=cmap, vmin=pivot.values.min()-0.02, vmax=min(1.0, pivot.values.max()+0.01),
        linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8}
    )
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Fingerprint', fontsize=9)
    ax.set_ylabel('Model', fontsize=9)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figures/benchmark_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/benchmark_heatmap.png")

fig, ax = plt.subplots(figsize=(12, 5))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_df)))[::-1]
bars = ax.barh(
    [f"{r['Model']} + {r['Fingerprint']}" for _, r in results_df.iterrows()],
    results_df['AUC_ROC'],
    xerr=results_df['AUC_ROC_std'],
    color=colors, edgecolor='white', height=0.6,
    error_kw=dict(ecolor='#555', capsize=3, linewidth=1)
)
ax.set_xlabel('AUC-ROC (5-fold CV mean ± std)', fontsize=10)
ax.set_title('All 16 Combinations Ranked by AUC-ROC', fontsize=12, fontweight='bold')
ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Random baseline')
ax.set_xlim(0.45, 1.0)
ax.legend(fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/benchmark_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/benchmark_barplot.png")

with open('results/best_combo.json', 'w') as f:
    json.dump({
        'best_model_name': best_model_name,
        'best_fp_key':     best_fp_key,
        'best_fp_label':   best['Fingerprint'],
        'auc_roc':         best['AUC_ROC'],
        'auc_roc_std':     best['AUC_ROC_std'],
        'auc_pr':          best['AUC_PR'],
        'mcc':             best['MCC'],
    }, f, indent=2)
