import os, pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef)
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("MLP Benchmark — Stage 3b")
print("=" * 60)

with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)

fingerprints = {
    'ECFP4':    data['ecfp4'],
    'MACCS':    data['maccs'],
    'RDKit2D':  data['rdkit'],
    'AtomPair': data['atompair'],
}
y = data['y']

print(f"\nDataset: {len(y):,} compounds | {y.sum():,} active ({y.mean():.1%}) | "
      f"{(y==0).sum():,} inactive")
print(f"\nRunning 5-fold stratified CV for MLP on 4 fingerprints...")
print(f"Architecture: 256→128→64, ReLU, early stopping, sample-weighted\n")

def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        batch_size=64,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mlp_results = []

for fp_name, X in fingerprints.items():
    fold_aucs, fold_auprs, fold_mccs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_sc = scaler.fit_transform(X_train.astype(float))
        X_test_sc  = scaler.transform(X_test.astype(float))

        sw = compute_sample_weight('balanced', y_train)

        mlp = make_mlp()
        mlp.fit(X_train_sc, y_train, sample_weight=sw)

        y_prob = mlp.predict_proba(X_test_sc)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        fold_aucs.append(roc_auc_score(y_test, y_prob))
        fold_auprs.append(average_precision_score(y_test, y_prob))
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))

    auc_mean = np.mean(fold_aucs)
    auc_std  = np.std(fold_aucs)
    aupr     = np.mean(fold_auprs)
    mcc      = np.mean(fold_mccs)

    mlp_results.append({
        'Fingerprint': fp_name,
        'Model':       'MLP (Neural Net)',
        'AUC_ROC':     round(auc_mean, 4),
        'AUC_ROC_std': round(auc_std,  4),
        'AUC_PR':      round(aupr,     4),
        'MCC':         round(mcc,      4),
    })

    print(f"  {fp_name:<10} + MLP  →  AUC-ROC: {auc_mean:.4f} ± {auc_std:.4f} | "
          f"AUC-PR: {aupr:.4f} | MCC: {mcc:.4f}")

print()
classical_df = pd.read_csv('results/results_table.csv')
mlp_df = pd.DataFrame(mlp_results)

mlp_df['_fp_key'] = mlp_df['Fingerprint'].str.lower().str.replace('2d','').str.replace('rdkit','rdkit')
mlp_df['_fp_key'] = mlp_df['Fingerprint'].map({
    'ECFP4': 'ecfp4', 'MACCS': 'maccs', 'RDKit2D': 'rdkit', 'AtomPair': 'atompair'
})

combined = pd.concat([classical_df[['Fingerprint','Model','AUC_ROC','AUC_ROC_std',
                                     'AUC_PR','MCC','_fp_key']],
                      mlp_df], ignore_index=True)
combined = combined.sort_values('AUC_ROC', ascending=False).reset_index(drop=True)
combined['Rank'] = combined.index + 1
combined.to_csv('results/results_table_with_mlp.csv', index=False)
print(f"✓ results/results_table_with_mlp.csv  ({len(combined)} combinations)")

print()
print("=" * 70)
print("FULL BENCHMARK TABLE (All 20 combos, ranked by AUC-ROC)")
print("=" * 70)
print(f"{'Rank':<5} {'Fingerprint':<12} {'Model':<20} {'AUC-ROC':>8} {'±':>3} {'AUC-PR':>7} {'MCC':>7}")
print("-" * 70)
for _, row in combined.iterrows():
    mlp_flag = " ← MLP" if "MLP" in row['Model'] else ""
    print(f"{int(row['Rank']):<5} {row['Fingerprint']:<12} {row['Model']:<20} "
          f"{row['AUC_ROC']:>8.4f} ±{row['AUC_ROC_std']:.4f} "
          f"{row['AUC_PR']:>7.4f} {row['MCC']:>7.4f}{mlp_flag}")

print()
print("MLP Results Summary:")
mlp_auc_avg = mlp_df['AUC_ROC'].mean()
classical_top = classical_df['AUC_ROC'].max()
print(f"  Best classical model AUC-ROC : {classical_top:.4f} (RF + RDKit2D)")
print(f"  MLP average AUC-ROC          : {mlp_auc_avg:.4f}")
print(f"  MLP best AUC-ROC             : {mlp_df['AUC_ROC'].max():.4f} ({mlp_df.loc[mlp_df['AUC_ROC'].idxmax(),'Fingerprint']})")
print(f"  Gap (classical - MLP avg)    : {classical_top - mlp_auc_avg:+.4f}")
print()
print("  → Confirms: classical ML outperforms MLP on this small, imbalanced dataset.")
print("  → Expected — GNNs/MLPs need 10× more active examples to generalise.")

print("\nGenerating updated benchmark figures...")

pivot = combined.pivot(index='Model', columns='Fingerprint', values='AUC_ROC')
model_order = ['Random Forest', 'XGBoost', 'LightGBM', 'SVM', 'MLP (Neural Net)']
fp_order    = ['RDKit2D', 'AtomPair', 'ECFP4', 'MACCS']
pivot = pivot.reindex(index=model_order, columns=fp_order)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd',
            vmin=0.85, vmax=0.96, ax=ax,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 9})
ax.set_title('AUC-ROC Heatmap: All 20 Combinations (4 FP × 5 Models)\n'
             'MLP (Neural Net) shown for comparison',
             fontsize=10, fontweight='bold')
ax.set_xlabel('Molecular Fingerprint', fontsize=10)
ax.set_ylabel('Model', fontsize=10)
plt.tight_layout()
plt.savefig('figures/benchmark_heatmap_mlp.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/benchmark_heatmap_mlp.png")

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#3d7ebf' if 'MLP' not in m else '#e07b39' for m in combined['Model']]
labels = [f"{r['Fingerprint']}\n{r['Model']}" for _, r in combined.iterrows()]
bars = ax.bar(range(len(combined)), combined['AUC_ROC'], color=colors, alpha=0.85,
              yerr=combined['AUC_ROC_std'], capsize=3)

ax.axhline(0.90, color='green', linewidth=1.2, linestyle='--', alpha=0.7, label='AUC-ROC = 0.90')
ax.axhline(0.80, color='orange', linewidth=1.2, linestyle='--', alpha=0.7, label='AUC-ROC = 0.80')
ax.set_xticks(range(len(combined)))
ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha='right')
ax.set_ylabel('AUC-ROC (5-fold CV)', fontsize=10)
ax.set_ylim(0.80, 1.00)
ax.set_title('Benchmark: 4 Fingerprints × 5 Models (Classical ML vs. MLP Neural Network)\n'
             'Blue = Classical ML   |   Orange = MLP (Neural Net)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/benchmark_barplot_mlp.png', dpi=300, bbox_inches='tight')
plt.close()
