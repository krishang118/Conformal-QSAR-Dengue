import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("Stage 10 — Top Predicted Actives Table")
print("=" * 60)

df = pd.read_csv('data/chembl_dengue_clean.csv')
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['rdkit']
y = data['y']
smiles_list = data['smiles'] if 'smiles' in data else df['canonical_smiles'].values

print(f"  Dataset: {len(y):,} compounds | {y.sum():,} active | {(y==0).sum():,} inactive")

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)),
    test_size=0.20, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=300, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_prob = rf.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)
print(f"  Test AUC-ROC (80/20 split): {test_auc:.4f}")

test_df = pd.DataFrame({
    'original_idx':    idx_test,
    'predicted_prob':  y_prob,
    'true_label':      y_test,
    'predicted_label': (y_prob > 0.5).astype(int),
})

if 'canonical_smiles' in df.columns:
    test_df['smiles'] = df['canonical_smiles'].values[idx_test]
if 'molecule_chembl_id' in df.columns:
    test_df['chembl_id'] = df['molecule_chembl_id'].values[idx_test]
elif 'chembl_id' in df.columns:
    test_df['chembl_id'] = df['chembl_id'].values[idx_test]
else:
    test_df['chembl_id'] = [f"COMPOUND_{i}" for i in idx_test]

for col in ['standard_value', 'ic50_nm', 'pIC50']:
    if col in df.columns:
        test_df['ic50_nm'] = df[col].values[idx_test]
        break
else:
    if 'standard_value' in df.columns:
        test_df['ic50_nm'] = df['standard_value'].values[idx_test]

predicted_actives = test_df[test_df['predicted_prob'] >= 0.50].copy()
predicted_actives = predicted_actives.sort_values('predicted_prob', ascending=False)

predicted_actives['classification'] = predicted_actives['true_label'].map(
    {1: 'True Positive (TP)', 0: 'False Positive (FP)'}
)

top15 = predicted_actives.head(15).copy()
top15['rank'] = range(1, len(top15) + 1)
top15['confidence_pct'] = (top15['predicted_prob'] * 100).round(1)

print(f"\n  Predicted actives in test set: {len(predicted_actives)}")
print(f"  True Positives (TP): {(predicted_actives['true_label']==1).sum()}")
print(f"  False Positives (FP): {(predicted_actives['true_label']==0).sum()}")
print(f"  Precision among predicted actives: "
      f"{(predicted_actives['true_label']==1).mean():.1%}")

save_cols = ['rank', 'chembl_id', 'confidence_pct', 'true_label', 'classification']
if 'ic50_nm' in top15.columns:
    save_cols.append('ic50_nm')
if 'smiles' in top15.columns:
    save_cols.append('smiles')

top15[save_cols].to_csv('results/top_predicted_actives.csv', index=False)
print(f"\n  ✓ results/top_predicted_actives.csv")

print()
print("  TOP 15 PREDICTED ACTIVES (ranked by RF + RDKit2D confidence)")
print("  " + "-" * 65)
print(f"  {'Rank':<5} {'ChEMBL ID':<20} {'Confidence':>10} {'True Label':>12} {'Result':<20}")
print("  " + "-" * 65)
for _, row in top15.iterrows():
    label = "✅ TP" if row['true_label'] == 1 else "❌ FP"
    print(f"  {int(row['rank']):<5} {row['chembl_id']:<20} "
          f"{row['confidence_pct']:>9.1f}%  {int(row['true_label']):>12}   {label}")

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#2ecc71' if t == 1 else '#e74c3c' for t in top15['true_label']]
bars = ax.barh(range(len(top15)), top15['confidence_pct'],
               color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

for i, (_, row) in enumerate(top15.iterrows()):
    label = "TP" if row['true_label'] == 1 else "FP"
    ax.text(row['confidence_pct'] + 0.5, i, f"{row['confidence_pct']:.1f}%  [{label}]",
            va='center', fontsize=8)

ax.set_yticks(range(len(top15)))
ax.set_yticklabels([f"#{int(r['rank'])} {r['chembl_id']}" for _, r in top15.iterrows()],
                   fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Predicted Probability of Activity (%)', fontsize=11)
ax.set_title('Top 15 Predicted NS2B-NS3 Protease Inhibitors\n'
             'RF + RDKit2D | Green = True Positive | Red = False Positive',
             fontsize=11, fontweight='bold')
ax.axvline(50, color='black', linewidth=1, linestyle='--', alpha=0.5, label='Decision threshold (50%)')
ax.set_xlim(0, 115)
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='True Positive (known active)'),
                   Patch(facecolor='#e74c3c', label='False Positive (known inactive)')]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('figures/top_actives_chart.png', dpi=300, bbox_inches='tight')
plt.close()
