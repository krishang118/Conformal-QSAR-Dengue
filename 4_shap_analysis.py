import os, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

os.makedirs('figures',  exist_ok=True)
os.makedirs('results',  exist_ok=True)

print("=" * 60)
print("Loading features and models...")
print("=" * 60)
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)
with open('results/best_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

y       = data['y']
smiles  = data['smiles']
desc_names = data['descriptor_names']

print(f"  Descriptor names available: {len(desc_names)}")
print(f"  ECFP4 shape : {data['ecfp4'].shape}")
print(f"  RDKit shape : {data['rdkit'].shape}")

print("\n" + "=" * 60)
print("Analysis A: SHAP on RDKit 2D descriptors (Random Forest)")
print("=" * 60)

X_rdkit = data['rdkit']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_rdkit, y, test_size=0.2, stratify=y, random_state=42
)

model_rdkit = RandomForestClassifier(
    n_estimators=200, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
model_rdkit.fit(X_train_r, y_train_r)
print(f"  Model     : Random Forest + RDKit2D (n_estimators=200)")
print(f"  Test set  : {len(y_test_r):,} compounds")

explainer_r  = shap.TreeExplainer(model_rdkit)

shap_vals_r  = explainer_r.shap_values(X_test_r)[:, :, 1]

mean_abs_shap = np.abs(shap_vals_r).mean(axis=0)
top20_idx     = np.argsort(mean_abs_shap)[-20:][::-1]
top20_names   = [desc_names[i] for i in top20_idx]
top20_shap    = mean_abs_shap[top20_idx]

shap_importance_df = pd.DataFrame({
    'descriptor': desc_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)
shap_importance_df.to_csv('results/shap_rdkit_importance.csv', index=False)
print(f"  Top 5 descriptors:\n{shap_importance_df.head().to_string(index=False)}")

print("\n  Generating Figure 2: SHAP beeswarm (RDKit)...")

fig, ax = plt.subplots(figsize=(11, 8))

shap_vals_top20 = shap_vals_r[:, top20_idx]
X_test_top20    = X_test_r[:, top20_idx]

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

cmap = plt.cm.RdBu_r
y_positions = np.arange(20)

for i, (feat_idx, feat_name) in enumerate(zip(range(20), top20_names)):
    sv    = shap_vals_top20[:, i]
    fv    = X_test_top20[:, i]
    fv_n  = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-9)

    jitter = np.random.default_rng(i).uniform(-0.3, 0.3, len(sv))

    ax.scatter(sv, y_positions[i] + jitter,
               c=fv_n, cmap=cmap, vmin=0, vmax=1,
               s=8, alpha=0.5, linewidths=0)

ax.set_yticks(y_positions)
ax.set_yticklabels(top20_names, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax.set_xlabel('SHAP value (impact on model output)', fontsize=10)
ax.set_title('SHAP Feature Importance — RDKit 2D Physicochemical Descriptors\n'
             'Random Forest model on dengue NS2B-NS3 bioactivity (IC50 ≤ 10,000 nM = 10 µM)',
             fontsize=11, fontweight='bold')

sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
cbar.set_label('Feature value\n(low → high)', fontsize=8)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Low', 'Mid', 'High'])

ax.grid(axis='x', alpha=0.3, linewidth=0.5)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/shap_beeswarm_rdkit.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/shap_beeswarm_rdkit.png")

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#e05c5c' if v > 0 else '#5c8ee0' for v in top20_shap]
ax.barh(top20_names[::-1], top20_shap[::-1], color='#7b5ea7', edgecolor='white', height=0.6)
ax.set_xlabel('Mean |SHAP| value', fontsize=10)
ax.set_title('Top 20 RDKit Descriptors by Mean |SHAP|\n(Random Forest, dengue NS2B-NS3 bioactivity, IC50 ≤ 10,000 nM)',
             fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/shap_bar_rdkit.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/shap_bar_rdkit.png")

print("\n" + "=" * 60)
print("Analysis B: SHAP on ECFP4 fingerprint (Random Forest)")
print("=" * 60)

X_ecfp4 = data['ecfp4']
X_train_e, X_test_e, y_train_e, y_test_e, smi_train, smi_test = train_test_split(
    X_ecfp4, y, smiles, test_size=0.2, stratify=y, random_state=42
)

model_ecfp4 = RandomForestClassifier(
    n_estimators=200, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
model_ecfp4.fit(X_train_e, y_train_e)
print(f"  Model     : Random Forest + ECFP4 (n_estimators=200)")

sample_size = min(500, len(X_test_e))
rng_idx     = np.random.default_rng(42).choice(len(X_test_e), sample_size, replace=False)
X_shap_e    = X_test_e[rng_idx]
y_shap_e    = y_test_e[rng_idx]

explainer_e  = shap.TreeExplainer(model_ecfp4)

shap_vals_e  = explainer_e.shap_values(X_shap_e)[:, :, 1]

mean_abs_e   = np.abs(shap_vals_e).mean(axis=0)
top30_bits   = np.argsort(mean_abs_e)[-30:][::-1]
top30_shap_e = mean_abs_e[top30_bits]

bit_importance_df = pd.DataFrame({
    'bit_index':      range(2048),
    'mean_abs_shap':  mean_abs_e
}).sort_values('mean_abs_shap', ascending=False)
bit_importance_df.to_csv('results/shap_ecfp4_bit_importance.csv', index=False)
print(f"  Top 5 ECFP4 bits:\n{bit_importance_df.head().to_string(index=False)}")

print("\n  Generating Figure 3: SHAP bar chart (ECFP4)...")

fig, ax = plt.subplots(figsize=(9, 6))
bit_labels = [f'Bit {b}' for b in top30_bits]
ax.barh(bit_labels[::-1], top30_shap_e[::-1],
        color='#2d8a6e', edgecolor='white', height=0.6)
ax.set_xlabel('Mean |SHAP| value', fontsize=10)
ax.set_title('Top 30 ECFP4 Bit Positions by Mean |SHAP|\n'
             '(Random Forest, dengue NS2B-NS3 bioactivity, IC50 ≤ 10,000 nM, n=500 test compounds)',
             fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/shap_bar_ecfp4.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/shap_bar_ecfp4.png")

print("\n  Analyzing top 5 confidently predicted actives...")
proba_e = model_ecfp4.predict_proba(X_test_e)[:, 1]
top5_idx = np.argsort(proba_e)[-5:][::-1]

compound_analysis = []
for rank, idx in enumerate(top5_idx, 1):
    sv_full = explainer_e.shap_values(X_test_e[idx:idx+1])[:, :, 1][0]
    top_bits = np.argsort(np.abs(sv_full))[-10:][::-1]
    compound_analysis.append({
        'rank':        rank,
        'smiles':      smi_test[idx],
        'true_label':  int(y_test_e[idx]),
        'pred_prob':   round(float(proba_e[idx]), 4),
        'top_10_bits': top_bits.tolist(),
        'top_10_shap': [round(float(sv_full[b]), 4) for b in top_bits],
    })
    print(f"  Rank {rank}: prob={proba_e[idx]:.3f}  true={'active' if y_test_e[idx]==1 else 'inactive'}")
    print(f"    SMILES: {smi_test[idx][:60]}...")
    print(f"    Key bits: {top_bits[:5].tolist()}")

with open('results/top5_active_shap.json', 'w') as f:
    json.dump(compound_analysis, f, indent=2)
print("  ✓ results/top5_active_shap.json")

shap_summary = {
    'top_rdkit_descriptor':      top20_names[0],
    'top_rdkit_mean_abs_shap':   float(round(top20_shap[0], 4)),
    'top5_rdkit_descriptors':    top20_names[:5],
    'top_ecfp4_bit':             int(top30_bits[0]),
    'top_ecfp4_mean_abs_shap':   float(round(top30_shap_e[0], 4)),
    'n_test_compounds_rdkit':    int(len(y_test_r)),
    'n_test_compounds_ecfp4':    int(sample_size),
}
with open('results/shap_summary.json', 'w') as f:
    json.dump(shap_summary, f, indent=2)

print(f"\n  Top RDKit descriptor: {top20_names[0]} (mean|SHAP|={top20_shap[0]:.4f})")
print(f"  Top ECFP4 bit      : Bit {top30_bits[0]} (mean|SHAP|={top30_shap_e[0]:.4f})")
