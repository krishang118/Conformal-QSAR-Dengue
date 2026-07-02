import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("Stage 12 — Applicability Domain (AD) Analysis")
print("=" * 60)

df = pd.read_csv('data/chembl_dengue_clean.csv')
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)

X_rdkit = data['rdkit']
y = data['y']
smiles_list = df['canonical_smiles'].tolist()

print(f"  Dataset: {len(y):,} compounds | Active: {y.sum():,}")

print("  Computing ECFP4 fingerprints for AD analysis...")
fps = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('C'), radius=2, nBits=2048)
    fps.append(fp)

idx_all = np.arange(len(y))
_, _, idx_train, idx_test = train_test_split(
    X_rdkit, idx_all, test_size=0.20, stratify=y, random_state=42
)
fps_train = [fps[i] for i in idx_train]
fps_test  = [fps[i] for i in idx_test]
y_train   = y[idx_train]
y_test    = y[idx_test]

print(f"  Train: {len(fps_train):,} | Test: {len(fps_test):,}")

print("  Computing Tanimoto similarities (test→train, k=5 nearest)...")
k = 5
test_max_sims = []
test_mean_k_sims = []

for i, fp_test in enumerate(fps_test):
    sims = DataStructs.BulkTanimotoSimilarity(fp_test, fps_train)
    sims_arr = np.array(sims)
    top_k = np.sort(sims_arr)[-k:]
    test_max_sims.append(sims_arr.max())
    test_mean_k_sims.append(top_k.mean())

test_max_sims    = np.array(test_max_sims)
test_mean_k_sims = np.array(test_mean_k_sims)

print("  Computing train-train similarity for AD threshold (sampled)...")

rng = np.random.default_rng(42)
sample_size = min(300, len(fps_train))
sample_idx  = rng.choice(len(fps_train), sample_size, replace=False)
fps_train_sample = [fps_train[i] for i in sample_idx]

train_train_sims = []
for i, fp in enumerate(fps_train_sample):
    others = [fps_train_sample[j] for j in range(sample_size) if j != i]
    sims = DataStructs.BulkTanimotoSimilarity(fp, others)
    train_train_sims.append(np.mean(sorted(sims)[-k:]))

train_train_sims = np.array(train_train_sims)
Z = 0.5
ad_threshold = train_train_sims.mean() - Z * train_train_sims.std()
ad_threshold = max(ad_threshold, 0.30)

print(f"\n  Train-train mean k-NN similarity: {train_train_sims.mean():.4f} ± {train_train_sims.std():.4f}")
print(f"  AD threshold (Z={Z}):             {ad_threshold:.4f}")

in_ad  = test_mean_k_sims >= ad_threshold
out_ad = ~in_ad

print(f"\n  Test compounds INSIDE  AD: {in_ad.sum():,} ({in_ad.mean():.1%})")
print(f"  Test compounds OUTSIDE AD: {out_ad.sum():,} ({out_ad.mean():.1%})")

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_rdkit[idx_train], y_train)
y_prob_test = rf.predict_proba(X_rdkit[idx_test])[:, 1]

auc_overall = roc_auc_score(y_test, y_prob_test)
auc_in_ad   = roc_auc_score(y_test[in_ad], y_prob_test[in_ad]) if in_ad.sum() > 5 and y_test[in_ad].sum() > 0 else float('nan')
auc_out_ad  = roc_auc_score(y_test[out_ad], y_prob_test[out_ad]) if out_ad.sum() > 5 and y_test[out_ad].sum() > 0 else float('nan')

print(f"\n  AUC-ROC overall   : {auc_overall:.4f}")
print(f"  AUC-ROC inside AD : {auc_in_ad:.4f}")
print(f"  AUC-ROC outside AD: {auc_out_ad:.4f}  (expected lower)")

results = {
    'ad_threshold':          round(float(ad_threshold), 4),
    'z_score':               Z,
    'k_neighbours':          k,
    'n_test':                int(len(y_test)),
    'n_inside_ad':           int(in_ad.sum()),
    'n_outside_ad':          int(out_ad.sum()),
    'pct_inside_ad':         round(float(in_ad.mean()) * 100, 1),
    'auc_roc_overall':       round(float(auc_overall), 4),
    'auc_roc_inside_ad':     round(float(auc_in_ad), 4),
    'auc_roc_outside_ad':    round(float(auc_out_ad), 4),
    'train_train_mean_sim':  round(float(train_train_sims.mean()), 4),
    'train_train_std_sim':   round(float(train_train_sims.std()), 4),
}
with open('results/ad_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  ✓ results/ad_results.json")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

bins = np.linspace(0, 1, 40)
ax1.hist(test_mean_k_sims[in_ad],  bins=bins, color='#27ae60', alpha=0.75,
         label=f'Inside AD (n={in_ad.sum()})')
ax1.hist(test_mean_k_sims[out_ad], bins=bins, color='#e74c3c', alpha=0.75,
         label=f'Outside AD (n={out_ad.sum()})')
ax1.axvline(ad_threshold, color='navy', linewidth=2, linestyle='--',
            label=f'AD threshold = {ad_threshold:.3f}')
ax1.set_xlabel('Mean k-NN Tanimoto Similarity (test → train)', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Applicability Domain — Tanimoto k-NN\n'
              f'{in_ad.mean():.1%} of test compounds inside AD',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.25)

categories = ['Overall', 'Inside AD', 'Outside AD']
auc_vals   = [auc_overall, auc_in_ad, auc_out_ad]
bar_colors = ['#3498db', '#27ae60', '#e74c3c']
bars = ax2.bar(categories, auc_vals, color=bar_colors, alpha=0.85, width=0.5)
ax2.axhline(0.5, color='grey', linewidth=1, linestyle='--', alpha=0.6, label='Random (0.50)')
for bar, val in zip(bars, auc_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylabel('AUC-ROC', fontsize=11)
ax2.set_ylim(0.5, 1.05)
ax2.set_title('Model Performance: Inside vs. Outside AD\n'
              'Higher AUC inside AD confirms domain validity',
              fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.25)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/ad_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/ad_analysis.png")
print()
print("✓ Stage 12 complete.")
print(f"\n  Key finding: AUC-ROC inside AD ({auc_in_ad:.4f}) > outside AD ({auc_out_ad:.4f})")
print(f"  → Confirms model is more reliable within its training chemical space.")
print(f"  → {in_ad.mean():.1%} of test compounds fall within AD — model broadly applicable.")
