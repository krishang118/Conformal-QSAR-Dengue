import os, pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, make_scorer)
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("Scaffold-Based Split Sensitivity Analysis")
print("=" * 60)

df = pd.read_csv('data/chembl_dengue_clean.csv')
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['rdkit']
y = data['y']
smiles = data['smiles'].tolist() if hasattr(data.get('smiles', []), 'tolist') else list(data.get('smiles', df['canonical_smiles']))

print(f"  Compounds : {len(smiles):,}  |  Active: {y.sum():,}  |  Inactive: {(y==0).sum():,}")

print("\nComputing ECFP4 for Butina scaffold clustering...")
fps = []
valid_idx = []
for i, smi in enumerate(smiles):
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(fp)
        valid_idx.append(i)

print(f"  Valid molecules: {len(fps):,}")

print("Running Butina clustering (cutoff=0.35, ~65% similarity threshold)...")

nfps = len(fps)
dists = []
for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - s for s in sims])

clusters = Butina.ClusterData(dists, nfps, 0.35, isDistData=True)
print(f"  Total clusters: {len(clusters):,}")
print(f"  Largest cluster: {max(len(c) for c in clusters):,} compounds")
print(f"  Singleton clusters: {sum(1 for c in clusters if len(c)==1):,}")

print("\nCreating scaffold-based 80/20 train/test split...")
rng = np.random.default_rng(42)
cluster_indices = list(range(len(clusters)))
rng.shuffle(cluster_indices)

train_idx, test_idx = [], []
total = len(valid_idx)
target_test_size = int(0.20 * total)
test_size = 0

for ci in cluster_indices:
    cluster_members = [valid_idx[j] for j in clusters[ci]]
    if test_size < target_test_size:
        test_idx.extend(cluster_members)
        test_size += len(cluster_members)
    else:
        train_idx.extend(cluster_members)

all_assigned = set(train_idx + test_idx)
unassigned = [i for i in valid_idx if i not in all_assigned]
train_idx.extend(unassigned)

train_idx = sorted(set(train_idx))
test_idx  = sorted(set(test_idx))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"  Train: {len(train_idx):,} compounds  ({y_train.sum():,} active)")
print(f"  Test : {len(test_idx):,} compounds  ({y_test.sum():,} active)")
print(f"  Test active rate: {y_test.mean():.1%}")

if y_test.sum() < 5:
    print("  WARNING: Too few actives in test set for reliable evaluation!")
    print("  Trying a more balanced scaffold split (cutoff=0.25)...")

    clusters2 = Butina.ClusterData(dists, nfps, 0.25, isDistData=True)
    cluster_indices2 = list(range(len(clusters2)))
    rng.shuffle(cluster_indices2)
    train_idx2, test_idx2 = [], []
    test_size2 = 0
    for ci in cluster_indices2:
        cluster_members = [valid_idx[j] for j in clusters2[ci]]
        if test_size2 < target_test_size:
            test_idx2.extend(cluster_members)
            test_size2 += len(cluster_members)
        else:
            train_idx2.extend(cluster_members)
    train_idx, test_idx = sorted(set(train_idx2)), sorted(set(test_idx2))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Re-split — Train: {len(train_idx):,} ({y_train.sum():,} active) | Test: {len(test_idx):,} ({y_test.sum():,} active)")

print("\nTraining RF + RDKit2D on scaffold train set...")
rf = RandomForestClassifier(
    n_estimators=200, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_prob = rf.predict_proba(X_test)[:, 1]

scaffold_auc   = roc_auc_score(y_test, y_prob)
scaffold_aupr  = average_precision_score(y_test, y_prob)
scaffold_mcc   = matthews_corrcoef(y_test, (y_prob > 0.5).astype(int))

print(f"  Scaffold split AUC-ROC : {scaffold_auc:.4f}")
print(f"  Scaffold split AUC-PR  : {scaffold_aupr:.4f}")
print(f"  Scaffold split MCC     : {scaffold_mcc:.4f}")

print("\nLoading canonical stratified CV result from Stage 3 benchmark...")
benchmark_df = pd.read_csv('results/results_table_with_mlp.csv')
best_row = benchmark_df[
    (benchmark_df['Model'] == 'Random Forest') &
    (benchmark_df['Fingerprint'] == 'RDKit2D')
].iloc[0]
strat_auc  = float(best_row['AUC_ROC'])
strat_aupr = float(best_row['AUC_PR'])
strat_mcc  = float(best_row['MCC'])
strat_std  = float(best_row['AUC_ROC_std']) if 'AUC_ROC_std' in best_row else 0.0071

print(f"  Canonical stratified CV AUC-ROC : {strat_auc:.4f} (from Stage 3)")
print(f"  Canonical stratified CV AUC-PR  : {strat_aupr:.4f}")
print(f"  Canonical stratified CV MCC     : {strat_mcc:.4f}")

print()
print("=" * 60)
print("SCAFFOLD vs. STRATIFIED COMPARISON")
print("=" * 60)
delta = scaffold_auc - strat_auc
print(f"  Scaffold split AUC-ROC  : {scaffold_auc:.4f}")
print(f"  Stratified CV  AUC-ROC  : {strat_auc:.4f}")
print(f"  Difference              : {delta:+.4f} ({'scaffold is lower, expected' if delta < 0 else 'scaffold is comparable'})")



results_out = {
    'scaffold_auc_roc':      float(round(scaffold_auc, 4)),
    'scaffold_auc_pr':       float(round(scaffold_aupr, 4)),
    'scaffold_mcc':          float(round(scaffold_mcc, 4)),
    'stratified_cv_auc_roc': float(round(strat_auc, 4)),
    'stratified_cv_auc_pr':  float(round(strat_aupr, 4)),
    'stratified_cv_mcc':     float(round(strat_mcc, 4)),
    'delta_auc_roc':         float(round(delta, 4)),
    'n_clusters':            len(clusters),
    'train_size':            len(train_idx),
    'test_size':             len(test_idx),
    'test_actives':          int(y_test.sum()),
}
with open('results/scaffold_split_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print("  ✓ results/scaffold_split_results.json")

print("\nGenerating Figure 8: Scaffold vs Stratified comparison...")
fig, ax = plt.subplots(figsize=(8, 5))

metrics = ['AUC-ROC', 'AUC-PR', 'MCC']
strat_vals    = [strat_auc, strat_aupr, strat_mcc]
scaffold_vals = [scaffold_auc, scaffold_aupr, scaffold_mcc]

x = np.arange(len(metrics))
w = 0.32
bars1 = ax.bar(x - w/2, strat_vals,    w, label='Stratified 5-fold CV', color='#3d7ebf', alpha=0.85)
bars2 = ax.bar(x + w/2, scaffold_vals, w, label='Scaffold split (Butina)', color='#e07b39', alpha=0.85)

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

ax.axhline(0.5, color='grey', linewidth=1, linestyle='--', label='Random (AUC-ROC=0.5)')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_title('Stratified CV vs. Scaffold-Based Split\n'
             f'RF + RDKit2D | ΔAUC-ROC = {delta:+.4f}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/scaffold_split_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
