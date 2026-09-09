"""
14_scaffold_ef_bedroc.py
========================
Scaffold-split Enrichment Factor (EF) and BEDROC analysis.

Motivation (Reviewer 1 Minor #2):
  The scaffold-split section reported only overall AUC-ROC (0.843).
  Under 6.9% class imbalance, AUC-ROC is overly optimistic.
  The reviewer requests standard early virtual screening metrics:
    - Enrichment Factor (EF) at 1%, 5%, 10% of ranked library
    - BEDROC (Boltzmann-Enhanced Discrimination of ROC) with alpha=20
      (standard for VS benchmarking; weights early recovery more heavily)

This script re-runs the scaffold split (same Butina clustering as
8_scaffold_split.py) and adds EF + BEDROC to the output, alongside
the stratified CV comparison already in the paper.

Output:
  results/scaffold_ef_bedroc.json
  figures/scaffold_ef_bedroc.png
"""

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
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Helper: Enrichment Factor ─────────────────────────────────────────────
def enrichment_factor(y_true, y_score, fraction):
    """
    EF at top `fraction` of the ranked library.
    EF = (actives in top fraction%) / (actives expected by random)
       = (actives in top-k / total actives) / (k / n)
    Maximum achievable EF = 1 / fraction  (all actives in top cut)
    """
    n     = len(y_true)
    k     = max(1, int(np.ceil(fraction * n)))
    top_k = np.argsort(y_score)[::-1][:k]
    hits  = y_true[top_k].sum()
    total_actives = y_true.sum()
    if total_actives == 0:
        return float('nan')
    ef = (hits / total_actives) / (k / n)
    return float(ef)

# ── Helper: BEDROC (alpha=20.0, standard for early enrichment in VS) ──────
def bedroc(y_true, y_score, alpha=20.0):
    """
    Boltzmann-Enhanced Discrimination of ROC (Truchon & Bayly, JCIM 2007).
    alpha=20: gives high weight to top ~8% of the ranked list.
    Formula from Truchon & Bayly 2007, eq. 14.
    """
    n       = len(y_true)
    ra      = y_true.sum() / n          # fraction of actives
    order   = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]

    # Boltzmann-weighted sum for ranked actives
    ranks   = np.where(y_sorted == 1)[0] + 1    # 1-indexed ranks
    if len(ranks) == 0:
        return 0.0

    risum  = np.sum(np.exp(-alpha * ranks / n))
    Ra     = (1 / ra) * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)

    # Random BEDROC (expected under random ranking)
    random_sum = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)

    bedroc_val = (risum * ra / n - random_sum) / (Ra - random_sum) * (
        Ra / (1 - np.exp(-alpha / n)) * (n / (n + 1)) + ra * (n - 1) / (n + 1)
    )
    # Simplified standard form
    # Using the corrected Truchon & Bayly formula directly
    n_a = int(y_true.sum())
    bedroc_numerator   = sum(np.exp(-alpha * r / n) for r in ranks)
    bedroc_denominator = (n_a / n) * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    bedroc_random      = (n_a * (1 - np.exp(-alpha))) / (n * (np.exp(alpha / n) - 1))
    bedroc_max         = (1 - np.exp(-alpha * n_a / n)) / (1 - np.exp(-alpha / n))

    if abs(bedroc_max - bedroc_random) < 1e-10:
        return float('nan')

    score = (bedroc_numerator - bedroc_random) / (bedroc_max - bedroc_random)
    return float(np.clip(score, 0.0, 1.0))

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 65)
print("Scaffold-Split: Enrichment Factor & BEDROC Analysis")
print("=" * 65)

df = pd.read_csv('data/chembl_dengue_clean.csv')
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)

X      = data['rdkit']
y      = data['y']
smiles = (data['smiles'].tolist() if hasattr(data.get('smiles', []), 'tolist')
          else list(data.get('smiles', df['canonical_smiles'])))

print(f"  Compounds: {len(y):,}  |  Active: {y.sum():,}  |  Inactive: {(y==0).sum():,}")

# ── Butina clustering (identical to 8_scaffold_split.py) ─────────────────
print("\nComputing ECFP4 for Butina clustering...")
fps, valid_idx = [], []
for i, smi in enumerate(smiles):
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        valid_idx.append(i)

nfps  = len(fps)
dists = []
for i in range(1, nfps):
    sims  = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - s for s in sims])

clusters = Butina.ClusterData(dists, nfps, 0.35, isDistData=True)
print(f"  Clusters: {len(clusters):,}")

# ── Scaffold split ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
cluster_indices = list(range(len(clusters)))
rng.shuffle(cluster_indices)

total          = len(valid_idx)
target_test    = int(0.20 * total)
train_idx, test_idx = [], []
test_size = 0
for ci in cluster_indices:
    members = [valid_idx[j] for j in clusters[ci]]
    if test_size < target_test:
        test_idx.extend(members); test_size += len(members)
    else:
        train_idx.extend(members)

all_assigned = set(train_idx + test_idx)
train_idx.extend([i for i in valid_idx if i not in all_assigned])
train_idx = sorted(set(train_idx)); test_idx = sorted(set(test_idx))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"  Train: {len(train_idx):,} ({y_train.sum():,} active) | Test: {len(test_idx):,} ({y_test.sum():,} active)")

# ── Train RF ──────────────────────────────────────────────────────────────
print("\nTraining RF + RDKit2D on scaffold train set...")
rf = RandomForestClassifier(
    n_estimators=300, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_prob = rf.predict_proba(X_test)[:, 1]

scaffold_auc  = roc_auc_score(y_test, y_prob)
scaffold_aupr = average_precision_score(y_test, y_prob)
scaffold_mcc  = matthews_corrcoef(y_test, (y_prob > 0.5).astype(int))
print(f"  AUC-ROC : {scaffold_auc:.4f}")
print(f"  AUC-PR  : {scaffold_aupr:.4f}")
print(f"  MCC     : {scaffold_mcc:.4f}")

# ── EF and BEDROC ─────────────────────────────────────────────────────────
print("\nComputing Enrichment Factor and BEDROC...")
ef_fractions = [0.01, 0.05, 0.10]
ef_results = {}
for frac in ef_fractions:
    ef_val = enrichment_factor(y_test, y_prob, frac)
    ef_max = 1.0 / frac   # theoretical maximum
    ef_results[f'EF{int(frac*100)}'] = {
        'ef': round(ef_val, 3),
        'ef_max': ef_max,
        'ef_random': 1.0,
        'ef_pct_max': round(ef_val / ef_max * 100, 1),
    }
    print(f"  EF{int(frac*100):2d}% : {ef_val:.3f}  (max={ef_max:.1f}, random=1.0, "
          f"{ef_val/ef_max*100:.1f}% of max)")

bedroc_val = bedroc(y_test, y_prob, alpha=20.0)
print(f"  BEDROC (α=20): {bedroc_val:.4f}  (random≈0.0, perfect=1.0)")

# Random baseline EF (for context)
print("\nRandom baseline (for reference):")
rng2  = np.random.default_rng(0)
n_rand_trials = 100
rand_efs = {f: [] for f in ef_fractions}
rand_bedrocs = []
for _ in range(n_rand_trials):
    y_rand = rng2.random(len(y_test))
    for frac in ef_fractions:
        rand_efs[frac].append(enrichment_factor(y_test, y_rand, frac))
    rand_bedrocs.append(bedroc(y_test, y_rand, alpha=20.0))
for frac in ef_fractions:
    print(f"  Random EF{int(frac*100):2d}% : {np.mean(rand_efs[frac]):.3f} ± {np.std(rand_efs[frac]):.3f}")
print(f"  Random BEDROC : {np.mean(rand_bedrocs):.4f} ± {np.std(rand_bedrocs):.4f}")

# Load stratified CV results for comparison
benchmark_df = pd.read_csv('results/results_table_with_mlp.csv')
best_row = benchmark_df[
    (benchmark_df['Model'] == 'Random Forest') &
    (benchmark_df['Fingerprint'] == 'RDKit2D')
].iloc[0]
strat_auc = float(best_row['AUC_ROC'])

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  Stratified CV AUC-ROC : {strat_auc:.4f}")
print(f"  Scaffold    AUC-ROC   : {scaffold_auc:.4f}  (Δ = {scaffold_auc - strat_auc:+.4f})")
print(f"  Scaffold    AUC-PR    : {scaffold_aupr:.4f}")
print(f"  Scaffold    MCC       : {scaffold_mcc:.4f}")
for k, v in ef_results.items():
    print(f"  {k}          : {v['ef']:.3f}  ({v['ef_pct_max']:.1f}% of theoretical max {v['ef_max']:.1f})")
print(f"  BEDROC (α=20)         : {bedroc_val:.4f}")

# Save
out = {
    'scaffold_auc_roc':   round(scaffold_auc, 4),
    'scaffold_auc_pr':    round(scaffold_aupr, 4),
    'scaffold_mcc':       round(scaffold_mcc, 4),
    'stratified_auc_roc': round(strat_auc, 4),
    'ef_results':         ef_results,
    'bedroc_alpha20':     round(bedroc_val, 4),
    'random_bedroc_mean': round(float(np.mean(rand_bedrocs)), 4),
    'n_test':             len(test_idx),
    'n_test_active':      int(y_test.sum()),
}
with open('results/scaffold_ef_bedroc.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\n  ✓ results/scaffold_ef_bedroc.json")

# ── Figure ────────────────────────────────────────────────────────────────
print("\nGenerating figure: scaffold_ef_bedroc.png ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Scaffold-Split Virtual Screening Performance\n'
             'RF + RDKit2D — Butina clustering (cutoff = 0.35)',
             fontsize=11, fontweight='bold')

# Panel A: EF bar chart
ax = axes[0]
ef_labels = [f'EF{int(f*100)}%' for f in ef_fractions]
ef_vals   = [ef_results[f'EF{int(f*100)}']['ef'] for f in ef_fractions]
ef_maxs   = [ef_results[f'EF{int(f*100)}']['ef_max'] for f in ef_fractions]

x = np.arange(len(ef_labels))
bars_model  = ax.bar(x - 0.2, ef_vals, 0.38, label='RF + RDKit2D', color='#3d7ebf', alpha=0.85)
bars_random = ax.bar(x + 0.2, [1.0]*3,  0.38, label='Random baseline', color='#bbb', alpha=0.7)
for i, (bar, mx) in enumerate(zip(bars_model, ef_maxs)):
    ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], [mx, mx],
            'r--', linewidth=1.5, label='Theoretical max' if i == 0 else '')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}', ha='center', fontsize=9, fontweight='bold', color='#3d7ebf')

ax.set_xticks(x); ax.set_xticklabels(ef_labels, fontsize=10)
ax.set_ylabel('Enrichment Factor', fontsize=10)
ax.set_title('(A)  Enrichment Factor (EF)\nScaffold-split test set', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(ef_maxs) * 1.15)

# Panel B: AUC-ROC + BEDROC summary
ax2 = axes[1]
metrics  = ['AUC-ROC\n(stratified CV)', 'AUC-ROC\n(scaffold split)', 'BEDROC\n(α=20, scaffold)']
vals     = [strat_auc, scaffold_auc, bedroc_val]
colors2  = ['#2ecc71', '#3d7ebf', '#e05c5c']
bars2    = ax2.bar(metrics, vals, color=colors2, alpha=0.85, width=0.5, edgecolor='white')
for bar, val in zip(bars2, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax2.axhline(0.5, color='grey', linewidth=1, linestyle='--', label='Random AUC=0.5', alpha=0.6)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel('Score', fontsize=10)
ax2.set_title('(B)  Overall Metrics Summary\nStratified vs. Scaffold split', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/scaffold_ef_bedroc.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/scaffold_ef_bedroc.png")
print()
print("✓ Scaffold EF + BEDROC analysis complete.")
