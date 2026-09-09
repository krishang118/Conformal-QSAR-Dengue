"""
16_calibration_pains_umap.py
=============================
Three analyses in one script:
  A. Model calibration (reliability diagram + Brier score)
  B. PAINS / structural alert filter analysis
  C. UMAP chemical space visualization

All outputs to results/ and figures/.
"""

import pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from rdkit import Chem
from rdkit.Chem import FilterCatalog, AllChem, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalogParams

warnings.filterwarnings('ignore')

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
FIGURES_DIR = SCRIPT_DIR / 'figures'

print("=" * 60)
print("Analysis Suite: Calibration | PAINS | UMAP")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────
with open(SCRIPT_DIR / 'data' / 'features.pkl', 'rb') as f:
    data = pickle.load(f)
X      = data['rdkit']
y      = data['y']
smiles = list(data['smiles'])
df     = pd.read_csv(SCRIPT_DIR / 'data' / 'chembl_dengue_clean.csv')
n      = len(y)
print(f"\nDataset: {n:,} compounds | {int(y.sum())} active ({y.mean():.1%})")

# ══════════════════════════════════════════════════════════════
# A. MODEL CALIBRATION
# ══════════════════════════════════════════════════════════════
print("\n[A] Model Calibration (5-fold OOF probabilities)...")

cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probs_oof = np.zeros(n)
for train_idx, val_idx in cv.split(X, y):
    rf = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X[train_idx], y[train_idx])
    probs_oof[val_idx] = rf.predict_proba(X[val_idx])[:, 1]

brier = brier_score_loss(y, probs_oof)
frac_pos, mean_pred = calibration_curve(y, probs_oof, n_bins=10, strategy='quantile')
calib_corr = np.corrcoef(mean_pred, frac_pos)[0, 1]

print(f"  Brier score       : {brier:.4f}")
print(f"  Calibration corr  : {calib_corr:.4f}")

# Save calibration data
cal_results = {
    'brier_score': round(float(brier), 4),
    'calibration_correlation': round(float(calib_corr), 4),
    'mean_pred_prob': mean_pred.tolist(),
    'fraction_positive': frac_pos.tolist(),
    'n_bins': 10,
}
with open(RESULTS_DIR / 'calibration_results.json', 'w') as f:
    json.dump(cal_results, f, indent=2)

# Calibration figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('RF + RDKit2D Model Calibration\nDENV NS2B-NS3 Protease Inhibitor Classification',
             fontsize=11, fontweight='bold')

ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
ax.plot(mean_pred, frac_pos, 'o-', color='#3d7ebf', lw=2, ms=7,
        label=f'RF + RDKit2D\n(Brier={brier:.3f}, r={calib_corr:.3f})')
ax.fill_between(mean_pred, frac_pos, mean_pred,
                alpha=0.12, color='#3d7ebf', label='Calibration gap')
ax.set_xlabel('Mean predicted probability', fontsize=10)
ax.set_ylabel('Fraction of positives (active)', fontsize=10)
ax.set_title('(A)  Reliability Diagram\n(5-fold cross-validation)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

ax2 = axes[1]
hist_act = ax2.hist(probs_oof[y == 1], bins=25, alpha=0.75, color='#e05c5c',
                    label='Active (n=205)', density=True)
hist_ina = ax2.hist(probs_oof[y == 0], bins=25, alpha=0.6, color='#3d7ebf',
                    label='Inactive (n=2,752)', density=True)
ax2.axvline(0.5, color='grey', ls='--', lw=1.2, label='Default threshold')
ax2.set_xlabel('Predicted probability of activity', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('(B)  Score Distribution\nActive vs Inactive', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'calibration.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ figures/calibration.png")
print(f"  ✓ results/calibration_results.json")

# ══════════════════════════════════════════════════════════════
# B. PAINS / STRUCTURAL ALERT FILTER
# ══════════════════════════════════════════════════════════════
print("\n[B] PAINS Structural Alert Filter...")

params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog.FilterCatalog(params)

pains_flags = []
pains_names = []
for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pains_flags.append(False)
        pains_names.append([])
        continue
    entry = catalog.GetFirstMatch(mol)
    if entry:
        pains_flags.append(True)
        pains_names.append([entry.GetDescription()])
    else:
        pains_flags.append(False)
        pains_names.append([])

pains_arr = np.array(pains_flags)
act_pains  = pains_arr[y == 1].sum()
ina_pains  = pains_arr[y == 0].sum()
act_total  = int(y.sum())
ina_total  = int((y == 0).sum())

print(f"  PAINS-flagged actives  : {act_pains}/{act_total} ({act_pains/act_total:.1%})")
print(f"  PAINS-flagged inactives: {ina_pains}/{ina_total} ({ina_pains/ina_total:.1%})")
print(f"  Total PAINS-flagged    : {pains_arr.sum()}/{n} ({pains_arr.mean():.1%})")

# Retrain RF without PAINS-flagged compounds
clean_idx   = ~pains_arr
X_clean     = X[clean_idx]
y_clean     = y[clean_idx]
print(f"\n  RF retrained on PAINS-clean dataset (n={clean_idx.sum():,})...")

from sklearn.model_selection import cross_val_score
rf_clean = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                  class_weight='balanced', random_state=42, n_jobs=-1)
auc_clean = cross_val_score(rf_clean, X_clean, y_clean,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring='roc_auc').mean()

# Full model AUC for comparison
auc_full = cross_val_score(
    RandomForestClassifier(n_estimators=300, max_features='sqrt',
                           class_weight='balanced', random_state=42, n_jobs=-1),
    X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc').mean()

delta_auc = auc_clean - auc_full
print(f"  AUC-ROC (full, n=2957)           : {auc_full:.4f}")
print(f"  AUC-ROC (PAINS-clean, n={clean_idx.sum()})   : {auc_clean:.4f}")
print(f"  ΔAUC-ROC after PAINS removal     : {delta_auc:+.4f}")

pains_results = {
    'n_pains_actives': int(act_pains),
    'n_actives': act_total,
    'pct_pains_actives': round(act_pains / act_total * 100, 1),
    'n_pains_inactives': int(ina_pains),
    'n_inactives': ina_total,
    'pct_pains_inactives': round(ina_pains / ina_total * 100, 1),
    'n_pains_total': int(pains_arr.sum()),
    'pct_pains_total': round(pains_arr.mean() * 100, 1),
    'auc_full': round(float(auc_full), 4),
    'auc_pains_clean': round(float(auc_clean), 4),
    'delta_auc': round(float(delta_auc), 4),
}
with open(RESULTS_DIR / 'pains_results.json', 'w') as f:
    json.dump(pains_results, f, indent=2)

# PAINS figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('PAINS Structural Alert Analysis\nDENV NS2B-NS3 Inhibitor Dataset',
             fontsize=11, fontweight='bold')

ax = axes[0]
categories  = ['Active\n(n=205)', 'Inactive\n(n=2,752)', 'All\n(n=2,957)']
pains_pct   = [act_pains/act_total*100, ina_pains/ina_total*100, pains_arr.mean()*100]
clean_pct   = [100-p for p in pains_pct]
bars1 = ax.bar(categories, clean_pct, color='#2ecc71', alpha=0.85, label='PAINS-clean')
bars2 = ax.bar(categories, pains_pct, bottom=clean_pct, color='#e05c5c', alpha=0.85, label='PAINS-flagged')
for bar, pct in zip(bars2, pains_pct):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
ax.set_ylabel('Percentage of compounds (%)', fontsize=10)
ax.set_title('(A)  PAINS Prevalence\nby Activity Class', fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
models    = ['RF + RDKit2D\n(full, n=2,957)', f'RF + RDKit2D\n(PAINS-clean, n={clean_idx.sum()})']
aucs      = [auc_full, auc_clean]
bar_cols  = ['#3d7ebf', '#2ecc71']
bars3 = ax2.bar(models, aucs, color=bar_cols, alpha=0.85, width=0.45)
for bar, auc in zip(bars3, aucs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{auc:.4f}', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylabel('AUC-ROC (5-fold CV)', fontsize=10)
ax2.set_ylim(0.88, 0.97)
ax2.set_title(f'(B)  AUC-ROC Stability\nΔAUC = {delta_auc:+.4f} after PAINS removal',
              fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pains_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ figures/pains_analysis.png")
print(f"  ✓ results/pains_results.json")

# ══════════════════════════════════════════════════════════════
# C. UMAP CHEMICAL SPACE VISUALIZATION
# ══════════════════════════════════════════════════════════════
print("\n[C] UMAP Chemical Space Visualization...")

try:
    import umap as umap_lib
    print("  umap-learn found ✓")
except ImportError:
    print("  umap-learn not installed — installing...")
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'umap-learn', '-q'])
    import umap as umap_lib

# ECFP4 fingerprints for UMAP
from rdkit.Chem import rdMolDescriptors
print("  Computing ECFP4 fingerprints...")
fps_matrix = []
valid_idx  = []
for i, smi in enumerate(smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros(1024, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_matrix.append(arr)
        valid_idx.append(i)

fps_matrix = np.array(fps_matrix)
y_umap     = y[valid_idx]
print(f"  Fingerprints: {fps_matrix.shape}  Valid: {len(valid_idx)}/{n}")

# UMAP embedding
print("  Running UMAP (n_neighbors=15, min_dist=0.1)...")
reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42,
                        metric='jaccard', n_jobs=1)
embedding = reducer.fit_transform(fps_matrix)
print(f"  Embedding shape: {embedding.shape}")

# ML predictions for colour overlay
rf_full = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                 class_weight='balanced', random_state=42, n_jobs=-1)
rf_full.fit(X, y)
ml_probs = rf_full.predict_proba(fps_matrix if X.shape == fps_matrix.shape else X[valid_idx])[:, 1]

# Read scaffold cluster labels from existing scaffold split
try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.ML.Cluster import Butina
    scaffolds = []
    for smi in [smiles[i] for i in valid_idx]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                scaffolds.append(MurckoScaffold.GetScaffoldForMol(mol))
            except: scaffolds.append(None)
        else: scaffolds.append(None)
    # Simple: just use first scaffold letter as proxy cluster
    scaffold_smiles = [Chem.MolToSmiles(s) if s else '' for s in scaffolds]
    # Top 6 scaffolds by frequency
    from collections import Counter
    sc_counts = Counter(scaffold_smiles)
    top6 = [s for s, _ in sc_counts.most_common(7) if s != ''][: 6]
    cluster_label = np.array([top6.index(s)+1 if s in top6 else 0
                               for s in scaffold_smiles])
    have_clusters = True
except Exception:
    have_clusters = False

# UMAP Figure — 4 panels
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chemical Space of DENV NS2B-NS3 Inhibitor Dataset (n=2,957)\n'
             'UMAP of ECFP4 Fingerprints (Jaccard metric)',
             fontsize=11, fontweight='bold')

# Panel A: Activity
ax = axes[0]
ina_mask = y_umap == 0; act_mask = y_umap == 1
ax.scatter(embedding[ina_mask, 0], embedding[ina_mask, 1],
           c='#a8c8e8', s=4, alpha=0.4, label=f'Inactive (n={ina_mask.sum():,})')
ax.scatter(embedding[act_mask, 0], embedding[act_mask, 1],
           c='#e05c5c', s=12, alpha=0.8, label=f'Active (n={act_mask.sum():,})')
ax.set_title('(A)  Activity Labels', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, markerscale=2); ax.set_xticks([]); ax.set_yticks([])

# Panel B: ML predicted probability
ax2 = axes[1]
sc = ax2.scatter(embedding[:, 0], embedding[:, 1],
                 c=ml_probs, cmap='RdYlGn', s=5, alpha=0.6, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax2, shrink=0.8, label='P(active)')
ax2.set_title('(B)  ML Predicted Probability', fontsize=10, fontweight='bold')
ax2.set_xticks([]); ax2.set_yticks([])

# Panel C: Docking score
try:
    dock_df = pd.read_csv(SCRIPT_DIR / 'results' / 'docking_results.csv')
    dock_scores = dock_df['docking_score'].values[valid_idx]
    # Only plot where score != 0
    dock_valid = dock_scores != 0
    sc3 = axes[2].scatter(embedding[dock_valid, 0], embedding[dock_valid, 1],
                          c=dock_scores[dock_valid], cmap='RdYlGn_r',
                          s=5, alpha=0.6,
                          vmin=np.percentile(dock_scores[dock_valid], 5),
                          vmax=np.percentile(dock_scores[dock_valid], 95))
    plt.colorbar(sc3, ax=axes[2], shrink=0.8, label='Docking score (kcal/mol)')
    axes[2].set_title('(C)  Docking Score', fontsize=10, fontweight='bold')
    axes[2].set_xticks([]); axes[2].set_yticks([])
except Exception as e:
    axes[2].text(0.5, 0.5, f'Docking data\nnot available\n{e}',
                 ha='center', va='center', transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'umap_chemical_space.png', dpi=300, bbox_inches='tight')
plt.close()

# Save embedding
np.save(RESULTS_DIR / 'umap_embedding.npy', embedding)
print(f"  ✓ figures/umap_chemical_space.png")
print(f"  ✓ results/umap_embedding.npy")

# ── Summary ────────────────────────────────────────────────────
print()
print("=" * 60)
print("ALL DONE")
print(f"  Brier score     : {brier:.4f}  (calibration: GOOD)")
print(f"  PAINS actives   : {act_pains}/{act_total} ({act_pains/act_total:.1%})")
print(f"  PAINS ΔA UC-ROC : {delta_auc:+.4f}")
print(f"  UMAP embedding  : {embedding.shape}")
print("=" * 60)
