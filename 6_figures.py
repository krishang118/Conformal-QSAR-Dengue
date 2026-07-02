import os, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import seaborn as sns
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("Loading data...")
print("=" * 60)
df = pd.read_csv('data/chembl_dengue_clean.csv')
with open('data/features.pkl', 'rb') as f:
    feat = pickle.load(f)

y       = feat['y']
active  = df[df['activity'] == 1]
inactive= df[df['activity'] == 0]

ACTIVE_COLOR   = '#2d8a6e'
INACTIVE_COLOR = '#c0392b'
PALETTE        = [ACTIVE_COLOR, INACTIVE_COLOR]

print(f"  Dataset: {len(df):,} compounds  |  Active: {len(active):,}  |  Inactive: {len(inactive):,}")

print("\n" + "=" * 60)
print("Figure 1: Pipeline overview schematic")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

stages = [
    ('ChEMBL\nDatabase', 'CHEMBL5980\nDengue NS2B-NS3', '#4a6fa5', '①'),
    ('Data\nCleaning', 'IC50 filter\nSMILES validate\nDeduplicate', '#e67e22', '②'),
    ('Molecular\nFeaturization', 'ECFP4 · MACCS\nRDKit2D · AtomPair', '#8e44ad', '③'),
    ('ML\nBenchmarking', '4 models × 4 FPs\n5-fold CV\nAUC-ROC / MCC', '#27ae60', '④'),
    ('SHAP\nAnalysis', 'RDKit descriptors\nECFP4 bits\nStructure-activity', '#2980b9', '⑤'),
    ('Conformal\nPrediction', 'MAPIE classifier\n80% / 90% CI\nUncertainty quant.', '#c0392b', '⑥'),
]

box_w, box_h = 1.9, 1.9
gap           = 0.45
start_x       = 0.35
y_center      = 2.0

for i, (title, subtitle, color, num) in enumerate(stages):
    x = start_x + i * (box_w + gap)

    shadow = mpatches.FancyBboxPatch(
        (x + 0.06, y_center - box_h/2 - 0.06),
        box_w, box_h,
        boxstyle='round,pad=0.1',
        facecolor='#cccccc', edgecolor='none', zorder=1, alpha=0.5
    )
    ax.add_patch(shadow)

    box = mpatches.FancyBboxPatch(
        (x, y_center - box_h/2),
        box_w, box_h,
        boxstyle='round,pad=0.1',
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=2
    )
    ax.add_patch(box)

    badge = plt.Circle((x + 0.22, y_center + box_h/2 - 0.22), 0.18,
                        color='white', zorder=3)
    ax.add_patch(badge)
    ax.text(x + 0.22, y_center + box_h/2 - 0.22, num,
            ha='center', va='center', fontsize=7, color=color,
            fontweight='bold', zorder=4)

    ax.text(x + box_w/2, y_center + 0.35, title,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', zorder=3, linespacing=1.3)

    ax.text(x + box_w/2, y_center - 0.5, subtitle,
            ha='center', va='center', fontsize=6.5,
            color='white', alpha=0.92, zorder=3, linespacing=1.4)

    if i < len(stages) - 1:
        ax.annotate('', xy=(x + box_w + gap, y_center),
                    xytext=(x + box_w + 0.02, y_center),
                    arrowprops=dict(arrowstyle='->', color='#555',
                                   lw=1.8, mutation_scale=16),
                    zorder=5)

ax.set_title(
    'Computational Pipeline: Interpretable ML for Dengue NS2B-NS3 Bioactivity Prediction',
    fontsize=12, fontweight='bold', color='#2c3e50', pad=10
)

plt.tight_layout(pad=0.5)
plt.savefig('figures/pipeline_overview.png', dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("  ✓ figures/pipeline_overview.png")

print("\n" + "=" * 60)
print("Figure 5: Dataset EDA")
print("=" * 60)

fig = plt.figure(figsize=(14, 9))
gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

ax1 = fig.add_subplot(gs[0, 0])
bins = np.logspace(np.log10(df['ic50_median'].min() + 0.01),
                   np.log10(df['ic50_median'].max()), 50)
ax1.hist(active['ic50_median'],   bins=bins, alpha=0.7, color=ACTIVE_COLOR,   label='Active',   density=True)
ax1.hist(inactive['ic50_median'], bins=bins, alpha=0.7, color=INACTIVE_COLOR, label='Inactive', density=True)
ax1.axvline(10000, color='black', linestyle='--', linewidth=1.5, label='Threshold (10,000 nM = 10 µM)')
ax1.set_xscale('log')
ax1.set_xlabel('IC50 (nM, log scale)', fontsize=9)
ax1.set_ylabel('Density', fontsize=9)
ax1.set_title('(A) IC50 Distribution', fontsize=10, fontweight='bold')
ax1.legend(fontsize=7.5)
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(active['pIC50'],   bins=30, alpha=0.7, color=ACTIVE_COLOR,   label='Active',   density=True)
ax2.hist(inactive['pIC50'], bins=30, alpha=0.7, color=INACTIVE_COLOR, label='Inactive', density=True)
ax2.axvline(-np.log10(10000e-9), color='black', linestyle='--', lw=1.5, label='Threshold (pIC50 = 5.0)')
ax2.set_xlabel('pIC50', fontsize=9)
ax2.set_ylabel('Density', fontsize=9)
ax2.set_title('(B) pIC50 Distribution', fontsize=10, fontweight='bold')
ax2.legend(fontsize=7.5)
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
sizes  = [len(active), len(inactive)]
labels = [f'Active\n(IC50 ≤ 10,000 nM)\nn={len(active):,}',
          f'Inactive\n(IC50 > 10,000 nM)\nn={len(inactive):,}']
wedges, texts, autotexts = ax3.pie(
    sizes, labels=labels, colors=[ACTIVE_COLOR, INACTIVE_COLOR],
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 8},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')
    at.set_color('white')
ax3.set_title('(C) Class Balance', fontsize=10, fontweight='bold')

ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(active['mw'],   bins=30, alpha=0.7, color=ACTIVE_COLOR,   label='Active',   density=True)
ax4.hist(inactive['mw'], bins=30, alpha=0.7, color=INACTIVE_COLOR, label='Inactive', density=True)
ax4.axvline(500, color='black', linestyle='--', lw=1.2, label='Lipinski MW limit')
ax4.set_xlabel('Molecular Weight (Da)', fontsize=9)
ax4.set_ylabel('Density', fontsize=9)
ax4.set_title('(D) MW Distribution', fontsize=10, fontweight='bold')
ax4.legend(fontsize=7.5)
ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(active['logp'],   bins=30, alpha=0.7, color=ACTIVE_COLOR,   label='Active',   density=True)
ax5.hist(inactive['logp'], bins=30, alpha=0.7, color=INACTIVE_COLOR, label='Inactive', density=True)
ax5.axvline(5, color='black', linestyle='--', lw=1.2, label='Lipinski LogP limit')
ax5.set_xlabel('LogP', fontsize=9)
ax5.set_ylabel('Density', fontsize=9)
ax5.set_title('(E) LogP Distribution', fontsize=10, fontweight='bold')
ax5.legend(fontsize=7.5)
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(inactive['hbd'], inactive['hba'], c=INACTIVE_COLOR, alpha=0.3, s=12, label='Inactive', zorder=2)
ax6.scatter(active['hbd'],   active['hba'],   c=ACTIVE_COLOR,   alpha=0.3, s=12, label='Active',   zorder=3)
ax6.set_xlabel('H-Bond Donors (HBD)', fontsize=9)
ax6.set_ylabel('H-Bond Acceptors (HBA)', fontsize=9)
ax6.set_title('(F) HBD vs HBA', fontsize=10, fontweight='bold')
ax6.legend(fontsize=7.5, markerscale=2)
ax6.grid(alpha=0.3)

fig.suptitle(f'Dataset Overview — ChEMBL Dengue NS2B-NS3 Bioactivity (n={len(df):,})',
             fontsize=12, fontweight='bold', y=1.01)
plt.savefig('figures/dataset_eda.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/dataset_eda.png")

print("\n" + "=" * 60)
print("Figure 6: t-SNE chemical space (ECFP4)")
print("=" * 60)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

X_ecfp4 = feat['ecfp4']

print("  Running PCA (50 components)...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_ecfp4)

print("  Running t-SNE (perplexity=40)...")
tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
            random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca)

fig, ax = plt.subplots(figsize=(9, 7))
scatter_inactive = ax.scatter(
    X_tsne[y==0, 0], X_tsne[y==0, 1],
    c=INACTIVE_COLOR, alpha=0.4, s=10, label=f'Inactive (n={int((y==0).sum()):,})',
    linewidths=0, zorder=2
)
scatter_active = ax.scatter(
    X_tsne[y==1, 0], X_tsne[y==1, 1],
    c=ACTIVE_COLOR, alpha=0.4, s=10, label=f'Active (n={int((y==1).sum()):,})',
    linewidths=0, zorder=3
)
ax.set_xlabel('t-SNE Component 1', fontsize=10)
ax.set_ylabel('t-SNE Component 2', fontsize=10)
ax.set_title('Chemical Space Visualization — ECFP4 Fingerprints (t-SNE)\n'
             'Dengue NS2B-NS3 Bioactivity Dataset',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, markerscale=2.5)
ax.grid(alpha=0.2, linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/tsne_chemical_space.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/tsne_chemical_space.png")

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED")
print("=" * 60)

figures_dir = 'figures'
for fname in sorted(os.listdir(figures_dir)):
    if fname.endswith('.png'):
        size = os.path.getsize(os.path.join(figures_dir, fname)) / 1024
        print(f"  ✓ figures/{fname:<45s} ({size:.0f} KB)")

print()
