import os, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import CrossConformalClassifier
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("Loading features...")
print("=" * 60)
with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)
with open('results/best_combo.json') as f:
    best = json.load(f)

y = data['y']
X = data['rdkit']

print(f"  Feature matrix : {X.shape}")
print(f"  Labels         : {y.sum()} active / {(1-y).sum()} inactive")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n  Train: {len(y_train):,}  |  Test: {len(y_test):,}")

print("\n" + "=" * 60)
print("Baseline model (RF + RDKit2D)...")
print("=" * 60)

base_model = RandomForestClassifier(
    n_estimators=300, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
base_model.fit(X_train, y_train)
base_proba         = base_model.predict_proba(X_test)[:, 1]
base_pred          = (base_proba >= 0.5).astype(int)
baseline_precision = precision_score(y_test, base_pred, zero_division=0)
baseline_recall    = recall_score(y_test, base_pred,    zero_division=0)

print(f"  Baseline precision : {baseline_precision:.3f}")
print(f"  Baseline recall    : {baseline_recall:.3f}")

print("\n" + "=" * 60)
print("Conformal prediction across confidence levels (MAPIE 1.4.1)...")
print("=" * 60)

confidence_levels = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
conf_results = []

for conf_level in confidence_levels:

    clf = CrossConformalClassifier(
        estimator=RandomForestClassifier(
            n_estimators=300, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        confidence_level=conf_level,
        conformity_score='lac',
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit_conformalize(X_train, y_train)
    _, y_pred_sets_raw = clf.predict_set(X_test)

    y_pred_sets = y_pred_sets_raw.squeeze(-1)

    in_class_0 = y_pred_sets[:, 0].astype(bool)
    in_class_1 = y_pred_sets[:, 1].astype(bool)

    ambiguous   = in_class_0 & in_class_1
    certain_neg = in_class_0 & ~in_class_1
    certain_pos = ~in_class_0 & in_class_1
    empty_set   = ~in_class_0 & ~in_class_1

    n_total    = len(y_test)
    n_certain  = int((~ambiguous & ~empty_set).sum())
    efficiency = n_certain / n_total

    true_in_set = ((y_test == 0) & in_class_0) | ((y_test == 1) & in_class_1)
    coverage    = true_in_set.mean()

    n_conf_active    = int(certain_pos.sum())
    prec_conf_active = float(y_test[certain_pos.flatten()].mean()) if n_conf_active > 0 else float('nan')

    conf_results.append({
        'confidence_level':      conf_level,
        'target_coverage':       conf_level,
        'empirical_coverage':    float(round(coverage, 4)),
        'n_certain':             n_certain,
        'n_ambiguous':           int(ambiguous.sum()),
        'efficiency':            float(round(efficiency, 4)),
        'n_conf_active':         n_conf_active,
        'precision_conf_active': round(prec_conf_active, 4) if not np.isnan(prec_conf_active) else None,
    })

    prec_str = f"{prec_conf_active:.3f}" if not np.isnan(prec_conf_active) else "N/A"
    print(f"  {conf_level:.0%} conf | "
          f"Coverage={coverage:.3f} | "
          f"Certain={n_certain}/{n_total} ({efficiency:.1%}) | "
          f"Conf.actives={n_conf_active} | Prec={prec_str}")

conf_df = pd.DataFrame(conf_results)

print("\n" + "=" * 60)
print("KEY RESULTS (Table 3 — 80% and 90% confidence)")
print("=" * 60)
for cl in [0.80, 0.90]:
    row = conf_df[conf_df['confidence_level'] == cl].iloc[0]
    print(f"\n  {cl:.0%} confidence:")
    print(f"    Empirical coverage    : {row['empirical_coverage']:.3f}  (target: {cl:.3f})")
    print(f"    Efficiency            : {row['efficiency']:.3f}  ({row['n_certain']}/{len(y_test)} compounds certain)")
    print(f"    Conf. actives         : {row['n_conf_active']}")
    print(f"    Precision (conf. act.): {row['precision_conf_active']}")
print(f"\n  Baseline precision: {baseline_precision:.3f}")

conf_df.to_csv('results/conformal_results.csv', index=False)
results_out = {
    'baseline_precision': float(round(baseline_precision, 4)),
    'baseline_recall':    float(round(baseline_recall, 4)),
    'n_test':             int(len(y_test)),
    'results_80pct':      conf_df[conf_df['confidence_level'] == 0.80].to_dict('records')[0],
    'results_90pct':      conf_df[conf_df['confidence_level'] == 0.90].to_dict('records')[0],
    'all_levels':         conf_df.to_dict('records'),
}
with open('results/conformal_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print("\n  ✓ results/conformal_results.csv")
print("  ✓ results/conformal_results.json")

print("\n" + "=" * 60)
print("Generating Figure 4 (combined two panels)...")
print("=" * 60)

fig = plt.figure(figsize=(13, 5.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40)

ax1 = fig.add_subplot(gs[0])
ax1.plot(conf_df['confidence_level'] * 100,
         conf_df['empirical_coverage'] * 100,
         'o-', color='#3d7ebf', linewidth=2.2, markersize=7,
         label='Empirical coverage', zorder=3)
ax1.plot(conf_df['confidence_level'] * 100,
         conf_df['confidence_level'] * 100,
         '--', color='#888', linewidth=1.5, label='Ideal (target)', zorder=2)
ax1.fill_between(
    conf_df['confidence_level'] * 100,
    conf_df['confidence_level'] * 100,
    conf_df['empirical_coverage'] * 100,
    alpha=0.13, color='#3d7ebf'
)
for cl in [0.80, 0.90]:
    ax1.axvline(cl * 100, color='#e05c5c', linestyle=':', alpha=0.6, linewidth=1.2)
    ax1.annotate(f"{cl:.0%}", xy=(cl*100 + 0.3, 61), fontsize=7.5, color='#e05c5c', fontweight='bold')

ax1.set_xlabel('Target confidence level (%)', fontsize=10)
ax1.set_ylabel('Empirical coverage (%)', fontsize=10)
ax1.set_title('(A)  Coverage Validity\nEmpirical vs Target', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8.5)
ax1.grid(alpha=0.3)
ax1.set_xlim(57, 98)
ax1.set_ylim(57, 103)

ax2 = fig.add_subplot(gs[1])
x     = np.arange(len(conf_df))
width = 0.35

ax2.bar(x - width/2, conf_df['efficiency'] * 100,
        width, label='Efficiency (% certain compounds)',
        color='#5c8ee0', alpha=0.85, edgecolor='white')

prec_vals = [r['precision_conf_active'] if r['precision_conf_active'] is not None
             else baseline_precision for _, r in conf_df.iterrows()]
ax2.bar(x + width/2, [p * 100 for p in prec_vals],
        width, label='Precision of confident actives',
        color='#e05c5c', alpha=0.85, edgecolor='white')

ax2.axhline(baseline_precision * 100, color='#333', linestyle='--',
            linewidth=1.5, label=f'Baseline precision ({baseline_precision:.1%})')

ax2.set_xticks(x)
ax2.set_xticklabels([f'{int(cl*100)}%' for cl in conf_df['confidence_level']], fontsize=8)
ax2.set_xlabel('Confidence level', fontsize=10)
ax2.set_ylabel('Percentage (%)', fontsize=10)
ax2.set_title('(B)  Efficiency vs Precision\nAcross Confidence Levels', fontsize=10, fontweight='bold')
ax2.legend(fontsize=7.5, loc='lower left')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 115)

fig.suptitle('Conformal Prediction — Dengue NS2B-NS3 Bioactivity\n'
             'MAPIE CrossConformalClassifier (LAC score, 5-fold CV) · Random Forest + RDKit2D',
             fontsize=11, fontweight='bold', y=1.02)

plt.savefig('figures/conformal_combined.png', dpi=300, bbox_inches='tight')
plt.close()

fig2, ax = plt.subplots(figsize=(7, 5))
ax.plot(conf_df['confidence_level']*100, conf_df['empirical_coverage']*100,
        'o-', color='#3d7ebf', lw=2.2, ms=7, label='Empirical coverage')
ax.plot(conf_df['confidence_level']*100, conf_df['confidence_level']*100,
        '--', color='#888', lw=1.5, label='Ideal (target)')
ax.fill_between(conf_df['confidence_level']*100, conf_df['confidence_level']*100,
                conf_df['empirical_coverage']*100, alpha=0.13, color='#3d7ebf')
ax.set_xlabel('Target confidence level (%)', fontsize=10)
ax.set_ylabel('Empirical coverage (%)', fontsize=10)
ax.set_title('Conformal Prediction Coverage Validity', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/conformal_coverage.png', dpi=300, bbox_inches='tight')
plt.close()
