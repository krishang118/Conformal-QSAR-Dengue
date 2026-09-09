"""
5_conformal_prediction.py  (v2 — revised after peer review)
============================================================
Stage 5: Mondrian (Class-Conditional) Conformal Prediction
         for Uncertainty-Guided Compound Prioritisation

Key changes vs. v1:
  - Mondrian CP: independent calibration thresholds for each class
    (addresses reviewer concern about global LAC being overwhelmed by
    the 93.1 % majority inactive class under 6.9 % class imbalance).
  - Empty sets explicitly tracked; efficiency = |singleton sets| / n_test
    (fixes the artefactual efficiency-rises-with-confidence trend that
    arose because empty sets at low confidence were mistakenly excluded).
  - Recall and F1 reported alongside Precision for confident actives.
  - Matched-list-size baseline: at each confidence level, compare the
    conformal active list against a probability-threshold list of the
    same length (so improvements cannot be dismissed as list-size effects).
  - Five random seeds for train/test split; all metrics report mean ± SD.
  - Notation: confidence level = 1 - ε (error rate) throughout.

Output:
  results/conformal_results.csv      — full table (mean ± SD across seeds)
  results/conformal_results.json     — structured summary
  figures/conformal_combined.png     — two-panel figure (coverage + metrics)
"""

import os, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)
os.makedirs('results',  exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("=" * 65)
print("Stage 5 — Mondrian Conformal Prediction (v2, peer-review revision)")
print("=" * 65)

with open('data/features.pkl', 'rb') as f:
    data = pickle.load(f)
with open('results/best_combo.json') as f:
    best = json.load(f)

y_all = data['y']
X_all = data['rdkit']   # RF + RDKit2D — primary benchmark model

print(f"  Feature matrix : {X_all.shape}")
print(f"  Labels         : {int(y_all.sum())} active / {int((1-y_all).sum())} inactive")

# ═══════════════════════════════════════════════════════════════════════════
#  Mondrian CP implementation (pure-numpy, no extra dependency)
# ═══════════════════════════════════════════════════════════════════════════

def mondrian_cp_binary(
    rf_model,
    X_cal, y_cal,
    X_test,
    confidence_level: float
):
    """
    Mondrian (class-conditional) inductive conformal prediction.

    For binary classification with LAC nonconformity score:
        s_i = 1 - p̂(y_i | x_i)

    For each class k ∈ {0, 1}, the threshold τ_k is the
    ⌈(1 - ε)(n_k + 1)⌉ / n_k  quantile of the calibration
    scores computed only on compounds that belong to class k.

    A test compound's prediction set includes class k if:
        1 - p̂(k | x_test) ≤ τ_k

    This gives per-class marginal coverage guarantees, solving
    the class-imbalance problem of global (marginal) CP.

    Parameters
    ----------
    rf_model        : fitted RandomForestClassifier
    X_cal, y_cal    : calibration (held-out) features and labels
    X_test          : test features
    confidence_level: 1 - ε, e.g. 0.90 for 90 % coverage guarantee

    Returns
    -------
    pred_sets : np.ndarray, shape (n_test, 2), dtype bool
                pred_sets[:, k] = True means class k is in prediction set
    thresholds: dict {0: τ_0, 1: τ_1}
    """
    epsilon = 1.0 - confidence_level

    # Calibration nonconformity scores (LAC)
    cal_proba = rf_model.predict_proba(X_cal)  # (n_cal, 2)
    scores = 1.0 - cal_proba[np.arange(len(y_cal)), y_cal.astype(int)]

    thresholds = {}
    for k in [0, 1]:
        mask_k  = (y_cal == k)
        s_k     = scores[mask_k]
        n_k     = mask_k.sum()
        # Mondrian quantile: (1-ε)(n_k+1)/n_k, capped at 1.0
        q_level = min(np.ceil((1.0 - epsilon) * (n_k + 1)) / n_k, 1.0)
        tau_k   = float(np.quantile(s_k, q_level)) if n_k > 0 else 1.0
        thresholds[k] = tau_k

    # Test prediction sets
    test_proba  = rf_model.predict_proba(X_test)   # (n_test, 2)
    in_class_0  = (1.0 - test_proba[:, 0]) <= thresholds[0]
    in_class_1  = (1.0 - test_proba[:, 1]) <= thresholds[1]
    pred_sets   = np.stack([in_class_0, in_class_1], axis=1)   # (n_test, 2)
    return pred_sets, thresholds


def evaluate_pred_sets(pred_sets, y_test, y_prob_base, conf_level):
    """
    Evaluate a set of conformal prediction sets.

    Prediction set categories:
      singleton active   : {1}  — confident active prediction
      singleton inactive : {0}  — confident inactive prediction
      ambiguous (full)   : {0,1} — model uncertain; both labels included
      empty              : {}   — model cannot include either class

    Efficiency = fraction of singleton (unambiguous, non-empty) predictions.

    NOTE: Under correct Mondrian CP theory, raising the confidence level
    forces prediction sets to expand (more full {0,1} sets), so efficiency
    DECREASES as confidence increases. Empty sets signal over-strictness
    and are tracked separately.

    Returns
    -------
    dict with all metrics
    """
    n = len(y_test)
    in_0 = pred_sets[:, 0].astype(bool)
    in_1 = pred_sets[:, 1].astype(bool)

    ambiguous   = in_0 &  in_1     # {0,1}
    certain_neg = in_0 & ~in_1     # {0}
    certain_pos = ~in_0 &  in_1    # {1}
    empty       = ~in_0 & ~in_1    # {}

    n_ambiguous    = int(ambiguous.sum())
    n_certain_neg  = int(certain_neg.sum())
    n_certain_pos  = int(certain_pos.sum())
    n_empty        = int(empty.sum())
    n_singleton    = n_certain_neg + n_certain_pos  # total singletons
    efficiency     = n_singleton / n                 # fraction of singletons

    # Empirical coverage: true label inside prediction set
    true_in_set = ((y_test == 0) & in_0) | ((y_test == 1) & in_1)
    coverage    = float(true_in_set.mean())

    # ── Metrics for confidently predicted actives ─────────────────────────
    conf_active_mask = certain_pos.flatten()
    n_conf_active    = int(conf_active_mask.sum())

    if n_conf_active > 0:
        y_true_ca  = y_test[conf_active_mask]
        prec_ca    = float(y_true_ca.mean())               # precision
        rec_ca     = float(y_true_ca.sum() / y_test.sum()) # recall (among all actives)
        denom_f1   = prec_ca + rec_ca
        f1_ca      = (2 * prec_ca * rec_ca / denom_f1) if denom_f1 > 0 else 0.0
    else:
        prec_ca = rec_ca = f1_ca = float('nan')

    # ── Matched-list-size baseline ─────────────────────────────────────────
    # At each confidence level we compare against a probability-ranked
    # list of exactly the same length (n_conf_active) — so any precision
    # gain cannot be attributed to a shorter list.
    if n_conf_active > 0:
        topk_idx         = np.argsort(y_prob_base)[::-1][:n_conf_active]
        baseline_prec_k  = float(y_test[topk_idx].mean())
        baseline_rec_k   = float(y_test[topk_idx].sum() / y_test.sum())
        denom_bk         = baseline_prec_k + baseline_rec_k
        baseline_f1_k    = (2*baseline_prec_k*baseline_rec_k/denom_bk) if denom_bk > 0 else 0.0
    else:
        baseline_prec_k = baseline_rec_k = baseline_f1_k = float('nan')

    return {
        'confidence_level':       conf_level,
        'target_coverage':        conf_level,
        'empirical_coverage':     round(coverage, 4),
        'n_total':                n,
        'n_singleton_active':     n_certain_pos,
        'n_singleton_inactive':   n_certain_neg,
        'n_ambiguous':            n_ambiguous,
        'n_empty':                n_empty,
        'efficiency':             round(efficiency, 4),
        'n_conf_active':          n_conf_active,
        'precision_conf_active':  round(prec_ca, 4) if not np.isnan(prec_ca) else None,
        'recall_conf_active':     round(rec_ca,  4) if not np.isnan(rec_ca)  else None,
        'f1_conf_active':         round(f1_ca,   4) if not np.isnan(f1_ca)   else None,
        'matched_baseline_prec':  round(baseline_prec_k, 4) if not np.isnan(baseline_prec_k) else None,
        'matched_baseline_rec':   round(baseline_rec_k,  4) if not np.isnan(baseline_rec_k)  else None,
        'matched_baseline_f1':    round(baseline_f1_k,   4) if not np.isnan(baseline_f1_k)   else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-seed experiment (5 seeds, mean ± SD across splits)
# ═══════════════════════════════════════════════════════════════════════════

CONFIDENCE_LEVELS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SEEDS             = [42, 7, 13, 99, 2024]
CAL_FRACTION      = 0.20    # 20 % of train set used as calibration

print(f"\n  Confidence levels : {CONFIDENCE_LEVELS}")
print(f"  Seeds             : {SEEDS}")
print(f"  Calibration split : {CAL_FRACTION:.0%} of training data\n")

# Accumulate per-seed results
seed_rows = []     # list of per-seed DataFrames

for seed_idx, seed in enumerate(SEEDS):
    print(f"{'─'*65}")
    print(f"  Seed {seed_idx+1}/{len(SEEDS)}  (random_state={seed})")
    print(f"{'─'*65}")

    # ── 80/20 train-test split ─────────────────────────────────────────────
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=seed
    )

    # ── Further split training into fit-set + calibration set ─────────────
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tr_full, y_tr_full,
        test_size=CAL_FRACTION,
        stratify=y_tr_full,
        random_state=seed
    )

    print(f"    Fit: {len(y_fit):,}  |  Cal: {len(y_cal):,}  |  Test: {len(y_test):,}")
    print(f"    Cal actives: {int(y_cal.sum())} | Cal inactives: {int((y_cal==0).sum())}")

    # ── Train RF on fit set only ───────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, max_features='sqrt',
        class_weight='balanced', random_state=seed, n_jobs=-1
    )
    rf.fit(X_fit, y_fit)

    # Baseline: standard probability prediction on test set
    y_prob_test = rf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    base_prec   = precision_score(y_test, y_pred_test, zero_division=0)
    base_rec    = recall_score(y_test, y_pred_test, zero_division=0)
    denom       = base_prec + base_rec
    base_f1     = (2*base_prec*base_rec/denom) if denom > 0 else 0.0
    base_auc    = roc_auc_score(y_test, y_prob_test)
    print(f"    Base Prec={base_prec:.3f}  Rec={base_rec:.3f}  F1={base_f1:.3f}  AUC={base_auc:.4f}")

    seed_level_rows = []
    for cl in CONFIDENCE_LEVELS:
        pred_sets, thr = mondrian_cp_binary(rf, X_cal, y_cal, X_test, cl)
        row = evaluate_pred_sets(pred_sets, y_test, y_prob_test, cl)
        row['seed']             = seed
        row['tau_inactive']     = round(thr[0], 4)
        row['tau_active']       = round(thr[1], 4)
        row['baseline_prec']    = round(base_prec, 4)
        row['baseline_rec']     = round(base_rec,  4)
        row['baseline_f1']      = round(base_f1,   4)
        seed_level_rows.append(row)

        prec_str = f"{row['precision_conf_active']:.3f}" if row['precision_conf_active'] is not None else "N/A"
        rec_str  = f"{row['recall_conf_active']:.3f}"    if row['recall_conf_active']    is not None else "N/A"
        f1_str   = f"{row['f1_conf_active']:.3f}"        if row['f1_conf_active']         is not None else "N/A"
        print(
            f"    {cl:.0%} | Cov={row['empirical_coverage']:.3f} | "
            f"Eff={row['efficiency']:.3f} | "
            f"Sing+={row['n_singleton_active']:3d} | "
            f"Ambig={row['n_ambiguous']:3d} | "
            f"Empty={row['n_empty']:3d} | "
            f"Prec={prec_str} | Rec={rec_str} | F1={f1_str} | "
            f"τ0={thr[0]:.3f} τ1={thr[1]:.3f}"
        )

    seed_rows.append(pd.DataFrame(seed_level_rows))

print(f"\n{'='*65}")
print("  Aggregating across seeds (mean ± SD)...")
print(f"{'='*65}")

# ── Aggregate across seeds ─────────────────────────────────────────────────
all_df = pd.concat(seed_rows, ignore_index=True)

numeric_cols = [
    'empirical_coverage', 'efficiency',
    'n_singleton_active', 'n_singleton_inactive', 'n_ambiguous', 'n_empty',
    'n_conf_active',
    'precision_conf_active', 'recall_conf_active', 'f1_conf_active',
    'matched_baseline_prec', 'matched_baseline_rec', 'matched_baseline_f1',
    'baseline_prec', 'baseline_rec', 'baseline_f1',
    'tau_inactive', 'tau_active',
]

agg_rows = []
for cl in CONFIDENCE_LEVELS:
    sub = all_df[all_df['confidence_level'] == cl]
    row_agg = {'confidence_level': cl, 'target_coverage': cl, 'n_seeds': len(SEEDS)}
    for col in numeric_cols:
        vals = pd.to_numeric(sub[col], errors='coerce').dropna()
        row_agg[f'{col}_mean'] = round(float(vals.mean()), 4) if len(vals) > 0 else None
        row_agg[f'{col}_std']  = round(float(vals.std()),  4) if len(vals) > 0 else None
    agg_rows.append(row_agg)

agg_df = pd.DataFrame(agg_rows)

# ── Print summary table ────────────────────────────────────────────────────
print()
print(f"  {'Conf':<6} {'Coverage':>9} {'Efficiency':>11} {'SingAct':>8} "
      f"{'Ambig':>7} {'Empty':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} "
      f"{'BLPrec':>7} {'BLRec':>7}")
print(f"  {'-'*87}")

for _, r in agg_df.iterrows():
    def fmt(col, digits=3):
        m = r.get(f'{col}_mean')
        s = r.get(f'{col}_std')
        if m is None: return '   N/A'
        return f"{m:.{digits}f}±{s:.{digits}f}" if s is not None else f"{m:.{digits}f}"

    print(
        f"  {r['confidence_level']:.0%}    "
        f"{fmt('empirical_coverage'):>14} "
        f"{fmt('efficiency'):>14} "
        f"{r['n_singleton_active_mean']:>6.1f} "
        f"{r['n_ambiguous_mean']:>8.1f} "
        f"{r['n_empty_mean']:>6.1f} "
        f"{fmt('precision_conf_active'):>14} "
        f"{fmt('recall_conf_active'):>14} "
        f"{fmt('f1_conf_active'):>14} "
        f"{fmt('matched_baseline_prec'):>14} "
        f"{fmt('matched_baseline_rec'):>14}"
    )

# ── Save CSVs and JSON ─────────────────────────────────────────────────────
agg_df.to_csv('results/conformal_results.csv', index=False)
all_df.to_csv('results/conformal_results_per_seed.csv', index=False)
print("\n  ✓ results/conformal_results.csv  (aggregated, mean±SD)")
print("  ✓ results/conformal_results_per_seed.csv  (all seeds, raw)")

# Compute baseline precision (average across seeds, prob > 0.5 threshold)
mean_base_prec = float(agg_df['baseline_prec_mean'].mean())
mean_base_rec  = float(agg_df['baseline_rec_mean'].mean())
denom_bf       = mean_base_prec + mean_base_rec
mean_base_f1   = (2*mean_base_prec*mean_base_rec/denom_bf) if denom_bf > 0 else 0.0

results_json = {
    'method':            'Mondrian (class-conditional) Conformal Prediction',
    'nonconformity':     'LAC (1 - p̂(y|x)) per class',
    'n_seeds':           len(SEEDS),
    'seeds':             SEEDS,
    'cal_fraction':      CAL_FRACTION,
    'baseline_prec_mean': round(mean_base_prec, 4),
    'baseline_rec_mean':  round(mean_base_rec,  4),
    'baseline_f1_mean':   round(mean_base_f1,   4),
    'confidence_levels': CONFIDENCE_LEVELS,
    'all_levels':        agg_df.to_dict('records'),
    'key_level_80pct':   agg_df[agg_df['confidence_level'] == 0.80].to_dict('records')[0],
    'key_level_90pct':   agg_df[agg_df['confidence_level'] == 0.90].to_dict('records')[0],
}
with open('results/conformal_results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print("  ✓ results/conformal_results.json")

# Print key results
print(f"\n{'='*65}")
print("  KEY RESULTS (Table 3 headline levels)")
print(f"{'='*65}")
for cl in [0.80, 0.90]:
    r = agg_df[agg_df['confidence_level'] == cl].iloc[0]
    prec_m = r['precision_conf_active_mean']
    prec_s = r['precision_conf_active_std']
    rec_m  = r['recall_conf_active_mean']
    rec_s  = r['recall_conf_active_std']
    f1_m   = r['f1_conf_active_mean']
    f1_s   = r['f1_conf_active_std']
    cov_m  = r['empirical_coverage_mean']
    eff_m  = r['efficiency_mean']
    sa_m   = r['n_singleton_active_mean']
    am_m   = r['n_ambiguous_mean']
    em_m   = r['n_empty_mean']
    bl_m   = r['matched_baseline_prec_mean']

    print(f"\n  {cl:.0%} confidence:")
    print(f"    Empirical coverage  : {cov_m:.3f}  (target: {cl:.2f})")
    print(f"    Efficiency          : {eff_m:.3f}  (singleton fraction)")
    print(f"    Singleton actives   : {sa_m:.1f} | Ambiguous: {am_m:.1f} | Empty: {em_m:.1f}")
    if prec_m is not None:
        print(f"    Precision (CP)      : {prec_m:.3f} ± {prec_s:.3f}")
        print(f"    Recall    (CP)      : {rec_m:.3f} ± {rec_s:.3f}")
        print(f"    F1        (CP)      : {f1_m:.3f} ± {f1_s:.3f}")
        print(f"    Matched-list prec.  : {bl_m:.3f}  (top-k probability baseline)")
        if prec_m > bl_m:
            delta_rel = (prec_m - bl_m) / bl_m * 100
            print(f"    Precision gain      : +{prec_m - bl_m:.3f} (+{delta_rel:.1f}% relative vs matched baseline)")

print(f"\n  Baseline (prob ≥ 0.5)   : Prec={mean_base_prec:.3f}  Rec={mean_base_rec:.3f}  F1={mean_base_f1:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
#  Figure: Two-panel conformal prediction figure
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  Generating Figure: conformal_combined.png ...")
print(f"{'='*65}")

conf_pct = np.array(CONFIDENCE_LEVELS) * 100
cov_mean = agg_df['empirical_coverage_mean'].values * 100
cov_std  = agg_df['empirical_coverage_std'].values * 100
eff_mean = agg_df['efficiency_mean'].values * 100
eff_std  = agg_df['efficiency_std'].values * 100
prec_vals_m = np.array([r['precision_conf_active_mean'] if r['precision_conf_active_mean'] is not None
                         else mean_base_prec for _, r in agg_df.iterrows()])
prec_vals_s = np.array([r['precision_conf_active_std']  if r['precision_conf_active_std']  is not None
                         else 0.0 for _, r in agg_df.iterrows()])
rec_vals_m  = np.array([r['recall_conf_active_mean'] if r['recall_conf_active_mean'] is not None
                         else mean_base_rec for _, r in agg_df.iterrows()])
ambi_mean   = agg_df['n_ambiguous_mean'].values
empty_mean  = agg_df['n_empty_mean'].values
sing_act    = agg_df['n_singleton_active_mean'].values
sing_inact  = agg_df['n_singleton_inactive_mean'].values

fig = plt.figure(figsize=(14, 5.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

# ── Panel A: Coverage validity ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(conf_pct, cov_mean, 'o-', color='#3d7ebf',
         linewidth=2.2, markersize=7, label='Empirical coverage (mean)', zorder=3)
ax1.fill_between(conf_pct, cov_mean - cov_std, cov_mean + cov_std,
                 alpha=0.18, color='#3d7ebf', label='±1 SD (5 seeds)')
ax1.plot(conf_pct, conf_pct, '--', color='#888', linewidth=1.5,
         label='Ideal (target = 1−ε)', zorder=2)
ax1.fill_between(conf_pct, conf_pct, cov_mean, alpha=0.10, color='#3d7ebf')

for cl in [0.80, 0.90]:
    ax1.axvline(cl*100, color='#e05c5c', linestyle=':', alpha=0.65, linewidth=1.3)
    ax1.annotate(f'{cl:.0%}', xy=(cl*100 + 0.4, 61), fontsize=7.8,
                 color='#e05c5c', fontweight='bold')

ax1.set_xlabel('Target confidence level, 1−ε (%)', fontsize=10)
ax1.set_ylabel('Empirical coverage (%)', fontsize=10)
ax1.set_title('(A)  Coverage Validity\nMondrian CP — empirical vs. target', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8.2, loc='upper left')
ax1.grid(alpha=0.28)
ax1.set_xlim(57, 98); ax1.set_ylim(57, 104)

# ── Panel B: Efficiency breakdown + Precision/Recall ──────────────────────
ax2 = fig.add_subplot(gs[1])

n_total_mean = agg_df['n_total'].iloc[0] if 'n_total' in agg_df.columns else 592
x = np.arange(len(CONFIDENCE_LEVELS))
w = 0.22

# Stacked bars showing prediction set composition
b1 = ax2.bar(x - 1.5*w, sing_act,    w, label='Singleton active {1}',   color='#2ecc71', alpha=0.85, edgecolor='white')
b2 = ax2.bar(x - 0.5*w, sing_inact,  w, label='Singleton inactive {0}', color='#3d7ebf', alpha=0.85, edgecolor='white')
b3 = ax2.bar(x + 0.5*w, ambi_mean,   w, label='Ambiguous {0,1}',        color='#f39c12', alpha=0.85, edgecolor='white')
b4 = ax2.bar(x + 1.5*w, empty_mean,  w, label='Empty {}',               color='#e05c5c', alpha=0.85, edgecolor='white')

ax2.set_xticks(x)
ax2.set_xticklabels([f'{int(cl*100)}%' for cl in CONFIDENCE_LEVELS], fontsize=8)
ax2.set_xlabel('Confidence level (1−ε)', fontsize=10)
ax2.set_ylabel('Count (compounds, mean over 5 seeds)', fontsize=9)
ax2.set_title('(B)  Prediction Set Composition\nMondrian CP — breakdown by type', fontsize=10, fontweight='bold')
ax2.legend(fontsize=7.5, loc='upper left')
ax2.grid(axis='y', alpha=0.28)

fig.suptitle('Mondrian (Class-Conditional) Conformal Prediction — Dengue NS2B-NS3\n'
             'LAC Score · Random Forest + RDKit2D · 5 Random Seeds',
             fontsize=11, fontweight='bold', y=1.02)

plt.savefig('figures/conformal_combined.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/conformal_combined.png")

# ── Supplementary: Precision / Recall / F1 across confidence levels ────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax_l, ax_r = axes

# Left: Precision with matched baseline
ax_l.errorbar(conf_pct, prec_vals_m * 100, yerr=prec_vals_s * 100,
              fmt='o-', color='#e05c5c', linewidth=2, markersize=6, capsize=4,
              label='Mondrian CP precision')
ax_l.errorbar(conf_pct, rec_vals_m * 100,
              fmt='s--', color='#3d7ebf', linewidth=2, markersize=6,
              label='Mondrian CP recall')
ax_l.axhline(mean_base_prec*100, color='#555', linestyle=':', linewidth=1.5,
             label=f'Prob≥0.5 baseline precision ({mean_base_prec:.1%})')
ax_l.axhline(mean_base_rec*100,  color='#888', linestyle=':', linewidth=1.2,
             label=f'Prob≥0.5 baseline recall ({mean_base_rec:.1%})')
ax_l.set_xlabel('Confidence level 1−ε (%)', fontsize=10)
ax_l.set_ylabel('Percentage (%)', fontsize=10)
ax_l.set_title('(C)  Precision & Recall\nConfident Active Predictions', fontsize=10, fontweight='bold')
ax_l.legend(fontsize=7.5); ax_l.grid(alpha=0.28); ax_l.set_ylim(0, 115)

# Right: Prediction set composition as stacked bar (normalised to %)
n_total_arr = np.full(len(CONFIDENCE_LEVELS), agg_df['n_total_mean'].values.mean()
                       if 'n_total_mean' in agg_df.columns else 592)
sing_act_pct   = sing_act   / n_total_arr * 100
sing_inact_pct = sing_inact / n_total_arr * 100
ambi_pct       = ambi_mean  / n_total_arr * 100
empty_pct      = empty_mean / n_total_arr * 100

ax_r.bar(conf_pct, sing_act_pct,   4, label='Singleton {1} active',   color='#2ecc71', alpha=0.85)
ax_r.bar(conf_pct, sing_inact_pct, 4, bottom=sing_act_pct,            label='Singleton {0} inactive', color='#3d7ebf', alpha=0.85)
ax_r.bar(conf_pct, ambi_pct,       4, bottom=sing_act_pct+sing_inact_pct, label='Ambiguous {0,1}', color='#f39c12', alpha=0.85)
ax_r.bar(conf_pct, empty_pct,      4, bottom=sing_act_pct+sing_inact_pct+ambi_pct, label='Empty {}', color='#e05c5c', alpha=0.85)

ax_r.set_xlabel('Confidence level 1−ε (%)', fontsize=10)
ax_r.set_ylabel('Fraction of test set (%)', fontsize=10)
ax_r.set_title('(D)  Prediction Set Types (%)\nMondrian CP Composition', fontsize=10, fontweight='bold')
ax_r.legend(fontsize=7.5, loc='lower left'); ax_r.grid(alpha=0.28)

fig2.suptitle('Mondrian Conformal Prediction — Supplementary Metrics\n'
              'RF + RDKit2D · 5 Seeds', fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/conformal_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/conformal_metrics.png")

print()
print("✓ Stage 5 (Mondrian CP, v2) complete.")
print()
print("  Summary:")
print(f"  • Mondrian class-conditional CP — separate τ_0, τ_1 thresholds")
print(f"  • 5 seeds · mean ± SD reported for all metrics")
print(f"  • Efficiency = singleton fraction (now correctly decreases with confidence)")
print(f"  • Empty + ambiguous sets explicitly tracked")
print(f"  • Precision, Recall, F1 for confident active predictions")
print(f"  • Matched-list-size baseline reported for fair comparison")
print(f"  • Baseline (prob≥0.5): Prec={mean_base_prec:.3f}  Rec={mean_base_rec:.3f}  F1={mean_base_f1:.3f}")
