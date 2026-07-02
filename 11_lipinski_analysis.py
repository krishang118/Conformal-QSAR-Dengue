import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("Stage 11 — Lipinski Rule of Five (Drug-likeness) Analysis")
print("=" * 60)

df = pd.read_csv('data/chembl_dengue_clean.csv')
actives   = df[df['activity'] == 1].copy()
inactives = df[df['activity'] == 0].copy()
print(f"  Actives  : {len(actives):,}")
print(f"  Inactives: {len(inactives):,}")

def get_props(smiles_series):
    rows = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            rows.append({
                'MW':   Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD':  rdMolDescriptors.CalcNumHBD(mol),
                'HBA':  rdMolDescriptors.CalcNumHBA(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotB': rdMolDescriptors.CalcNumRotatableBonds(mol),
            })
        else:
            rows.append({'MW': np.nan, 'LogP': np.nan, 'HBD': np.nan,
                         'HBA': np.nan, 'TPSA': np.nan, 'RotB': np.nan})
    return pd.DataFrame(rows)

print("  Computing descriptors for actives...")
act_props = get_props(actives['canonical_smiles'])
print("  Computing descriptors for inactives...")
inact_props = get_props(inactives['canonical_smiles'])

def lipinski_pass(row):
    violations = 0
    if row['MW']  > 500: violations += 1
    if row['LogP'] > 5:  violations += 1
    if row['HBD']  > 5:  violations += 1
    if row['HBA']  > 10: violations += 1
    return violations <= 1

act_pass   = act_props.apply(lipinski_pass, axis=1)
inact_pass = inact_props.apply(lipinski_pass, axis=1)

act_pass_n   = act_pass.sum()
inact_pass_n = inact_pass.sum()

print(f"\n  Actives passing Ro5  : {act_pass_n}/{len(actives)} ({act_pass_n/len(actives):.1%})")
print(f"  Inactives passing Ro5: {inact_pass_n}/{len(inactives)} ({inact_pass_n/len(inactives):.1%})")

print("\n  Active compound descriptor statistics:")
for col in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotB']:
    vals = act_props[col].dropna()
    print(f"    {col:<6}: mean={vals.mean():.2f}  std={vals.std():.2f}  "
          f"[{vals.min():.1f} – {vals.max():.1f}]")

print("\n  Inactive compound descriptor statistics:")
for col in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotB']:
    vals = inact_props[col].dropna()
    print(f"    {col:<6}: mean={vals.mean():.2f}  std={vals.std():.2f}  "
          f"[{vals.min():.1f} – {vals.max():.1f}]")

results = {
    'n_actives':              int(len(actives)),
    'n_actives_pass_ro5':     int(act_pass_n),
    'pct_actives_pass_ro5':   round(act_pass_n/len(actives)*100, 1),
    'n_inactives':            int(len(inactives)),
    'n_inactives_pass_ro5':   int(inact_pass_n),
    'pct_inactives_pass_ro5': round(inact_pass_n/len(inactives)*100, 1),
    'active_descriptors': {
        col: {'mean': round(act_props[col].mean(),2), 'std': round(act_props[col].std(),2)}
        for col in ['MW','LogP','HBD','HBA','TPSA','RotB']
    }
}
with open('results/lipinski_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n  ✓ results/lipinski_results.json")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
props_to_plot = [
    ('MW',   'Molecular Weight (Da)', 500,  'MW ≤ 500'),
    ('LogP', 'LogP',                  5,    'LogP ≤ 5'),
    ('HBD',  'H-Bond Donors',         5,    'HBD ≤ 5'),
    ('HBA',  'H-Bond Acceptors',      10,   'HBA ≤ 10'),
]

for ax, (col, xlabel, threshold, rule) in zip(axes.flat, props_to_plot):
    a_vals = act_props[col].dropna()
    i_vals = inact_props[col].dropna()

    all_vals = pd.concat([a_vals, i_vals])
    bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 35)

    ax.hist(i_vals, bins=bins, alpha=0.55, color='#7f8c8d', label='Inactive', density=True)
    ax.hist(a_vals, bins=bins, alpha=0.75, color='#e74c3c', label='Active',   density=True)
    ax.axvline(threshold, color='navy', linewidth=1.8, linestyle='--',
               label=f'Ro5 limit ({rule})')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'{xlabel}\n(Actives: {a_vals.mean():.1f} ± {a_vals.std():.1f})', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

fig.suptitle(f'Lipinski Rule of Five — Drug-likeness of Active vs Inactive Compounds\n'
             f'{act_pass_n}/{len(actives)} actives ({act_pass_n/len(actives):.1%}) '
             f'and {inact_pass_n}/{len(inactives)} inactives ({inact_pass_n/len(inactives):.1%}) '
             f'pass all Ro5 criteria',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/lipinski_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
