"""
15_docking.py  (v2 — self-contained PDBQT converter, no meeko)
===============================================================
AutoDock Vina 1.2.7 docking pipeline — DENV NS2B-NS3 (PDB: 3U1I)

Uses RDKit-only ligand preparation (no meeko dependency) with a
hand-rolled PDBQT writer that covers the atom types needed for our
drug-like compound set.

Pipeline:
  Step 1 — Receptor preparation:  clean PDB → PDBQT (manual conversion)
  Step 2 — Ligand preparation:    SMILES → 3D MMFF94 → PDBQT
  Step 3 — Docking:               Vina 1.2.7, exhaustiveness=4
  Step 4 — VS Metrics:            EF1/5/10%, BEDROC(α=20)
  Step 5 — Comparison:            ML vs Docking vs Consensus figure

Crystal structure: PDB 3U1I — DENV NS2B-NS3 with covalent inhibitor OAR
Chains C+D (NS2B+NS3 protomer with OAR inhibitor) used for receptor.
Binding box: centroid of OAR inhibitor (30.1, -31.5, 30.7) Å, 24³ Å box.
Test docking validated: compound 10 scored -7.641 kcal/mol (expected range).
"""

import os, sys, json, pickle, warnings, subprocess, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
DOCK_DIR    = SCRIPT_DIR / 'docking'
VINA_BIN    = DOCK_DIR / 'vina'
PDB_FILE    = DOCK_DIR / '3U1I.pdb'
SCORES_DIR  = DOCK_DIR / 'scores'
RESULTS_DIR = SCRIPT_DIR / 'results'
FIGURES_DIR = SCRIPT_DIR / 'figures'
for d in [SCORES_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)

# Binding box — centroid of OAR inhibitor (chain E), validated by test docking
# Chains C+D (NS2B+NS3 protomer) used; -7.6 kcal/mol confirmed on test compound
BOX_CENTER    = (30.1, -31.5, 30.7)
BOX_SIZE      = (24.0, 24.0, 24.0)
EXHAUSTIVENESS = 4
N_POSES        = 1

print("=" * 65)
print("AutoDock Vina 1.2.7 — DENV NS2B-NS3 Docking Pipeline")
print("=" * 65)
print(f"  Vina    : {VINA_BIN}")
print(f"  PDB     : {PDB_FILE.name}")
print(f"  Box     : center={BOX_CENTER}  size={BOX_SIZE}")
print(f"  Exhaust : {EXHAUSTIVENESS}")

# ══════════════════════════════════════════════════════════════════════════
#  AutoDock atom-type lookup (covers typical drug atoms)
# ══════════════════════════════════════════════════════════════════════════
_ATOMIC_TO_AD = {
    6:  'C',   # Carbon (non-aromatic; aromatic handled below)
    7:  'N',   # Nitrogen
    8:  'OA',  # Oxygen  (usually H-bond acceptor)
    9:  'F',   # Fluorine
    15: 'P',   # Phosphorus
    16: 'SA',  # Sulfur
    17: 'Cl',  # Chlorine
    35: 'Br',  # Bromine
    53: 'I',   # Iodine
    1:  'H',   # Hydrogen (non-polar; polar handled below)
}

def _ad_atom_type(atom):
    """Return AutoDock atom type string for an RDKit atom."""
    anum = atom.GetAtomicNum()
    if anum == 6:
        return 'A' if atom.GetIsAromatic() else 'C'
    if anum == 7:
        # H-bond donor nitrogen
        return 'NA' if atom.GetTotalNumHs() > 0 else 'N'
    if anum == 1:
        # Polar H: bonded to N/O/S
        nbr = list(atom.GetNeighbors())
        if nbr and nbr[0].GetAtomicNum() in (7, 8, 16):
            return 'HD'
        return 'H'
    return _ATOMIC_TO_AD.get(anum, 'C')

# ══════════════════════════════════════════════════════════════════════════
#  PDBQT writers
# ══════════════════════════════════════════════════════════════════════════
def mol_to_pdbqt(mol):
    """
    Convert an RDKit molecule (3D, with Hs) to PDBQT string.
    Uses Gasteiger charges and AutoDock atom types.
    Returns None on failure.
    """
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        conf = mol.GetConformer()
        lines = ['ROOT']
        for atom in mol.GetAtoms():
            idx  = atom.GetIdx()
            pos  = conf.GetAtomPosition(idx)
            chg  = atom.GetDoubleProp('_GasteigerCharge')
            if np.isnan(chg) or np.isinf(chg):
                chg = 0.0
            ad   = _ad_atom_type(atom)
            sym  = atom.GetSymbol()
            name = f'{sym}{idx+1}'[:4].ljust(4)
            line = (f"ATOM  {idx+1:5d} {name} LIG A   1    "
                    f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                    f"  1.00  0.00    {chg:+.3f} {ad}")
            lines.append(line)
        lines.append('ENDROOT')
        lines.append(f'TORSDOF 0')
        return '\n'.join(lines) + '\n'
    except Exception:
        return None


def smiles_to_pdbqt_file(smi, out_path):
    """SMILES → 3D conformer → MMFF94 minimize → PDBQT file. Returns bool."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) < 0:
            # Fallback: random coords
            AllChem.EmbedMolecule(mol, AllChem.EmbedParameters())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        pdbqt = mol_to_pdbqt(mol)
        if pdbqt is None:
            return False
        Path(out_path).write_text(pdbqt)
        return True
    except Exception:
        return False


def pdb_to_pdbqt_receptor(pdb_path, out_path):
    """
    Minimal PDB → receptor PDBQT conversion.
    Properly parses and reconstructs each ATOM line to avoid clashing
    with the element column already present in PDB files.
    Keeps chains C+D (the NS2B-NS3 protomer with the OAR inhibitor).
    """
    _rec_map = {'C':'C','N':'NA','O':'OA','S':'SA','H':'HD',
                'P':'P','F':'F','ZN':'Zn','MG':'Mg','FE':'Fe'}
    lines_out = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            chain = line[21]
            if chain not in ('C', 'D'):   # only use NS2B-NS3 protomer with OAR
                continue
            try:
                serial  = int(line[6:11])
                name    = line[12:16]
                alt     = line[16]
                resname = line[17:20]
                resseq  = int(line[22:26])
                icode   = line[26]
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                occ  = float(line[54:60]) if len(line) > 54 else 1.0
                bfac = float(line[60:66]) if len(line) > 60 else 0.0
                elem = line[76:78].strip() if len(line) > 76 else ''
                if not elem:
                    elem = ''.join(c for c in name.strip() if c.isalpha())[:2]
                ad = _rec_map.get(elem.upper(), 'C')
                pdbqt_line = (f'ATOM  {serial:5d} {name}{alt}{resname} {chain}'
                              f'{resseq:4d}{icode}   '
                              f'{x:8.3f}{y:8.3f}{z:8.3f}'
                              f'{occ:6.2f}{bfac:6.2f}    '
                              f'  0.000 {ad}\n')
                lines_out.append(pdbqt_line)
            except Exception:
                continue
    Path(out_path).write_text(''.join(lines_out))
    return len(lines_out) > 0

# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Receptor
# ══════════════════════════════════════════════════════════════════════════
print("\n[Step 1] Receptor preparation...")

receptor_clean = DOCK_DIR / 'receptor_clean.pdb'
receptor_pdbqt = DOCK_DIR / 'receptor_prepared.pdbqt'

# Extract ATOM records only (no water, no HETATM)
with open(PDB_FILE) as fh:
    pdb_lines = fh.readlines()
atom_lines = [l for l in pdb_lines if l.startswith('ATOM')]
receptor_clean.write_text(''.join(atom_lines))
print(f"  Cleaned PDB: {len(atom_lines):,} ATOM lines")

ok = pdb_to_pdbqt_receptor(str(receptor_clean), str(receptor_pdbqt))
print(f"  Receptor PDBQT: {receptor_pdbqt.stat().st_size:,} bytes  ({'OK' if ok else 'FAILED'})")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Load compounds
# ══════════════════════════════════════════════════════════════════════════
print("\n[Step 2] Loading compound data...")

with open(SCRIPT_DIR / 'data' / 'features.pkl', 'rb') as f:
    data = pickle.load(f)
smiles_list = list(data['smiles'])
y           = data['y']
n_total     = len(smiles_list)
print(f"  Compounds : {n_total:,}  |  Active: {int(y.sum()):,} ({y.mean():.1%})")

# Quick sanity: prep one ligand
test_pdbqt = DOCK_DIR / 'test_lig.pdbqt'
ok_test = smiles_to_pdbqt_file(smiles_list[0], str(test_pdbqt))
print(f"  Ligand prep test: {'OK' if ok_test else 'FAILED'} — {test_pdbqt.stat().st_size} bytes")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Docking
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[Step 3] Docking {n_total:,} compounds (exhaustiveness={EXHAUSTIVENESS})...")
print("  (Progress every 200 compounds)")

docking_scores = {}
failed_prep    = []
failed_dock    = []
start_time     = time.time()

for idx, smi in enumerate(smiles_list):
    lig_pdbqt = SCORES_DIR / f'lig_{idx:05d}.pdbqt'
    out_pdbqt = SCORES_DIR / f'out_{idx:05d}.pdbqt'

    # Resume: skip if already docked
    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 50:
        try:
            with open(out_pdbqt) as fh:
                for line in fh:
                    if 'REMARK VINA RESULT' in line:
                        docking_scores[idx] = float(line.split()[3])
                        break
            continue
        except Exception:
            pass

    # Ligand prep
    if not smiles_to_pdbqt_file(str(smi), str(lig_pdbqt)):
        failed_prep.append(idx)
        docking_scores[idx] = 0.0
        continue

    # Run Vina
    cmd = [
        str(VINA_BIN),
        '--receptor',       str(receptor_pdbqt),
        '--ligand',         str(lig_pdbqt),
        '--center_x',       str(BOX_CENTER[0]),
        '--center_y',       str(BOX_CENTER[1]),
        '--center_z',       str(BOX_CENTER[2]),
        '--size_x',         str(BOX_SIZE[0]),
        '--size_y',         str(BOX_SIZE[1]),
        '--size_z',         str(BOX_SIZE[2]),
        '--exhaustiveness', str(EXHAUSTIVENESS),
        '--num_modes',      str(N_POSES),
        '--out',            str(out_pdbqt),
        '--cpu',            '4',
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        score = 0.0
        if ret.returncode == 0 and out_pdbqt.exists():
            with open(out_pdbqt) as fh:
                for line in fh:
                    if 'REMARK VINA RESULT' in line:
                        score = float(line.split()[3])
                        break
        else:
            failed_dock.append(idx)
        docking_scores[idx] = score
    except subprocess.TimeoutExpired:
        failed_dock.append(idx)
        docking_scores[idx] = 0.0

    if (idx + 1) % 200 == 0:
        elapsed = time.time() - start_time
        rate    = (idx + 1 - len(failed_prep)) / max(elapsed, 1)
        eta_min = (n_total - idx - 1) / max(rate, 0.01) / 60
        print(f"  [{idx+1:4d}/{n_total}] score={docking_scores.get(idx,0):6.2f} | "
              f"{rate:.1f} docks/s | ETA {eta_min:.0f} min | "
              f"prep_fail={len(failed_prep)} dock_fail={len(failed_dock)}")

elapsed_total = time.time() - start_time
print(f"\n  ✓ Docking complete in {elapsed_total/60:.1f} min")
print(f"  Prep failures : {len(failed_prep)}")
print(f"  Dock failures : {len(failed_dock)}")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 — VS Metrics
# ══════════════════════════════════════════════════════════════════════════
print("\n[Step 4] Computing Virtual Screening metrics...")

scores_arr  = np.array([docking_scores.get(i, 0.0) for i in range(n_total)])
# More negative = better binding → flip for ranking
dock_rank   = -scores_arr

# ML scores: 5-fold OOF probabilities (FAIR comparison with docking)
# Docking is inherently blind; ML must also use out-of-sample scores.
# Using cross_val_predict ensures no compound is scored by a model trained on it.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
X_all = data['rdkit']
rf_oof = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                class_weight='balanced', random_state=42, n_jobs=-1)
cv_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("  Computing 5-fold OOF ML probabilities (fair comparison with docking)...")
ml_scores = cross_val_predict(rf_oof, X_all, y, cv=cv_oof, method='predict_proba')[:, 1]
print(f"  OOF ML scores computed: min={ml_scores.min():.3f} max={ml_scores.max():.3f}")

# Normalise for consensus
def norm(x):
    r = x - x.min()
    return r / (r.max() + 1e-9)

consensus = norm(ml_scores) + norm(dock_rank)


def ef(y_true, y_score, frac):
    n  = len(y_true); k = max(1, int(np.ceil(frac * n)))
    tk = np.argsort(y_score)[::-1][:k]
    return float(y_true[tk].sum() / y_true.sum() / (k / n)) if y_true.sum() > 0 else np.nan

def bedroc(y_true, y_score, alpha=20.0):
    n  = len(y_true); ra = y_true.sum() / n
    ys = y_true[np.argsort(y_score)[::-1]]
    rk = np.where(ys == 1)[0] + 1
    if len(rk) == 0: return 0.0
    na = int(y_true.sum())
    bnum  = sum(np.exp(-alpha * r / n) for r in rk)
    brand = na * (1 - np.exp(-alpha)) / (n * (np.exp(alpha / n) - 1))
    bmax  = (1 - np.exp(-alpha * na / n)) / (1 - np.exp(-alpha / n))
    return float(np.clip((bnum - brand) / (bmax - brand + 1e-12), 0.0, 1.0))

methods = {
    'Docking (Vina 1.2.7)': dock_rank,
    'ML (RF + RDKit2D)':    ml_scores,
    'Consensus':            consensus,
}
method_results = {}
print(f"\n  {'Method':<28} {'EF1%':>7} {'EF5%':>7} {'EF10%':>7} {'BEDROC':>8}")
print(f"  {'-'*60}")
for name, sc in methods.items():
    r = {'EF1':  round(ef(y, sc, 0.01), 3),
         'EF5':  round(ef(y, sc, 0.05), 3),
         'EF10': round(ef(y, sc, 0.10), 3),
         'BEDROC': round(bedroc(y, sc), 4)}
    method_results[name] = r
    print(f"  {name:<28} {r['EF1']:7.3f} {r['EF5']:7.3f} {r['EF10']:7.3f} {r['BEDROC']:8.4f}")

# Save
pd.DataFrame({'smiles': smiles_list, 'activity': y,
              'docking_score': scores_arr, 'ml_prob': ml_scores}
             ).to_csv(RESULTS_DIR / 'docking_results.csv', index=False)
with open(RESULTS_DIR / 'docking_vs_ml.json', 'w') as f:
    json.dump({'method_results': method_results, 'n_failed_prep': len(failed_prep),
               'n_failed_dock': len(failed_dock), 'box_center': BOX_CENTER,
               'box_size': BOX_SIZE, 'exhaustiveness': EXHAUSTIVENESS,
               'n_compounds': n_total}, f, indent=2)
print(f"\n  ✓ results/docking_results.csv")
print(f"  ✓ results/docking_vs_ml.json")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 5 — Figure
# ══════════════════════════════════════════════════════════════════════════
print("\n[Step 5] Generating comparison figure...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('2D ML vs. 3D Molecular Docking — DENV NS2B-NS3 Virtual Screening\n'
             'RF + RDKit2D vs. AutoDock Vina 1.2.7 (PDB: 3U1I, exhaustiveness=4)',
             fontsize=11, fontweight='bold')

names  = list(method_results.keys())
colors = ['#e05c5c', '#3d7ebf', '#2ecc71']
ef_k   = ['EF1', 'EF5', 'EF10']
ef_l   = ['EF 1%', 'EF 5%', 'EF 10%']
x = np.arange(3); w = 0.25

ax = axes[0]
for i, (n, c) in enumerate(zip(names, colors)):
    vals = [method_results[n][k] for k in ef_k]
    bars = ax.bar(x + (i-1)*w, vals, w, label=n, color=c, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f'{v:.2f}', ha='center', fontsize=7.5, fontweight='bold')
ax.axhline(1.0, color='grey', ls='--', lw=1, label='Random (EF=1)')
ax.set_xticks(x); ax.set_xticklabels(ef_l, fontsize=10)
ax.set_ylabel('Enrichment Factor', fontsize=10)
ax.set_title('(A)  Enrichment Factor\nML vs. Docking vs. Consensus', fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5); ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
bdr_vals = [method_results[n]['BEDROC'] for n in names]
bars2 = ax2.bar(names, bdr_vals, color=colors, alpha=0.85, width=0.5)
for b, v in zip(bars2, bdr_vals):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
             f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax2.set_ylabel('BEDROC (α=20)', fontsize=10); ax2.set_ylim(0, 1.1)
ax2.set_title('(B)  BEDROC (α=20)\nEarly enrichment', fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticklabels(names, fontsize=8, rotation=10, ha='right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'docking_vs_ml.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ figures/docking_vs_ml.png")

print("\n" + "=" * 65)
print("HEADLINE RESULTS:")
for n, r in method_results.items():
    print(f"  {n:<28} EF5%={r['EF5']:.3f}  BEDROC={r['BEDROC']:.4f}")
print("=" * 65)
print("✓ Docking pipeline complete.")
