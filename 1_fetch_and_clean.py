import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
RAW_CSV    = 'data/chembl_dengue_raw.csv'
CLEAN_CSV  = 'data/chembl_dengue_clean.csv'
STATS_JSON = 'data/dataset_stats.json'

TARGET_ID              = 'CHEMBL5980'
ACTIVITY_THRESHOLD_NM  = 10_000

print("=" * 60)
print("STEP 1: Fetching ChEMBL data")
print("=" * 60)

df_raw = None

try:
    import chembl_downloader
    print(f"  Using chembl_downloader to query {TARGET_ID}...")
    query = f"""
        SELECT
            act.molregno,
            cs.canonical_smiles,
            act.standard_type,
            act.standard_value,
            act.standard_units,
            act.pchembl_value,
            act.activity_comment,
            cp.full_mwt,
            cp.alogp,
            cp.hba,
            cp.hbd,
            cp.rtb,
            cp.psa,
            ass.assay_type,
            ass.description AS assay_description
        FROM activities act
        JOIN compound_structures cs  ON act.molregno  = cs.molregno
        JOIN compound_properties cp  ON act.molregno  = cp.molregno
        JOIN assays ass               ON act.assay_id  = ass.assay_id
        WHERE ass.tid = (
            SELECT tid FROM target_dictionary WHERE chembl_id = '{TARGET_ID}'
        )
        AND act.standard_type IN ('IC50', 'Ki', 'Inhibition')
        AND act.standard_units = 'nM'
        AND act.standard_value IS NOT NULL
        AND cs.canonical_smiles IS NOT NULL
    """
    df_raw = chembl_downloader.query(query)
    print(f"  ✓ Raw records fetched: {len(df_raw)}")
    df_raw.to_csv(RAW_CSV, index=False)

except Exception as e:
    print(f"  chembl_downloader failed: {e}")
    print("  Falling back to chembl-webresource-client...")

if df_raw is None or len(df_raw) == 0:
    try:
        from chembl_webresource_client.new_client import new_client
        activity = new_client.activity
        print(f"  Fetching activities for target {TARGET_ID}...")
        res = activity.filter(
            target_chembl_id=TARGET_ID,
            standard_type__in=['IC50', 'Ki', 'Inhibition'],
            standard_units='nM',
        ).only([
            'molecule_chembl_id', 'canonical_smiles', 'standard_type',
            'standard_value', 'standard_units', 'pchembl_value',
            'activity_comment', 'assay_type'
        ])
        df_raw = pd.DataFrame(list(res))
        print(f"  ✓ Raw records fetched: {len(df_raw)}")
        df_raw.to_csv(RAW_CSV, index=False)

    except Exception as e2:
        print(f"  chembl-webresource-client also failed: {e2}")
        sys.exit(1)

print("\n" + "=" * 60)
print("STEP 2: Filtering to IC50 measurements")
print("=" * 60)

print(f"  Records by standard_type:")
if 'standard_type' in df_raw.columns:
    print(df_raw['standard_type'].value_counts().to_string())

df = df_raw[df_raw['standard_type'] == 'IC50'].copy()
print(f"\n  After IC50 filter: {len(df)} records")

df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')

smiles_col = 'canonical_smiles'
df = df.dropna(subset=['standard_value', smiles_col])
print(f"  After dropping missing values: {len(df)} records")

print("\n" + "=" * 60)
print("STEP 2b: Removing non-protease assay contamination")
print("=" * 60)

before_filter = len(df)
if 'assay_description' in df.columns:
    desc = df['assay_description'].fillna('')

    is_protease = (
        desc.str.contains('NS2B|NS3 protease|NS2B-NS3|NS2B/NS3|NS3pro|serine protease|NS3.*protease|dengue.*protease|protease.*dengue',
                          case=False, regex=True)
        & ~desc.str.contains('helicase|capping|methyltransfer|MTase|Capping Enzyme|NS5',
                             case=False, regex=True)
    )

    df = df[is_protease].copy()
    removed = before_filter - len(df)
    print(f"  Records before filter : {before_filter:,}")
    print(f"  Removed (non-protease): {removed:,}  (NS5-MTase capping + NS3 helicase)")
    print(f"  Records after filter  : {len(df):,}")
else:
    print("  Warning: no assay_description column — skipping contamination filter")

print("\n" + "=" * 60)
print("STEP 3: Activity labeling")
print("=" * 60)

df['activity'] = (df['standard_value'] <= ACTIVITY_THRESHOLD_NM).astype(int)
print(f"  Threshold: IC50 ≤ {ACTIVITY_THRESHOLD_NM:,} nM → active (1)")
print(f"  Active:   {df['activity'].sum():>5d} ({df['activity'].mean():.1%})")
print(f"  Inactive: {(df['activity']==0).sum():>5d} ({1-df['activity'].mean():.1%})")

print("\n" + "=" * 60)
print("STEP 4: SMILES validation")
print("=" * 60)

def is_valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return mol is not None
    except:
        return False

before = len(df)
df = df[df[smiles_col].apply(is_valid_smiles)].copy()
print(f"  Invalid SMILES removed: {before - len(df)}")
print(f"  Valid compounds: {len(df)}")

print("\n" + "=" * 60)
print("STEP 5: Deduplication")
print("=" * 60)

before = len(df)

def canonicalize(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        return Chem.MolToSmiles(mol)
    return smi

print("  Canonicalizing SMILES...")
df['canonical_smiles'] = df[smiles_col].apply(canonicalize)

df_dedup = (df.groupby('canonical_smiles')
              .agg(
                  ic50_median=('standard_value', 'median'),
                  ic50_min=('standard_value', 'min'),
                  ic50_max=('standard_value', 'max'),
                  n_measurements=('standard_value', 'count'),
                  activity_mean=('activity', 'mean'),
              )
              .reset_index())

df_dedup['activity']  = (df_dedup['ic50_median'] <= ACTIVITY_THRESHOLD_NM).astype(int)
df_dedup['pIC50']     = -np.log10(df_dedup['ic50_median'] * 1e-9)

print(f"  Before dedup: {before} records")
print(f"  After dedup:  {len(df_dedup)} unique compounds")
print(f"  Active:   {df_dedup['activity'].sum():>5d} ({df_dedup['activity'].mean():.1%})")
print(f"  Inactive: {(df_dedup['activity']==0).sum():>5d} ({1-df_dedup['activity'].mean():.1%})")

print("\n" + "=" * 60)
print("STEP 6: Lipinski rule-of-five flagging")
print("=" * 60)

def lipinski_filter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, np.nan, np.nan, np.nan, np.nan
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    passes = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    return passes, mw, logp, hbd, hba

lipinski_results = df_dedup['canonical_smiles'].apply(lipinski_filter)
df_dedup['lipinski_pass'] = [r[0] for r in lipinski_results]
df_dedup['mw']            = [r[1] for r in lipinski_results]
df_dedup['logp']          = [r[2] for r in lipinski_results]
df_dedup['hbd']           = [r[3] for r in lipinski_results]
df_dedup['hba']           = [r[4] for r in lipinski_results]

n_pass = df_dedup['lipinski_pass'].sum()
print(f"  Lipinski pass: {n_pass} / {len(df_dedup)} ({n_pass/len(df_dedup):.1%})")
print(f"  (All compounds KEPT — natural products often violate Lipinski)")

print("\n" + "=" * 60)
print("STEP 7: Saving clean dataset")
print("=" * 60)

df_dedup.to_csv(CLEAN_CSV, index=False)
print(f"  ✓ Saved to {CLEAN_CSV}")

print("\n" + "=" * 60)
print("DATASET SUMMARY (Table 1 data)")
print("=" * 60)

stats = {
    'total_compounds': int(len(df_dedup)),
    'n_active': int(df_dedup['activity'].sum()),
    'n_inactive': int((df_dedup['activity'] == 0).sum()),
    'class_balance_active_pct': float(round(df_dedup['activity'].mean() * 100, 1)),
    'lipinski_pass': int(df_dedup['lipinski_pass'].sum()),
    'lipinski_pass_pct': float(round(n_pass / len(df_dedup) * 100, 1)),
    'activity_threshold_nm': ACTIVITY_THRESHOLD_NM,
    'median_ic50_active': float(round(df_dedup[df_dedup['activity']==1]['ic50_median'].median(), 1)),
    'median_ic50_inactive': float(round(df_dedup[df_dedup['activity']==0]['ic50_median'].median(), 1)),
    'mw_mean': float(round(df_dedup['mw'].mean(), 1)),
    'logp_mean': float(round(df_dedup['logp'].mean(), 2)),
}

import json
with open(STATS_JSON, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"""
  Total unique compounds : {stats['total_compounds']:,}
  Active (IC50 ≤ 10 µM)  : {stats['n_active']:,} ({stats['class_balance_active_pct']}%)
  Inactive               : {stats['n_inactive']:,} ({100 - stats['class_balance_active_pct']}%)
  Lipinski pass          : {stats['lipinski_pass']:,} ({stats['lipinski_pass_pct']}%)
  Median IC50 (active)   : {stats['median_ic50_active']:,.1f} nM
  Median IC50 (inactive) : {stats['median_ic50_inactive']:,.1f} nM
  Mean MW                : {stats['mw_mean']} Da
  Mean LogP              : {stats['logp_mean']}

  Stats saved to {STATS_JSON}
""")
