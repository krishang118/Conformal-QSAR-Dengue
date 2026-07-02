import os, pickle, warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)

print("=" * 60)
print("Loading clean dataset...")
print("=" * 60)
df = pd.read_csv('data/chembl_dengue_clean.csv')
smiles_list = df['canonical_smiles'].tolist()
y = df['activity'].values
print(f"  Compounds : {len(df):,}")
print(f"  Active    : {y.sum():,} ({y.mean():.1%})")
print(f"  Inactive  : {(1-y).sum():,} ({1-y.mean():.1%})")

print("\nParsing SMILES → RDKit mol objects...")
mols = [Chem.MolFromSmiles(s) for s in smiles_list]
assert all(m is not None for m in mols), "Some SMILES failed to parse!"
print(f"  ✓ All {len(mols):,} molecules parsed")

print("\n" + "=" * 60)
print("Fingerprint 1: ECFP4 / Morgan (radius=2, 2048 bits)")
print("=" * 60)

def ecfp4(mol, n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return list(fp)

X_ecfp4 = np.array([ecfp4(m) for m in tqdm(mols, desc="  ECFP4")], dtype=np.float32)
print(f"  Shape: {X_ecfp4.shape}  |  Sparsity: {(X_ecfp4==0).mean():.1%} zeros")

print("\n" + "=" * 60)
print("Fingerprint 2: MACCS keys (167 bits)")
print("=" * 60)

def maccs(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    return list(fp)

X_maccs = np.array([maccs(m) for m in tqdm(mols, desc="  MACCS")], dtype=np.float32)
print(f"  Shape: {X_maccs.shape}  |  Sparsity: {(X_maccs==0).mean():.1%} zeros")

print("\n" + "=" * 60)
print("Fingerprint 3: RDKit 2D descriptors (physicochemical)")
print("=" * 60)

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]

def rdkit_descriptors(mol):
    vals = []
    for _, fn in Descriptors.descList:
        try:
            v = fn(mol)
            vals.append(v if np.isfinite(v) else 0.0)
        except:
            vals.append(0.0)
    return vals

X_rdkit_raw = np.array(
    [rdkit_descriptors(m) for m in tqdm(mols, desc="  RDKit2D")],
    dtype=np.float32
)

X_rdkit_raw = np.nan_to_num(X_rdkit_raw, nan=0.0, posinf=0.0, neginf=0.0)

var = X_rdkit_raw.var(axis=0)
nonzero_var_mask = var > 0
X_rdkit_raw = X_rdkit_raw[:, nonzero_var_mask]
DESCRIPTOR_NAMES_FILTERED = [n for n, keep in zip(DESCRIPTOR_NAMES, nonzero_var_mask) if keep]

scaler = StandardScaler()
X_rdkit = scaler.fit_transform(X_rdkit_raw).astype(np.float32)

print(f"  Total descriptors      : {len(DESCRIPTOR_NAMES)}")
print(f"  After zero-var removal : {X_rdkit.shape[1]}")
print(f"  Final shape            : {X_rdkit.shape}")

print("\n" + "=" * 60)
print("Fingerprint 4: Atom-pair fingerprints (2048 bits)")
print("=" * 60)

_ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

def atom_pair(mol):
    fp = _ap_gen.GetFingerprint(mol)
    return list(fp)

X_ap = np.array([atom_pair(m) for m in tqdm(mols, desc="  AtomPair")], dtype=np.float32)
print(f"  Shape: {X_ap.shape}  |  Sparsity: {(X_ap==0).mean():.1%} zeros")

print("\n" + "=" * 60)
print("Saving feature matrices...")
print("=" * 60)

features = {
    'ecfp4':                X_ecfp4,
    'maccs':                X_maccs,
    'rdkit':                X_rdkit,
    'atompair':             X_ap,
    'y':                    y,
    'smiles':               smiles_list,
    'descriptor_names':     DESCRIPTOR_NAMES_FILTERED,
    'rdkit_scaler':         scaler,
    'n_active':             int(y.sum()),
    'n_inactive':           int((1-y).sum()),
}

with open('data/features.pkl', 'wb') as f:
    pickle.dump(features, f, protocol=4)

size_mb = os.path.getsize('data/features.pkl') / 1e6
print(f"  ✓ Saved to data/features.pkl  ({size_mb:.1f} MB)")

print("\n" + "=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)
print(f"  ECFP4      : {X_ecfp4.shape}   (Morgan circular, r=2)")
print(f"  MACCS      : {X_maccs.shape}   (structural keys dictionary)")
print(f"  RDKit 2D   : {X_rdkit.shape}   (physicochemical, standardised)")
print(f"  Atom-pair  : {X_ap.shape}   (pairwise atom topology)")
print(f"  Labels (y) : {y.shape}   ({y.sum()} active, {(1-y).sum()} inactive)")
