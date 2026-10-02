"""Mordred 2D descriptors from SMILES only -> mordred.npz (tr, pr, names). Cleaning (const/NaN-heavy cols) uses no labels."""
import numpy as np, pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
if __name__ == '__main__':
    D = np.load('features.npz', allow_pickle=True)
    smi = list(D['smiles_tr']) + list(D['smiles_pr'])
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas([Chem.MolFromSmiles(s) for s in smi], nproc=4, quiet=True)
    A = df.apply(pd.to_numeric, errors='coerce').astype(np.float64).replace([np.inf, -np.inf], np.nan)
    A = A.loc[:, A.isna().mean() < 0.05]
    A = A.loc[:, A.nunique() > 1]
    A = A.clip(-1e6, 1e6)
    n = len(D['smiles_tr']); X = A.to_numpy(np.float32)
    np.savez('mordred.npz', tr=X[:n], pr=X[n:], names=np.array(A.columns.astype(str)))
    print(X.shape, np.isnan(X).mean())
