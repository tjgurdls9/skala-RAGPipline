"""추가 fingerprint: atom-pair, topological torsion, RDKit, Avalon."""
import pathlib, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools
RDLogger.DisableLog('rdApp.*')
ROOT = pathlib.Path(__file__).resolve().parent
AP = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
TT = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
RD = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)

def run(smiles):
    n = len(smiles)
    ap = np.zeros((n, 1024), np.uint8); tt = np.zeros((n, 1024), np.uint8)
    rd = np.zeros((n, 1024), np.uint8); av = np.zeros((n, 1024), np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        ap[i] = AP.GetFingerprintAsNumPy(m); tt[i] = TT.GetFingerprintAsNumPy(m)
        rd[i] = RD.GetFingerprintAsNumPy(m)
        av[i] = np.array(pyAvalonTools.GetAvalonFP(m, 1024), np.uint8)
    return ap, tt, rd, av

D = np.load(ROOT / 'features.npz', allow_pickle=True)
out = {}
for tag in ('tr', 'pr'):
    ap, tt, rd, av = run(list(D[f'smiles_{tag}']))
    out |= {f'ap_{tag}': ap, f'tt_{tag}': tt, f'rd_{tag}': rd, f'av_{tag}': av}
np.savez_compressed(ROOT / 'fp2.npz', **out)
print('saved fp2.npz')
