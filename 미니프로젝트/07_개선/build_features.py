"""원본 train.csv/predict_input.csv -> 모델 입력 행렬 캐시.

제공 컬럼(ecfp/fcfp/ptfp 1024*3 + MolWt/clogp/sa_score/qed)은 그대로 쓰고
RDKit 2D descriptor와 count 기반 Morgan을 덧붙인다.
"""
import sys, pathlib, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'features.npz'
TRAIN = ROOT.parent / '01_원본자료' / 'train_original_labels.csv.bak'
PRED = ROOT.parent / '01_원본자료' / 'predict_input.csv'

DESC = [(n, f) for n, f in Descriptors.descList if n != 'Ipc']  # Ipc는 값이 폭주한다
MORGAN_C = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, countSimulation=False)
MORGAN_C3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024, includeChirality=True)


def mol_block(smiles):
    desc = np.zeros((len(smiles), len(DESC)), dtype=np.float32)
    cnt = np.zeros((len(smiles), 1024), dtype=np.float32)
    r3 = np.zeros((len(smiles), 1024), dtype=np.uint8)
    maccs = np.zeros((len(smiles), 167), dtype=np.uint8)
    scaf = []
    bad = 0
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad += 1
            scaf.append(f'INVALID_{i}')
            continue
        for j, (_, fn) in enumerate(DESC):
            try:
                v = fn(m)
            except Exception:
                v = np.nan
            desc[i, j] = v
        cnt[i] = MORGAN_C.GetCountFingerprintAsNumPy(m)
        r3[i] = MORGAN_C3.GetFingerprintAsNumPy(m)
        maccs[i] = np.array(rdMolDescriptors.GetMACCSKeysFingerprint(m), dtype=np.uint8)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            sc = ''
        scaf.append(sc if sc else 'ACYCLIC')
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    return desc, cnt, r3, maccs, np.array(scaf), bad


def main():
    tr = pd.read_csv(TRAIN)
    pr = pd.read_csv(PRED)
    y = tr['label'].to_numpy(np.int8)          # 원본 규약: 1=비독성, 0=독성
    given = [c for c in tr.columns if c not in ('SMILES', 'label')]
    assert given == [c for c in pr.columns if c != 'SMILES'], '제공 컬럼 순서 불일치'
    Xg_tr = tr[given].to_numpy(np.float32)
    Xg_pr = pr[given].to_numpy(np.float32)

    out = {}
    for tag, smiles in (('tr', tr.SMILES.tolist()), ('pr', pr.SMILES.tolist())):
        d, c, r3, mk, sc, bad = mol_block(smiles)
        print(tag, 'parse 실패', bad, flush=True)
        out[f'desc_{tag}'], out[f'cnt_{tag}'], out[f'r3_{tag}'], out[f'maccs_{tag}'], out[f'scaf_{tag}'] = d, c, r3, mk, sc
    np.savez_compressed(
        OUT, y=y, given_tr=Xg_tr, given_pr=Xg_pr,
        given_names=np.array(given), desc_names=np.array([n for n, _ in DESC]),
        smiles_tr=tr.SMILES.to_numpy(), smiles_pr=pr.SMILES.to_numpy(), **out)
    print('saved', OUT, OUT.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
