"""Multi-block averaged Tanimoto/MinMax kernel SVM (precomputed kernel; rows addressed by index)."""
import sys
from zoo import *
from sklearn.svm import SVC
F = np.load(ROOT / 'fp2.npz')
G = lambda t: D[f'given_{t}'][:, :3072]
blocks = []   # each: all-sample binary matrix [train; test]
cat2 = lambda a, b: np.vstack([a, b]).astype(np.float32)
for i in range(3): blocks.append(cat2(G('tr')[:, i*1024:(i+1)*1024], G('pr')[:, i*1024:(i+1)*1024]))
for k in ('r3', 'maccs'): blocks.append(cat2(D[f'{k}_tr'], D[f'{k}_pr']))
for k in ('ap', 'tt', 'rd', 'av'): blocks.append(cat2(F[f'{k}_tr'], F[f'{k}_pr']))
def tani(A):
    i = A @ A.T; s = A.sum(1); return i / (s[:, None] + s[None] - i + 1e-9)
K = np.mean([tani(b) for b in blocks], 0)
NTR = len(Y); IDX = np.arange(NTR + len(D['smiles_pr']))[:, None]
def svm_k(C_, Kmat):
    def f(Xtr, ytr, Xva, Xp):
        a, b, c = Xtr[:, 0], Xva[:, 0], Xp[:, 0]
        m = SVC(C=C_, kernel='precomputed').fit(Kmat[np.ix_(a, a)], ytr)
        return m.decision_function(Kmat[np.ix_(b, a)]), m.decision_function(Kmat[np.ix_(c, a)])
    return f
if __name__ == '__main__':
    cv('svm_mk10', svm_k(10, K), IDX[:NTR], IDX[NTR:])
    for p in (2, 3):
        for C_ in (3, 10, 30):
            cv(f'svm_mk_p{p}_c{C_}', svm_k(C_, K ** p), IDX[:NTR], IDX[NTR:])
    for p, C_ in ((2, 1), (2, 2), (1.5, 3), (2.5, 2)):
        cv(f'svm_mk_p{p}_c{C_}', svm_k(C_, K ** p), IDX[:NTR], IDX[NTR:])
