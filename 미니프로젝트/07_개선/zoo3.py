"""svm (Tanimoto kernel, decision_function), catboost, seed-bagged GBDTs."""
import sys
from zoo import *
from sklearn.svm import SVC
def tani(A, B):
    i = A @ B.T; return i / (A.sum(1)[:, None] + B.sum(1)[None] - i + 1e-9)
def svm_tani(C):
    def f(Xtr, ytr, Xva, Xp):
        m = SVC(C=C, kernel='precomputed').fit(tani(Xtr, Xtr), ytr)
        return m.decision_function(tani(Xva, Xtr)), m.decision_function(tani(Xp, Xtr))
    return f
def cat(**kw):
    def f(Xtr, ytr, Xva, Xp):
        from catboost import CatBoostClassifier
        p = dict(iterations=1500, learning_rate=0.05, depth=7, rsm=0.1, l2_leaf_reg=5, thread_count=5, verbose=0, random_seed=0, border_count=64)
        p.update(kw); m = CatBoostClassifier(**p).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xp)[:, 1]
    return f
M3 = {
 'svm_tani1': (svm_tani(1.0), BIN, BINP),
 'svm_tani3': (svm_tani(3.0), BIN, BINP),
 'svm_tani10': (svm_tani(10.0), BIN, BINP),
 'svm_tani30': (svm_tani(30.0), BIN, BINP),
 'cat_fp2': (cat(), X2, X2P),
}
for s in (11, 12, 13):
    M3[f'lgb_fp2_s{s}'] = (lgb_fp(random_state=s, num_leaves=63, n_estimators=1100, colsample_bytree=0.25, n_jobs=5), X2, X2P)
if __name__ == '__main__':
    for n in sys.argv[1:]:
        fn, X, Xp = M3[n]; cv(n, fn, X, Xp)
