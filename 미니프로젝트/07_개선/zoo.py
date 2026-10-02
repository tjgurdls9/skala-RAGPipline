"""모델 zoo: 동일 폴드에서 여러 모델의 OOF/test 예측을 만들어 저장한다."""
import sys, time, pathlib, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent
PRED_DIR = ROOT / 'preds'; PRED_DIR.mkdir(exist_ok=True)
D = np.load(ROOT / 'features.npz', allow_pickle=True)
Y = D['y'].astype(int)
GN = list(D['given_names'])
PROP = [GN.index(c) for c in ('MolWt', 'clogp', 'sa_score', 'qed')]

def stack(tag):
    g = D[f'given_{tag}']
    return np.hstack([g[:, :1024], g[:, 1024:2048], g[:, 2048:3072], g[:, PROP],
                      D[f'desc_{tag}'], D[f'cnt_{tag}'], D[f'r3_{tag}'], D[f'maccs_{tag}']]).astype(np.float32)

XT, XP = stack('tr'), stack('pr')
KEEP = XT.std(0) > 0
XT, XP = XT[:, KEEP], XP[:, KEEP]
# 추가 fingerprint 블록(atom-pair/torsion/RDKit/Avalon)을 붙인 확장 행렬
F2 = np.load(ROOT / 'fp2.npz')
X2 = np.hstack([XT] + [F2[f'{k}_tr'] for k in ('ap', 'tt', 'rd', 'av')]).astype(np.float32)
X2P = np.hstack([XP] + [F2[f'{k}_pr'] for k in ('ap', 'tt', 'rd', 'av')]).astype(np.float32)
K2 = X2.std(0) > 0
X2, X2P = X2[:, K2], X2P[:, K2]
# 희소 이진 블록만 (선형/거리 모델용)
BIN = np.hstack([D['given_tr'][:, :3072], D['r3_tr'], D['maccs_tr']]).astype(np.float32)
BINP = np.hstack([D['given_pr'][:, :3072], D['r3_pr'], D['maccs_pr']]).astype(np.float32)
BK = BIN.std(0) > 0
BIN, BINP = BIN[:, BK], BINP[:, BK]
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(XT, Y))
print('X', XT.shape, 'X2', X2.shape, 'bin', BIN.shape, flush=True)


def cv(name, fit_predict, X, Xp):
    f = PRED_DIR / f'{name}.npz'
    if f.exists():
        z = np.load(f); return z['oof'], z['test']
    t = time.time()
    oof = np.zeros(len(Y)); test = np.zeros(len(Xp))
    for tr, va in FOLDS:
        pv, pt = fit_predict(X[tr], Y[tr], X[va], Xp)
        oof[va] = pv; test += pt / len(FOLDS)
    np.savez(f, oof=oof, test=test)
    print(f'{name:22s} AUC {roc_auc_score(Y, oof):.4f}  F1max {max(f1_score(Y,(oof>=t2).astype(int)) for t2 in np.arange(.3,.7,.005)):.4f}  {time.time()-t:.0f}s', flush=True)
    return oof, test


def lgb_fp(**kw):
    def f(Xtr, ytr, Xva, Xp):
        import lightgbm as lgb
        p = dict(n_estimators=900, num_leaves=63, learning_rate=0.03, colsample_bytree=0.3,
                 subsample=0.8, subsample_freq=1, min_child_samples=20, reg_lambda=1.0,
                 n_jobs=-1, verbose=-1)
        p.update(kw)
        m = lgb.LGBMClassifier(**p).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xp)[:, 1]
    return f


def xgb_fp(**kw):
    def f(Xtr, ytr, Xva, Xp):
        import xgboost as xgb
        p = dict(n_estimators=900, max_depth=7, learning_rate=0.03, colsample_bytree=0.3,
                 subsample=0.8, min_child_weight=3, reg_lambda=1.0, tree_method='hist',
                 n_jobs=-1, eval_metric='logloss')
        p.update(kw)
        m = xgb.XGBClassifier(**p).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xp)[:, 1]
    return f


def et_fp(Xtr, ytr, Xva, Xp):
    from sklearn.ensemble import ExtraTreesClassifier
    m = ExtraTreesClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
                             n_jobs=-1, random_state=42).fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xp)[:, 1]


def rf_fp(Xtr, ytr, Xva, Xp):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
                               n_jobs=-1, random_state=42).fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xp)[:, 1]


def logit(Xtr, ytr, Xva, Xp):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=0.3, max_iter=3000, n_jobs=-1).fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xp)[:, 1]


def svm_fp(Xtr, ytr, Xva, Xp):
    from sklearn.svm import SVC
    m = SVC(C=3.0, gamma='scale', probability=True, random_state=42).fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xp)[:, 1]


def mlp(Xtr, ytr, Xva, Xp):
    from sklearn.neural_network import MLPClassifier
    sc = StandardScaler(with_mean=False).fit(Xtr)
    m = MLPClassifier((512, 128), alpha=1e-3, batch_size=128, learning_rate_init=1e-3,
                      max_iter=60, early_stopping=True, n_iter_no_change=6,
                      random_state=42).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xva))[:, 1], m.predict_proba(sc.transform(Xp))[:, 1]


def knn_tanimoto(k=25):
    def f(Xtr, ytr, Xva, Xp):
        out = []
        nt = Xtr.sum(1)
        for Q in (Xva, Xp):
            inter = Q @ Xtr.T
            sim = inter / (Q.sum(1)[:, None] + nt[None, :] - inter + 1e-9)
            idx = np.argpartition(-sim, k, axis=1)[:, :k]
            w = np.take_along_axis(sim, idx, 1)
            out.append((w * ytr[idx]).sum(1) / (w.sum(1) + 1e-9))
        return out[0], out[1]
    return f


MODELS = {
    'lgb_a': (lgb_fp(random_state=42), XT, XP),
    'lgb_b': (lgb_fp(random_state=7, num_leaves=127, learning_rate=0.02, n_estimators=1400,
                     colsample_bytree=0.2, min_child_samples=10), XT, XP),
    'lgb_goss': (lgb_fp(random_state=11, boosting_type='goss', num_leaves=63, n_estimators=1200,
                        subsample=1.0, subsample_freq=0), XT, XP),
    'lgb_fp2': (lgb_fp(random_state=3, num_leaves=63, n_estimators=1100, colsample_bytree=0.25), X2, X2P),
    'xgb_fp2': (xgb_fp(random_state=3, max_depth=8, n_estimators=1100, colsample_bytree=0.25), X2, X2P),
    'xgb_a': (xgb_fp(random_state=42), XT, XP),
    'xgb_b': (xgb_fp(random_state=7, max_depth=10, min_child_weight=8, colsample_bytree=0.2,
                     n_estimators=1400, learning_rate=0.02), XT, XP),
    'et': (et_fp, XT, XP),
    'rf': (rf_fp, XT, XP),
    'logit_bin': (logit, BIN, BINP),
    'svm_bin': (svm_fp, BIN, BINP),
    'mlp': (mlp, XT, XP),
    'knn_tan': (knn_tanimoto(), BIN, BINP),
}

if __name__ == '__main__':
    want = sys.argv[1:] or list(MODELS)
    for n in want:
        fn, X, Xp = MODELS[n]
        cv(n, fn, X, Xp)
