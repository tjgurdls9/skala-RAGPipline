"""피처셋/모델 실험. 랜덤 층화 CV(리더보드 근사)와 골격 CV를 함께 본다."""
import sys, time, json, pathlib, numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parent
D = np.load(ROOT / 'features.npz', allow_pickle=True)
Y = D['y'].astype(int)
SCAF = D['scaf_tr']

GIVEN_NAMES = list(D['given_names'])
PROP_IDX = [GIVEN_NAMES.index(c) for c in ('MolWt', 'clogp', 'sa_score', 'qed')]
ECFP = slice(0, 1024); FCFP = slice(1024, 2048); PTFP = slice(2048, 3072)


def blocks(tag):
    g = D[f'given_{tag}']
    return dict(ecfp=g[:, ECFP], fcfp=g[:, FCFP], ptfp=g[:, PTFP], prop=g[:, PROP_IDX],
                desc=D[f'desc_{tag}'], cnt=D[f'cnt_{tag}'], r3=D[f'r3_{tag}'], maccs=D[f'maccs_{tag}'])


BTR, BPR = blocks('tr'), blocks('pr')
FEATSETS = {
    'given3fp+prop': ['ecfp', 'fcfp', 'ptfp', 'prop'],
    '+desc': ['ecfp', 'fcfp', 'ptfp', 'prop', 'desc'],
    '+desc+cnt': ['ecfp', 'fcfp', 'ptfp', 'prop', 'desc', 'cnt'],
    'all': ['ecfp', 'fcfp', 'ptfp', 'prop', 'desc', 'cnt', 'r3', 'maccs'],
    'desc+prop only': ['prop', 'desc'],
}


def build(names):
    Xt = np.hstack([BTR[n] for n in names]).astype(np.float32)
    Xp = np.hstack([BPR[n] for n in names]).astype(np.float32)
    keep = Xt.std(0) > 0
    return Xt[:, keep], Xp[:, keep]


def best_f1(y, p):
    """양성(=1, 비독성) F1 최대 임계값."""
    ths = np.unique(np.round(p, 3))
    best = max(((f1_score(y, (p >= t).astype(int)), t) for t in ths))
    return best


def splits(kind, n):
    if kind == 'random':
        return list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), Y))
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(n), Y, SCAF))


def run_lgb(X, folds, **kw):
    import lightgbm as lgb
    oof = np.zeros(len(Y))
    params = dict(n_estimators=600, num_leaves=31, learning_rate=0.05, colsample_bytree=0.4,
                  subsample=0.8, subsample_freq=1, min_child_samples=20, n_jobs=-1,
                  random_state=42, verbose=-1)
    params.update(kw)
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params).fit(X[tr], Y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def report(name, oof):
    f1, th = best_f1(Y, oof)
    pred = (oof >= th).astype(int)
    print(f'{name:28s} AUC {roc_auc_score(Y, oof):.4f}  F1(pos) {f1:.4f}@{th:.3f}  '
          f'acc {accuracy_score(Y, pred):.4f}  F1@0.5 {f1_score(Y, (oof>=.5).astype(int)):.4f}', flush=True)
    return oof


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'random'
    folds = splits(kind, len(Y))
    print(f'== {kind} 5-fold ==')
    for name, names in FEATSETS.items():
        X, _ = build(names)
        t = time.time()
        oof = run_lgb(X, folds)
        np.save(ROOT / f'oof_{kind}_{name.replace("+","p").replace(" ","_")}.npy', oof)
        report(f'{name} ({X.shape[1]}) {time.time()-t:.0f}s', oof)
