"""Blend/stack evaluation with optimistic and nested (leakage-free) estimates, on the fixed folds."""
import glob, pathlib, numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parent
Y = np.load(ROOT / 'features.npz', allow_pickle=True)['y'].astype(int)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(Y)), Y))
THS = np.linspace(0.30, 0.80, 101)

def load(exclude=()):
    names, O, T = [], [], []
    for f in sorted((ROOT / 'preds').glob('*.npz')):
        if f.stem in exclude: continue
        z = np.load(f); names.append(f.stem); O.append(z['oof']); T.append(z['test'])
    rk = lambda a: rankdata(a) / len(a)
    return names, np.array([rk(o) for o in O]), np.array([rk(t) for t in T])

def best_th(y, s):
    f = [f1_score(y, (s >= t).astype(int)) for t in THS]
    return THS[int(np.argmax(f))]

def metrics(y, s, th):
    q = (s >= th).astype(int)
    return dict(auc=roc_auc_score(y, s), f1=f1_score(y, q), acc=accuracy_score(y, q), mf1=f1_score(y, q, average='macro'), th=th)

def greedy(O, y, n_iter=40):
    chosen, cur, best = [], np.zeros(len(y)), -1
    while len(chosen) < n_iter:
        sc = [roc_auc_score(y, (cur * len(chosen) + O[i]) / (len(chosen) + 1)) for i in range(len(O))]
        i = int(np.argmax(sc))
        if sc[i] <= best + 1e-6: break
        cur = (cur * len(chosen) + O[i]) / (len(chosen) + 1); chosen.append(i); best = sc[i]
    return np.bincount(chosen, minlength=len(O)) / len(chosen)

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def fit_lr(X, y, C=0.1):
    return LogisticRegression(C=C, max_iter=2000).fit(logit(X), y)

def fit_lgbm(X, y):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=150, num_leaves=4, learning_rate=0.03, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, n_jobs=2)

# a "method" = fit(Otr [m,n], ytr) -> predict(O [m,n]) -> score
def m_greedy(O, y):
    w = greedy(O, y); return lambda Z: w @ Z
def m_lr(C):
    def f(O, y):
        mdl = fit_lr(O.T, y, C); return lambda Z: mdl.predict_proba(logit(Z.T))[:, 1]
    return f
def m_lgbm(O, y):
    mdl = fit_lgbm(O.T, y).fit(O.T, y); return lambda Z: mdl.predict_proba(Z.T)[:, 1]

def evaluate(O, method, inner_th=True):
    """returns (optimistic, nested). Nested: for each outer fold, fit method (+threshold from inner CV oof) on other 4 folds."""
    y = Y
    pred = method(O, y)
    s = pred(O); opt = metrics(y, s, best_th(y, s))   # in-sample (optimistic) for the same data
    sn = np.zeros(len(y)); ths = []
    for k, (tr, va) in enumerate(FOLDS):
        # inner OOF over the 4 training folds to pick the threshold honestly
        inner = np.zeros(len(tr))
        skf = StratifiedKFold(4, shuffle=True, random_state=k).split(tr, y[tr])
        for a, b in skf:
            inner[b] = method(O[:, tr[a]], y[tr[a]])(O[:, tr[b]])
        th = best_th(y[tr], inner); ths.append(th)
        sn[va] = method(O[:, tr], y[tr])(O[:, va])
    # pooled nested score; per-fold thresholds applied per fold
    q = np.zeros(len(y), int)
    for (tr, va), th in zip(FOLDS, ths): q[va] = (sn[va] >= th)
    nest = dict(auc=roc_auc_score(y, sn), f1=f1_score(y, q), acc=accuracy_score(y, q), mf1=f1_score(y, q, average='macro'), th=float(np.mean(ths)))
    return opt, nest

def fmt(d): return f"AUC {d['auc']:.4f} F1 {d['f1']:.4f} acc {d['acc']:.4f} mF1 {d['mf1']:.4f} th {d['th']:.3f}"
