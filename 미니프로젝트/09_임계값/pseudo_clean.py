"""누수 없는 pseudo-labeling.

fold별로 그 fold의 학습 행만으로 base를 학습해 평가셋 가짜 라벨을 만든다.
검증 fold의 라벨은 가짜 라벨 생성에 전혀 관여하지 않는다.
"""
import numpy as np, pathlib, time, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = np.load(ROOT / '07_개선' / 'features.npz', allow_pickle=True)
F2 = np.load(ROOT / '07_개선' / 'fp2.npz')
GN = list(D['given_names']); PROP = [GN.index(c) for c in ('MolWt','clogp','sa_score','qed')]

def stack(tag):
    g = D[f'given_{tag}']
    return np.hstack([g[:, :3072], g[:, PROP], D[f'desc_{tag}'], D[f'cnt_{tag}'],
                      D[f'r3_{tag}'], D[f'maccs_{tag}']] +
                     [F2[f'{k}_{tag}'] for k in ('ap','tt','rd','av')]).astype(np.float32)

XT, XP = stack('tr'), stack('pr')
K = XT.std(0) > 0; XT, XP = XT[:, K], XP[:, K]
Y = D['y'].astype(int)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(XT, Y))
BASE = dict(n_estimators=700, num_leaves=63, learning_rate=0.04, colsample_bytree=0.25,
            subsample=0.8, subsample_freq=1, min_child_samples=20, reg_lambda=1.0,
            n_jobs=-1, verbose=-1)

oof = np.zeros(len(Y)); test = np.zeros(927)
t0 = time.time()
for fi, (tr, va) in enumerate(FOLDS):
    # 1) 이 fold의 학습 행만으로 base 학습 -> 평가셋 가짜 라벨
    b = lgb.LGBMClassifier(random_state=100 + fi, **BASE).fit(XT[tr], Y[tr])
    pt = b.predict_proba(XP)[:, 1]
    conf = (pt > 0.82) | (pt < 0.18)
    py = (pt[conf] >= 0.5).astype(int)
    # 2) 학습 행 + 가짜 라벨로 재학습 -> 검증 fold와 평가셋 예측
    Xa = np.vstack([XT[tr], XP[conf]]); ya = np.concatenate([Y[tr], py])
    w = np.concatenate([np.ones(len(tr)), np.full(int(conf.sum()), 0.4)])
    m = lgb.LGBMClassifier(random_state=200 + fi, **BASE).fit(Xa, ya, sample_weight=w)
    oof[va] = m.predict_proba(XT[va])[:, 1]
    test += m.predict_proba(XP)[:, 1] / 5
    print(f'fold{fi} 가짜라벨 {int(conf.sum())}개  {time.time()-t0:.0f}s', flush=True)

f1 = max(f1_score(Y, (oof >= t).astype(int)) for t in np.arange(.3, .6, .005))
print(f'\n누수없는 pseudo LGBM  OOF AUC {roc_auc_score(Y, oof):.4f}  F1 {f1:.4f}')
np.savez(ROOT / '09_임계값' / 'pseudo_clean.npz', oof=oof, test=test)
