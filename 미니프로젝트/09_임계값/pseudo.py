"""전이학습(pseudo-labeling): 평가셋 927개의 '특징'을 학습에 쓴다.

평가셋 라벨은 모른다. 확신이 높은 예측만 가짜 라벨로 붙여 재학습하면
평가셋 분포를 모델이 한 번 더 본다. OOF는 train만으로 정직하게 측정한다.
"""
import numpy as np, pathlib, lightgbm as lgb
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
P = dict(n_estimators=1100, num_leaves=63, learning_rate=0.03, colsample_bytree=0.25,
         subsample=0.8, subsample_freq=1, min_child_samples=20, reg_lambda=1.0,
         n_jobs=-1, verbose=-1, random_state=3)

# 1단계: 평가셋 가짜 라벨 생성
base = np.zeros(927)
for tr, va in FOLDS:
    base += lgb.LGBMClassifier(**P).fit(XT[tr], Y[tr]).predict_proba(XP)[:, 1] / 5
conf = (base > 0.80) | (base < 0.20)
pl_y = (base[conf] >= 0.5).astype(int)
print(f'가짜 라벨 채택 {conf.sum()}/927 (양성 {pl_y.sum()})', flush=True)

# 2단계: train 학습 fold + 가짜 라벨을 합쳐 재학습. 검증 fold는 train에서만 온다.
oof = np.zeros(len(Y)); test = np.zeros(927)
for tr, va in FOLDS:
    Xa = np.vstack([XT[tr], XP[conf]]); ya = np.concatenate([Y[tr], pl_y])
    w = np.concatenate([np.ones(len(tr)), np.full(conf.sum(), 0.5)])
    m = lgb.LGBMClassifier(**P).fit(Xa, ya, sample_weight=w)
    oof[va] = m.predict_proba(XT[va])[:, 1]
    test += m.predict_proba(XP)[:, 1] / 5
f1 = max(f1_score(Y, (oof >= t).astype(int)) for t in np.arange(.3, .6, .005))
print(f'pseudo-label LGBM  OOF AUC {roc_auc_score(Y, oof):.4f}  F1 {f1:.4f}')
np.savez(ROOT / '09_임계값' / 'pseudo.npz', oof=oof, test=test)
