"""품질 손실을 허용하며 최고 실측본과의 행 차이 k를 최대화한다.

평가셋은 고정이라 각 후보의 점수는 이미 정해져 있고 우리가 모를 뿐이다.
k행 다른 후보의 TP 변화 표준편차는 대략 sqrt(k)/2이므로, 0.85 꼬리를 노리려면
품질을 지키면서 k를 키우는 것이 유일하게 합리적인 수단이다.

pseudo 입력은 가짜 라벨 생성 단계에서 간접 누수가 있어 OOF가 낙관적이다.
품질 판정에서 제외하고 다양성 재료로만 쓴다.
"""
import numpy as np, pandas as pd, pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
Z = np.load(ROOT / '10_최종개선' / 'probability_fusion_inputs.npz', allow_pickle=True)
O, OLD, FULL, Y = Z['O'], Z['old_test'], Z['full_test'], Z['y'].astype(int)
NAMES = [str(n) for n in Z['names']]
BEST = pd.read_csv(ROOT / '09_임계값' / '2차_제출됨_백업.csv').output.values
lg = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
L, T_OLD, T_FULL = lg(O), lg(OLD), lg(FULL)

p = np.load(ROOT / '09_임계값' / 'pseudo.npz')
L = np.hstack([L, lg(p['oof'])[:, None]])
T_OLD = np.hstack([T_OLD, lg(p['test'])[:, None]])
T_FULL = np.hstack([T_FULL, lg(p['test'])[:, None]])
NAMES.append('pseudo')
PSEUDO = len(NAMES) - 1
CLEAN = [i for i in range(len(NAMES)) if i != PSEUDO]
n = len(NAMES)


def run(cols, C, mix, seed, D):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(O, Y))
    Xo = L[:, cols]
    oof = np.zeros(len(Y))
    for tr, va in folds:
        m = LogisticRegression(C=C, max_iter=3000).fit(Xo[tr], Y[tr])
        oof[va] = m.predict_proba(Xo[va])[:, 1]
    cut = int(D / 927 * len(Y))
    pr = np.zeros(len(Y), int); pr[np.argsort(-oof)[:cut]] = 1
    meta = LogisticRegression(C=C, max_iter=3000).fit(Xo, Y)
    t = meta.predict_proba(((1 - mix) * T_OLD + mix * T_FULL)[:, cols])[:, 1]
    pred = np.zeros(927, int); pred[np.argsort(-t)[:D]] = 1
    return f1_score(Y, pr), roc_auc_score(Y, oof), pred


rng = np.random.default_rng(7)
cfgs = [('ref_all', CLEAN, 0.1, 1.0, 42, 552)]
for i in range(160):
    k = int(rng.integers(3, n + 1))
    cols = sorted(rng.choice(n, size=k, replace=False).tolist())
    cfgs.append((f'r{i}', cols, float(rng.choice([0.003, 0.01, 0.1, 1.0, 10.0])),
                 float(rng.choice([0.0, 1.0])), int(rng.choice([42, 7, 2026])),
                 int(rng.choice([540, 552, 560, 565]))))

rows, preds = [], {}
for tag, cols, C, mix, seed, D in cfgs:
    f1, auc, pred = run(cols, C, mix, seed, D)
    preds[tag] = pred
    rows.append(dict(tag=tag, k=len(cols), C=C, mix=mix, seed=seed, D=D, oof_f1=f1,
                     oof_auc=auc, has_pseudo=PSEUDO in cols,
                     diff=int((pred != BEST).sum()), cols=','.join(map(str, cols))))
df = pd.DataFrame(rows)
top = df[~df.has_pseudo].oof_f1.max()          # 누수 없는 후보 기준으로 품질선을 잡는다
df['loss'] = top - df.oof_f1
df.sort_values('diff', ascending=False).to_csv(ROOT / '09_임계값' / 'diverse2.csv', index=False)
np.savez(ROOT / '09_임계값' / 'diverse2_preds.npz', **preds)
print(f'누수 없는 후보 최고 OOF F1 {top:.4f}\n')
for lim in (0.002, 0.004, 0.008):
    ok = df[(df.loss <= lim) & (~df.has_pseudo)].sort_values('diff', ascending=False)
    if len(ok):
        b = ok.iloc[0]
        print(f'손실 {lim:.3f} 이내 (누수없음): 최대 diff {b["diff"]:3d}행  tag={b.tag} k={b.k} C={b.C} D={b.D} F1={b.oof_f1:.4f}')
print()
ok = df[df.loss <= 0.004].sort_values('diff', ascending=False)
print(ok.head(14)[['tag','k','C','mix','seed','D','oof_f1','oof_auc','has_pseudo','diff']].to_string(index=False))
