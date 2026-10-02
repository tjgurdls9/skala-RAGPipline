"""최고 실측 제출본(0.8393)과 최대한 다르면서 OOF F1은 거의 같은 후보를 찾는다.

평가셋은 고정이므로 제출마다 점수가 새로 뽑히는 복권이 아니다. 다만 우리가 모르는
점수가 후보마다 다르고 순위는 제출 행별로 매겨지므로, 이미 0.8393을 확보한 이상
남은 제출은 '평균을 지키면서 최대한 다른' 후보여야 꼬리를 노릴 수 있다.
"""
import numpy as np, pandas as pd, pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
Z = np.load(ROOT / '10_최종개선' / 'probability_fusion_inputs.npz', allow_pickle=True)
O, OLD, FULL, Y = Z['O'], Z['old_test'], Z['full_test'], Z['y'].astype(int)
NAMES = [str(n) for n in Z['names']]
BEST = pd.read_csv(ROOT / '09_임계값' / '2차_제출됨_백업.csv').output.values  # 실측 0.8393
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(O, Y))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
L, LT_OLD, LT_FULL = logit(O), logit(OLD), logit(FULL)


def evaluate(cols, C, mix):
    """메타 LR을 4개 fold에서 학습해 나머지 fold를 예측(교차적합). 임계값도 안쪽에서만 고른다."""
    Xo = L[:, cols]
    oof = np.zeros(len(Y)); cuts = []
    for tr, va in FOLDS:
        m = LogisticRegression(C=C, max_iter=2000).fit(Xo[tr], Y[tr])
        oof[va] = m.predict_proba(Xo[va])[:, 1]
        p_in = m.predict_proba(Xo[tr])[:, 1]
        ths = np.quantile(p_in, np.linspace(.3, .6, 120))
        cuts.append(max((f1_score(Y[tr], (p_in >= t).astype(int)), t) for t in ths)[1])
    cut = float(np.median(cuts))
    f1 = f1_score(Y, (oof >= cut).astype(int))
    meta = LogisticRegression(C=C, max_iter=2000).fit(Xo, Y)
    Xt = (1 - mix) * LT_OLD[:, cols] + mix * LT_FULL[:, cols]
    test = meta.predict_proba(Xt)[:, 1]
    return f1, roc_auc_score(Y, oof), (test >= cut).astype(int), cut


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    rows = []
    n = len(NAMES)
    trials = [('전체25', list(range(n)), 0.01, 0.5), ('전체25', list(range(n)), 0.1, 0.5),
              ('전체25', list(range(n)), 1.0, 0.5), ('전체25-full', list(range(n)), 0.1, 1.0),
              ('전체25-old', list(range(n)), 0.1, 0.0)]
    for i in range(60):
        k = rng.integers(6, n + 1)
        cols = sorted(rng.choice(n, size=k, replace=False).tolist())
        trials.append((f'rand{i}', cols, float(rng.choice([0.01, 0.03, 0.1, 0.3, 1.0])),
                       float(rng.choice([0.0, 0.5, 1.0]))))
    for tag, cols, C, mix in trials:
        f1, auc, pred, cut = evaluate(cols, C, mix)
        rows.append(dict(tag=tag, k=len(cols), C=C, mix=mix, oof_f1=f1, oof_auc=auc,
                         pos=int(pred.sum()), diff=int((pred != BEST).sum()),
                         cols=','.join(map(str, cols))))
    df = pd.DataFrame(rows).sort_values('oof_f1', ascending=False)
    df.to_csv(ROOT / '09_임계값' / 'diverse_candidates.csv', index=False)
    top = df.oof_f1.max()
    ok = df[df.oof_f1 >= top - 0.002].sort_values('diff', ascending=False)
    print(f'최고 OOF F1 {top:.4f} / 허용선 {top-0.002:.4f} 이상인 후보 {len(ok)}개\n')
    print(ok.head(12)[['tag','k','C','mix','oof_f1','oof_auc','pos','diff']].to_string(index=False))
