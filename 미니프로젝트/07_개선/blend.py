"""OOF 기반 greedy 가중 블렌딩 + 임계값 선정 + 제출 파일 생성."""
import sys, json, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parent
D = np.load(ROOT / 'features.npz', allow_pickle=True)
Y = D['y'].astype(int)
SMILES_PR = D['smiles_pr']
N_ITER = 60


def load():
    names, oof, test = [], [], []
    for f in sorted((ROOT / 'preds').glob('*.npz')):
        z = np.load(f)
        names.append(f.stem); oof.append(z['oof']); test.append(z['test'])
    return names, np.array(oof), np.array(test)


def rank01(p):
    return rankdata(p) / len(p)


def greedy(names, O, Y, metric):
    """중복 허용 greedy 선택. 각 단계에서 평균 점수가 가장 좋아지는 모델을 추가한다."""
    chosen, cur = [], np.zeros(len(Y))
    best_hist = []
    for it in range(N_ITER):
        scores = [metric(Y, (cur * len(chosen) + O[i]) / (len(chosen) + 1)) for i in range(len(names))]
        i = int(np.argmax(scores))
        if best_hist and scores[i] <= best_hist[-1] + 1e-6:
            break
        cur = (cur * len(chosen) + O[i]) / (len(chosen) + 1)
        chosen.append(i); best_hist.append(scores[i])
    w = np.bincount(chosen, minlength=len(names)) / len(chosen)
    return w, best_hist[-1]


def best_threshold(y, p, metric):
    ths = np.quantile(p, np.linspace(0.05, 0.95, 400))
    vals = [metric(y, (p >= t).astype(int)) for t in ths]
    i = int(np.argmax(vals))
    return float(ths[i]), float(vals[i])


def report(tag, y, p):
    th_f1, f1v = best_threshold(y, p, f1_score)
    th_ac, acv = best_threshold(y, p, accuracy_score)
    print(f'{tag}: AUC {roc_auc_score(y, p):.4f} | F1(pos) {f1v:.4f}@{th_f1:.3f} '
          f'(acc {accuracy_score(y, (p>=th_f1).astype(int)):.4f}) | acc {acv:.4f}@{th_ac:.3f} '
          f'(F1 {f1_score(y, (p>=th_ac).astype(int)):.4f})')
    return th_f1, th_ac


if __name__ == '__main__':
    names, O, T = load()
    Or, Tr = np.array([rank01(o) for o in O]), np.array([rank01(t) for t in T])
    for n, o in zip(names, O):
        print(f'  {n:12s} AUC {roc_auc_score(Y, o):.4f}')
    w, sc = greedy(names, Or, Y, roc_auc_score)
    print('\n가중치:', {n: round(float(x), 3) for n, x in zip(names, w) if x > 0}, f'OOF AUC {sc:.4f}')
    oof_b, test_b = w @ Or, w @ Tr
    th_f1, th_ac = report('blend', Y, oof_b)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'f1'
    th = {'f1': th_f1, 'acc': th_ac}[mode]
    # rank 기준이라 테스트에도 같은 분위수를 적용한다
    q = float((oof_b < th).mean())
    cut = np.quantile(test_b, q)
    out = (test_b >= cut).astype(int)
    name = sys.argv[2] if len(sys.argv) > 2 else f'predict_output_blend_{mode}'
    pd.DataFrame({'SMILES': SMILES_PR, 'output': out}).to_csv(ROOT / f'{name}.csv', index=False)
    json.dump({'weights': {n: float(x) for n, x in zip(names, w)}, 'oof_auc': float(sc),
               'mode': mode, 'oof_threshold': th, 'test_quantile': q,
               'positive_rate_test': float(out.mean()), 'positive_rate_train': float(Y.mean())},
              open(ROOT / f'{name}.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\n{name}.csv  양성(비독성) 비율 {out.mean():.3f} (train {Y.mean():.3f})')
