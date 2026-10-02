"""최종 블렌드 -> 제출 CSV. 임계값은 OOF에서 F1(pos)/정확도/macro-F1을 함께 보고 고른다."""
import sys, json, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parent
D = np.load(ROOT / 'features.npz', allow_pickle=True)
Y = D['y'].astype(int)
SM = D['smiles_pr']
EXCLUDE = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else set()


def load():
    n, o, t = [], [], []
    for f in sorted((ROOT / 'preds').glob('*.npz')):
        if f.stem in EXCLUDE:
            continue
        z = np.load(f); n.append(f.stem); o.append(z['oof']); t.append(z['test'])
    return n, np.array(o), np.array(t)


def greedy(O, y, n_iter=60):
    chosen, cur, best = [], np.zeros(len(y)), -1
    while len(chosen) < n_iter:
        sc = [roc_auc_score(y, (cur * len(chosen) + O[i]) / (len(chosen) + 1)) for i in range(len(O))]
        i = int(np.argmax(sc))
        if sc[i] <= best + 1e-6:
            break
        cur = (cur * len(chosen) + O[i]) / (len(chosen) + 1); chosen.append(i); best = sc[i]
    return np.bincount(chosen, minlength=len(O)) / len(chosen), best


def scan(y, p):
    rows = []
    for rate in np.arange(0.50, 0.68, 0.005):
        th = np.quantile(p, 1 - rate); q = (p >= th).astype(int)
        rows.append((rate, f1_score(y, q), accuracy_score(y, q), f1_score(y, q, average='macro')))
    return pd.DataFrame(rows, columns=['pos_rate', 'f1_pos', 'accuracy', 'macro_f1'])


if __name__ == '__main__':
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.577
    names, O, T = load()
    Or = np.array([rankdata(x) / len(x) for x in O])
    Tr = np.array([rankdata(x) / len(x) for x in T])
    w, auc = greedy(Or, Y)
    used = {n: round(float(x), 3) for n, x in zip(names, w) if x > 0}
    oof, test = w @ Or, w @ Tr
    df = scan(Y, oof)
    print('후보 모델:', names)
    print('선택 가중치:', used, f'OOF AUC {auc:.4f}')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    cut = np.quantile(test, 1 - rate)
    out = (test >= cut).astype(int)
    pd.DataFrame({'SMILES': SM, 'output': out}).to_csv(ROOT / 'predict_output_최종.csv', index=False)
    best = df.iloc[int(df.f1_pos.idxmax())]
    json.dump({'weights': used, 'oof_auc': float(auc), 'chosen_pos_rate': rate,
               'oof_at_chosen': scan(Y, oof).query('abs(pos_rate-@rate)<1e-9').to_dict('records'),
               'oof_best_f1': best.to_dict(), 'test_pos_rate': float(out.mean())},
              open(ROOT / 'submission_meta.json', 'w'), indent=2, ensure_ascii=False)
    print(f'\n저장: predict_output_최종.csv  양성비율 {out.mean():.4f}')
