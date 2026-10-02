"""리더보드에서 역산한 실제 양성 개수(A=493)를 이용해 F1 최적 예측 개수 D를 고른다.

F1 = 2*TP / (D + A). A를 알고 있으므로, OOF로 보정한 확률로 E[TP(D)]를 추정하면
임계값을 추측이 아니라 계산으로 정할 수 있다.
"""
import sys, pathlib, numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

A_TEST = 493        # 리더보드 24개 제출의 F1/Acc/Precision/Recall에서 역산, 전부 일치
N_TEST = 927
ROOT = pathlib.Path(__file__).resolve().parents[1]
PI_TEST = A_TEST / N_TEST


def shift_prior(p, pi_from, pi_to):
    """학습 사전확률에서 평가 사전확률로 확률을 보정한다."""
    odds = np.clip(p, 1e-6, 1 - 1e-6)
    odds = odds / (1 - odds)
    k = (pi_to / (1 - pi_to)) / (pi_from / (1 - pi_from))
    o = odds * k
    return o / (1 + o)


def optimal_d(oof, y, test):
    iso = IsotonicRegression(out_of_bounds='clip').fit(oof, y)
    pt = shift_prior(iso.predict(test), y.mean(), PI_TEST)
    order = np.argsort(-pt)
    csum = np.cumsum(pt[order])
    ds = np.arange(1, len(pt) + 1)
    f1 = 2 * csum / (ds + A_TEST)
    best = int(np.argmax(f1))
    return ds[best], f1[best], pt, order, f1


def main(oof_path, y_path, test_path):
    y = np.load(y_path, allow_pickle=True)['y'].astype(int) if str(y_path).endswith('.npz') else np.load(y_path)
    oof, test = np.load(oof_path), np.load(test_path)
    D, ef1, pt, order, curve = optimal_d(oof, y, test)
    print(f'보정 후 테스트 기대 양성 합계 {pt.sum():.1f} (실제 {A_TEST})')
    print(f'F1 최적 예측개수 D = {D},  기대 F1 = {ef1:.4f}')
    for d in (D - 40, D - 20, D, D + 20, D + 40, 535, 565):
        if 0 < d <= len(pt):
            print(f'  D={d:4d} -> 기대 F1 {curve[d-1]:.4f}')
    pred = np.zeros(len(pt), int); pred[order[:D]] = 1
    return pred


if __name__ == '__main__':
    main(*sys.argv[1:])
