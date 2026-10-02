"""Cross-fit a strongly shrunk chemical-neighbor residual correction."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SEEDS = [2026, 730, 731]
# Fixed before evaluation: narrow/high-order and broader residual neighborhoods.
CONFIGS = [(power, alpha) for power in [4, 8] for alpha in [.1, .25, .5]]


def neighbors():
    path = ROOT / 'neighbors.npz'
    if path.exists():
        return np.load(path)
    d = np.load(PROJECT / '07_개선/features.npz', allow_pickle=True)
    n, m = len(d['y']), len(d['smiles_pr'])
    K = np.zeros((n + m, n), np.float32)
    for atr, apr in [(d['given_tr'][:, :1024], d['given_pr'][:, :1024]),
                     (d['r3_tr'], d['r3_pr'])]:
        a = atr.astype(np.float32)
        b = np.vstack([a, apr]).astype(np.float32)
        sizes = a.sum(axis=1)
        for start in range(0, len(b), 256):
            v = b[start:start + 256]
            intersection = v @ a.T
            union = v.sum(axis=1)[:, None] + sizes[None, :] - intersection
            K[start:start + len(v)] += .5 * np.divide(intersection, union,
                out=np.zeros_like(intersection), where=union > 0)
    K[np.arange(n), np.arange(n)] = -1
    idx = np.argpartition(K, -128, axis=1)[:, -128:]
    sim = np.take_along_axis(K, idx, axis=1)
    order = np.argsort(-sim, axis=1)
    idx, sim = np.take_along_axis(idx, order, axis=1), np.take_along_axis(sim, order, axis=1)
    assert not np.any(idx[:n] == np.arange(n)[:, None])
    np.savez_compressed(path, idx=idx, sim=sim)
    return np.load(path)


def correction(idx, sim, train, residual, power):
    permitted = np.zeros(len(residual), bool)
    permitted[train] = True
    valid = permitted[idx]
    # 128 precomputed neighbors safely provide >=32 training neighbors for these splits.
    assert np.all(valid.sum(axis=1) >= 32)
    chosen = valid & (valid.cumsum(axis=1) <= 32)
    weights = np.where(chosen, sim ** power, 0.)
    assert np.all(weights[~valid] == 0)
    return (weights * residual[idx]).sum(axis=1) / (weights.sum(axis=1) + 1.)


def metrics(y, p, cut, band):
    return dict(f1=float(f1_score(y, p >= cut)), auc=float(roc_auc_score(y, p)),
                band_auc=float(roc_auc_score(y[band], p[band])),
                log_loss=float(log_loss(y, p)), predicted_positive=int((p >= cut).sum()))


def main():
    with threadpool_limits(2):
        source = np.load(PROJECT / '10_최종개선/probability_fusion_selected.npz')
        final = np.load(PROJECT / '10_최종개선/final_scores_0.5.npz')
        y, base, test, cut = source['y'], source['oof'], final['test'], float(final['threshold'])
        assert np.array_equal(base, final['oof'])
        assert len(base) == 8349 and len(test) == 927
        band = (base >= .35) & (base <= .50)
        z = neighbors(); idx, sim = z['idx'], z['sim']; n = len(y)
        residual = y - base
        baseline = metrics(y, base, cut, band)
        result = {}
        for power in [4, 8]:
            oofs, tests = [], []
            for seed in SEEDS:
                r, t = np.zeros(n), np.zeros(len(test))
                for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(base, y):
                    assert not np.intersect1d(tr, va).size
                    r[va] = correction(idx[va], sim[va], tr, residual, power)
                    t += correction(idx[n:], sim[n:], tr, residual, power) / 5
                oofs.append(r); tests.append(t)
            for alpha in [.1, .25, .5]:
                name = f'power{power}_shrink{alpha:g}'
                ps = np.clip(base + alpha * np.array(oofs), 1e-7, 1-1e-7)
                p = ps.mean(axis=0)
                q = np.clip(test + alpha * np.mean(tests, axis=0), 1e-7, 1-1e-7)
                scores = [float(f1_score(y, a >= cut)) for a in ps]
                result[name] = dict(power=power, alpha=alpha, **metrics(y, p, cut, band),
                    seed_f1=scores, repeat_f1=float(np.mean(scores)),
                    changed_test=int(((q >= cut) != (test >= cut)).sum()))
                np.savez_compressed(ROOT / f'{name}.npz', y=y, oof=p, repeated_oof=ps,
                    test=q, threshold=cut, base_oof=base, base_test=test)
                print(name, json.dumps(result[name]), flush=True)
        best_name = max(result, key=lambda k: result[k]['repeat_f1'])
        best = result[best_name]
        chosen = np.load(ROOT / f'{best_name}.npz')
        repeat_dec = chosen['repeated_oof'] >= cut
        base_dec = base >= cut
        rng = np.random.default_rng(431)
        deltas = []
        for _ in range(1000):
            ids = rng.integers(n, size=n); yy = y[ids]
            new = repeat_dec[:, ids]
            f = np.mean(2 * (new * yy).sum(axis=1) / (new.sum(axis=1) + yy.sum()))
            old = base_dec[ids]
            b = 2 * (old * yy).sum() / (old.sum() + yy.sum())
            deltas.append(float(f-b))
        ci = np.quantile(deltas, [.025, .975]).tolist()
        adopt = (best['repeat_f1'] >= baseline['f1'] + .001
                 and min(best['seed_f1']) >= baseline['f1']
                 and best['band_auc'] > baseline['band_auc'])
        report = dict(baseline=baseline, cutoff=cut, best_name=best_name, best=best,
            adopt=adopt, candidates=result, paired_gain_interval_95=ci,
            direction_gate='Mean F1 gain >=.001, every repeated seed nonnegative, boundary AUC increases.',
            scope='Outer validation labels excluded from direct residual fit. Cached base OOF models already used other fold labels: indirect leakage and prior selection remain. Fixed cutoff was selected in prior work; no new cutoff optimization. Hidden-test F1 is unknown.',
            labels='1=safe/nontoxic', neighbors='Mean Tanimoto of provided Morgan and chiral radius3, top32, denominator +1 shrinkage')
        (ROOT / 'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
