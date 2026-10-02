"""Reuse Claude's feature/prediction caches; write only to 08_재개선."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                            precision_score, recall_score, roc_auc_score, log_loss)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'07_개선'
PRED = ROOT/'preds'
PRED.mkdir(exist_ok=True)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def data():
    d = np.load(PRIOR/'features.npz', allow_pickle=True)
    extra = np.load(PRIOR/'fp2.npz')
    original = pd.read_csv(ROOT.parent/'01_원본자료/train_original_labels.csv.bak', usecols=['SMILES', 'label'])
    incoming = pd.read_csv(ROOT.parent/'01_원본자료/predict_input.csv', usecols=['SMILES'])
    assert np.array_equal(d['y'], original.label) and np.array_equal(d['smiles_tr'], original.SMILES)
    assert np.array_equal(d['smiles_pr'], incoming.SMILES)
    matrices = {}
    for tag in ['tr', 'pr']:
        blocks = [d[f'given_{tag}'], d[f'desc_{tag}'], d[f'cnt_{tag}'], d[f'r3_{tag}'], d[f'maccs_{tag}']]
        blocks += [extra[f'{name}_{tag}'] for name in ['ap', 'tt', 'rd', 'av']]
        matrices[tag] = np.hstack(blocks).astype(np.float32)
        assert np.isfinite(matrices[tag]).all()
    keep = np.std(matrices['tr'], axis=0) > 0
    return d, matrices['tr'][:, keep], matrices['pr'][:, keep]


def threshold(y, p):
    precision, recall, th = precision_recall_curve(y, p)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(precision), where=precision+recall>0)[:-1]
    feasible = (th >= .05) & (th <= .95)
    best = float(f1[feasible].max())
    # ponytail: median near-optimal cutoff stabilizes discrete F1; nested threshold CV if more labels arrive.
    plateau = th[feasible & (f1 >= best-.0005)]
    return float(np.median(plateau)), best, [float(plateau.min()), float(plateau.max())]


def metrics(y, p, cutoff=None):
    cutoff = threshold(y, p)[0] if cutoff is None else cutoff
    q = p >= cutoff
    return {'auc': float(roc_auc_score(y, p)), 'f1_safe': float(f1_score(y, q)),
            'accuracy': float(accuracy_score(y, q)), 'precision_safe': float(precision_score(y, q)),
            'recall_safe': float(recall_score(y, q)), 'recall_toxic': float(recall_score(y, q, pos_label=0)),
            'threshold': float(cutoff), 'positive_rate': float(q.mean()),
            'log_loss': float(log_loss(y, np.clip(p, 1e-7, 1-1e-7)))}


CONFIGS = {
    'lgb_leaf31': dict(num_leaves=31, min_child_samples=20, reg_lambda=2, colsample_bytree=.35),
    'lgb_leaf63_reg': dict(num_leaves=63, min_child_samples=15, reg_lambda=5, reg_alpha=.1, colsample_bytree=.3),
    'lgb_leaf127_reg': dict(num_leaves=127, min_child_samples=12, reg_lambda=5, colsample_bytree=.2),
    'lgb_leaf31_lowchild': dict(num_leaves=31, min_child_samples=8, reg_lambda=1, colsample_bytree=.2),
    'lgb_extra31': dict(num_leaves=31, min_child_samples=10, reg_lambda=2, colsample_bytree=.4, extra_trees=True),
    'lgb_extra63': dict(num_leaves=63, min_child_samples=15, reg_lambda=3, colsample_bytree=.3, extra_trees=True),
    'lgb_smooth': dict(num_leaves=63, min_child_samples=10, reg_lambda=2, colsample_bytree=.25, path_smooth=5),
    'lgb_large': dict(num_leaves=127, min_child_samples=5, reg_lambda=1, colsample_bytree=.15),
}


def fit_lgb(name, X, Q, y, folds, seed, screened=False):
    config = CONFIGS[name]
    oof = np.full(len(y), np.nan)
    test = np.zeros(len(Q))
    fold_scores = []
    begin = perf_counter()
    for fold, (tr, va) in enumerate(folds):
        model = LGBMClassifier(n_estimators=1400, learning_rate=.025, subsample=.85,
                               subsample_freq=1, n_jobs=4, verbosity=-1,
                               deterministic=True, force_col_wise=True, random_state=seed,
                               **config).fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        test += model.predict_proba(Q)[:, 1]/len(folds)
        fold_scores.append(metrics(y[va], oof[va]))
        print(name, 'seed', seed, 'fold', fold+1, 'AUC', round(fold_scores[-1]['auc'], 5),
              'elapsed', round(perf_counter()-begin, 1), flush=True)
    valid = np.isfinite(oof)
    result = metrics(y[valid], oof[valid])
    result.update(name=name, seed=seed, config=config, feature_count=X.shape[1],
                  seconds=perf_counter()-begin, folds=fold_scores, screening=screened)
    if not screened:
        np.savez_compressed(PRED/f'{name}_seed{seed}.npz', oof=oof, test=test)
        write_json(PRED/f'{name}_seed{seed}.json', result)
    return result


def build_extra(d):
    path = ROOT/'chiral_counts.npz'
    if path.exists():
        cache = np.load(path)
        return cache['tr'], cache['pr']
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    generators = [rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True),
                  rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())]
    blocks = {}
    for tag in ['tr', 'pr']:
        records = []
        for s in d[f'smiles_{tag}']:
            mol = Chem.MolFromSmiles(s)
            assert mol is not None, s
            records.append(np.hstack([g.GetCountFingerprintAsNumPy(mol) for g in generators]))
        blocks[tag] = np.array(records, dtype=np.float32)
    np.savez_compressed(path, **blocks)
    return blocks['tr'], blocks['pr']


def tune():
    d, X, Q = data()
    y = d['y'].astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    screen = []
    for name in CONFIGS:
        screen.append(fit_lgb(name, X, Q, y, folds[:2], 42, screened=True))
        write_json(ROOT/'tuning_screen.json', screen)
    winners = sorted(screen, key=lambda r: (r['f1_safe'], r['auc']), reverse=True)[:3]
    print('SCREEN WINNERS', [r['name'] for r in winners], flush=True)
    results = []
    for result in winners:
        results.append(fit_lgb(result['name'], X, Q, y, folds, 42))
        write_json(ROOT/'tuning_full.json', results)
    a, b = build_extra(d)
    Xextra, Qextra = np.hstack([X, a]), np.hstack([Q, b])
    keep = np.std(Xextra, axis=0) > 0
    CONFIGS['lgb_chiralcounts'] = dict(num_leaves=63, min_child_samples=15, reg_lambda=3, colsample_bytree=.25)
    results.append(fit_lgb('lgb_chiralcounts', Xextra[:, keep], Qextra[:, keep], y, folds, 42))
    write_json(ROOT/'tuning_full.json', results)


def tanimoto(A, B):
    intersection = A @ B.T
    union = A.sum(axis=1)[:, None] + B.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union>0)


def tune_xgb():
    from xgboost import XGBClassifier
    d, X, Q = data()
    y = d['y'].astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    configurations = {
        'xgb_depth5_reg': dict(max_depth=5, min_child_weight=3, reg_lambda=3, colsample_bytree=.4),
        'xgb_depth7_reg': dict(max_depth=7, min_child_weight=8, reg_lambda=8, colsample_bytree=.25),
    }
    def run(name, selected_folds, screen):
        oof, test = np.full(len(y), np.nan), np.zeros(len(Q))
        begin = perf_counter()
        for fold, (tr, va) in enumerate(selected_folds):
            model = XGBClassifier(n_estimators=1600, learning_rate=.025, subsample=.85,
                reg_alpha=.1, tree_method='hist', n_jobs=4, random_state=42,
                eval_metric='logloss', **configurations[name]).fit(X[tr], y[tr])
            oof[va] = model.predict_proba(X[va])[:, 1]
            test += model.predict_proba(Q)[:, 1]/len(selected_folds)
            print(name, fold+1, 'elapsed', round(perf_counter()-begin, 1), flush=True)
        finite = np.isfinite(oof)
        result = {'name': name, 'config': configurations[name], 'seconds': perf_counter()-begin,
                  'screening': screen, **metrics(y[finite], oof[finite])}
        if not screen:
            np.savez_compressed(PRED/f'{name}_seed42.npz', oof=oof, test=test)
            write_json(PRED/f'{name}_seed42.json', result)
        return result
    screen = [run(name, folds[:2], True) for name in configurations]
    write_json(ROOT/'xgb_screen.json', screen)
    winner = max(screen, key=lambda r: (r['f1_safe'], r['auc']))['name']
    run(winner, folds, False)


def kernel():
    d, _, _ = data()
    y = d['y'].astype(int)
    # Morgan-only similarity avoids dense Pattern bits dominating molecular similarity.
    cache = np.load(ROOT.parent/'06_모델링/feature_cache.npz')
    A, B = cache['X'][:, :1024].astype(np.float32), cache['Q'][:, :1024].astype(np.float32)
    K, T = tanimoto(A, A), tanimoto(B, A)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(A, y))
    for C in [.3, 1., 3., 10., 30.]:
        name = f'svm_tanimoto_C{C:g}'
        if (PRED/f'{name}_seed42.npz').exists():
            continue
        oof, test = np.zeros(len(y)), np.zeros(len(B))
        begin = perf_counter()
        for fold, (tr, va) in enumerate(folds):
            model = SVC(C=C, kernel='precomputed', cache_size=512).fit(K[np.ix_(tr, tr)], y[tr])
            # decision sigmoid is only a monotonic score; final calibration is learned on OOF.
            oof[va] = expit(model.decision_function(K[np.ix_(va, tr)]))
            test += expit(model.decision_function(T[:, tr]))/len(folds)
            print('tanimoto SVM', C, fold+1, 'elapsed', round(perf_counter()-begin, 1), flush=True)
        np.savez_compressed(PRED/f'{name}_seed42.npz', oof=oof, test=test)
        write_json(PRED/f'{name}_seed42.json', {'name': name, 'C': C, 'seconds': perf_counter()-begin,
                                             **metrics(y, oof)})


def multi_kernel():
    d = np.load(PRIOR/'features.npz', allow_pickle=True)
    y = d['y'].astype(int)
    extra = np.load(PRIOR/'fp2.npz')
    K, T = np.zeros((len(y), len(y)), dtype=np.float32), np.zeros((927, len(y)), dtype=np.float32)
    blocks = [(d['given_tr'][:, :1024], d['given_pr'][:, :1024]),
              (d['given_tr'][:, 1024:2048], d['given_pr'][:, 1024:2048]),
              (d['cnt_tr'], d['cnt_pr']), (extra['rd_tr'], extra['rd_pr'])]
    for A, B in blocks:
        A, B = A.astype(np.float32), B.astype(np.float32)
        norms = np.square(A).sum(axis=1)
        inter = A @ A.T
        union = norms[:, None]+norms[None, :]-inter
        K += np.divide(inter, union, out=np.zeros_like(inter), where=union>0)/len(blocks)
        inter = B @ A.T
        union = np.square(B).sum(axis=1)[:, None]+norms[None, :]-inter
        T += np.divide(inter, union, out=np.zeros_like(inter), where=union>0)/len(blocks)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1)
    props, qprops = d['given_tr'][:, -4:], d['given_pr'][:, -4:]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(props, y))
    for C, prop_weight in [(1., 0.), (3., 0.), (10., 0.), (3., .15), (10., .15)]:
        name = f'svm_multi_C{C:g}_prop{prop_weight:g}'
        oof, test = np.zeros(len(y)), np.zeros(927)
        begin = perf_counter()
        for fold, (tr, va) in enumerate(folds):
            Kfit, Kval, Ktest = K[np.ix_(tr, tr)], K[np.ix_(va, tr)], T[:, tr]
            if prop_weight:
                scaler = StandardScaler().fit(props[tr])
                fitprop = scaler.transform(props[tr])
                Kfit = (1-prop_weight)*Kfit+prop_weight*rbf_kernel(fitprop, gamma=.25)
                Kval = (1-prop_weight)*Kval+prop_weight*rbf_kernel(scaler.transform(props[va]), fitprop, gamma=.25)
                Ktest = (1-prop_weight)*Ktest+prop_weight*rbf_kernel(scaler.transform(qprops), fitprop, gamma=.25)
            model = SVC(C=C, kernel='precomputed', cache_size=512).fit(Kfit, y[tr])
            oof[va] = expit(model.decision_function(Kval))
            test += expit(model.decision_function(Ktest))/5
            print('multi SVM', C, prop_weight, fold+1, 'elapsed', round(perf_counter()-begin, 1), flush=True)
        np.savez_compressed(PRED/f'{name}_seed42.npz', oof=oof, test=test)
        write_json(PRED/f'{name}_seed42.json', {'name': name, 'C': C, 'property_weight': prop_weight,
                  'seconds': perf_counter()-begin, **metrics(y, oof)})


def collect():
    names, O, T = [], [], []
    for directory in [PRIOR/'preds', PRED]:
        for path in sorted(directory.glob('*.npz')):
            if directory == PRED and not path.stem.endswith('_seed42'):
                continue
            z = np.load(path)
            o, t = z['oof'], z['test']
            repeated = directory/f'{path.stem.removesuffix("_seed42")}_seed2026.npz'
            if directory == PRED and repeated.exists():
                additional = np.load(repeated)
                o, t = (o+additional['oof'])/2, (t+additional['test'])/2
                print('Repeat-average:', path.stem, flush=True)
            assert o.shape == (8349,) and t.shape == (927,) and np.isfinite(o).all() and np.isfinite(t).all()
            names.append(path.stem); O.append(np.clip(o, 1e-6, 1-1e-6)); T.append(np.clip(t, 1e-6, 1-1e-6))
    return names, np.array(O).T, np.array(T).T


def convex_weights(O, y):
    count = O.shape[1]
    def loss(w):
        p = np.clip(O @ w, 1e-6, 1-1e-6)
        return -(y*np.log(p)+(1-y)*np.log1p(-p)).mean()+.01*np.square(w).sum()
    result = minimize(loss, np.full(count, 1/count), method='SLSQP', bounds=[(0, 1)]*count,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1}], options={'maxiter': 250, 'ftol': 1e-9})
    assert result.success, result.message
    return result.x


def greedy_weights(O, y, goal):
    current, chosen, best = np.zeros(len(y)), [], -1.
    for _ in range(30):
        candidates = [(current*len(chosen)+O[:, i])/(len(chosen)+1) for i in range(O.shape[1])]
        values = [roc_auc_score(y, p) if goal == 'auc' else threshold(y, p)[1] for p in candidates]
        index = int(np.argmax(values))
        if values[index] <= best+1e-6:
            break
        current, best = candidates[index], values[index]
        chosen.append(index)
    return np.bincount(chosen, minlength=O.shape[1])/len(chosen)


def meta_candidate(O, T, y, folds, mode, C=1, context=None):
    oof, test = np.zeros(len(y)), np.zeros(len(T))
    fold_cutoffs, decisions = [], np.zeros(len(y), dtype=int)
    coefs = []
    for tr, va in folds:
        if mode.startswith('greedy'):
            sorted_scores = np.sort(O[tr], axis=0)
            train_rank = rankdata(O[tr], axis=0)/len(tr)
            valid_rank = np.column_stack([np.searchsorted(sorted_scores[:, j], O[va, j], side='right')/len(tr)
                                         for j in range(O.shape[1])])
            test_rank = np.column_stack([np.searchsorted(sorted_scores[:, j], T[:, j], side='right')/len(tr)
                                        for j in range(O.shape[1])])
            coef = greedy_weights(train_rank, y[tr], mode.split('_')[-1])
            fitted, validated, predicted = train_rank @ coef, valid_rank @ coef, test_rank @ coef
        elif mode == 'convex':
            coef = convex_weights(O[tr], y[tr])
            fitted, validated, predicted = O[tr] @ coef, O[va] @ coef, T @ coef
        else:
            fit_inputs, valid_inputs, test_inputs = logit(O[tr]), logit(O[va]), logit(T)
            if mode == 'context':
                ctx, ctx_test = context
                scaler = StandardScaler().fit(ctx[tr])
                def contextual(base, properties):
                    properties = np.clip(scaler.transform(properties), -5, 5)
                    interactions = (base[:, :, None]*properties[:, None, :]).reshape(len(base), -1)
                    return np.hstack([base, properties, interactions])
                fit_inputs = contextual(fit_inputs, ctx[tr])
                valid_inputs = contextual(valid_inputs, ctx[va])
                test_inputs = contextual(test_inputs, ctx_test)
            model = LogisticRegression(C=C, max_iter=3000).fit(fit_inputs, y[tr])
            fitted = model.predict_proba(fit_inputs)[:, 1]
            validated = model.predict_proba(valid_inputs)[:, 1]
            predicted = model.predict_proba(test_inputs)[:, 1]
            coef = model.coef_[0]
        cut = threshold(y[tr], fitted)[0]
        fold_cutoffs.append(cut)
        decisions[va] = validated >= cut
        oof[va] = validated
        test += predicted/len(folds)
        coefs.append(coef)
    return oof, test, decisions, {'cross_threshold_f1': float(f1_score(y, decisions)),
                        'fold_thresholds': fold_cutoffs, 'coefficients': np.mean(coefs, axis=0).tolist()}


def ensemble():
    d = np.load(PRIOR/'features.npz', allow_pickle=True)
    y = d['y'].astype(int)
    names, O, T = collect()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(O, y))
    individual = [{'name': n, **metrics(y, O[:, i])} for i, n in enumerate(names)]
    pd.DataFrame(individual).sort_values('f1_safe', ascending=False).to_csv(ROOT/'candidate_comparison.csv', index=False)
    # Do not let weak/redundant models overwhelm the regularized meta model.
    keep = [i for i, r in enumerate(individual) if r['auc'] >= .86]
    names = [names[i] for i in keep]
    O, T = O[:, keep], T[:, keep]
    scaffold_seen = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(O, y):
        scaffold_seen[va] = np.isin(d['scaf_tr'][va], d['scaf_tr'][tr])
    context = (np.column_stack([d['given_tr'][:, -4:], scaffold_seen]),
               np.column_stack([d['given_pr'][:, -4:], np.isin(d['scaf_pr'], d['scaf_tr'])]))
    rows, predictions = [], {}
    for mode, C in [('convex', 1), ('greedy_auc', 1), ('greedy_f1', 1),
                    *[('logistic', c) for c in [.01, .03, .1, .3, 1., 3.]],
                    *[('context', c) for c in [.003, .01, .03, .1]]]:
        name = f'{mode}_C{C:g}'
        p, q, decisions, details = meta_candidate(O, T, y, folds, mode, C, context)
        row = {'name': name, **metrics(y, p), **details}
        rows.append(row); predictions[name] = (p, q, decisions)
        print('META', name, row['auc'], row['f1_safe'], row['cross_threshold_f1'], flush=True)
    selected = max(rows, key=lambda r: (r['cross_threshold_f1'], r['auc']))
    p, q, decisions = predictions[selected['name']]
    cut, best, interval = threshold(y, p)
    # Blend estimates of plateau cutoff and per-fold threshold to reduce selection jitter.
    stable_cut = float(np.median([cut, *selected['fold_thresholds']]))
    write_json(ROOT/'meta_comparison.json', rows)
    write_json(ROOT/'ensemble_selection.json', {'name': selected['name'], 'models': names,
               'selected_metrics': metrics(y, p, stable_cut), 'threshold_plateau': interval,
               'global_plateau_cutoff': cut, 'stable_cutoff': stable_cut, 'coefficients': selected['coefficients'],
               'fold_thresholds': selected['fold_thresholds'],
               'scope': 'Meta cross-fit reuses existing base OOF; selection estimate, not fully nested unbiased CV.'})
    np.savez_compressed(ROOT/'ensemble_predictions.npz', oof=p, test=q, cross_threshold_prediction=decisions)
    np.savez_compressed(ROOT/'ensemble_inputs.npz', O=O, T=T, names=np.array(names), y=y)
    template = pd.read_csv(ROOT.parent/'01_원본자료/predict_output_홍길동.csv')
    assert np.array_equal(template.SMILES, d['smiles_pr'])
    submission = template.copy()
    submission['output'] = (q >= stable_cut).astype(int)
    target = ROOT/'predict_output_서혁인_개선2차.csv'
    submission.to_csv(target, index=False, encoding='utf-8-sig')
    assert list(submission) == ['SMILES', 'output'] and len(submission) == 927
    assert pd.read_csv(target).equals(submission)
    ready = ROOT/'제출용/predict_output_서혁인.csv'
    ready.parent.mkdir(exist_ok=True)
    submission.to_csv(ready, index=False, encoding='utf-8-sig')
    assert pd.read_csv(ready).equals(submission)
    print('SAVED', target, 'positive rate', submission.output.mean(), flush=True)


def repeat(names, seed):
    d, X, Q = data()
    y = d['y'].astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    for name in names:
        if name == 'lgb_chiralcounts':
            CONFIGS[name] = dict(num_leaves=63, min_child_samples=15, reg_lambda=3, colsample_bytree=.25)
            a, b = build_extra(d)
            Xe, Qe = np.hstack([X, a]), np.hstack([Q, b])
            keep = np.std(Xe, axis=0)>0
            fit_lgb(name, Xe[:, keep], Qe[:, keep], y, folds, seed)
        else:
            fit_lgb(name, X, Q, y, folds, seed)


def connected_cv():
    d, X, Q = data()
    y = d['y'].astype(int)
    cache = np.load(ROOT.parent/'06_모델링/feature_cache.npz')
    connectivity = cache['connectivity']
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, connectivity))
    for tr, va in folds:
        assert not set(connectivity[tr]) & set(connectivity[va])
    result = fit_lgb('lgb_leaf31_lowchild', X, Q, y, folds, 31415)
    write_json(ROOT/'connected_group_cv.json', result)
    print('PASS: connected-family CV, all opposite stereoisomers grouped', flush=True)


def self_check():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .2, .8, .9])
    assert metrics(y, p)['f1_safe'] == metrics(y, p)['auc'] == 1
    A = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
    K = tanimoto(A, A)
    assert np.allclose(K, [[1, 1/3], [1/3, 1]])
    O = np.column_stack([p, np.full(4, .5)])
    weights = convex_weights(O, y)
    assert weights[0]>.99 and abs(weights.sum()-1)<1e-8
    for goal in ['auc', 'f1']:
        weights = greedy_weights(O, y, goal)
        assert weights[0] == 1 and metrics(y, O @ weights)['f1_safe'] == 1
    print('PASS: F1 positive=safe, threshold plateau, kernel, convex blend')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['tune', 'xgb', 'kernel', 'multi-kernel', 'ensemble', 'repeat', 'connected', 'self-check'])
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.action == 'tune': tune()
    elif args.action == 'xgb': tune_xgb()
    elif args.action == 'kernel': kernel()
    elif args.action == 'multi-kernel': multi_kernel()
    elif args.action == 'ensemble': ensemble()
    elif args.action == 'repeat': repeat(args.names, args.seed)
    elif args.action == 'connected': connected_cv()
    else: self_check()
