import glob, os
from blendlib import *
allnames, Oall, Tall = load()
def best_rate(y, s):
    best = max(((f1_score(y, (s >= np.quantile(s, 1 - r)).astype(int)), r) for r in np.arange(0.45, 0.70, 0.005)))
    return best[1]
print('| model | AUC | F1(pos) @best pos-rate | acc | macroF1 |'); print('|---|---|---|---|---|')
for n, o in sorted(zip(allnames, Oall), key=lambda x: -roc_auc_score(Y, x[1])):
    r = best_rate(Y, o); q = (o >= np.quantile(o, 1 - r)).astype(int)
    print(f'| {n} | {roc_auc_score(Y,o):.4f} | {f1_score(Y,q):.4f} | {accuracy_score(Y,q):.4f} | {f1_score(Y,q,average="macro"):.4f} |')
ex = set(open('exclude_pool.txt').read().strip().split(','))
names, O, T = load(ex)
C = 0.01; m = m_lr(C)
opt, nest = evaluate(O, m)
print('\nFINAL pool', names); print('LR stack C=0.01 optimistic:', fmt(opt)); print('LR stack C=0.01 nested    :', fmt(nest))
mdl = fit_lr(O.T, Y, C); print('coefs', {n: round(float(c), 3) for n, c in zip(names, mdl.coef_[0])})
# nested OOF scores + test scores saved for re-blending
sn = np.zeros(len(Y))
for tr, va in FOLDS: sn[va] = m(O[:, tr], Y[tr])(O[:, va])
test = mdl.predict_proba(logit(T.T))[:, 1]
np.savez('preds/final_stack_lr.npz', oof=sn, test=test)
# bootstrap SE of nested F1 at fixed per-fold thresholds approx: use pooled nested th
th = nest['th']; q = (sn >= th).astype(int); rng = np.random.default_rng(0)
bs = [f1_score(Y[i], q[i]) for i in (rng.integers(0, len(Y), len(Y)) for _ in range(300))]
print('nested F1 at th', th, f1_score(Y, q), 'bootstrap SE', np.std(bs).round(4), 'test pos rate at th', (test >= th).mean().round(3))
# greedy-rank blend for comparison
w = greedy(O, Y); print('greedy weights', {n: round(float(x), 3) for n, x in zip(names, w) if x > 0})
