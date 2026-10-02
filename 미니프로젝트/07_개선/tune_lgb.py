"""Optuna TPE over LGBM on fp2 matrix; scored on folds 0,1 of the fixed split (early-stopped, tuning only)."""
import json, time, optuna, lightgbm as lgb
from zoo import *
TUNE_FOLDS = FOLDS[:2]
def obj(t):
    p = dict(n_estimators=1500, learning_rate=0.05, n_jobs=-1, verbose=-1, random_state=3, subsample_freq=1,
             num_leaves=t.suggest_int('num_leaves', 15, 255, log=True),
             min_child_samples=t.suggest_int('min_child_samples', 3, 60, log=True),
             colsample_bytree=t.suggest_float('colsample_bytree', 0.05, 0.6, log=True),
             subsample=t.suggest_float('subsample', 0.5, 1.0),
             reg_lambda=t.suggest_float('reg_lambda', 1e-2, 30, log=True),
             reg_alpha=t.suggest_float('reg_alpha', 1e-3, 5, log=True),
             max_bin=t.suggest_categorical('max_bin', [15, 31, 63, 255]))
    aucs, its = [], []
    for tr, va in TUNE_FOLDS:
        m = lgb.LGBMClassifier(**p).fit(X2[tr], Y[tr], eval_set=[(X2[va], Y[va])], eval_metric='auc',
                                        callbacks=[lgb.early_stopping(100, verbose=False)])
        aucs.append(m.best_score_['valid_0']['auc']); its.append(m.best_iteration_)
    t.set_user_attr('iters', its)
    return sum(aucs) / len(aucs)
if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial(dict(num_leaves=63, min_child_samples=20, colsample_bytree=0.25, subsample=0.8, reg_lambda=1.0, reg_alpha=1e-3, max_bin=255))
    t0 = time.time()
    for i in range(36):
        st.optimize(obj, n_trials=1)
        b = st.best_trial; l = st.trials[-1]
        print(f'{i} {time.time()-t0:.0f}s auc {l.value:.4f} its {l.user_attrs["iters"]} best {b.value:.4f}', flush=True)
        json.dump({'params': b.params, 'iters': b.user_attrs['iters'], 'auc': b.value,
                   'all': [(x.value, x.params, x.user_attrs.get('iters')) for x in st.trials if x.value]},
                  open('tune_lgb.json', 'w'), indent=1)
