"""Extra models on top of zoo.py (same folds, same cache dir)."""
import sys
from zoo import *
T1 = dict(num_leaves=59, min_child_samples=15, colsample_bytree=0.216, subsample=0.802, reg_lambda=20.1, reg_alpha=0.012, max_bin=63)
T2 = dict(num_leaves=91, min_child_samples=17, colsample_bytree=0.583, subsample=0.875, reg_lambda=23.9, reg_alpha=0.019, max_bin=63)
T3 = dict(num_leaves=95, min_child_samples=24, colsample_bytree=0.378, subsample=0.884, reg_lambda=27.1, reg_alpha=0.009, max_bin=63)
M2 = {
 'lgb_t1': (lgb_fp(learning_rate=0.05, n_estimators=420, random_state=1, **T1), X2, X2P),
 'lgb_t2': (lgb_fp(learning_rate=0.05, n_estimators=330, random_state=2, **T2), X2, X2P),
 'lgb_t3': (lgb_fp(learning_rate=0.05, n_estimators=420, random_state=3, **T3), X2, X2P),
}
if __name__ == '__main__':
    for n in sys.argv[1:]:
        fn, X, Xp = M2[n]; cv(n, fn, X, Xp)
