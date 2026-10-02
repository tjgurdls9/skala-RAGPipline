import sys
from zoo import *
M = np.load(ROOT / 'mordred.npz')
XM, XMP = np.hstack([X2, M['tr']]), np.hstack([X2P, M['pr']])
M5 = {
 'lgb_mord': (lgb_fp(random_state=3, num_leaves=63, n_estimators=1100, colsample_bytree=0.25, n_jobs=5), XM, XMP),
 'lgb_mordonly': (lgb_fp(random_state=3, num_leaves=31, n_estimators=600, colsample_bytree=0.4, n_jobs=5), M['tr'], M['pr']),
}
if __name__ == '__main__':
    for n in sys.argv[1:]:
        fn, X, Xp = M5[n]; cv(n, fn, X, Xp)
