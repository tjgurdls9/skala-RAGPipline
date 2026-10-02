"""Extra small-leaf LGBM: rebuild inner OOF separately for every outer fold."""
from pathlib import Path
import os,sys,time,json
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'07_개선'))
from zoo import Y,X2,FOLDS
for fold,(outer,hold) in enumerate(FOLDS):
 target=ROOT/f'small_leaf_fold{fold}.npz'
 if target.exists():continue
 t=time.time();o=np.zeros(len(outer));bags=[]
 def fit(tr,va):
  assert not set(tr)&set(hold)
  m=lgb.LGBMClassifier(n_estimators=1400,learning_rate=.025,num_leaves=31,min_child_samples=8,reg_lambda=1.,colsample_bytree=.2,subsample=.85,subsample_freq=1,random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True).fit(X2[tr],Y[tr])
  return m.predict_proba(X2[va])[:,1],m.predict_proba(X2[hold])[:,1]
 for a,b in StratifiedKFold(4,shuffle=True,random_state=730).split(outer,Y[outer]):
  p,q=fit(outer[a],outer[b]);o[b]=p;bags.append(q)
 _,full=fit(outer,hold);np.savez(target,inner_oof=o,hold_bag=np.mean(bags,axis=0),hold_full=full)
 print('SMALL LEAF DONE FOLD',fold,'seconds',round(time.time()-t),flush=True)
