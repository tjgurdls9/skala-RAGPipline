"""Fixed-structure LightGBM with generalized CE gradients; two declared q values.
Positive Fisher-like Hessian avoids the true GCE Hessian's non-convex tail.
No pseudo labels or out-of-fold confidence weighting enters model fitting.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.special import expit
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,f1_score,precision_recall_curve
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'07_개선'))
import zoo
Y=zoo.Y;X=zoo.X2;XP=zoo.X2P
P=dict(num_leaves=63,learning_rate=.03,colsample_bytree=.25,subsample=.8,subsample_freq=1,min_child_samples=20,reg_lambda=1.,n_jobs=2,verbosity=-1,seed=3)
base=np.load(PROJECT/'07_개선/preds/lgb_fp2.npz');reference=np.load(PROJECT/'10_최종개선/probability_fusion_selected.npz')['oof'];band=(reference>=.35)&(reference<=.5)
def cut(y,p):
 a,b,t=precision_recall_curve(y,p);f=2*a[:-1]*b[:-1]/(a[:-1]+b[:-1]+1e-15);return float(np.median(t[f>=f.max()-.0005]))
def metric(p):
 d=np.zeros(len(Y),int);cuts=[]
 for tr,va in zoo.FOLDS:
  c=cut(Y[tr],p[tr]);cuts.append(c);d[va]=p[va]>=c
 return dict(auc=float(roc_auc_score(Y,p)),f1_crossfit_cut=float(f1_score(Y,d)),band_auc=float(roc_auc_score(Y[band],p[band])),cut=float(np.median(cuts)))
def objective(q):
 def fn(preds,data):
  y=data.get_label();p=expit(preds);pt=np.where(y==1,p,1-p);w=pt**q
  return w*(p-y),np.maximum(w*p*(1-p),1e-5)
 return fn
# Numerical gradient check, independent of the optimizer.
r=np.linspace(-3,3,31);yy=(np.arange(len(r))%2).astype(float)
for q in [.2,.5]:
 def loss(z):pt=np.where(yy==1,expit(z),expit(-z));return (1-pt**q)/q
 pt=np.where(yy==1,expit(r),expit(-r));g=pt**q*(expit(r)-yy)
 assert np.allclose(g,(loss(r+1e-6)-loss(r-1e-6))/2e-6,atol=1e-8)
rows={'baseline':metric(base['oof']),'settings':P,'method':'GCE exact gradients; positive Fisher-like approximate Hessian; q=.2,.5 fixed before results; 1100 trees, no early stopping or validation labels'}
(ROOT/'results.json').write_text(json.dumps(rows,indent=2));print('BASE',rows,flush=True)
for q in [.2,.5]:
 tag=f'gce_q{q:g}';oof=np.zeros(len(Y));tests=[];started=time.time()
 for fold,(tr,va) in enumerate(zoo.FOLDS):
  file=ROOT/f'{tag}_fold{fold}.npz'
  if file.exists():z=np.load(file);pv,pt=z['valid'],z['test']
  else:
   model=lgb.train(dict(P,objective=objective(q)),lgb.Dataset(X[tr],label=Y[tr]),num_boost_round=1100)
   pv=expit(model.predict(X[va]));pt=expit(model.predict(XP));np.savez(file,valid=pv,test=pt,valid_indices=va)
  oof[va]=pv;tests.append(pt);print(tag,'fold',fold,'AUC',roc_auc_score(Y[va],pv),'seconds',round(time.time()-started),flush=True)
 test=np.mean(tests,axis=0);np.savez(ROOT/f'{tag}.npz',oof=oof,test=test,y=Y)
 rows[tag]=metric(oof);rows[tag]['seconds']=time.time()-started;(ROOT/'results.json').write_text(json.dumps(rows,indent=2));print('COMPLETE',tag,rows[tag],flush=True)
print('ALL FINISHED',flush=True)
