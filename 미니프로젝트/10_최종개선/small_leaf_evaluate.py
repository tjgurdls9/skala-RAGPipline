from pathlib import Path
import json,numpy as np
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score,precision_recall_curve
ROOT=Path(__file__).resolve().parent
preds=[];ys=[];cuts=[];reports=[]
def cut(y,p):
 a,b,t=precision_recall_curve(y,p);f=2*a[:-1]*b[:-1]/(a[:-1]+b[:-1]+1e-15);return float(np.median(t[f>=f.max()-.001]))
for fold in range(5):
 folder=ROOT/('strict_probe' if fold==0 else f'strict_probe_fold{fold}')
 old=np.load(folder/'evaluation_predictions.npz');zs=[np.load(folder/f'{n}.npz') for n in ['lgb_fp2','svm_mk_p2_c3','chemprop']]+[np.load(ROOT/f'small_leaf_fold{fold}.npz')]
 D=np.load(ROOT.parent/'07_개선/features.npz',allow_pickle=True);y=D['y'].astype(int);hold=old['hold_ids'];outer=next(tr for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y) if np.array_equal(va,hold));yt=y[outer];yh=y[hold]
 O=np.column_stack([z['inner_oof'] for z in zs]);F=np.column_stack([z['hold_full'] for z in zs]);A=logit(np.clip(rankdata(O,axis=0)/len(O),1e-4,1-1e-4));innerp=np.zeros(len(outer))
 for tr,va in StratifiedKFold(4,shuffle=True,random_state=731).split(A,yt):
  m=LogisticRegression(C=.01,max_iter=2000).fit(A[tr],yt[tr]);innerp[va]=m.predict_proba(A[va])[:,1]
 c=cut(yt,innerp);m=LogisticRegression(C=.01,max_iter=2000).fit(A,yt);p=m.predict_proba(logit(np.clip(rankdata(F,axis=0)/len(F),1e-4,1-1e-4)))[:,1]
 preds.append(p);ys.append(yh);cuts.append(np.full(len(yh),c));reports.append({'fold':fold,'baseline_refit_f1':float(f1_score(yh,old['refit']>=old['cut'])),'small_leaf_added_f1':float(f1_score(yh,p>=c))})
Y=np.concatenate(ys);P=np.concatenate(preds);C=np.concatenate(cuts);base=np.load(ROOT/'strict_nested_predictions.npz');old=base['refit']>=base['cuts'];q=P>=C;assert np.array_equal(Y,base['y']);band=(base['fold_average']>=.35)&(base['fold_average']<=.50)
rng=np.random.default_rng(17);delta=[]
for _ in range(1000):
 a=rng.integers(len(Y),size=len(Y));yy=Y[a]
 def f(p):return 2*(p[a]*yy).sum()/(p[a].sum()+yy.sum())
 delta.append(float(f(q)-f(old)))
report={'baseline_refit_f1':float(f1_score(Y,old)),'small_leaf_added_f1':float(f1_score(Y,q)),'gain':float(f1_score(Y,q)-f1_score(Y,old)),'folds':reports,'paired_gain_interval_95':np.quantile(delta,[.025,.975]).tolist(),'band_auc_gain':float(roc_auc_score(Y[band],P[band])-roc_auc_score(Y[band],base['refit'][band])),'scope':'4-model fixed-parameter nested fitting experiment; earlier global model tuning remains a caveat.'}
report['adopt_direction']=report['gain']>=.002 and report['band_auc_gain']>0
(ROOT/'small_leaf_evaluation.json').write_text(json.dumps(report,indent=2));np.savez(ROOT/'small_leaf_nested_predictions.npz',y=Y,oof=P,cuts=C);print(json.dumps(report,indent=2))
