"""Test raw-logit fusion around the best real submission, not rank fusion."""
from pathlib import Path
import json,numpy as np
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score,precision_recall_curve
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'07_개선';OLD=ROOT.parent/'08_재개선';A=np.load(OLD/'ensemble_inputs.npz');y=A['y'];names=list(A['names']);O=A['O'];oldT=A['T'];full=np.column_stack([np.load(ROOT/'full_predictions'/f'{n}.npy') for n in names])
extras=['bag_cp','cat_fp2','lgb_mord','svm_cb77','svm_molf']
for n in extras:
 z=np.load(PRIOR/'preds'/f'{n}.npz');o,t,f=z['oof'],z['test'],np.load(ROOT/'full_predictions'/f'{n}.npy')
 if n.startswith('svm_'):o,t,f=expit(o),expit(t),expit(f)
 names.append(n);O=np.column_stack([O,o]);oldT=np.column_stack([oldT,t]);full=np.column_stack([full,f])
X=logit(np.clip(O,1e-4,1-1e-4));Q=logit(np.clip(full,1e-4,1-1e-4));reference=np.load(OLD/'ensemble_predictions.npz')['oof'];band=(reference>=.35)&(reference<=.50)
def threshold(y,p):
 a,b,t=precision_recall_curve(y,p);f=2*a[:-1]*b[:-1]/(a[:-1]+b[:-1]+1e-15);return float(np.median(t[f>=f.max()-.0005]))
rows=[];cache={}
for name,width,C in [('best20',20,.1),('plus_cp',21,.1),('plus_three',23,.1),('plus_five',25,.1),('plus_five_reg',25,.03)]:
 ps=[];ds=[];cuts=[]
 for seed in [2026,730,731]:
  p=np.zeros(len(y));dec=np.zeros(len(y),int)
  for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
   inside=np.zeros(len(tr))
   for a,b in StratifiedKFold(3,shuffle=True,random_state=seed+1).split(X[tr],y[tr]):
    m=LogisticRegression(C=C,max_iter=2500).fit(X[tr[a],:width],y[tr[a]]);inside[b]=m.predict_proba(X[tr[b],:width])[:,1]
   cut=threshold(y[tr],inside);cuts.append(cut);m=LogisticRegression(C=C,max_iter=2500).fit(X[tr,:width],y[tr]);p[va]=m.predict_proba(X[va,:width])[:,1];dec[va]=p[va]>=cut
  ps.append(p);ds.append(dec)
 mean=np.mean(ps,axis=0);scores=[float(f1_score(y,d)) for d in ds];cut=float(np.median(cuts));row={'name':name,'width':width,'C':C,'auc':float(roc_auc_score(y,mean)),'f1':float(np.mean(scores)),'seed_f1':scores,'cut':cut,'band_auc':float(roc_auc_score(y[band],mean[band]))}
 rows.append(row);cache[name]=(mean,np.array(ds));print('RAW FUSION',json.dumps(row),flush=True)
base=rows[0];best=max(rows,key=lambda r:r['f1']);adopt=best['f1']>=base['f1']+.001 and all(a>=b for a,b in zip(best['seed_f1'],base['seed_f1'])) and best['band_auc']>base['band_auc']
report={'baseline':base,'best_exploratory':best,'adopt':adopt,'candidates':rows,'scope':'Repeated meta CV reuses base OOF; indirect leakage and prior selection remain.'};np.savez(ROOT/'probability_fusion_inputs.npz',O=O,old_test=oldT,full_test=full,y=y,names=np.array(names));np.savez(ROOT/'probability_fusion_selected.npz',oof=cache[best['name']][0],decisions=cache[best['name']][1],y=y)
rng=np.random.default_rng(48);deltas=[];bd=cache[base['name']][1];nd=cache[best['name']][1]
for _ in range(1000):
 ids=rng.integers(len(y),size=len(y));yy=y[ids]
 def f(q):return np.mean(2*(q[:,ids]*yy).sum(axis=1)/(q[:,ids].sum(axis=1)+yy.sum()))
 deltas.append(float(f(nd)-f(bd)))
report['paired_gain_interval_95']=np.quantile(deltas,[.025,.975]).tolist();(ROOT/'probability_fusion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
