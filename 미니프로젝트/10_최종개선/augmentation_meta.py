"""One frozen comparison: add augmented ChemBERTa to the 16-model stack."""
from pathlib import Path
import sys,json,numpy as np
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score,precision_recall_curve
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'07_개선'
y=np.load(PRIOR/'features.npz',allow_pickle=True)['y'].astype(int);names=json.loads((ROOT/'claude_audit.json').read_text())['pool'];cols=[np.load(PRIOR/'preds'/f'{n}.npz')['oof'] for n in names]
bert=np.load(ROOT/'chemberta_augmented.npz');cols.append(bert['oof']);names.append('chemberta_augmented');A=logit(np.clip(np.column_stack([rankdata(p)/len(y) for p in cols]),1e-4,1-1e-4))
reference=np.load(ROOT/'claude_stack_reproduced.npz')['oof'];band=(reference>=.35)&(reference<=.50)
def cut(y,p):
 a,b,t=precision_recall_curve(y,p);f=2*a[:-1]*b[:-1]/(a[:-1]+b[:-1]+1e-15);return float(np.median(t[f>=f.max()-.0005]))
ps=[];ds=[];cuts=[]
for seed in [2026,730,731]:
 p=np.zeros(len(y));dec=np.zeros(len(y),int)
 for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(A,y):
  inner=np.zeros(len(tr))
  for a,b in StratifiedKFold(3,shuffle=True,random_state=seed+1).split(A[tr],y[tr]):
   m=LogisticRegression(C=.01,max_iter=2500).fit(A[tr[a]],y[tr[a]]);inner[b]=m.predict_proba(A[tr[b]])[:,1]
  c=cut(y[tr],inner);cuts.append(c);m=LogisticRegression(C=.01,max_iter=2500).fit(A[tr],y[tr]);p[va]=m.predict_proba(A[va])[:,1];dec[va]=p[va]>=c
 ps.append(p);ds.append(dec)
p=np.mean(ps,axis=0);dec=np.array(ds);scores=[float(f1_score(y,q)) for q in ds];old=np.load(ROOT/'baseline.npz');old_scores=[float(f1_score(y,q)) for q in old['decisions']]
gain=float(np.mean(scores)-np.mean(old_scores));band_gain=float(roc_auc_score(y[band],p[band])-roc_auc_score(y[band],old['oof'][band]));adopt=gain>=.001 and all(a>=b for a,b in zip(scores,old_scores)) and band_gain>=.002
rng=np.random.default_rng(49);diff=[]
for _ in range(1000):
 ids=rng.integers(len(y),size=len(y));yy=y[ids]
 def f(q):return float(np.mean(2*(q[:,ids]*yy).sum(axis=1)/(q[:,ids].sum(axis=1)+yy.sum())))
 diff.append(f(dec)-f(old['decisions']))
report={'base_bert_auc':float(roc_auc_score(y,bert['oof'])),'stack_auc':float(roc_auc_score(y,p)),'seed_f1':scores,'baseline_seed_f1':old_scores,'repeated_cross_threshold_f1':float(np.mean(scores)),'gain':gain,'band_auc_gain':band_gain,'paired_f1_gain_interval_95':np.quantile(diff,[.025,.975]).tolist(),'adopt':adopt,'predeclared_gate':'mean F1 gain>=.001, nonnegative in every seed, band AUC gain>=.002','scope':'Meta CV reuses base OOF; indirect leakage and prior global parameter selection remain.'}
(ROOT/'augmentation_meta.json').write_text(json.dumps(report,indent=2));np.savez(ROOT/'augmentation_stack_oof.npz',oof=p,decisions=dec,y=y,names=np.array(names),cut=np.median(cuts));print(json.dumps(report,indent=2))
