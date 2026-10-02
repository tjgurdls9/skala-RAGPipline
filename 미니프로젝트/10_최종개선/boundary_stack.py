"""Repeated meta CV: evaluate boundary emphasis, no derived stacks as inputs."""
from pathlib import Path
import json,sys,time,os
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_recall_curve
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'07_개선';OLD=ROOT.parent/'08_재개선'
y=np.load(PRIOR/'features.npz',allow_pickle=True)['y'].astype(int)
audit=json.loads((ROOT/'claude_audit.json').read_text());base_names=audit['pool'];base=[np.load(PRIOR/'preds'/f'{n}.npz') for n in base_names]
NAMES=list(base_names);O=[z['oof'] for z in base];T=[z['test'] for z in base]
for name in ['lgb_leaf31_lowchild','lgb_leaf31','lgb_extra63','lgb_chiralcounts','xgb_depth5_reg','svm_multi_C10_prop0.15']:
    a=np.load(OLD/'preds'/f'{name}_seed42.npz');op,tp=a['oof'],a['test'];repeat=OLD/'preds'/f'{name}_seed2026.npz'
    if repeat.exists():
      b=np.load(repeat);op,tp=(op+b['oof'])/2,(tp+b['test'])/2
    NAMES.append(name);O.append(op);T.append(tp)
O=np.column_stack([rankdata(p)/len(y) for p in O]);T=np.column_stack([rankdata(p)/927 for p in T])
X=logit(np.clip(O,1e-4,1-1e-4));Q=logit(np.clip(T,1e-4,1-1e-4))
reference=np.load(ROOT/'claude_stack_reproduced.npz')['oof'];band=(reference>=.35)&(reference<=.50)

def threshold(y,p):
    a,b,t=precision_recall_curve(y,p);f=2*a[:-1]*b[:-1]/(a[:-1]+b[:-1]+1e-15);return float(np.median(t[f>=f.max()-.0005]))
def fit(A,b,C,emphasis):
    m=LogisticRegression(C=C,max_iter=2500).fit(A,b)
    if emphasis:
      ref=m.predict_proba(A)[:,1];w=.5+1.5*np.exp(-((ref-.425)/.10)**2)
      m=LogisticRegression(C=C,max_iter=2500).fit(A,b,sample_weight=w)
    return m

def run():
    rows=[];outputs={};start=time.time()
    # Test the old pool and our expanded pool on the same folds, thresholds, seeds.
    specs=[('baseline',16,.01,False),('expanded_C.003',len(NAMES),.003,False),('expanded_C.01',len(NAMES),.01,False),('expanded_C.03',len(NAMES),.03,False),('expanded_C.1',len(NAMES),.1,False),('band_C.003',len(NAMES),.003,True),('band_C.01',len(NAMES),.01,True),('band_C.03',len(NAMES),.03,True)]
    for name,width,C,emphasis in specs:
      A,B=X[:,:width],Q[:,:width];ps=[];dec=[];fold_cuts=[];seed_scores=[]
      for seed in [2026,730,731]:
        p=np.zeros(len(y));decisions=np.zeros(len(y),int);cuts=[]
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(A,y):
          inside=np.zeros(len(tr))
          for a,b in StratifiedKFold(3,shuffle=True,random_state=seed+1).split(A[tr],y[tr]):
            m=fit(A[tr[a]],y[tr[a]],C,emphasis);inside[b]=m.predict_proba(A[tr[b]])[:,1]
          cut=threshold(y[tr],inside);cuts.append(cut)
          model=fit(A[tr],y[tr],C,emphasis);p[va]=model.predict_proba(A[va])[:,1];decisions[va]=p[va]>=cut
        ps.append(p);dec.append(decisions);fold_cuts+=cuts;seed_scores.append(float(f1_score(y,decisions)))
      mean=np.mean(ps,axis=0);cut=float(np.median(fold_cuts));full=fit(A,y,C,emphasis);test=full.predict_proba(B)[:,1]
      row=dict(name=name,width=width,C=C,boundary_weighted=emphasis,auc=float(roc_auc_score(y,mean)),f1_fixed=float(f1_score(y,mean>=cut)),repeated_cross_threshold_f1=float(np.mean(seed_scores)),seed_f1=seed_scores,cut=cut,accuracy=float(accuracy_score(y,mean>=cut)),band_auc=float(roc_auc_score(y[band],mean[band])),band_accuracy=float(accuracy_score(y[band],mean[band]>=cut)),band_rows=int(band.sum()))
      rows.append(row);np.savez(ROOT/f'{name}.npz',oof=mean,test=test,decisions=np.array(dec),names=np.array(NAMES[:width]),cut=cut);outputs[name]=(mean,test)
      print('META',name,json.dumps(row),'seconds',round(time.time()-start),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'boundary_meta_comparison.csv',index=False)
    baseline=rows[0];best=max(rows,key=lambda x:x['repeated_cross_threshold_f1'])
    # Keep evidence of all choices, including failed boundary weighting.
    selected=best if best['repeated_cross_threshold_f1']>=baseline['repeated_cross_threshold_f1']+.001 else baseline
    p,q=outputs[selected['name']];(ROOT/'meta_selected.json').write_text(json.dumps({'selected':selected,'best_exploratory':best,'baseline':baseline,'scope':'Repeated meta CV reuses base OOF: indirect leakage and selection bias remain.'},indent=2))
    np.savez(ROOT/'meta_selected.npz',oof=p,test=q,names=np.array(NAMES[:selected['width']]),cut=selected['cut'])
    print('SELECTED',selected['name'],flush=True)

if __name__=='__main__':
    assert len(set(NAMES))==len(NAMES) and not any(n.startswith('final_stack') for n in NAMES)
    assert np.isfinite(X).all() and np.isfinite(Q).all()
    run()
