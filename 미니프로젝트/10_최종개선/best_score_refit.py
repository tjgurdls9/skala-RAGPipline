"""Refit the exact 20-input model that scored 0.8393, reusing full caches."""
from pathlib import Path
import os,sys,json,time,numpy as np,pandas as pd
os.environ.setdefault('OMP_NUM_THREADS','2')
from scipy.special import expit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'08_재개선';FULL=ROOT/'full_predictions';sys.path.insert(0,str(OLD));import improve
D,X,Q=improve.data();Y=D['y'].astype(int);sys.path.insert(0,str(ROOT.parent/'07_개선'));from zoo import XT,XP,X2
assert np.array_equal(X,X2),'Feature order differs between caches'

def save(n,p):
 assert p.shape==(927,) and np.isfinite(p).all()
 np.save(FULL/f'{n}.npy',p);print('BEST REFIT DONE',n,flush=True)

def train():
 for n,kw in [('xgb_a',dict(random_state=42)),('xgb_b',dict(random_state=7,max_depth=10,min_child_weight=8,colsample_bytree=.2,n_estimators=1400,learning_rate=.02))]:
  if (FULL/f'{n}.npy').exists():continue
  params=dict(n_estimators=900,max_depth=7,learning_rate=.03,colsample_bytree=.3,subsample=.8,min_child_weight=3,reg_lambda=1.,tree_method='hist',n_jobs=2,eval_metric='logloss');params.update(kw);m=XGBClassifier(**params).fit(XT,Y);save(n,m.predict_proba(XP)[:,1])
 for name,seeds in [('lgb_leaf31_lowchild',[42,2026]),('lgb_leaf31',[42,2026]),('lgb_extra63',[42]),('lgb_chiralcounts',[42])]:
  n=f'{name}_seed42'
  if (FULL/f'{n}.npy').exists():continue
  XX,QQ=X,Q
  config=improve.CONFIGS.get(name)
  if name=='lgb_chiralcounts':
   extra=np.load(OLD/'chiral_counts.npz');XX=np.hstack([X,extra['tr']]);QQ=np.hstack([Q,extra['pr']]);keep=XX.std(0)>0;XX,QQ=XX[:,keep],QQ[:,keep];config=dict(num_leaves=63,min_child_samples=15,reg_lambda=3,colsample_bytree=.25)
  ps=[]
  for seed in seeds:
   reusable=FULL/f'lgb_lowchild_full_s{seed}.npy'
   if name=='lgb_leaf31_lowchild' and reusable.exists():p=np.load(reusable)
   else:
    m=LGBMClassifier(n_estimators=1400,learning_rate=.025,subsample=.85,subsample_freq=1,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True,random_state=seed,**config).fit(XX,Y);p=m.predict_proba(QQ)[:,1]
   ps.append(p)
  save(n,np.mean(ps,axis=0))
 n='xgb_depth5_reg_seed42'
 if not (FULL/f'{n}.npy').exists():
  m=XGBClassifier(n_estimators=1600,learning_rate=.025,subsample=.85,reg_alpha=.1,tree_method='hist',n_jobs=2,random_state=42,eval_metric='logloss',max_depth=5,min_child_weight=3,reg_lambda=3,colsample_bytree=.4).fit(X,Y);save(n,m.predict_proba(Q)[:,1])
 cache=np.load(ROOT.parent/'06_모델링/feature_cache.npz');A,B=cache['X'][:,:1024].astype(np.float32),cache['Q'][:,:1024].astype(np.float32);K,T=improve.tanimoto(A,A),improve.tanimoto(B,A)
 n='svm_tanimoto_C3_seed42'
 if not (FULL/f'{n}.npy').exists():m=SVC(C=3,kernel='precomputed',cache_size=512).fit(K,Y);save(n,expit(m.decision_function(T)))
 F=np.load(ROOT.parent/'07_개선/fp2.npz');K=np.zeros((len(Y),len(Y)),np.float32);T=np.zeros((927,len(Y)),np.float32)
 blocks=[(D['given_tr'][:,:1024],D['given_pr'][:,:1024]),(D['given_tr'][:,1024:2048],D['given_pr'][:,1024:2048]),(D['cnt_tr'],D['cnt_pr']),(F['rd_tr'],F['rd_pr'])]
 for A,B in blocks:
  A,B=A.astype(np.float32),B.astype(np.float32);norm=(A*A).sum(1);dot=A@A.T;union=norm[:,None]+norm[None,:]-dot;K+=np.divide(dot,union,out=np.zeros_like(dot),where=union>0)/len(blocks);dot=B@A.T;union=(B*B).sum(1)[:,None]+norm[None,:]-dot;T+=np.divide(dot,union,out=np.zeros_like(dot),where=union>0)/len(blocks)
 sc=StandardScaler().fit(D['given_tr'][:,-4:]);P=sc.transform(D['given_tr'][:,-4:]);PP=sc.transform(D['given_pr'][:,-4:]);R=rbf_kernel(P,gamma=.25);RT=rbf_kernel(PP,P,gamma=.25)
 for C,w in [(1,0),(3,0),(10,0),(3,.15),(10,.15)]:
  n=f'svm_multi_C{C:g}_prop{w:g}_seed42'
  if (FULL/f'{n}.npy').exists():continue
  m=SVC(C=C,kernel='precomputed',cache_size=512).fit((1-w)*K+w*R,Y);save(n,expit(m.decision_function((1-w)*T+w*RT)))

def finalize():
 a=np.load(OLD/'ensemble_inputs.npz');O=a['O'];y=a['y'];names=list(a['names']);T=np.column_stack([np.load(FULL/f'{n}.npy') for n in names]);old=a['T'];boundary_count=int(np.sum((T<=0)|(T>=1)));T=np.clip(T,1e-7,1-1e-7);print('Numerical logit clipping at boundaries:',boundary_count,flush=True);s=json.loads((OLD/'ensemble_selection.json').read_text());cut=s['stable_cutoff'];folds=list(StratifiedKFold(5,shuffle=True,random_state=2026).split(O,y));mode,C=s['name'].rsplit('_C',1)
 baseline=np.load(OLD/'ensemble_predictions.npz');template=pd.read_csv(ROOT.parent/'01_원본자료/predict_output_홍길동.csv');current=pd.read_csv(ROOT.parent/'predict_output_서혁인.csv');best_csv=pd.read_csv(OLD/'제출용/predict_output_서혁인.csv');rows=[]
 for alpha in [0.,.5,1.]:
  p,q,_,details=improve.meta_candidate(O,(1-alpha)*old+alpha*T,y,folds,mode,float(C));assert np.allclose(p,baseline['oof'],rtol=0,atol=1e-12)
  out=template.copy();out.output=(q>=cut).astype(int)
  if alpha==0:assert out.equals(best_csv) and np.allclose(q,baseline['test'],rtol=0,atol=1e-12)
  out.to_csv(ROOT/f'predict_output_best_refit_{alpha:g}.csv',index=False,encoding='utf-8-sig');np.savez(ROOT/f'best_refit_scores_{alpha:g}.npz',oof=p,test=q,cut=cut)
  rows.append({'full_data_fraction':alpha,'safe_count':int(out.output.sum()),'changed_vs_current':int((out.output!=current.output).sum()),'changed_vs_best_submitted':int((out.output!=best_csv.output).sum()),'old_oof_f1_not_refit_validation':s['selected_metrics']['f1_safe'],'old_oof_auc_not_refit_validation':s['selected_metrics']['auc']})
 np.savez(ROOT/'best_refit_inputs.npz',O=O,old_test=old,full_test=T,y=y,names=np.array(names));(ROOT/'best_score_refit.json').write_text(json.dumps({'verified_best_submitted_score':.8393,'best_file_verified_by_backup':True,'model':s['name'],'cut':cut,'candidates':rows,'scope':'Frozen model/threshold selected earlier. Base models refit on all 8349 rows; new test F1 unknown.'},indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':train();finalize()
