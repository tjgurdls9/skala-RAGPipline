"""Locked outer fold: rebuild inner base OOF without any outer labels."""
from pathlib import Path
import os,sys,time,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/toxicity-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit,logit
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'07_개선';FOLD_ID=int(os.environ.get('TOX_OUTER_FOLD','0'))
OUT=ROOT/('strict_probe' if FOLD_ID==0 else f'strict_probe_fold{FOLD_ID}');OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(PRIOR))
from zoo import D,Y,X2,XT,FOLDS
outer,hold=FOLDS[FOLD_ID];inner=list(StratifiedKFold(4,shuffle=True,random_state=730).split(outer,Y[outer]))

def metrics(y,p,cut):return dict(auc=float(roc_auc_score(y,p)),f1=float(f1_score(y,p>=cut)),accuracy=float(accuracy_score(y,p>=cut)),cut=float(cut))
def best_cut(y,p):
    a,b,t=precision_recall_curve(y,p);f=2*a[:-1]*b[:-1]/(a[:-1]+b[:-1]+1e-15);best=f.max();return float(np.median(t[f>=best-.001]))

def tabular():
    import lightgbm as lgb
    from sklearn.svm import SVC
    F=np.load(PRIOR/'fp2.npz');K=np.zeros((len(Y),len(Y)),np.float32)
    blocks=[D['given_tr'][:,i*1024:(i+1)*1024] for i in range(3)]+[D[f'{k}_tr'] for k in ['r3','maccs']]+[F[f'{k}_tr'] for k in ['ap','tt','rd','av']]
    for a in blocks:
      a=a.astype(np.float32);norm=a.sum(1);dot=a@a.T;K+=dot/(norm[:,None]+norm[None,:]-dot+1e-9)/len(blocks)
    K=K**2
    for name in ['lgb_fp2','svm_mk_p2_c3']:
      if (OUT/f'{name}.npz').exists():continue
      t0=time.time();o=np.zeros(len(outer));bags=[]
      def fit(tr,va):
        assert not set(tr)&set(hold)
        if name=='lgb_fp2':
          m=lgb.LGBMClassifier(n_estimators=1100,num_leaves=63,learning_rate=.03,colsample_bytree=.25,subsample=.8,subsample_freq=1,min_child_samples=20,reg_lambda=1.,n_jobs=2,verbose=-1,random_state=3).fit(X2[tr],Y[tr]);return m.predict_proba(X2[va])[:,1],m.predict_proba(X2[hold])[:,1]
        m=SVC(C=3,kernel='precomputed').fit(K[np.ix_(tr,tr)],Y[tr]);return m.decision_function(K[np.ix_(va,tr)]),m.decision_function(K[np.ix_(hold,tr)])
      for i,(a,b) in enumerate(inner):
        pv,ph=fit(outer[a],outer[b]);o[b]=pv;bags.append(ph);print('STRICT',name,'inner',i,'sec',round(time.time()-t0),flush=True)
      _,full=fit(outer,hold)
      np.savez(OUT/f'{name}.npz',inner_oof=o,hold_bag=np.mean(bags,axis=0),hold_full=full)
      print('STRICT DONE',name,round(time.time()-t0),flush=True)

def chemprop():
    import torch,warnings
    warnings.filterwarnings('ignore');torch.set_num_threads(3)
    from lightning import pytorch as pl
    from chemprop import data,featurizers,models,nn
    target=OUT/'chemprop.npz'
    if target.exists():return
    t0=time.time();desc=np.nan_to_num(D['desc_tr'].astype(float));o=np.zeros(len(outer));bags=[]
    def fit(tr,va):
      assert not set(tr)&set(hold);pl.seed_everything(1,workers=True)
      sc=StandardScaler().fit(desc[tr]);xd=np.clip(sc.transform(desc),-6,6)
      def loader(ids,shuffle):
        pts=[data.MoleculeDatapoint.from_smi(str(D['smiles_tr'][i]),[float(Y[i])],x_d=xd[i].astype(np.float32)) for i in ids]
        return data.build_dataloader(data.MoleculeDataset(pts,featurizers.SimpleMoleculeMolGraphFeaturizer()),batch_size=64,num_workers=0,shuffle=shuffle,seed=1)
      mp=nn.BondMessagePassing(d_h=300,depth=3,dropout=.1);ffn=nn.BinaryClassificationFFN(input_dim=mp.output_dim+desc.shape[1],hidden_dim=300,n_layers=2,dropout=.2)
      m=models.MPNN(mp,nn.MeanAggregation(),ffn,batch_norm=True,warmup_epochs=2,init_lr=1e-4,max_lr=1e-3,final_lr=1e-4)
      t=pl.Trainer(max_epochs=30,accelerator='cpu',logger=False,enable_progress_bar=False,enable_checkpointing=False,enable_model_summary=False);t.fit(m,loader(tr,True))
      def pred(ids):return torch.cat(t.predict(m,loader(ids,False))).numpy().ravel()
      return pred(va),pred(hold)
    for i,(a,b) in enumerate(inner):
      pv,ph=fit(outer[a],outer[b]);o[b]=pv;bags.append(ph);print('STRICT CP inner',i,'sec',round(time.time()-t0),flush=True)
    _,full=fit(outer,hold);np.savez(target,inner_oof=o,hold_bag=np.mean(bags,axis=0),hold_full=full)
    print('STRICT DONE CP',round(time.time()-t0),flush=True)

def evaluate():
    names=['lgb_fp2','svm_mk_p2_c3','chemprop'];zs=[np.load(OUT/f'{n}.npz') for n in names]
    O=np.column_stack([z['inner_oof'] for z in zs]);B=np.column_stack([z['hold_bag'] for z in zs]);F=np.column_stack([z['hold_full'] for z in zs]);yt=Y[outer];yh=Y[hold]
    # Global base hyperparameters were explored earlier on all folds. This tests
    # fitting isolation, not a previously untouched model-selection holdout.
    A=logit(np.clip(rankdata(O,axis=0)/len(O),1e-4,1-1e-4));m=LogisticRegression(C=.01,max_iter=2000).fit(A,yt)
    sc=[]
    for tr,va in StratifiedKFold(4,shuffle=True,random_state=731).split(A,yt):
      mm=LogisticRegression(C=.01,max_iter=2000).fit(A[tr],yt[tr]);sc.append((va,mm.predict_proba(A[va])[:,1]))
    innerp=np.zeros(len(yt))
    for va,p in sc:innerp[va]=p
    cut=best_cut(yt,innerp);report={'hold_rows':len(hold),'train_pool_rows':len(outer),'inner_training_rows':len(outer)*.75,'hyperparameter_selection_caveat':'Earlier global tuning saw this fold; no outer labels enter fitting here.','cutoff_selected_inside_outer_train':cut,'inference_rank_normalization':'Each prediction cohort separately, matching Claude/final submission path','models':{}}
    for n,z in zip(names,zs):
      c=best_cut(yt,z['inner_oof']);report['models'][n]={'fold_average':metrics(yh,z['hold_bag'],c),'refit':metrics(yh,z['hold_full'],c)}
    predictions={}
    for name,q in [('fold_average',B),('refit',F),('half_refit_half_cv',(B+F)/2)]:
      p=m.predict_proba(logit(np.clip(rankdata(q,axis=0)/len(q),1e-4,1-1e-4)))[:,1];report[name]=metrics(yh,p,cut);predictions[name]=p
    rng=np.random.default_rng(9);base=predictions['fold_average'];new=predictions['refit'];ds=[]
    for _ in range(1000):
      ids=rng.integers(len(yh),size=len(yh));ds.append(f1_score(yh[ids],new[ids]>=cut)-f1_score(yh[ids],base[ids]>=cut))
    report['paired_refit_f1_difference_interval_95']=np.quantile(ds,[.025,.975]).tolist()
    (ROOT/f'strict_refit_probe_fold{FOLD_ID}.json').write_text(json.dumps(report,indent=2));np.savez(OUT/'evaluation_predictions.npz',y=yh,hold_ids=hold,cut=cut,**predictions);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
