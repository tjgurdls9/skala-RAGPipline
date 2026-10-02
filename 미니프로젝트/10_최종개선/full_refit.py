"""Full-data refits; preserve existing submissions and all prior caches."""
from pathlib import Path
import os,sys,json,time,argparse,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/toxicity-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'07_개선'
OUT=ROOT/'full_predictions';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(PRIOR))

def save(name,p,seconds,details):
    assert p.shape==(927,) and np.isfinite(p).all()
    np.save(OUT/f'{name}.npy',p)
    (OUT/f'{name}.json').write_text(json.dumps({'seconds':seconds,'train_rows':8349,**details},indent=2))
    print('DONE',name,'seconds',round(seconds,1),flush=True)

def tabular():
    from zoo import D,Y,XT,XP,X2,X2P,BIN,BINP
    import lightgbm as lgb,xgboost as xgb
    from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from catboost import CatBoostClassifier
    import joblib
    models={}
    lp=dict(n_estimators=900,num_leaves=63,learning_rate=.03,colsample_bytree=.3,subsample=.8,subsample_freq=1,min_child_samples=20,reg_lambda=1.,n_jobs=4,verbosity=-1)
    specs={'lgb_a':dict(random_state=42),'lgb_b':dict(random_state=7,num_leaves=127,learning_rate=.02,n_estimators=1400,colsample_bytree=.2,min_child_samples=10),
      'lgb_goss':dict(random_state=11,boosting_type='goss',num_leaves=63,n_estimators=1200,subsample=1.,subsample_freq=0)}
    for n,kw in specs.items():models[n]=(lgb.LGBMClassifier(**{**lp,**kw}),XT,XP)
    for seed in [3,11,12,13]:models['lgb_fp2' if seed==3 else f'lgb_fp2_s{seed}']=(lgb.LGBMClassifier(**{**lp,'random_state':seed,'n_estimators':1100,'colsample_bytree':.25}),X2,X2P)
    for seed in [42,2026,31415]:
      models[f'lgb_lowchild_full_s{seed}']=(lgb.LGBMClassifier(n_estimators=1400,learning_rate=.025,num_leaves=31,min_child_samples=8,reg_lambda=1.,colsample_bytree=.2,subsample=.85,subsample_freq=1,random_state=seed,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True),X2,X2P)
    models['xgb_fp2']=(xgb.XGBClassifier(n_estimators=1100,max_depth=8,learning_rate=.03,colsample_bytree=.25,subsample=.8,min_child_weight=3,reg_lambda=1.,tree_method='hist',n_jobs=4,eval_metric='logloss',random_state=3),X2,X2P)
    models['cat_fp2']=(CatBoostClassifier(iterations=1500,learning_rate=.05,depth=7,rsm=.1,l2_leaf_reg=5,thread_count=4,verbose=False,random_seed=0,border_count=64,allow_writing_files=False),X2,X2P)
    M=np.load(PRIOR/'mordred.npz')
    models['lgb_mord']=(lgb.LGBMClassifier(**{**lp,'random_state':3,'n_estimators':1100,'colsample_bytree':.25}),np.hstack([X2,M['tr']]),np.hstack([X2P,M['pr']]))
    models['lgb_mordonly']=(lgb.LGBMClassifier(**{**lp,'random_state':3,'num_leaves':31,'n_estimators':600,'colsample_bytree':.4}),M['tr'],M['pr'])
    models['et']=(ExtraTreesClassifier(n_estimators=1000,max_features='sqrt',min_samples_leaf=1,n_jobs=4,random_state=42),XT,XP)
    models['rf']=(RandomForestClassifier(n_estimators=1000,max_features='sqrt',min_samples_leaf=1,n_jobs=4,random_state=42),XT,XP)
    models['logit_bin']=(LogisticRegression(C=.3,max_iter=3000),BIN,BINP)
    for n,(model,X,Q) in models.items():
      if (OUT/f'{n}.npy').exists():continue
      t=time.time();model.fit(X,Y);save(n,model.predict_proba(Q)[:,1],time.time()-t,{'params':model.get_params(),'feature_count':X.shape[1]})
      if n in ['lgb_fp2','lgb_lowchild_full_s42']:joblib.dump(model,ROOT/f'{n}.joblib')
    for tag in ['cb77','molf']:
      n=f'svm_{tag}'
      if (OUT/f'{n}.npy').exists():continue
      t=time.time();E=np.load(PRIOR/f'emb_{tag}.npz');sc=StandardScaler().fit(E['tr']);m=SVC(C=3,gamma='scale').fit(sc.transform(E['tr']),Y)
      save(n,m.decision_function(sc.transform(E['pr'])),time.time()-t,{'C':3,'gamma':'scale','embedding':tag})
    if not (OUT/'svm_tani10.npy').exists():
      t=time.time();dot=BIN@BIN.T;norm=BIN.sum(1);K=dot/(norm[:,None]+norm[None,:]-dot+1e-9);m=SVC(C=10,kernel='precomputed').fit(K,Y);dot=BINP@BIN.T;Q=dot/(BINP.sum(1)[:,None]+norm[None,:]-dot+1e-9)
      save('svm_tani10',m.decision_function(Q),time.time()-t,{'C':10,'kernel':'binary Tanimoto'})
    if not (OUT/'svm_mk_p2_c3.npy').exists():
      t=time.time();F=np.load(PRIOR/'fp2.npz');K=np.zeros((len(Y),len(Y)),dtype=np.float32);Q=np.zeros((927,len(Y)),dtype=np.float32)
      blocks=[(D['given_tr'][:,i*1024:(i+1)*1024],D['given_pr'][:,i*1024:(i+1)*1024]) for i in range(3)]
      blocks += [(D[f'{k}_tr'],D[f'{k}_pr']) for k in ['r3','maccs']]+[(F[f'{k}_tr'],F[f'{k}_pr']) for k in ['ap','tt','rd','av']]
      for A,B in blocks:
        A,B=A.astype(np.float32),B.astype(np.float32);n=A.sum(1);dot=A@A.T;K+=dot/(n[:,None]+n[None,:]-dot+1e-9)/len(blocks);dot=B@A.T;Q+=dot/(B.sum(1)[:,None]+n[None,:]-dot+1e-9)/len(blocks)
      m=SVC(C=3,kernel='precomputed').fit(K**2,Y);save('svm_mk_p2_c3',m.decision_function(Q**2),time.time()-t,{'C':3,'kernel':'mean nine Tanimoto squared'})
    for n in ['mlp','knn_tan']:
      if (OUT/f'{n}.npy').exists():continue
      t=time.time()
      if n=='mlp':
        sc=StandardScaler(with_mean=False).fit(XT);m=MLPClassifier((512,128),alpha=1e-3,batch_size=128,learning_rate_init=1e-3,max_iter=60,early_stopping=True,n_iter_no_change=6,random_state=42).fit(sc.transform(XT),Y);p=m.predict_proba(sc.transform(XP))[:,1]
      else:
        norm=BIN.sum(1);dot=BINP@BIN.T;sim=dot/(BINP.sum(1)[:,None]+norm[None,:]-dot+1e-9);idx=np.argpartition(-sim,25,axis=1)[:,:25];w=np.take_along_axis(sim,idx,1);p=(w*Y[idx]).sum(1)/(w.sum(1)+1e-9)
      save(n,p,time.time()-t,{'model':n})

def chemprop():
    import torch,warnings
    warnings.filterwarnings('ignore');torch.set_num_threads(4)
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import Callback
    from chemprop import data,featurizers,models,nn
    from sklearn.preprocessing import StandardScaler
    D=np.load(PRIOR/'features.npz',allow_pickle=True);y=D['y'].astype(int)
    desc=np.nan_to_num(D['desc_tr'].astype(float));sc=StandardScaler().fit(desc)
    def mk(smis,labels,xd,shuffle,seed):
      points=[data.MoleculeDatapoint.from_smi(str(s),None if labels is None else [float(labels[i])],x_d=xd[i].astype(np.float32)) for i,s in enumerate(smis)]
      return data.build_dataloader(data.MoleculeDataset(points,featurizers.SimpleMoleculeMolGraphFeaturizer()),batch_size=64,num_workers=0,shuffle=shuffle,seed=seed)
    train_desc=np.clip(sc.transform(desc),-6,6);test_desc=np.clip(sc.transform(np.nan_to_num(D['desc_pr'].astype(float))),-6,6)
    class Progress(Callback):
      def on_train_epoch_end(self,t,m):
        if (t.current_epoch+1)%5==0:print('Chemprop epoch',t.current_epoch+1,flush=True)
    for seed,epochs in [(0,25),(1,30),(2,30),(3,30)]:
      name=f'cp_full_s{seed}'
      if (OUT/f'{name}.npy').exists():continue
      t0=time.time();pl.seed_everything(seed,workers=True)
      L=mk(D['smiles_tr'],y,train_desc,True,seed);Q=mk(D['smiles_pr'],None,test_desc,False,seed)
      mp=nn.BondMessagePassing(d_h=300,depth=3,dropout=.1);ffn=nn.BinaryClassificationFFN(input_dim=mp.output_dim+desc.shape[1],hidden_dim=300,n_layers=2,dropout=.2)
      m=models.MPNN(mp,nn.MeanAggregation(),ffn,batch_norm=True,warmup_epochs=2,init_lr=1e-4,max_lr=1e-3,final_lr=1e-4)
      tr=pl.Trainer(max_epochs=epochs,accelerator='cpu',logger=False,enable_progress_bar=False,enable_checkpointing=False,enable_model_summary=False,callbacks=[Progress()]);tr.fit(m,L)
      p=torch.cat(tr.predict(m,Q)).numpy().ravel();save(name,p,time.time()-t0,{'seed':seed,'epochs':epochs,'descriptor_scaler_train_rows':len(y)})
      tr.save_checkpoint(ROOT/f'{name}.ckpt')
    save('bag_cp',np.mean([np.load(OUT/f'cp_full_s{s}.npy') for s in range(4)],axis=0),0,{'components':[0,1,2,3]})

def extra():
    from zoo import Y,X2,X2P
    import lightgbm as lgb
    from zoo2 import T1,T3
    for n,cfg,seed in [('lgb_t1',T1,1),('lgb_t3',T3,3)]:
      t=time.time();params=dict(n_estimators=420,learning_rate=.05,subsample_freq=1,random_state=seed,n_jobs=2,verbose=-1,**cfg)
      m=lgb.LGBMClassifier(**params).fit(X2,Y);save(n,m.predict_proba(X2P)[:,1],time.time()-t,{'params':params})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['tabular','chemprop','extra']);args=a.parse_args()
    globals()[args.action]()
