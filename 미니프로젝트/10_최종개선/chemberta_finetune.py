"""Local cached ChemBERTa: last two encoder layers + toxicity head."""
from pathlib import Path
import os,sys,time,json,random,gc
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent;D=np.load(ROOT.parent/'07_개선/features.npz',allow_pickle=True);y=D['y'].astype(int)
MODEL='DeepChem/ChemBERTa-77M-MTR';DEVICE='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(2)
AUGMENT=os.environ.get('CB_AUGMENT')=='1'
EPOCHS=24 if AUGMENT else 8
TAG='chemberta_augmented' if AUGMENT else 'chemberta_finetuned'
OUT=ROOT/('chemberta_aug' if AUGMENT else 'chemberta');OUT.mkdir(exist_ok=True)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
smiles=list(D['smiles_tr'])+list(D['smiles_pr']);enc=tok(smiles,truncation=True,max_length=256,padding=False)
views=[enc]
if AUGMENT:
    from rdkit import Chem
    rejected=0
    for j in range(3):
      augmented=[]
      for i,smi in enumerate(smiles):
        mol=Chem.MolFromSmiles(str(smi));assert mol is not None
        random_smi=Chem.MolToRandomSmilesVect(mol,1,randomSeed=420+i*3+j,isomericSmiles=True)[0]
        canonical=Chem.MolToSmiles(mol,isomericSmiles=True)
        if Chem.MolToSmiles(Chem.MolFromSmiles(random_smi),isomericSmiles=True)!=canonical:
          random_smi=str(smi);rejected+=1
        assert Chem.MolToSmiles(Chem.MolFromSmiles(random_smi),isomericSmiles=True)==canonical
        augmented.append(random_smi)
      views.append(tok(augmented,truncation=True,max_length=256,padding=False))
if AUGMENT:
    (OUT/'augmentation_validation.json').write_text(json.dumps({'views':4,'molecules':len(smiles),'rejected_random_views':rejected,'stereo_canonical_preserved':True}))
    print('PASS random SMILES stereo check; rejected views',rejected,flush=True)
prop_ids=[list(D['given_names']).index(n) for n in ['MolWt','clogp','sa_score','qed']]
props=np.vstack([D['given_tr'][:,prop_ids],D['given_pr'][:,prop_ids]]).astype(np.float32)
N=len(y);test_ids=np.arange(N,N+927)

class Classifier(nn.Module):
    def __init__(self):
      super().__init__();self.base=AutoModel.from_pretrained(MODEL,local_files_only=True,add_pooling_layer=False,attn_implementation='eager')
      for p in self.base.parameters():p.requires_grad=False
      for layer in self.base.encoder.layer[-(3 if AUGMENT else 2):]:
        for p in layer.parameters():p.requires_grad=True
      self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(self.base.config.hidden_size+4,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
    def forward(self,b,prop):
      h=self.base(**b).last_hidden_state;mask=b['attention_mask'].unsqueeze(-1).to(h.dtype);pooled=(h*mask).sum(1)/mask.sum(1)
      return self.head(torch.cat([pooled,prop],1)).squeeze(-1)

def fit_predict(tr,va,seed,tag,epochs=None):
    epochs=EPOCHS if epochs is None else epochs
    target=OUT/f'{tag}.npz'
    if target.exists():z=np.load(target);return z['valid'],z['test']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    sc=StandardScaler().fit(props[tr]);xp=np.clip(sc.transform(props),-6,6).astype(np.float32)
    def batch(ids,training=False,view=0):
      indices=np.random.randint(0,len(views),len(ids)) if training else np.full(len(ids),view)
      inp=tok.pad([{k:views[j][k][i] for k in enc} for i,j in zip(ids,indices)],padding=True,return_tensors='pt')
      return {k:v.to(DEVICE) for k,v in inp.items()},torch.tensor(xp[ids],device=DEVICE)
    def loader(ids,shuffle):return DataLoader(list(map(int,ids)),batch_size=32,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(seed),collate_fn=lambda a:a)
    L=loader(tr,True);V=loader(va,False);T=loader(test_ids,False)
    m=Classifier().to(DEVICE);base=[p for p in m.base.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':base,'lr':5e-5 if AUGMENT else 2e-5},{'params':m.head.parameters(),'lr':1e-3}],weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(len(L)*epochs*.1),len(L)*epochs)
    loss_fn=nn.BCEWithLogitsLoss();valids=[];tests=[];history=[];start=time.time()
    def predict(L):
      m.eval();all_views=[]
      with torch.no_grad():
        for view in range(len(views)):
          out=[]
          for ids in L:
            b,p=batch(ids,view=view);out.append(torch.sigmoid(m(b,p)).cpu().numpy())
          all_views.append(np.concatenate(out))
      return np.mean(all_views,axis=0)
    for epoch in range(epochs):
      m.train();losses=[]
      for ids in L:
        b,p=batch(ids,training=True);target_y=torch.tensor(y[ids],dtype=torch.float32,device=DEVICE);optimizer.zero_grad(set_to_none=True);loss=loss_fn(m(b,p),target_y);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step();scheduler.step();losses.append(float(loss.detach().cpu()))
      v=predict(V);auc=float(roc_auc_score(y[va],v)) if len(va) and len(np.unique(y[va]))==2 else None
      history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'valid_auc':auc,'seconds':time.time()-start})
      print('FINETUNE',tag,'epoch',epoch+1,'AUC',auc,'seconds',round(time.time()-start),flush=True)
      if epoch>=epochs-3:valids.append(v);tests.append(predict(T))
    v,p=np.mean(valids,axis=0),np.mean(tests,axis=0)
    assert np.isfinite(v).all() and np.isfinite(p).all()
    np.savez(target,valid=v,test=p,valid_ids=va)
    (OUT/f'{tag}.json').write_text(json.dumps({'device':DEVICE,'history':history,'seed':seed,'train_rows':len(tr),'epochs':epochs,'last_three_epochs_average':True,'outer_epoch_selection':False,'random_smiles_augmentation':AUGMENT,'TTA_views':len(views)},indent=2))
    torch.save({'state_dict':{k:v.cpu() for k,v in m.state_dict().items()},'scaler_mean':sc.mean_,'scaler_scale':sc.scale_},OUT/f'{tag}.pt')
    del m;gc.collect()
    if DEVICE=='mps':torch.mps.empty_cache()
    return v,p

def cv():
    print('DEVICE',DEVICE,'cached model',MODEL,flush=True);folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(N),y));o=np.zeros(N);p=np.zeros(927)
    for k,(tr,va) in enumerate(folds):
      v,t=fit_predict(tr,va,2026+k,f'fold{k}');o[va]=v;p+=t/5
    np.savez(ROOT/f'{TAG}.npz',oof=o,test=p);print('FINETUNED OOF AUC',roc_auc_score(y,o),flush=True)

def full():
    for seed in [2026,2027]:fit_predict(np.arange(N),np.array([0,1]),seed,f'full{seed}')
    ps=[np.load(OUT/f'full{s}.npz')['test'] for s in [2026,2027]];np.save(ROOT/f'full_predictions/{TAG}.npy',np.mean(ps,axis=0))

if __name__=='__main__':
    assert DEVICE=='mps','GPU required for this time-bounded experiment'
    globals()[sys.argv[1] if len(sys.argv)>1 else 'cv']()
