"""Reproduce frozen 16-model stack with full-data predictions; no submission."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'07_개선';FULL=ROOT/'full_predictions'
sys.path.insert(0,str(ROOT.parent/'09_임계값'))
from optimal_d import optimal_d

def build():
    data=np.load(PRIOR/'features.npz',allow_pickle=True);y=data['y'].astype(int)
    audit=json.loads((ROOT/'claude_audit.json').read_text());names=audit['pool']
    baseline=np.load(ROOT/'claude_stack_reproduced.npz')
    recipe=json.loads((ROOT/'gbdt_bag_recipe.json').read_text())
    np.save(FULL/'bag_gbdt_all.npy',sum(w*np.load(FULL/f'{n}.npy') for n,w in recipe.items()))
    np.save(FULL/'bag_lgb_fp2.npy',np.mean([np.load(FULL/f'{n}.npy') for n in ['lgb_fp2','lgb_fp2_s11','lgb_fp2_s12','lgb_fp2_s13']],axis=0))
    O=np.column_stack([rankdata(np.load(PRIOR/'preds'/f'{n}.npz')['oof'])/len(y) for n in names]);oldT=np.column_stack([np.load(PRIOR/'preds'/f'{n}.npz')['test'] for n in names]);fullT=np.column_stack([np.load(FULL/f'{n}.npy') for n in names])
    model=LogisticRegression(C=.01,max_iter=2000).fit(logit(np.clip(O,1e-4,1-1e-4)),y)
    template=pd.read_csv(ROOT.parent/'01_원본자료/predict_output_홍길동.csv');current=pd.read_csv(ROOT.parent/'predict_output_서혁인.csv');rows=[]
    for alpha in [0.,.5,1.]:
      raw=(1-alpha)*oldT+alpha*fullT
      T=np.column_stack([rankdata(raw[:,i])/927 for i in range(raw.shape[1])]);score=model.predict_proba(logit(np.clip(T,1e-4,1-1e-4)))[:,1]
      if alpha==0:assert np.allclose(score,baseline['test'],atol=1e-14)
      D,expected,calibrated,order,curve=optimal_d(baseline['oof'],y,score)
      # Preserve ordering within isotonic plateaus by the unrounded model score.
      order=np.lexsort((-score,-calibrated));pred=np.zeros(927,int);pred[order[:D]]=1
      out=template.copy();out['output']=pred
      assert out.SMILES.equals(current.SMILES) and set(out.output)<={0,1}
      target=ROOT/f'predict_output_fullmix_{alpha:g}.csv';out.to_csv(target,index=False,encoding='utf-8-sig');assert pd.read_csv(target).equals(out)
      rows.append({'full_data_fraction':alpha,'D':int(D),'estimated_f1_not_measured':float(expected),'changed_from_current':int((pred!=current.output.to_numpy()).sum()),'file':target.name})
      np.savez(ROOT/f'refit_scores_{alpha:g}.npz',oof=baseline['oof'],test=score,calibrated_test=calibrated,optimal_D=D,estimated_f1_curve=curve)
    if (ROOT/'strict_refit_probe.json').exists():
      probe=json.loads((ROOT/'strict_refit_probe.json').read_text());gain=probe['refit']['f1']-probe['fold_average']['f1']
      chosen=1. if gain>=.003 else .5 if gain>=-.002 else 0.
      reason=f'Frozen outer probe refit-minus-fold-average F1={gain:.6f}; use full at >=.003, half in [-.002,.003), otherwise retain CV.'
    else:raise RuntimeError('Finish strict holdout probe before choosing the final file')
    selected=next(row for row in rows if row['full_data_fraction']==chosen)
    ready=ROOT/'제출용/predict_output_서혁인.csv';ready.parent.mkdir(exist_ok=True);ready.write_bytes((ROOT/selected['file']).read_bytes())
    selected={'selected':selected,'choice_reason':reason,'candidates':rows,'models':names,'scope':'OOF belongs to 80%-trained fold models. Full refit changes test predictions; full-refit test F1 is unknown.'}
    (ROOT/'refit_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    np.savez(ROOT/'refit_meta_inputs.npz',O=O,old_test=oldT,full_test=fullT,y=y,names=np.array(names),coefficients=model.coef_,intercept=model.intercept_)
    for rel,h in json.loads((ROOT/'protected_hashes.json').read_text()).items():assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==h
    validation={'rows':927,'columns':['SMILES','output'],'SMILES_order_exact':True,'binary_output':True,'all_protected_sources_unchanged':True,'full_models_match_original_pool':True,'label_1':'safe/nontoxic','submission_sha256':hashlib.sha256(ready.read_bytes()).hexdigest(),'submitted':False}
    (ROOT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2));print(json.dumps(selected,ensure_ascii=False,indent=2))

if __name__=='__main__':build()
