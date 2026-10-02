"""Finalize the repeated probability stack; preserve earlier outputs and never submit."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'09_임계값'))
from optimal_d import optimal_d

def build():
    report=json.loads((ROOT/'probability_fusion.json').read_text())
    assert report['adopt'], 'No probability-fusion improvement passed the declared gate'
    selected=report['best_exploratory'];width=selected['width'];cut=selected['cut']
    z=np.load(ROOT/'probability_fusion_inputs.npz')
    O=z['O'][:,:width];old=z['old_test'][:,:width];full=z['full_test'][:,:width];y=z['y'].astype(int)
    names=z['names'][:width].tolist()
    assert len(set(names))==width and 'final_stack_lr' not in names
    X=logit(np.clip(O,1e-4,1-1e-4));models=[];ps=[]
    for seed in [2026,730,731]:
        p=np.zeros(len(y))
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
            model=LogisticRegression(C=selected['C'],max_iter=2500).fit(X[tr],y[tr])
            p[va]=model.predict_proba(X[va])[:,1];models.append(model)
        ps.append(p)
    oof=np.mean(ps,axis=0)
    assert np.allclose(oof,np.load(ROOT/'probability_fusion_selected.npz')['oof'],atol=1e-14)
    probe=json.loads((ROOT/'strict_refit_probe.json').read_text())
    gain=probe['refit']['f1']-probe['fold_average']['f1']
    fraction=1. if gain>=.003 else .5 if gain>=-.002 else 0.
    template=pd.read_csv(ROOT.parent/'01_원본자료/predict_output_홍길동.csv')
    current=pd.read_csv(ROOT.parent/'predict_output_서혁인.csv')
    best=pd.read_csv(ROOT.parent/'09_임계값/2차_제출됨_백업.csv')
    original=pd.read_csv(ROOT.parent/'01_원본자료/train_original_labels.csv.bak')
    assert np.array_equal(original['label'].to_numpy(),y)
    rows=[]
    for alpha in [0.,.5,1.]:
        T=logit(np.clip((1-alpha)*old+alpha*full,1e-4,1-1e-4))
        score=np.mean([m.predict_proba(T)[:,1] for m in models],axis=0)
        pred=(score>=cut).astype(int);out=template.copy();out['output']=pred
        assert out.shape==(927,2) and list(out)==['SMILES','output']
        assert out.SMILES.equals(current.SMILES) and set(out.output)<={0,1}
        target=ROOT/f'predict_output_finalmix_{alpha:g}.csv'
        out.to_csv(target,index=False,encoding='utf-8-sig');assert pd.read_csv(target).equals(out)
        optimal_D,expected,calibrated,order,curve=optimal_d(oof,y,score)
        # Diagnostic expectation only: threshold stays fixed from inner meta CV.
        expected_at_fixed_cut=float(2*calibrated[pred==1].sum()/(pred.sum()+493))
        np.savez(ROOT/f'final_scores_{alpha:g}.npz',oof=oof,test=score,calibrated_test=calibrated,estimated_f1_curve=curve,threshold=cut)
        rows.append({'full_data_fraction':alpha,'D':int(pred.sum()),'threshold':cut,
                     'estimated_f1_not_measured':expected_at_fixed_cut,
                     'changed_from_current':int((pred!=current.output.to_numpy()).sum()),
                     'changed_from_best_actual_submission':int((pred!=best.output.to_numpy()).sum()),
                     'file':target.name})
    chosen=next(r for r in rows if r['full_data_fraction']==fraction)
    ready=ROOT/'제출용/predict_output_서혁인.csv';ready.parent.mkdir(exist_ok=True)
    ready.write_bytes((ROOT/chosen['file']).read_bytes())
    reason=f'5-fold nested-fitting proxy refit F1 gain={gain:.6f}; full at >=.003, half in [-.002,.003), otherwise CV. Proxy uses 3 families, not final 25.'
    result={'selected':chosen,'choice_reason':reason,'candidates':rows,'models':names,
            'meta_model':'15 logistic meta models; 3 repeated 5-folds, C=0.1, logit probability inputs',
            'repeat_meta_f1':selected['f1'],'repeat_meta_baseline_f1':report['baseline']['f1'],
            'oof_auc':float(roc_auc_score(y,oof)),'pooled_oof_f1_at_fixed_cut':float(f1_score(y,oof>=cut)),
            'scope':'Repeated meta CV reuses base OOF and has indirect leakage/prior selection bias. Full-refit hidden-test F1 is unknown.'}
    (ROOT/'final_selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for rel,h in json.loads((ROOT/'protected_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==h,rel
    validation={'rows':927,'columns':['SMILES','output'],'SMILES_order_exact':True,
                'binary_output':True,'label_1':'safe/nontoxic','all_protected_sources_unchanged':True,
                'OOF_replay_exact':True,'predictions_match_frozen_cutoff':True,
                'submission_sha256':hashlib.sha256(ready.read_bytes()).hexdigest(),'submitted':False}
    (ROOT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':build()
