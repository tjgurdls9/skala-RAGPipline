from pathlib import Path
import json,numpy as np
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
ROOT=Path(__file__).resolve().parent
zs=[np.load(ROOT/('strict_probe' if k==0 else f'strict_probe_fold{k}')/'evaluation_predictions.npz') for k in range(5)]
y=np.concatenate([z['y'] for z in zs]);cuts=np.concatenate([np.full(len(z['y']),float(z['cut'])) for z in zs]);ids=np.concatenate([z['hold_ids'] for z in zs]);assert len(y)==8349 and len(np.unique(ids))==8349
report={'hold_rows':len(y),'outer_folds':5,'inner_folds':4,'base_models':['lgb_fp2','svm_mk_p2_c3','chemprop'],'hyperparameter_selection_caveat':'Fixed parameters were selected earlier using full-data CV. Base/meta fitting here excludes every outer validation label.','inference_rank_normalization':'Each holdout cohort, matching submission path','fold_reports':[json.loads((ROOT/f'strict_refit_probe_fold{k}.json').read_text()) for k in range(5)]}
preds={};decisions={}
for name in ['fold_average','refit','half_refit_half_cv']:
 p=np.concatenate([z[name] for z in zs]);q=p>=cuts;preds[name]=p;decisions[name]=q
 report[name]={'f1':float(f1_score(y,q)),'auc':float(roc_auc_score(y,p)),'accuracy':float(accuracy_score(y,q)),'fold_f1':[float(f1_score(z['y'],z[name]>=z['cut'])) for z in zs]}
band=(preds['fold_average']>=.35)&(preds['fold_average']<=.50)
report['boundary_band']={'rows':int(band.sum()),'source':'Frozen nested fold-average scores in [.35,.50]','metrics':{}}
for name in preds:
 report['boundary_band']['metrics'][name]={'auc':float(roc_auc_score(y[band],preds[name][band])),'accuracy':float(accuracy_score(y[band],decisions[name][band]))}
rng=np.random.default_rng(19);diffs=[]
for _ in range(2000):
 a=rng.integers(len(y),size=len(y));yy=y[a]
 def f(q):return 2*(q[a]*yy).sum()/(q[a].sum()+yy.sum())
 diffs.append(float(f(decisions['refit'])-f(decisions['fold_average'])))
report['paired_refit_f1_difference_interval_95']=np.quantile(diffs,[.025,.975]).tolist();report['scope']='3-model nested fitting comparison; not the final 16-model submission validation. No outer label enters fitting, but prior parameter selection remains global.'
(ROOT/'strict_refit_probe.json').write_text(json.dumps(report,indent=2));np.savez(ROOT/'strict_nested_predictions.npz',y=y,cuts=cuts,ids=ids,**preds);print(json.dumps(report,indent=2))
