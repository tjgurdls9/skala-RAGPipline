from blendlib import *
names,O,T=load(set(n for n in load()[0] if n.startswith(('svm_tani','svm_mk_','svm_mk10'))))
w=greedy(O,Y); b=w@O
lo,hi=np.quantile(b,[.25,.55]); m=(b>=lo)&(b<=hi)   # ~ prob band around the cutoff (cutoff rate .42 -> rank pct .42)
print('band n',m.sum(),'pos rate in band',Y[m].mean().round(3), ' blend overall AUC',round(roc_auc_score(Y,b),4),'band AUC',round(roc_auc_score(Y[m],b[m]),4))
for n,o in sorted(zip(names,O),key=lambda x:-roc_auc_score(Y,x[1])):
    print(f'{n:14s} AUC {roc_auc_score(Y,o):.4f}  band AUC (blend-defined band) {roc_auc_score(Y[m],o[m]):.4f}')
