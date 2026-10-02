import sys,time; from zoo import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
for tag in sys.argv[1:]:
    E=np.load(f'emb_{tag}.npz'); Xe,Xp=E['tr'],E['pr']
    def svm(Xtr,ytr,Xva,Xp):
        m=make_pipeline(StandardScaler(),SVC(C=3,gamma='scale')).fit(Xtr,ytr); return m.decision_function(Xva), m.decision_function(Xp)
    cv(f'svm_{tag}',svm,Xe,Xp)
    cv(f'lgb_{tag}',lgb_fp(n_estimators=500,learning_rate=0.04,num_leaves=31,colsample_bytree=0.5,random_state=1),Xe,Xp)
