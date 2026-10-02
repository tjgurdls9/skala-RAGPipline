from zoo import *
import lightgbm as lgb
rng=np.random.default_rng(0); res={}
for frac in (0.25,0.5,0.75,1.0):
    a=[]
    for tr,va in FOLDS[:3]:
        sub=rng.permutation(tr)[:int(len(tr)*frac)]
        m=lgb.LGBMClassifier(n_estimators=900,num_leaves=63,learning_rate=0.03,colsample_bytree=0.25,subsample=0.8,subsample_freq=1,reg_lambda=1,n_jobs=-1,verbose=-1,random_state=3).fit(X2[sub],Y[sub])
        a.append(roc_auc_score(Y[va],m.predict_proba(X2[va])[:,1]))
    print(frac,int(len(FOLDS[0][0])*frac),round(np.mean(a),4),flush=True)
