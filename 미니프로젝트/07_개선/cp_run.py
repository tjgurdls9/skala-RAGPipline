"""Chemprop v2 D-MPNN (+ optional RDKit-descriptor x_d) under the fixed folds. 
usage: cp_run.py <name> <epochs> <xd 0/1> <seed> [fold ...]   (curve mode if folds given: prints val AUC per epoch, no save)"""
import sys, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from chemprop import data, featurizers, models, nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from blendlib import Y, FOLDS, ROOT
import os; torch.set_num_threads(int(os.environ.get("CPT", 10)))
name, EP, XD, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
curve = [int(f) for f in sys.argv[5:]]
D = np.load(ROOT / 'features.npz', allow_pickle=True)
smi_tr, smi_pr = list(D['smiles_tr']), list(D['smiles_pr'])
dtr, dpr = np.nan_to_num(D['desc_tr'].astype(np.float64)), np.nan_to_num(D['desc_pr'].astype(np.float64))
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

def mk(smis, ys, xd, shuffle):
    pts = [data.MoleculeDatapoint.from_smi(s, None if ys is None else [float(y)], x_d=None if xd is None else xd[i].astype(np.float32))
           for i, (s, y) in enumerate(zip(smis, ys if ys is not None else [None] * len(smis)))]
    return data.build_dataloader(data.MoleculeDataset(pts, feat), batch_size=64, num_workers=0, shuffle=shuffle, seed=SEED)

class Curve(Callback):
    def __init__(s, vl, yv): s.vl, s.yv = vl, yv
    def on_train_epoch_end(s, tr, m):
        if (tr.current_epoch + 1) % 3: return
        m.eval()
        with torch.no_grad(): p = torch.cat([m(b[0], b[1], b[2]) for b in s.vl]).numpy().ravel()
        m.train()
        print(f'  ep {tr.current_epoch+1} valAUC {roc_auc_score(s.yv, p):.4f}', flush=True)
def Trainer_pred(m, loader):
    t = pl.Trainer(accelerator='cpu', logger=False, enable_progress_bar=False, enable_checkpointing=False, enable_model_summary=False)
    return t.predict(m, loader)

def run(tr, va):
    pl.seed_everything(SEED, workers=True)
    xd_tr = xd_va = xd_pr = None
    if XD:
        sc = StandardScaler().fit(dtr[tr]); f = lambda a: np.clip(sc.transform(a), -6, 6)
        xd_tr, xd_va, xd_pr = f(dtr[tr]), f(dtr[va]), f(dpr)
    ytr = Y[tr]
    L = mk([smi_tr[i] for i in tr], ytr, xd_tr, True)
    Lv = mk([smi_tr[i] for i in va], Y[va], xd_va, False)
    Lp = mk(smi_pr, None, xd_pr, False)
    mp = nn.BondMessagePassing(d_h=300, depth=3, dropout=0.1); agg = nn.MeanAggregation()
    nxd = dtr.shape[1] if XD else 0
    ffn = nn.BinaryClassificationFFN(input_dim=mp.output_dim + nxd, hidden_dim=300, n_layers=2, dropout=0.2)
    model = models.MPNN(mp, agg, ffn, batch_norm=True, warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)
    cb = [Curve(Lv, Y[va])] if curve else []
    t = pl.Trainer(max_epochs=EP, accelerator='cpu', logger=False, enable_progress_bar=False, enable_checkpointing=False,
                   enable_model_summary=False, callbacks=cb)
    t.fit(model, L)
    pv = torch.cat(Trainer_pred(model, Lv)).numpy().ravel(); pt = torch.cat(Trainer_pred(model, Lp)).numpy().ravel()
    return pv, pt

if curve:
    for k in curve:
        t0 = time.time(); pv, _ = run(*FOLDS[k]); print(f'fold {k} AUC {roc_auc_score(Y[FOLDS[k][1]], pv):.4f} {time.time()-t0:.0f}s')
else:
    oof = np.zeros(len(Y)); test = np.zeros(len(smi_pr)); t0 = time.time()
    for k, (tr, va) in enumerate(FOLDS):
        pv, pt = run(tr, va); oof[va] = pv; test += pt / 5
        print(f'fold {k} AUC {roc_auc_score(Y[va], pv):.4f} {time.time()-t0:.0f}s', flush=True)
    np.savez(ROOT / 'preds' / f'{name}.npz', oof=oof, test=test)
    print(f'{name} OOF AUC {roc_auc_score(Y, oof):.4f} {time.time()-t0:.0f}s')
