from pathlib import Path
import os,sys,subprocess,time
root=Path(__file__).resolve().parent
for fold in range(1,5):
    env={**os.environ,'TOX_OUTER_FOLD':str(fold)}
    print('START OUTER FOLD',fold,flush=True)
    with open(root/f'strict_tabular_fold{fold}.log','w') as f,open(root/f'strict_cp_fold{fold}.log','w') as g:
      tabular=subprocess.Popen([sys.executable,str(root/'strict_probe.py'),'tabular'],env=env,stdout=f,stderr=subprocess.STDOUT)
      subprocess.run([sys.executable,str(root/'strict_probe.py'),'chemprop'],env=env,stdout=g,stderr=subprocess.STDOUT,check=True)
      assert tabular.wait()==0
    subprocess.run([sys.executable,str(root/'strict_probe.py'),'evaluate'],env=env,check=True)
    print('DONE OUTER FOLD',fold,flush=True)
