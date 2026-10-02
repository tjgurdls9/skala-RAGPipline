"""Compact EDA alternatives and locally measured feature extraction costs.

Run ../.venv_eda/bin/python make_compact_eda.py from this directory.
Selection is conflict-focused EDA, not an independent generalization test.
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median
from time import perf_counter
import csv
import hashlib
import json
import platform

from rdkit import Chem, rdBase
from rdkit.Chem import rdCIPLabeler, rdFingerprintGenerator, rdMolDescriptors
import run_stereo_eda as exploratory

ROOT = Path(__file__).resolve().parent
BASE = ['stereo_R_minus_S', 'stereo_r_minus_s', 'stereo_unassigned_count']
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=False)


def parse(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:raise ValueError(smiles)
    potential = Chem.FindPotentialStereo(m)
    rdCIPLabeler.AssignCIPLabels(m)
    centers={s.centeredOn for s in potential if s.type==Chem.StereoType.Atom_Tetrahedral}
    centers.update(a.GetIdx() for a in m.GetAtoms() if a.HasProp('_CIPCode'))
    return m, centers


def compact(smiles, selected):
    m,centers=parse(smiles)
    states={i:m.GetAtomWithIdx(i).GetProp('_CIPCode') if m.GetAtomWithIdx(i).HasProp('_CIPCode') else '?' for i in centers}
    c=Counter(states.values())
    counts=Counter((exploratory.environment(m,i),state) for i,state in states.items() if state=='R')
    return [c['R']-c['S'],c['r']-c['s'],c['?']]+[counts[k] for k in selected]


def cip_fp(smiles):
    m,centers=parse(smiles)
    decorated=[]
    for a,inv in zip(m.GetAtoms(),rdMolDescriptors.GetConnectivityInvariants(m)):
        state=a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else ('?' if a.GetIdx() in centers else '-')
        decorated.append(int.from_bytes(hashlib.blake2s(f'{inv}:{state}'.encode(),digest_size=4).digest(),'little'))
    return GEN.GetFingerprint(m,customAtomInvariants=decorated).ToBitString()


def collision_count(records, signatures):
    groups=defaultdict(Counter)
    for r,s in zip(records,signatures):groups[s][int(r[-1])]+=1
    return sum(len(c)>1 for c in groups.values()),sum(min(c.values()) for c in groups.values() if len(c)>1)


def main():
    exploratory.self_check()
    source=ROOT/'train.csv'; source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open() as f:
        reader=csv.reader(f);header=next(reader);raw=list(reader)
    dropped=[j for j in range(1,len(header)-1) if len({r[j] for r in raw})==1]
    assert len(dropped)==49 and len(raw)==8349
    meta=list(csv.DictReader((ROOT/'충돌33그룹_입체분석.csv').open(encoding='utf-8-sig')))
    conflict_rows={int(x['csv_row']) for x in meta}
    dictionary={r['feature']:(r['environment_radius2'],r['CIP_state']) for r in csv.DictReader((ROOT/'입체_파생변수_사전.csv').open(encoding='utf-8-sig')) if r['kind']=='local_stereo_count'}
    full_path=ROOT/'train_eda_exploratory126.csv'
    if not full_path.exists():
        assert 'stereo_env_097' in next(csv.reader((ROOT/'train_eda.csv').open()))
        (ROOT/'train_eda.csv').rename(full_path)
    with full_path.open() as f:
        reader=csv.DictReader(f)
        candidate={n:{k:float(v) for k,v in r.items() if k.startswith('stereo_')} for n,r in enumerate(reader,2) if n in conflict_rows}
    grouped=defaultdict(list)
    for r in meta:grouped[r['conflict_group']].append(r)
    unresolved=[]
    for gid,g in grouped.items():
        for a,b in combinations(g,2):
            if a['label']==b['label']:continue
            ia,ib=int(a['csv_row']),int(b['csv_row'])
            if all(candidate[ia][k]==candidate[ib][k] for k in BASE):unresolved.append((gid,ia,ib))
    masks={}
    for name in dictionary:
        mask=sum(1<<n for n,(_,a,b) in enumerate(unresolved) if candidate[a][name]!=candidate[b][name])
        if mask:masks.setdefault(mask,name)
    goal=(1<<len(unresolved))-1; selected_names=None
    for size in range(1,len(masks)+1):
        for choice in combinations(masks.items(),size):
            mask=0
            for bits,name in choice:mask|=bits
            if mask==goal:selected_names=[name for bits,name in choice];break
        if selected_names is not None:break
    assert selected_names is not None and len(selected_names)==5
    selected=[dictionary[n] for n in selected_names]
    for s in ('N[C@@H](C)C(=O)O',*[r['SMILES'] for r in meta]):
        m=Chem.MolFromSmiles(s);alternate=Chem.MolToSmiles(m,doRandom=True,isomericSmiles=True)
        assert compact(s,selected)==compact(alternate,selected)
        assert cip_fp(s)==cip_fp(alternate)
    print('Extracting compact features and replacement fingerprint...',flush=True)
    small=[compact(r[0],selected) for r in raw]
    replacement=[cip_fp(r[0]) for r in raw]
    fp_keys=[(''.join(r[1:1025]),''.join(r[1025:2049]),''.join(r[2049:3073]),*(float(x) for x in r[3073:3077])) for r in raw]
    base_result=collision_count(raw,fp_keys)
    small_result=collision_count(raw,[(key,tuple(v)) for key,v in zip(fp_keys,small)])
    fp_result=collision_count(raw,[(fp,*key[1:]) for fp,key in zip(replacement,fp_keys)])
    assert base_result==(33,34) and small_result==(0,0) and fp_result==(0,0)
    keep=[j for j in range(len(header)-1) if j not in dropped]
    names=BASE+selected_names
    with (ROOT/'train_eda.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow([header[j] for j in keep]+names+['label'])
        for r,v in zip(raw,small):w.writerow([r[j] for j in keep]+v+[r[-1]])
    with (ROOT/'train_eda_cip_fp.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['SMILES']+[f'cip_morgan_{n}' for n in range(1024)]+[header[j] for j in keep if j>=1025]+['label'])
        for r,bits in zip(raw,replacement):w.writerow([r[0]]+list(bits)+[r[j] for j in keep if j>=1025]+[r[-1]])
    def full_extract(r):
        d=exploratory.describe(r[0],with_morgan=False)
        vec=list(d['scalar'].values())
        vec += [sum(v for k,v in d['local'].items() if k not in dictionary.values())]
        vec += [d['local'][k] for k in dictionary.values()]
        vec += [d['scalar'][balance]*float(r[header.index(prop)]) for balance in ('stereo_R_minus_S','stereo_r_minus_s','stereo_defined_fraction') for prop in exploratory.PROPS]
        vec += [''.join(r[a:b]).count('1') for a,b in ((1,1025),(1025,2049),(2049,3073))]
        assert len(vec)==126
        return vec
    print('Benchmarking three modes, three runs each, excluding CSV I/O...',flush=True)
    functions={'탐색126열':full_extract,'축소8열':lambda r:compact(r[0],selected),'기존1024열대체':lambda r:cip_fp(r[0])}
    timings={}
    for mode,func in functions.items():
        rounds=[]
        for repeat in range(3):
            begin=perf_counter()
            for row in raw:func(row)
            rounds.append(perf_counter()-begin)
        timings[mode]={'runs_seconds':rounds,'median_seconds_8349':median(rounds),'median_ms_per_molecule':median(rounds)/len(raw)*1000}
    for filename,expected in [('train_eda.csv',3037),('train_eda_cip_fp.csv',3029)]:
        with (ROOT/filename).open() as f:
            r=csv.reader(f);h=next(r);assert len(h)==expected
            output=[]
            for n,row in enumerate(r):
                assert row[0]==raw[n][0] and row[-1]==raw[n][-1]
                if filename=='train_eda.csv':assert row[:len(keep)]==[raw[n][j] for j in keep]
                else:assert row[1025:-1]==[raw[n][j] for j in keep if j>=1025]
                output.append((tuple(float(x) for x in row[1:-1]),int(row[-1])))
        output.sort();assert len(output)==8349
        assert not any(a[0]==b[0] and a[1]!=b[1] for a,b in zip(output,output[1:]))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    summary={'source_sha256':source_hash,'rdkit':rdBase.rdkitVersion,'python':platform.python_version(),
        'machine':platform.machine(),'rows':8349,'removed_constants':49,'exploratory_added':126,
        'compact_added':8,'compact_columns':3037,'replacement_added':0,'replacement_columns':3029,
        'compact_features':names,'local_features':[{'name':n,'environment':k[0],'CIP':k[1]} for n,k in zip(selected_names,selected)],
        'selection_scope':'전역 요약 3개를 보존하고 알려진 잔여 대립 쌍 8개를 구분하는 조건에서 국소 변수 최소5개를 탐색했다. 전체 문제의 최적 변수 수나 독립 예측 검증은 아니다.',
        'unresolved_after_three_summaries':unresolved,
        'remaining_conflicts':{'original':base_result[0],'compact8':small_result[0],'replacement':fp_result[0]},
        'benchmark':timings,'csv_sizes_bytes':{n:(ROOT/n).stat().st_size for n in ['train.csv','train_eda_exploratory126.csv','train_eda.csv','train_eda_cip_fp.csv']},
        'extra_numeric_float64_bytes':{'exploratory126':8349*126*8,'compact8':8349*8*8,'replacement':0},
        'replacement_definition':{'radius':2,'bits':1024,'custom_atom_invariants':'BLAKE2s32(connectivity invariant:CIP R/S/r/s/?/-)',
            'includeChirality':False,'why_false':'CIP states already explicitly decorated; bond stereo retained by RDKit',
            'requires_retraining':True,'original_ECFP_generation_unknown':True},
        'limits':['known-dataset representation collision resolution, not causal toxicity explanation',
            '97 candidates selected from known conflicts; independent chemical-family validation still required',
            'fixed 1024-bit hashing can collide on future molecules','cost excludes model training, services, I/O and infrastructure pricing']}
    (ROOT/'compact_eda_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
