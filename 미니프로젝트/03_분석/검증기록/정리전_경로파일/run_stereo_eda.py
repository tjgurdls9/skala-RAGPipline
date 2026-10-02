"""Conflict-focused EDA. Run with ../.venv_eda/bin/python run_stereo_eda.py.

RDKit 2025.9.6, numpy and scipy. Input labels and SMILES are never changed.
The context vocabulary is discovered in the known conflict cases: exploratory,
not held-out prediction validation. No label, row ID or group ID is a feature.
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import sys
import textwrap

import numpy as np
from scipy.stats import fisher_exact
from rdkit import Chem, rdBase
from rdkit.Chem import Draw, rdCIPLabeler, rdFingerprintGenerator
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PROPS = ('MolWt', 'clogp', 'sa_score', 'qed')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)


def write_csv(name, records, fields=None):
    fields = fields or list(records[0])
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def environment(mol, center):
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, 2, center))
    atoms = {center}
    for bid in bonds:
        bond = mol.GetBondWithIdx(bid)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=bonds,
                                  rootedAtAtom=center, canonical=True, isomericSmiles=False)


def describe(smiles, with_morgan=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot parse SMILES: {smiles}')
    potential = list(Chem.FindPotentialStereo(mol))
    rdCIPLabeler.AssignCIPLabels(mol)
    centers = {s.centeredOn for s in potential if s.type == Chem.StereoType.Atom_Tetrahedral}
    centers.update(a.GetIdx() for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
    local, atom_info = Counter(), []
    for idx in sorted(centers):
        atom = mol.GetAtomWithIdx(idx)
        code = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else '?'
        env = environment(mol, idx)
        local[(env, code)] += 1
        atom_info.append({'atom_index': idx, 'CIP': code, 'environment_radius2': env})
    codes = Counter(a['CIP'] for a in atom_info)
    double = [s for s in potential if s.type == Chem.StereoType.Bond_Double]
    scalar = {f'stereo_{code}_count': codes[code] for code in ('R', 'S', 'r', 's')}
    scalar.update(stereo_unassigned_count=codes['?'],
                  stereo_potential_count=len(centers),
                  stereo_specified_count=len(centers)-codes['?'],
                  stereo_R_minus_S=codes['R']-codes['S'],
                  stereo_r_minus_s=codes['r']-codes['s'],
                  stereo_defined_fraction=(len(centers)-codes['?'])/len(centers) if centers else 0,
                  stereo_E_count=sum(b.GetStereo() == Chem.BondStereo.STEREOE for b in mol.GetBonds()),
                  stereo_Z_count=sum(b.GetStereo() == Chem.BondStereo.STEREOZ for b in mol.GetBonds()),
                  stereo_double_unassigned_count=sum(s.specified == Chem.StereoSpecified.Unspecified for s in double))
    return {'mol': mol, 'canonical_isomeric': Chem.MolToSmiles(mol, isomericSmiles=True),
            'canonical_connectivity': Chem.MolToSmiles(mol, isomericSmiles=False),
            'atoms': atom_info, 'local': local, 'scalar': scalar,
            'chiral_morgan': GEN.GetFingerprint(mol).ToBitString() if with_morgan else None}


def minimum_errors(items, signature):
    bins = defaultdict(Counter)
    for item in items:
        bins[signature(item)][item['label']] += 1
    mixed = [v for v in bins.values() if len(v) > 1]
    return len(mixed), sum(sum(v.values()) for v in mixed), sum(min(v.values()) for v in mixed)


def pair_kind(a, b):
    if a['canonical_isomeric'] == b['canonical_isomeric']:
        return '같은 정규화 구조'
    if a['scalar']['stereo_unassigned_count'] != b['scalar']['stereo_unassigned_count']:
        return '입체정보 지정 범위 차이'
    if any(x['CIP'] in ('r', 's') for d in (a,b) for x in d['atoms']):
        return 'r/s를 포함한 입체배치 차이'
    if a['scalar']['stereo_unassigned_count'] or b['scalar']['stereo_unassigned_count']:
        return '일부 미지정 상태에서의 입체배치 차이'
    mirror = Chem.Mol(a['mol'])
    for atom in mirror.GetAtoms():
        if atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                  Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            atom.InvertChirality()
    return ('전체 사면체 배치 반전 관계' if Chem.MolToSmiles(mirror, isomericSmiles=True)
            == b['canonical_isomeric'] else '부분 입체배치 차이')


def self_check():
    a, b = describe('N[C@@H](C)C(=O)O'), describe('N[C@H](C)C(=O)O')
    assert a['canonical_connectivity'] == b['canonical_connectivity']
    assert a['canonical_isomeric'] != b['canonical_isomeric'] and a['local'] != b['local']
    random_smiles = Chem.MolToSmiles(a['mol'], doRandom=True, isomericSmiles=True)
    assert describe(random_smiles)['local'] == a['local'], 'Atom-order-dependent environment'
    assert describe('NC(C)C(=O)O')['scalar']['stereo_unassigned_count'] == 1
    assert describe('CCO')['scalar']['stereo_potential_count'] == 0


def main():
    self_check()
    source = ROOT / 'train.csv'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='') as f:
        reader = csv.reader(f)
        header, raw = next(reader), list(reader)
    assert len(raw) == 8349 and len(header) == 3078
    dropped = [i for i in range(1,len(header)-1) if len({r[i] for r in raw}) == 1]
    assert len(dropped) == 49 and all(header[i].startswith('ptfp_') for i in dropped)
    groups, items = defaultdict(list), []
    print('Parsing 8349 molecules...', flush=True)
    for idx, row in enumerate(raw):
        desc = describe(row[0])
        fp = (''.join(row[1:1025]), ''.join(row[1025:2049]), ''.join(row[2049:3073]))
        key = (*fp, *(float(v) for v in row[3073:3077]))
        item = {'idx': idx, 'row': idx+2, 'label': int(row[-1]), 'SMILES': row[0],
                'key': key, 'desc': desc, 'properties': dict(zip(PROPS, key[3:]))}
        item['desc']['scalar'].update(ecfp_onbits=fp[0].count('1'), fcfp_onbits=fp[1].count('1'),
                                     ptfp_onbits=fp[2].count('1'))
        items.append(item)
        groups[key].append(item)
    mixed = [g for g in groups.values() if len({i['label'] for i in g}) > 1]
    assert len(mixed) == 33 and sum(map(len,mixed)) == 73
    contexts = sorted({k for g in mixed for i in g for k in i['desc']['local']})
    context_names = {key:f'stereo_env_{n:03d}' for n,key in enumerate(contexts,1)}
    for item in items:
        d = item['desc']
        derived = dict(d['scalar'])
        derived['stereo_uncovered_environment_count'] = sum(v for k,v in d['local'].items() if k not in context_names)
        for balance in ('stereo_R_minus_S', 'stereo_r_minus_s', 'stereo_defined_fraction'):
            for prop in PROPS:
                derived[f'{balance}_x_{prop}'] = derived[balance]*item['properties'][prop]
        derived.update({name:d['local'][key] for key,name in context_names.items()})
        item['derived'] = derived
    feature_names = list(items[0]['derived'])
    assert not {'label','row','group'}.intersection(feature_names)
    keep = [i for i in range(len(header)-1) if i not in dropped]
    output_header = [header[i] for i in keep] + feature_names + ['label']
    target = ROOT / 'train_eda_exploratory126.csv'
    temp = target.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        w = csv.writer(f); w.writerow(output_header)
        for row,item in zip(raw,items):
            w.writerow([row[i] for i in keep] + [item['derived'][n] for n in feature_names] + [row[-1]])
    temp.replace(target)
    dictionary = [{'feature':name,'kind':'local_stereo_count','environment_radius2':key[0],
                   'CIP_state':key[1],'meaning':f'중심 배치 {key[1]} / 반경2 환경 {key[0]}의 개수'}
                  for key,name in context_names.items()]
    for name in feature_names:
        if name not in context_names.values():
            dictionary.append({'feature':name,'kind':'interaction' if '_x_' in name else 'global_count_or_summary',
                               'environment_radius2':'','CIP_state':'','meaning':name})
    write_csv('입체_파생변수_사전.csv', dictionary)
    conflict_rows, pair_rows = [], []
    for n,g in enumerate(mixed,1):
        gid = f'C{n:02d}'
        for item in g:
            d = item['desc']
            conflict_rows.append({'conflict_group':gid,'csv_row':item['row'],'label':item['label'],
                'SMILES':item['SMILES'],'canonical_isomeric':d['canonical_isomeric'],
                'canonical_connectivity':d['canonical_connectivity'],
                'CIP_by_atom':json.dumps(d['atoms'],ensure_ascii=False),
                'local_features':json.dumps({context_names[k]:v for k,v in d['local'].items()},ensure_ascii=False),
                **item['properties'], **{k:v for k,v in d['scalar'].items() if k.startswith('stereo_')}})
        assert len({i['desc']['canonical_connectivity'] for i in g}) == 1
        for a,b in combinations(g,2):
            if a['label'] == b['label']:
                continue
            changes = [{'feature':name,'environment':key[0],'CIP':key[1],
                        'row_a_count':a['desc']['local'][key],'row_b_count':b['desc']['local'][key]}
                       for key,name in context_names.items() if a['desc']['local'][key] != b['desc']['local'][key]]
            pair_rows.append({'conflict_group':gid,'row_a':a['row'],'label_a':a['label'],
                'row_b':b['row'],'label_b':b['label'],'difference_type':pair_kind(a['desc'],b['desc']),
                'changed_features':json.dumps(changes,ensure_ascii=False),
                'chiral_morgan_bits_different':a['desc']['chiral_morgan'] != b['desc']['chiral_morgan']})
    write_csv('충돌33그룹_입체분석.csv', conflict_rows)
    write_csv('충돌_대립label_쌍별차이.csv',pair_rows)
    signatures = {
        '기존 숫자 특징':lambda i:i['key'],
        '기존 + R/S/r/s/미지정 개수':lambda i:(i['key'],*(i['desc']['scalar'][f'stereo_{c}_count'] for c in ('R','S','r','s')),i['desc']['scalar']['stereo_unassigned_count']),
        '기존 + chirality Morgan r2 1024':lambda i:(i['key'],i['desc']['chiral_morgan']),
        '기존 + 주변환경×CIP count':lambda i:(i['key'],*(i['desc']['local'][k] for k in contexts)),
        '기존 + 정규화 입체 SMILES (진단용)':lambda i:(i['key'],i['desc']['canonical_isomeric'])}
    validation = []
    for name,signature in signatures.items():
        a,b,c = minimum_errors(items,signature)
        validation.append({'features':name,'remaining_conflict_groups':a,'remaining_conflict_rows':b,
                           'minimum_identical_input_errors':c,'scope':'전체8349행의 동일입력 진단; 예측성능 아님'})
    write_csv('충돌_파생변수검증.csv',validation)

    print('Scanning structural patterns and property strata...', flush=True)
    labels = np.array([i['label'] for i in items])
    pattern_rows = []
    for key,name in context_names.items():
        present = np.array([bool(i['desc']['local'][key]) for i in items])
        yes,noyes = int(labels[present].sum()), int(present.sum()-labels[present].sum())
        no,nono = int(labels[~present].sum()), int((~present).sum()-labels[~present].sum())
        odds,p = fisher_exact([[yes,noyes],[no,nono]])
        plus,minus,varied = [],[],[]
        for n,g in enumerate(mixed,1):
            values = [i['desc']['local'][key] for i in g]
            if len(set(values)) < 2:
                continue
            gid=f'C{n:02d}';varied.append(gid)
            tox = [i['desc']['local'][key] for i in g if i['label']==1]
            safe = [i['desc']['local'][key] for i in g if i['label']==0]
            if min(tox)>max(safe):plus.append(gid)
            elif max(tox)<min(safe):minus.append(gid)
        pattern_rows.append({'feature':name,'environment':key[0],'CIP':key[1],
            'present_n':yes+noyes,'present_toxic_n':yes,'present_toxic_rate':yes/(yes+noyes),
            'present_unique_connectivity_n':len({i['desc']['canonical_connectivity'] for i in items if i['desc']['local'][key]}),
            'present_unique_isomeric_n':len({i['desc']['canonical_isomeric'] for i in items if i['desc']['local'][key]}),
            'absent_n':no+nono,'absent_toxic_rate':no/(no+nono) if no+nono else '',
            'odds_ratio':odds,'fisher_p':p,'BH_q':None,
            'separated_conflict_group_n':len(plus)+len(minus),'toxic_direction_group_n':len(plus),
            'nontoxic_direction_group_n':len(minus),'direction_consistency':max(len(plus),len(minus))/(len(plus)+len(minus)) if plus or minus else '',
            'toxic_direction_groups':','.join(plus),'nontoxic_direction_groups':','.join(minus),
            'varying_groups':','.join(varied),'interpretation':'관찰 연관; 충돌사례에서 선정한 후보이며 인과/독립검증 아님'})
    order = sorted(range(len(pattern_rows)),key=lambda j:pattern_rows[j]['fisher_p'])
    prev=1.0
    for rank in range(len(order),0,-1):
        j=order[rank-1];prev=min(prev,pattern_rows[j]['fisher_p']*len(order)/rank)
        pattern_rows[j]['BH_q']=prev
    write_csv('입체환경_독성패턴.csv',pattern_rows)

    # Compare CIP states within the same local environment, not arbitrary unrelated molecules.
    strata=[]; edges={p:np.quantile([i['properties'][p] for i in items],[.25,.5,.75]).tolist() for p in PROPS}
    for env in sorted({k[0] for k in contexts}):
        for prop in PROPS:
            for q in range(4):
                subset=[i for i in items if int(np.searchsorted(edges[prop],i['properties'][prop],side='right'))==q]
                for state in ('R','S','r','s','?'):
                    chosen=[i for i in subset if i['desc']['local'][(env,state)]>0]
                    if not chosen:continue
                    strata.append({'environment':env,'property':prop,'quartile':q+1,'CIP':state,
                        'n':len(chosen),'toxic_n':sum(i['label'] for i in chosen),
                        'toxic_rate':sum(i['label'] for i in chosen)/len(chosen),
                        'lower_bound':edges[prop][q-1] if q else '-inf',
                        'upper_bound':edges[prop][q] if q<3 else 'inf',
                        'note':'행 단위 관찰 비율; 상태간 집합은 겹칠 수 있으며 사분위 합계만 합산 가능'})
    write_csv('입체환경_물성조건별패턴.csv',strata)
    correlations=[]
    scalar_covariates=['ecfp_onbits','fcfp_onbits','ptfp_onbits']+list(PROPS)
    for key,name in context_names.items():
        x=np.array([i['desc']['local'][key] for i in items],dtype=float)
        rec={'feature':name,'environment':key[0],'CIP':key[1]}
        for cov in scalar_covariates:
            y=np.array([i['properties'][cov] if cov in PROPS else i['desc']['scalar'][cov] for i in items])
            rec[cov+'_pearson']=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None
        correlations.append(rec)
    write_csv('입체환경_기존지표연관.csv',correlations)

    # Independent verification from exported rows, using sorted vectors rather than the grouping helper.
    label_position=output_header.index('label')
    with target.open() as f:
        check=csv.reader(f);assert next(check)==output_header
        vectors=[]
        for n,row in enumerate(check):
            assert row[0]==raw[n][0] and row[label_position]==raw[n][-1]
            assert row[:len(keep)]==[raw[n][j] for j in keep]
            vectors.append((tuple(float(x) for x in row[1:label_position]),int(row[label_position])))
    assert len(vectors)==8349
    vectors.sort(); independently_mixed=0
    for j in range(1,len(vectors)):
        if vectors[j][0]==vectors[j-1][0] and vectors[j][1]!=vectors[j-1][1]:
            independently_mixed+=1
    assert independently_mixed==0
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
    # Randomized atom ordering must preserve all candidate fragment counts for all 73 cases.
    for g in mixed:
        for i in g:
            alternate=Chem.MolToSmiles(i['desc']['mol'],doRandom=True,isomericSmiles=True)
            assert describe(alternate)['local']==i['desc']['local']
    assert validation[0]['remaining_conflict_groups']==33
    assert validation[3]['remaining_conflict_groups']==0

    pdf_pages=[]
    font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
    title_font=ImageFont.truetype(font_path,30);small_font=ImageFont.truetype(font_path,18)
    pdf_path=ROOT/'충돌33그룹_구조비교.pdf'
    for n,g in enumerate(mixed,1):
        page=Image.new('RGB',(1400,1100),'white');draw=ImageDraw.Draw(page)
        draw.text((35,20),f'C{n:02d} | identical numeric inputs; preserved stereochemistry',font=title_font,fill='black')
        draw.text((35,65),'CIP: R/S and r/s are distinct states; ? = unspecified. Label 1 = toxic, 0 = nontoxic.',font=small_font,fill='black')
        props=' | '.join(f'{p}: {g[0]["properties"][p]:.4g}' for p in PROPS)
        draw.text((35,100),props,font=small_font,fill='black')
        for j,i in enumerate(g):
            mol=Chem.Mol(i['desc']['mol'])
            for atom in i['desc']['atoms']:
                mol.GetAtomWithIdx(atom['atom_index']).SetProp('atomNote',str(atom['atom_index'])+':'+atom['CIP'])
            pic=Draw.MolToImage(mol,size=(660,390),highlightAtoms=[a['atom_index'] for a in i['desc']['atoms']])
            x,y=35+(j%2)*690,150+(j//2)*470
            page.paste(pic,(x,y))
            cip=', '.join(f"atom {a['atom_index']}:{a['CIP']}" for a in i['desc']['atoms'])
            caption=f"CSV row {i['row']} | label {i['label']} | {cip}"
            for line_no,line in enumerate(textwrap.wrap(caption,width=65)):
                draw.text((x,y+395+line_no*22),line,font=small_font,fill='black')
        draw.text((35,1060),'Atom indices are local to each input SMILES. These are 2D depictions, not calculated 3D conformers.',font=small_font,fill='black')
        pdf_pages.append(page)
        if n==4:page.save(ROOT/'입체차이_예시_C04.png')
    pdf_pages[0].save(pdf_path,save_all=True,append_images=pdf_pages[1:],resolution=150)

    summary={'source':'train.csv','sha256':original_hash,'rows':len(raw),'original_columns':len(header),
        'label_mapping':{'0':'nontoxic','1':'toxic'},'dropped_columns':[header[j] for j in dropped],
        'retained_original_columns':len(keep)+1,'added_columns':len(feature_names),
        'output_columns':len(output_header),'local_environment_state_columns':len(contexts),
        'radius':2,'rdkit_version':rdBase.rdkitVersion,'group_count':len(mixed),
        'conflict_rows':len(conflict_rows),'opposite_label_pairs':len(pair_rows),
        'pair_types':dict(Counter(r['difference_type'] for r in pair_rows)),
        'validation':validation,'quartile_edges':edges,
        'checks':['8349 original rows/SMILES/labels/retained values preserved','49 PTFP constants removed',
                  'all 33 groups have identical canonical connectivity',
                  'atom-order invariance of local features on all 73 conflict rows',
                  'independent full exported numeric vector check has zero mixed-label identical inputs',
                  'source hash unchanged'],
        'scope':'Exploratory conflict-selected vocabulary, no causal or held-out performance claim'}
    (ROOT/'eda_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    repeated=sorted([p for p in pattern_rows if p['separated_conflict_group_n']>=2],
                    key=lambda p:(-p['separated_conflict_group_n'],-p['direction_consistency']))
    report=['# 충돌을 설명하기 위한 입체구조 EDA 결과','',
        f'입력: 8,349행. RDKit {rdBase.rdkitVersion}. label 1=독성, 0=비독성. 원본 SMILES·label은 유지했다.',
        '',f'**상수 49열을 제거하고 구조 파생변수 {len(feature_names)}열을 추가했다. 기존 33개 label 충돌은 새 숫자 입력에서 0개로 줄었다.**',
        '', '**이 결과는 현재 기록의 구조 차이를 표현해 구분했다는 뜻이다. 독성 기전이나 새로운 분자의 예측 성능을 검증한 결과가 아니다.**',
        '', '## 무엇을 추가했는가','',
        f'- 입체중심 주변 반경2 원자·결합 환경과 CIP 상태(R/S/r/s/미지정)를 결합한 count 변수 {len(contexts)}개.',
        '- 전체 R/S/r/s/미지정 개수, 지정 비율, E/Z, 기존 fingerprint on-bit 수 및 후보 환경 밖 입체중심 수.',
        '- R−S, r−s, 지정 비율과 MolWt·clogp·sa_score·qed의 곱 12개. 같은 숫자 물성끼리의 조합은 구분력을 만들지 못하므로 구조 변수를 먼저 추출했다.',
        '- 후보 환경은 충돌 73행에서 탐색했다. label 값은 특징 계산에 사용하지 않았지만, 충돌을 찾는 데 label을 사용했으므로 독립 검증은 아니다.',
        '- 주변 환경 코드는 중심 원자로부터 두 결합까지의 부분 그래프를 직렬화한 것이다. 완전한 분자 SMILES나 고정 독성 작용기의 이름으로 해석하지 않는다. 기존 fingerprint on-bit 수는 원래 1,024비트를 기준으로 계산했다.',
        '', '## 동일 입력의 충돌 비교','', '|입력|남은 충돌 그룹|해당 행|동일입력 최소 오분류|','|---|---:|---:|---:|']
    for r in validation:report.append(f"|{r['features']}|{r['remaining_conflict_groups']}|{r['remaining_conflict_rows']}|{r['minimum_identical_input_errors']}|")
    report += ['', '33그룹 모두 정규화한 연결 구조가 같았고, 입체정보를 유지한 정규화 SMILES는 대립 label 기록을 구분했다. 미지정 상태는 R/S·라세미 혼합물로 추정하지 않았다.',
        '', '## 반복되는 후보 패턴','',
        '동일한 숫자 특징의 그룹 안에서 해당 변수 하나가 모든 독성/비독성 기록을 나누는지 확인했다. 아래는 두 그룹 이상에서 나눈 변수이며, 방향이 반대인 사례도 함께 표시한다. 비트·부분 환경은 생물학적 작용기 확정이 아니다.',
        '', '|변수|CIP|주변 환경|구분한 그룹|독성 쪽 증가|비독성 쪽 증가|전체 데이터 해당 상태 n|전체 독성 비율|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for p in repeated[:12]:report.append(f"|{p['feature']}|{p['CIP']}|`{p['environment']}`|{p['separated_conflict_group_n']}|{p['toxic_direction_group_n']}|{p['nontoxic_direction_group_n']}|{p['present_n']}|{p['present_toxic_rate']:.1%}|")
    if not repeated:report.append('|두 그룹 이상에서 반복된 단일 변수 없음||||||||')
    specific=next(p for p in pattern_rows if p['environment']=='C(CC)(NC)C(=O)N' and p['CIP']=='R')
    report += ['', f"**구체적인 반복 후보:** `C(CC)(NC)C(=O)N` 주변 환경의 R 배치는 C10·C12·C17에서 독성 쪽을 구분했다. 전체 데이터에서 이 환경/배치를 가진 {specific['present_n']}행은 모두 독성 label이었고, 입체구조 포함 정규화 구조 {specific['present_unique_isomeric_n']}개, 연결 구조 {specific['present_unique_connectivity_n']}개였다. 같은 환경의 S 배치 7행은 3행이 독성이다. 작은 화학 계열에서 발견한 관찰이며 보편 규칙은 아니다.",
        '', '**반례:** `C(C)(CC)N(C)C` 환경에서는 C05가 S 쪽 독성, C14·C15는 R 쪽 독성이었다. C05의 MolWt/clogp는 377.488/3.94732, C14·C15는 503.546/3.4078로 다르고 전체 연결 구조도 다르다. 따라서 이 반경2 환경과 R/S만으로 공통 독성 방향을 정할 수 없다. 물성·더 넓은 구조를 조건으로 추가하는 가설은 가능하지만 세 그룹으로 임계값을 정해 기전으로 선언하지 않는다.',
        '', '**물성 조건 예시:** `C(C)(CC)N(C)C` 환경의 R 배치를 가진 기록의 독성 비율은 clogp 3.7351–4.6525 구간에서 19/46(41.3%), 4.6525 이상에서는 31/45(68.9%)였다. 같은 두 구간의 S 배치는 각각 3/11(27.3%), 5/13(38.5%)였다. 이는 친유성 구간과 label의 관찰 관계이며, 다른 구조·표본 구성의 영향을 통제한 인과 효과는 아니다.',
        '', '전체 97개 후보의 Fisher 검정 및 BH 보정은 `입체환경_독성패턴.csv`에 있다. 작은 사례·중복 화학 계열·후보 선택 때문에 q값을 원인 증명으로 해석하지 않는다.',
        'MolWt·clogp·SA·QED 사분위 안에서 같은 주변 환경의 배치별 label 비율은 `입체환경_물성조건별패턴.csv`에 있다. 원자료의 행 수를 분모로 사용했고, 배치별 집합은 겹칠 수 있다.',
        'ECFP/FCFP/PTFP와의 수치 연관은 on-bit 수를 사용해 요약했으며, 전체 비트별 상호작용 분석을 수행한 것은 아니다.',
        '', '## 33그룹 개별 설명','', '[모든 그룹 구조 그림 PDF](충돌33그룹_구조비교.pdf). 원자의 R/S/r/s/미지정 상태를 표시했다. 그림은 2차원 표현이며 실제 3차원 좌표가 아니다.']
    for n,g in enumerate(mixed,1):
        gid=f'C{n:02d}';pairs=[p for p in pair_rows if p['conflict_group']==gid]
        prop_text=', '.join('{}={:.6g}'.format(p,g[0]['properties'][p]) for p in PROPS)
        report += ['',f'### {gid}', '', f'공통 물성: {prop_text}.',
                   '', '|CSV 행|label|입체중심 원자 인덱스와 배치|','|---|---:|---|']
        for i in g:
            cip=', '.join(f"{a['atom_index']}:{a['CIP']}" for a in i['desc']['atoms']) or '없음'
            report.append(f"|{i['row']}|{i['label']}|{cip}|")
        report.append('')
        for p in pairs:
            changes=json.loads(p['changed_features'])
            changed=', '.join(f"{c['feature']}({c['CIP']}):{c['row_a_count']}→{c['row_b_count']}" for c in changes)
            report.append(f"- 행 {p['row_a']} ↔ {p['row_b']}: **{p['difference_type']}**. 달라진 파생변수: {changed}.")
        report.append('관측된 입체표현 차이는 기존 입력이 놓친 구분 정보다. 해당 입체 차이가 독성 발생 원인이라는 기전은 현재 CSV만으로 확정하지 않는다.')
    report += ['', '## 검증·재현','',
        '전체 출력의 기존 값·SMILES·label 보존, 원본 파일 hash, 상수 열 수, 73행의 원자 순서 무작위 변경 후 특징 불변성을 확인했다. 출력 CSV를 다시 읽고 전체 숫자 벡터를 정렬해 label 충돌 0개를 별도로 검증했다.',
        '', '```sh','../.venv_eda/bin/python run_stereo_eda.py','```',
        '', '재실행은 생성 결과만 갱신한다. `train.csv`는 변경하지 않는다. 환경 의존성: RDKit 2025.9.6, numpy, scipy, Pillow.',
        '', '방법 근거: [RDKit CIP 배정](https://www.rdkit.org/docs/source/rdkit.Chem.rdCIPLabeler.html), [Morgan generator](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html).']
    (ROOT/'충돌_EDA_분석.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        self_check();print('Stereo parsing and atom-order invariance self-check passed.')
    else:
        main()
