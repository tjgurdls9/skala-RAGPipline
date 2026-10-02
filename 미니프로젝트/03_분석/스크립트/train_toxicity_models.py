"""Local toxicity modeling; original files are read-only. Run with .venv_eda Python."""
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import json
import os
import platform
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/toxicity-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import joblib
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                            confusion_matrix, f1_score, precision_score, recall_score,
                            roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import run_stereo_eda as history

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / '06_모델링'
SEED = 42
BASE = ['stereo_R_minus_S', 'stereo_r_minus_s', 'stereo_unassigned_count']
PROPS = list(history.PROPS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_frame(path):
    header = pd.read_csv(path, nrows=0).columns
    types = {c: 'int8' for c in header if c.startswith(('ecfp_', 'fcfp_', 'ptfp_')) or c == 'label'}
    return pd.read_csv(path, dtype=types)


def transform(frame, columns, local_dictionary):
    """Use the existing EDA parser for both training and prediction."""
    records = [history.describe(s, with_morgan=False) for s in frame.SMILES]
    fingerprints = np.empty((len(frame), 1024), dtype=np.int8)
    for i, record in enumerate(records):
        DataStructs.ConvertToNumpyArray(history.GEN.GetFingerprint(record['mol']), fingerprints[i])
    result = frame.drop(columns=['label'], errors='ignore').copy()
    result[[f'ecfp_{i}' for i in range(1024)]] = fingerprints
    derived = {name: [r['scalar'][name] for r in records] for name in BASE}
    derived.update({name: [r['local'][key] for r in records] for name, key in local_dictionary.items()})
    result = pd.concat([result, pd.DataFrame(derived, index=result.index)], axis=1)
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=r['mol'], includeChirality=False)
                       or 'ACYCLIC' for r in records])
    connectivity = np.array([r['canonical_connectivity'] for r in records])
    isomeric = np.array([r['canonical_isomeric'] for r in records])
    matrix = result[columns].to_numpy(dtype=np.float32)
    assert np.isfinite(matrix).all()
    return matrix, groups, connectivity, isomeric


def scores(y, probability, threshold=0.5):
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, probability)),
            'pr_auc_toxic': float(average_precision_score(y, probability)),
            'pr_auc_safe': float(average_precision_score(1-y, 1-probability)),
            'accuracy': float(accuracy_score(y, pred)),
            'precision_toxic': float(precision_score(y, pred, zero_division=0)),
            'recall_toxic': float(recall_score(y, pred, zero_division=0)),
            'f1_toxic': float(f1_score(y, pred, zero_division=0)),
            'precision_safe': float(precision_score(y, pred, pos_label=0, zero_division=0)),
            'recall_safe': float(recall_score(y, pred, pos_label=0, zero_division=0)),
            'f1_safe': float(f1_score(y, pred, pos_label=0, zero_division=0)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'specificity_toxic': float(tn/(tn+fp)),
            'specificity_safe': float(tp/(tp+fn)),
            'brier': float(brier_score_loss(y, probability)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def choose_threshold(y, probability):
    values = [(float(t), f1_score(y, probability >= t, average='macro'))
              for t in np.linspace(0.15, 0.85, 141)]
    return max(values, key=lambda pair: (pair[1], -abs(pair[0]-0.5)))[0], values


def safe_submission(template, toxic_pred, path):
    """Create a new CSV, preserving the exact template header and SMILES strings."""
    assert template.resolve() != path.resolve(), 'Do not overwrite the original template'
    with template.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ['SMILES', 'output'] and len(rows)-1 == len(toxic_pred)
    assert set(np.unique(toxic_pred)) <= {0, 1}
    saved = [rows[0]] + [[row[0], str(1-int(pred))] for row, pred in zip(rows[1:], toxic_pred)]
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        csv.writer(stream).writerows(saved)
    with temporary.open(encoding='utf-8-sig', newline='') as stream:
        check = list(csv.reader(stream))
    assert check == saved and [r[0] for r in check] == [r[0] for r in rows]
    temporary.replace(path)


def self_check():
    import tempfile
    history.self_check()
    metrics = scores(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert metrics['roc_auc'] == metrics['accuracy'] == metrics['macro_f1'] == 1
    assert metrics['tn'] == metrics['tp'] == 2 and metrics['fp'] == metrics['fn'] == 0
    with tempfile.TemporaryDirectory() as directory:
        template = Path(directory)/'original.csv'
        output = Path(directory)/'new.csv'
        template.write_text('SMILES,output\nCCO,1\nCCN,1\n', encoding='utf-8')
        before = digest(template)
        safe_submission(template, np.array([1, 0]), output)
        assert pd.read_csv(output).output.tolist() == [0, 1]
        assert digest(template) == before
        try:
            safe_submission(template, np.array([1, 0]), template)
        except AssertionError:
            pass
        else:
            raise AssertionError('Original overwrite was allowed')
    print('PASS: label inversion, CSV preservation, metrics and stereo parsing', flush=True)


def main():
    self_check()
    DEST.mkdir(exist_ok=True)
    started = perf_counter()
    source = ROOT/'02_데이터/train.csv'
    final_path = ROOT/'02_데이터/train_final_morgan_8vars.csv'
    input_path = ROOT/'01_원본자료/predict_input.csv'
    template = ROOT/'01_원본자료/predict_output_홍길동.csv'
    backup = ROOT/'01_원본자료/train_original_labels.csv.bak'
    protected = [source, final_path, input_path, template, backup,
                 next(ROOT.glob('04_노트북/*Morgan8_EDA.ipynb'))]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    raw, final, incoming = (read_frame(p) for p in [source, final_path, input_path])
    original_labels = pd.read_csv(backup, usecols=['label']).label.to_numpy()
    y = final.label.to_numpy(dtype=int)
    assert np.array_equal(y, 1-original_labels) and np.array_equal(y, raw.label)
    assert raw.SMILES.equals(final.SMILES) and final.shape == (8349, 3037)
    assert not raw.isna().any().any() and not incoming.isna().any().any()
    template_frame = pd.read_csv(template)
    assert template_frame.SMILES.equals(incoming.SMILES) and len(incoming) == 927
    columns = [c for c in final if c not in ['SMILES', 'label']]
    dictionary = pd.read_csv(ROOT/'03_분석/진단표/운영후보_8변수_사전.csv').fillna('')
    local_dictionary = {r.feature: (r.environment_radius2, r.CIP_state)
                        for r in dictionary.itertuples() if r.environment_radius2}
    print('Recomputing all training/prediction Morgan and stereo features...', flush=True)
    X, groups, connectivity, isomeric = transform(raw, columns, local_dictionary)
    assert np.array_equal(X, final[columns].to_numpy(dtype=np.float32)), 'EDA/train-serving mismatch'
    Q, qgroups, qconnectivity, qisomeric = transform(incoming, columns, local_dictionary)
    original_columns = [c for c in columns if not c.startswith('stereo_')]
    legacy = raw[original_columns].to_numpy(dtype=np.float32)
    np.savez_compressed(DEST/'feature_cache.npz', X=X, Q=Q, legacy=legacy, y=y,
                        groups=groups, connectivity=connectivity, isomeric=isomeric,
                        qgroups=qgroups, qconnectivity=qconnectivity, qisomeric=qisomeric)
    dev, holdout = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    assert not set(groups[dev]) & set(groups[holdout])
    assert not set(connectivity[dev]) & set(connectivity[holdout])
    cv = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(X[dev], y[dev], groups[dev]))
    split_rows = pd.DataFrame({'csv_row': np.arange(len(y))+2, 'SMILES': raw.SMILES,
                              'label_toxic': y, 'scaffold': groups, 'connectivity': connectivity,
                              'split': 'holdout', 'dev_fold': -1})
    split_rows.loc[dev, 'split'] = 'development'
    for fold, (train, valid) in enumerate(cv):
        assert not set(groups[dev[train]]) & set(groups[dev[valid]])
        assert not set(connectivity[dev[train]]) & set(connectivity[dev[valid]])
        split_rows.loc[dev[valid], 'dev_fold'] = fold
        assert len(set(y[dev[train]])) == len(set(y[dev[valid]])) == 2
    split_rows.to_csv(DEST/'split_manifest.csv', index=False)
    eligible_columns = [i for i, c in enumerate(columns) if not c.startswith('stereo_env_')]
    proper = X[:, eligible_columns]
    qproper = Q[:, eligible_columns]
    scaled = [i for i, c in enumerate(np.array(columns)[eligible_columns]) if c in PROPS+BASE]
    legacy_scaled = [i for i, c in enumerate(original_columns) if c in PROPS]
    def pipeline(estimator, indices):
        return make_pipeline(ColumnTransformer([('continuous', StandardScaler(), indices)],
                                               remainder='passthrough'), estimator)
    def boost(leaves, child):
        return LGBMClassifier(n_estimators=600, learning_rate=.035, num_leaves=leaves,
                              min_child_samples=child, colsample_bytree=.8, subsample=.85,
                              subsample_freq=1, reg_lambda=5, random_state=SEED,
                              n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
    candidates = [
        ('dummy_majority', 'proper', DummyClassifier(strategy='prior'), True),
        ('logistic_original', 'legacy', pipeline(LogisticRegression(C=.1, max_iter=1500), legacy_scaled), True),
        ('logistic_morgan_global3', 'proper', pipeline(LogisticRegression(C=.1, max_iter=1500), scaled), True),
        ('random_forest', 'proper', RandomForestClassifier(n_estimators=350, max_features='sqrt',
                                    min_samples_leaf=2, random_state=SEED, n_jobs=4), True),
        ('extra_trees', 'proper', ExtraTreesClassifier(n_estimators=400, max_features='sqrt',
                                    min_samples_leaf=2, random_state=SEED, n_jobs=4), True),
        ('lightgbm_31', 'proper', boost(31, 30), True),
        ('lightgbm_15', 'proper', boost(15, 50), True),
        ('svm_rbf', 'proper', pipeline(SVC(C=3, gamma='scale', probability=True, random_state=SEED), scaled), True),
        ('lightgbm_plus8_exploratory', 'full', boost(31, 30), False)]
    matrices = {'proper': proper, 'legacy': legacy, 'full': X}
    prediction_matrices = {'proper': qproper, 'legacy': incoming[original_columns].to_numpy(dtype=np.float32), 'full': Q}
    fold_rows, comparison, oof = [], [], {}
    for name, kind, estimator, eligible in candidates:
        oof[name] = np.empty(len(dev), dtype=float)
        candidate_started = perf_counter()
        for fold, (train, valid) in enumerate(cv):
            begin = perf_counter()
            model = clone(estimator).fit(matrices[kind][dev[train]], y[dev[train]])
            probability = model.predict_proba(matrices[kind][dev[valid]])[:, 1]
            assert np.isfinite(probability).all() and np.all((0 <= probability) & (probability <= 1))
            oof[name][valid] = probability
            metrics = scores(y[dev[valid]], probability)
            fold_rows.append({'model': name, 'fold': fold, 'train_n': len(train), 'valid_n': len(valid),
                              'seconds': perf_counter()-begin, **metrics})
            print(f'{name} fold={fold+1}/4 AUC={metrics["roc_auc"]:.4f} seconds={perf_counter()-begin:.1f}', flush=True)
            pd.DataFrame(fold_rows).to_csv(DEST/'cv_folds.csv', index=False)
        row = {'model': name, 'eligible': eligible, 'feature_count': matrices[kind].shape[1],
               'seconds': perf_counter()-candidate_started, **scores(y[dev], oof[name])}
        row['fold_auc_std'] = float(np.std([r['roc_auc'] for r in fold_rows if r['model'] == name], ddof=1))
        comparison.append(row)
        pd.DataFrame(comparison).to_csv(DEST/'model_comparison.csv', index=False)
    ranked = sorted([r for r in comparison if r['eligible']], key=lambda r: r['roc_auc'], reverse=True)
    selected_names = [ranked[0]['model']]
    selected_oof = oof[selected_names[0]]
    top_three = [r['model'] for r in ranked[:3]]
    ensemble_oof = np.mean([oof[name] for name in top_three], axis=0)
    ensemble_metrics = scores(y[dev], ensemble_oof)
    comparison.append({'model': 'mean_top3', 'eligible': True, 'feature_count': None,
                       'seconds': 0, 'fold_auc_std': None, **ensemble_metrics})
    if ensemble_metrics['roc_auc'] >= ranked[0]['roc_auc'] + .002:
        selected_names, selected_oof = top_three, ensemble_oof
    threshold, curve = choose_threshold(y[dev], selected_oof)
    lock = {'models': selected_names, 'threshold_toxic': threshold, 'primary_metric': 'OOF ROC-AUC',
            'threshold_metric': 'OOF macro-F1', 'ensemble_minimum_auc_gain': .002,
            'development_scores_at_threshold': scores(y[dev], selected_oof, threshold)}
    write_json(DEST/'selection_locked.json', lock)
    pd.DataFrame(comparison).sort_values('roc_auc', ascending=False).to_csv(DEST/'model_comparison.csv', index=False)
    pd.DataFrame(curve, columns=['threshold_toxic', 'oof_macro_f1']).to_csv(DEST/'threshold_search.csv', index=False)
    pd.DataFrame({'csv_row': dev+2, 'label_toxic': y[dev], **oof,
                  'selected_probability_toxic': selected_oof}).to_csv(DEST/'development_oof.csv', index=False)
    print('LOCKED', lock, flush=True)
    candidate_map = {name: (kind, estimator) for name, kind, estimator, _ in candidates}
    holdout_probabilities = []
    for name in selected_names:
        kind, estimator = candidate_map[name]
        model = clone(estimator).fit(matrices[kind][dev], y[dev])
        holdout_probabilities.append(model.predict_proba(matrices[kind][holdout])[:, 1])
    test_probability = np.mean(holdout_probabilities, axis=0)
    test_pred = (test_probability >= threshold).astype(int)
    final_scores = scores(y[holdout], test_probability, threshold)
    rng = np.random.default_rng(SEED)
    group_indices = [np.flatnonzero(groups[holdout] == group) for group in np.unique(groups[holdout])]
    auc_samples, f1_samples = [], []
    for _ in range(300):
        indices = np.concatenate([group_indices[i] for i in rng.integers(0, len(group_indices), len(group_indices))])
        if len(set(y[holdout][indices])) != 2:
            continue
        auc_samples.append(roc_auc_score(y[holdout][indices], test_probability[indices]))
        f1_samples.append(f1_score(y[holdout][indices], test_pred[indices], average='macro'))
    ci = {'roc_auc_95_percent': np.quantile(auc_samples, [.025, .975]).tolist(),
          'macro_f1_95_percent': np.quantile(f1_samples, [.025, .975]).tolist(), 'group_bootstrap_repeats': 300}
    errors = pd.DataFrame({'csv_row': holdout+2, 'SMILES': raw.SMILES.iloc[holdout].to_numpy(),
                          'scaffold': groups[holdout], 'label_toxic': y[holdout],
                          'probability_toxic': test_probability, 'prediction_toxic': test_pred})
    errors['error_kind'] = np.where(test_pred == y[holdout], 'correct', np.where(test_pred == 1, 'false_positive', 'false_negative'))
    errors.to_csv(DEST/'holdout_predictions.csv', index=False)
    errors.loc[errors.error_kind != 'correct'].to_csv(DEST/'holdout_errors.csv', index=False)
    slice_rows = []
    stereo = X[:, columns.index('stereo_unassigned_count')] > 0
    for name, mask in [('stereo_unassigned', stereo[holdout]), ('no_unassigned', ~stereo[holdout])]:
        if mask.sum() and len(set(y[holdout][mask])) == 2:
            slice_rows.append({'slice': name, 'n': int(mask.sum()), **scores(y[holdout][mask], test_probability[mask], threshold)})
    pd.DataFrame(slice_rows).to_csv(DEST/'holdout_slices.csv', index=False)
    final_models, final_probabilities = [], []
    for name in selected_names:
        kind, estimator = candidate_map[name]
        print('Refitting all 8349 rows:', name, flush=True)
        model = clone(estimator).fit(matrices[kind], y)
        final_models.append({'name': name, 'kind': kind, 'model': model})
        final_probabilities.append(model.predict_proba(prediction_matrices[kind])[:, 1])
    probability = np.mean(final_probabilities, axis=0)
    pred = (probability >= threshold).astype(int)
    assert np.isfinite(probability).all() and np.all((0 <= probability) & (probability <= 1))
    output_path = DEST/'predict_output_모델예측.csv'
    safe_submission(template, pred, output_path)
    # ponytail: chunked exact Tanimoto scan for 927x8349; indexed search if data grows substantially.
    train_fp = X[:, :1024].astype(np.float32)
    query_fp = Q[:, :1024].astype(np.float32)
    nearest, similarity = [], []
    train_sums = train_fp.sum(axis=1)
    for start in range(0, len(query_fp), 100):
        batch = query_fp[start:start+100]
        intersection = batch @ train_fp.T
        union = batch.sum(axis=1)[:, None] + train_sums - intersection
        values = np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)
        indices = values.argmax(axis=1)
        nearest.extend(indices.tolist()); similarity.extend(values[np.arange(len(batch)), indices].tolist())
    diagnostics = pd.DataFrame({'SMILES': incoming.SMILES, 'probability_toxic_uncalibrated': probability,
                                'prediction_toxic': pred, 'submission_output_safe': 1-pred,
                                'distance_to_threshold': np.abs(probability-threshold),
                                'scaffold_seen_in_train': np.isin(qgroups, groups),
                                'connectivity_seen_in_train': np.isin(qconnectivity, connectivity),
                                'isomeric_seen_in_train': np.isin(qisomeric, isomeric),
                                'nearest_train_csv_row': np.array(nearest)+2,
                                'nearest_morgan_tanimoto': similarity})
    diagnostics.to_csv(DEST/'prediction_diagnostics.csv', index=False)
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'sklearn': sklearn.__version__, 'rdkit': rdBase.rdkitVersion,
                'lightgbm': __import__('lightgbm').__version__, 'joblib': joblib.__version__}
    bundle = {'models': final_models, 'columns': columns, 'eligible_column_indices': eligible_columns,
              'original_columns': original_columns, 'local_dictionary': local_dictionary,
              'threshold_toxic': threshold, 'label_mapping_internal': {'0': 'safe', '1': 'toxic'},
              'label_mapping_submission': {'0': 'toxic', '1': 'safe'}, 'versions': versions,
              'source_hashes': hashes}
    joblib.dump(bundle, DEST/'toxicity_model.joblib', compress=3)
    restored = joblib.load(DEST/'toxicity_model.joblib')
    restored_probability = np.mean([item['model'].predict_proba(prediction_matrices[item['kind']])[:, 1]
                                    for item in restored['models']], axis=0)
    assert np.allclose(restored_probability, probability, atol=1e-12)
    assert all(digest(ROOT/name) == value for name, value in hashes.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    RocCurveDisplay.from_predictions(y[holdout], test_probability, ax=axes[0], name='Toxic class')
    PrecisionRecallDisplay.from_predictions(y[holdout], test_probability, ax=axes[1], name='Toxic class')
    matrix = confusion_matrix(y[holdout], test_pred)
    axes[2].imshow(matrix, cmap='Blues'); axes[2].set(xticks=[0, 1], yticks=[0, 1],
                        xticklabels=['Safe', 'Toxic'], yticklabels=['Safe', 'Toxic'],
                        xlabel='Predicted', ylabel='Actual', title='Scaffold holdout confusion matrix')
    for i in range(2):
        for j in range(2): axes[2].text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=18)
    fig.tight_layout(); fig.savefig(DEST/'holdout_evaluation.png', dpi=160); plt.close(fig)
    summary = {'seed': SEED, 'train_rows': len(y), 'prediction_rows': len(incoming),
               'development_rows': len(dev), 'holdout_rows': len(holdout), 'scaffold_groups': len(set(groups)),
               'group_overlap': 0, 'cv_folds': 4, 'selection': lock,
               'holdout': final_scores, 'holdout_at_0_5': scores(y[holdout], test_probability),
               'holdout_confidence_interval': ci,
               'holdout_dummy': scores(y[holdout], np.full(len(holdout), y[dev].mean())),
               'prediction_output_counts_original_mapping': {str(k): int(v) for k, v in pd.Series(1-pred).value_counts().items()},
               'prediction_known_scaffolds': int(np.isin(qgroups, groups).sum()),
               'prediction_known_connectivity': int(np.isin(qconnectivity, connectivity).sum()),
               'prediction_known_isomeric': int(np.isin(qisomeric, isomeric).sum()),
               'median_nearest_tanimoto': float(np.median(similarity)),
               'versions': versions, 'protected_source_hashes': hashes,
               'source_files_unchanged': True, 'feature_recalculation_matches_all_training_rows': True,
               'model_reload_probability_matches': True, 'submission_columns': ['SMILES', 'output'],
               'submission_sha256': digest(output_path), 'elapsed_seconds': perf_counter()-started,
               'scope': 'Internal scaffold holdout. Prior full-label EDA; no external validation. Conflict-selected local5 excluded from final selection.'}
    write_json(DEST/'modeling_summary.json', summary)
    print('DONE', json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_ablation():
    """Post-selection development-only feature comparisons; do not reopen the holdout."""
    cache = np.load(DEST/'feature_cache.npz')
    split = pd.read_csv(DEST/'split_manifest.csv')
    dev = np.flatnonzero(split.split.to_numpy() == 'development')
    columns = [c for c in pd.read_csv(ROOT/'02_데이터/train_final_morgan_8vars.csv', nrows=0)
               if c not in ['SMILES', 'label']]
    eligible = [i for i, c in enumerate(columns) if not c.startswith('stereo_env_')]
    features = np.array(columns)[eligible]
    X = cache['X'][:, eligible]
    y = cache['y']
    matrices = {
        'original_ecfp_same_lightgbm': cache['legacy'],
        'morgan_without_global3': X[:, [i for i, c in enumerate(features) if c not in BASE]],
        'morgan_without_ptfp': X[:, [i for i, c in enumerate(features) if not c.startswith('ptfp_')]],
        'morgan_only_plus_properties_global3': X[:, [i for i, c in enumerate(features)
                                                 if c.startswith('ecfp_') or c in PROPS+BASE]],
        'fingerprints_without_properties_global3': X[:, [i for i, c in enumerate(features)
                                                       if c.startswith(('ecfp_', 'fcfp_', 'ptfp_'))]]}
    folds = []
    for fold in sorted(split.dev_fold.iloc[dev].unique()):
        valid = np.flatnonzero(split.dev_fold.iloc[dev].to_numpy() == fold)
        train = np.flatnonzero(split.dev_fold.iloc[dev].to_numpy() != fold)
        assert not set(cache['groups'][dev[train]]) & set(cache['groups'][dev[valid]])
        folds.append((train, valid))
    results, probabilities = [], {}
    reference = pd.read_csv(DEST/'model_comparison.csv').set_index('model').loc['lightgbm_31']
    for name, matrix in matrices.items():
        p = np.empty(len(dev))
        for fold, (train, valid) in enumerate(folds):
            model = LGBMClassifier(n_estimators=600, learning_rate=.035, num_leaves=31,
                                   min_child_samples=30, colsample_bytree=.8, subsample=.85,
                                   subsample_freq=1, reg_lambda=5, random_state=SEED, n_jobs=4,
                                   verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(matrix[dev[train]], y[dev[train]])
            p[valid] = model.predict_proba(matrix[dev[valid]])[:, 1]
            print(f'ablation {name} {fold+1}/4', flush=True)
        probabilities[name] = p
        row = {'experiment': name, 'feature_count': matrix.shape[1], **scores(y[dev], p)}
        row['auc_delta_vs_morgan_global3'] = row['roc_auc']-float(reference.roc_auc)
        results.append(row)
        pd.DataFrame(results).to_csv(DEST/'feature_ablation.csv', index=False)
    pd.DataFrame({'csv_row': dev+2, 'label_toxic': y[dev], **probabilities}).to_csv(DEST/'ablation_oof.csv', index=False)
    print('PASS: five matched feature comparisons, holdout not consulted', flush=True)


def predict(input_path, output_path):
    bundle = joblib.load(DEST/'toxicity_model.joblib')
    frame = read_frame(input_path)
    assert set(bundle['original_columns']+['SMILES']) <= set(frame.columns)
    assert not frame.isna().any().any()
    matrix, _, _, _ = transform(frame, bundle['columns'], bundle['local_dictionary'])
    matrices = {'proper': matrix[:, bundle['eligible_column_indices']],
                'legacy': frame[bundle['original_columns']].to_numpy(dtype=np.float32), 'full': matrix}
    probability = np.mean([item['model'].predict_proba(matrices[item['kind']])[:, 1]
                           for item in bundle['models']], axis=0)
    assert np.isfinite(probability).all()
    assert output_path.resolve() not in {input_path.resolve(), (ROOT/'01_원본자료/predict_output_홍길동.csv').resolve()}
    output = pd.DataFrame({'SMILES': frame.SMILES, 'output': 1-(probability >= bundle['threshold_toxic']).astype(int)})
    temporary = output_path.with_suffix('.csv.tmp')
    output.to_csv(temporary, index=False, encoding='utf-8-sig')
    assert pd.read_csv(temporary).equals(output)
    temporary.replace(output_path)
    print(f'Wrote {len(output)} rows: {output_path}', flush=True)


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        self_check()
    elif '--ablation' in sys.argv:
        run_ablation()
    elif '--predict' in sys.argv:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--predict', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args()
        predict(args.predict, args.output)
    else:
        main()
