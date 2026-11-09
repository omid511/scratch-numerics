"""One fixed modal-history tree per grouped development fold; no held predictions."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn
import experiment_raw_modal as R
import analyze_raw_modal as A

D, S, H = R.D, R.S, R.H
ROOT, SCRIPT = R.ROOT, R.SCRIPT
OUT = SCRIPT / 'modal_tree_20261102'
GROUPED = SCRIPT / 'raw_tcn_grouped4_20261010' / 'protocol.json'
COMPARATOR = R.OUT / 'results.json'
ARM = 'modal_tree'
CONFIG = dict(loss='squared_error', max_iter=200, learning_rate=.05, max_leaf_nodes=15,
              min_samples_leaf=100, l2_regularization=1, early_stopping=False, random_state=42)
PROGRESS = 'ae7d58419fd25e3013568aaeb9f42e9fa2946fd6e7fe8af7ccd074cc835db69f'


def sha(path):
    return D._sha256(Path(path))


def sources(grouped):
    names = ('support_modal_20261009_v2/dataset.npz', 'support_modal_20261009_v2/protocol.json',
             'raw_modal_data.py', 'experiment_raw_modal.py', 'score_additive_spectral.py',
             'experiment_multivel_history.py', 'experiment_support_modal_v2.py',
             'diagnostic_matrix.py', 'analyze_gate_bypass_receipt.py',
             'analyze_raw_modal.py', 'raw_modal_20261010/results.json',
             'raw_modal_20261010/protocol.json')
    paths = [SCRIPT / name for name in names] + [ROOT / 'src/mechanics/p4_margin_estimation/baselines.py']
    hashes = {}
    for path in paths:
        name = str(path.relative_to(ROOT))
        actual = sha(path)
        if grouped['inputs'].get(name) != actual:
            raise RuntimeError(f'Frozen input hash mismatch: {name}')
        hashes[name] = actual
    for path in (GROUPED, Path(__file__).resolve(), ROOT / 'P4_PROGRESS.json'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    if hashes['P4_PROGRESS.json'] != PROGRESS:
        raise RuntimeError('Protected progress ledger changed')
    return hashes


def protocol():
    grouped = json.loads(GROUPED.read_text())
    hashes = sources(grouped)
    original = json.loads((R.OUT / 'protocol.json').read_text())
    for name in ('split', 'additional_designs', 'evaluation_folds'):
        if grouped[name] != original[name]:
            raise RuntimeError(f'Grouped and original protocols disagree: {name}')
    folds = grouped['evaluation_folds']
    split = grouped['split']
    groups = [set(split[k]) for k in ('train', 'validation', 'held')] + [set(grouped['additional_designs'])]
    if (len(folds) != 3 or any(len(f) != 4 for f in folds)
            or len(set(sum(folds, []))) != 12 or set(sum(folds, [])) != groups[3]
            or any(a & b for i, a in enumerate(groups) for b in groups[i+1:])):
        raise RuntimeError('Invalid design partition')
    frozen = dict(scope='Modal-only limited-input development competitor; not matched-input waveform attribution',
        hypothesis='Fixed nonlinear conversion of causal modal history may reduce signed-headroom error',
        inputs=hashes, split=split, additional_designs=grouped['additional_designs'],
        evaluation_folds=folds, primary_noise=grouped['primary_noise'],
        independence_unit='design', estimator=CONFIG, sklearn_version=sklearn.__version__,
        inputs_contract='174 measured modal features, observation flag and age, lags 4/2/0, 528 flattened values; no hidden physical inputs',
        training='One deterministic fit per fold, all 124 frames of proper clean and iid1 rows only; no validation selection',
        targets='Equal-design all/boundary MAE <= .010/.005 clean and <= .015/.010 each 1% view',
        safeguards='Boundary optimism >.02 <=5%; each eligible current sign side <=10%; retreat no regression and farther-stable MAE delta <=.005 versus frozen references',
        continuation='Only consider as development reference if point targets and dangerous-error, retreat and farther-stable safeguards hold; otherwise partial/negative without tuning',
        limits='One seed versus three-seed comparators; no search, promotion, operational or safety claim. Uncertainty, warning, blind 120 designs and independent physics remain required.')
    path = OUT / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise RuntimeError('Existing experiment protocol or source hashes differ')
    else:
        if OUT.exists() and any(OUT.iterdir()):
            raise RuntimeError('Output directory exists without frozen protocol')
        OUT.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            json.dump(frozen, handle, indent=2, allow_nan=False)
            handle.write('\n')
    return frozen


def rows_for_fold(data, p, fold):
    records = data['records']
    fit_ids = set(p['split']['train']) | (set(p['additional_designs']) - set(p['evaluation_folds'][fold]))
    forbidden = set(p['evaluation_folds'][fold]) | set(p['split']['validation']) | set(p['split']['held'])
    if fit_ids & forbidden:
        raise RuntimeError('Fitting and excluded design populations overlap')
    proper = np.array([r['design'] in fit_ids and r['noise'] in ('clean', 'iid1') for r in records])
    if {r['design'] for r, keep in zip(records, proper) if keep} != fit_ids:
        raise RuntimeError('Missing proper fitting design')
    held = np.array([r['design'] in p['split']['held'] for r in records])
    if np.any(proper & held) or not held.any():
        raise RuntimeError('Held exclusion failure')
    return proper, held


def fit(fold, data, p):
    paths = [OUT / f'fold{fold}_{suffix}' for suffix in ('model.pkl', 'predictions.npz', 'complete.json')]
    if any(path.exists() for path in paths):
        raise RuntimeError(f'Fold {fold} has existing or incomplete artifacts; refusing overwrite')
    started = time.perf_counter()
    proper, held = rows_for_fold(data, p, fold)
    if data['truth'].shape[1] != 124 or not np.isfinite(data['truth'][proper]).all():
        raise RuntimeError('Invalid proper labels or native frame count')
    modal, (mean, scale, active) = D.modal_inputs(data, proper)
    if modal.shape != (*data['truth'].shape, 3, 176) or not np.isfinite(modal[~held]).all():
        raise RuntimeError('Invalid modal slots')
    # Check the source's fold-local scale and every causal history slot without labels.
    expected_mean, expected_scale, expected_active = D.fit_ridge_scaling(
        data['features'][proper][data['observed'][proper]])
    if not (np.array_equal(mean, expected_mean) and np.array_equal(scale, expected_scale)
            and np.array_equal(active, expected_active)):
        raise RuntimeError('Fold-local scaling changed')
    current = (data['features'][~held] - mean) / scale
    current[..., ~active] = 0
    current[~data['observed'][~held]] = 0
    current = np.concatenate((current, data['observed'][~held, :, None], data['age'][~held, :, None]), axis=-1).astype(np.float32)
    for slot, lag in enumerate(D.LAGS):
        if lag and np.any(modal[~held, :lag, slot] != 0):
            raise RuntimeError('History warmup is not zero')
        if not np.array_equal(modal[~held, lag:, slot], current[:, :124-lag]):
            raise RuntimeError('Causal modal history mismatch')
    del current
    x = modal.reshape(len(modal), 124, 528)
    model = HistGradientBoostingRegressor(**CONFIG)
    model.fit(x[proper].reshape(-1, 528), data['truth'][proper].reshape(-1))
    point = np.full(data['truth'].shape, np.nan, dtype=np.float64)
    for row in np.flatnonzero(~held):
        point[row] = model.predict(x[row])
    if not np.isfinite(point[~held]).all() or not np.isnan(point[held]).all():
        raise RuntimeError('Prediction scope or finiteness failure')
    with paths[0].open('xb') as handle:
        pickle.dump(dict(model=model, mean=mean, scale=scale, active=active), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with paths[1].open('xb') as handle:
        np.savez_compressed(handle, predictions=point, proper=proper, held=held,
                            protocol_sha256=np.array(sha(OUT / 'protocol.json')))
    receipt = dict(fold=fold, seed=42, protocol_sha256=sha(OUT / 'protocol.json'),
        model_sha256=sha(paths[0]), predictions_sha256=sha(paths[1]),
        proper_rows=int(proper.sum()), training_decisions=int(proper.sum()) * 124,
        elapsed_seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_scope='process high-water mark; Linux KiB',
        checks=['proper finite', 'fit-local scaling', 'design exclusion', 'causal slots', 'held prediction exclusion'])
    with paths[2].open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return receipt


def completed(fold, data, p):
    model = OUT / f'fold{fold}_model.pkl'
    prediction = OUT / f'fold{fold}_predictions.npz'
    receipt = json.loads((OUT / f'fold{fold}_complete.json').read_text())
    if (receipt['fold'] != fold or receipt['seed'] != 42
            or receipt['protocol_sha256'] != sha(OUT / 'protocol.json')
            or receipt['model_sha256'] != sha(model) or receipt['predictions_sha256'] != sha(prediction)):
        raise RuntimeError('Completed fold artifact provenance mismatch')
    proper, held = rows_for_fold(data, p, fold)
    with np.load(prediction, allow_pickle=False) as z:
        point = z['predictions'].copy()
        if (str(z['protocol_sha256']) != receipt['protocol_sha256']
                or not np.array_equal(z['proper'], proper) or not np.array_equal(z['held'], held)
                or point.shape != data['truth'].shape or not np.isnan(point[held]).all()
                or not np.isfinite(point[~held]).all()):
            raise RuntimeError('Completed fold scope mismatch')
    return point, receipt


def score(data, p, points, receipts):
    if (OUT / 'results.json').exists() or (OUT / 'comparison_summary.json').exists():
        raise RuntimeError('Completed scoring outputs already exist')
    records = data['records']
    designs = np.array([r['design'] for r in records])
    truth = data['truth']
    strata = H.strata(data)
    strata['farther_stable15'] = truth >= .15
    strata['profile_retreat'] = np.array([r['profile'] == 'retreat' for r in records])[:, None] & strata['all']
    metrics = json.loads(COMPARATOR.read_text())['metrics']
    names = ('all', 'boundary', 'current', 'boundary_current', 'profile_retreat', 'farther_stable', 'farther_stable15')
    for fold, ids in enumerate(p['evaluation_folds']):
        for noise in sorted({r['noise'] for r in records if r['design'] in ids}):
            key = f'fold{fold}/evaluation/{noise}'
            pop = np.array([r['design'] in ids and r['noise'] == noise for r in records])[:, None]
            metrics[key][ARM] = {name: S.summary(points[fold][None], truth, pop & strata[name], designs) for name in names}
            for control in A.ARMS:
                for name in names:
                    tree = metrics[key][ARM][name]['per_design']
                    other = metrics[key][control][name]['per_design']
                    if set(tree) != set(other) or any(tree[d]['frames'] != other[d]['frames'] for d in tree):
                        raise RuntimeError(f'Comparator denominator mismatch: {key}/{control}/{name}')
    primary = {}
    for noise in p['primary_noise']:
        by_arm = {}
        for arm in (ARM, *A.ARMS):
            by_arm[arm] = dict(slices={name: A.mae(A.joined(metrics, noise, arm, name)) for name in names},
                optimism=A.event(metrics, noise, arm, 'boundary', 'optimism_over02'),
                stable_wrong_sign=A.event(metrics, noise, arm, 'current', 'stable_wrong_sign'),
                unstable_wrong_sign=A.event(metrics, noise, arm, 'current', 'unstable_exclusion_wrong_sign'))
        paired = {}
        for name in names:
            tree = A.joined(metrics, noise, ARM, name)
            paired[name] = {}
            for control in A.ARMS:
                other = A.joined(metrics, noise, control, name)
                if set(tree) != set(other):
                    raise RuntimeError('Paired design populations differ')
                differences = {d: (np.asarray(tree[d]['mae_by_seed']) - np.asarray(other[d]['mae_by_seed'])).tolist() for d in sorted(tree)}
                paired[name][control] = dict(design_seed_delta=differences,
                    mean=float(np.mean(list(differences.values()))) if differences else None)
        primary[noise] = dict(arms=by_arm, tree_minus_comparators=paired,
            targets=dict(all=.010 if noise == 'clean' else .015, boundary=.005 if noise == 'clean' else .010))
    if sha(ROOT / 'P4_PROGRESS.json') != PROGRESS:
        raise RuntimeError('Protected progress ledger changed')
    R.save_json(OUT / 'results.json', dict(protocol_sha256=sha(OUT / 'protocol.json'),
        fit_receipts=receipts, metrics=metrics, held_predictions_absent=True))
    R.save_json(OUT / 'comparison_summary.json', dict(protocol_sha256=sha(OUT / 'protocol.json'),
        primary=primary, limits=p['limits'], continuation=p['continuation']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'run'))
    args = parser.parse_args()
    p = protocol()
    data = D.load_data()
    if args.phase == 'preflight':
        if (OUT / 'fold0_complete.json').exists():
            _, receipt = completed(0, data, p)
        else:
            receipt = fit(0, data, p)
        if sha(ROOT / 'P4_PROGRESS.json') != PROGRESS:
            raise RuntimeError('Protected progress ledger changed')
        print(json.dumps(receipt, allow_nan=False), flush=True)
        return
    if (OUT / 'results.json').exists() or (OUT / 'comparison_summary.json').exists():
        raise RuntimeError('Completed scoring outputs already exist')
    points, receipts = [], []
    for fold in range(3):
        if fold and not (OUT / f'fold{fold}_complete.json').exists():
            fit(fold, data, p)
        point, receipt = completed(fold, data, p)
        points.append(point)
        receipts.append(receipt)
    score(data, p, points, receipts)
    print('MODAL_TREE_SCORED', flush=True)


if __name__ == '__main__':
    main()
