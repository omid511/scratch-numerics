#!/usr/bin/env python3
"""Training-design-only P4 boosting/MLP comparison on cached symmetric features.

Run with the p4-tabular extra installed. No held-out dataset scoring.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import warnings
from types import SimpleNamespace
from scipy.optimize import minimize, check_grad

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from experiment_p4_ridge_cv import design_folds, per_design_mae
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from mechanics.p4_margin_estimation.skip_mlp import objective_setup
from mechanics.p4_margin_estimation.sidecar import extract
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize


def model(kind, setting):
    if kind == 'boosting':
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=.05, max_leaf_nodes=setting,
            min_samples_leaf=100, l2_regularization=1.,
            early_stopping=False, random_state=42)
    return MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu',
                        solver='lbfgs', alpha=setting, max_iter=2000,
                        max_fun=50000, tol=1e-7, random_state=42,
                        early_stopping=False)


def fit(kind, setting, x, y):
    if kind == 'mlp':
        predict, metadata = fit_network(x, y, max_iter=2000, max_fun=50000, alpha=setting)
        for message in metadata['warnings']:
            print(kind, 'fit warning:', message, flush=True)
        return predict, metadata['warnings']
    mean, scale, active = fit_ridge_scaling(x)
    # Both estimators see the same training-supported columns; MLP needs scaling.
    def transform(q):
        q = (q - mean) / scale
        q[:, ~active] = 0.
        return q
    estimator = model(kind, setting)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(transform(x), y)
    messages = [str(w.message) for w in caught]
    for message in messages:
        print(kind, 'fit warning:', message, flush=True)
    return lambda q: estimator.predict(transform(q)), messages


def score(y, pred, ids):
    errors = pred - y
    dm = per_design_mae(y, pred[:, None], ids)[:, 0]
    near = abs(y) < .15
    return {'design_mae': float(dm.mean()),
            'per_design_mae': dict(zip(np.unique(ids).tolist(), dm.tolist())),
            'p95_signed_error': float(np.quantile(errors, .95)),
            'p99_signed_error': float(np.quantile(errors, .99)),
            'near_clip_mae': float(abs(errors[near]).mean()),
            'predictions': pred.tolist()}


def fit_network(x, y, hidden=(32, 16), seed=42, alpha=1., custom=False, skip=False,
                max_iter=10000, max_fun=200000, *,
                consistency_pairs=None, consistency_weight=0.,
                consistency_pair_weights=None, sample_weight=None):
    """Train-fold scaling; optional custom paired penalty or native sample weighting."""
    if custom and sample_weight is not None:
        raise ValueError("sample weights require the native fitting path")
    if not custom and (consistency_pairs is not None or consistency_weight != 0
                       or consistency_pair_weights is not None):
        raise ValueError("paired consistency requires custom=True")
    mean, scale, active = fit_ridge_scaling(x)
    def transform(q):
        z = (q - mean) / scale
        z[:, ~active] = 0
        return z
    z = transform(x)
    start = time.perf_counter()
    if custom:
        initial, objective, forward = objective_setup(
            z, y, hidden, seed, alpha, skip,
            consistency_pairs=consistency_pairs, consistency_weight=consistency_weight,
            consistency_pair_weights=consistency_pair_weights)
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'maxfun': max_fun, 'gtol': 1e-7})
        metadata = {'success': bool(result.success), 'message': str(result.message),
                    'iterations': int(result.nit), 'loss': float(result.fun)}
        predict = lambda q: forward(result.x, transform(q))
    else:
        estimator = MLPRegressor(hidden_layer_sizes=hidden, activation='relu', solver='lbfgs',
                                alpha=alpha, max_iter=max_iter, max_fun=max_fun,
                                tol=1e-7, random_state=seed, early_stopping=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            estimator.fit(z, y, sample_weight=sample_weight)
        metadata = {'success': not bool(caught), 'warnings': [str(w.message) for w in caught],
                    'iterations': int(estimator.n_iter_), 'loss': float(estimator.loss_)}
        predict = lambda q: estimator.predict(transform(q))
    metadata['seconds'] = time.perf_counter() - start
    return predict, metadata


def validate_mlp(output):
    out = ROOT / output
    out.mkdir(exist_ok=False)
    with np.load(ROOT / 'p4_groups/robustness_symmetric_spacing_20260929/features.npz') as z:
        g, j, y, ids, rows = z['features'], z['independent_features'], z['y'], z['design_ids'], z['source_indices']
    n = len(y)
    old = json.loads((ROOT / 'p4_groups/nonlinear_tabular_20260929/results.json').read_text())['models']['mlp']
    payload = {'protocol': {'hidden_baseline': [32,16], 'larger': [64,32],
        'skip': 'Jointly trained, L2-regularized input-to-output linear path initialized to zero; matched custom plain control.',
        'validation_seeds': [42,43,44], 'ablation_seed': 42, 'alpha': 1.,
        'folds': 'Same 5 outer design folds seed42, equal clean/1%/5% mixture; all design copies grouped. No reserved designs.',
        'regularization': 'Fixed at prior unanimously selected alpha1 for ablations. Not retuned for new arms.',
        'convergence_diagnostic': 'Outer fold0 three inner folds, both original alpha candidates, maxiter10000 versus saved maxiter2000 aggregate scores.',
        'features': ['zcr','xcorr','kurt'], 'independent_noise': 'seed929, no refits',
        'limitations': 'Exploratory training-design comparisons; ablations one initialization; larger iteration budget is not a convergence guarantee.'}, 'diagnostic': {}, 'arms': {}}
    def save():
        (out/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    # Objective checks exercise both skip and non-skip gradients.
    rng = np.random.default_rng(71)
    xx, yy = rng.normal(size=(20,3)), rng.normal(size=20)
    for skip in (False, True):
        initial, fun, forward = objective_setup(xx, yy, (4,3), 42, 1., skip)
        error = check_grad(lambda t: fun(t)[0], lambda t: fun(t)[1], initial)
        assert error < 1e-5, error
    p1, _ = fit_network(xx, yy, hidden=(4,3))
    p2, _ = fit_network(xx, yy, hidden=(4,3), custom=True)
    np.testing.assert_allclose(p1(xx), p2(xx), atol=1e-6, rtol=1e-6)
    payload['objective_checks'] = 'Finite-difference gradients for plain/skip; plain predictions agree with sklearn within1e-6.'
    outer0 = design_folds(ids, 5, 42)[0]
    train0 = np.setdiff1d(np.arange(n), outer0)
    for alpha in (.01, 1.):
        errors, fits = [], []
        for ii, local in enumerate(design_folds(ids[train0], 3, 142)):
            val = train0[local]; train = np.setdiff1d(train0, val)
            predict, info = fit_network(g[:,train].reshape(-1,41), np.tile(y[train],3), alpha=alpha)
            pred = predict(g[:,val].reshape(-1,41))
            errors.extend(per_design_mae(np.tile(y[val],3), pred[:,None], np.tile(ids[val],3))[:,0].tolist())
            fits.append(info)
            print('diagnostic',alpha,ii,info,flush=True)
        payload['diagnostic'][str(alpha)] = {'inner_design_mae':float(np.mean(errors)), 'fits':fits}
        save()
    payload['diagnostic']['prior_fold0_inner_scores'] = old['folds'][0]['inner_scores']
    # Recover candidate features on the same acquired signals and noise realizations.
    extras = np.empty((5,n,3)); dts=np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz')['dts']
    layout=np.array([5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23])
    rowmap={int(row):k for k,row in enumerate(rows)}
    for di,did in enumerate(np.unique(ids)):
        with np.load(ROOT/'p4_sensor_candidates'/(str(did)+'.npz')) as z:
            raw, source, scales=z['raw'],z['source_indices'],z['noise_scale']
        for k,index in enumerate(source):
            for seed, levels, offset in ((928,(0,.01,.05),0),(929,(.01,.05),3)):
                white=np.random.default_rng(np.random.SeedSequence([seed,int(index)])).normal(size=raw[k].shape)
                for ni,level in enumerate(levels):
                    signal=np.clip(causal_calibration_normalize(raw[k]+level*scales[k]*white,calibration_samples=51,normalize_mode='per_channel'),-50,50).astype(np.float32)[layout]
                    clip=SimpleNamespace(sensor_signals=signal,dt=float(dts[index]))
                    extras[offset+ni,rowmap[int(index)]]=extract([clip],['zcr','xcorr','kurt'])[0]
        if (di+1)%20==0: print('features',di+1,flush=True)
    assert np.isfinite(extras).all()
    np.savez_compressed(out/'extra_features.npz',features=extras,source_indices=rows,design_ids=ids,names=np.array(['zcr','xcorr','kurt']))
    configs=[('baseline_s'+str(s),(32,16),s,False,False,[]) for s in (42,43,44)]
    configs += [('larger',(64,32),42,False,False,[]),('plain_control',(32,16),42,True,False,[]),('skip',(32,16),42,True,True,[])]
    configs += [(name,(32,16),42,False,False,cols) for name,cols in [('plus_zcr',[0]),('plus_xcorr',[1]),('plus_kurt',[2]),('plus_all',[0,1,2])]]
    for name,hidden,seed,custom,skip,columns in configs:
        gg = np.concatenate([g,extras[:3,:,columns]],axis=2) if columns else g
        jj = np.concatenate([j,extras[3:,:,columns]],axis=2) if columns else j
        preds=np.full((3,n),np.nan); ipreds=np.full((2,n),np.nan); folds=[]
        for fold,test in enumerate(design_folds(ids,5,42)):
            train=np.setdiff1d(np.arange(n),test)
            assert not set(ids[train]) & set(ids[test])
            predict,info=fit_network(gg[:,train].reshape(-1,gg.shape[2]),np.tile(y[train],3),hidden,seed,custom=custom,skip=skip)
            preds[:,test]=predict(gg[:,test].reshape(-1,gg.shape[2])).reshape(3,len(test))
            ipreds[:,test]=predict(jj[:,test].reshape(-1,jj.shape[2])).reshape(2,len(test))
            info.update({'fold':fold,'train_designs':np.unique(ids[train]).tolist(),'validation_designs':np.unique(ids[test]).tolist()}); folds.append(info)
            print(name,fold,info['iterations'],info['success'],flush=True)
        assert np.isfinite(preds).all() and np.isfinite(ipreds).all()
        records={}
        for prefix,pp,levels in [('original',preds,(0,.01,.05)),('independent',ipreds,(.01,.05))]:
            for ni,level in enumerate(levels):
                key=f'{prefix}_noise{level:g}'; r=score(y,pp[ni],ids)
                reference=payload['arms'].get('baseline_s42',{}).get('records',old['records'])[key]
                r['paired_vs_baseline']=paired_design_comparison(r['per_design_mae'].items(),reference['per_design_mae'].items(),n_bootstrap=10000,seed=42)
                records[key]=r
                print(name,key,r['design_mae'],r['p99_signed_error'],flush=True)
        payload['arms'][name]={'hidden':hidden,'seed':seed,'custom':custom,'skip':skip,'added_columns':columns,'folds':folds,'records':records}
        save()
    print('Saved',out,flush=True)
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='p4_groups/nonlinear_tabular_20260929')
    ap.add_argument('--mlp-validation', action='store_true',
                    help='Run seed, convergence, capacity, skip and sidecar ablations.')
    args = ap.parse_args()
    if args.mlp_validation:
        validate_mlp(args.output)
        return
    out = ROOT / args.output
    out.mkdir(exist_ok=False)
    source = ROOT / 'p4_groups/robustness_symmetric_spacing_20260929/features.npz'
    with np.load(source) as z:
        g, j, y, ids = z['features'], z['independent_features'], z['y'], z['design_ids']
        names = z['original_feature_names'].tolist()
    names[37] = 'symmetric_lead_neighbor_frequency_gap'
    names[39] = 'symmetric_lead_neighbor_pole_distance'
    baseline_path = ROOT / 'p4_groups/robustness_symmetric_mixed_20260929/results.json'
    baseline = json.loads(baseline_path.read_text())
    n = len(y)
    assert np.isfinite(g).all() and np.isfinite(j).all()
    families = {'ridge7': list(range(7)), 'modal': list(range(7, 22)),
                'damping': list(range(22, 27)), 'stability': list(range(27, 37)),
                'spacing': list(range(37, 41))}
    importance_groups = {name: [i] for i, name in enumerate(names)}
    importance_groups.update({'family:' + k: v for k, v in families.items()})
    grids = {'boosting': [7, 15], 'mlp': [.01, 1.]}
    payload = {'protocol': {'sklearn': sklearn.__version__, 'features': names,
        'grids': grids, 'mlp_hidden_layers': [32, 16],
        'training': 'Equal clean/1%/5% mixture; all copies of designs grouped; seed42, 5 outer/3 inner folds; select equal-noise design MAE.',
        'importance': 'Outer-validation permutation MAE increase, 3 repeats per column/family per noise. Same clip permutation across noise levels. Correlated features can mask importance; not causal.',
        'independent_noise': 'Seed929, no retuning; training seed928.',
        'limitations': 'Exploratory training-design CV, single initialization seed, no test scoring. 5% noise is an uncalibrated stress case.',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, 'models': {}}
    for kind, grid in grids.items():
        pred = np.full((3, n), np.nan)
        independent = np.full((2, n), np.nan)
        folds, importances, latency = [], [], []
        started = time.perf_counter()
        for fold, test in enumerate(design_folds(ids, 5, 42)):
            train = np.setdiff1d(np.arange(n), test)
            assert not set(ids[train]) & set(ids[test])
            curves, fit_warnings = [], []
            for setting in grid:
                inner_scores = []
                for val_local in design_folds(ids[train], 3, 142 + fold):
                    val = train[val_local]
                    fit_rows = np.setdiff1d(train, val)
                    assert not set(ids[fit_rows]) & set(ids[val])
                    predict, messages = fit(kind, setting, g[:, fit_rows].reshape(-1, g.shape[2]), np.tile(y[fit_rows], 3))
                    fit_warnings.extend(messages)
                    pp = predict(g[:, val].reshape(-1, g.shape[2]))
                    inner_scores.extend(per_design_mae(np.tile(y[val], 3), pp[:, None], np.tile(ids[val], 3))[:, 0].tolist())
                curves.append(float(np.mean(inner_scores)))
            best = int(np.argmin(curves))
            predict, messages = fit(kind, grid[best], g[:, train].reshape(-1, g.shape[2]), np.tile(y[train], 3))
            fit_warnings.extend(messages)
            pred[:, test] = predict(g[:, test].reshape(-1, g.shape[2])).reshape(3, len(test))
            independent[:, test] = predict(j[:, test].reshape(-1, j.shape[2])).reshape(2, len(test))
            for _ in range(5):
                predict(g[0, test[:1]])
            for index in test[:30]:
                tick = time.perf_counter_ns()
                predict(g[0, index:index+1])
                latency.append((time.perf_counter_ns() - tick) / 1e6)
            if kind == 'boosting':
                for label, columns in importance_groups.items():
                    for repeat in range(3):
                        perm = np.random.default_rng(1000 + fold * 10 + repeat).permutation(len(test))
                        for ni in range(3):
                            q = g[ni, test].copy()
                            q[:, columns] = g[ni, test[perm]][:, columns]
                            delta = abs(predict(q) - y[test]) - abs(pred[ni, test] - y[test])
                            by_design = {str(d): float(delta[ids[test] == d].mean()) for d in np.unique(ids[test])}
                            importances.append({'feature': label, 'noise_index': ni, 'repeat': repeat, 'fold': fold, 'design_deltas': by_design})
            folds.append({'fold': fold, 'train_designs': np.unique(ids[train]).tolist(), 'validation_designs': np.unique(ids[test]).tolist(), 'inner_scores': curves, 'selected': grid[best], 'warnings': fit_warnings})
            print(kind, 'fold', fold, 'setting', grid[best], 'inner', curves, flush=True)
        assert np.isfinite(pred).all() and np.isfinite(independent).all()
        records = {}
        for prefix, predictions, levels, refs in [('original', pred, [0., .01, .05], baseline['condition_scores']), ('independent', independent, [.01, .05], baseline['independent_noise_scores'])]:
            for ni, level in enumerate(levels):
                r = score(y, predictions[ni], ids)
                b = refs[f'noise{level:g}']
                r['ridge_design_mae'] = b['design_mae']
                r['paired_vs_ridge'] = paired_design_comparison(r['per_design_mae'].items(), b['per_design_mae'].items(), n_bootstrap=10000, seed=42)
                records[f'{prefix}_noise{level:g}'] = r
                print(kind, prefix, level, 'MAE', r['design_mae'], 'p99', r['p99_signed_error'], flush=True)
        payload['models'][kind] = {'folds': folds, 'records': records, 'permutation_importance': importances, 'fit_seconds': time.perf_counter()-started, 'latency': {'median_ms': float(np.median(latency)), 'p95_ms': float(np.quantile(latency, .95)), 'n': len(latency), 'scope': 'one-window scaling and prediction only; excludes extraction/acquisition'}}
        (out/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print('Saved', out, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
