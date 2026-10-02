"""TRAIN-only joint physical prior and soft frequency-growth experiment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
spec = importlib.util.spec_from_file_location('physical_frozen_contract', ROOT / 'p4_groups/boundary_noise_consistency_20260930/experiment_helpers.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
np, cloudpickle = helpers.np, helpers.cloudpickle
save_json, save_npz, sha256 = helpers.save_json, helpers.save_npz, helpers.sha256
from scipy.special import logsumexp, ndtr
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from experiment_p4_tabular import fit_network

ARMS = ['growth_only', 'joint_naive', 'joint_soft']
REFERENCES = ['one107', 'full174', 'blend50']
SEEDS = [42, 43, 44]
BUDGETS = [.01, .05, .10, .20]
FIELDS = ['point', 'lower', 'upper', 'p_unsafe', 'alpha', 'alpha_sd']


def mixture_output(weights, means, variances):
    """Exact central normal-mixture quantiles, not a moment-normal interval."""
    scale = np.sqrt(np.maximum(variances, 1e-14))
    low, high = np.min(means - 12 * scale, axis=1), np.max(means + 12 * scale, axis=1)
    quantiles = []
    for probability in (.05, .95):
        left, right = low.copy(), high.copy()
        for _ in range(40):
            middle = .5 * (left + right)
            cdf = np.sum(weights * ndtr((middle[:, None] - means) / scale), axis=1)
            left = np.where(cdf < probability, middle, left)
            right = np.where(cdf >= probability, middle, right)
        quantiles.append(.5 * (left + right))
    return np.column_stack([np.sum(weights * means, axis=1), *quantiles,
                            np.sum(weights * ndtr(-means / scale), axis=1)])


class PhysicalPrior:
    def fit(self, truth, observed, seed):
        self.mean = truth.mean(axis=0)
        self.scale = truth.std(axis=0)
        if not np.all(self.scale > 1e-8):
            raise ValueError('physical prior requires varying modal quantities and margin')
        z = (truth - self.mean) / self.scale
        self.prior = GaussianMixture(8, covariance_type='full', reg_covar=1e-4,
                                    max_iter=1000, n_init=1, random_state=seed).fit(z)
        errors = (observed - truth[None, :, :2]).reshape(-1, 2) / self.scale[:2]
        self.error = GaussianMixture(2, covariance_type='full', reg_covar=1e-4,
                                    max_iter=1000, n_init=1, random_state=seed).fit(errors)
        if not self.prior.converged_ or not self.error.converged_:
            raise RuntimeError('nonconverged physical/error mixture')
        return {'prior_iterations': int(self.prior.n_iter_), 'error_iterations': int(self.error.n_iter_),
                'prior_converged': bool(self.prior.converged_), 'error_converged': bool(self.error.converged_),
                'error_weights': self.error.weights_.tolist(),
                'error_means_physical': (self.error.means_ * self.scale[:2]).tolist(),
                'error_std_physical': (np.sqrt(np.diagonal(self.error.covariances_, axis1=1, axis2=2)) * self.scale[:2]).tolist()}

    def components(self, query, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        dimensions = np.array([0] if arm == 'growth_only' else [0, 1])
        q = ((query - self.mean[:2]) / self.scale[:2])[:, dimensions]
        if not np.isfinite(q).all():
            raise ValueError('expected finite measured pole; missing-mode policy must be explicit')
        if arm == 'joint_naive':
            noise_weights, noise_mean, noise_covariance = np.ones(1), np.zeros((1, 2)), np.zeros((1, 2, 2))
        else:
            noise_weights, noise_mean, noise_covariance = self.error.weights_, self.error.means_, self.error.covariances_
        logs, conditional_means, conditional_variances = [], [], []
        for weight, mean, covariance in zip(self.prior.weights_, self.prior.means_, self.prior.covariances_):
            cross = covariance[:, dimensions]
            for nw, nm, nc in zip(noise_weights, noise_mean, noise_covariance):
                observed_covariance = covariance[np.ix_(dimensions, dimensions)] + nc[np.ix_(dimensions, dimensions)]
                sign, logdet = np.linalg.slogdet(observed_covariance)
                if sign <= 0:
                    raise FloatingPointError('nonpositive observation covariance')
                gain = np.linalg.solve(observed_covariance, cross.T).T
                difference = q - mean[dimensions] - nm[dimensions]
                solved = np.linalg.solve(observed_covariance, difference.T).T
                logs.append(np.log(weight * nw) - .5 * (len(dimensions) * np.log(2 * np.pi) + logdet + np.sum(difference * solved, axis=1)))
                conditional_means.append((mean + difference @ gain.T) * self.scale + self.mean)
                conditional_variances.append(np.maximum(np.diag(covariance - gain @ cross.T), 0) * self.scale ** 2)
        log_weights = np.column_stack(logs)
        weights = np.exp(log_weights - logsumexp(log_weights, axis=1)[:, None])
        return weights, np.stack(conditional_means, axis=1), np.asarray(conditional_variances)

    def predict(self, query, arm):
        weights, means, variances = self.components(query, arm)
        margin = mixture_output(weights, means[:, :, 2], np.broadcast_to(variances[:, 2], weights.shape))
        alpha = np.sum(weights * means[:, :, 0], axis=1)
        alpha_variance = np.sum(weights * (variances[:, 0] + (means[:, :, 0] - alpha[:, None]) ** 2), axis=1)
        out = np.column_stack([margin, alpha, np.sqrt(np.maximum(alpha_variance, 0))])
        if not np.isfinite(out).all() or not np.all(out[:, 1] <= out[:, 2]):
            raise FloatingPointError('invalid posterior output')
        return out


def load_data():
    train, evaluation, y, ids, rows, names, folds, layout, pairs = helpers.load_data()
    fresh = ROOT / 'p4_groups/boundary_noise_consistency_20260930/fresh_features.npz'
    with np.load(fresh) as z:
        for key, value in [('y', y), ('ids', ids), ('source_indices', rows), ('row_fold', folds)]:
            np.testing.assert_array_equal(z[key], value)
        evaluation = np.concatenate([evaluation, z['features']])
        names += z['conditions'].tolist()
    metadata = json.loads((ROOT / 'p4_dataset_saturation/metadata.json').read_text())
    assert set(ids) == set(metadata['train_designs'])
    with np.load(ROOT / 'p4_dataset_saturation/metadata_arrays.npz') as z:
        np.testing.assert_array_equal(z['margins'][rows], y)
        np.testing.assert_array_equal(z['design_ids'][rows], ids)
        dt, clamps = z['dts'][rows], z['clamp_fracs'][rows]
        expected = np.column_stack([z['label_alphas'][rows], z['label_omegas'][rows] / (2 * np.pi)])
    truth = np.empty((len(y), 3))
    raw_hashes = {}
    row_map = {int(row): i for i, row in enumerate(rows)}
    for did in np.unique(ids):
        path = ROOT / 'p4_sensor_candidates' / f'{did}.npz'
        with np.load(path) as z:
            source, poles = z['source_indices'], z['poles']
        target = np.array([row_map[int(row)] for row in source])
        assert set(target) == set(np.flatnonzero(ids == did))
        truth[target, :2] = np.column_stack([poles[:, 0].real, poles[:, 0].imag / (2 * np.pi)])
        raw_hashes[did] = sha256(path)
    np.testing.assert_allclose(truth[:, :2], expected, rtol=2e-5, atol=2e-5)
    truth[:, 2] = y
    assert np.all(train[:, :, 36] >= 1) and np.all(evaluation[:, :, 36] >= 1)
    return train, evaluation, truth, ids, rows, names, folds, dt, clamps, raw_hashes


def split(ids, folds, fold):
    outer = np.flatnonzero(folds != fold)
    test = np.flatnonzero(folds == fold)
    designs = np.unique(ids[outer])
    np.random.default_rng(947 + fold).shuffle(designs)
    count = int(np.ceil(.2 * len(designs)))
    cal, fit = outer[np.isin(ids[outer], designs[:count])], outer[np.isin(ids[outer], designs[count:])]
    assert not set(ids[fit]) & set(ids[cal]) and not set(ids[outer]) & set(ids[test])
    return fit, cal, test


def adjustment(y, output, ids):
    scores = np.maximum.reduce([output[:, 1] - y, y - output[:, 2], np.zeros(len(y))])
    maxima = np.sort([scores[ids == did].max() for did in np.unique(ids)])
    rank = int(np.ceil((len(maxima) + 1) * .9))
    if rank > len(maxima):
        raise ValueError('too few calibration designs for finite90% group-max correction')
    return float(maxima[rank - 1])


def warning_threshold(probabilities, y, budget):
    scores = np.sort(-probabilities[y > 0])
    threshold = float(np.nextafter(scores[int(np.floor(budget * len(scores)))], -np.inf))
    assert np.mean(-probabilities[y > 0] <= threshold) <= budget
    return threshold


def source_signature(raw_hashes):
    paths = [Path(__file__), OUT / 'protocol.json', ROOT / 'p4_dataset_saturation/metadata_arrays.npz',
             helpers.base.PRIOR / 'features.npz', helpers.base.VERIFY / 'features.npz',
             helpers.BASE / 'confirmation_features.npz',
             ROOT / 'p4_groups/boundary_noise_consistency_20260930/fresh_features.npz',
             ROOT / 'experiment_p4_tabular.py', ROOT / 'src/mechanics/p4_margin_estimation/modal_features.py']
    references = {f'warning_{arm}_seed{seed}_fold{fold}.npz': sha256(helpers.BASE / f'warning_{arm}_seed{seed}_fold{fold}.npz')
                  for arm in ('one', 'PCS') for seed in SEEDS for fold in range(5)}
    return {'sources': {str(p.relative_to(ROOT)): sha256(p) for p in paths}, 'raw': raw_hashes,
            'references': references, 'versions': helpers.base.source_signature()['versions']}


def reference_predictors(seed, fold, fit, cal, test):
    predictors = []
    for arm in ('one', 'PCS'):
        path = helpers.BASE / f'warning_{arm}_seed{seed}_fold{fold}.npz'
        with np.load(path) as z:
            for key, value in [('proper_rows', fit), ('calibration_rows', cal), ('held_rows', test)]:
                np.testing.assert_array_equal(z[key], value)
        predictors.append(helpers.base.trusted_predictor(path))
    return predictors


def run():
    train, evaluation, truth, ids, rows, names, folds, dt, clamps, raw_hashes = load_data()
    y = truth[:, 2]
    signature = source_signature(raw_hashes)
    measured_train, measured = train[:, :, [27, 28]], evaluation[:, :, [27, 28]]
    n, nc, ns = len(y), len(names), len(SEEDS)
    output = np.full((len(ARMS) + len(REFERENCES), ns, nc, n, len(FIELDS)), np.nan)
    oracle = np.full((ns, n, len(FIELDS)), np.nan)
    mlp_diagnostic = np.full((2, ns, nc, n), np.nan)
    corrections = np.full((len(ARMS) + len(REFERENCES), ns, 5), np.nan)
    thresholds = np.full((len(ARMS) + len(REFERENCES), ns, 5, len(BUDGETS)), np.nan)
    splits, fits = [], []
    for fold in range(5):
        fit, cal, test = split(ids, folds, fold)
        splits.append({'fold': fold, 'proper_designs': np.unique(ids[fit]).tolist(),
                       'calibration_designs': np.unique(ids[cal]).tolist(), 'heldout_designs': np.unique(ids[test]).tolist()})
        ycal, idcal = np.tile(y[cal], 4), np.tile(ids[cal], 4)
        for si, seed in enumerate(SEEDS):
            cache = OUT / f'physical_seed{seed}_fold{fold}.npz'
            if cache.exists():
                with np.load(cache) as z:
                    assert json.loads(str(z['fingerprint'])) == signature, 'stale physical model'
                    for key, value in [('proper_rows', fit), ('calibration_rows', cal), ('held_rows', test)]:
                        np.testing.assert_array_equal(z[key], value)
                    prior, meta = cloudpickle.loads(z['model'].tobytes()), json.loads(str(z['metadata']))
            else:
                prior = PhysicalPrior()
                start = time.perf_counter()
                meta = prior.fit(truth[fit], measured_train[:, fit], seed)
                meta['seconds'] = time.perf_counter() - start
                save_npz(cache, model=np.frombuffer(cloudpickle.dumps(prior), dtype=np.uint8),
                         fingerprint=json.dumps(signature), metadata=json.dumps(meta),
                         proper_rows=fit, calibration_rows=cal, held_rows=test)
            fits.append({'seed': seed, 'fold': fold, 'model_hash': sha256(cache), **meta})
            cal_queries = measured_train[:, cal].reshape(-1, 2)
            test_queries = measured[:, test].reshape(-1, 2)
            for ai, arm in enumerate(ARMS):
                calibration = prior.predict(cal_queries, arm)
                held = prior.predict(test_queries, arm).reshape(nc, len(test), len(FIELDS))
                output[ai, si][:, test] = held
                corrections[ai, si, fold] = adjustment(ycal, calibration, idcal)
                thresholds[ai, si, fold] = [warning_threshold(calibration[:, 3], ycal, b) for b in BUDGETS]
            oracle[si, test] = prior.predict(truth[test, :2], 'joint_naive')
            native = reference_predictors(seed, fold, fit, cal, test)
            reference_cal = [native[0](train[:, cal, :107].reshape(-1, 107)), native[1](train[:, cal].reshape(-1, 174))]
            reference_test = [native[0](evaluation[:, test, :107].reshape(-1, 107)), native[1](evaluation[:, test].reshape(-1, 174))]
            reference_cal.append(.5 * (reference_cal[0] + reference_cal[1]))
            reference_test.append(.5 * (reference_test[0] + reference_test[1]))
            for ri, (cp, hp) in enumerate(zip(reference_cal, reference_test)):
                radius = float(np.quantile(abs(cp - ycal), .9))
                rms = max(float(np.sqrt(np.mean((cp - ycal) ** 2))), 1e-12)
                calibration = np.column_stack([cp, cp - radius, cp + radius, ndtr(-cp / rms), np.zeros((len(cp), 2))])
                held = np.column_stack([hp, hp - radius, hp + radius, ndtr(-hp / rms), np.zeros((len(hp), 2))])
                ai = len(ARMS) + ri
                output[ai, si][:, test] = held.reshape(nc, len(test), len(FIELDS))
                corrections[ai, si, fold] = adjustment(ycal, calibration, idcal)
                thresholds[ai, si, fold] = [warning_threshold(calibration[:, 3], ycal, b) for b in BUDGETS]
            for di, kind in enumerate(('true', 'measured')):
                path = OUT / f'diagnostic_{kind}_seed{seed}_fold{fold}.npz'
                if path.exists():
                    with np.load(path) as z:
                        assert json.loads(str(z['fingerprint'])) == signature
                        predict = cloudpickle.loads(z['predictor'].tobytes())
                        fit_meta = json.loads(str(z['metadata']))
                else:
                    tx = truth[fit, :2] if kind == 'true' else measured_train[:, fit].reshape(-1, 2)
                    ty = y[fit] if kind == 'true' else np.tile(y[fit], 4)
                    predict, fit_meta = fit_network(tx, ty, seed=seed)
                    if not fit_meta['success']:
                        raise RuntimeError(f'failed diagnostic fit: {path.name}: {fit_meta}')
                    save_npz(path, predictor=np.frombuffer(cloudpickle.dumps(predict), dtype=np.uint8),
                             metadata=json.dumps(fit_meta), fingerprint=json.dumps(signature))
                q = np.tile(truth[test, :2], (nc, 1)) if kind == 'true' else test_queries
                mlp_diagnostic[di, si][:, test] = predict(q).reshape(nc, len(test))
                fits.append({'kind': kind, 'seed': seed, 'fold': fold, 'model_hash': sha256(path), **fit_meta})
            print('Completed physical fold', fold, 'seed', seed, json.dumps(meta), flush=True)
    for value in (output, oracle, mlp_diagnostic, corrections, thresholds):
        assert np.isfinite(value).all()
    save_npz(OUT / 'predictions.npz', predictions=output, oracle=oracle, diagnostics=mlp_diagnostic,
             corrections=corrections, thresholds=thresholds, truth=truth, y=y, ids=ids,
             source_indices=rows, row_fold=folds, conditions=np.array(names), arms=np.array(ARMS + REFERENCES),
             seeds=np.array(SEEDS), fields=np.array(FIELDS), budgets=np.array(BUDGETS),
             measured=measured, usable_time=np.maximum(0, np.rint(evaluation[:, :, 20] * 512) - 1) * dt,
             fit_error=evaluation[:, :, 19], energy=evaluation[:, :, 29], clamps=clamps,
             fingerprint=json.dumps(signature))
    assert source_signature(raw_hashes) == signature, 'reference/source changed during run'
    save_json(OUT / 'fits.json', {'splits': splits, 'fits': fits, 'fingerprint': signature})
    print('Completed all physical inference arms', output.shape, flush=True)


def smoke():
    # Closed-form one-component check: observing correlated frequency reduces alpha uncertainty.
    from types import SimpleNamespace
    p = PhysicalPrior()
    p.mean, p.scale = np.zeros(3), np.ones(3)
    p.prior = SimpleNamespace(weights_=np.ones(1), means_=np.zeros((1, 3)),
        covariances_=np.array([[[1., .8, -.5], [.8, 1., -.4], [-.5, -.4, 1.]]]))
    p.error = SimpleNamespace(weights_=np.ones(1), means_=np.zeros((1, 2)), covariances_=np.array([np.diag([1., .01])]))
    query = np.array([[.5, .5], [-.5, -.5]])
    soft, growth = p.predict(query, 'joint_soft'), p.predict(query, 'growth_only')
    expected = np.linalg.solve(np.array([[2., .8], [.8, 1.01]]), query[0])
    np.testing.assert_allclose(soft[0, 0], np.array([-.5, -.4]) @ expected, atol=1e-12)
    assert np.all(soft[:, 5] < growth[:, 5])
    np.testing.assert_allclose(soft[0, 3] + soft[1, 3], 1., atol=1e-12)
    assert soft[0, 0] < 0 and soft[1, 0] > 0
    train, evaluation, truth, ids, rows, names, folds, dt, clamps, hashes = load_data()
    fit, cal, test = split(ids, folds, 0)
    prior = PhysicalPrior()
    metadata = prior.fit(truth[fit], train[:, fit][:, :, [27, 28]], 42)
    actual = prior.predict(evaluation[names.index('iid1_new949'), test[:32]][:, [27, 28]], 'joint_soft')
    assert np.all(actual[:, 3] >= 0) and np.all(actual[:, 3] <= 1)
    assert np.all(actual[:, 1] < actual[:, 2])
    reference_predictors(42, 0, fit, cal, test)
    print('Analytical covariance/quantile and actual32-clip posterior smoke passed', json.dumps(metadata),
          'first posterior', actual[0].tolist(), 'designs', len(np.unique(ids)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        smoke() if args.smoke else run()
