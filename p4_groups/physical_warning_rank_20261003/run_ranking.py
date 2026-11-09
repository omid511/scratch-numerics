"""Warning-only empirical low-FPR ranking; frozen margin/interval outputs."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'p4_groups/stability_correction_20261003'
sys.path.insert(0, str(PARENT))
import run_correction as E
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
C, np = E.C, E.np
ARMS = ['blend_frozen', 'rank_control', 'rank_products', 'rank_hurwitz']
FIELDS = ['point', 'raw_lower', 'raw_upper', 'warning_score', 'group_lower', 'group_upper']
UNIT, FRACTION, REG = .01, .05, .1


def signature():
    return dict(parent=E.signature(), sources={str(p.relative_to(ROOT)): C.sha256(p)
                for p in (OUT / 'protocol.json', Path(__file__))})


def features(control, q, arm):
    return control if arm == 'rank_control' else np.column_stack((control, E.H.additions(q, arm.split('_')[1])))


def standardized(head, x):
    z = (x-head['mean'])/head['scale']
    z[:, ~head['active']] = 0.
    return z


def score(head, control, q, arm):
    return -control[:, 0]/UNIT + standardized(head, features(control, q, arm)) @ head['coef']


def objective(z, prior, y):
    stable, unsafe = np.flatnonzero(y>0), np.flatnonzero((y>-.05) & (y<=0))
    if not len(stable) or not len(unsafe):
        raise ValueError('empty warning objective cohort')
    count = int(np.ceil(FRACTION*len(stable)))
    def evaluate(w):
        s = prior + z @ w
        selected = stable[np.argpartition(s[stable], -count)[-count:]]
        hinge = np.maximum(1+s[selected, None]-s[unsafe][None, :], 0.)
        denominator = hinge.size
        grad = z[selected].T @ (2*hinge.sum(axis=1)/denominator)
        grad -= z[unsafe].T @ (2*hinge.sum(axis=0)/denominator)
        grad += REG*w
        return float(np.mean(hinge*hinge) + .5*REG*np.dot(w, w)), grad
    return evaluate, dict(stable_copies=len(stable), unsafe_copies=len(unsafe), top_stable_copies=count)


def fit_head(control, q, y, arm, check=False):
    x = features(control, q, arm)
    mean, scale, active = fit_ridge_scaling(x)
    head = dict(mean=mean, scale=scale, active=active)
    z = standardized(head, x)
    function, cohort = objective(z, -control[:, 0]/UNIT, y)
    proof = {}
    if check:
        w = np.random.default_rng(71).normal(size=x.shape[1])*.001
        analytic = function(w)[1]; numeric = np.empty_like(w)
        for j in range(len(w)):
            delta = np.zeros_like(w); delta[j] = 1e-6
            numeric[j] = (function(w+delta)[0]-function(w-delta)[0])/(2e-6)
        np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-5)
        proof['gradient_max_error'] = float(abs(analytic-numeric).max())
    start = time.perf_counter()
    initial = np.zeros(x.shape[1])
    result = minimize(function, initial, jac=True, method='L-BFGS-B',
        options=dict(maxiter=2000, maxfun=20000, ftol=1e-12, gtol=1e-7))
    meta = dict(arm=arm, success=bool(result.success), message=str(result.message), iterations=int(result.nit),
        evaluations=int(result.nfev), loss=float(result.fun), initial_loss=function(initial)[0],
        seconds=time.perf_counter()-start, active_columns=int(active.sum()), **cohort, **proof)
    C.save_json(OUT / 'latest_optimizer.json', meta)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'warning-ranking fit failed: {meta}')
    head['coef'] = result.x
    return head, meta


def run():
    d = E.setup()
    fp = signature()
    predictions = np.repeat(d['baseline'][None, None], len(ARMS), axis=0)
    thresholds = np.full((len(ARMS), 1, 5, len(C.BUDGETS)), np.nan)
    records = []
    for fold in range(5):
        proper, cal, held = C.split(d['ids'], d['folds'], fold)
        tx, cx, hx, _, _, _, _ = E.previous_fold(fold, proper, cal, held)
        tq, cq, hq = [v[:, rows].reshape(-1, 174) for v, rows in
                      ((d['train'], proper), (d['train'], cal), (d['evaluation'], held))]
        tc, cc, hc = tx.reshape(-1, 7), cx.reshape(-1, 7), hx.reshape(-1, 7)
        y, cy = np.tile(d['y'][proper], 4), np.tile(d['y'][cal], 4)
        heads, cal_scores, held_scores, fold_records = {}, [], [], []
        frozen_radius = C.calibration(cc[:, 0], cy, np.tile(d['ids'][cal], 4))
        cal_scores.append(C.distribution(cc[:, 0], frozen_radius)[:, 3])
        held_scores.append(d['baseline'][:, held, 3].reshape(-1))
        for arm in ARMS[1:]:
            head, meta = fit_head(tc, tq, y, arm, check=fold==0)
            heads[arm] = head
            cal_scores.append(score(head, cc, cq, arm)); held_scores.append(score(head, hc, hq, arm))
            fold_records.append(dict(fold=fold, seed=42, **meta))
            print('WARNING_RANK_FIT', json.dumps(fold_records[-1]), flush=True)
        levels, rates = [], []
        for ai, arm in enumerate(ARMS):
            thresholds_ai = [C.threshold(cal_scores[ai], cy, b) for b in C.BUDGETS]
            observed = [float(np.mean(-cal_scores[ai][cy>0] <= t)) for t in thresholds_ai]
            assert np.all(np.array(observed) <= C.BUDGETS)
            levels.append(thresholds_ai); rates.append(observed)
            predictions[ai, 0, :, held, 3] = np.array(held_scores[ai]).reshape(37, len(held)).T
        np.testing.assert_array_equal(levels[0], d['baseline_thresholds'][fold])
        thresholds[:, 0, fold] = levels
        C.save_npz(OUT / f'fold{fold}.npz', heads=np.frombuffer(C.cloudpickle.dumps(heads), dtype=np.uint8),
            calibration_scores=cal_scores, calibration_false_warning_rates=rates, thresholds=levels,
            proper_rows=proper, calibration_rows=cal, held_rows=held, records=json.dumps(fold_records), fingerprint=json.dumps(fp))
        records.extend(fold_records)
        print('WARNING_RANK_FOLD_COMPLETE', fold, 'frozen outputs, source replay and calibration budgets passed', flush=True)
    assert len(records) == 15 and all(r['success'] for r in records)
    assert np.isfinite(predictions).all() and np.isfinite(thresholds).all()
    for i in range(len(ARMS)):
        np.testing.assert_array_equal(predictions[i, 0][..., [0,1,2,4,5]], d['baseline'][..., [0,1,2,4,5]])
    np.testing.assert_array_equal(predictions[0, 0], d['baseline'])
    assert signature() == fp
    C.save_npz(OUT / 'predictions.npz', predictions=predictions, thresholds=thresholds, y=d['y'], ids=d['ids'],
        source_indices=d['rows'], row_fold=d['folds'], conditions=d['names'], arms=ARMS, seeds=[42],
        budgets=C.BUDGETS, fields=FIELDS, fingerprint=json.dumps(fp))
    C.save_json(OUT / 'fits.json', dict(all_converged=True, fits=records, fingerprint=fp,
                                      baseline_compatibility=d.get('baseline_compatibility', {})))
    print('WARNING_RANK_EXPERIMENT_COMPLETE', predictions.shape, 'fifteen low-FPR heads, frozen margin outputs', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
