"""TRAIN-only feature-coordinate experiment, with frozen blend replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'p4_groups/local_observability_20261001'))
import contract as C
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(OUT))
from stability_coordinates import augmented, additions, numerical_check
np, cloudpickle = C.np, C.cloudpickle
ARMS = ['blend_frozen', 'full_frozen', 'full_products', 'blend_products', 'full_hurwitz', 'blend_hurwitz']
SEED = 42
DATA = None


def signature():
    return {'sources': {str(p.relative_to(ROOT)): C.sha256(p) for p in
            (OUT / 'protocol.json', Path(__file__), OUT / 'stability_coordinates.py')},
            'contract': C.signature()}


def setup():
    global DATA
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = C.load_data()
    with np.load(C.OUT / 'predictions.npz') as z:
        for key, value in [('y', y), ('ids', ids), ('source_indices', rows), ('row_fold', folds)]:
            np.testing.assert_array_equal(z[key], value)
        assert z['conditions'].tolist() == names
        baseline = z['predictions'][0, C.SEEDS.index(SEED)].copy()
        thresholds = z['thresholds'][0, C.SEEDS.index(SEED)].copy()
    DATA = dict(train=train, evaluation=evaluation, y=y, ids=ids, rows=rows, names=names,
                folds=folds, layout=layout, pairs=pairs, dt=dt, baseline=baseline,
                baseline_thresholds=thresholds, fingerprint=signature())
    for value in DATA.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return DATA


def fit_model(kind, fold, proper, cal, held):
    d = DATA
    path = OUT / f'{kind}_seed{SEED}_fold{fold}.npz'
    partitions = dict(proper_rows=proper, calibration_rows=cal, held_rows=held)
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['fingerprint'])) == d['fingerprint'], 'stale coordinate model'
            for key, value in partitions.items():
                np.testing.assert_array_equal(z[key], value)
            predict = cloudpickle.loads(z['predictor'].tobytes())
            meta = json.loads(str(z['metadata']))
    else:
        x = augmented(d['train'][:, proper], kind).reshape(-1, 195)
        predict, meta = C.fit_network(x, np.tile(d['y'][proper], 4), seed=SEED,
                                     alpha=1., max_iter=20000, max_fun=400000)
        C.save_npz(path, predictor=np.frombuffer(cloudpickle.dumps(predict), dtype=np.uint8),
                   metadata=json.dumps(meta), fingerprint=json.dumps(d['fingerprint']), **partitions)
        print('COORDINATE_FIT', path.name, json.dumps(meta), flush=True)
    if not meta['success']:
        raise RuntimeError(f'nonconverged coordinate model: {path}: {meta}')
    return predict, dict(kind=kind, seed=SEED, fold=fold, path=str(path.relative_to(ROOT)),
                         hash=C.sha256(path), **meta)


def fold_job(fold):
    with threadpool_limits(limits=1):
        d = DATA
        proper, cal, held = C.split(d['ids'], d['folds'], fold)
        path = OUT / f'fold{fold}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint'])) == d['fingerprint'], 'stale coordinate fold'
            return fold
        one, full = C.reference_predictors(SEED, fold, proper, cal, held)
        qcal = d['train'][:, cal].reshape(-1, 174)
        qheld = d['evaluation'][:, held].reshape(-1, 174)
        onecal, oneheld = one(qcal[:, :107]), one(qheld[:, :107])
        fullcal, fullheld = full(qcal), full(qheld)
        points_cal = [.5 * (onecal + fullcal), fullcal]
        points_held = [.5 * (oneheld + fullheld), fullheld]
        records = []
        for kind in ('products', 'hurwitz'):
            model, record = fit_model(kind, fold, proper, cal, held)
            records.append(record)
            cp, hp = model(augmented(qcal, kind)), model(augmented(qheld, kind))
            points_cal.extend((cp, .5 * (onecal + cp)))
            points_held.extend((hp, .5 * (oneheld + hp)))
            with np.load(ROOT / record['path']) as z:
                restored = cloudpickle.loads(z['predictor'].tobytes())
            np.testing.assert_array_equal(restored(augmented(qheld[:3], kind)), model(augmented(qheld[:3], kind)))
        target, cids = np.tile(d['y'][cal], 4), np.tile(d['ids'][cal], 4)
        predictions, thresholds, calibration, budgets_observed = [], [], [], []
        for cp, hp in zip(points_cal, points_held):
            radius = C.calibration(cp, target, cids)
            cdist, hdist = C.distribution(cp, radius), C.distribution(hp, radius)
            levels = [C.threshold(cdist[:, 3], target, b) for b in C.BUDGETS]
            actual = [float(np.mean(-cdist[target > 0, 3] <= level)) for level in levels]
            assert np.all(np.array(actual) <= C.BUDGETS)
            predictions.append(hdist.reshape(len(d['names']), len(held), 6))
            thresholds.append(levels)
            calibration.append(radius)
            budgets_observed.append(actual)
        predictions, thresholds = np.array(predictions), np.array(thresholds)
        np.testing.assert_allclose(predictions[0], d['baseline'][:, held], atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(thresholds[0], d['baseline_thresholds'][fold])
        assert np.isfinite(predictions).all() and np.isfinite(thresholds).all()
        C.save_npz(path, predictions=predictions, thresholds=thresholds,
                   calibration_points=np.array(points_cal).reshape(len(ARMS), 4, len(cal)),
                   calibration=json.dumps(calibration), records=json.dumps(records),
                   calibration_false_warning_rates=budgets_observed,
                   proper_rows=proper, calibration_rows=cal, held_rows=held,
                   fingerprint=json.dumps(d['fingerprint']))
        print('COORDINATE_FOLD_COMPLETE', fold, len(held), 'frozen replay and calibration budgets passed', flush=True)
        return fold


def run(workers):
    d = setup()
    proof = numerical_check()
    for kind in ('hurwitz', 'products'):
        assert additions(d['evaluation'], kind).shape == (37, 4454, 21)
    C.save_json(OUT / 'numerical_verification.json', proof)
    print('COORDINATE_NUMERICAL_CHECK', json.dumps(proof), flush=True)
    if workers == 1:
        for fold in range(5):
            fold_job(fold)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for future in as_completed([pool.submit(fold_job, fold) for fold in range(5)]):
                future.result()
    predictions = np.full((len(ARMS), 1, len(d['names']), len(d['y']), 6), np.nan)
    thresholds = np.full((len(ARMS), 1, 5, len(C.BUDGETS)), np.nan)
    records, partitions = [], []
    for fold in range(5):
        with np.load(OUT / f'fold{fold}.npz') as z:
            assert json.loads(str(z['fingerprint'])) == d['fingerprint']
            held = z['held_rows']
            predictions[:, 0][:, :, held] = z['predictions']
            thresholds[:, 0, fold] = z['thresholds']
            records.extend(json.loads(str(z['records'])))
            partitions.append(dict(fold=fold, **{k: z[k].tolist() for k in ('proper_rows', 'calibration_rows', 'held_rows')}))
    assert len(records) == 10 and all(r['success'] for r in records)
    assert np.isfinite(predictions).all() and np.isfinite(thresholds).all()
    assert signature() == d['fingerprint'], 'source changed during execution'
    C.save_npz(OUT / 'predictions.npz', predictions=predictions, thresholds=thresholds, y=d['y'], ids=d['ids'],
               source_indices=d['rows'], row_fold=d['folds'], conditions=d['names'], arms=ARMS, seeds=[SEED],
               budgets=C.BUDGETS, fields=C.FIELDS, fingerprint=json.dumps(d['fingerprint']))
    C.save_json(OUT / 'fits.json', dict(all_converged=True, fits=records, partitions=partitions, fingerprint=d['fingerprint']))
    print('COORDINATE_EXPERIMENT_COMPLETE', predictions.shape, 'ten converged fits; five design folds', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.workers)
