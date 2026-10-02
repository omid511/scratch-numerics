"""Fixed three-arm comparison; native predictors and predictions resume atomically."""
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'p4_groups/correlated_training_20260929'
SPATIAL = ROOT / 'p4_groups/spatial_modal_20260930'
sys.path[:0] = [str(ROOT), str(PARENT)]
import cloudpickle
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from experiment_p4_tabular import fit_network
from run_augmentation_experiment import sha256


def main():
    with np.load(OUT / 'features.npz') as z:
        train, extra_train = z['training107'], z['training_additions']
        evaluation, extra_eval = z['evaluation107'], z['evaluation_additions']
        y, ids, rows, conditions = z['y'], z['ids'], z['source_indices'], z['conditions']
    with np.load(SPATIAL / 'predictions.npz') as z:
        for key, value in [('y', y), ('ids', ids), ('source_indices', rows)]:
            np.testing.assert_array_equal(z[key], value)
        row_fold, historical = z['row_fold'], z['conditions'].tolist()
        reference_tr, reference_ho = z['training_predictions'][0], z['heldout_predictions'][0]
    assert conditions[:len(historical)].tolist() == historical
    seeds, arms = [42, 43, 44], ['spatial107', 'two_mode174', 'shuffled_addition174']
    fingerprint = {
        'protocol': sha256(OUT / 'protocol.json'), 'execution': sha256(Path(__file__)),
        'features': sha256(OUT / 'features.npz'), 'prior_predictions': sha256(SPATIAL / 'predictions.npz'),
        'sources': {f: sha256(ROOT / f) for f in
                    ('experiment_p4_tabular.py', 'src/mechanics/p4_margin_estimation/baselines.py')},
        'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__, 'cloudpickle': cloudpickle.__version__},
    }
    n, nc, nh = len(y), len(conditions), len(historical)
    training, held = np.full((3, 3, 4, nc, n), np.nan), np.full((3, 3, nc, n), np.nan)
    fits = []
    replay_error = 0.
    for si, seed in enumerate(seeds):
        for fi in range(5):
            fit, test = np.flatnonzero(row_fold != fi), np.flatnonzero(row_fold == fi)
            assert not set(ids[fit]) & set(ids[test])
            rng = np.random.default_rng(934+fi)
            shuffled_fit, shuffled_test = rng.permutation(fit), rng.permutation(test)
            for ai, arm in enumerate(arms):
                added_fit = shuffled_fit if ai == 2 else fit
                added_test = shuffled_test if ai == 2 else test
                width = 107 if ai == 0 else 174
                cache = OUT / f'{arm}_seed{seed}_fold{fi}.npz'
                if cache.exists():
                    with np.load(cache) as z:
                        assert json.loads(str(z['fingerprint'])) == fingerprint, 'stale fit cache'
                        for key, value in [('train_rows', fit), ('held_rows', test),
                                           ('addition_train_rows', added_fit), ('addition_held_rows', added_test)]:
                            np.testing.assert_array_equal(z[key], value)
                        tr, ho = z['training_predictions'], z['heldout_predictions']
                        metadata = json.loads(str(z['metadata']))
                    print('Resume', arm, seed, fi, flush=True)
                else:
                    x = train[:, fit] if ai == 0 else np.concatenate([train[:, fit], extra_train[:, added_fit]], axis=2)
                    predict, metadata = fit_network(x.reshape(-1, width), np.tile(y[fit], 4), seed=seed,
                                                    alpha=1., max_iter=10000, max_fun=200000)
                    tr = np.stack([predict(q[fit] if ai == 0 else np.column_stack([q[fit], e[added_fit]]))
                                   for q, e in zip(evaluation, extra_eval)])
                    ho = np.stack([predict(q[test] if ai == 0 else np.column_stack([q[test], e[added_test]]))
                                   for q, e in zip(evaluation, extra_eval)])
                    model = cloudpickle.dumps(predict)
                    restored = cloudpickle.loads(model)  # Only our own trusted in-memory predictor.
                    query = evaluation[0, test] if ai == 0 else np.column_stack([evaluation[0, test], extra_eval[0, added_test]])
                    np.testing.assert_array_equal(restored(query), ho[0])
                    metadata.update({'features': width, 'training_rows': len(x)*len(fit)})
                    with cache.with_suffix('.tmp').open('wb') as handle:
                        np.savez_compressed(handle, train_rows=fit, held_rows=test,
                                            addition_train_rows=added_fit, addition_held_rows=added_test,
                                            training_predictions=tr, heldout_predictions=ho,
                                            predictor=np.frombuffer(model, dtype=np.uint8),
                                            metadata=json.dumps(metadata), fingerprint=json.dumps(fingerprint))
                    cache.with_suffix('.tmp').replace(cache)
                    print('Fit', arm, seed, fi, json.dumps(metadata), flush=True)
                assert np.isfinite(tr).all() and np.isfinite(ho).all()
                slots = fi-(row_fold[fit] < fi).astype(int)
                for ci in range(nc):
                    training[ai, si, slots, ci, fit] = tr[ci]
                    held[ai, si, ci, test] = ho[ci]
                if ai == 0:
                    np.testing.assert_array_equal(ho[:nh], reference_ho[si][:, test])
                fits.append({'arm': arm, 'seed': seed, 'fold': fi, **metadata})
    assert np.isfinite(training).all() and np.isfinite(held).all()
    np.testing.assert_array_equal(held[0, :, :nh], reference_ho)
    np.testing.assert_array_equal(training[0, :, :, :nh], reference_tr)
    replay_error = float(abs(held[0, :, :nh]-reference_ho).max())
    path = OUT / 'predictions.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, training_predictions=training, heldout_predictions=held,
                            y=y, ids=ids, source_indices=rows, row_fold=row_fold,
                            conditions=conditions, seeds=seeds, arms=arms)
    path.with_suffix('.tmp').replace(path)
    (OUT / 'fits.json').write_text(json.dumps({'fingerprint': fingerprint, 'fits': fits,
        'historical_prediction_replay_max_error': replay_error,
        'all_fits_converged': all(f['success'] for f in fits)}, indent=2, allow_nan=False)+'\n')
    print('Completed', len(fits), 'fits; converged:', all(f['success'] for f in fits),
          'historical replay delta', replay_error, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
