"""Capture unavailable in-sample predictions of the frozen P4 candidate.

Completed fits resume atomically; saved held-out predictions remain primary.
No reserved split is loaded. Refit settings match the existing seed experiment.
"""
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'p4_groups/correlated_training_20260929'
sys.path[:0] = [str(ROOT), str(PARENT)]
import numpy as np
from threadpoolctl import threadpool_limits
from experiment_p4_tabular import fit_network
from run_augmentation_experiment import sha256


def main():
    sources = ('experiment_p4_tabular.py', 'experiment_p4_ridge_cv.py',
               'src/mechanics/p4_margin_estimation/baselines.py')
    fingerprint = {
        'inputs': sha256(PARENT / 'shared.npz'),
        'protocol': sha256(OUT / 'protocol.json'),
        'execution': sha256(Path(__file__)),
        'sources': {p: sha256(ROOT / p) for p in sources},
    }
    with np.load(PARENT / 'shared.npz') as z:
        views = np.concatenate((z['train3'], z['independent_extra'][None]), axis=0)
        y, ids, rows = z['y'], z['ids'], z['source_indices']
        names = z['evaluation_names'].tolist()
        conditions = ['clean', 'iid1_fresh932']
        evaluation = z['evaluation'][[names.index(c) for c in conditions]]
    seeds = [42, 43, 44]
    n = len(y)
    train = np.full((3, 4, 2, n), np.nan)
    held = np.full((3, 2, n), np.nan)
    saved_held = np.full_like(held, np.nan)
    metadata = []
    assignments = None
    for si, (seed, filename) in enumerate(zip(seeds, (
            'augmentation_results.json', 'augmentation_seed43.json', 'augmentation_seed44.json'))):
        run = json.loads((PARENT / filename).read_text())
        fingerprint[f'saved_seed{seed}'] = sha256(PARENT / filename)
        np.testing.assert_array_equal(run['design_assignments']['source_indices'], rows)
        row_fold = np.asarray(run['design_assignments']['row_fold'])
        if assignments is None:
            assignments = row_fold
        np.testing.assert_array_equal(assignments, row_fold)
        for ci, condition in enumerate(conditions):
            saved_held[si, ci] = run['arms']['extra_independent4']['records'][condition]['predictions']
        for fold in run['arms']['extra_independent4']['folds']:
            fi = fold['fold']
            test = np.asarray(fold['test_row_indices'])
            fit_rows = np.flatnonzero(row_fold != fi)
            assert not set(ids[fit_rows]) & set(ids[test])
            cache = OUT / f'seed{seed}_fold{fi}.npz'
            if cache.exists():
                with np.load(cache) as z:
                    assert json.loads(str(z['fingerprint'])) == fingerprint, 'stale fit cache'
                    np.testing.assert_array_equal(z['train_rows'], fit_rows)
                    np.testing.assert_array_equal(z['held_rows'], test)
                    train_predictions, held_predictions = z['train_predictions'], z['held_predictions']
                    fit_metadata = json.loads(str(z['metadata']))
                print('Resume', seed, fi, flush=True)
            else:
                x = np.concatenate([view[fit_rows] for view in views])
                target = np.tile(y[fit_rows], 4)
                predict, fit_metadata = fit_network(x, target, seed=seed, alpha=1.,
                                                    max_iter=10000, max_fun=200000)
                train_predictions = np.stack([predict(q[fit_rows]) for q in evaluation])
                held_predictions = np.stack([predict(q[test]) for q in evaluation])
                if not np.isfinite(train_predictions).all() or not np.isfinite(held_predictions).all():
                    raise ValueError('nonfinite candidate predictions')
                temporary = cache.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, train_rows=fit_rows, held_rows=test,
                                        train_predictions=train_predictions, held_predictions=held_predictions,
                                        metadata=json.dumps(fit_metadata), fingerprint=json.dumps(fingerprint))
                temporary.replace(cache)
                print('Fit', seed, fi, json.dumps(fit_metadata), flush=True)
            # Compare only as a reconstruction check: no new held-out model selection.
            np.testing.assert_allclose(held_predictions, saved_held[si][:, test], atol=1e-10, rtol=1e-10)
            slots = fi - (row_fold[fit_rows] < fi).astype(int)
            for ci in range(2):
                train[si, slots, ci, fit_rows] = train_predictions[ci]
                held[si, ci, test] = held_predictions[ci]
            metadata.append({'seed': seed, 'fold': fi, **fit_metadata})
    assert np.isfinite(train).all() and np.isfinite(held).all()
    final = OUT / 'predictions.npz'
    with final.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, training_predictions=train, heldout_predictions=saved_held,
                            y=y, ids=ids, source_indices=rows, row_fold=assignments,
                            conditions=conditions, seeds=seeds)
    final.with_suffix('.tmp').replace(final)
    result = {'fingerprint': fingerprint, 'fits': metadata,
              'all_fits_converged': all(m['success'] for m in metadata),
              'max_reconstruction_error': float(np.max(abs(held - saved_held))),
              'train_shape': list(train.shape), 'held_shape': list(held.shape)}
    (OUT / 'fits.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Completed', json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
