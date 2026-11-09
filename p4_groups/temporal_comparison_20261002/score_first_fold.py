"""Score and replay the approved, fully trained seed42/fold0 screening only."""
import json
from pathlib import Path
import experiment_temporal as T
import evaluate_temporal as scoring
import verify_temporal as verification

C, np, OUT = T.C, T.np, T.OUT


def assemble():
    d = T.setup()
    selected = np.flatnonzero(d['folds'] == 0)
    with np.load(OUT / 'fold_seed42_fold0.npz') as z:
        assert json.loads(str(z['fingerprint'])) == d['fingerprint'], 'stale first-fold result'
        np.testing.assert_array_equal(z['held_rows'], selected)
        predictions = z['predictions'][:, None]
        native = z['native_quantiles'][:, None]
        thresholds = np.full((len(T.ARMS), 1, 5, len(C.BUDGETS)), np.nan)
        thresholds[:, 0, 0] = z['thresholds']
        records = json.loads(str(z['records']))
        partitions = [dict(seed=42, fold=0, **{k: z[k].tolist() for k in
            ('proper_rows', 'calibration_rows', 'held_rows')})]
        budgets = [dict(seed=42, fold=0, rates=z['calibration_false_warning'].tolist())]
    assert np.isfinite(predictions).all() and np.isfinite(native).all()
    assert np.isfinite(thresholds[:, :, 0]).all()
    assert sum(not record.get('frozen_base', False) for record in records) == 8
    scope = dict(kind='first_fold_screening', seed=42, outer_fold=0,
        training='Full epoch selection and refit; original optimizer and split protocol',
        limitation='One held-design group and one initialization; not full confirmation.',
        assembly_source_hash=C.sha256(Path(__file__)))
    C.save_npz(OUT / 'screening_predictions.npz', predictions=predictions,
        thresholds=thresholds, native_quantiles=native, y=d['y'][selected],
        ids=d['ids'][selected], source_indices=d['rows'][selected],
        row_fold=d['folds'][selected], conditions=d['names'], arms=T.ARMS,
        seeds=[42], fields=C.FIELDS, fingerprint=json.dumps(d['fingerprint']),
        scope=json.dumps(scope))
    C.save_json(OUT / 'screening_fits.json', dict(fingerprint=d['fingerprint'],
        fits=records, partitions=partitions, calibration_budgets=budgets,
        all_finite=True, scope=scope))
    print('FIRST_FOLD_SCREENING_ASSEMBLED', predictions.shape, flush=True)


if __name__ == '__main__':
    with T.E.threadpool_limits(limits=1):
        assemble()
        scoring.analyze(first_fold=True)
        verification.verify(first_fold=True)
    print('FIRST_FOLD_SCREENING_COMPLETE', flush=True)
