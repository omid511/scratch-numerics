"""Actual raw-waveform inference and saved ranking objective/calibration replay."""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ranking as R
from threadpoolctl import threadpool_limits
C, np, OUT, E = R.C, R.np, R.OUT, R.E
spec = importlib.util.spec_from_file_location('ranking_acquisition', C.OUT / 'prepare_features.py')
acquisition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquisition)


def verify():
    d = E.setup()
    fits = json.loads((OUT / 'fits.json').read_text())
    with np.load(OUT / 'predictions.npz') as z:
        assert json.loads(str(z['fingerprint'])) == R.signature() == fits['fingerprint']
        p, t = z['predictions'][:, 0], z['thresholds'][:, 0]
        assert z['fields'].tolist() == R.FIELDS and z['arms'].tolist() == R.ARMS
        for key, value in [('y', d['y']), ('ids', d['ids']), ('source_indices', d['rows']), ('row_fold', d['folds'])]:
            np.testing.assert_array_equal(z[key], value)
    assert fits['all_converged'] and len(fits['fits']) == 15 and np.isfinite(p).all() and np.isfinite(t).all()
    np.testing.assert_array_equal(p[0], d['baseline'])
    np.testing.assert_array_equal(t[0], d['baseline_thresholds'])
    for arm_prediction in p:
        np.testing.assert_array_equal(arm_prediction[..., [0,1,2,4,5]], d['baseline'][..., [0,1,2,4,5]])
    models, radii, partitions, loss_proofs = {}, {}, [], []
    max_calibration_error = 0.
    for fold in range(5):
        proper, cal, held = C.split(d['ids'], d['folds'], fold)
        sets = [set(d['ids'][rows]) for rows in (proper, cal, held)]
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
        with np.load(OUT / f'fold{fold}.npz') as z:
            assert json.loads(str(z['fingerprint'])) == fits['fingerprint']
            for key, rows in [('proper_rows', proper), ('calibration_rows', cal), ('held_rows', held)]:
                np.testing.assert_array_equal(z[key], rows)
            heads = C.cloudpickle.loads(z['heads'].tobytes())
            recorded = z['calibration_scores']
            fold_records = json.loads(str(z['records']))
        with np.load(E.OLD / f'fusion_seed42_fold{fold}.npz') as z:
            E._verify_archived_contract(json.loads(str(z['fingerprint']))['frozen'],
                                        fits['fingerprint']['parent']['coordinates']['contract'])
            tc = z['train_control'].reshape(-1, 7)
        tq = d['train'][:, proper].reshape(-1, 174); target = np.tile(d['y'][proper], 4)
        for arm in R.ARMS[1:]:
            x = R.features(tc, tq, arm); head = heads[arm]
            mean, scale, active = R.fit_ridge_scaling(x)
            for key, value in [('mean',mean), ('scale',scale), ('active',active)]:
                np.testing.assert_array_equal(head[key], value)
            function, cohort = R.objective(R.standardized(head,x), -tc[:,0]/R.UNIT, target)
            record = next(r for r in fold_records if r['arm'] == arm)
            assert record == next(r for r in fits['fits'] if r['arm'] == arm and r['fold'] == fold)
            np.testing.assert_allclose(function(head['coef'])[0], record['loss'], atol=1e-12, rtol=1e-12)
            assert all(record[key] == value for key,value in cohort.items())
            loss_proofs.append(dict(fold=fold, arm=arm, reloaded_loss=function(head['coef'])[0]))
        one, full = C.reference_predictors(42, fold, proper, cal, held)
        models[fold] = (one, full, heads)
        q = d['train'][:, cal].reshape(-1, 174)
        control = E.control_features(q, np.column_stack((one(q[:,:107]), full(q))))
        cy, cids = np.tile(d['y'][cal],4), np.tile(d['ids'][cal],4)
        radius = C.calibration(control[:,0], cy, cids); radii[fold] = radius
        actual = [C.distribution(control[:,0], radius)[:,3]]
        actual += [R.score(heads[arm],control,q,arm) for arm in R.ARMS[1:]]
        actual = np.array(actual)
        np.testing.assert_allclose(actual, recorded, atol=1e-12, rtol=1e-12)
        max_calibration_error = max(max_calibration_error,float(abs(actual-recorded).max()))
        observed = []
        for ai, scores in enumerate(actual):
            levels = [C.threshold(scores,cy,b) for b in C.BUDGETS]
            np.testing.assert_array_equal(levels,t[ai,fold])
            rates = [float(np.mean(-scores[cy>0]<=level)) for level in levels]
            assert np.all(np.array(rates)<=C.BUDGETS)
            observed.append(rates)
        partitions.append(dict(fold=fold, proper_designs=len(sets[0]), calibration_designs=len(sets[1]),
                               held_designs=len(sets[2]), calibration_false_warnings=observed))
    wanted = json.loads((R.PARENT / 'verification.json').read_text())['replays']
    gains = np.array(json.loads((R.ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    with np.load(C.OUT / 'features.npz') as z:
        raw_hashes = json.loads(str(z['raw_hashes']))
    B = C.helpers.base; ridge = B.PhysicsFeatureRidge()
    replays, max_features, max_outputs = [], 0., 0.
    for request in wanted:
        source, name = request['source_index'], request['condition']
        match = np.flatnonzero(d['rows']==source); assert len(match)==1
        k = int(match[0]); fold = int(d['folds'][k]); did = str(d['ids'][k]); ci = d['names'].index(name)
        one, full, heads = models[fold]
        path = R.ROOT / 'p4_sensor_candidates' / f'{did}.npz'
        assert C.sha256(path)==raw_hashes[did]
        with np.load(path) as z:
            j = int(np.flatnonzero(z['source_indices']==source)[0]); raw, sigma = z['raw'][j], z['noise_scale'][j]
        signal, dt = acquisition.measured_signal(raw,sigma,source,name,d['layout'],gains,float(d['dt'][k]),{},{})
        modes = B.estimate_modes(signal,dt,include_shapes=True)
        q = np.r_[B.base_feature_row(signal,dt,modes,ridge), B.modal_spatial_row(modes,d['pairs']),
                  B.modal_spatial_addition_row(modes,d['pairs'])][None]
        feature_error = float(abs(q-d['evaluation'][ci,k]).max())
        np.testing.assert_allclose(q[0],d['evaluation'][ci,k],atol=1e-9,rtol=1e-9)
        control = E.control_features(q,np.column_stack((one(q[:,:107]),full(q))))
        frozen = C.distribution(control[:,0],radii[fold])[0]
        actual = np.tile(frozen,(len(R.ARMS),1))
        for ai,arm in enumerate(R.ARMS[1:],1):
            actual[ai,3] = R.score(heads[arm],control,q,arm)[0]
        error = float(abs(actual-p[:,ci,k]).max())
        np.testing.assert_allclose(actual,p[:,ci,k],atol=1e-8,rtol=1e-8)
        max_features = max(max_features,feature_error); max_outputs = max(max_outputs,error)
        replays.append(dict(source_index=source,condition=name,design=did,max_feature_error=feature_error,max_output_error=error))
        print('WARNING_RANK_WAVEFORM_REPLAY',source,name,feature_error,error,flush=True)
    assert len(loss_proofs)==15 and len(replays)==15
    proof = dict(all_predictions_finite=True, frozen_margin_and_intervals_exact=True,
        all_partitions_disjoint=True, all_calibration_budgets_respected=True, fitted_heads=15,
        objective_replays=loss_proofs, partitions=partitions, real_waveform_replays=len(replays),replays=replays,
        max_feature_error=max_features,max_output_error=max_outputs,max_calibration_error=max_calibration_error,
        predictions_sha256=C.sha256(OUT / 'predictions.npz'), warning_score_is_uncalibrated=True,
        limitation='One initialization, repeated TRAIN exploration, no refit/multiplicity guarantee.')
    C.save_json(OUT / 'verification.json',proof)
    print('WARNING_RANK_VERIFICATION_COMPLETE',len(replays),'actual raw waveform replays; frozen outputs exact',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
