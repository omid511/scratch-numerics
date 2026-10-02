"""Fixed TRAIN-only observability addition to both frozen-blend representations."""
import argparse
from contract import *
from local_features import numerical_smoke


def load_features():
    data = load_data()
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    with np.load(OUT / 'features.npz') as z:
        for key, value in [('y', y), ('ids', ids), ('source_indices', rows), ('row_fold', folds)]:
            np.testing.assert_array_equal(z[key], value)
        assert z['conditions'].tolist() == names
        assert json.loads(str(z['fingerprint'])) == signature(), 'stale descriptor table'
        controls, local = z['control'], z['local']
        train_controls, train_local = z['training_control'], z['training_local']
        raw_hashes = json.loads(str(z['raw_hashes']))
    for did, digest in raw_hashes.items():
        assert sha256(ROOT / 'p4_sensor_candidates' / f'{did}.npz') == digest, 'raw acquisition changed'
    return data, (train_controls, train_local, controls, local)


def query(base, controls, local, rows, width, arm, permutation):
    parts = [base[:, rows, :width], controls[:, rows]]
    if arm == 'blend_local':
        parts.append(local[:, rows])
    elif arm == 'blend_sham':
        parts.append(local[:, rows[permutation]])
    elif arm != 'blend_global':
        raise ValueError(f'unknown refitted arm: {arm}')
    features = np.concatenate(parts, axis=2)
    return features.reshape(-1, features.shape[-1])


def fitted(path, x, y, seed, fingerprint, proper, cal, held):
    arrays = {'proper_rows': proper, 'calibration_rows': cal, 'held_rows': held}
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['fingerprint'])) == fingerprint, 'stale observability fit'
            for key, value in arrays.items():
                np.testing.assert_array_equal(z[key], value)
            predict, meta = cloudpickle.loads(z['predictor'].tobytes()), json.loads(str(z['metadata']))
    else:
        predict, meta = fit_network(x, y, seed=seed, alpha=1., max_iter=10000, max_fun=200000)
        save_npz(path, predictor=np.frombuffer(cloudpickle.dumps(predict), dtype=np.uint8),
                 metadata=json.dumps(meta), fingerprint=json.dumps(fingerprint), **arrays)
        print('Fit observability', path.name, json.dumps(meta), flush=True)
    if not meta['success']:
        raise RuntimeError(f'nonconverged observability fit: {path}: {meta}')
    return predict, meta


def run(smoke=False):
    (train, evaluation, y, ids, rows, names, folds, layout, pairs, dt), additions = load_features()
    tc, tl, ec, el = additions
    old_path = ROOT / 'p4_groups/physical_joint_inference_20261001/predictions.npz'
    with np.load(old_path) as z:
        for key, value in [('y', y), ('ids', ids), ('source_indices', rows), ('row_fold', folds)]:
            np.testing.assert_array_equal(z[key], value)
        assert z['conditions'].tolist() == names
        recorded = z['predictions'][z['arms'].tolist().index('blend50'), :, :, :, 0]
    fingerprint = {**signature(), 'feature_table': sha256(OUT / 'features.npz'), 'frozen_point_record': sha256(old_path)}
    seeds = SEEDS[:1] if smoke else SEEDS
    predictions = np.full((4, len(seeds), len(names), len(y), 6), np.nan)
    levels = np.full((4, len(seeds), 5, len(BUDGETS)), np.nan)
    records, splits = [], []
    for fold in (range(1) if smoke else range(5)):
        proper, cal, held = split(ids, folds, fold)
        partition = {'fold': fold}
        for label, idx in [('proper', proper), ('calibration', cal), ('held', held)]:
            partition[label + '_rows'] = idx.tolist()
            partition[label + '_designs'] = np.unique(ids[idx]).tolist()
        splits.append(partition)
        permutations = [np.random.default_rng(3047 + 3 * fold + i).permutation(len(idx)) for i, idx in enumerate((proper, cal, held))]
        for si, seed in enumerate(seeds):
            one, full = reference_predictors(seed, fold, proper, cal, held)
            qcal, qheld = train[:, cal].reshape(-1, 174), evaluation[:, held].reshape(-1, 174)
            frozen_cal = .5 * (one(qcal[:, :107]) + full(qcal))
            frozen_held = .5 * (one(qheld[:, :107]) + full(qheld))
            np.testing.assert_allclose(frozen_held.reshape(len(names), len(held)), recorded[SEEDS.index(seed)][:, held], atol=1e-12, rtol=1e-12)
            for ai, arm in enumerate(ARMS):
                if arm == 'blend_frozen':
                    cp, hp = frozen_cal, frozen_held
                else:
                    members_cal, members_held = [], []
                    for width in (107, 174):
                        x = query(train, tc, tl, proper, width, arm, permutations[0])
                        predict, meta = fitted(OUT / f'{arm}_{width}_seed{seed}_fold{fold}.npz', x, np.tile(y[proper], 4), seed,
                                               fingerprint, proper, cal, held)
                        members_cal.append(predict(query(train, tc, tl, cal, width, arm, permutations[1])))
                        members_held.append(predict(query(evaluation, ec, el, held, width, arm, permutations[2])))
                        records.append({'arm': arm, 'width': width, 'fold': fold, 'seed': seed, **meta,
                                        'model_hash': sha256(OUT / f'{arm}_{width}_seed{seed}_fold{fold}.npz')})
                    cp, hp = .5 * (members_cal[0] + members_cal[1]), .5 * (members_held[0] + members_held[1])
                ctarget, cids = np.tile(y[cal], 4), np.tile(ids[cal], 4)
                radius = calibration(cp, ctarget, cids)
                cp_dist, hp_dist = distribution(cp, radius), distribution(hp, radius)
                predictions[ai, si][:, held] = hp_dist.reshape(len(names), len(held), 6)
                levels[ai, si, fold] = [threshold(cp_dist[:, 3], ctarget, b) for b in BUDGETS]
                save_npz(OUT / f'calibration_{arm}_seed{seed}_fold{fold}.npz', calibration=json.dumps(radius),
                         thresholds=levels[ai, si, fold], fingerprint=json.dumps(fingerprint),
                         proper_rows=proper, calibration_rows=cal, held_rows=held)
            print('Completed observability fold', fold, 'seed', seed, flush=True)
    selected = np.flatnonzero(folds == 0) if smoke else np.arange(len(y))
    assert np.isfinite(predictions[:, :, :, selected]).all()
    assert np.isfinite(levels[:, :, :1] if smoke else levels).all()
    assert {k: fingerprint[k] for k in signature()} == signature(), 'source/reference changed during fit'
    save_npz(OUT / ('smoke_predictions.npz' if smoke else 'predictions.npz'),
             predictions=predictions[:, :, :, selected], thresholds=levels, budgets=BUDGETS,
             y=y[selected], ids=ids[selected], source_indices=rows[selected], row_fold=folds[selected],
             conditions=names, arms=ARMS, seeds=seeds, fields=FIELDS, fingerprint=json.dumps(fingerprint))
    save_json(OUT / ('smoke_fits.json' if smoke else 'fits.json'),
              {'splits': splits, 'fits': records, 'fingerprint': fingerprint, 'partial_smoke': smoke,
               'all_fits_converged': all(r['success'] for r in records)})
    print('LOCAL_SMOKE_PASSED' if smoke else 'LOCAL_EXPERIMENT_COMPLETE', predictions[:, :, :, selected].shape,
          'frozen point replay passed; proper/calibration/held designs disjoint', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.smoke:
            numerical_smoke()
        run(args.smoke)
