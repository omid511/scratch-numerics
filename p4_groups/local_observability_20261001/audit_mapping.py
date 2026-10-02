"""Actual fit/held errors and proper-training neighborhoods, without fitting."""
from audit_failures import *
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
from scipy.spatial.distance import cdist

NFIELDS = ['nearest_rms_distance', 'nearest_margin', 'nearest_margin_minus_truth',
           'five_design_margin_range', 'five_design_margin_median',
           'nearest_distance_training_percentile']


def summary(pred, truth, design):
    error = np.asarray(pred) - truth
    return {'rows': len(truth), 'designs': len(np.unique(design)),
            'design_mae': float(np.mean([abs(error[design == d]).mean() for d in np.unique(design)])),
            'bias': float(error.mean()), 'optimism_gt03': float(np.mean(error > .03)),
            'signed_error_p99': float(np.quantile(error, .99))}


def masks(truth):
    return {'all': np.ones(len(truth), bool), 'stable_last1pct': (truth > 0) & (truth < .01),
            'unstable_last1pct': (truth < 0) & (truth > -.01), 'far_stable': truth >= .15}


def cohort(data, recorded):
    y, ids, rows, names = data[2:6]
    events = recorded - y > .03
    primary = np.array([name in ['clean', *helpers.FRESH] for name in names])
    result = {'definition': 'prediction - truth > .03; strict inequality', 'scopes': {}}
    for scope, cond in [('all37', np.ones(len(names), bool)), ('primary9', primary)]:
        result['scopes'][scope] = {}
        for label, ym in masks(y).items():
            selected = events & cond[None, None, :, None] & ym[None, None, None, :]
            acquisitions = selected.any(axis=(0, 1))
            clips = acquisitions.any(axis=0)
            result['scopes'][scope][label] = {'exposures': int(selected.sum()),
                'condition_clips': int(acquisitions.sum()), 'physical_clips': int(clips.sum()),
                'designs': len(np.unique(ids[clips])),
                'per_arm_exposures': dict(zip(ARMS[:3], selected.sum(axis=(1, 2, 3)).astype(int).tolist()))}
    return result, events.any(axis=(0, 1))


def fitting(data, additions, recorded):
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    tc, tl, ec, el = additions
    original = [names.index(n) for n in helpers.base.TRAIN_NAMES]
    np.testing.assert_array_equal(train, evaluation[original])
    result = {'folds': [], 'aggregate': {}, 'named_seen_design': []}
    pooled = {a: {'proper_fit': [[], [], []], 'held_same_acquisitions': [[], [], []]} for a in ARMS[:3]}
    for fold in range(5):
        proper, cal, held = split(ids, folds, fold)
        for si, seed in enumerate(SEEDS):
            for ai, arm in enumerate(ARMS[:3]):
                models = predictors(arm, seed, fold, proper, cal, held)
                record = {'fold': fold, 'seed': seed, 'arm': arm, 'slices': {}}
                for label, selected in [('proper_fit', proper), ('held_same_acquisitions', held)]:
                    pred, _ = predict(models, train, tc, tl, selected, arm)
                    truth, design = np.tile(y[selected], 4), np.tile(ids[selected], 4)
                    if label == 'held_same_acquisitions':
                        np.testing.assert_allclose(pred.reshape(4, -1), recorded[ai, si, original][:, held], atol=1e-12, rtol=1e-12)
                    for target, value in zip(pooled[arm][label], (pred, truth, design)):
                        target.append(value)
                    record['slices'][label] = {s: summary(pred[m], truth[m], design[m]) for s, m in masks(truth).items()}
                result['folds'].append(record)
                for source in (3696, 5198):
                    k = int(np.flatnonzero(rows == source)[0])
                    if k not in proper:
                        continue
                    ci = names.index('iid1_new949')
                    point, members = predict(models, evaluation[ci:ci+1], ec[ci:ci+1], el[ci:ci+1], np.array([k]), arm)
                    result['named_seen_design'].append({'source': source, 'seed': seed, 'model_fold': fold, 'arm': arm,
                        'condition': names[ci], 'point': float(point[0]), 'error': float(point[0] - y[k]),
                        'members': members[:, 0].tolist()})
        print('AUDIT_ACTUAL_FITTING_FOLD', fold, flush=True)
    for arm, scopes in pooled.items():
        result['aggregate'][arm] = {}
        for label, arrays in scopes.items():
            pred, truth, design = map(np.concatenate, arrays)
            result['aggregate'][arm][label] = {s: summary(pred[m], truth[m], design[m]) for s, m in masks(truth).items()}
    result['aggregation'] = 'Pooled actual row predictions; proper designs repeat across outer models and seeds, held rows across seeds. No independence or uncertainty claim.'
    save_json(OUT / 'audit_fitting_results.json', result)
    return result


def distances(q, r, n):
    # Four training views collapse to the closest view of each physical clip.
    return cdist(q, r, metric='sqeuclidean').reshape(len(q), 4, n).min(axis=1)


def neighborhoods(data, additions, selected):
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    tc, tl, ec, el = additions
    ci, ki = np.nonzero(selected)
    values = np.empty((3, 2, len(ki), len(NFIELDS)))
    references = np.empty((3, 2, len(ki), 5), dtype=int)
    reference_distances = np.empty_like(references, dtype=float)
    supports = []
    for fold in range(5):
        proper, cal, held = split(ids, folds, fold)
        assert not set(ids[proper]) & set(ids[held])
        picked = np.flatnonzero(folds[ki] == fold)
        designs = np.unique(ids[proper])
        design_rows = [np.flatnonzero(ids[proper] == d) for d in designs]
        for ai, arm in enumerate(ARMS[:3]):
            for wi, width in enumerate((107, 174)):
                ref = inputs(train, tc, tl, proper, width, arm)
                mean, scale, active = fit_ridge_scaling(ref)
                ref = ((ref - mean) / scale)[:, active]
                assert ref.shape[1] > 0
                normalizer = ref.shape[1]
                # Empirical support scale: clean proper observations vs other proper designs.
                baseline = []
                for start in range(0, len(proper), 64):
                    idx = np.arange(start, min(start + 64, len(proper)))
                    distance = distances(ref[idx], ref, len(proper))
                    distance[ids[proper[idx]][:, None] == ids[proper][None]] = np.inf
                    baseline.extend(np.sqrt(distance.min(axis=1) / normalizer))
                baseline = np.sort(baseline)
                supports.append({'fold': fold, 'arm': arm, 'width': width, 'active_features': normalizer,
                                 'proper_designs': len(designs), 'leave_own_design_distance_quantiles': np.quantile(baseline, [0, .25, .5, .75, .95, 1]).tolist()})
                for start in range(0, len(picked), 64):
                    positions = picked[start:start+64]
                    q = evaluation[ci[positions], ki[positions], :width]
                    if arm != 'blend_frozen':
                        q = np.column_stack([q, ec[ci[positions], ki[positions]]])
                        if arm == 'blend_local':
                            q = np.column_stack([q, el[ci[positions], ki[positions]]])
                    q = ((q - mean) / scale)[:, active]
                    distance = distances(q, ref, len(proper))
                    nearest = distance.argmin(axis=1)
                    design_nearest = np.stack([r[distance[:, r].argmin(axis=1)] for r in design_rows], axis=1)
                    design_distances = np.take_along_axis(distance, design_nearest, axis=1)
                    five = np.argsort(design_distances, axis=1)[:, :5]
                    nn = np.take_along_axis(design_nearest, five, axis=1)
                    nearest_distance = np.sqrt(distance[np.arange(len(q)), nearest] / normalizer)
                    margin = y[proper[nn]]
                    values[ai, wi, positions] = np.column_stack([nearest_distance, y[proper[nearest]],
                        y[proper[nearest]] - y[ki[positions]], np.ptp(margin, axis=1), np.median(margin, axis=1),
                        np.searchsorted(baseline, nearest_distance, side='right') / len(baseline)])
                    references[ai, wi, positions] = rows[proper[nn]]
                    reference_distances[ai, wi, positions] = np.sqrt(np.take_along_axis(distance, nn, axis=1) / normalizer)
        print('AUDIT_NEIGHBORHOOD_FOLD', fold, len(picked), flush=True)
    assert np.isfinite(values).all()
    result = {'fields': NFIELDS, 'query_acquisitions': len(ki), 'supports': supports, 'named': {}, 'populations': {}}
    for source in (3696, 5198):
        hits = np.flatnonzero((rows[ki] == source) & (np.array(names)[ci] == 'iid1_new949'))
        assert len(hits) == 1
        j = int(hits[0])
        result['named'][str(source)] = {}
        for ai, arm in enumerate(ARMS[:3]):
            for wi, width in enumerate((107, 174)):
                refs = references[ai, wi, j]
                idx = np.searchsorted(rows, refs)
                np.testing.assert_array_equal(rows[idx], refs)
                result['named'][str(source)][f'{arm}:{width}'] = {'statistics': dict(zip(NFIELDS, values[ai, wi, j].tolist())),
                    'references': [{'source': int(r), 'design': str(ids[k]), 'margin': float(y[k]), 'distance': float(d)}
                                   for r, k, d in zip(refs, idx, reference_distances[ai, wi, j])]}
    primary = np.array([names[i] in ['clean', *helpers.FRESH] for i in ci])
    for scope, condition_mask in [('all37', np.ones(len(ki), bool)), ('primary9', primary)]:
        for label, margin_mask in masks(y[ki]).items():
            mask = condition_mask & margin_mask
            result['populations'][f'{scope}:{label}'] = {'acquisitions': int(mask.sum()), 'representations': {}}
            if not mask.any():
                continue
            for ai, arm in enumerate(ARMS[:3]):
                for wi, width in enumerate((107, 174)):
                    block = values[ai, wi, mask]
                    result['populations'][f'{scope}:{label}']['representations'][f'{arm}:{width}'] = {
                        field: {'median': float(np.median(block[:, j])), 'p95': float(np.quantile(block[:, j], .95))}
                        for j, field in enumerate(NFIELDS)}
    save_npz(OUT / 'audit_neighborhoods.npz', values=values, fields=NFIELDS, condition_indices=ci, clip_indices=ki,
             references=references, reference_distances=reference_distances, source_indices=rows, conditions=names)
    save_json(OUT / 'audit_neighborhood_results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitting-only', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        data, additions, recorded = load_audit_data()
        result, selected = cohort(data, recorded)
        save_json(OUT / 'audit_cohort.json', result)
        fitting(data, additions, recorded)
        if not args.fitting_only:
            neighborhoods(data, additions, selected)
        print('AUDIT_MAPPING_COMPLETE', json.dumps(result['scopes']), flush=True)
