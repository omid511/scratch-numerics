"""Complete-fold comparison; deterministic candidate is not replicated as fits."""
import argparse
import json
from pathlib import Path
import time
import neighborhood_reference as exp
np, frozen, helpers = exp.np, exp.frozen, exp.helpers
OUT, ROOT = exp.OUT, exp.ROOT
ARMS = ['learned_blend', 'neighborhood_blend']
FAMILIES = {'iid1': ['iid1_new949', 'iid1_new951'],
            'correlated1': ['correlated1_new949_950', 'correlated1_new951_952'],
            'iid5': ['iid5_new949', 'iid5_new951'],
            'correlated5': ['correlated5_new949_950', 'correlated5_new951_952']}


def paired(a, b, ids):
    return helpers.paired_rate_difference(a, b, ids)


def warnings(values, thresholds, folds, budget):
    if budget == 0:
        return values[..., 0] <= 0
    bi = frozen.BUDGETS.index(budget)
    return -values[..., 3] <= thresholds[:, folds, bi]


def warning_scores(warned, y, ids, slices):
    result = {}
    for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0),
                        ('stable_last1_warning', slices['stable0.01']), ('unstable_last1_miss', slices['unstable0.01'])]:
        events = warned[:, mask] if label.startswith('stable') else ~warned[:, mask]
        result[label] = {'rate': float(events.mean()), 'design_rate': frozen.interval_helpers.macro(events, ids[mask]),
                         'per_predictor': events.mean(axis=1).tolist(), 'physical_clips': int(mask.sum())}
    return result


def paired_warning(a, b, y, ids, slices):
    result = {}
    for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0),
                        ('unstable_last1_miss', slices['unstable0.01'])]:
        control, candidate = (a[:, mask], b[:, mask]) if label.startswith('stable') else (~a[:, mask], ~b[:, mask])
        result[label] = paired(control, candidate, ids[mask])
    return result


def analyze(smoke=False):
    path = OUT / ('smoke_predictions.npz' if smoke else 'predictions.npz')
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    y, ids, folds = data['y'], data['ids'], data['row_fold']
    names = data['conditions'].tolist()
    assert json.loads(str(data['fingerprint'])) == exp.signature(), 'stale experiment outputs'
    assert np.isfinite(data['candidate']).all() and np.isfinite(data['baseline']).all()
    p = [data['baseline'], data['candidate'][None]]
    thresholds = [data['baseline_thresholds'], data['candidate_thresholds'][None]]
    slices = frozen.interval_helpers.point_slices(y, ids)
    primary = ['clean', *helpers.FRESH]
    result = {'protocol': json.loads((OUT / 'protocol.json').read_text()), 'partial_smoke': smoke,
              'physical_clips': len(y), 'designs': len(np.unique(ids)), 'primary_conditions': primary,
              'baseline_predictors': 3, 'candidate_predictors': 1, 'conditions': {}, 'families': {}, 'folds': {},
              'limits': 'Previously scored draws; fixed-prediction2000 design bootstrap averages baseline seeds before resampling; no candidate seed replication, refit uncertainty or multiplicity correction.',
              'hashes': {'evaluation': exp.sha256(Path(__file__)), 'predictions': exp.sha256(path)}}
    for ci, name in enumerate(names):
        values = [v[:, ci] for v in p]
        entry = {'slices': {}, 'uncertainty': {}, 'warning': {}, 'warning_paired': {}}
        for label, mask in slices.items():
            a, b = [v[:, mask, 0] for v in values]
            entry['slices'][label] = {'physical_clips': int(mask.sum()), 'designs': len(np.unique(ids[mask])),
                'arms': {arm: {**helpers.point(y[mask], point, ids[mask]),
                    'per_predictor': [helpers.point(y[mask], q, ids[mask]) for q in point],
                    'signed_error_p99': np.quantile(point - y[mask], .99, axis=1).tolist()}
                         for arm, point in zip(ARMS, (a, b))}}
            if name in primary:
                entry['slices'][label]['candidate_minus_reference'] = helpers.paired_metrics(y[mask], a, b, ids[mask])
        for ai, arm in enumerate(ARMS):
            entry['uncertainty'][arm] = {}
            for label in ('all', 'stable0.01', 'unstable0.01', 'far_stable'):
                mask, v = slices[label], values[ai][:, slices[label]]
                entry['uncertainty'][arm][label] = {
                    'raw': frozen.interval_helpers.intervals(y[mask], v[:, :, :4], ids[mask]),
                    'group': frozen.interval_helpers.intervals(y[mask], v[:, :, [0, 4, 5, 3]], ids[mask]),
                    'probability': frozen.interval_helpers.probability_scores(y[mask], v[:, :, 3], ids[mask])}
            entry['warning'][arm] = {str(budget): warning_scores(warnings(values[ai], thresholds[ai], folds, budget), y, ids, slices)
                                      for budget in [0., *frozen.BUDGETS]}
        if name in primary:
            for budget in frozen.BUDGETS:
                a, b = [warnings(v, t, folds, budget) for v, t in zip(values, thresholds)]
                entry['warning_paired'][str(budget)] = paired_warning(a, b, y, ids, slices)
        result['conditions'][name] = entry
        print('NEIGHBOR_EVALUATION_CONDITION', name, {arm: round(entry['slices']['all']['arms'][arm]['design_mae'], 6) for arm in ARMS}, flush=True)
    for family, conditions in FAMILIES.items():
        indices = [names.index(n) for n in conditions]
        points = [np.concatenate([v[:, ci, :, 0] for ci in indices], axis=0) for v in p]
        probs = [np.concatenate([v[:, ci, :, 3] for ci in indices], axis=0) for v in p]
        warned = [np.concatenate([warnings(v[:, ci], t, folds, .05) for ci in indices], axis=0) for v, t in zip(p, thresholds)]
        entries = {'slices': {}, 'warning5': {}, 'mean': {}}
        for arm in ARMS:
            entries['mean'][arm] = {metric: float(np.mean([result['conditions'][n]['slices']['all']['arms'][arm][metric] for n in conditions]))
                                    for metric in ('design_mae', 'bias', 'over03', 'under05')}
        for label, mask in slices.items():
            entries['slices'][label] = helpers.paired_metrics(y[mask], points[0][:, mask], points[1][:, mask], ids[mask])
        entries['brier'] = paired((probs[0] - (y <= 0))**2, (probs[1] - (y <= 0))**2, ids)
        entries['warning5'] = paired_warning(warned[0], warned[1], y, ids, slices)
        result['families'][family] = entries
    checks = []
    for name in helpers.FRESH:
        for label, metric in [('all', 'design_mae'), ('stable0.01', 'design_mae'), ('unstable0.01', 'design_mae'),
                              ('far_stable', 'design_mae'), ('stable0.01', 'over_0.03'),
                              ('unstable0.01', 'over_0.03'), ('far_stable', 'under_0.05')]:
            interval = result['conditions'][name]['slices'][label]['candidate_minus_reference'][metric]['design_bootstrap_95']
            checks.append({'condition': name, 'slice': label, 'metric': metric, 'upper': interval[1], 'pass': bool(interval[1] <= 0)})
    result['strict_guardrails'] = {'passed': sum(c['pass'] for c in checks), 'checks': len(checks), 'all_pass': all(c['pass'] for c in checks),
                                    'details': checks, 'interpretation': 'Conservative zero-deterioration upper-bound screen, not formal noninferiority or deployment certification.'}
    for fold in np.unique(folds):
        selected = folds == fold
        result['folds'][str(fold)] = {name: {arm: helpers.point(y[selected], v[:, ci][:, selected, 0], ids[selected])
                                            for arm, v in zip(ARMS, p)} for ci, name in enumerate(names) if name in primary}
    result['named'] = {}
    for source in (3696, 5198):
        matches = np.flatnonzero(data['source_indices'] == source)
        if not len(matches):
            continue
        k, ci = int(matches[0]), names.index('iid1_new949')
        result['named'][str(source)] = {'margin': float(y[k]), 'baseline_points': p[0][:, ci, k, 0].tolist(),
            'neighborhood_point': float(p[1][0, ci, k, 0]), 'neighborhood_error': float(p[1][0, ci, k, 0] - y[k])}
    exp.save_json(OUT / ('smoke_results.json' if smoke else 'results.json'), result)
    print('NEIGHBOR_EVALUATION_COMPLETE', len(names), 'conditions', 'guardrails', result['strict_guardrails']['passed'], '/', len(checks), flush=True)
    return result


def timing():
    data, baseline, _ = exp.load_data()
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    proper, cal, held = frozen.split(ids, folds, 0)
    members = [exp.DesignNeighborhood(train[:, proper, :w], y[proper], ids[proper], rows[proper]) for w in (107, 174)]
    one, full = frozen.reference_predictors(42, 0, proper, cal, held)
    with np.load(OUT / 'fold0.npz') as z:
        radius = json.loads(str(z['manifest']))['calibration']
    with np.load(frozen.OUT / 'calibration_blend_frozen_seed42_fold0.npz') as z:
        old_radius = json.loads(str(z['calibration']))
    ci = names.index('iid1_new949')
    selected = held[np.random.default_rng(137).choice(len(held), 150, replace=False)]
    elapsed = {'learned_blend': [], 'neighborhood_blend': []}
    replay_delta = 0.
    with np.load(OUT / 'predictions.npz') as z:
        candidate = z['candidate']
    for k in selected:
        q = evaluation[ci, k:k+1]
        for arm in ARMS:
            def predict():
                point = .5 * (one(q[:, :107]) + full(q)) if arm == 'learned_blend' else .5 * (members[0].predict(q[:, :107])[0] + members[1].predict(q)[0])
                return frozen.distribution(point, old_radius if arm == 'learned_blend' else radius)[0]
            predict()
            start = time.perf_counter()
            observed = predict()
            elapsed[arm].append(1000 * (time.perf_counter() - start))
            reference = baseline[0, ci, k] if arm == 'learned_blend' else candidate[ci, k]
            np.testing.assert_allclose(observed, reference, atol=1e-12, rtol=1e-12)
            replay_delta = max(replay_delta, float(abs(observed-reference).max()))
    result = {'boundary': 'One cached174-feature row through both members and calibrated distribution; waveform normalization/modal/spatial extraction unchanged and excluded. Warm single-row timings; not full sensor-to-output latency.',
              'physical_queries': len(selected), 'condition': names[ci], 'fold': 0, 'baseline_seed': 42,
              'prediction_replay_max_delta': replay_delta, 'arms': {a: {'median_ms': float(np.median(v)), 'p95_ms': float(np.quantile(v, .95)), 'max_ms': float(max(v))} for a, v in elapsed.items()}}
    exp.save_npz(OUT / 'inference_measurements.npz', source_indices=rows[selected], **{a: np.array(v) for a, v in elapsed.items()})
    exp.save_json(OUT / 'inference_timing.json', result)
    print('NEIGHBOR_INFERENCE_TIMING', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with frozen.threadpool_limits(limits=1):
        analyze(args.smoke)
        if not args.smoke:
            timing()
