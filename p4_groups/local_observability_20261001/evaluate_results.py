"""Paired boundary/decision evaluation and descriptive local-quality diagnostics."""
import time
from contract import *
from run_experiment import load_features
from prepare_features import measured_signal


def paired(a, b, ids):
    return helpers.paired_rate_difference(a, b, ids)


def run_analysis():
    with np.load(OUT / 'predictions.npz') as z:
        data = {k: z[k] for k in z.files}
    p, y, ids, folds = (data[k] for k in ('predictions', 'y', 'ids', 'row_fold'))
    names = data['conditions'].tolist()
    assert data['arms'].tolist() == ARMS and data['seeds'].tolist() == SEEDS
    fits = json.loads((OUT / 'fits.json').read_text())
    assert not fits['partial_smoke'] and fits['all_fits_converged'] and len(fits['fits']) == 90
    for split_record in fits['splits']:
        groups = [set(split_record[n + '_designs']) for n in ('proper', 'calibration', 'held')]
        assert all(not groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    masks = interval_helpers.point_slices(y, ids)
    primary = ['clean', *helpers.FRESH]
    result = {'protocol': json.loads((OUT / 'protocol.json').read_text()), 'physical_clips': len(y),
              'physical_designs': len(np.unique(ids)), 'new_member_fits': 90, 'all_fits_converged': True,
              'primary_conditions': primary, 'conditions': {}, 'fresh_mean': {}, 'fresh_paired_local': {}, 'strict_guardrails': {},
              'hashes': {f: sha256(OUT / f) for f in ('protocol.json', 'run_experiment.py', 'evaluate_results.py', 'predictions.npz', 'fits.json')}}
    for ci, name in enumerate(names):
        condition = {'slices': {}, 'uncertainty': {}, 'warning': {}}
        for label, mask in masks.items():
            record = {'physical_clips': int(mask.sum()), 'designs': len(np.unique(ids[mask])), 'arms': {}}
            for ai, arm in enumerate(ARMS):
                pred = p[ai, :, ci][:, mask, 0]
                record['arms'][arm] = {**helpers.point(y[mask], pred, ids[mask]),
                    'per_seed': [helpers.point(y[mask], v, ids[mask]) for v in pred],
                    'signed_p99_per_seed': np.quantile(pred - y[mask], .99, axis=1).tolist()}
            if name in primary:
                record['paired_local'] = {ref: helpers.paired_metrics(y[mask], p[ARMS.index(ref), :, ci][:, mask, 0],
                    p[2, :, ci][:, mask, 0], ids[mask]) for ref in ('blend_frozen', 'blend_global', 'blend_sham')}
                record['global_minus_frozen'] = helpers.paired_metrics(y[mask], p[0, :, ci][:, mask, 0], p[1, :, ci][:, mask, 0], ids[mask])
            condition['slices'][label] = record
        for ai, arm in enumerate(ARMS):
            condition['uncertainty'][arm] = {}
            for label in ('all', 'stable0.01', 'unstable0.01', 'far_stable'):
                mask = masks[label]
                v = p[ai, :, ci][:, mask]
                condition['uncertainty'][arm][label] = {
                    'raw': interval_helpers.intervals(y[mask], v[:, :, :4], ids[mask]),
                    'group': interval_helpers.intervals(y[mask], v[:, :, [0, 4, 5, 3]], ids[mask]),
                    'probability': interval_helpers.probability_scores(y[mask], v[:, :, 3], ids[mask])}
            condition['warning'][arm] = {}
            for bi, budget in enumerate([0., *BUDGETS]):
                warned = p[ai, :, ci, :, 0] <= 0 if bi == 0 else -p[ai, :, ci, :, 3] <= data['thresholds'][ai, :, :, bi - 1][:, folds]
                entries = {}
                for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0),
                                    ('stable_last1_warning', masks['stable0.01']), ('unstable_last1_miss', masks['unstable0.01'])]:
                    events = warned[:, mask] if label.startswith('stable') else ~warned[:, mask]
                    entries[label] = {'rate': float(events.mean()), 'design_rate': interval_helpers.macro(events, ids[mask]),
                                      'per_seed': events.mean(axis=1).tolist(), 'physical_clips': int(mask.sum()),
                                      'designs': len(np.unique(ids[mask]))}
                condition['warning'][arm][str(budget)] = entries
        if name in primary:
            condition['warning_paired_local'], condition['uncertainty_paired_local'] = {}, {}
            for bi, budget in enumerate(BUDGETS):
                warned = [-p[ai, :, ci, :, 3] <= data['thresholds'][ai, :, :, bi][:, folds] for ai in range(4)]
                comparisons = {}
                for ref in ('blend_frozen', 'blend_global', 'blend_sham'):
                    a, b = warned[ARMS.index(ref)], warned[2]
                    comparisons[ref] = {}
                    for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0), ('unstable_last1_miss', masks['unstable0.01'])]:
                        control, proposed = (a[:, mask], b[:, mask]) if label.startswith('stable') else (~a[:, mask], ~b[:, mask])
                        comparisons[ref][label] = paired(control, proposed, ids[mask])
                condition['warning_paired_local'][str(budget)] = comparisons
            for ref in ('blend_frozen', 'blend_global', 'blend_sham'):
                a, b = p[ARMS.index(ref), :, ci], p[2, :, ci]
                entries = {'brier': paired((a[:, :, 3] - (y <= 0)) ** 2, (b[:, :, 3] - (y <= 0)) ** 2, ids)}
                for label, lo, hi in [('raw', 1, 2), ('group', 4, 5)]:
                    entries[label] = {
                        'width': paired(a[:, :, hi] - a[:, :, lo], b[:, :, hi] - b[:, :, lo], ids),
                        'coverage': paired((y >= a[:, :, lo]) & (y <= a[:, :, hi]), (y >= b[:, :, lo]) & (y <= b[:, :, hi]), ids),
                        'stable_safe_availability': paired(a[:, y > 0, lo] > 0, b[:, y > 0, lo] > 0, ids[y > 0]),
                        'unsafe_false_safe': paired(a[:, y <= 0, lo] > 0, b[:, y <= 0, lo] > 0, ids[y <= 0])}
                condition['uncertainty_paired_local'][ref] = entries
        result['conditions'][name] = condition
        print(name, 'designMAE', {a: round(condition['slices']['all']['arms'][a]['design_mae'], 6) for a in ARMS}, flush=True)
    for ref in ('blend_frozen', 'blend_global', 'blend_sham'):
        checks = []
        for name in helpers.FRESH:
            slices = result['conditions'][name]['slices']
            for label, metric in [('all', 'design_mae'), ('stable0.01', 'design_mae'), ('unstable0.01', 'design_mae'),
                                  ('far_stable', 'design_mae'), ('stable0.01', 'over_0.03'), ('far_stable', 'under_0.05')]:
                ci = slices[label]['paired_local'][ref][metric]['design_bootstrap_95']
                checks.append({'condition': name, 'slice': label, 'metric': metric, 'upper': ci[1], 'pass': bool(ci[1] <= 0)})
        result['strict_guardrails'][ref] = {'passed': sum(c['pass'] for c in checks), 'checks': len(checks),
            'all_pass': all(c['pass'] for c in checks), 'details': checks,
            'interpretation': 'Conservative zero-deterioration upper-bound screen; not formal noninferiority or a deployment certificate.'}
    for family, names_family in [('iid1', ['iid1_new949', 'iid1_new951']), ('correlated1', ['correlated1_new949_950', 'correlated1_new951_952']),
                                 ('iid5', ['iid5_new949', 'iid5_new951']), ('correlated5', ['correlated5_new949_950', 'correlated5_new951_952'])]:
        result['fresh_mean'][family] = {a: {metric: float(np.mean([result['conditions'][n]['slices']['all']['arms'][a][metric] for n in names_family]))
                                          for metric in ('design_mae', 'bias', 'over03', 'under05')} for a in ARMS}
        indices = [names.index(n) for n in names_family]
        points = [np.concatenate([p[ai, :, ci, :, 0] for ci in indices]) for ai in range(4)]
        probabilities = [np.concatenate([p[ai, :, ci, :, 3] for ci in indices]) for ai in range(4)]
        warned = [np.concatenate([-p[ai, :, ci, :, 3] <= data['thresholds'][ai, :, :, BUDGETS.index(.05)][:, folds]
                                  for ci in indices]) for ai in range(4)]
        entries = {}
        for ref in ('blend_frozen', 'blend_global', 'blend_sham'):
            ri = ARMS.index(ref)
            entries[ref] = {'slices': {label: helpers.paired_metrics(y[mask], points[ri][:, mask], points[2][:, mask], ids[mask])
                                     for label, mask in masks.items()},
                           'brier': paired((probabilities[ri] - (y <= 0)) ** 2, (probabilities[2] - (y <= 0)) ** 2, ids),
                           'warning5': {}}
            for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0), ('unstable_last1_miss', masks['unstable0.01'])]:
                a, b = (warned[ri][:, mask], warned[2][:, mask]) if label.startswith('stable') else (~warned[ri][:, mask], ~warned[2][:, mask])
                entries[ref]['warning5'][label] = paired(a, b, ids[mask])
        result['fresh_paired_local'][family] = entries
    result['observability'] = diagnostics(p, y, ids, folds, names, masks)
    save_json(OUT / 'results.json', result)
    print('LOCAL_EVALUATION_COMPLETE', len(names), 'conditions;90 fits converged', flush=True)


def diagnostics(predictions, y, ids, folds, names, masks):
    with np.load(OUT / 'features.npz') as z:
        controls, local, training_local, training_controls = z['control'], z['local'], z['training_local'], z['training_control']
    result = {'availability': {}, 'local_rmse_quartiles': {}, 'interpretation':
              'Cutoffs use clean proper-fit features only, separately per outer fold. Held-design error associations are descriptive, not causal confidence or a deployment cutoff. Unavailable observations are retained.'}
    for ci, name in enumerate(names):
        result['availability'][name] = {label: {f'band{i+1}': {'available_fraction': float(controls[ci, mask, 6*i+2].mean()),
             'valid_center_fraction_mean': float(controls[ci, mask, 6*i+3].mean())} for i in range(2)}
             for label, mask in masks.items()}
    for band in range(2):
        assignments = np.full((len(names), len(y)), -1, dtype=int)
        cutoffs = []
        for fold in range(5):
            proper, cal, held = split(ids, folds, fold)
            available = training_controls[0, proper, 6*band+2] > 0
            values = training_local[0, proper, 6*band+5][available]
            edges = np.quantile(values, [.25, .5, .75])
            cutoffs.append({'fold': fold, 'edges': edges.tolist(), 'proper_available_clips': len(values)})
            for ci in range(len(names)):
                valid = controls[ci, held, 6*band+2] > 0
                assignments[ci, held[valid]] = np.searchsorted(edges, local[ci, held[valid], 6*band+5], side='right')
        records = {}
        for name in ['clean', *helpers.FRESH]:
            ci = names.index(name)
            entries = {}
            for label in ('all', 'stable0.01', 'unstable0.01'):
                groups = {}
                for quartile in (-1, 0, 1, 2, 3):
                    mask = masks[label] & (assignments[ci] == quartile)
                    groups[str(quartile)] = {'physical_clips': int(mask.sum()), 'designs': len(np.unique(ids[mask])),
                        'frozen_error': helpers.point(y[mask], predictions[0, :, ci][:, mask, 0], ids[mask]) if mask.any() else None}
                entries[label] = groups
            records[name] = entries
        result['local_rmse_quartiles'][f'band{band+1}'] = {'cutoffs': cutoffs, 'conditions': records}
    return result


def timing():
    (train, evaluation, y, ids, rows, names, folds, layout, pairs, dt), additions = load_features()
    proper, cal, held = split(ids, folds, 0)
    frozen = reference_predictors(42, 0, proper, cal, held)
    models = {'blend_frozen': frozen}
    for arm in ('blend_global', 'blend_local'):
        models[arm] = [helpers.base.trusted_predictor(OUT / f'{arm}_{width}_seed42_fold0.npz') for width in (107, 174)]
    radii = {}
    for arm in models:
        with np.load(OUT / f'calibration_{arm}_seed42_fold0.npz') as z:
            radii[arm] = json.loads(str(z['calibration']))
    selected = held[np.linspace(0, len(held) - 1, 60, dtype=int)]
    ci = names.index('iid1_new949')
    gains = np.array(json.loads((ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    raw_cache, signals = {}, []
    for k in selected:
        did = str(ids[k])
        if did not in raw_cache:
            with np.load(ROOT / 'p4_sensor_candidates' / f'{did}.npz') as z:
                raw_cache[did] = (z['raw'], z['noise_scale'], z['source_indices'])
        raw, sigma, source = raw_cache[did]
        j = int(np.flatnonzero(source == rows[k])[0])
        signals.append(measured_signal(raw[j], sigma[j], int(rows[k]), names[ci], layout, gains, float(dt[k]), {}, {}))
    extraction, durations = [], np.empty((3, len(selected)))
    def predict(q, signal, reported_dt, arm):
        if arm == 'blend_frozen':
            extras = np.empty(0)
        else:
            c, local = feature_row(signal, reported_dt, q[FREQUENCY_COLUMNS])
            extras = c if arm == 'blend_global' else np.r_[c, local]
        a, b = [model(np.r_[q[:width], extras][None])[0] for model, width in zip(models[arm], (107, 174))]
        return distribution(np.array([.5 * (a+b)]), radii[arm])
    for k, (signal, reported_dt) in zip(selected, signals):
        start = time.perf_counter()
        c, q = feature_row(signal, reported_dt, evaluation[ci, k, FREQUENCY_COLUMNS])
        extraction.append(1000 * (time.perf_counter() - start))
        np.testing.assert_allclose(c, additions[2][ci, k], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(q, additions[3][ci, k], rtol=1e-10, atol=1e-10)
    for ai, arm in enumerate(models):
        for k, (signal, reported_dt) in list(zip(selected, signals))[:5]:
            predict(evaluation[ci, k], signal, reported_dt, arm)
        for j, (k, (signal, reported_dt)) in enumerate(zip(selected, signals)):
            start = time.perf_counter()
            value = predict(evaluation[ci, k], signal, reported_dt, arm)
            durations[ai, j] = 1000 * (time.perf_counter() - start)
            assert np.isfinite(value).all()
    save_npz(OUT / 'inference_measurements.npz', durations_ms=durations, extraction_ms=extraction,
             source_indices=rows[selected], arms=list(models))
    summary = lambda v: {'median_ms': float(np.median(v)), 'p95_ms': float(np.quantile(v, .95)), 'max_ms': float(np.max(v))}
    value = {'boundary': 'Actual selected held iid1_new949 waveforms; calibrated/capped waveform and174 base features already available. Additions include two-band extraction, both refitted blend members and calibrated distribution. Frozen arm includes only both frozen members and distribution. Full original normalization/modal/spatial extraction excluded; sham is nondeployable and not timed.',
             'local_extraction': summary(extraction), 'arms': {a: summary(v) for a, v in zip(models, durations)},
             'hash': sha256(OUT / 'inference_measurements.npz')}
    save_json(OUT / 'inference_timing.json', value)
    print('LOCAL_INFERENCE_TIMING', json.dumps(value), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run_analysis()
        timing()
