"""Separate margin uncertainty, soft identification and observation sufficiency."""
from experiment import *
from mechanics.p4_margin_estimation.decision_metrics import interval_score


def macro(values, ids):
    values = np.asarray(values).reshape(-1, len(ids)).mean(axis=0)
    return float(np.mean([values[ids == did].mean() for did in np.unique(ids)]))


def intervals(y, prediction, ids):
    point, lower, upper = prediction[:, :, 0], prediction[:, :, 1], prediction[:, :, 2]
    safe = lower > 0
    unsafe = y <= 0
    coverage = (y >= lower) & (y <= upper)
    return {'coverage': float(coverage.mean()), 'design_coverage': macro(coverage, ids),
            'width': float((upper - lower).mean()),
            'score': float(np.mean([interval_score(y, l, u, alpha=.1) for l, u in zip(lower, upper)])),
            'safe_availability': float(safe.mean()),
            'stable_safe_availability': float(safe[:, ~unsafe].mean()) if (~unsafe).any() else None,
            'unsafe_false_safe': float(safe[:, unsafe].mean()) if unsafe.any() else None,
            'unsafe_fraction_among_safe': float(np.sum(safe & unsafe) / np.sum(safe)) if safe.any() else None,
            'safe_prediction_exposures': int(safe.sum())}


def probability_scores(y, probabilities, ids):
    unsafe = (y <= 0).astype(float)
    p = np.clip(probabilities, 1e-15, 1 - 1e-15)  # Numerical scoring only; saved probabilities stay unchanged.
    brier = (probabilities - unsafe) ** 2
    loss = -(unsafe * np.log(p) + (1 - unsafe) * np.log1p(-p))
    bins = []
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (probabilities >= left) & (probabilities < right if right < 1 else probabilities <= right)
        if mask.any():
            bins.append({'range': [float(left), float(right)], 'prediction_exposures': int(mask.sum()),
                         'physical_clips': int(np.any(mask, axis=0).sum()),
                         'mean_probability': float(probabilities[mask].mean()),
                         'observed_unsafe': float(np.broadcast_to(unsafe, mask.shape)[mask].mean())})
    return {'brier': float(brier.mean()), 'design_brier': macro(brier, ids),
            'log_loss': float(loss.mean()), 'log_scoring_epsilon': 1e-15, 'reliability_bins': bins}


def alpha_metrics(truth, predicted, ids):
    error = predicted - truth
    return {'design_mae': macro(abs(error), ids), 'clip_mae': float(abs(error).mean()),
            'bias': float(error.mean()), 'wrong_sign': float((np.sign(predicted) != np.sign(truth)).mean())}


def point_slices(y, ids):
    masks = {'all': np.ones(len(y), bool), 'far_stable': y >= .15}
    for eps in (.01, .02, .05):
        masks[f'stable{eps}'] = (y > 0) & (y < eps)
        masks[f'unstable{eps}'] = (y < 0) & (y > -eps)
    return masks


def run_analysis():
    with np.load(OUT / 'predictions.npz') as z:
        data = {key: z[key] for key in z.files}
    p, truth, y, ids, folds = (data[k] for k in ('predictions', 'truth', 'y', 'ids', 'row_fold'))
    names, arms = data['conditions'].tolist(), data['arms'].tolist()
    signature = json.loads(str(data['fingerprint']))
    assert source_signature(signature['raw']) == signature, 'source/reference fingerprint changed'
    fits = json.loads((OUT / 'fits.json').read_text())
    assert all(f.get('success', f.get('prior_converged', False) and f.get('error_converged', False)) for f in fits['fits'])
    masks = point_slices(y, ids)
    results = {'protocol': json.loads((OUT / 'protocol.json').read_text()),
               'physical_designs': int(len(np.unique(ids))), 'physical_clips': len(y),
               'all_fits_converged': True, 'fits': len(fits['fits']), 'conditions': {},
               'oracle': {}, 'observation_diagnostics': {},
               'hashes': {k: sha256(OUT / v) for k, v in {'analysis': 'analyze_results.py',
                   'experiment': 'experiment.py', 'protocol': 'protocol.json', 'predictions': 'predictions.npz', 'fits': 'fits.json'}.items()}}
    for label, mask in masks.items():
        results['oracle'][label] = {
            'physical_clips': int(mask.sum()), 'designs': int(len(np.unique(ids[mask]))),
            'mixture_true_critical_pole': helpers.point(y[mask], data['oracle'][:, mask, 0], ids[mask]),
            'mlp_true_critical_pole': helpers.point(y[mask], data['diagnostics'][0, :, 0][:, mask], ids[mask])}
    fresh_names = helpers.FRESH
    for ci, name in enumerate(names):
        condition = {'slices': {}, 'uncertainty': {}, 'warning': {}, 'identification': {}, 'diagnostics': {}}
        corrected = p[:, :, ci].copy()
        for ai in range(len(arms)):
            delta = data['corrections'][ai][:, folds]
            corrected[ai, :, :, 1] -= delta
            corrected[ai, :, :, 2] += delta
        for label, mask in masks.items():
            target, designs = y[mask], ids[mask]
            candidate = p[:, :, ci][:, :, mask, 0]
            condition['slices'][label] = {'physical_clips': int(mask.sum()), 'designs': int(len(np.unique(designs))),
                'arms': {arm: helpers.point(target, candidate[ai], designs) for ai, arm in enumerate(arms)}}
            # Fresh draws and clean are primary comparisons; retain all descriptive outcomes.
            if name in fresh_names or name == 'clean':
                condition['slices'][label]['paired_joint_soft'] = {
                    ref: helpers.paired_metrics(target, candidate[arms.index(ref)], candidate[2], designs)
                    for ref in ('growth_only', 'joint_naive', 'one107', 'full174', 'blend50')}
        for ai, arm in enumerate(arms):
            record = {}
            for label in ('all', 'stable0.01', 'unstable0.01'):
                mask = masks[label]
                record[label] = {'raw': intervals(y[mask], p[ai, :, ci][:, mask], ids[mask]),
                                 'group_max_adjusted': intervals(y[mask], corrected[ai][:, mask], ids[mask]),
                                 'probability': probability_scores(y[mask], p[ai, :, ci, :, 3][:, mask], ids[mask])}
            condition['uncertainty'][arm] = record
            condition['warning'][arm] = {}
            for bi, budget in enumerate(BUDGETS):
                threshold = data['thresholds'][ai, :, :, bi][:, folds]
                warned = -p[ai, :, ci, :, 3] <= threshold
                entries = {}
                for label, mask in [('stable_all', y > 0), ('unstable_all', y <= 0),
                                    ('stable_last1', masks['stable0.01']), ('unstable_last1', masks['unstable0.01'])]:
                    events = warned[:, mask] if label.startswith('stable') else ~warned[:, mask]
                    entries[label] = {'rate': float(events.mean()), 'design_rate': macro(events, ids[mask]),
                                      'physical_clips': int(mask.sum()), 'designs': int(len(np.unique(ids[mask])))}
                condition['warning'][arm][str(budget)] = entries
        if name in fresh_names or name == 'clean':
            condition['warning_paired'] = {}
            for bi, budget in enumerate(BUDGETS):
                warned = [-p[ai, :, ci, :, 3] <= data['thresholds'][ai, :, :, bi][:, folds]
                          for ai in range(len(arms))]
                comparisons = {}
                for candidate in ('growth_only', 'joint_soft'):
                    comparisons[candidate] = {}
                    for reference in ('one107', 'full174', 'blend50'):
                        a, b = warned[arms.index(reference)], warned[arms.index(candidate)]
                        comparisons[candidate][reference] = {}
                        for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0),
                                            ('unstable_last1_miss', masks['unstable0.01'])]:
                            control, proposed = (a[:, mask], b[:, mask]) if label.startswith('stable') else (~a[:, mask], ~b[:, mask])
                            comparisons[candidate][reference][label] = helpers.paired_rate_difference(control, proposed, ids[mask])
                condition['warning_paired'][str(budget)] = comparisons
        alpha_candidates = {'measured': np.broadcast_to(data['measured'][ci, :, 0], (len(SEEDS), len(y))),
                            **{arm: p[ai, :, ci, :, 4] for ai, arm in enumerate(ARMS)}}
        for label, mask in masks.items():
            condition['identification'][label] = {
                key: alpha_metrics(truth[mask, 0], value[:, mask], ids[mask]) for key, value in alpha_candidates.items()}
            if name in fresh_names or name == 'clean':
                condition['identification'][label]['soft_minus_measured'] = helpers.paired_metrics(
                    truth[mask, 0], alpha_candidates['measured'][:, mask], alpha_candidates['joint_soft'][:, mask], ids[mask])
                condition['identification'][label]['soft_minus_growth_only'] = helpers.paired_metrics(
                    truth[mask, 0], alpha_candidates['growth_only'][:, mask], alpha_candidates['joint_soft'][:, mask], ids[mask])
        condition['diagnostics']['mlp_measured_pole'] = helpers.point(y, data['diagnostics'][1, :, ci], ids)
        condition['diagnostics']['measurement_error'] = {
            'alpha_bias': float(np.mean(data['measured'][ci, :, 0] - truth[:, 0])),
            'frequency_abs_error_hz': float(np.mean(abs(data['measured'][ci, :, 1] - truth[:, 1])))}
        results['conditions'][name] = condition
        print(name, 'designMAE', {a: round(condition['slices']['all']['arms'][a]['design_mae'], 6) for a in arms},
              'alphaMAE', {a: round(v['design_mae'], 4) for a, v in condition['identification']['all'].items() if isinstance(v, dict) and 'design_mae' in v and isinstance(v['design_mae'], (float, int))}, flush=True)
        if name in fresh_names or name == 'clean':
            time_usable = data['usable_time'][ci]
            signals = {'usable_seconds': time_usable, 'critical_cycles': truth[:, 1] * time_usable,
                       'critical_growth_aperture': abs(truth[:, 0]) * time_usable,
                       'lead_energy': data['energy'][ci], 'fit_error': data['fit_error'][ci], 'clamp_fraction': data['clamps']}
            diagnostic = {}
            for quantity, values in signals.items():
                cuts = np.quantile(values, [.25, .5, .75])
                labels = np.searchsorted(cuts, values, side='right')
                records = []
                for group in range(4):
                    mask = labels == group
                    if not mask.any():
                        continue
                    records.append({'bin': group, 'range': [float(values[mask].min()), float(values[mask].max())],
                        'physical_clips': int(mask.sum()), 'designs': int(len(np.unique(ids[mask]))),
                        'measured_growth': alpha_metrics(truth[mask, 0], alpha_candidates['measured'][:, mask], ids[mask]),
                        'soft_growth': alpha_metrics(truth[mask, 0], alpha_candidates['joint_soft'][:, mask], ids[mask]),
                        'soft_margin': helpers.point(y[mask], p[2, :, ci, :, 0][:, mask], ids[mask]),
                        'soft_interval': intervals(y[mask], p[2, :, ci][:, mask], ids[mask])})
                diagnostic[quantity] = {'quartile_edges': cuts.tolist(), 'bins': records}
            diagnostic['clamped_vs_unclamped'] = {}
            for label, mask in [('unclamped', data['clamps'] == 0), ('clamped', data['clamps'] > 0)]:
                if mask.any():
                    diagnostic['clamped_vs_unclamped'][label] = {
                        'physical_clips': int(mask.sum()), 'designs': int(len(np.unique(ids[mask]))),
                        'measured_growth': alpha_metrics(truth[mask, 0], alpha_candidates['measured'][:, mask], ids[mask]),
                        'soft_growth': alpha_metrics(truth[mask, 0], alpha_candidates['joint_soft'][:, mask], ids[mask]),
                        'soft_margin': helpers.point(y[mask], p[2, :, ci, :, 0][:, mask], ids[mask])}
            results['observation_diagnostics'][name] = diagnostic
    results['descriptive_guardrails'] = {}
    for ref in ('one107', 'full174', 'blend50'):
        checks = []
        for name in fresh_names:
            slices = results['conditions'][name]['slices']
            checks.extend(slices[label]['arms']['joint_soft']['design_mae'] <= slices[label]['arms'][ref]['design_mae']
                          for label in ('all', 'stable0.01', 'unstable0.01', 'far_stable'))
        results['descriptive_guardrails'][ref] = {'no_point_MAE_deterioration': bool(all(checks)),
                                                'passed': int(sum(checks)), 'checks': len(checks),
                                                'note': 'Descriptive point check, not formal noninferiority or automatic adoption.'}
    save_json(OUT / 'results.json', results)
    print('Completed physical uncertainty, identification and observation analysis', len(names), 'conditions', flush=True)


def timing():
    with np.load(OUT / 'physical_seed42_fold0.npz') as z:
        prior = cloudpickle.loads(z['model'].tobytes())
    with np.load(OUT / 'predictions.npz') as z:
        names = z['conditions'].tolist()
        query = z['measured'][names.index('iid1_new949')]
        candidates = np.flatnonzero(z['row_fold'] == 0)
    selected = candidates[np.linspace(0, len(candidates) - 1, 60, dtype=int)]
    measurements = np.empty((len(ARMS), len(selected)))
    for ai, arm in enumerate(ARMS):
        for row in selected[:5]:
            prior.predict(query[row:row + 1], arm)
        for j, row in enumerate(selected):
            start = time.perf_counter()
            prior.predict(query[row:row + 1], arm)
            measurements[ai, j] = 1000 * (time.perf_counter() - start)
    save_npz(OUT / 'inference_measurements.npz', durations_ms=measurements, rows=selected, arms=np.array(ARMS))
    result = {'boundary': 'Cached observed pole to distribution including exact mixture quantiles; excludes unchanged waveform extraction.',
              'arms': {a: {'median_ms': float(np.median(v)), 'p95_ms': float(np.quantile(v, .95)), 'max_ms': float(v.max())}
                       for a, v in zip(ARMS, measurements)},
              'hashes': {'measurements': sha256(OUT / 'inference_measurements.npz'), 'model': sha256(OUT / 'physical_seed42_fold0.npz')}}
    save_json(OUT / 'inference_timing.json', result)
    print('Observed distribution inference timing', json.dumps(result['arms']), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run_analysis()
        timing()
