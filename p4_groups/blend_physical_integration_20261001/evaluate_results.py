"""Paired accuracy/risk/availability evaluation of the incremental blend readouts."""
import importlib.util
import json
from pathlib import Path
import time
from scipy.special import ndtr  # Frozen physical shards resolve this ufunc through __main__.
import run_experiment as exp
np, OUT, ROOT = exp.np, exp.OUT, exp.ROOT
helpers = exp.physical.helpers
spec = importlib.util.spec_from_file_location('physical_interval_analysis', exp.PRIOR / 'analyze_results.py')
interval_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interval_helpers)
intervals, probability_scores, macro = interval_helpers.intervals, interval_helpers.probability_scores, interval_helpers.macro


def paired(control, candidate, ids):
    return helpers.paired_rate_difference(control, candidate, ids)


def run_analysis():
    with np.load(OUT / 'predictions.npz') as z:
        values = {key: z[key] for key in z.files}
    p, y, ids, folds = (values[k] for k in ('predictions', 'y', 'ids', 'row_fold'))
    names, arms = values['conditions'].tolist(), values['arms'].tolist()
    fits = json.loads((OUT / 'fits.json').read_text())
    assert not fits['partial_smoke']
    component_records = [f for f in fits['fits'] if f.get('kind') != 'heads']
    assert len(component_records) == 45
    assert all(f['one']['success'] and f['full']['success'] and f['physical']['prior_converged'] and f['physical']['error_converged'] for f in component_records)
    masks = interval_helpers.point_slices(y, ids)
    fresh = helpers.FRESH
    primary = ['clean', *fresh]
    additions = ['blend_joint', 'blend_soft', 'blend_both']
    references = ['blend_frozen', 'blend_adaptive', 'blend_affine', 'blend_control', 'blend_sham']
    result = {'protocol': json.loads((OUT / 'protocol.json').read_text()), 'physical_clips': len(y),
              'physical_designs': len(np.unique(ids)), 'component_partitions': len(component_records),
              'base_fits': 2 * len(component_records), 'physical_fits': len(component_records),
              'all_component_fits_converged': True, 'primary_conditions': primary, 'conditions': {},
              'fresh_mean': {}, 'strict_guardrails': {},
              'hashes': {key: exp.sha256(path) for key, path in {
                  'evaluation': Path(__file__), 'execution': OUT / 'run_experiment.py', 'protocol': OUT / 'protocol.json',
                  'predictions': OUT / 'predictions.npz', 'fits': OUT / 'fits.json'}.items()}}
    for ci, name in enumerate(names):
        condition = {'slices': {}, 'uncertainty': {}, 'warning': {}}
        for label, mask in masks.items():
            q, target, designs = p[:, :, ci][:, :, mask, 0], y[mask], ids[mask]
            record = {'physical_clips': int(mask.sum()), 'designs': len(np.unique(designs)),
                      'arms': {arm: helpers.point(target, q[ai], designs) for ai, arm in enumerate(arms)}}
            if name in primary:
                record['paired'] = {arm: {ref: helpers.paired_metrics(target, q[arms.index(ref)], q[arms.index(arm)], designs)
                                         for ref in references} for arm in additions}
                record['generic_control_minus_frozen'] = helpers.paired_metrics(target, q[0], q[3], designs)
            condition['slices'][label] = record
        for ai, arm in enumerate(arms):
            condition['uncertainty'][arm] = {}
            for label in ('all', 'stable0.01', 'unstable0.01', 'far_stable'):
                mask = masks[label]
                v = p[ai, :, ci][:, mask]
                condition['uncertainty'][arm][label] = {
                    'raw': intervals(y[mask], v[:, :, :4], ids[mask]),
                    'group': intervals(y[mask], v[:, :, [0, 4, 5, 3]], ids[mask]),
                    'probability': probability_scores(y[mask], v[:, :, 3], ids[mask])}
            condition['warning'][arm] = {}
            for bi, budget in enumerate(exp.BUDGETS):
                level = values['thresholds'][ai, :, :, bi][:, folds]
                warned = -p[ai, :, ci, :, 3] <= level
                condition['warning'][arm][str(budget)] = {}
                for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0),
                                    ('stable_last1_warning', masks['stable0.01']), ('unstable_last1_miss', masks['unstable0.01'])]:
                    events = warned[:, mask] if label.startswith('stable') else ~warned[:, mask]
                    condition['warning'][arm][str(budget)][label] = {'rate': float(events.mean()),
                        'design_rate': macro(events, ids[mask]), 'physical_clips': int(mask.sum()), 'designs': len(np.unique(ids[mask]))}
        if name in primary:
            condition['warning_paired'] = {}
            for bi, budget in enumerate(exp.BUDGETS):
                warned = [-p[ai, :, ci, :, 3] <= values['thresholds'][ai, :, :, bi][:, folds] for ai in range(len(arms))]
                comparisons = {}
                for arm in additions:
                    comparisons[arm] = {}
                    for ref in ('blend_frozen', 'blend_control', 'blend_adaptive'):
                        a, b = warned[arms.index(ref)], warned[arms.index(arm)]
                        comparisons[arm][ref] = {}
                        for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0),
                                            ('unstable_last1_miss', masks['unstable0.01'])]:
                            control, proposed = (a[:, mask], b[:, mask]) if label.startswith('stable') else (~a[:, mask], ~b[:, mask])
                            comparisons[arm][ref][label] = paired(control, proposed, ids[mask])
                condition['warning_paired'][str(budget)] = comparisons
            condition['uncertainty_paired'] = {}
            for arm in additions:
                ai = arms.index(arm)
                condition['uncertainty_paired'][arm] = {}
                for ref in ('blend_frozen', 'blend_control', 'blend_adaptive'):
                    ri = arms.index(ref)
                    entries = {'brier': paired((p[ri, :, ci, :, 3] - (y <= 0)) ** 2,
                                              (p[ai, :, ci, :, 3] - (y <= 0)) ** 2, ids)}
                    for width, lo, hi in [('raw', 1, 2), ('group', 4, 5)]:
                        a, b = p[ri, :, ci], p[ai, :, ci]
                        entries[width] = {
                            'coverage': paired((y >= a[:, :, lo]) & (y <= a[:, :, hi]), (y >= b[:, :, lo]) & (y <= b[:, :, hi]), ids),
                            'width': paired(a[:, :, hi] - a[:, :, lo], b[:, :, hi] - b[:, :, lo], ids),
                            'stable_safe_availability': paired(a[:, y > 0, lo] > 0, b[:, y > 0, lo] > 0, ids[y > 0]),
                            'unsafe_false_safe': paired(a[:, y <= 0, lo] > 0, b[:, y <= 0, lo] > 0, ids[y <= 0])}
                    condition['uncertainty_paired'][arm][ref] = entries
        result['conditions'][name] = condition
        print(name, 'designMAE', {arm: round(condition['slices']['all']['arms'][arm]['design_mae'], 6) for arm in arms}, flush=True)
    for arm in additions:
        for ref in ('blend_frozen', 'blend_control'):
            checks = []
            for name in fresh:
                c = result['conditions'][name]
                for label in ('all', 'stable0.01', 'unstable0.01', 'far_stable'):
                    ci = c['slices'][label]['paired'][arm][ref]['design_mae']['design_bootstrap_95']
                    checks.append({'condition': name, 'slice': label, 'upper': ci[1], 'pass': bool(ci[1] <= 0)})
                for label, metric in [('stable0.01', 'over_0.03'), ('far_stable', 'under_0.05')]:
                    ci = c['slices'][label]['paired'][arm][ref][metric]['design_bootstrap_95']
                    checks.append({'condition': name, 'slice': label, 'metric': metric, 'upper': ci[1], 'pass': bool(ci[1] <= 0)})
            result['strict_guardrails'].setdefault(arm, {})[ref] = {
                'passed': sum(c['pass'] for c in checks), 'checks': len(checks), 'all_pass': all(c['pass'] for c in checks),
                'details': checks, 'interpretation': 'Strict zero-deterioration fixed-prediction upper-bound screen, not a deployment certificate or multiplicity-adjusted claim.'}
    for family, group in [('iid1', ['iid1_new949', 'iid1_new951']),
                          ('correlated1', ['correlated1_new949_950', 'correlated1_new951_952']),
                          ('iid5', ['iid5_new949', 'iid5_new951']),
                          ('correlated5', ['correlated5_new949_950', 'correlated5_new951_952'])]:
        result['fresh_mean'][family] = {arm: {metric: float(np.mean([result['conditions'][c]['slices']['all']['arms'][arm][metric] for c in group]))
                                            for metric in ('design_mae', 'bias', 'over03', 'under05')} for arm in arms}
    exp.save_json(OUT / 'results.json', result)
    print('EVALUATION_COMPLETE', len(names), 'conditions', '45 physical and90 base crossfit fits converged', flush=True)


def timing():
    train, evaluation, truth, ids, rows, names, folds, dt, clamps, raw_hashes = exp.physical.load_data()
    proper, cal, test = exp.physical.split(ids, folds, 0)
    one, full = exp.physical.reference_predictors(42, 0, proper, cal, test)
    with np.load(exp.PRIOR / 'physical_seed42_fold0.npz') as z:
        prior = exp.cloudpickle.loads(z['model'].tobytes())
    models = {}
    for arm in exp.ARMS[:-1]:
        with np.load(OUT / f'heads_{arm}_seed42_fold0.npz') as z:
            models[arm] = (exp.cloudpickle.loads(z['heads'].tobytes()), json.loads(str(z['calibration'])))
    selected = test[np.linspace(0, len(test) - 1, 60, dtype=int)]
    queries = evaluation[names.index('iid1_new949'), selected]
    def predict(q, arm):
        a, b = one(q[:, :107]), full(q)
        post = prior.predict(q[:, [27, 28]], 'joint_soft') if arm in ('blend_joint', 'blend_soft', 'blend_both') else np.zeros((len(q), 6))
        components = np.column_stack([a, b, .5 * (a + b), post])
        x = exp.feature_blocks(q, components, 1, np.arange(len(q)))[arm]
        heads, levels = models[arm]
        point, scale = exp.point_scale(heads, x, components)
        return exp.distribution(point, scale, levels)
    durations = np.empty((len(models), len(selected)))
    for ai, arm in enumerate(models):
        for q in queries[:5]:
            predict(q[None], arm)
        for j, q in enumerate(queries):
            start = time.perf_counter()
            out = predict(q[None], arm)
            durations[ai, j] = 1000 * (time.perf_counter() - start)
            assert np.isfinite(out).all()
    exp.save_npz(OUT / 'inference_measurements.npz', durations_ms=durations, source_indices=rows[selected], arms=np.array(list(models)))
    result = {'boundary': 'Cached174 features through both base predictors, designated physical readouts, residual/scale heads and calibrated distributions; unchanged waveform extraction excluded. Sham timing excluded because its cross-clip permutation is diagnostic only.',
              'arms': {a: {'median_ms': float(np.median(v)), 'p95_ms': float(np.quantile(v, .95)), 'max_ms': float(v.max())}
                       for a, v in zip(models, durations)}, 'hash': exp.sha256(OUT / 'inference_measurements.npz')}
    exp.save_json(OUT / 'inference_timing.json', result)
    print('INFERENCE_TIMING', json.dumps(result['arms']), flush=True)


if __name__ == '__main__':
    with exp.threadpool_limits(limits=1):
        run_analysis()
        timing()
