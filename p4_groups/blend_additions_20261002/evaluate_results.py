"""Paired complete-population evaluation; freeze retention before cross-family scoring."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as E
C, np, OUT = E.C, E.np, E.OUT
FAMILIES = {
    'iid1': ['iid1_new949', 'iid1_new951'],
    'correlated1': ['correlated1_new949_950', 'correlated1_new951_952'],
    'iid5': ['iid5_new949', 'iid5_new951'],
    'correlated5': ['correlated5_new949_950', 'correlated5_new951_952']}
ELIGIBLE = ['neighborhood', 'global_consistency', 'localized_consistency', 'local_observability']


def references(arm):
    refs = ['blend_frozen']
    if arm.startswith('learned_'):
        refs.append('blend_control')
    if arm not in ('blend_affine', 'blend_control', 'blend_frozen'):
        route, addon = arm.split('_', 1)
        if addon in ('global_consistency', 'localized_consistency'):
            refs.append(f'{route}_supervised')
        if addon == 'localized_consistency':
            refs.append(f'{route}_global_consistency')
        if addon == 'local_observability':
            refs.append(f'{route}_global_observability')
    return refs


def warn(prediction, thresholds, folds, budget):
    if budget == 0:
        return prediction[:, :, 0] <= 0
    return -prediction[:, :, 3] <= thresholds[:, :, C.BUDGETS.index(budget)][:, folds]


def point_scores(y, p, ids):
    return {**C.helpers.point(y, p, ids), 'over02': float(((p - y) > .02).mean()),
            'signed_p99_per_seed': np.quantile(p - y, .99, axis=1).tolist()}


def describe(p, thresholds, y, ids, folds, arms):
    masks = C.interval_helpers.point_slices(y, ids)
    result = dict(slices={}, uncertainty={}, warning={}, stability={})
    for label, mask in masks.items():
        result['slices'][label] = {'physical_clips': int(mask.sum()), 'designs': len(np.unique(ids[mask])),
            'arms': {a: point_scores(y[mask], p[i][:, mask, 0], ids[mask]) for i, a in enumerate(arms)}}
    for i, a in enumerate(arms):
        result['uncertainty'][a] = {}
        for label in ('all', 'stable0.01', 'unstable0.01', 'far_stable'):
            mask = masks[label]
            v = p[i][:, mask]
            result['uncertainty'][a][label] = {
                'raw': C.interval_helpers.intervals(y[mask], v[:, :, :4], ids[mask]),
                'group': C.interval_helpers.intervals(y[mask], v[:, :, [0, 4, 5, 3]], ids[mask]),
                'probability': C.interval_helpers.probability_scores(y[mask], v[:, :, 3], ids[mask])}
        result['warning'][a] = {}
        for b in [0., *C.BUDGETS]:
            w = warn(p[i], thresholds[i], folds, b)
            entries = {}
            for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0),
                                ('stable_last1_warning', masks['stable0.01']), ('unstable_last1_miss', masks['unstable0.01'])]:
                event = w[:, mask] if label.startswith('stable') else ~w[:, mask]
                entries[label] = dict(rate=float(event.mean()), design_rate=C.interval_helpers.macro(event, ids[mask]),
                                      per_seed=event.mean(axis=1).tolist(), physical_clips=int(mask.sum()),
                                      designs=len(np.unique(ids[mask])))
            result['warning'][a][str(b)] = entries
        result['stability'][a] = {
            'per_seed': [C.helpers.point(y, v[:, 0], ids)['design_mae'] for v in p[i]],
            'per_seed_fold': [[C.helpers.point(y[folds == f], v[folds == f, 0], ids[folds == f])['design_mae']
                               for f in np.unique(folds)] for v in p[i]]}
    return result


def comparison(a, b, at, bt, y, ids, folds):
    masks = C.interval_helpers.point_slices(y, ids)
    paired = C.helpers.paired_rate_difference
    out = {'slices': {label: C.helpers.paired_metrics(y[mask], a[:, mask, 0], b[:, mask, 0], ids[mask])
                      for label, mask in masks.items()},
           'brier': paired((a[:, :, 3] - (y <= 0)) ** 2, (b[:, :, 3] - (y <= 0)) ** 2, ids),
           'warning5': {}, 'intervals': {}}
    wa, wb = warn(a, at, folds, .05), warn(b, bt, folds, .05)
    for label, mask in [('stable_false_warning', y > 0), ('unstable_miss', y <= 0),
                        ('unstable_last1_miss', masks['unstable0.01'])]:
        av, bv = (wa[:, mask], wb[:, mask]) if label.startswith('stable') else (~wa[:, mask], ~wb[:, mask])
        out['warning5'][label] = paired(av, bv, ids[mask])
    for label, lo, hi in [('raw', 1, 2), ('group', 4, 5)]:
        out['intervals'][label] = {
            'width': paired(a[:, :, hi] - a[:, :, lo], b[:, :, hi] - b[:, :, lo], ids),
            'coverage': paired((y >= a[:, :, lo]) & (y <= a[:, :, hi]), (y >= b[:, :, lo]) & (y <= b[:, :, hi]), ids),
            'stable_safe_availability': paired(a[:, y > 0, lo] > 0, b[:, y > 0, lo] > 0, ids[y > 0]),
            'unsafe_false_safe': paired(a[:, y <= 0, lo] > 0, b[:, y <= 0, lo] > 0, ids[y <= 0])}
    return out


def check_entries(record):
    yield 'all_MAE', record['slices']['all']['design_mae']
    yield 'stable_last1_MAE', record['slices']['stable0.01']['design_mae']
    yield 'unstable_last1_MAE', record['slices']['unstable0.01']['design_mae']
    yield 'unstable_last1_optimism_gt03', record['slices']['unstable0.01']['over_0.03']
    yield 'far_stable_underestimation_gt05', record['slices']['far_stable']['under_0.05']
    yield 'brier', record['brier']
    yield 'warning5_stable_false_warning', record['warning5']['stable_false_warning']
    yield 'warning5_unstable_last1_miss', record['warning5']['unstable_last1_miss']


def retention(result):
    decisions, selected = {}, []
    for addon in ELIGIBLE:
        decisions[addon] = {}
        for route in ('fixed', 'learned'):
            arm = f'{route}_{addon}'
            required = references(arm)
            gains = [family for family in FAMILIES if all(
                result['families'][family]['paired'][arm][ref]['slices']['all']['design_mae']['design_bootstrap_95'][1] < 0
                for ref in required)]
            harms, unresolved = [], []
            for condition in ['clean', *FAMILIES]:
                source = result['conditions'][condition] if condition == 'clean' else result['families'][condition]
                for name, score in check_entries(source['paired'][arm]['blend_frozen']):
                    lo, hi = score['design_bootstrap_95']
                    row = dict(condition=condition, metric=name, **score)
                    if lo > 0:
                        harms.append(row)
                    elif hi > 0:
                        unresolved.append(row)
            decisions[addon][route] = dict(gain_families=gains, matched_references=required, supported_harms=harms,
                                           unresolved_guardrails=unresolved, retained=bool(gains and not harms))
        good = [r for r in ('learned', 'fixed') if decisions[addon][r]['retained']]
        if good:
            selected.append(f'{good[0]}_{addon}')
    # A localized candidate must beat the identically fused global candidate; already required above.
    # Keep one consistency readout, never count global/localized as independent families.
    if any(a.endswith('_localized_consistency') for a in selected):
        selected = [a for a in selected if not a.endswith('_global_consistency')]
    value = dict(independent_results_hash=E.sha256(OUT / 'results.json'), protocol_hash=E.sha256(OUT / 'protocol.json'),
                 decisions=decisions, retained=selected,
                 interpretation='Exploratory combination eligibility only; unresolved guardrails are not noninferiority proof.')
    path = OUT / 'retention.json'
    if path.exists():
        existing = json.loads(path.read_text())
        assert {k: v for k, v in existing.items() if k != 'frozen_at_utc'} == value, 'retention decisions already frozen'
    else:
        value['frozen_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        E.save_json(path, value)
    print('RETENTION_FROZEN', selected, flush=True)
    for addon, routes in decisions.items():
        print(addon, {r: {'gain_families': v['gain_families'], 'supported_harms': len(v['supported_harms']),
                         'unresolved_guardrails': len(v['unresolved_guardrails']), 'retained': v['retained']} for r, v in routes.items()}, flush=True)


def analyze(smoke=False):
    prefix = 'smoke_' if smoke else ''
    with np.load(OUT / f'{prefix}predictions.npz') as z:
        p, t, y, ids, folds = [z[k] for k in ('predictions', 'thresholds', 'y', 'ids', 'row_fold')]
        names, arms, seeds = z['conditions'].tolist(), z['arms'].tolist(), z['seeds'].tolist()
    assert np.isfinite(p).all() and arms == E.ARMS
    fits = json.loads((OUT / f'{prefix}fits.json').read_text())
    assert fits['all_converged']
    for row in fits['fits']:
        assert E.sha256(E.ROOT / row['path']) == row['hash'], 'changed fit artifact'
    for split in fits['partitions']:
        if not smoke:
            groups = [set(ids[split[k]]) for k in ('proper_rows', 'calibration_rows', 'held_rows')]
            assert all(not groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    result = dict(protocol=json.loads((OUT / 'protocol.json').read_text()), physical_clips=len(y), designs=len(np.unique(ids)),
                  seeds=seeds, all_converged=True, fit_records=len(fits['fits']), conditions={}, families={},
                  hashes={f: E.sha256(OUT / f) for f in ('protocol.json', 'run_experiment.py', 'evaluate_results.py',
                                                                       prefix + 'predictions.npz', prefix + 'fits.json')})
    primary = ['clean', *C.helpers.FRESH]
    candidate_arms = [a for a in arms if a != 'blend_frozen']
    for ci, name in enumerate(names):
        result['conditions'][name] = describe(p[:, :, ci], t, y, ids, folds, arms)
        result['conditions'][name]['prediction_rows'] = [{'condition': name, 'seed': s} for s in seeds]
        if name in primary:
            result['conditions'][name]['paired'] = {a: {
                ref: comparison(p[arms.index(ref), :, ci], p[arms.index(a), :, ci],
                                t[arms.index(ref)], t[arms.index(a)], y, ids, folds)
                for ref in references(a)} for a in candidate_arms}
        print('ADDITION_CONDITION', name, {a: round(result['conditions'][name]['slices']['all']['arms'][a]['design_mae'], 6) for a in arms}, flush=True)
    for family, members in FAMILIES.items():
        indices = [names.index(n) for n in members]
        family_p = np.stack([np.concatenate([p[ai, :, ci] for ci in indices]) for ai in range(len(arms))])
        family_t = np.concatenate([t, t], axis=1)
        result['families'][family] = describe(family_p, family_t, y, ids, folds, arms)
        result['families'][family]['prediction_rows'] = [{'condition': n, 'seed': s} for n in members for s in seeds]
        result['families'][family]['row_note'] = 'Per-seed arrays contain one row per acquisition/seed pair in prediction_rows order; draws are not new physical designs.'
        result['families'][family]['paired'] = {a: {
            ref: comparison(family_p[arms.index(ref)], family_p[arms.index(a)],
                            family_t[arms.index(ref)], family_t[arms.index(a)], y, ids, folds)
            for ref in references(a)} for a in candidate_arms}
        print('ADDITION_FAMILY', family, {a: round(result['families'][family]['slices']['all']['arms'][a]['design_mae'], 6) for a in arms}, flush=True)
    E.save_json(OUT / f'{prefix}results.json', result)
    if not smoke:
        retention(result)
    print('EVALUATION_SMOKE_PASSED' if smoke else 'INDEPENDENT_EVALUATION_COMPLETE', len(y), len(names), len(arms), 'point, intervals, probabilities, warnings and paired families', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    with E.threadpool_limits(limits=1):
        analyze(args.smoke)
