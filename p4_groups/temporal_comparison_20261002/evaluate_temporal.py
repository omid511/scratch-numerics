"""Complete populations and paired warning/accuracy comparisons; no result tuning."""
import argparse
import json
import sys
import experiment_temporal as T

sys.path.insert(0, str(T.ROOT / 'p4_groups/blend_additions_20261002'))
A = T.module('temporal_scoring', T.ROOT / 'p4_groups/blend_additions_20261002/evaluate_results.py')
C, np, OUT = T.C, T.np, T.OUT
FAMILIES = A.FAMILIES


def references(arm):
    return ['blend_frozen', 'blend_control'] if arm.startswith('learned_') else ['blend_frozen']


def analyze(smoke=False):
    prefix = 'smoke_' if smoke else ''
    with np.load(OUT / f'{prefix}predictions.npz') as z:
        p, thresholds, y, ids, folds = (z[k] for k in ('predictions', 'thresholds', 'y', 'ids', 'row_fold'))
        native = z['native_quantiles']
        names, arms, seeds = (z[k].tolist() for k in ('conditions', 'arms', 'seeds'))
        fingerprint = json.loads(str(z['fingerprint']))
    d = T.setup()
    assert fingerprint == d['fingerprint'] and arms == T.ARMS
    assert np.isfinite(p).all() and np.isfinite(native).all()
    fits = json.loads((OUT / f'{prefix}fits.json').read_text())
    assert fits['fingerprint'] == fingerprint
    for fit in fits['fits']:
        assert C.sha256(T.ROOT / fit['path']) == fit['hash']
        if not fit.get('frozen_base'):
            assert not set(fit['training_designs']) & set(fit['selection_designs'])
            assert set(fit['fit_designs']) == set(fit['training_designs']) | set(fit['selection_designs'])
            assert np.isfinite(fit['selection_history']['val_loss']).all()
    for partition in fits['partitions']:
        groups = [set(d['ids'][partition[k]]) for k in ('proper_rows', 'calibration_rows', 'held_rows')]
        assert all(not groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        assert np.all(np.array(next(b['rates'] for b in fits['calibration_budgets']
            if b['seed'] == partition['seed'] and b['fold'] == partition['fold'])) <= C.BUDGETS)
    primary = ['clean', *C.helpers.FRESH]
    result = dict(protocol=json.loads((OUT / 'protocol.json').read_text()), physical_clips=len(y),
        designs=len(np.unique(ids)), seeds=seeds, conditions={}, families={}, fingerprint=fingerprint,
        verification={'all_predictions_finite': True, 'all_partitions_design_disjoint': True,
                      'all_calibration_budgets_respected': True, 'fit_records': len(fits['fits']),
                      'neural_fits': sum(not f.get('frozen_base', False) for f in fits['fits']),
                      'optimization_note': 'Finite gradients/losses and completed fixed protocol; not a claim of global optimizer convergence.'})
    for ci, name in enumerate(names):
        condition = A.describe(p[:, :, ci], thresholds, y, ids, folds, arms)
        condition['native_quantiles'] = {}
        for ki, kind in enumerate(('gru', 'tcn')):
            v = native[ki, :, ci]
            condition['native_quantiles'][kind] = C.interval_helpers.intervals(y, np.concatenate((v[:, :, 1:2], v[:, :, :1], v[:, :, 2:3]), axis=2), ids)
        if name in primary:
            condition['paired'] = {arm: {ref: A.comparison(p[arms.index(ref), :, ci], p[ai, :, ci],
                thresholds[arms.index(ref)], thresholds[ai], y, ids, folds) for ref in references(arm)}
                for ai, arm in enumerate(arms) if arm != 'blend_frozen'}
        result['conditions'][name] = condition
        print('TEMPORAL_CONDITION', name, {arm: round(condition['slices']['all']['arms'][arm]['design_mae'], 6)
            for arm in arms}, flush=True)
    for family, members in FAMILIES.items():
        indices = [names.index(n) for n in members]
        family_p = np.stack([np.concatenate([p[ai, :, ci] for ci in indices]) for ai in range(len(arms))])
        family_t = np.concatenate([thresholds, thresholds], axis=1)
        record = A.describe(family_p, family_t, y, ids, folds, arms)
        record['prediction_rows'] = [{'condition': name, 'seed': seed} for name in members for seed in seeds]
        record['paired'] = {arm: {ref: A.comparison(family_p[arms.index(ref)], family_p[ai],
            family_t[arms.index(ref)], family_t[ai], y, ids, folds) for ref in references(arm)}
            for ai, arm in enumerate(arms) if arm != 'blend_frozen'}
        result['families'][family] = record
        print('TEMPORAL_FAMILY', family, {arm: round(record['slices']['all']['arms'][arm]['design_mae'], 6)
            for arm in arms}, flush=True)
    result['hashes'] = {name: C.sha256(OUT / name) for name in ('protocol.json', 'experiment_temporal.py',
        'evaluate_temporal.py', prefix + 'predictions.npz', prefix + 'fits.json')}
    C.save_json(OUT / f'{prefix}results.json', result)
    print('TEMPORAL_EVALUATION_COMPLETE', len(y), len(names), len(arms), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with T.E.threadpool_limits(limits=1):
        analyze(args.smoke)
