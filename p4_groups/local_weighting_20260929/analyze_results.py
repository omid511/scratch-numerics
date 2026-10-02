"""Paired local-weighting analysis; metrics, not predictions, average across seeds."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'src'),
               str(ROOT / 'p4_groups/conditional_bias_diagnosis_20260929'),
               str(ROOT / 'p4_groups/correlated_training_20260929')]
import numpy as np
from threadpoolctl import threadpool_limits
from analyze_diagnosis import conditional, paired_gaps
from run_augmentation_experiment import sha256


def summarize(y, predictions, ids, kind):
    result = conditional(y, predictions, ids, kind)
    error = predictions-y
    per_seed = np.quantile(error.reshape(3,-1), .99, axis=1)
    result['p99_signed_error'] = {'mean_seed_quantile': float(per_seed.mean()),
                                 'per_seed': per_seed.tolist()}
    result['zero_threshold_proxies'] = {}
    for side, mask in [('stable', y>0), ('unstable', y<0)]:
        if mask.any():
            p = predictions[..., mask]
            wrong = p <= 0 if side == 'stable' else p > 0
            result['zero_threshold_proxies'][side] = {
                'physical_clips': int(mask.sum()), 'mean_wrong_side_rate': float(wrong.mean()),
                'per_seed_wrong_side_rate': wrong.reshape(3,-1).mean(axis=1).tolist()}
    return result


def arm_difference(y, control, local, ids, difference_name='local_minus_control'):
    result = paired_gaps(y, control, local, ids)
    for item in result.values():
        item[difference_name] = item.pop('held_minus_train')
    return result


def main():
    with np.load(OUT / 'predictions.npz') as z:
        tr, ho, historical = z['training_predictions'], z['heldout_predictions'], z['historical_predictions']
        y, ids, rows, conditions, arms = z['y'], z['ids'], z['source_indices'], z['conditions'], z['arms']
    assert arms.tolist() == ['unit_control', 'local2']
    fits = json.loads((OUT / 'fits.json').read_text())
    slices = {'all': np.ones(len(y), dtype=bool), 'safer_at_least_0.15': y>=.15}
    for eps in (.01, .02, .05):
        slices[f'stable_below_{eps}'] = (y>0)&(y<eps)
        slices[f'unstable_within_{eps}'] = (y<0)&(y>-eps)
    output = {'protocol': json.loads((OUT / 'protocol.json').read_text()),
              'all_fits_converged': fits['all_fits_converged'], 'conditions': {},
              'unit_control_versus_historical': {},
              'hashes': {'analysis': sha256(Path(__file__)), 'predictions': sha256(OUT / 'predictions.npz'),
                         'fits': sha256(OUT / 'fits.json'),
                         'conditional_helpers': sha256(ROOT / 'p4_groups/conditional_bias_diagnosis_20260929/analyze_diagnosis.py'),
                         'metrics': sha256(ROOT / 'src/mechanics/p4_margin_estimation/decision_metrics.py')}}
    for ci, condition in enumerate(conditions):
        records = {}
        for name, mask in slices.items():
            target, designs = y[mask], ids[mask]
            a, b = tr[:, :, :, ci, :][..., mask], ho[:, :, ci, :][..., mask]
            records[name] = {
                'physical_clips': int(mask.sum()), 'designs': int(len(np.unique(designs))),
                'arms': {str(arm): {
                    'training': summarize(target, a[ai], designs, 'three seeds x four in-sample fit exposures'),
                    'heldout': summarize(target, b[ai], designs, 'three seeds x one design-held-out prediction'),
                    'held_minus_train': paired_gaps(target, a[ai], b[ai], designs)}
                    for ai, arm in enumerate(arms)},
                'local_minus_control': {
                    'training': arm_difference(target, a[0], a[1], designs),
                    'heldout': arm_difference(target, b[0], b[1], designs)}}
        output['conditions'][str(condition)] = records
        output['unit_control_versus_historical'][str(condition)] = {
            'max_absolute_prediction_difference': float(abs(ho[0,:,ci]-historical[:,ci]).max()),
            'mean_absolute_prediction_difference': float(abs(ho[0,:,ci]-historical[:,ci]).mean()),
            'paired_metric_differences': arm_difference(y, historical[:,ci], ho[0,:,ci], ids,
                                                       'unit_minus_historical')}
        if condition in ('clean', 'iid1_fresh932', 'correlated_fresh932_933', 'iid5_seed929'):
            print(condition, 'whole design MAE',
                  [records['all']['arms'][str(arm)]['heldout']['over']['design_mae'] for arm in arms],
                  'local bias', [records['stable_below_0.05']['arms'][str(arm)]['heldout']['over']['mean_signed_error'] for arm in arms], flush=True)
    with np.load(ROOT / 'p4_groups/correlated_training_20260929/shared.npz') as z:
        np.testing.assert_array_equal(rows, z['source_indices'])
        np.testing.assert_array_equal(y, z['y'])
        np.testing.assert_array_equal(ids, z['ids'])
    (OUT / 'results.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print('Saved paired conditional scores and guardrails for', len(conditions), 'conditions', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
