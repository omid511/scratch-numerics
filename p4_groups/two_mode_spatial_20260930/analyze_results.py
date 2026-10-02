"""Two-mode versus current spatial baseline; fixed-design conditional guardrails."""
import importlib.util
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'src'),
               str(ROOT/'p4_groups/conditional_bias_diagnosis_20260929'),
               str(ROOT/'p4_groups/correlated_training_20260929')]
import numpy as np
from threadpoolctl import threadpool_limits
from analyze_diagnosis import paired_gaps, distribution
from run_augmentation_experiment import sha256
spec = importlib.util.spec_from_file_location('p4_weighting_analysis',
    ROOT/'p4_groups/local_weighting_20260929/analyze_results.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
summarize, arm_difference = helpers.summarize, helpers.arm_difference


def paired_rate_difference(control, candidate, ids):
    """Clip-weighted event-rate change; seeds averaged before design resampling."""
    delta = candidate.reshape(-1, len(ids)).mean(axis=0)-control.reshape(-1, len(ids)).mean(axis=0)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = np.random.default_rng(42).multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=2000)
    sampled = (weights @ np.bincount(inverse, weights=delta))/(weights @ counts)
    return {'candidate_minus_reference': float(delta.mean()),
            'design_bootstrap_95': np.quantile(sampled, [.025, .975]).tolist()}


def main():
    with np.load(OUT/'predictions.npz') as z:
        tr, ho = z['training_predictions'], z['heldout_predictions']
        y, ids, rows, folds = z['y'], z['ids'], z['source_indices'], z['row_fold']
        conditions, arms = z['conditions'].tolist(), z['arms'].tolist()
    assert arms == ['spatial107', 'two_mode174', 'shuffled_addition174']
    with np.load(OUT/'features.npz') as z:
        np.testing.assert_array_equal(z['source_indices'], rows)
        additions = z['evaluation_additions']
    fits = json.loads((OUT/'fits.json').read_text())
    slices = {'all': np.ones(len(y), bool), 'safer_at_least_0.15': y>=.15}
    for eps in (.01, .02, .05):
        slices[f'stable_below_{eps}'] = (y>0)&(y<eps)
        slices[f'unstable_within_{eps}'] = (y<0)&(y>-eps)
    files = {'analysis': Path(__file__), 'predictions': OUT/'predictions.npz',
             'features': OUT/'features.npz', 'fits': OUT/'fits.json',
             'summary_helpers': ROOT/'p4_groups/local_weighting_20260929/analyze_results.py',
             'conditional_helpers': ROOT/'p4_groups/conditional_bias_diagnosis_20260929/analyze_diagnosis.py',
             'metrics': ROOT/'src/mechanics/p4_margin_estimation/decision_metrics.py'}
    output = {'protocol': json.loads((OUT/'protocol.json').read_text()),
              'all_fits_converged': fits['all_fits_converged'],
              'historical_prediction_replay_max_error': fits['historical_prediction_replay_max_error'],
              'hashes': {k: sha256(p) for k,p in files.items()},
              'conditions': {}, 'seed_fold_stability': {}, 'baseline_severe_cases': {}, 'availability': {}}
    for ci, condition in enumerate(conditions):
        records = {}
        for name, mask in slices.items():
            target, designs = y[mask], ids[mask]
            a, b = tr[:,:,:,ci,:][...,mask], ho[:,:,ci,:][...,mask]
            records[name] = {
                'physical_clips': int(mask.sum()), 'designs': int(len(np.unique(designs))),
                'arms': {arm: {'training': summarize(target, a[ai], designs, 'three seeds x four in-sample exposures'),
                               'heldout': summarize(target, b[ai], designs, 'three seeds x one OOF prediction'),
                               'held_minus_train': paired_gaps(target, a[ai], b[ai], designs)}
                         for ai, arm in enumerate(arms)},
                'two_mode_minus_reference': {reference: {
                    'training': arm_difference(target, a[ri], a[1], designs, 'two_mode_minus_reference'),
                    'heldout': arm_difference(target, b[ri], b[1], designs, 'two_mode_minus_reference')}
                    for ri, reference in [(0, arms[0]), (2, arms[2])]},
                'wrong_side_paired': {},
            }
            for side, side_mask in [('stable', target>0), ('unstable', target<0)]:
                if side_mask.any():
                    p = b[...,side_mask]
                    wrong = p<=0 if side == 'stable' else p>0
                    records[name]['wrong_side_paired'][side] = {
                        ref: paired_rate_difference(wrong[ri], wrong[1], designs[side_mask])
                        for ri,ref in [(0,arms[0]), (2,arms[2])]}
        output['conditions'][condition] = records
        per_design = np.array([[abs(p[ids==d]-y[ids==d]).mean() for d in np.unique(ids)]
                               for p in ho[1,:,ci]])
        baseline = np.array([[abs(p[ids==d]-y[ids==d]).mean() for d in np.unique(ids)]
                             for p in ho[0,:,ci]])
        delta = per_design-baseline
        design_folds = np.array([folds[np.flatnonzero(ids==d)[0]] for d in np.unique(ids)])
        output['seed_fold_stability'][condition] = {
            'per_seed_design_mae': per_design.mean(axis=1).tolist(),
            'per_seed_design_mae_delta': delta.mean(axis=1).tolist(),
            'per_seed_designs_improved': (delta<0).sum(axis=1).tolist(),
            'per_seed_outer_fold_mae_delta': np.column_stack([delta[:,design_folds==f].mean(axis=1) for f in range(5)]).tolist()}
        stable = (y>0)&(y<.05)
        errors = ho[:,:,ci]-y
        severe = stable[None,:]&((errors>.03).sum(axis=1)>=2)
        original = severe[0]
        output['baseline_severe_cases'][condition] = {
            'baseline_clips': int(original.sum()), 'source_indices': rows[original].tolist(),
            'arms': {arm: {'severe_total': int(severe[ai].sum()),
                           'baseline_cases_still_severe': int((original&severe[ai]).sum()),
                           'new_severe_cases': int((~original&severe[ai]).sum())}
                     for ai,arm in enumerate(arms)}}
        output['availability'][condition] = {
            'second_mode_missing_clips': int((additions[ci,:,-3]==0).sum()),
            'mode_pair_missing_clips': int((additions[ci,:,-1]==0).sum()),
            'valid_pair_similarity': distribution(additions[ci,additions[ci,:,-1]==1,-2])}
        print(condition, 'design MAE', {a: records['all']['arms'][a]['heldout']['over']['design_mae'] for a in arms}, flush=True)
    path = OUT/'results.json'
    path.with_suffix('.tmp').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    path.with_suffix('.tmp').replace(path)
    print('Saved two-mode comparisons and guardrails for', len(conditions), 'conditions', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
