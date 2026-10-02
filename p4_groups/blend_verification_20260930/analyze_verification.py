"""Confirm fixed blend on new draws; report boundary failures without retuning."""
import importlib.util
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT/'p4_groups/two_mode_spatial_20260930'
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from threadpoolctl import threadpool_limits
spec = importlib.util.spec_from_file_location('two_mode_analysis', PRIOR/'analyze_results.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
summarize, arm_difference = helpers.summarize, helpers.arm_difference
paired_gaps, paired_rate_difference, sha256 = helpers.paired_gaps, helpers.paired_rate_difference, helpers.sha256


def main():
    with np.load(OUT/'predictions.npz') as z:
        tr, ho = z['training_predictions'], z['heldout_predictions']
        y, ids, rows, folds = z['y'], z['ids'], z['source_indices'], z['row_fold']
        conditions, arms = z['conditions'].tolist(), z['arms'].tolist()
    assert arms == ['spatial107', 'two_mode174', 'blend50']
    np.testing.assert_array_equal(ho[2], .5*(ho[0]+ho[1]))
    np.testing.assert_array_equal(tr[2], .5*(tr[0]+tr[1]))
    slices = {'all': np.ones(len(y),bool), 'safer_at_least_0.15': y>=.15}
    for eps in (.01,.02,.05):
        slices[f'stable_below_{eps}'] = (y>0)&(y<eps)
        slices[f'unstable_within_{eps}'] = (y<0)&(y>-eps)
    files = {'analysis': Path(__file__), 'predictions': OUT/'predictions.npz',
             'features': OUT/'features.npz', 'models': OUT/'model_manifest.json',
             'protocol': OUT/'protocol.json', 'comparison_helpers': PRIOR/'analyze_results.py',
             'summary_helpers': ROOT/'p4_groups/local_weighting_20260929/analyze_results.py',
             'conditional_helpers': ROOT/'p4_groups/conditional_bias_diagnosis_20260929/analyze_diagnosis.py',
             'metrics': ROOT/'src/mechanics/p4_margin_estimation/decision_metrics.py'}
    result = {'protocol': json.loads((OUT/'protocol.json').read_text()),
              'hashes': {k: sha256(p) for k,p in files.items()},
              'conditions': {}, 'descriptive_checks': {}, 'seed_fold_stability': {}}
    designs = np.unique(ids)
    design_folds = np.array([folds[np.flatnonzero(ids==d)[0]] for d in designs])
    for ci,c in enumerate(conditions):
        records = {}
        for label,mask in slices.items():
            target, groups = y[mask],ids[mask]
            a,b = tr[:,:,:,ci,:][...,mask],ho[:,:,ci,:][...,mask]
            records[label] = {
                'physical_clips': int(mask.sum()), 'designs': int(len(np.unique(groups))),
                'arms': {arm: {'training': summarize(target,a[ai],groups,'three seeds x four in-sample exposures'),
                               'heldout': summarize(target,b[ai],groups,'three paired seeds x one OOF prediction'),
                               'held_minus_train': paired_gaps(target,a[ai],b[ai],groups)}
                         for ai,arm in enumerate(arms)},
                'blend_minus_reference': {ref: {
                    'training': arm_difference(target,a[ri],a[2],groups,'blend_minus_reference'),
                    'heldout': arm_difference(target,b[ri],b[2],groups,'blend_minus_reference')}
                    for ri,ref in enumerate(arms[:2])}, 'wrong_side_paired': {}}
            for side,m in [('stable',target>0),('unstable',target<0)]:
                if m.any():
                    wrong = b[...,m]<=0 if side=='stable' else b[...,m]>0
                    records[label]['wrong_side_paired'][side] = {
                        ref: paired_rate_difference(wrong[ri],wrong[2],groups[m]) for ri,ref in enumerate(arms[:2])}
        result['conditions'][c] = records
        def value(label, arm, metric):
            return metric(records[label]['arms'][arm]['heldout'])
        global_mae = lambda q:q['over']['design_mae']
        stable_sign = lambda q:q['zero_threshold_proxies']['stable']['mean_wrong_side_rate']
        unstable_sign = lambda q:q['zero_threshold_proxies']['unstable']['mean_wrong_side_rate']
        checks = {
            'global_mae_improves': value('all','blend50',global_mae)<value('all','spatial107',global_mae),
            'stable_last1_mae_nonworse': value('stable_below_0.01','blend50',global_mae)<=value('stable_below_0.01','spatial107',global_mae),
            'stable_last1_wrong_sign_nonworse': value('stable_below_0.01','blend50',stable_sign)<=value('stable_below_0.01','spatial107',stable_sign),
            'unstable_last1_wrong_sign_nonworse': value('unstable_within_0.01','blend50',unstable_sign)<=value('unstable_within_0.01','spatial107',unstable_sign)}
        result['descriptive_checks'][c] = {'checks': checks, 'all_pass': all(checks.values()),
            'interpretation': 'Point comparisons only; not equivalence/noninferiority or operational safety proof.'}
        macro = np.array([[[abs(p[ids==d]-y[ids==d]).mean() for d in designs] for p in ho[ai,:,ci]] for ai in range(3)])
        delta = macro[2]-macro[0]
        result['seed_fold_stability'][c] = {
            'per_seed_design_mae': {a:macro[ai].mean(axis=1).tolist() for ai,a in enumerate(arms)},
            'blend_minus_one_per_seed': delta.mean(axis=1).tolist(),
            'per_seed_designs_improved': (delta<0).sum(axis=1).tolist(),
            'per_seed_outer_fold_delta': np.column_stack([delta[:,design_folds==f].mean(axis=1) for f in range(5)]).tolist()}
        print(c,'global MAE',{a:records['all']['arms'][a]['heldout']['over']['design_mae'] for a in arms},
              'checks',checks,flush=True)
    result['all_four_new1pct_descriptive_checks_pass'] = all(result['descriptive_checks'][c]['all_pass'] for c in conditions[:4])
    result['both_new5pct_descriptive_checks_pass'] = all(result['descriptive_checks'][c]['all_pass'] for c in conditions[4:])
    path = OUT/'results.json'
    path.with_suffix('.tmp').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    path.with_suffix('.tmp').replace(path)
    print('Saved fixed-blend verification; all new1% point checks:',result['all_four_new1pct_descriptive_checks_pass'],
          'both5% stress checks:',result['both_new5pct_descriptive_checks_pass'],flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
