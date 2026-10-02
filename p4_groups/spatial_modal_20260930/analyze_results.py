"""Fixed spatial ablation versus baseline, wide sham and local2 references."""
import json
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT/'src'),str(ROOT/'p4_groups/local_weighting_20260929'),
               str(ROOT/'p4_groups/conditional_bias_diagnosis_20260929'),
               str(ROOT/'p4_groups/correlated_training_20260929')]
import numpy as np
from threadpoolctl import threadpool_limits
from analyze_diagnosis import paired_gaps
spec = importlib.util.spec_from_file_location(
    "p4_weighting_analysis", ROOT/'p4_groups/local_weighting_20260929/analyze_results.py')
weighting_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weighting_analysis)
summarize, arm_difference = weighting_analysis.summarize, weighting_analysis.arm_difference
from run_augmentation_experiment import sha256


def main():
    with np.load(OUT/'predictions.npz') as z:
        new_tr,new_ho = z['training_predictions'],z['heldout_predictions']
        baseline,weighted = z['baseline_predictions'],z['local2_reference_predictions']
        y,ids,rows,conditions = z['y'],z['ids'],z['source_indices'],z['conditions'].tolist()
        assert z['arms'].tolist() == ['spatial','shuffled_spatial']
    with np.load(ROOT/'p4_groups/local_weighting_20260929/predictions.npz') as z:
        for key,reference in [('y',y),('ids',ids),('source_indices',rows),('conditions',conditions)]:
            np.testing.assert_array_equal(z[key],reference)
        reference_tr = z['training_predictions']
        np.testing.assert_array_equal(z['heldout_predictions'][0],baseline)
        np.testing.assert_array_equal(z['heldout_predictions'][1],weighted)
    training = np.stack([reference_tr[0],new_tr[0],new_tr[1],reference_tr[1]])
    held = np.stack([baseline,new_ho[0],new_ho[1],weighted])
    arms = ['baseline','spatial','shuffled_spatial','local2_reference']
    fits = json.loads((OUT/'fits.json').read_text())
    slices = {'all':np.ones(len(y),dtype=bool),'safer_at_least_0.15':y>=.15}
    for eps in (.01,.02,.05):
        slices[f'stable_below_{eps}'] = (y>0)&(y<eps)
        slices[f'unstable_within_{eps}'] = (y<0)&(y>-eps)
    with np.load(OUT/'spatial_features.npz') as z:
        layout,pairs,positions = z['sensor_indices'],z['sensor_pairs'],z['sensor_positions_yx']
        feature_labels = ([f'sensor{int(i)}_modal_power' for i in layout]
            +[f'pair{int(layout[i])}_{int(layout[j])}_real' for i,j in pairs]
            +[f'pair{int(layout[i])}_{int(layout[j])}_imag' for i,j in pairs]+['spatial_available'])
        availability = z['evaluation_spatial'][:,:,-1]
        modal_replay_error = float(z['max_modal_replay_error'])
    output = {'protocol':json.loads((OUT/'protocol.json').read_text()),
              'all_fits_converged':fits['all_fits_converged'],'conditions':{},'baseline_severe_cases':{},
              'spatial_definition':{'columns':feature_labels,'sensor_indices':layout.tolist(),
                  'positions_yx':positions.tolist(),'channel_pairs':pairs.tolist(),
                  'baseline_modal_replay_max_error':modal_replay_error},
              'hashes':{'analysis':sha256(Path(__file__)),'predictions':sha256(OUT/'predictions.npz'),
                  'spatial_features':sha256(OUT/'spatial_features.npz'),'fits':sha256(OUT/'fits.json'),
                  'summary_helpers':sha256(ROOT/'p4_groups/local_weighting_20260929/analyze_results.py'),
                  'conditional_helpers':sha256(ROOT/'p4_groups/conditional_bias_diagnosis_20260929/analyze_diagnosis.py'),
                  'metrics':sha256(ROOT/'src/mechanics/p4_margin_estimation/decision_metrics.py')}}
    for ci,condition in enumerate(conditions):
        records = {}
        for name,mask in slices.items():
            target,designs = y[mask],ids[mask]
            a,b = training[:,:,:,ci,:][...,mask],held[:,:,ci,:][...,mask]
            records[name] = {'physical_clips':int(mask.sum()),'designs':int(len(np.unique(designs))),
                'arms':{arm:{'training':summarize(target,a[ai],designs,'three seeds x four in-sample fit exposures'),
                             'heldout':summarize(target,b[ai],designs,'three seeds x one OOF prediction'),
                             'held_minus_train':paired_gaps(target,a[ai],b[ai],designs)}
                        for ai,arm in enumerate(arms)},
                'spatial_minus_reference':{reference:{
                    'training':arm_difference(target,a[ri],a[1],designs,'spatial_minus_reference'),
                    'heldout':arm_difference(target,b[ri],b[1],designs,'spatial_minus_reference')}
                    for ri,reference in [(0,'baseline'),(2,'shuffled_spatial'),(3,'local2_reference')]}}
        output['conditions'][condition] = records
        stable = slices['stable_below_0.05']
        errors = held[:,:,ci]-y
        severe = stable[None,:]&((errors>.03).sum(axis=1)>=2)
        baseline_mask = severe[0]
        output['baseline_severe_cases'][condition] = {
            'baseline_clips':int(baseline_mask.sum()),'baseline_designs':int(len(np.unique(ids[baseline_mask]))),
            'source_indices':rows[baseline_mask].tolist(),
            'arms':{arm:{'severe_total':int(severe[ai].sum()),
                         'baseline_cases_still_severe':int((baseline_mask&severe[ai]).sum()),
                         'new_severe_cases':int((~baseline_mask&severe[ai]).sum()),
                         'mean_error_on_baseline_cases':float(errors[ai,:,baseline_mask].mean()) if baseline_mask.any() else None,
                         'mean_absolute_error_on_baseline_cases':float(abs(errors[ai,:,baseline_mask]).mean()) if baseline_mask.any() else None}
                    for ai,arm in enumerate(arms)},
            'spatial_unavailable_clips':int((availability[ci]==0).sum())}
        if condition in ('clean','iid1_fresh932','correlated_fresh932_933','iid5_seed929'):
            print(condition,'whole design MAE',
                  {arm:records['all']['arms'][arm]['heldout']['over']['design_mae'] for arm in arms},flush=True)
            print(' near-stable bias',
                  {arm:records['stable_below_0.05']['arms'][arm]['heldout']['over']['mean_signed_error'] for arm in arms},flush=True)
    (OUT/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Saved spatial comparisons and conditional guardrails for',len(conditions),'conditions',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
