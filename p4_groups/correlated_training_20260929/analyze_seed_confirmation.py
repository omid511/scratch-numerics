"""Analyze fixed augmentation seed repeats without ensembling predictions."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison, error_exceedance
from run_augmentation_experiment import sha256


def main():
    output = Path(__file__).resolve().parent
    paths = {42: output / 'augmentation_results.json',
             43: output / 'augmentation_seed43.json', 44: output / 'augmentation_seed44.json'}
    runs = {seed: json.loads(path.read_text()) for seed, path in paths.items()}
    with np.load(output / 'shared.npz') as z:
        y, ids, rows = z['y'], z['ids'], z['source_indices']
    designs = np.unique(ids)
    names = ('extra_independent4', 'correlated4')
    conditions = runs[42]['settings']['evaluation_names']
    groups = [np.flatnonzero(ids == did) for did in designs]
    rng = np.random.default_rng(42)
    resamples = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(2000)]
    payload = {'protocol': 'Seeds42/43/44; unchanged five design folds seed42; mean of seed metrics, not an ensemble. Fixed OOF paired design bootstrap, no refit uncertainty or multiplicity correction.',
               'parents_sha256': {str(path): sha256(path) for path in paths.values()},
               'source_sha256': sha256(Path(__file__)), 'conditions': {}, 'boundaries': {}, 'fits': {}}
    pred = {}
    for seed, run in runs.items():
        assert run['settings']['model_seed'] == seed and run['settings']['fold_seed'] == 42
        np.testing.assert_array_equal(run['design_assignments']['source_indices'], rows)
        assert run['design_assignments']['row_fold'] == runs[42]['design_assignments']['row_fold']
        for arm in names:
            payload['fits'][f'{arm}_seed{seed}'] = [f['fit_metadata'] for f in run['arms'][arm]['folds']]
            for fold in run['arms'][arm]['folds']:
                assert not set(fold['train_design_ids']) & set(fold['test_design_ids'])
            for condition in conditions:
                pred[arm, seed, condition] = np.array(run['arms'][arm]['records'][condition]['predictions'])
                assert pred[arm, seed, condition].shape == y.shape
                assert np.isfinite(pred[arm, seed, condition]).all()
    for condition in conditions:
        summary = {}
        for arm in names:
            records = [runs[seed]['arms'][arm]['records'][condition] for seed in paths]
            metrics = ('design_mae', 'p99_signed_error', 'near_clip_mae')
            summary[arm] = {metric: {'mean': float(np.mean([r[metric] for r in records])),
                                      'seed_sd': float(np.std([r[metric] for r in records], ddof=1)),
                                      'per_seed': {str(seed): r[metric] for seed, r in zip(paths, records)}}
                            for metric in metrics}
        design_losses = {arm: [(str(d), float(np.mean([runs[s]['arms'][arm]['records'][condition]['per_design_mae'][str(d)] for s in paths]))) for d in designs] for arm in names}
        summary['correlated_minus_independent_mae'] = paired_design_comparison(
            design_losses['correlated4'], design_losses['extra_independent4'], n_bootstrap=10000, seed=42)
        if condition in ('clean', 'iid1_fresh932', 'correlated_fresh932_933', 'iid5_seed929'):
            errors = {arm: np.stack([pred[arm, s, condition]-y for s in paths]) for arm in names}
            bootstrap = [float(np.mean(np.quantile(errors['correlated4'][:, ix], .99, axis=1)
                                      - np.quantile(errors['extra_independent4'][:, ix], .99, axis=1))) for ix in resamples]
            summary['correlated_minus_independent_p99'] = {
                'difference': summary['correlated4']['p99_signed_error']['mean']-summary['extra_independent4']['p99_signed_error']['mean'],
                'design_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist()}
        payload['conditions'][condition] = summary
    for epsilon in (.01, .02, .05):
        payload['boundaries'][str(epsilon)] = {}
        for side in ('stable', 'unstable'):
            mask = (abs(y) < epsilon) & ((y > 0) if side == 'stable' else (y < 0))
            local_designs = np.unique(ids[mask])
            part = {'clips': int(mask.sum()), 'designs': len(local_designs), 'records': {}}
            for condition in conditions:
                part['records'][condition] = {}
                for arm in names:
                    metrics = []
                    for seed in paths:
                        p = pred[arm, seed, condition][mask]; target = y[mask]; error = p-target
                        loss = [float(np.mean(abs(error[ids[mask] == did]))) for did in local_designs]
                        metrics.append({'seed': seed, 'design_mae': float(np.mean(loss)),
                            'clip_mae': float(np.mean(abs(error))), 'mean_signed_error': float(np.mean(error)),
                            'p99_signed_error': float(np.quantile(error, .99)),
                            'wrong_side_fraction': float(np.mean(p <= 0 if side == 'stable' else p > 0))})
                    part['records'][condition][arm] = {'per_seed': metrics,
                        'mean': {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0] if key != 'seed'}}
            payload['boundaries'][str(epsilon)][side] = part
    risk = {
        'definition': 'Strict clip-weighted error exceedance; mean of seed rates, not ensemble predictions.',
        'thresholds': [0., .005, .01, .02, .03, .05],
        'uncertainty': '2000 bootstrap resamples of represented designs, all clips/seeds retained together; fixed OOF, no refits or multiplicity correction. Zero observed events do not establish zero population risk.',
        'safer_case_definition': 'm >= .15, a descriptive guardrail, not a certified safe operating region.',
        'metric_source_sha256': sha256(ROOT / 'src/mechanics/p4_margin_estimation/decision_metrics.py'),
        'slices': {},
    }
    slices = [(f'stable_below_{epsilon}', (y > 0) & (y < epsilon), 'over')
              for epsilon in (.01, .02, .05)]
    slices.append(('safer_at_least_0.15', y >= .15, 'under'))
    for label, mask, direction in slices:
        risk['slices'][label] = {}
        for condition in conditions:
            risk['slices'][label][condition] = {
                arm: error_exceedance(
                    y[mask], np.stack([pred[arm, s, condition][mask] for s in paths]),
                    ids[mask], risk['thresholds'], direction=direction)
                for arm in names}
    payload['conditional_error_risk'] = risk
    payload['all_fits_converged'] = all(f['success'] for folds in payload['fits'].values() for f in folds)
    (output / 'seed_confirmation_analysis.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print('All fits converged:', payload['all_fits_converged'])
    for condition in ('clean', 'iid1_fresh932', 'correlated_fresh932_933', 'iid5_seed929'):
        r = payload['conditions'][condition]
        print(condition, json.dumps(r), flush=True)
    for epsilon, sides in payload['boundaries'].items():
        for side, part in sides.items():
            print('boundary', epsilon, side, 'clips', part['clips'], 'designs', part['designs'])
            for condition in ('iid1_fresh932', 'correlated_fresh932_933'):
                print(condition, {arm: data['mean'] for arm, data in part['records'][condition].items()})


if __name__ == '__main__':
    main()
