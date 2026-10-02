"""Fixed-OOF paired design comparisons for the predeclared noise experiment."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from mechanics.p4_margin_estimation.decision_metrics import paired_design_comparison
from run_augmentation_experiment import sha256


def main():
    output = Path(__file__).resolve().parent
    with np.load(output / 'shared.npz') as z:
        y, ids = z['y'], z['ids']
    arms = {}
    parents = [output / name for name in ('augmentation_results.json', 'consistency_results.json')]
    for parent in parents:
        arms.update(json.loads(parent.read_text())['arms'])
    conditions = list(arms['original3']['records'])
    contrasts = [('correlated4', 'extra_independent4'), ('correlated4', 'original3'),
                 ('consistency_lambda1', 'custom_lambda0'), ('custom_lambda0', 'correlated4')]
    result = {'limits': 'One model seed; fixed OOF clustered bootstrap, no refits or multiplicity correction. No reserved designs.',
              'parents_sha256': {str(p): sha256(p) for p in parents}, 'comparisons': {},
              'arms': {}, 'bootstrap_replicates': {'mae': 10000, 'p99': 2000}}
    groups = [np.flatnonzero(ids == did) for did in np.unique(ids)]
    rng = np.random.default_rng(42)
    resampled = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(2000)]
    for name, arm in arms.items():
        fits = [fold.get('fit_metadata', fold) for fold in arm['folds']]
        for record in arm['records'].values():
            assert np.isfinite(record['predictions']).all() and len(record['predictions']) == len(y)
        iid = np.array(arm['records']['iid1_fresh932']['predictions'])
        corr = np.array(arm['records']['correlated_fresh932_933']['predictions'])
        result['arms'][name] = {'all_fits_converged': all(f['success'] for f in fits),
            'fresh_paired_prediction_gap_mae': float(np.mean(abs(corr-iid))),
            'fresh_paired_prediction_gap_rmse': float(np.sqrt(np.mean((corr-iid)**2))),
            'records': {c: {k: v for k, v in r.items() if k not in ('predictions', 'per_design_mae')}
                        for c, r in arm['records'].items()}}
    for left, right in contrasts:
        key = left + '_minus_' + right
        result['comparisons'][key] = {}
        for condition in conditions:
            a, b = arms[left]['records'][condition], arms[right]['records'][condition]
            comparison = {'design_mae': paired_design_comparison(
                a['per_design_mae'].items(), b['per_design_mae'].items(), n_bootstrap=10000, seed=42)}
            if condition in ('clean', 'iid1_fresh932', 'correlated_fresh932_933', 'iid5_seed929'):
                ea, eb = np.array(a['predictions']) - y, np.array(b['predictions']) - y
                boot = [np.quantile(ea[ix], .99) - np.quantile(eb[ix], .99) for ix in resampled]
                comparison['p99_signed_error'] = {'difference': float(np.quantile(ea, .99)-np.quantile(eb, .99)),
                                                  'design_bootstrap_95': np.quantile(boot, [.025, .975]).tolist()}
            result['comparisons'][key][condition] = comparison
        print(key, flush=True)
        for condition in ('clean', 'iid1_fresh932', 'correlated_fresh932_933', 'iid5_seed929'):
            print(condition, result['comparisons'][key][condition], flush=True)
    for name, arm in result['arms'].items():
        print(name, 'converged', arm['all_fits_converged'], 'fresh gap', arm['fresh_paired_prediction_gap_mae'], flush=True)
        for condition in ('clean', 'iid1_fresh932', 'correlated_fresh932_933', 'iid5_seed929'):
            print(condition, arm['records'][condition], flush=True)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
