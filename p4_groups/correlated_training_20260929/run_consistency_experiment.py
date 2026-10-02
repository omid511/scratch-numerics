"""Execute the fixed paired-loss protocol; no reserved-design evaluation."""
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from experiment_p4_ridge_cv import design_folds
from experiment_p4_tabular import fit_network, score
from run_augmentation_experiment import sha256


def main():
    output = Path(__file__).resolve().parent
    with np.load(output / 'shared.npz') as z:
        views = np.concatenate([z['train3'], z['correlated_extra'][None]], axis=0)
        evaluation, names = z['evaluation'], z['evaluation_names'].tolist()
        y, ids, source = z['y'], z['ids'], z['source_indices']
    n = len(y)
    payload = {'protocol': json.loads((output / 'protocol.json').read_text()),
               'hashes': {str(path): sha256(path) for path in (
                   output / 'shared.npz', Path(__file__), ROOT / 'experiment_p4_tabular.py',
                   ROOT / 'src/mechanics/p4_margin_estimation/skip_mlp.py')}, 'arms': {}}
    for arm, weight in [('custom_lambda0', 0.), ('consistency_lambda1', 1.)]:
        predictions = np.full((len(names), n), np.nan)
        folds = []
        for fold, held in enumerate(design_folds(ids, 5, 42)):
            fit = np.setdiff1d(np.arange(n), held)
            assert not set(ids[fit]) & set(ids[held])
            m = len(fit)
            x = views[:, fit].reshape(4*m, 42)
            target = np.tile(y[fit], 4)
            pairs = np.column_stack([m + np.arange(m), 3*m + np.arange(m)])
            for identities in (np.tile(source[fit], 4), np.tile(ids[fit], 4), target):
                np.testing.assert_array_equal(identities[pairs[:, 0]], identities[pairs[:, 1]])
            predict, info = fit_network(
                x, target, seed=42, alpha=1., custom=True,
                consistency_pairs=pairs, consistency_weight=weight)
            predictions[:, held] = predict(evaluation[:, held].reshape(-1, 42)).reshape(len(names), len(held))
            fitted = predict(x)
            info.update({'fold': fold, 'train_design_ids': np.unique(ids[fit]).tolist(),
                         'test_design_ids': np.unique(ids[held]).tolist(),
                         'test_source_indices': source[held].tolist(),
                         'consistency_weight': weight, 'n_pairs': m,
                         'supervised_half_mse': float(np.mean((fitted-target)**2)/2),
                         'paired_half_mse': float(np.mean((fitted[pairs[:, 0]]-fitted[pairs[:, 1]])**2)/2)})
            folds.append(info)
            print(arm, 'fold', fold, 'converged', info['success'], 'iterations', info['iterations'], flush=True)
        assert np.isfinite(predictions).all()
        payload['arms'][arm] = {'folds': folds, 'records': {
            name: score(y, predictions[i], ids) for i, name in enumerate(names)}}
        (output / 'consistency_results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
        print(arm, [(name, round(record['design_mae'], 6), round(record['p99_signed_error'], 6))
                    for name, record in payload['arms'][arm]['records'].items()], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
