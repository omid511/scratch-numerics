"""Reuse TRAIN-only frozen spatial data, predictors, calibration and metrics."""
import importlib.util
import json
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(OUT), str(ROOT), str(ROOT / 'src')]
spec = importlib.util.spec_from_file_location('local_frozen_contract', ROOT / 'p4_groups/boundary_noise_consistency_20260930/experiment_helpers.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
np, cloudpickle = helpers.np, helpers.cloudpickle
save_json, save_npz, sha256, fit_network = helpers.save_json, helpers.save_npz, helpers.sha256, helpers.fit_network
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from local_features import feature_row, GLOBAL_NAMES, LOCAL_NAMES, FREQUENCY_COLUMNS
spec = importlib.util.spec_from_file_location('local_interval_helpers', ROOT / 'p4_groups/physical_joint_inference_20261001/analyze_results.py')
# Its functions are reused, but no physical prior or oracle-data routine is run.
sys.path.append(str(ROOT / 'p4_groups/physical_joint_inference_20261001'))
interval_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interval_helpers)
ARMS = ['blend_frozen', 'blend_global', 'blend_local', 'blend_sham']
SEEDS = [42, 43, 44]
BUDGETS = [.01, .05, .10, .20]
FIELDS = ['point', 'raw_lower', 'raw_upper', 'p_unsafe', 'group_lower', 'group_upper']


def load_data():
    train, evaluation, y, ids, rows, names, folds, layout, pairs = helpers.load_data()
    with np.load(helpers.OUT / 'fresh_features.npz') as z:
        for key, value in [('y', y), ('ids', ids), ('source_indices', rows), ('row_fold', folds)]:
            np.testing.assert_array_equal(z[key], value)
        evaluation = np.concatenate([evaluation, z['features']])
        names += z['conditions'].tolist()
    metadata = json.loads((ROOT / 'p4_dataset_saturation/metadata.json').read_text())
    assert set(ids) == set(metadata['train_designs'])
    with np.load(ROOT / 'p4_dataset_saturation/metadata_arrays.npz') as z:
        np.testing.assert_array_equal(z['design_ids'][rows], ids)
        np.testing.assert_array_equal(z['margins'][rows], y)
        dt = z['dts'][rows]
    assert train.shape == (4, 4454, 174) and evaluation.shape == (37, 4454, 174)
    assert len(np.unique(ids)) == 77 and len(set(names)) == len(names)
    return train, evaluation, y, ids, rows, names, folds, layout, pairs, dt


def signature():
    files = [OUT / f for f in ('protocol.json', 'contract.py', 'local_features.py', 'prepare_features.py', 'run_experiment.py')]
    files += [ROOT / 'experiment_p4_tabular.py', ROOT / 'src/mechanics/p4_margin_estimation/baselines.py',
              ROOT / 'src/mechanics/p4_margin_estimation/transient.py', ROOT / 'src/mechanics/p4_margin_estimation/modal_features.py',
              ROOT / 'p4_dataset_saturation/metadata_arrays.npz', ROOT / 'p4_dataset_saturation/metadata.json',
              helpers.base.PRIOR / 'features.npz', helpers.base.VERIFY / 'features.npz',
              helpers.BASE / 'confirmation_features.npz', helpers.OUT / 'fresh_features.npz',
              ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json']
    return {'sources': {str(p.relative_to(ROOT)): sha256(p) for p in files},
            'versions': helpers.base.source_signature()['versions'],
            'references': {f'{a}_{s}_{f}': sha256(helpers.BASE / f'warning_{a}_seed{s}_fold{f}.npz')
                           for a in ('one', 'PCS') for s in SEEDS for f in range(5)}}


def split(ids, folds, fold):
    outer, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    designs = np.unique(ids[outer])
    np.random.default_rng(947 + fold).shuffle(designs)
    cal_ids = designs[:int(np.ceil(.2 * len(designs)))]
    cal, proper = outer[np.isin(ids[outer], cal_ids)], outer[~np.isin(ids[outer], cal_ids)]
    assert not set(ids[proper]) & set(ids[cal]) and not set(ids[outer]) & set(ids[held])
    return proper, cal, held


def reference_predictors(seed, fold, proper, cal, held):
    predictors = []
    for arm in ('one', 'PCS'):
        path = helpers.BASE / f'warning_{arm}_seed{seed}_fold{fold}.npz'
        with np.load(path) as z:
            for key, rows in [('proper_rows', proper), ('calibration_rows', cal), ('held_rows', held)]:
                np.testing.assert_array_equal(z[key], rows)
        predictors.append(helpers.base.trusted_predictor(path))
    return predictors


def calibration(point, y, ids):
    residual = abs(y - point)
    maxima = np.sort([residual[ids == d].max() for d in np.unique(ids)])
    rank = int(np.ceil((len(maxima) + 1) * .9))
    if rank > len(maxima):
        raise ValueError('too few calibration designs for finite90% group calibration')
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if rms <= 0:
        raise ValueError('zero calibration RMS')
    return {'raw': float(np.quantile(residual, .9)), 'group': float(maxima[rank - 1]), 'rms': rms}


def distribution(point, levels):
    return np.column_stack([point, point - levels['raw'], point + levels['raw'],
                            ndtr(-point / levels['rms']), point - levels['group'], point + levels['group']])


def threshold(probability, y, budget):
    scores = np.sort(-probability[y > 0])
    value = float(np.nextafter(scores[int(np.floor(budget * len(scores)))], -np.inf))
    assert np.mean(-probability[y > 0] <= value) <= budget
    return value
