"""Warm single-thread full inference timing; acquisition and disk IO excluded."""
import json
import os
from pathlib import Path
import platform
import sys
import time
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(OUT), str(ROOT), str(ROOT/'src'),
               str(ROOT/'p4_groups/correlated_training_20260929')]
import cloudpickle
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from mechanics.p4_margin_estimation.modal_features import estimate_modes, modal_spatial_row, modal_spatial_addition_row
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
from prepare_features import base_feature_row
from run_augmentation_experiment import sha256


def inference(measured, dt, layout, pairs, ridge, predict, two_mode):
    signal = np.clip(causal_calibration_normalize(measured, calibration_samples=51)[layout], -50, 50).astype(np.float32)
    modes = estimate_modes(signal, dt, include_shapes=True)
    base = base_feature_row(signal, dt, modes, ridge)
    spatial = modal_spatial_row(modes, pairs)
    pieces = [base, spatial]
    if two_mode:
        pieces.append(modal_spatial_addition_row(modes, pairs))
    query = np.concatenate(pieces)
    return float(predict(query[None])[0]), query


def main():
    with np.load(OUT/'features.npz') as z:
        ids, rows = z['ids'], z['source_indices']
        layout, pairs = z['sensor_indices'], z['sensor_pairs']
        ci = z['conditions'].tolist().index('iid1_fresh935')
        reference, additions = z['evaluation107'][ci], z['evaluation_additions'][ci]
    with np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz') as z:
        dts = z['dts'][rows]
    selected = np.array([np.flatnonzero(ids==d)[0] for d in np.unique(ids)])
    measured = []
    for k in selected:
        with np.load(ROOT/'p4_sensor_candidates'/f'{ids[k]}.npz') as z:
            j = int(np.flatnonzero(z['source_indices']==rows[k])[0])
            raw, sigma = z['raw'][j], z['noise_scale'][j]
        white = np.random.default_rng(np.random.SeedSequence([935, int(rows[k])])).normal(size=raw.shape)
        measured.append(raw+.01*sigma*white)
    arms = ['spatial107', 'two_mode174']
    predictors, signatures = [], {}
    for arm in arms:
        path = OUT/f'{arm}_seed42_fold0.npz'
        with np.load(path) as z:
            predictors.append(cloudpickle.loads(z['predictor'].tobytes()))  # Trusted experiment artifact only.
        signatures[arm] = sha256(path)
    ridge = PhysicsFeatureRidge()
    expected = []
    for ai, predict in enumerate(predictors):
        query = reference[selected] if ai == 0 else np.column_stack([reference[selected], additions[selected]])
        expected.append(predict(query))
        for j,k in enumerate(selected):
            prediction, actual = inference(measured[j], float(dts[k]), layout, pairs, ridge, predict, ai==1)
            np.testing.assert_allclose(actual, query[j], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(prediction, expected[-1][j], atol=1e-12, rtol=1e-12)
    durations = np.empty((2, 10, len(selected)))
    outputs = np.empty_like(durations)
    for repeat in range(10):
        for j,k in enumerate(selected):
            for ai in ((0,1) if (repeat+j)%2 == 0 else (1,0)):
                start = time.perf_counter_ns()
                prediction, _ = inference(measured[j], float(dts[k]), layout, pairs, ridge, predictors[ai], ai==1)
                durations[ai,repeat,j] = (time.perf_counter_ns()-start)/1e6
                outputs[ai,repeat,j] = prediction
    for ai in range(2):
        np.testing.assert_allclose(outputs[ai], np.broadcast_to(expected[ai], outputs[ai].shape), atol=1e-12, rtol=1e-12)
    result = {
        'scope': json.loads((OUT/'protocol.json').read_text())['timing'],
        'platform': platform.platform(), 'python': platform.python_version(),
        'threadpools': threadpool_info(), 'physical_clips': len(selected), 'repetitions': 10,
        'arms': {arm: {'median_ms': float(np.median(durations[ai])),
                        'p95_ms': float(np.quantile(durations[ai], .95)),
                        'max_ms': float(durations[ai].max())} for ai,arm in enumerate(arms)},
        'paired_overhead_ms': {'median': float(np.median(durations[1]-durations[0])),
                               'mean': float((durations[1]-durations[0]).mean()),
                               'p95': float(np.quantile(durations[1]-durations[0], .95))},
        'hashes': {'measurement': sha256(Path(__file__)), 'features': sha256(OUT/'features.npz'),
                   'prepare_features': sha256(OUT/'prepare_features.py'), 'protocol': sha256(OUT/'protocol.json'),
                   'models': signatures},
    }
    output = OUT/'inference_measurements.npz'
    with output.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, durations_ms=durations, predictions=outputs, source_indices=rows[selected],
                            ids=ids[selected], arms=arms)
    output.with_suffix('.tmp').replace(output)
    result['hashes']['measurements'] = sha256(output)
    path = OUT/'inference_timing.json'
    path.with_suffix('.tmp').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    path.with_suffix('.tmp').replace(path)
    print('Full inference timing', json.dumps(result['arms']), 'paired overhead', result['paired_overhead_ms'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
