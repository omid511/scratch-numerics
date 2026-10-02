"""Replay fixed training-design acquisitions and retain two-mode additions."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'p4_groups/correlated_training_20260929'
SPATIAL = ROOT / 'p4_groups/spatial_modal_20260930'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(PARENT)]
import numpy as np
from threadpoolctl import threadpool_limits
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from mechanics.p4_margin_estimation.modal_features import (
    estimate_modes, modal_feature_row, physics_feature_groups,
    modal_spatial_row, modal_spatial_addition_row,
)
from mechanics.p4_margin_estimation.sidecar import extract
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
from run_augmentation_experiment import sha256

FRESH = ['iid1_fresh935', 'correlated_fresh935_936',
         'iid1_fresh937', 'correlated_fresh937_938']
TRAIN_VIEWS = ['clean', 'iid1_seed928', 'iid5_seed928', 'iid1_seed929']


def base_feature_row(signal, dt, modes, ridge):
    """Same42-feature convention as the existing symmetric visible-mode replay."""
    groups = physics_feature_groups(modes)
    a, f, e = modes['alpha'], modes['frequency'], modes['energy_fraction']
    visible = np.flatnonzero(e >= .01)
    if len(visible) > 1:
        order = visible[np.lexsort((f[visible], -a[visible]))]
        lead, others = order[0], order[1:]
        neighbor = others[np.argmin(abs(f[others] - f[lead]))]
        natural = np.hypot(a, 2*np.pi*f)
        groups['spacing'][0] = 2*abs(f[lead]-f[neighbor])/(f[lead]+f[neighbor])
        groups['spacing'][2] = 2*np.hypot(a[lead]-a[neighbor], 2*np.pi*(f[lead]-f[neighbor]))/(natural[lead]+natural[neighbor])
    correlation = extract([SimpleNamespace(sensor_signals=signal, dt=dt)], ['xcorr'])[0]
    return np.concatenate([ridge._extract_features(signal, dt=dt), modal_feature_row(modes), *groups.values(), correlation])


def main():
    with np.load(PARENT / 'shared.npz') as z:
        ids, rows, y = z['ids'], z['source_indices'], z['y']
        historical, base_reference = z['evaluation_names'].tolist(), z['evaluation']
    with np.load(SPATIAL / 'spatial_features.npz') as z:
        for key, value in [('ids', ids), ('source_indices', rows), ('y', y), ('conditions', historical)]:
            np.testing.assert_array_equal(z[key], value)
        spatial_reference = z['evaluation_spatial']
        layout, pairs = z['sensor_indices'], z['sensor_pairs']
    names = historical + FRESH
    nh = len(historical)
    with np.load(ROOT / 'p4_dataset_saturation/metadata_arrays.npz') as z:
        dt = z['dts'][rows]
    gains = np.asarray(json.loads((ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    rowmap = {int(row): k for k, row in enumerate(rows)}
    source_files = ['src/mechanics/p4_margin_estimation/' + f for f in
                    ('modal_features.py', 'baselines.py', 'sidecar.py', 'transient.py')]
    fingerprint = {
        'protocol': sha256(OUT / 'protocol.json'), 'execution': sha256(Path(__file__)),
        'shared_features': sha256(PARENT / 'shared.npz'),
        'spatial_reference': sha256(SPATIAL / 'spatial_features.npz'),
        'acquisition_protocol': sha256(ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json'),
        'timebase_source': sha256(ROOT / 'p4_dataset_saturation/metadata_arrays.npz'),
        'sources': {f: sha256(ROOT / f) for f in source_files},
    }
    original = np.full((len(names), len(rows), 107), np.nan)
    original[:nh] = np.concatenate([base_reference, spatial_reference], axis=2)
    additions = np.full((len(names), len(rows), 67), np.nan)
    ridge, max_replay = PhysicsFeatureRidge(), 0.
    for did in np.unique(ids):
        source = ROOT / 'p4_sensor_candidates' / f'{did}.npz'
        signature = {**fingerprint, 'raw': sha256(source)}
        cache = OUT / f'{did}.npz'
        if cache.exists():
            with np.load(cache) as z:
                assert json.loads(str(z['fingerprint'])) == signature, 'stale acquisition cache'
                source_rows, added, fresh = z['source_indices'], z['additions'], z['fresh107']
                replay = float(z['max_replay_error'])
            print('Resume two-mode', did, flush=True)
        else:
            with np.load(source) as z:
                raw, source_rows, sigma = z['raw'], z['source_indices'], z['noise_scale']
            added = np.full((len(names), len(source_rows), 67), np.nan)
            fresh = np.full((len(FRESH), len(source_rows), 107), np.nan)
            replay = 0.
            for j, index in enumerate(source_rows):
                k = rowmap[int(index)]
                assert ids[k] == did
                white = {s: np.random.default_rng(np.random.SeedSequence([s, int(index)])).normal(size=raw[j].shape)
                         for s in (928, 929, 932, 935, 937)}
                common = {s: np.random.default_rng(np.random.SeedSequence([s, int(index)])).normal(size=(1, raw[j].shape[1]))
                          for s in (931, 933, 936, 938)}
                modes_by_name = {}
                for ni, name in enumerate(names):
                    if name.startswith('dt_'):
                        factor = .98 if name == 'dt_minus2pct' else 1.02
                        base = modes_by_name['iid1_seed929']
                        modes = {**base, 'alpha': base['alpha']/factor, 'frequency': base['frequency']/factor}
                    else:
                        if name == 'clean':
                            measured = raw[j]
                        elif name.startswith('iid'):
                            level = .01 if name.startswith('iid1') else .05
                            seed = int(name[-3:])
                            measured = raw[j] + level*sigma[j]*white[seed]
                        elif name == 'gain_bias':
                            measured = raw[j] + .01*sigma[j]*white[929]
                        elif name.startswith('correlated'):
                            seeds = {'correlated_train_draw': (929, 931),
                                     'correlated_fresh932_933': (932, 933),
                                     'correlated_fresh935_936': (935, 936),
                                     'correlated_fresh937_938': (937, 938)}[name]
                            measured = raw[j] + .01*sigma[j]*(np.sqrt(.2)*white[seeds[0]] + np.sqrt(.8)*common[seeds[1]])
                        else:
                            raise ValueError(f'unknown condition {name}')
                        signal = causal_calibration_normalize(measured, calibration_samples=51)[layout]
                        if name == 'gain_bias':
                            signal = signal*gains[:, None]
                        signal = np.clip(signal, -50, 50).astype(np.float32)
                        modes = estimate_modes(signal, float(dt[k]), include_shapes=True)
                    modes_by_name[name] = modes
                    primary = modal_spatial_row(modes, pairs)
                    if ni < nh:
                        modal = modal_feature_row(modes)
                        np.testing.assert_allclose(modal, base_reference[ni, k, 7:22], atol=1e-10, rtol=1e-10)
                        np.testing.assert_allclose(primary, spatial_reference[ni, k], atol=1e-10, rtol=1e-10)
                        replay = max(replay, float(abs(modal-base_reference[ni, k, 7:22]).max()),
                                     float(abs(primary-spatial_reference[ni, k]).max()))
                    else:
                        fresh[ni-nh, j] = np.r_[base_feature_row(signal, float(dt[k]), modes, ridge), primary]
                    added[ni, j] = modal_spatial_addition_row(modes, pairs)
            with cache.with_suffix('.tmp').open('wb') as handle:
                np.savez_compressed(handle, source_indices=source_rows, additions=added, fresh107=fresh,
                                    max_replay_error=replay, fingerprint=json.dumps(signature))
            cache.with_suffix('.tmp').replace(cache)
            print('Extracted two-mode', did, 'rows', len(source_rows), flush=True)
        target = np.array([rowmap[int(row)] for row in source_rows])
        assert set(target) == set(np.flatnonzero(ids == did))
        additions[:, target] = added
        original[nh:, target] = fresh
        max_replay = max(max_replay, replay)
    assert np.isfinite(original).all() and np.isfinite(additions).all()
    assert np.all((additions[:, :, -2] >= 0) & (additions[:, :, -2] <= 1+1e-12))
    train = [names.index(c) for c in TRAIN_VIEWS]
    path = OUT / 'features.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, evaluation107=original, evaluation_additions=additions,
                            training107=original[train], training_additions=additions[train],
                            y=y, ids=ids, source_indices=rows, conditions=names,
                            sensor_indices=layout, sensor_pairs=pairs,
                            max_replay_error=max_replay, fingerprint=json.dumps(fingerprint))
    path.with_suffix('.tmp').replace(path)
    print('Completed acquisition replay', original.shape, additions.shape, 'max prior delta', max_replay, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
