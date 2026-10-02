"""Replay training-design acquisitions; retain only waveform-estimated spatial features."""
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'p4_groups/correlated_training_20260929'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(PARENT)]
import numpy as np
from threadpoolctl import threadpool_limits
from generate_p4_sensor_candidates import candidate_positions
from mechanics.p4_margin_estimation.modal_features import estimate_modes, modal_feature_row, modal_spatial_row
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
from run_augmentation_experiment import sha256


def sensor_graph(layout):
    positions = candidate_positions()[0][layout]
    rank = np.column_stack([np.searchsorted(np.unique(positions[:, k]), positions[:, k]) for k in range(2)])
    pairs = np.array([(i,j) for i in range(len(layout)) for j in range(i+1,len(layout))
                      if abs(rank[i]-rank[j]).sum() == 1], dtype=int)
    assert len(pairs) == 24
    return positions, pairs


def main():
    with np.load(PARENT / 'shared.npz') as z:
        ids, rows, y, names = z['ids'], z['source_indices'], z['y'], z['evaluation_names'].tolist()
        reference = z['evaluation']
        np.testing.assert_array_equal(z['train3'], reference[[names.index(c) for c in ('clean','iid1_seed928','iid5_seed928')]])
        np.testing.assert_array_equal(z['independent_extra'], reference[names.index('iid1_seed929')])
    acquisition = json.loads((ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())
    layout = np.asarray(acquisition['selected_sensor_indices'])
    gains = np.asarray(acquisition['gains'])
    positions, pairs = sensor_graph(layout)
    with np.load(ROOT / 'p4_dataset_saturation/metadata_arrays.npz') as z:
        dt = z['dts'][rows]
    rowmap = {int(row): i for i,row in enumerate(rows)}
    fingerprint = {
        'protocol': sha256(OUT / 'protocol.json'), 'execution': sha256(Path(__file__)),
        'shared_features': sha256(PARENT / 'shared.npz'),
        'acquisition_protocol': sha256(ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json'),
        'source': sha256(ROOT / 'src/mechanics/p4_margin_estimation/modal_features.py'),
        'timebase_source': sha256(ROOT / 'p4_dataset_saturation/metadata_arrays.npz')}
    features = np.full((len(names),len(rows),65), np.nan)
    max_replay_error = 0.
    for did in np.unique(ids):
        source = ROOT / 'p4_sensor_candidates' / f'{did}.npz'
        shard_fingerprint = {**fingerprint, 'raw_acquisition': sha256(source)}
        cache = OUT / f'{did}.npz'
        if cache.exists():
            with np.load(cache) as z:
                assert json.loads(str(z['fingerprint'])) == shard_fingerprint, 'stale spatial cache'
                source_rows, spatial, replay_error = z['source_indices'], z['features'], float(z['max_replay_error'])
            print('Resume spatial', did, flush=True)
        else:
            with np.load(source) as z:
                raw, source_rows, sigma = z['raw'], z['source_indices'], z['noise_scale']
            spatial = np.full((len(names),len(source_rows),65), np.nan)
            replay_error = 0.
            for j,index in enumerate(source_rows):
                k = rowmap[int(index)]
                assert str(ids[k]) == str(did)
                white = {seed: np.random.default_rng(np.random.SeedSequence([seed,int(index)])).normal(size=raw[j].shape)
                         for seed in (928,929,932)}
                common = {seed: np.random.default_rng(np.random.SeedSequence([seed,int(index)])).normal(size=(1,raw[j].shape[1]))
                          for seed in (931,933)}
                modes_by_name = {}
                for ni,name in enumerate(names):
                    if name.startswith('dt_'):
                        factor = .98 if name == 'dt_minus2pct' else 1.02
                        base = modes_by_name['iid1_seed929']
                        modes = {**base, 'alpha': base['alpha']/factor, 'frequency': base['frequency']/factor}
                    else:
                        if name == 'clean':
                            measured = raw[j]
                        elif name.startswith('iid'):
                            level = .01 if name.startswith('iid1') else .05
                            seed = 928 if '928' in name else (929 if '929' in name else 932)
                            measured = raw[j]+level*sigma[j]*white[seed]
                        elif name == 'gain_bias':
                            measured = raw[j]+.01*sigma[j]*white[929]
                        elif name == 'correlated_train_draw':
                            measured = raw[j]+.01*sigma[j]*(np.sqrt(.2)*white[929]+np.sqrt(.8)*common[931])
                        elif name == 'correlated_fresh932_933':
                            measured = raw[j]+.01*sigma[j]*(np.sqrt(.2)*white[932]+np.sqrt(.8)*common[933])
                        else:
                            raise ValueError(f'unknown acquisition condition: {name}')
                        selected = causal_calibration_normalize(measured, calibration_samples=51)[layout]
                        if name == 'gain_bias':
                            selected = selected*gains[:,None]
                        signal = np.clip(selected,-50,50).astype(np.float32)
                        modes = estimate_modes(signal,float(dt[k]),include_shapes=True)
                    modes_by_name[name] = modes
                    row = modal_feature_row(modes)
                    np.testing.assert_allclose(row, reference[ni,k,7:22], atol=1e-10, rtol=1e-10)
                    replay_error = max(replay_error,float(abs(row-reference[ni,k,7:22]).max()))
                    spatial[ni,j] = modal_spatial_row(modes,pairs)
            temporary = cache.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, features=spatial, source_indices=source_rows,
                                    max_replay_error=replay_error, fingerprint=json.dumps(shard_fingerprint))
            temporary.replace(cache)
            print('Extracted spatial', did, 'rows',len(source_rows),flush=True)
        target_rows = np.array([rowmap[int(row)] for row in source_rows])
        assert set(target_rows) == set(np.flatnonzero(ids == did))
        features[:,target_rows] = spatial
        max_replay_error = max(max_replay_error,replay_error)
    assert np.isfinite(features).all()
    view_names = ['clean','iid1_seed928','iid5_seed928','iid1_seed929']
    output = OUT / 'spatial_features.npz'
    with output.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, evaluation_spatial=features,
                            training_spatial=features[[names.index(c) for c in view_names]],
                            y=y, ids=ids, source_indices=rows, conditions=names,
                            sensor_indices=layout, sensor_positions_yx=positions, sensor_pairs=pairs,
                            max_modal_replay_error=max_replay_error, fingerprint=json.dumps(fingerprint))
    output.with_suffix('.tmp').replace(output)
    print('Completed spatial replay:',features.shape,'max modal delta',max_replay_error,flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
