"""Frozen separate-family ablation on existing measured development trajectories."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import cloudpickle
from threadpoolctl import threadpool_limits
import cuda_backend as C
import diagnostic_matrix as M
import pulse_aware_probe as P
import temporal_convergence as T
import spectral_features as S
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge, fit_ridge_scaling

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUT = SCRIPT / 'additive_spectral_20261009'
SOURCE = SCRIPT / 'fitter_margin_comparison_20261008'
FREQUENCY_COLUMNS = np.array([8, 11, 14])  # Seven signal summaries, then modal(alpha,f,energy) triples.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for field in ('inputs', 'helpers'):
        for name, digest in protocol[field].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError(f'Frozen {field} changed: {name}')
    if 'implementation' in protocol:
        for name, digest in protocol['implementation'].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError(f'Frozen implementation changed: {name}')
    return protocol


def key(record):
    return tuple(record[name] for name in ('split', 'design', 'replicate', 'profile', 'noise'))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        data = {name: z[name].copy() for name in ('features', 'truth', 'observed', 'age', 'times')}
        data['records'] = json.loads(str(z['records']))
    return data


def append(signal, features, observed, ends):
    values = np.full((len(ends), 3), np.nan)
    valid = np.zeros((len(ends), 3), dtype=bool)
    bins = np.full((len(ends), 3), -1, dtype=np.int16)
    for frame in np.flatnonzero(observed):
        window = signal[:, ends[frame] - 128:ends[frame]]
        values[frame], valid[frame], bins[frame] = S.descriptors(window, features[frame, FREQUENCY_COLUMNS])
    return values, valid, bins


def fresh_features(signals, command, times, d, engine):
    ends = np.arange(128, signals[0].shape[1] + 1, 64)
    q = np.zeros((len(signals), len(ends), 174))
    observed = np.zeros(q.shape[:2], dtype=bool)
    age = np.zeros(q.shape[:2])
    last = np.zeros(len(signals))
    for frame, end in enumerate(ends):
        windows, indices = [], []
        for i, signal in enumerate(signals):
            window = signal[:, end - 128:end]
            if not command[end - 128:end].any() and np.isfinite(window).all() and not (abs(window) >= 50).any():
                windows.append(window)
                indices.append(i)
        if windows:
            prefix = C.estimate_batch([window[:, :64] for window in windows])
            skills = P.forecast_skills(windows, prefix, 64, 64)
            chosen = [i for i, skill in enumerate(skills) if skill >= .5]
            modes = C.estimate_batch([windows[i] for i in chosen]) if chosen else []
            for i, mode in zip(chosen, modes):
                if len(mode['alpha']):
                    row = M.features(windows[i], mode, d, engine)
                    if np.isfinite(row).all():
                        q[indices[i], frame] = row
                        observed[indices[i], frame] = True
                        last[indices[i]] = times[frame]
        age[:, frame] = times[frame] - last
    return q, observed, age


def extract():
    protocol = contract()
    if (OUT / 'dataset.npz').exists():
        raise RuntimeError('Acquired experiment is immutable')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA acquisition required, no numerical CPU fallback')
    checks = S.smoke()
    data = load(SOURCE / 'dataset.npz')
    canonical = load(SCRIPT / 'temporal_probe_20261007/data.npz')
    canonical_index = {key(record): i for i, record in enumerate(canonical['records'])}
    replaced = 0
    for i, record in enumerate(data['records']):
        if key(record) in canonical_index:
            j = canonical_index[key(record)]
            for name in ('truth', 'observed', 'age'):
                np.testing.assert_array_equal(data[name][i], canonical[name][j])
            np.testing.assert_allclose(data['features'][i], canonical['features'][j], atol=1e-5, rtol=0)
            data['features'][i] = canonical['features'][j]
            replaced += 1
    with np.load(SOURCE / 'raw_signals.npz', allow_pickle=False) as z:
        physical, commands = z['physical'], z['command']
        raw_records = json.loads(str(z['records']))
    with np.load(SCRIPT / 'remote_input.npz', allow_pickle=False) as z:
        d = cloudpickle.loads(z['data'].tobytes())
    engine = SimpleNamespace(ridge=PhysicsFeatureRidge())
    index = {key(record): i for i, record in enumerate(data['records'])}
    ends = np.arange(128, physical.shape[-1] + 1, 64)
    values = np.full(data['observed'].shape + (3,), np.nan)
    valid = np.zeros(values.shape, dtype=bool)
    bins = np.full(values.shape, -1, dtype=np.int16)
    extra_q, extra_y, extra_observed, extra_age, extra_values, extra_valid, extra_bins, extra_records = ([] for _ in range(8))
    for raw_index, record in enumerate(raw_records):
        raw = physical[raw_index]
        seed = np.random.SeedSequence([20261010, int(record['design'][1:]), record['replicate'],
                                      ('cross', 'stable', 'retreat').index(record['profile'])])
        noises, _ = M.noises(raw, seed)
        for noise in protocol['evaluation']['original_views']:
            i = index[key(dict(record, noise=noise))]
            signal = M.normalize(raw, noises[noise])[0]
            values[i], valid[i], bins[i] = append(signal, data['features'][i], data['observed'][i], ends)
        if record['split'] == 'held':
            source_i = index[key(dict(record, noise='clean'))]
            for fresh_seed in protocol['evaluation']['fresh_draw_seeds']:
                noise_seed = np.random.SeedSequence([fresh_seed, int(record['design'][1:]), record['replicate'],
                                                    ('cross', 'stable', 'retreat').index(record['profile'])])
                fresh_noises, _ = M.noises(raw, noise_seed)
                noise_names = protocol['evaluation']['fresh_views']
                signals = [M.normalize(raw, fresh_noises[name])[0] for name in noise_names]
                q, observed, age = fresh_features(signals, commands[raw_index], data['times'], d, engine)
                for j, noise in enumerate(noise_names):
                    v, mask, centers = append(signals[j], q[j], observed[j], ends)
                    extra_q.append(q[j]); extra_y.append(data['truth'][source_i])
                    extra_observed.append(observed[j]); extra_age.append(age[j])
                    extra_values.append(v); extra_valid.append(mask); extra_bins.append(centers)
                    extra_records.append(dict(record, noise=f'{noise}_fresh{fresh_seed}',
                                              accepted_windows=int(observed[j].sum()), frames=len(ends)))
        print('ADDITIVE_EXTRACTED', record['design'], record['replicate'], record['profile'], flush=True)
    for name, added in (('features', extra_q), ('truth', extra_y), ('observed', extra_observed), ('age', extra_age)):
        data[name] = np.concatenate((data[name], np.stack(added)))
    data['records'].extend(extra_records)
    values = np.concatenate((values, np.stack(extra_values)))
    valid = np.concatenate((valid, np.stack(extra_valid)))
    bins = np.concatenate((bins, np.stack(extra_bins)))
    assert np.array_equal(np.isfinite(values), valid)
    assert not valid[~data['observed']].any()
    np.savez_compressed(OUT / 'dataset.npz', **{name: data[name] for name in ('features', 'truth', 'observed', 'age', 'times')},
                        records=json.dumps(data['records']), descriptors=values, descriptor_valid=valid, bins=bins,
                        protocol_sha256=sha(OUT / 'protocol.json'))
    acquisition = dict(smoke=checks, canonical_rows=replaced, original_records=len(index), fresh_records=len(extra_records),
                       original_acceptance_retained=True, source_hashes=protocol['inputs'], gpu=torch.cuda.get_device_name())
    (OUT / 'acquisition.json').write_text(json.dumps(acquisition, indent=2) + '\n')
    print('ADDITIVE_ACQUIRED', json.dumps(acquisition), flush=True)


def input_matrix(data, values, valid, columns, proper, shuffled=False):
    q, observed = data['features'], data['observed']
    mean, scale, active = fit_ridge_scaling(q[proper][observed[proper]])
    normalized = (q - mean) / scale
    normalized[..., ~active] = 0
    normalized[~observed] = 0
    blocks = [normalized, observed[..., None], data['age'][..., None]]
    if columns:
        v, mask = values[..., columns], valid[..., columns]
        impute = np.array([v[proper, :, j][mask[proper, :, j]].mean() if mask[proper, :, j].any() else 0.
                           for j in range(len(columns))])
        imputed = np.where(mask, v, impute)
        a_mean, a_scale, a_active = fit_ridge_scaling(imputed[proper][observed[proper]])
        extra = (imputed - a_mean) / a_scale
        extra[..., ~a_active] = 0
        if shuffled:
            rng = np.random.default_rng(20261009)
            for split in ('train', 'validation', 'held'):
                groups = sorted(set((r['design'], r['replicate'], r['profile']) for r in data['records'] if r['split'] == split))
                permutation = rng.permutation(len(groups))
                group_map = {group: groups[permutation[j]] for j, group in enumerate(groups)}
                lookup = {(r['design'], r['replicate'], r['profile'], r['noise']): j for j, r in enumerate(data['records']) if r['split'] == split}
                original = extra.copy()
                for j, record in enumerate(data['records']):
                    if record['split'] == split:
                        donor = (*group_map[(record['design'], record['replicate'], record['profile'])], record['noise'])
                        extra[j] = original[lookup[donor]]
        extra[~observed] = 0
        # Raw, explicit descriptor validity flags. They are not variance-scaled or permuted.
        blocks.extend((extra, mask.astype(np.float64)))
    return np.concatenate(blocks, axis=2).astype(np.float32), (mean, scale, active)


def train_one(x, y, proper, validation, arm, seed, protocol):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = T.make_model('framewise', x)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
    xt, yt, xv, yv = x[proper], y[proper], x[validation], y[validation]
    best, best_epoch, stale, history, best_state = np.inf, None, 0, [], None
    for epoch in range(protocol['training']['maximum_epochs']):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = ((model(xt)[..., 0] - yt) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite training loss')
        loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad():
            mae = float(abs(model(xv)[..., 0] - yv).mean())
        history.append(dict(epoch=epoch + 1, train_mse=float(loss.detach()), validation_mae=mae))
        if mae < best:
            best, best_epoch, stale = mae, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= protocol['training']['patience']:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        point = model(x)[..., 0].cpu().numpy()
    if not np.isfinite(point).all():
        raise RuntimeError('Nonfinite trained predictions')
    torch.save(dict(state=best_state, arm=arm, seed=seed, protocol_sha256=sha(OUT / 'protocol.json')),
               OUT / f'{arm}_seed{seed}.pt')
    meta = dict(seed=seed, best_epoch=best_epoch, epochs=len(history), validation_mae=best, history=history,
                input_width=x.shape[-1], parameters=sum(p.numel() for p in model.parameters()))
    print('ADDITIVE_TRAINED', arm, seed, best_epoch, best, flush=True)
    return point, meta


def train():
    protocol = contract()
    if (OUT / 'training.json').exists():
        raise RuntimeError('Completed training is immutable')
    if not torch.cuda.is_available():
        raise RuntimeError('Matched training requires CUDA')
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = load(OUT / 'dataset.npz')
    with np.load(OUT / 'dataset.npz', allow_pickle=False) as z:
        assert str(z['protocol_sha256']) == sha(OUT / 'protocol.json')
        values, valid = z['descriptors'], z['descriptor_valid']
    proper = np.array([r['split'] == 'train' and r['noise'] in ('clean', 'iid1') for r in data['records']])
    validation = np.array([r['split'] == 'validation' and r['noise'] in ('clean', 'iid1') for r in data['records']])
    assert set(r['design'] for r, keep in zip(data['records'], proper) if keep) == set(protocol['split']['train'])
    assert set(r['design'] for r, keep in zip(data['records'], validation) if keep) == set(protocol['split']['validation'])
    y = torch.as_tensor(data['truth'], device='cuda', dtype=torch.float32)
    prediction_rows, fits = [], {}
    for arm in protocol['arms']:
        columns = [0, 1] if 'spectral' in arm else [2] if 'spatial' in arm else []
        array, scaling = input_matrix(data, values, valid, columns, proper, arm.startswith('shuffled_'))
        x = torch.as_tensor(array, device='cuda')
        points, arm_fits = [], []
        for seed in protocol['training']['seeds']:
            point, metadata = train_one(x, y, proper, validation, arm, seed, protocol)
            points.append(point); arm_fits.append(metadata)
        prediction_rows.append(np.stack(points)); fits[arm] = arm_fits
        if arm == 'baseline174':
            with np.load(SCRIPT / 'temporal_convergence_20261007/predictions.npz', allow_pickle=False) as z:
                for actual, name in zip(scaling, ('mean', 'scale', 'active')):
                    np.testing.assert_array_equal(actual, z[name])
                old_records = json.loads(str(z['records']))
                current_index = {key(r): i for i, r in enumerate(data['records'])}
                old_indices = [current_index[key(r)] for r in old_records]
                expected = z['predictions'][0]
                np.testing.assert_allclose(points[0][old_indices], expected, rtol=0, atol=2e-6)
                reproduction_error = float(abs(points[0][old_indices] - expected).max())
        del x
    np.savez_compressed(OUT / 'predictions.npz', predictions=np.stack(prediction_rows),
                        arms=json.dumps(protocol['arms']), seeds=protocol['training']['seeds'],
                        dataset_sha256=sha(OUT / 'dataset.npz'), protocol_sha256=sha(OUT / 'protocol.json'))
    result = dict(fits=fits, baseline_seed42_reproduction_error=reproduction_error,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, dataset_sha256=sha(OUT / 'dataset.npz'),
                  protocol_sha256=sha(OUT / 'protocol.json'), predictions_sha256=sha(OUT / 'predictions.npz'))
    (OUT / 'training.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ADDITIVE_TRAINING_COMPLETE', result['gpu'], reproduction_error, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('extract', 'train'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.phase]()
