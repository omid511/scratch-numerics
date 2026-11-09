"""Acquire same-observation waveform features and waveform-only modal fits."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits

import cuda_backend as C
import diagnostic_matrix as M
import gpu_execution as G
import pulse_aware_probe as P
import temporal_probe as T
from time_varying_modal_fit import fit_single_mode

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT / 'fitter_margin_comparison_20261008'
PROTOCOL = json.loads((OUT / 'protocol.json').read_text())
FIT_PROTOCOL = json.loads((SCRIPT / 'time_varying_fit_20261007' / 'protocol.json').read_text())
FIT_SETTINGS = FIT_PROTOCOL['solver']
NOISES = tuple(PROTOCOL['noise'])
PROFILES = tuple(T.PROTOCOL['profiles'])
FEATURE_TOLERANCE = 1e-5
WINDOW, STRIDE, PREFIX = 128, 64, 64


def _key(record):
    return (record['split'], record['design'], int(record['replicate']),
            record['profile'], record['noise'])


def _raw_records(smoke):
    rows = []
    for split, designs in PROTOCOL['split'].items():
        for design in designs[:1] if smoke else designs:
            for replicate in range(1 if smoke else T.PROTOCOL['schedule_replicates']):
                for profile in PROFILES:
                    rows.append(dict(design=design, split=split, replicate=replicate,
                                     profile=profile,
                                     schedule=M.clean_json(T.schedule(design, replicate, profile))))
    return rows


def _hashes():
    input_hashes = {}
    for relative, expected in PROTOCOL['inputs'].items():
        path = G.O.SCRIPT / relative
        actual = G.O.C.sha256(path)
        if actual != expected:
            raise RuntimeError(f'Frozen input hash mismatch: {relative}')
        input_hashes[relative] = actual
    source_paths = {
        'comparison_acquisition': Path(__file__), 'temporal_probe': Path(T.__file__),
        'diagnostic_matrix': Path(M.__file__), 'pulse_aware_probe': Path(P.__file__),
        'gpu_execution': Path(G.__file__), 'cuda_backend': Path(C.__file__),
        'trajectory_runner': Path(G.O.__file__),
        'time_varying_modal_fit': SCRIPT / 'time_varying_modal_fit.py',
    }
    sources = {name: G.O.C.sha256(path) for name, path in source_paths.items()}
    physical_names = ('temporal_probe', 'diagnostic_matrix', 'gpu_execution',
                      'cuda_backend', 'trajectory_runner')
    physical = {name: sources[name] for name in physical_names}
    physical['temporal_probe_protocol'] = G.O.C.sha256(
        G.O.SCRIPT / 'temporal_probe_20261007' / 'protocol.json')
    return input_hashes, sources, physical


def _protocol_hash():
    if json.loads((OUT / 'protocol.json').read_text()) != PROTOCOL:
        raise RuntimeError('Comparison protocol changed after module initialization')
    return G.O.C.sha256(OUT / 'protocol.json')


def _runtime():
    return dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                scipy=scipy.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(torch.cuda.current_device()))


def _physical_cache(path, smoke, fp, input_hashes, physical_sources, protocol_hash, d):
    expected = _raw_records(smoke)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            physical, command = z['physical'].copy(), z['command'].copy()
            records = json.loads(str(z['records']))
            protocol = json.loads(str(z['protocol']))
            provenance = json.loads(str(z['provenance']))
        if (protocol != PROTOCOL or provenance.get('fingerprint') != fp
                or provenance.get('comparison_protocol_sha256') != protocol_hash):
            raise RuntimeError(f'Raw-waveform cache contract mismatch: {path}')
        if (provenance.get('input_hashes') != input_hashes
                or provenance.get('physical_source_hashes') != physical_sources):
            raise RuntimeError(f'Raw-waveform cache inputs changed: {path}')
        if provenance.get('smoke') is not smoke or records != expected:
            raise RuntimeError(f'Raw-waveform cache schedule mismatch: {path}')
        if (physical.ndim != 3 or physical.shape[0] != len(records) or physical.shape[2] != 8001
                or command.shape != (len(records), 8001) or command.dtype.kind != 'b'):
            raise RuntimeError(f'Raw-waveform cache shape mismatch: {path}')
        if not np.isfinite(physical).all():
            raise RuntimeError(f'Raw-waveform cache contains nonfinite samples: {path}')
        print('COMPARISON_RAW_CACHE_REUSED', path.name, len(records), flush=True)
        return physical, command, records

    physical_rows, command_rows = [], []
    for item in expected:
        _, plate = M.context(d, fp, item)
        description = {key: np.asarray(value) if key in ('knots', 'ratios', 'onsets', 'amplitudes') else value
                       for key, value in item['schedule'].items()}
        times, response, command = T.acquire(plate, description, item['replicate'])
        if len(times) != 8001 or response.shape[1] != len(times) or not np.isfinite(response).all():
            raise RuntimeError(f'Invalid physical response for {item["design"]}/{item["profile"]}')
        physical_rows.append(response)
        command_rows.append(command)
        print('COMPARISON_RAW_ACQUIRED', item['design'], item['replicate'], item['profile'], flush=True)
    physical, command = np.stack(physical_rows), np.stack(command_rows).astype(bool, copy=False)
    provenance = dict(fingerprint=fp, input_hashes=input_hashes,
                      physical_source_hashes=physical_sources,
                      comparison_protocol_sha256=protocol_hash, smoke=smoke)
    G.O.C.save_npz(path, physical=physical, command=command, records=json.dumps(expected),
                   protocol=json.dumps(PROTOCOL), provenance=json.dumps(provenance))
    return physical, command, expected


def _load_models(d, fp):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU propagation or inference fallback')
    models, hashes = [], {}
    for width in (107, 174):
        path = G.O.OUT / f'trajectory_baseline{width}.npz'
        hashes[str(width)] = G.O.C.sha256(path)
        with np.load(path) as z:
            if json.loads(str(z['fingerprint'])) != fp:
                raise RuntimeError(f'Frozen spatial predictor fingerprint mismatch: {path.name}')
            models.append(G.O.C.cloudpickle.loads(z['predictor'].tobytes()))
    return C.CudaInference(models, d['pairs']), hashes


def _fit_window(signal):
    stationary = fit_single_mode(signal, .001, varying=False, settings=FIT_SETTINGS)
    varying = fit_single_mode(signal, .001, varying=True, settings=FIT_SETTINGS,
                              stationary=stationary)
    interior = all(result.get('success') is True and not result.get('bound_hits')
                   for result in (stationary, varying))
    values = None
    if interior:
        alpha, frequency = varying['alpha_end'], varying['frequency_end']
        values = np.asarray((alpha, frequency, varying['parameters']['alpha_slope'],
                             varying['parameters']['frequency_slope'],
                             alpha / (2 * np.pi * frequency),
                             varying['relative_fit_error']), dtype=np.float64)
        if not np.isfinite(values).all():
            values = None
    return stationary, varying, values


def _process(item, raw, command, offset, records, fit_records, d, engine):
    times = np.arange(8001, dtype=np.float64) / 1000.
    schedule = item['schedule']
    description = {key: np.asarray(value) if key in ('knots', 'ratios', 'onsets', 'amplitudes') else value
                   for key, value in schedule.items()}
    seed = np.random.SeedSequence([20261010, int(item['design'][1:]), item['replicate'],
                                   PROFILES.index(item['profile'])])
    noises, _ = M.noises(raw, seed)
    signals = [M.normalize(raw, noises[name])[0] for name in NOISES]
    ends = np.arange(WINDOW, len(times) + 1, STRIDE)
    frame_times = times[ends - 1]
    truth = 1 - np.interp(frame_times, description['knots'], description['ratios'])
    count = len(NOISES)
    features = np.zeros((count, len(ends), 174), dtype=np.float64)
    observed = np.zeros((count, len(ends)), dtype=bool)
    age = np.zeros((count, len(ends)), dtype=np.float64)
    fitter = np.zeros((count, len(ends), 6), dtype=np.float64)
    fitter_observed = np.zeros((count, len(ends)), dtype=bool)
    fitter_age = np.zeros((count, len(ends)), dtype=np.float64)
    last, last_fit = np.zeros(count), np.zeros(count)
    candidates, qualified, attempts = np.zeros(count, int), np.zeros(count, int), np.zeros(count, int)
    station_failures, varying_failures = np.zeros(count, int), np.zeros(count, int)

    for frame, end in enumerate(ends):
        windows, indices = [], []
        for noise_index, signal in enumerate(signals):
            window = signal[:, end - WINDOW:end]
            if (not command[end - WINDOW:end].any() and np.isfinite(window).all()
                    and not (np.abs(window) >= 50).any()):
                candidates[noise_index] += 1
                windows.append(window)
                indices.append(noise_index)
        source_noises = {NOISES.index(name) for name in ('clean', 'iid1')}
        groups = ([k for k, noise_index in enumerate(indices) if noise_index in source_noises],
                  [k for k, noise_index in enumerate(indices) if noise_index not in source_noises])
        for group in groups:
            if not group:
                continue
            group_windows = [windows[k] for k in group]
            prefix_modes = C.estimate_batch([window[:, :PREFIX] for window in group_windows])
            skills = P.forecast_skills(group_windows, prefix_modes, PREFIX, WINDOW - PREFIX)
            selected = [group[k] for k, skill in enumerate(skills) if skill >= .5]
            for k in selected:
                qualified[indices[k]] += 1
            if not selected:
                continue
            modes = C.estimate_batch([windows[k] for k in selected])
            for k, mode in zip(selected, modes):
                noise_index = indices[k]
                if not len(mode['alpha']):
                    continue
                row = M.features(windows[k], mode, d, engine)
                if row.shape != (174,) or not np.isfinite(row).all():
                    continue
                features[noise_index, frame] = row
                observed[noise_index, frame] = True
                station, varying, values = _fit_window(windows[k])
                attempts[noise_index] += 1
                station_failures[noise_index] += int(station.get('success') is not True
                                                     or bool(station.get('bound_hits')))
                varying_failures[noise_index] += int(varying.get('success') is not True
                                                      or bool(varying.get('bound_hits')))
                fit_records.append(dict(record_index=offset + noise_index, frame_index=frame,
                                        stationary=station, varying=varying))
                if values is not None:
                    fitter[noise_index, frame] = values
                    fitter_observed[noise_index, frame] = True
                    last_fit[noise_index] = frame_times[frame]
        for noise_index in range(count):
            if observed[noise_index, frame]:
                last[noise_index] = frame_times[frame]
            age[noise_index, frame] = frame_times[frame] - last[noise_index]
            fitter_age[noise_index, frame] = frame_times[frame] - last_fit[noise_index]

    for noise_index, name in enumerate(NOISES):
        records[offset + noise_index].update(
            finite_command_free_windows=int(candidates[noise_index]),
            forecast_qualified_windows=int(qualified[noise_index]),
            accepted_windows=int(observed[noise_index].sum()), fit_attempts=int(attempts[noise_index]),
            stationary_failures=int(station_failures[noise_index]),
            varying_failures=int(varying_failures[noise_index]),
            fitter_observed=int(fitter_observed[noise_index].sum()), frames=len(ends))
    print('COMPARISON_TRAJECTORY_PROCESSED', item['design'], item['replicate'], item['profile'],
          'accepted', observed.sum(axis=1).tolist(), 'fitter', fitter_observed.sum(axis=1).tolist(), flush=True)
    return features, truth, observed, age, fitter, fitter_observed, fitter_age, frame_times


def _verify_source(dataset, smoke):
    source = G.O.SCRIPT / 'temporal_probe_20261007' / 'data.npz'
    with np.load(source) as z:
        if json.loads(str(z['protocol'])) != T.PROTOCOL:
            raise RuntimeError('Original temporal-probe dataset protocol mismatch')
        original = {key: z[key].copy() for key in ('features', 'truth', 'observed', 'age', 'times')}
        source_records = json.loads(str(z['records']))
    old = {_key(record): index for index, record in enumerate(source_records)}
    new = {_key(record): index for index, record in enumerate(dataset['records'])}
    if len(old) != len(source_records) or len(new) != len(dataset['records']):
        raise RuntimeError('Duplicate source or comparison record key')
    selected = [key for key in old if key in new and key[-1] in ('clean', 'iid1')]
    expected = {(split, design, replicate, profile, noise)
                for split, designs in PROTOCOL['split'].items()
                for design in (designs[:1] if smoke else designs)
                for replicate in range(1 if smoke else T.PROTOCOL['schedule_replicates'])
                for profile in PROFILES for noise in ('clean', 'iid1')}
    if set(selected) != expected or len(selected) != len(expected):
        raise RuntimeError('Original clean/iid1 source subset is incomplete')
    if not np.array_equal(dataset['times'], original['times']):
        raise RuntimeError('Frame times differ from the original temporal-probe dataset')
    maximum = 0.0
    for key in selected:
        new_index, old_index = new[key], old[key]
        for name in ('truth', 'observed', 'age'):
            if not np.array_equal(dataset[name][new_index], original[name][old_index]):
                raise RuntimeError(f'Original {name} mismatch for {key}')
        difference = np.abs(dataset['features'][new_index] - original['features'][old_index])
        maximum = max(maximum, float(difference.max(initial=0.0)))
        if not np.allclose(dataset['features'][new_index], original['features'][old_index],
                           rtol=0.0, atol=FEATURE_TOLERANCE):
            raise RuntimeError(f'Original 174-feature mismatch for {key}; max_abs={difference.max()}')
    return dict(source=str(source.relative_to(G.O.SCRIPT)), records_compared=len(selected),
                feature_tolerance_absolute=FEATURE_TOLERANCE,
                maximum_absolute_feature_difference=maximum)


def _load_complete(path, input_hashes, source_hashes, predictor_hashes,
                   fp, smoke, raw_hash, protocol_hash):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        data = {name: z[name].copy() for name in
                ('features', 'truth', 'observed', 'age', 'times', 'fitter_features',
                 'fitter_observed', 'fitter_age')}
        data['records'] = json.loads(str(z['records']))
        fit_records = json.loads(str(z['fit_records']))
        if json.loads(str(z['protocol'])) != PROTOCOL:
            raise RuntimeError(f'Completed output protocol mismatch: {path}')
        provenance = json.loads(str(z['provenance']))
    expected = 45 if smoke else 240
    if (data['features'].shape != (expected, 124, 174)
            or data['fitter_features'].shape != (expected, 124, 6)
            or len(data['records']) != expected
            or len(fit_records) != sum(record.get('fit_attempts', 0) for record in data['records'])
            or any(not {'record_index', 'frame_index', 'stationary', 'varying'}.issubset(record)
                   for record in fit_records)):
        raise RuntimeError(f'Completed output is incomplete or malformed: {path}')
    if (provenance.get('input_hashes') != input_hashes
            or provenance.get('source_hashes') != source_hashes
            or provenance.get('spatial_predictor_sha256') != predictor_hashes
            or provenance.get('fingerprint') != fp or provenance.get('smoke') != smoke
            or provenance.get('raw_signals_sha256') != raw_hash
            or provenance.get('comparison_protocol_sha256') != protocol_hash):
        raise RuntimeError(f'Refusing to overwrite completed output with changed inputs: {path}')
    _verify_source(data, smoke)
    print('COMPARISON_COMPLETED_OUTPUT_REUSED', path.name, len(data['records']), flush=True)
    return True


def run(smoke=False):
    started = time.perf_counter()
    protocol_hash = _protocol_hash()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU propagation or inference fallback is forbidden')
    d, fp = G.inputs()
    input_hashes, source_hashes, physical_sources = _hashes()
    device = torch.device('cuda', index=torch.cuda.current_device())
    engine, predictor_hashes = _load_models(d, fp)
    stem = 'smoke_' if smoke else ''
    raw_path, output_path = OUT / f'{stem}raw_signals.npz', OUT / f'{stem}dataset.npz'
    physical, commands, raw_records = _physical_cache(
        raw_path, smoke, fp, input_hashes, physical_sources, protocol_hash, d)
    raw_hash = G.O.C.sha256(raw_path)
    if _load_complete(output_path, input_hashes, source_hashes, predictor_hashes,
                      fp, smoke, raw_hash, protocol_hash):
        return

    records, fit_records = [], []
    feature_rows, truth_rows, observation_rows, age_rows = [], [], [], []
    fitter_rows, fitter_observation_rows, fitter_age_rows = [], [], []
    frame_times = None
    for index, item in enumerate(raw_records):
        offset = len(records)
        records.extend(dict(design=item['design'], split=item['split'], replicate=item['replicate'],
                            profile=item['profile'], noise=noise, schedule=item['schedule'])
                       for noise in NOISES)
        result = _process(item, physical[index], commands[index], offset, records,
                          fit_records, d, engine)
        features, truth, observed, age, fitter, fitter_observed, fitter_age, frame_times = result
        feature_rows.extend(features)
        truth_rows.extend([truth.copy() for _ in NOISES])
        observation_rows.extend(observed)
        age_rows.extend(age)
        fitter_rows.extend(fitter)
        fitter_observation_rows.extend(fitter_observed)
        fitter_age_rows.extend(fitter_age)

    expected = 45 if smoke else 240
    dataset = dict(features=np.stack(feature_rows), truth=np.stack(truth_rows),
                   observed=np.stack(observation_rows), age=np.stack(age_rows), times=frame_times,
                   fitter_features=np.stack(fitter_rows),
                   fitter_observed=np.stack(fitter_observation_rows),
                   fitter_age=np.stack(fitter_age_rows), records=records)
    if dataset['features'].shape != (expected, 124, 174):
        raise RuntimeError(f'Unexpected comparison dataset shape: {dataset["features"].shape}')
    if dataset['fitter_features'].shape != (expected, 124, 6):
        raise RuntimeError('Unexpected waveform-fitter feature shape')
    verification = _verify_source(dataset, smoke)
    runtime = _runtime()
    signature = json.dumps(runtime, sort_keys=True, separators=(',', ':'))
    provenance = dict(fingerprint=fp, smoke=smoke, input_hashes=input_hashes,
                      source_hashes=source_hashes, physical_source_hashes=physical_sources,
                      spatial_predictor_sha256=predictor_hashes, raw_signals_sha256=raw_hash,
                      comparison_protocol_sha256=protocol_hash, runtime=runtime,
                      runtime_signature_sha256=hashlib.sha256(signature.encode()).hexdigest(),
                      feature_tolerance_absolute=FEATURE_TOLERANCE,
                      fitter_feature_order=PROTOCOL['physical_features'], solver_settings=FIT_SETTINGS,
                      verification=verification, device=str(device),
                      elapsed_seconds=time.perf_counter() - started)
    arrays = {key: value for key, value in dataset.items() if key != 'records'}
    G.O.C.save_npz(output_path, **arrays, records=json.dumps(records),
                   fit_records=json.dumps(M.clean_json(fit_records)),
                   protocol=json.dumps(PROTOCOL), provenance=json.dumps(M.clean_json(provenance)))
    print('COMPARISON_DATASET_COMPLETE', output_path.name, len(records),
          'max_feature_difference', verification['maximum_absolute_feature_difference'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=1):
        run(parser.parse_args().smoke)
