"""Frozen paired pulse/observability diagnostic; never overwrites O1 results."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import gpu_execution as G
from cuda_backend import CudaPlate, CudaInference, estimate_batch, least_squares, tensor
from mechanics.p4_margin_estimation.design_sampler import DesignSample

OUT = G.O.SCRIPT / 'pulse_aware_probe_20261007'
PROTOCOL = json.loads((OUT / 'protocol.json').read_text())


def archive_rows(design, profile, forcing, replicate, duration):
    with np.load(G.O.OUT / f'trajectories_{design}.npz') as z:
        identity = json.loads(str(z['identity']))
        offsets = z['offsets']
        rows = {}
        for i, descriptor in enumerate(json.loads(str(z['descriptors']))):
            if (descriptor['profile'], descriptor['forcing'], descriptor['replicate']) != (profile, forcing, replicate):
                continue
            if descriptor['noise'] not in PROTOCOL['noise']:
                continue
            begin, end = offsets[i:i + 2]
            count = int((z['times'][begin:end] <= duration + 1e-9).sum())
            rows[descriptor['noise']] = dict(descriptor=descriptor, **{
                key: z[key][begin:begin + count].copy() for key in
                ('times', 'delivery', 'point', 'score', 'available', 'identified', 'reason')})
    assert set(rows) == set(PROTOCOL['noise'])
    return identity, rows


def forecast_skills(signals, modes, prefix, horizon):
    """Predict an already-observed suffix from its strictly earlier prefix."""
    skills = np.full(len(signals), -np.inf)
    groups = {}
    for i, mode in enumerate(modes):
        if len(mode['alpha']):
            groups.setdefault(len(mode['alpha']), []).append(i)
    for count, indices in groups.items():
        x = tensor(np.stack([signals[i] for i in indices]))
        # Fixed horizon is known, but suffix sample values do not enter fitting.
        normalizer = x[:, :, :prefix].abs().amax(dim=(1, 2), keepdim=True)
        x = x / normalizer
        alpha = tensor(np.stack([modes[i]['alpha'] for i in indices]))
        frequency = tensor(np.stack([modes[i]['frequency'] for i in indices]))
        t = torch.arange(prefix + horizon, device='cuda', dtype=torch.float64) * .001
        growth = alpha[:, :, None] * t
        envelope = torch.exp(growth - growth.amax(dim=2, keepdim=True))
        phase = 2 * np.pi * frequency[:, :, None] * t
        basis = torch.cat((torch.ones((len(indices), len(t), 1), device='cuda', dtype=torch.float64),
                           (envelope * torch.cos(phase)).transpose(1, 2),
                           (envelope * torch.sin(phase)).transpose(1, 2)), dim=2)
        coefficients = least_squares(basis[:, :prefix], x[:, :, :prefix].transpose(1, 2))
        prediction = basis[:, prefix:] @ coefficients
        suffix = x[:, :, prefix:prefix + horizon].transpose(1, 2)
        centered = ((suffix - suffix.mean(dim=1, keepdim=True)) ** 2).sum(dim=(1, 2))
        energy = (suffix ** 2).sum(dim=(1, 2))
        residual = ((suffix - prediction) ** 2).sum(dim=(1, 2))
        skill = 1 - residual / centered
        good = (centered > torch.finfo(torch.float64).eps * energy) & torch.isfinite(skill)
        values = skill.cpu().numpy()
        for j, i in enumerate(indices):
            if bool(good[j]):
                skills[i] = float(values[j])
    return skills


def command_free_windows(command_activity, ends):
    # Entire activity log is recorded during acquisition. Each decision uses
    # only entries up to its own window end, not future commands or labels.
    cumulative = np.r_[0, np.cumsum(command_activity, dtype=np.int64)]
    return cumulative[ends] == cumulative[ends - PROTOCOL['window_samples']]


def verify_prediction(engine, signal, archived_point, pairs):
    mode = estimate_batch([signal])[0]
    assert len(mode['alpha'])
    ridge = G.O.B.PhysicsFeatureRidge()
    row = np.r_[G.O.B.base_feature_row(signal, .001, mode, ridge),
                G.O.B.modal_spatial_row(mode, pairs), G.O.B.modal_spatial_addition_row(mode, pairs)]
    point = float(engine.predict(row[None])[0])
    np.testing.assert_allclose(point, archived_point, atol=1e-6, rtol=1e-6)
    return abs(point - archived_point)


def summarize(row, eligible, profile, duration, thresholds):
    selected = row['available'] & eligible
    truth = 1 - G.O.TP.ratio_at(profile, row['times'])
    result = dict(windows=len(selected), baseline_available_windows=int(row['available'].sum()),
                  accepted_windows=int(selected.sum()), availability_fraction=float(selected.mean()),
                  accepted_margin_below_minus_one=int((selected & (row['point'] < -1)).sum()),
                  conditional_margin_mae=None if not selected.any() else float(np.abs(row['point'][selected] - truth[selected]).mean()),
                  policies={})
    crossing = row['descriptor']['crossing']
    if crossing is not None:
        approach = (row['times'] >= row['descriptor']['approach_start']) & (row['delivery'] <= crossing - 1)
        result['accepted_one_second_lead_approach_windows'] = int((selected & approach).sum())
    for name, threshold in thresholds.items():
        # Keep the ORIGINAL end for alarm hysteresis, then clip episode exposure
        # to the probe horizon. Processing no future updates cannot start an alarm.
        operating_end = max(row['descriptor']['duration'], float(row['delivery'][-1]))
        episodes = G.O.alarm_episodes(row['delivery'], row['score'] >= threshold, selected,
                                    operating_end=operating_end)
        episodes = episodes[episodes[:, 0] <= duration]
        result['policies'][name] = dict(episodes=int(len(episodes)),
            event=None if crossing is None else G.O.crossing_metrics(
                episodes, crossing, row['descriptor']['approach_start']))
    return result


def run(smoke):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU numerical fallback')
    d, fp = G.inputs()
    models = []
    for width in (107, 174):
        with np.load(G.O.OUT / f'trajectory_baseline{width}.npz') as z:
            assert json.loads(str(z['fingerprint'])) == fp
            models.append(G.O.C.cloudpickle.loads(z['predictor'].tobytes()))
    engine = CudaInference(models, d['pairs'])
    calibration = json.loads((G.O.OUT / 'trajectory_calibration.json').read_text())
    assert calibration['fingerprint'] == fp
    thresholds = {name: calibration['thresholds'][name] for name in ('episodes_per_hour_1.0', 'legacy_clip1')}
    records = []
    for case in PROTOCOL['designs'][:1] if smoke else PROTOCOL['designs']:
        plate = None
        profiles = {'safe_dwell': PROTOCOL['profiles']['safe_dwell']} if smoke else PROTOCOL['profiles']
        for profile, duration in profiles.items():
            forcings = ['intermittent_physical_pulses'] if smoke else PROTOCOL['forcing']
            for forcing in forcings:
                identity, rows = archive_rows(case['design'], profile, forcing, case['replicate'], duration)
                assert identity['fingerprint'] == fp
                audit = json.loads((G.O.OUT / f"audit_{case['design']}.json").read_text())
                assert audit['acquisition_supported'] and audit['fingerprint'] == fp
                if plate is None:
                    plate = CudaPlate(G.O.TP.PlateDynamics(DesignSample(**identity['design']), identity['ucrit'], G.sensor_yx(d)))
                rawfile = OUT / f"raw_{case['design']}_{profile}_{forcing}_rep{case['replicate']}.npz"
                raw_identity = dict(identity=identity, duration=duration, profile=profile, forcing=forcing,
                                    replicate=case['replicate'], backend=G.backend_signature())
                if rawfile.exists():
                    with np.load(rawfile) as z:
                        assert json.loads(str(z['identity'])) == raw_identity
                        raw = z['response'].copy()
                        assert z['valid'].all()
                else:
                    acquired = plate.acquire(profile, case['replicate'], forcing, duration=duration)
                    assert acquired.valid.all() and acquired.termination is None
                    raw = acquired.response
                    G.O.C.save_npz(rawfile, response=raw, valid=acquired.valid, identity=json.dumps(raw_identity))
                ends = np.arange(PROTOCOL['window_samples'], raw.shape[1] + 1, 100)
                times = (ends - 1) / 1000.
                activity = np.array([G.O.TP.physical_pulse(float(t)) != 0 for t in np.arange(raw.shape[1]) / 1000.]) if forcing == 'intermittent_physical_pulses' else np.zeros(raw.shape[1], bool)
                free = command_free_windows(activity, ends)
                # Noise-only fixture preserves original RNG shape and REAL leading
                # calibration samples. Its unused future entries are not a waveform
                # approximation and never reach normalization/estimation. These
                # three declared noise families use only prefix and shape.
                original_count = int(rows['clean']['descriptor']['duration'] * 1000) + 1
                rng_fixture = np.zeros((raw.shape[0], original_count))
                rng_fixture[:, :51] = raw[:, :51]
                measured = []
                for noise in PROTOCOL['noise']:
                    injected, sigma = G.O.TP.continuous_noise(rng_fixture, profile, forcing, case['replicate'], int(case['design'][1:]), noise)
                    np.testing.assert_allclose(sigma, rows[noise]['descriptor']['physical_noise_reference_m'], rtol=1e-10, atol=1e-15)
                    signal, _, _ = G.O.causal_observation(raw, injected[:, :raw.shape[1]])
                    np.testing.assert_array_equal(times, rows[noise]['times'])
                    measured.append(signal)
                pulse = np.zeros((len(measured), len(ends)), bool)
                skill_values = np.full((len(measured), len(ends), 2), -np.inf)
                replay_errors = []
                for i, noise in enumerate(PROTOCOL['noise']):
                    first = int(np.flatnonzero(rows[noise]['available'])[0])
                    end = ends[first]
                    replay_errors.append(verify_prediction(engine, measured[i][:, end - 512:end], rows[noise]['point'][first], d['pairs']))
                for j, end in enumerate(ends):
                    indices = []
                    windows = []
                    for i, noise in enumerate(PROTOCOL['noise']):
                        signal = measured[i][:, end - 512:end]
                        eligible = bool(free[j] and np.isfinite(signal).all() and not np.any(abs(signal) >= 50.))
                        pulse[i, j] = eligible
                        if eligible and rows[noise]['available'][j]:
                            windows.append(signal)
                            indices.append(i)
                    if not windows:
                        continue
                    for fold, prefix in enumerate(PROTOCOL['forecast_prefix_samples']):
                        modes = estimate_batch([window[:, :prefix] for window in windows])
                        skill_values[indices, j, fold] = forecast_skills(windows, modes, prefix, PROTOCOL['forecast_samples'])
                predictive = pulse & np.all(skill_values >= PROTOCOL['minimum_forecast_skill'], axis=2)
                for i, noise in enumerate(PROTOCOL['noise']):
                    row = rows[noise]
                    masks = dict(baseline=np.ones(len(ends), bool), pulse_guard=pulse[i], pulse_guard_predictive=predictive[i])
                    record = dict(design=case['design'], profile=profile, forcing=forcing,
                                  replicate=case['replicate'], noise=noise, duration=duration,
                                  prediction_replay_error=float(replay_errors[i]),
                                  arms={name: summarize(row, mask, profile, duration, thresholds) for name, mask in masks.items()})
                    records.append(record)
                    G.O.C.save_npz(OUT / f"gate_{case['design']}_{profile}_{forcing}_rep{case['replicate']}_{noise}.npz",
                        times=times, point=row['point'], baseline_available=row['available'], pulse_guard=pulse[i],
                        predictive_guard=predictive[i], forecast_skills=skill_values[i])
                print('PULSE_AWARE_CASE_COMPLETE', case['design'], profile, forcing, flush=True)
    proof = dict(future_commands_do_not_change_past_masks=True)
    activity = np.zeros(2001, bool)
    activity[1001:1020] = True
    ends = np.arange(512, 2002, 100)
    before = command_free_windows(activity, ends)
    activity[1500:] = ~activity[1500:]
    np.testing.assert_array_equal(before[ends <= 1500], command_free_windows(activity, ends)[ends <= 1500])
    output = dict(protocol=PROTOCOL, records=records, thresholds=thresholds, causal_command_check=proof,
                  backend=G.backend_signature(), source_sha256=G.O.C.sha256(Path(__file__)),
                  baseline_fingerprint=fp, limitations=PROTOCOL['frozen'])
    G.O.C.save_json(OUT / ('smoke_results.json' if smoke else 'results.json'), output)
    print('PULSE_AWARE_PROBE_COMPLETE', len(records), 'paired noise cases', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.smoke)
