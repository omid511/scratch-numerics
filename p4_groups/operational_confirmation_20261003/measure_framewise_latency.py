"""Measure frozen framewise sensor-to-output computation on existing CUDA hardware."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import cuda_backend as C
import diagnostic_matrix as M
import gpu_execution as G
import pulse_aware_probe as P
import temporal_convergence as T

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUT = SCRIPT/'framewise_latency_20261008'
COMPARISON = SCRIPT/'fitter_margin_comparison_20261008'
CONVERGENCE = SCRIPT/'temporal_convergence_20261007'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(measured, command, end, state, d, engine, model, scaling, threshold):
    """No truth, cached features, labels or simulator poles enter this path."""
    if state['offset'] is None:
        prefix = measured[:, :51]
        _, state['offset'], state['scale'] = M.normalize(prefix, np.zeros_like(prefix))
    window = np.clip((measured[:, end-128:end]-state['offset'][:, None])/
                     state['scale'][:, None], -50., 50.).astype(np.float32)
    accepted = False
    row = np.zeros(174)
    if not command[end-128:end].any() and np.isfinite(window).all() and not (abs(window) >= 50).any():
        prefix_mode = C.estimate_batch([window[:, :64]])
        if P.forecast_skills([window], prefix_mode, 64, 64)[0] >= .5:
            mode = C.estimate_batch([window])[0]
            if len(mode['alpha']):
                row = M.features(window, mode, d, engine)
                if row.shape != (174,) or not np.isfinite(row).all():
                    raise FloatingPointError('Invalid live framewise feature row')
                accepted = True
    timestamp = (end-1)*.001
    if accepted:
        state['last'] = timestamp
    age = timestamp-state['last']
    mean, scale, active = scaling
    normalized = (row-mean)/scale if accepted else np.zeros(174)
    normalized[~active] = 0
    inputs = np.r_[normalized, float(accepted), age].astype(np.float32)
    with torch.no_grad():
        point = float(model(torch.as_tensor(inputs[None], device='cuda'))[0, 0])
    if not np.isfinite(point):
        raise FloatingPointError('Nonfinite live framewise prediction')
    score = -point
    return dict(point=point, score=score, warning=bool(score > threshold),
                observed=accepted, age=age, features=row, normalized_features=inputs[:174])


def summary(milliseconds):
    values = np.asarray(milliseconds)
    if not len(values):
        return None
    return dict(count=len(values), median_ms=float(np.median(values)),
                p95_ms=float(np.quantile(values, .95)), p99_ms=float(np.quantile(values, .99)),
                max_ms=float(values.max()), deadline_misses=int((values > 10).sum()),
                deadline_miss_fraction=float((values > 10).mean()))


def run():
    if (OUT/'results.json').exists():
        raise RuntimeError('Completed timing evidence is immutable; use a new version for another run')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU inference fallback')
    protocol = json.loads((OUT/'protocol.json').read_text())
    for relative, expected in protocol['inputs'].items():
        if sha(ROOT/relative) != expected:
            raise RuntimeError('Frozen timing prerequisite changed: '+relative)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with np.load(COMPARISON/'raw_signals.npz', allow_pickle=False) as z:
        physical, command = z['physical'], z['command']
        raw_records = json.loads(str(z['records']))
    with np.load(COMPARISON/'dataset.npz', allow_pickle=False) as z:
        archived = {key: z[key] for key in ('features', 'observed', 'age')}
        records = json.loads(str(z['records']))
    with np.load(COMPARISON/'predictions.npz', allow_pickle=False) as z:
        names = json.loads(str(z['prediction_names']))
        points = z['predictions'][names.index('framewise')]
        threshold = json.loads(str(z['warning_thresholds']))['framewise']['0.05']['threshold']
    with np.load(CONVERGENCE/'predictions.npz', allow_pickle=False) as z:
        scaling = tuple(z[key] for key in ('mean', 'scale', 'active'))
    d, _ = G.inputs()
    engine = SimpleNamespace(ridge=G.O.B.PhysicsFeatureRidge())
    sample = torch.empty((1, 1, 176), device='cuda', dtype=torch.float32)
    model = T.make_model('framewise', sample).to(device='cuda', dtype=torch.float32)
    checkpoint = torch.load(CONVERGENCE/'framewise.pt', map_location='cuda', weights_only=False)
    if checkpoint['kind'] != 'framewise':
        raise RuntimeError('Wrong frozen checkpoint kind')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    index = {(r['design'], r['replicate'], r['profile'], r['noise']): i for i, r in enumerate(records)}
    selected = [i for i, r in enumerate(raw_records) if r['split'] == protocol['split']]
    ends = np.arange(128, physical.shape[-1]+1, 64)
    timings, observations, replay_rows = [], [], []
    replay_points, replay_ages, feature_errors = [], [], []
    normalized_feature_errors = []
    maximum_feature_error = maximum_prediction_error = maximum_age_error = 0.
    maximum_normalized_feature_error = 0.
    availability_mismatches = prediction_mismatches = age_mismatches = warning_mismatches = 0
    replay_warnings = []
    cold_first = None
    for raw_index in selected:
        record = raw_records[raw_index]
        seed = np.random.SeedSequence([20261010, int(record['design'][1:]), record['replicate'],
                                      ('cross', 'stable', 'retreat').index(record['profile'])])
        noises, _ = M.noises(physical[raw_index], seed)
        for noise in protocol['noise']:
            measured = physical[raw_index]+noises[noise]
            if cold_first is None:
                state = dict(offset=None, scale=None, last=0.)
                for warmup in range(protocol['warmup_decisions']):
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    decision(measured, command[raw_index], int(ends[warmup]), state,
                             d, engine, model, scaling, threshold)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns()-started)/1e6
                    if warmup == 0:
                        cold_first = elapsed
            state = dict(offset=None, scale=None, last=0.)
            row_timings, row_observed = [], []
            row_points, row_ages, row_feature_errors = [], [], []
            row_normalized_errors = []
            row_warnings = []
            archived_index = index[(record['design'], record['replicate'], record['profile'], noise)]
            for frame, end in enumerate(ends):
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                output = decision(measured, command[raw_index], int(end), state,
                                  d, engine, model, scaling, threshold)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns()-started)/1e6
                feature_error = float(np.max(abs(output['features']-archived['features'][archived_index, frame])))
                expected_input = (archived['features'][archived_index, frame]-scaling[0])/scaling[1]
                expected_input[~scaling[2]] = 0.
                if not archived['observed'][archived_index, frame]:
                    expected_input[:] = 0.
                normalized_error = float(np.max(abs(output['normalized_features']-
                    expected_input.astype(np.float32))))
                maximum_normalized_feature_error = max(maximum_normalized_feature_error, normalized_error)
                prediction_error = abs(output['point']-float(points[archived_index, frame]))
                age_error = abs(output['age']-float(archived['age'][archived_index, frame]))
                maximum_feature_error = max(maximum_feature_error, feature_error)
                maximum_prediction_error = max(maximum_prediction_error, prediction_error)
                maximum_age_error = max(maximum_age_error, age_error)
                availability_mismatches += output['observed'] != bool(archived['observed'][archived_index, frame])
                prediction_mismatches += prediction_error > protocol['prediction_atol']
                age_mismatches += age_error > 1e-12
                warning_mismatches += output['warning'] != bool(-points[archived_index, frame] > threshold)
                row_timings.append(elapsed)
                row_observed.append(output['observed'])
                row_points.append(output['point'])
                row_ages.append(output['age'])
                row_feature_errors.append(feature_error)
                row_normalized_errors.append(normalized_error)
                row_warnings.append(output['warning'])
            timings.append(row_timings)
            observations.append(row_observed)
            replay_points.append(row_points)
            replay_ages.append(row_ages)
            feature_errors.append(row_feature_errors)
            normalized_feature_errors.append(row_normalized_errors)
            replay_warnings.append(row_warnings)
            replay_rows.append(dict(record_index=archived_index, record=records[archived_index]))
        print('FRAMEWISE_TIMING_TRAJECTORY', record['design'], record['replicate'], record['profile'], flush=True)
    milliseconds, accepted = np.array(timings), np.array(observations)
    replay_valid = not (availability_mismatches or prediction_mismatches or age_mismatches or warning_mismatches)
    per_noise = {}
    for noise in protocol['noise']:
        mask = np.array([row['record']['noise'] == noise for row in replay_rows])
        per_noise[noise] = dict(all_frames=summary(milliseconds[mask].ravel()),
                               accepted=summary(milliseconds[mask][accepted[mask]]),
                               rejected=summary(milliseconds[mask][~accepted[mask]]))
    if any(row['accepted'] is None for row in per_noise.values()):
        raise RuntimeError('Every noise condition needs accepted-path timing evidence')
    scored = max(row['accepted']['p95_ms'] for row in per_noise.values())
    measurement_path = OUT/'measurements.npz'
    np.savez_compressed(measurement_path, milliseconds=milliseconds, observed=accepted,
                        points=np.array(replay_points), ages=np.array(replay_ages),
                        feature_errors=np.array(feature_errors),
                        normalized_feature_errors=np.array(normalized_feature_errors),
                        warnings=np.array(replay_warnings), records=json.dumps(replay_rows))
    result = dict(candidate=protocol['candidate'], protocol_sha256=sha(OUT/'protocol.json'),
                  source_sha256=sha(Path(__file__)), measurements_sha256=sha(measurement_path),
                  inputs=protocol['inputs'], hardware=dict(host=platform.node(), platform=platform.platform(),
                      processor=next((line.split(':', 1)[1].strip() for line in
                          Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                          platform.processor()), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      cuda=torch.version.cuda, numpy=np.__version__, torch_cpu_threads=torch.get_num_threads()),
                  scope=protocol['scope'], included=protocol['included'], excluded=protocol['excluded'],
                  aperture_seconds=.128, cold_first_decision_ms=cold_first, all_frames=summary(milliseconds.ravel()),
                  accepted=summary(milliseconds[accepted]), rejected=summary(milliseconds[~accepted]),
                  per_noise=per_noise, scored_p95_ms=scored,
                  replay=dict(frames=int(milliseconds.size), accepted_frames=int(accepted.sum()),
                      maximum_feature_error=maximum_feature_error, maximum_prediction_error=maximum_prediction_error,
                      maximum_normalized_feature_error=maximum_normalized_feature_error,
                      maximum_age_error=maximum_age_error, availability_exact=availability_mismatches == 0,
                      availability_mismatches=availability_mismatches, prediction_mismatches=prediction_mismatches,
                      age_mismatches=age_mismatches, warning_mismatches=warning_mismatches),
                  replay_valid=replay_valid, criterion_met=bool(replay_valid and scored <= protocol['deadline_ms']))
    (OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('FRAMEWISE_LATENCY_COMPLETE', json.dumps(dict(scored_p95_ms=scored,
        frames=int(milliseconds.size), replay_valid=replay_valid, criterion_met=result['criterion_met'])), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
