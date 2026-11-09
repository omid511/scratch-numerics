"""Independently recompute timing statistics and check live replay outputs."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUT = SCRIPT/'framewise_latency_20261008'
COMPARISON = SCRIPT/'fitter_margin_comparison_20261008'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values, deadline):
    values = sorted(float(value) for value in values)
    if not values:
        return None

    def quantile(probability):
        position = probability*(len(values)-1)
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower]+(values[upper]-values[lower])*(position-lower)

    misses = sum(value > deadline for value in values)
    return dict(count=len(values), median_ms=quantile(.5), p95_ms=quantile(.95),
                p99_ms=quantile(.99), max_ms=values[-1], deadline_misses=misses,
                deadline_miss_fraction=misses/len(values))


def run():
    result = json.loads((OUT/'results.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert result['protocol_sha256'] == sha(OUT/'protocol.json')
    assert result['source_sha256'] == sha(SCRIPT/'measure_framewise_latency.py')
    assert result['measurements_sha256'] == sha(OUT/'measurements.npz')
    assert result['inputs'] == protocol['inputs']
    for relative, expected in result['inputs'].items():
        assert sha(ROOT/relative) == expected, relative
    with np.load(OUT/'measurements.npz', allow_pickle=False) as archive:
        measured = {key: archive[key] for key in ('milliseconds', 'observed', 'points', 'ages',
            'feature_errors', 'normalized_feature_errors', 'warnings')}
        rows = json.loads(str(archive['records']))
    with np.load(COMPARISON/'predictions.npz', allow_pickle=False) as archive:
        records = json.loads(str(archive['records']))
        names = json.loads(str(archive['prediction_names']))
        points = archive['predictions'][names.index('framewise')]
        expected_age, expected_observed = archive['age'], archive['observed']
        threshold = json.loads(str(archive['warning_thresholds']))['framewise']['0.05']['threshold']
    indices = [row['record_index'] for row in rows]
    expected = {i for i, row in enumerate(records) if row['split'] == protocol['split']}
    assert len(indices) == len(set(indices)) and set(indices) == expected
    assert all(row['record'] == records[index] for row, index in zip(rows, indices))
    milliseconds, observed = measured['milliseconds'], measured['observed']
    assert all(value.shape == (len(indices), points.shape[1]) for value in measured.values())
    assert np.isfinite(milliseconds).all() and (milliseconds > 0).all()
    availability_mismatches = int(np.count_nonzero(observed != expected_observed[indices]))
    assert np.isfinite(measured['points']).all() and np.isfinite(measured['ages']).all()
    expected_warnings = -points[indices] > threshold
    assert np.array_equal(measured['warnings'], -measured['points'] > threshold)
    warning_mismatches = int(np.count_nonzero(measured['warnings'] != expected_warnings))
    prediction_mismatches = int(np.count_nonzero(abs(measured['points']-points[indices]) > protocol['prediction_atol']))
    age_mismatches = int(np.count_nonzero(abs(measured['ages']-expected_age[indices]) > 1e-12))
    replay_valid = not (availability_mismatches or prediction_mismatches or age_mismatches or warning_mismatches)
    for key, value in (('availability_mismatches', availability_mismatches),
                       ('prediction_mismatches', prediction_mismatches),
                       ('age_mismatches', age_mismatches), ('warning_mismatches', warning_mismatches)):
        assert result['replay'][key] == value
    assert result['replay_valid'] == replay_valid
    assert result['replay']['availability_exact'] == (availability_mismatches == 0)
    point_error = float(np.max(abs(measured['points']-points[indices])))
    age_error = float(np.max(abs(measured['ages']-expected_age[indices])))
    feature_error = float(measured['feature_errors'].max())
    assert np.isfinite(measured['feature_errors']).all() and (measured['feature_errors'] >= 0).all()
    normalized_error = float(measured['normalized_feature_errors'].max())
    assert np.isfinite(measured['normalized_feature_errors']).all()
    assert (measured['normalized_feature_errors'] >= 0).all()
    assert math.isclose(normalized_error, result['replay']['maximum_normalized_feature_error'], abs_tol=1e-12)
    assert math.isclose(point_error, result['replay']['maximum_prediction_error'], abs_tol=1e-12)
    assert math.isclose(age_error, result['replay']['maximum_age_error'], abs_tol=1e-12)
    assert math.isclose(feature_error, result['replay']['maximum_feature_error'], abs_tol=1e-12)
    checked = 0

    def check(reported, values):
        nonlocal checked
        independent = statistics(values, protocol['deadline_ms'])
        if independent is None:
            assert reported is None
            return
        for key, expected_value in independent.items():
            assert math.isclose(reported[key], expected_value, rel_tol=1e-12, abs_tol=1e-12), key
            checked += 1

    check(result['all_frames'], milliseconds.ravel())
    check(result['accepted'], milliseconds[observed])
    check(result['rejected'], milliseconds[~observed])
    condition_p95 = []
    for noise in protocol['noise']:
        mask = np.array([row['record']['noise'] == noise for row in rows])
        for category, values in (('all_frames', milliseconds[mask].ravel()),
                                 ('accepted', milliseconds[mask][observed[mask]]),
                                 ('rejected', milliseconds[mask][~observed[mask]])):
            check(result['per_noise'][noise][category], values)
        condition_p95.append(statistics(milliseconds[mask][observed[mask]], protocol['deadline_ms'])['p95_ms'])
    assert math.isclose(result['scored_p95_ms'], max(condition_p95), abs_tol=1e-12)
    assert result['criterion_met'] == (replay_valid and max(condition_p95) <= protocol['deadline_ms'])
    assert result['replay']['frames'] == milliseconds.size
    assert result['replay']['accepted_frames'] == int(observed.sum())
    assert result['aperture_seconds'] == protocol['window_samples']*protocol['dt_seconds']
    assert result['hardware']['torch_cpu_threads'] == 1 and result['hardware']['gpu']
    verification = dict(results_sha256=sha(OUT/'results.json'),
        measurements_sha256=sha(OUT/'measurements.npz'), verification_source_sha256=sha(Path(__file__)),
        statistic_checks=checked, decisions=int(milliseconds.size), exact_availability=availability_mismatches == 0,
        replay_valid=replay_valid, availability_mismatches=availability_mismatches,
        prediction_mismatches=prediction_mismatches, age_mismatches=age_mismatches,
        warning_mismatches=warning_mismatches, maximum_prediction_error=point_error, maximum_age_error=age_error,
        maximum_feature_error=feature_error, maximum_normalized_feature_error=normalized_error,
        scored_p95_ms=max(condition_p95),
        limitation='Development GPU replay and independent arithmetic; not deployment or safety qualification')
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print('FRAMEWISE_LATENCY_VERIFIED', json.dumps(verification), flush=True)


if __name__ == '__main__':
    run()
