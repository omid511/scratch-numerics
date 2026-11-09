"""Replay and score the frozen cached128ms waveform-only modal-fit diagnostic."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import scipy
from threadpoolctl import threadpool_limits
import short_window_timing as S
from time_varying_modal_fit import fit_single_mode

M = S.M
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUT = SCRIPT / 'time_varying_fit_20261007'
PROTOCOL_PATH, RESULTS_PATH = OUT / 'protocol.json', OUT / 'results.json'
SOURCE_OUT = SCRIPT / 'short_window_timing_20261007'
SOURCE_FILES = {name: SOURCE_OUT / name for name in
                ('protocol.json', 'results.json', 'analysis.json', 'raw_controls.npz')}

def _verify_inputs():
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError(f'Frozen protocol missing: {PROTOCOL_PATH}')
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if set(protocol.get('inputs', {})) != set(SOURCE_FILES):
        raise RuntimeError('Frozen protocol must pin the four cached source artifacts')
    inputs = {name: M.G.O.C.sha256(path) for name, path in SOURCE_FILES.items()}
    if any(inputs[name] != digest for name, digest in protocol['inputs'].items()):
        raise RuntimeError('Frozen cached input SHA256 mismatch')
    source_protocol = json.loads(SOURCE_FILES['protocol.json'].read_text())
    result = json.loads(SOURCE_FILES['results.json'].read_text())
    analysis = json.loads(SOURCE_FILES['analysis.json'].read_text())
    if result.get('protocol') != source_protocol or analysis.get('scope') != source_protocol.get('scope'):
        raise RuntimeError('Cached result/analysis metadata does not match source protocol')
    verified, estimator_sources = {}, result.get('source_sha256')
    if not isinstance(estimator_sources, dict) or not estimator_sources:
        raise RuntimeError('Cached results lack original estimator-source provenance')
    for name, digest in estimator_sources.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or M.G.O.C.sha256(SCRIPT / path) != digest:
            raise RuntimeError(f'Original estimator-source hash mismatch: {name}')
        verified[f'estimator/{name}'] = digest
    fingerprint = result.get('fingerprint', {})
    sources = fingerprint.get('sources', {}) if isinstance(fingerprint, dict) else {}
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError('Cached results lack original physical-source provenance')
    for name, digest in sources.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or M.G.O.C.sha256(ROOT / path) != digest:
            raise RuntimeError(f'Original physical-source hash mismatch: {name}')
        verified[f'physical/{name}'] = digest
    metadata_name = 'p4_dataset_saturation/metadata.json'
    metadata_path = ROOT / metadata_name
    if result.get('dataset_metadata_sha256') != sources.get(metadata_name):
        raise RuntimeError('Dataset metadata hash disagrees with original-source fingerprint')
    train = json.loads(metadata_path.read_text()).get('train_designs', [])
    if len(train) != 77 or any(case['design'] not in train for case in source_protocol['cases']):
        raise RuntimeError('Cached cases are not in the original 77-design TRAIN split')
    replay = analysis.get('source_sha256')
    if not replay or M.G.O.C.sha256(SCRIPT / 'analyze_short_window_timing.py') != replay:
        raise RuntimeError('Cached replay-analysis source hash mismatch')
    if analysis.get('result_sha256') != inputs['results.json'] or analysis.get('raw_sha256') != inputs['raw_controls.npz']:
        raise RuntimeError('Cached analysis does not reference verified results/raw controls')
    verified['replay_analysis'] = replay
    provenance = {'time_varying_protocol_sha256': M.G.O.C.sha256(PROTOCOL_PATH),
        'cached_inputs_sha256': inputs, 'original_sources_sha256': verified,
        'dataset_train_design_count': len(train), 'case_designs': [c['design'] for c in source_protocol['cases']]}
    return protocol, source_protocol, result, analysis, provenance

def _key(row):
    return (row['design'], int(row['replicate']), int(row['samples']), float(row['end_ratio']),
            float(row['ratio_rate_per_second']), row['excitation'], row['noise'])

def _load_controls(protocol, result, analysis):
    rows, analysis_rows = result.get('rows'), analysis.get('rows')
    if not isinstance(rows, list) or not isinstance(analysis_rows, list):
        raise RuntimeError('Cached result and analysis must contain window records')
    count = math.prod(map(len, (protocol['cases'], protocol['samples'], protocol['end_ratios'],
                               protocol['ratio_rates_per_second'], protocol['excitation'], protocol['noise'])))
    maps = [{_key(row): row for row in items} for items in (rows, analysis_rows)]
    if (len(rows) != count or len(analysis_rows) != count or len(maps[0]) != count or
            len(maps[1]) != count or maps[0].keys() != maps[1].keys()):
        raise RuntimeError(f'Cached rows have duplicate/mismatched conditions (expected {count})')
    for row in rows:
        bank = maps[1][_key(row)].get('pole_bank')
        names = ('alpha', 'frequency', 'energy_fraction')
        if not isinstance(bank, dict) or any(name not in bank for name in names) or len({len(bank[n]) for n in names}) != 1:
            raise RuntimeError(f'Archived pole bank missing/malformed for {_key(row)}')
    with np.load(SOURCE_FILES['raw_controls.npz']) as archive:
        required = {'physical', 'noise_white', 'records', 'protocol'}
        if not required.issubset(archive.files):
            raise RuntimeError(f'raw_controls.npz missing arrays: {sorted(required - set(archive.files))}')
        physical, white = archive['physical'].copy(), archive['noise_white'].copy()
        records, embedded = archive['records'], archive['protocol']
        def decode_json(value, label):
            value = np.asarray(value)
            if value.shape != ():
                raise RuntimeError(f'raw_controls.npz {label} must be a scalar JSON string')
            value = value.item()
            return json.loads(value.decode('utf-8') if isinstance(value, bytes) else str(value))
        records, embedded = decode_json(records, 'records'), decode_json(embedded, 'protocol')
    if records != rows or embedded != protocol:
        raise RuntimeError('raw_controls.npz records/protocol differ from cached JSON')
    waves = math.prod(map(len, (protocol['cases'], protocol['samples'], protocol['end_ratios'],
                               protocol['ratio_rates_per_second'], protocol['excitation'])))
    if (physical.ndim != 3 or white.ndim != 3 or physical.shape[0] != waves or
            white.shape[0] != len(protocol['cases']) or physical.shape[1] != white.shape[1] or
            physical.shape[2] < max(protocol['samples']) or white.shape[2] < max(protocol['samples'])):
        raise RuntimeError(f'Unexpected cached physical/noise shapes: {physical.shape}, {white.shape}')
    return rows, maps[1], physical, white

def _prepare(row, physical, white):
    samples, wave, draw_index = int(row['samples']), int(row['wave_index']), int(row['noise_index'])
    if samples != 128 or not 0 <= wave < len(physical) or not 0 <= draw_index < len(white):
        raise RuntimeError(f'Invalid cached logical-window index for {_key(row)}')
    raw, draw = physical[wave, :, :samples], white[draw_index, :, :samples]
    if raw.shape != draw.shape or raw.shape[1] != 128 or not np.isfinite(raw).all() or not np.isfinite(draw).all():
        raise RuntimeError(f'Invalid cached signal/noise samples for {_key(row)}')
    if row['noise'] == 'clean':
        noise = np.zeros_like(raw)
    elif row['noise'] == 'iid1':
        draw_rms, noise_rms = float(np.sqrt(np.mean(draw ** 2))), float(row['absolute_noise_rms_m'])
        if not math.isfinite(draw_rms) or draw_rms <= 0 or not math.isfinite(noise_rms) or noise_rms < 0:
            raise RuntimeError(f'Cannot reconstruct stored noise for {_key(row)}')
        noise = (noise_rms / draw_rms) * draw
    else:
        raise RuntimeError(f'Unexpected cached noise type: {row["noise"]!r}')
    measured, offset, scale = M.normalize(raw, noise)  # leading51 calibration and cap50
    signal = np.asarray(measured, dtype=np.float64)
    measurement = {'wave_index': wave, 'noise_index': draw_index, 'logical_samples': 128,
        'span_seconds': float(row['span_seconds']), 'channels': int(signal.shape[0]),
        'calibration_samples': 51, 'calibration_offset': np.asarray(offset, dtype=np.float64).tolist(),
        'calibration_scale': np.asarray(scale, dtype=np.float64).tolist(), 'cap': 50.0,
        'saturation_fraction': float(np.mean(np.abs(measured) >= 50.0)), 'normalized_dtype': str(measured.dtype),
        'fit_dtype': str(signal.dtype), 'fit_signal_sha256': hashlib.sha256(np.ascontiguousarray(signal).tobytes()).hexdigest()}
    return signal, measurement

def _score(alpha, frequency, reference):
    try:
        alpha, frequency = float(alpha), float(frequency)
        ra, rf = float(reference['alpha']), float(reference['frequency'])
    except (KeyError, TypeError, ValueError, OverflowError):
        alpha = frequency = ra = rf = math.nan
    if not np.isfinite((alpha, frequency, ra, rf)).all() or rf == 0:
        return {'available': False, 'alpha_absolute_error_per_second': None, 'frequency_relative_error': None,
                'end_pole_matches_reference': False, 'growth_sign_correct': False}
    ae, fe = abs(alpha - ra), abs(frequency / rf - 1.)
    return {'available': True, 'alpha_absolute_error_per_second': ae, 'frequency_relative_error': fe,
            'end_pole_matches_reference': bool(fe <= .05 and ae <= max(1., .1 * abs(ra))),
            'growth_sign_correct': bool(np.sign(alpha) == np.sign(ra))}

def _archived_score(original, analysis_row, reference):
    alpha, frequency = original.get('estimated_alpha'), original.get('estimated_frequency')
    nearest = _score(alpha, frequency, reference)
    source, bank, matching = analysis_row['pole_bank'], [], []
    for index, (a, f, energy) in enumerate(zip(source['alpha'], source['frequency'], source['energy_fraction'])):
        score = _score(a, f, reference)
        bank.append({'index': index, 'alpha': a, 'frequency': f, 'energy_fraction': energy,
                     'end_reference_score': score})
        if score['end_pole_matches_reference']:
            matching.append(index)
    return {'nearest_end_frequency': {'alpha': alpha, 'frequency': frequency, 'available': nearest['available'],
        'end_reference_score': nearest, 'archived_end_frequency_retained': original.get('end_frequency_retained')},
        'full_pole_bank': bank, 'full_bank_matching_indices': matching, 'any_full_bank_end_match': bool(matching),
        'archived_end_quality': original.get('end_quality'), 'archived_saturation_fraction': original.get('saturation_fraction')}

def _fit_window(original, analysis_row, physical, white, settings, dt):
    signal, measurement = _prepare(original, physical, white)
    stationary = fit_single_mode(signal, dt, varying=False, settings=settings)
    varying = fit_single_mode(signal, dt, varying=True, settings=settings, stationary=stationary)
    references, end = original['references'], original['references']['end']  # truth only after both fits
    def scored(fit):
        return _score(fit.get('alpha_end'), fit.get('frequency_end'), end) if fit.get('success') is True else _score(None, None, end)
    return {'condition': {name: original[name] for name in (
        'design', 'replicate', 'samples', 'span_seconds', 'end_ratio', 'ratio_rate_per_second', 'start_ratio',
        'excitation', 'noise')}, 'measurement': measurement, 'references_after_fit': references,
        'stationary_fit': stationary, 'stationary_end_score': scored(stationary),
        'varying_fit': varying, 'varying_end_score': scored(varying),
        'archived_ls_esprit': _archived_score(original, analysis_row, end)}

def _summary(rows):
    def fit(fk, sk):
        fits, scores = [r[fk] for r in rows], [r[sk] for r in rows]
        errors = [s['alpha_absolute_error_per_second'] for s in scores if s.get('available')]
        return {'denominator': len(rows), 'successful': sum(f.get('success') is True for f in fits),
            'error_available': len(errors), 'error_missing': len(rows) - len(errors),
            'mean_alpha_error_per_second': float(np.mean(errors)) if errors else None}
    archive = [r['archived_ls_esprit'] for r in rows]
    return {'windows': len(rows), 'stationary_fit': fit('stationary_fit', 'stationary_end_score'),
        'varying_fit': fit('varying_fit', 'varying_end_score'),
        'archived_ls_esprit': {'denominator': len(rows),
            'nearest_end_matches': sum(r['nearest_end_frequency']['end_reference_score'].get(
                'end_pole_matches_reference') is True for r in archive),
            'any_bank_end_matches': sum(r['any_full_bank_end_match'] for r in archive)}}

def _pass_checks(rows, frozen):
    n = len(rows)
    interior = [r['varying_fit'].get('success') is True and r['varying_fit'].get('alpha_end') is not None and
                r['varying_fit'].get('frequency_end') is not None and not r['varying_fit'].get('bound_hits') for r in rows]
    matches = [r['varying_end_score'].get('end_pole_matches_reference') is True for r in rows]
    signs = [r['varying_end_score'].get('growth_sign_correct') is True for r in rows]
    clean = [r for r in rows if r['condition']['noise'] == 'clean' and float(r['condition']['ratio_rate_per_second']) == 0.]
    ce = [r['varying_end_score'].get('alpha_absolute_error_per_second') if r['varying_end_score'].get('available') else None for r in clean]
    ce = [v for v in ce if v is not None]
    clean_limit, clean_max = float(frozen['clean_stationary_max_alpha_error_per_second']), max(ce) if ce else None
    clean_pass = len(clean) == 8 and len(ce) == 8 and clean_max <= clean_limit
    noisy = [r for r in rows if r['condition']['noise'] == 'iid1' and float(r['condition']['ratio_rate_per_second']) == 0.]
    nf = [r['varying_end_score'].get('alpha_absolute_error_per_second') if r['varying_end_score'].get('available') else None for r in noisy]
    nl = [r['archived_ls_esprit']['nearest_end_frequency']['end_reference_score'].get(
        'alpha_absolute_error_per_second') if r['archived_ls_esprit']['nearest_end_frequency']['end_reference_score'].get(
        'available') else None for r in noisy]
    nf, nl = [v for v in nf if v is not None], [v for v in nl if v is not None]
    noisy_fit_median, noisy_ls_median = (float(np.median(v)) if v else None for v in (nf, nl))
    allowance = float(frozen['noisy_stationary_median_alpha_error_allowance_over_esprit_per_second'])
    noisy_pass = len(noisy) == 8 and len(nf) == len(nl) == 8 and noisy_fit_median <= noisy_ls_median + allowance

    ramping = {}
    for noise in ('clean', 'iid1'):
        group = [r for r in rows if r['condition']['noise'] == noise and float(r['condition']['ratio_rate_per_second']) != 0.]
        values = {
            'varying': [r['varying_end_score'].get('alpha_absolute_error_per_second') if r['varying_end_score'].get('available') else None for r in group],
            'stationary_single_mode': [r['stationary_end_score'].get('alpha_absolute_error_per_second') if r['stationary_end_score'].get('available') else None for r in group],
            'archived_nearest_ls_esprit': [r['archived_ls_esprit']['nearest_end_frequency']['end_reference_score'].get(
                'alpha_absolute_error_per_second') if r['archived_ls_esprit']['nearest_end_frequency']['end_reference_score'].get('available') else None for r in group]}
        stats = {k: {'denominator': len(group), 'available': sum(v is not None for v in vals),
            'missing': sum(v is None for v in vals),
            'alpha_mae_per_second': float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else None}
            for k, vals in values.items()}
        complete = len(group) == 16 and all(v['available'] == 16 for v in stats.values())
        vm, sm, lm = (stats[k]['alpha_mae_per_second'] for k in values)
        lower_s, lower_l = bool(complete and vm < sm), bool(complete and vm < lm)
        ramping[noise] = {**stats, 'complete_comparison': complete,
            'lower_than_stationary_single_mode': lower_s, 'lower_than_archived_nearest_ls_esprit': lower_l,
            'passed': lower_s and lower_l}
    checks = {
        'all48_critical_fits_successful_and_interior': {'passed': n == 48 and len(interior) == 48 and all(interior),
            'denominator': 48, 'successful_interior': sum(interior),
            'failed_or_noninterior_conditions': [r['condition'] for r, ok in zip(rows, interior) if not ok]},
        'all48_end_poles_match': {'passed': n == 48 and len(matches) == 48 and all(matches), 'denominator': 48,
            'matched': sum(matches), 'tolerance': {'frequency_relative_error_max': .05,
            'alpha_absolute_error_per_second_max': 'max(1, 0.1 * abs(reference alpha))'}},
        'all48_end_growth_signs_correct': {'passed': n == 48 and len(signs) == 48 and all(signs),
            'denominator': 48, 'correct': sum(signs)},
        'clean_stationary_max_alpha_error_per_second': {'passed': clean_pass, 'denominator': 8,
            'available': len(ce), 'model': 'varying_fit_on_stationary_windows', 'maximum': clean_max, 'threshold': clean_limit},
        'noisy_stationary_median_alpha_error_allowance_over_esprit_per_second': {'passed': noisy_pass,
            'denominator': 8, 'varying_fit_available': len(nf), 'archived_nearest_ls_available': len(nl),
            'varying_fit_median': noisy_fit_median, 'archived_nearest_ls_median': noisy_ls_median, 'allowance': allowance,
            'threshold': noisy_ls_median + allowance if noisy_ls_median is not None else None},
        'ramping_alpha_mae': {'passed': all(v['passed'] for v in ramping.values()), 'by_noise': ramping}}
    booleans = {name: value['passed'] for name, value in checks.items()}
    return {'checks': checks, 'booleans': booleans, 'passed': all(booleans.values())}

def _stage(excitation, rows, analysis_map, physical, white, settings, dt, frozen=None):
    chosen = [r for r in rows if int(r['samples']) == 128 and r['excitation'] == excitation]
    if len(chosen) != 48:
        raise RuntimeError(f'Expected 48 logical128 {excitation} conditions, found {len(chosen)}')
    fitted = [_fit_window(r, analysis_map[_key(r)], physical, white, settings, dt) for r in chosen]
    stage = {'status': 'complete', 'windows': len(fitted), 'summary': _summary(fitted), 'rows': fitted}
    if frozen is not None:
        stage['pass_checks'] = _pass_checks(fitted, frozen)
    return stage

def run():
    if RESULTS_PATH.exists():
        raise RuntimeError(f'Completed results are immutable; refusing to overwrite {RESULTS_PATH}')
    started = time.perf_counter()
    protocol, source_protocol, result, analysis, provenance = _verify_inputs()
    provenance['implementation_sha256'] = {name: M.G.O.C.sha256(SCRIPT / name)
        for name in (Path(__file__).name, 'time_varying_modal_fit.py')}
    rows, analysis_map, physical, white = _load_controls(source_protocol, result, analysis)
    selection = protocol.get('selection', {})
    if selection.get('samples') != 128 or selection.get('initial_stage') != 'critical_only_oracle':
        raise RuntimeError('Frozen protocol must select critical-only logical128 first')
    settings, dt = protocol['solver'], float(source_protocol['dt_seconds'])
    critical = _stage('critical_only_oracle', rows, analysis_map, physical, white, settings, dt, protocol['pass_checks'])
    checks = critical['pass_checks']
    if checks['passed']:
        ordinary = _stage('ordinary', rows, analysis_map, physical, white, settings, dt)
        ordinary.update(status='complete_after_critical_pass', reason='All frozen critical-stage checks passed.')
    else:
        ordinary = {'status': 'skipped', 'reason': 'Frozen critical-stage checks failed; ordinary controls were not run.',
            'failed_checks': [name for name, passed in checks['booleans'].items() if not passed],
            'windows': 0, 'summary': None, 'rows': []}
    output = {'scope': protocol['scope'], 'experiment_protocol': protocol, 'cached_source_protocol': source_protocol,
        'provenance': provenance, 'execution': {'device': 'CPU', 'threadpool_limit': 1, 'fitter': 'fit_single_mode',
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'solver_settings': settings, 'fit_samples': 128,
        'truth_scored_only_after_fitting': True, 'ordinary_stage_requires_all_critical_pass_checks': True},
        'critical_only_stage': critical, 'critical_pass_checks': checks, 'ordinary_stage': ordinary,
        'ordinary_stage_executed': ordinary['status'] == 'complete_after_critical_pass',
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': ['Four inspected TRAIN designs, one paired noise draw; not blind confirmation; critical-only is a diagnostic positive control, not a monitor input; no new simulations, training, warning-policy, or reserved-design evaluation.']}
    M.G.O.C.save_json(RESULTS_PATH, M.clean_json(output))
    print('TIME_VARYING_FIT_COMPLETE', json.dumps({'results': str(RESULTS_PATH),
        'critical_windows': critical['windows'], 'critical_checks_passed': checks['passed'],
        'ordinary_stage': ordinary['status']}, sort_keys=True), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
