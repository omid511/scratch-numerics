"""Quadratic pole trends, constant channel coefficients; experimental only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
import experiment_time_varying_fit as E
from time_varying_modal_fit import _solver_options, _fft_seed, _bound_hits, _failure, _FitIssue

OUT = Path(__file__).resolve().parent / 'modal_transport_20261008'


def fit_curved(signal, dt, linear, settings):
    started = time.perf_counter()
    x = np.asarray(signal, float)
    if x.ndim != 2 or x.shape[1] < 4 or not np.isfinite(x).all():
        return _failure('Invalid measured waveform', 'invalid_input', started)
    if np.any(abs(x) >= 50):
        return _failure('Calibrated cap50 reached', 'saturated_input', started)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive physical dt required')
    if not linear.get('success') or linear.get('bound_hits'):
        return _failure('Interior linear waveform fit required', 'linear_unavailable', started)
    duration = (x.shape[1] - 1) * dt
    centered = x - x.mean(axis=1, keepdims=True)
    energy = float(np.sum(centered**2))
    if energy <= 0 or not np.isfinite(energy):
        return _failure('Nonzero centered energy required', 'invalid_energy', started)
    bounds, halfwidth, max_nfev, tolerances = _solver_options(settings)
    seed, fmin, fmax = _fft_seed(centered, dt, duration)
    if seed is None:
        return _failure('Measured FFT seed unavailable', 'no_fft_seed', started)
    lo, hi = max(fmin, seed-halfwidth/duration), min(fmax, seed+halfwidth/duration)
    margin = min(lo-fmin, fmax-hi)
    if lo >= hi or margin <= 0:
        return _failure('No curvature slack in physical frequency band', 'frequency_bounds', started)
    # Half the band slack each for linear and quadratic frequency variation.
    slope_cap = margin / duration
    curvature_cap = 4 * margin / duration**2
    a, adot, fdot = bounds['alpha'], bounds['alpha_slope'], bounds['frequency_slope']
    acurve = (2*adot[0]/duration, 2*adot[1]/duration)
    fcurve = (2*fdot[0]/duration, 2*fdot[1]/duration)
    lower = np.array([a[0]*duration, adot[0]*duration**2, acurve[0]*duration**3,
        lo*duration, max(fdot[0], -slope_cap)*duration**2, max(fcurve[0], -curvature_cap)*duration**3])
    upper = np.array([a[1]*duration, adot[1]*duration**2, acurve[1]*duration**3,
        hi*duration, min(fdot[1], slope_cap)*duration**2, min(fcurve[1], curvature_cap)*duration**3])
    p = linear['parameters']
    initial = np.array([p['alpha_mid']*duration, p['alpha_slope']*duration**2, 0,
        p['frequency_mid']*duration, p['frequency_slope']*duration**2, 0])
    if np.any(lower >= upper) or not np.isfinite(initial).all() or np.any(initial < lower) or np.any(initial > upper):
        return _failure('Linear warm start outside curved bounds', 'invalid_warm_start', started)
    u = (np.arange(x.shape[1]) - .5*(x.shape[1]-1)) * dt / duration

    def residual(theta):
        am, ad, ac, fm, fd, fc = theta
        envelope = np.exp(am*u + .5*ad*u**2 + ac*u**3/6)
        phase = 2*np.pi*(fm*u + .5*fd*u**2 + fc*u**3/6)
        basis = np.column_stack((np.ones_like(u), envelope*np.cos(phase), envelope*np.sin(phase)))
        if not np.isfinite(basis).all():
            raise _FitIssue('nonfinite_basis', 'Nonfinite curved basis')
        coefficients, _, rank, _ = np.linalg.lstsq(basis, x.T, rcond=None)
        if rank != 3:
            raise _FitIssue('rank_deficient', 'Curved carrier rank deficient', rank)
        values = (basis @ coefficients - x.T).ravel()
        if not np.isfinite(values).all():
            raise _FitIssue('nonfinite_residual', 'Nonfinite curved residual')
        return values

    try:
        result = least_squares(residual, initial, bounds=(lower, upper), jac='3-point', method='trf',
            max_nfev=max_nfev, ftol=tolerances[0], xtol=tolerances[1], gtol=tolerances[2])
        values = residual(result.x)
    except _FitIssue as issue:
        return _failure(str(issue), issue.status, started, basis_rank=issue.rank)
    names = ('alpha_mid', 'alpha_slope', 'alpha_curvature', 'frequency_mid', 'frequency_slope', 'frequency_curvature')
    hits = _bound_hits(result.x, lower, upper, names)
    if not result.success or not np.isfinite(result.x).all():
        return _failure(str(result.message), str(result.status), started, nfev=result.nfev, bound_hits=hits)
    parameters = dict(zip(names, (result.x / np.array([duration,duration**2,duration**3]*2)).tolist()))
    h = .5*duration
    alpha = parameters['alpha_mid'] + h*parameters['alpha_slope'] + .5*h*h*parameters['alpha_curvature']
    frequency = parameters['frequency_mid'] + h*parameters['frequency_slope'] + .5*h*h*parameters['frequency_curvature']
    return dict(success=True, parameters=parameters, alpha_end=float(alpha), frequency_end=float(frequency),
        relative_fit_error=float(values @ values / energy), bound_hits=hits, nfev=int(result.nfev),
        elapsed_seconds=time.perf_counter()-started, status=int(result.status))


def run():
    target = OUT / 'curvature_results.json'
    if target.exists():
        raise RuntimeError('Completed correction results are immutable')
    protocol, source, result, analysis, provenance = E._verify_inputs()
    rows, _, physical, white = E._load_controls(source, result, analysis)
    transport = json.loads((OUT/'results.json').read_text())
    prior = json.loads((E.OUT/'results.json').read_text())
    baseline = {E._key(row['condition']): row for row in prior['critical_only_stage']['rows']}
    probes = {E._key(row['condition']): row for row in transport['rows']}
    records = []
    with np.load(OUT/'traces.npz') as traces:
        for row in rows:
            if row['samples'] != 128 or row['excitation'] != 'critical_only_oracle':
                continue
            key = E._key(row)
            signal, measurement = E._prepare(row, physical, white)
            if measurement['fit_signal_sha256'] != baseline[key]['measurement']['fit_signal_sha256']:
                raise RuntimeError('Original measurement hash mismatch')
            offset = np.asarray(measurement['calibration_offset'])[:,None]
            scale = np.asarray(measurement['calibration_scale'])[:,None]
            draw = white[row['noise_index'],:,:128]
            noise = np.zeros_like(draw) if row['noise']=='clean' else draw*row['absolute_noise_rms_m']/np.sqrt(np.mean(draw**2))
            frozen = np.clip((traces[f'w{row["wave_index"]}_frozen_carrier'].T + noise-offset)/scale,-50,50).astype(np.float32).astype(float)
            fits = {}
            for arm, x, warm in (('physical',signal,baseline[key]['varying_fit']),
                    ('frozen_carrier',frozen,probes[key]['counterfactuals']['frozen_carrier']['fit'])):
                fit = fit_curved(x, source['dt_seconds'], warm, protocol['solver'])
                fits[arm] = dict(fit=fit, score=E._score(fit['alpha_end'],fit['frequency_end'],row['references']['end']))
            records.append(dict(condition=probes[key]['condition'], references=row['references'], fits=fits))
    provenance.update(curvature_protocol_sha256=E.M.G.O.C.sha256(OUT/'curvature_protocol.json'),
        implementation_sha256=E.M.G.O.C.sha256(Path(__file__)), transport_results_sha256=E.M.G.O.C.sha256(OUT/'results.json'),
        transport_traces_sha256=E.M.G.O.C.sha256(OUT/'traces.npz'))
    E.M.G.O.C.save_json(target,E.M.clean_json(dict(protocol=json.loads((OUT/'curvature_protocol.json').read_text()),
        provenance=provenance, rows=records, execution='CPU waveform-only fits, unchanged original controls and physical criteria')))
    print('CARRIER_CURVATURE_COMPLETE',len(records),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
