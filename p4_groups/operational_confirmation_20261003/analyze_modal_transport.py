"""Oracle mechanism contrasts on pinned cached controls, not predictor inputs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import time

import cloudpickle
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eig, expm, solve
from threadpoolctl import threadpool_limits

import experiment_time_varying_fit as E
from time_varying_modal_fit import fit_single_mode
from mechanics.p4_margin_estimation.design_sampler import DesignSample

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT / 'modal_transport_20261008'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum(plate, ratio, previous=None):
    matrix = plate.matrix(ratio)
    roots, vectors = eig(matrix)
    q = plate.basis[:, :plate.count] @ (vectors[:plate.count] / plate.omega[:plate.count, None])
    participation = E.M.G.O.TP._transverse_participation(q, len(plate.omega))
    visible = np.flatnonzero((roots.imag > E.M.G.O.TP.TRANSIENT_FILTER.omega_min) &
                            (participation >= E.M.G.O.TP.TRANSIENT_FILTER.eta_w_min))
    if not len(visible):
        raise RuntimeError('No physically transverse critical branch')
    critical = int(visible[np.argmax(roots.real[visible])])
    chosen = critical if previous is None else int(visible[np.argmin(abs(roots[visible] - previous[0]))])
    vector = vectors[:, chosen].copy()
    if previous is not None:
        overlap = np.vdot(previous[1], vector)
        if abs(overlap) < 1e-8:
            raise RuntimeError('Critical eigenvector overlap collapses')
        vector *= np.exp(-1j * np.angle(overlap))
    vectors[:, chosen] = vector
    unit = np.zeros(len(roots), complex)
    unit[chosen] = 1
    left = solve(vectors.T, unit)
    residual = np.linalg.norm(matrix @ vector - roots[chosen] * vector) / (
        np.linalg.norm(matrix) * np.linalg.norm(vector))
    return roots[chosen], vector, left, bool(chosen == critical), float(residual)


def replay(plate, ratios, times, vectors, raw):
    """Replay the existing zero-input trajectory; no new initial-state draw."""
    initial = np.column_stack((vectors[0].real, vectors[0].imag))
    states = np.empty((len(times), len(initial), 2))
    states[0] = initial
    transition = None
    for j in range(len(times) - 1):
        if j % 2 == 0:
            end = min(j + 2, len(times) - 1)
            midpoint_ratio = .5 * (ratios[j] + ratios[end])
            transition = expm(plate.matrix(midpoint_ratio) * (times[1] - times[0]))
        states[j + 1] = transition @ states[j]
    projection = plate.reduced['projection']
    response_basis = np.einsum('cd,ndk->nck', projection, states)
    coefficients, _, rank, _ = np.linalg.lstsq(response_basis.reshape(-1, 2), raw.T.ravel(), rcond=None)
    if rank != 2:
        raise RuntimeError('Critical-pair initial phase is not identifiable in cached waveform')
    state = states @ coefficients
    reproduced = state @ projection.T
    relative_rms = np.linalg.norm(reproduced - raw.T) / np.linalg.norm(raw)
    initial_coefficient = (coefficients[0] - 1j * coefficients[1]) / 2
    return state, reproduced, initial_coefficient, float(relative_rms)


def physical_contrasts(plate, row, raw, dt):
    times = np.arange(raw.shape[1]) * dt
    ratios = row['end_ratio'] - row['ratio_rate_per_second'] * (times[-1] - times)
    poles, vectors, lefts, branches, residuals = [], [], [], [], []
    previous = None
    for ratio in ratios:
        pole, vector, left, branch, residual = spectrum(plate, ratio, previous)
        poles.append(pole); vectors.append(vector); lefts.append(left)
        branches.append(branch); residuals.append(residual)
        previous = pole, vector
    poles, vectors, lefts = map(np.asarray, (poles, vectors, lefts))
    reference_errors = {}
    for name, index in (('start', 0), ('end', -1)):
        reference = row['references'][name]
        reference_errors[name] = dict(alpha=abs(poles[index].real - reference['alpha']),
            frequency=abs(poles[index].imag / (2 * np.pi) - reference['frequency']))
    midpoint, *_ = spectrum(plate, float(.5 * (ratios[0] + ratios[-1])))
    reference = row['references']['midpoint']
    reference_errors['midpoint'] = dict(alpha=abs(midpoint.real - reference['alpha']),
        frequency=abs(midpoint.imag / (2 * np.pi) - reference['frequency']))
    state, reproduced, coefficient0, replay_error = replay(plate, ratios, times, vectors, raw)
    vector_dot = np.gradient(vectors, dt, axis=0, edge_order=2)
    left_dot = np.gradient(lefts, dt, axis=0, edge_order=2)
    connection = np.einsum('nd,nd->n', lefts, vector_dot)
    projection = plate.reduced['projection']
    shapes = vectors @ projection.T
    integral = cumulative_trapezoid(poles, times, initial=0)
    corrected_integral = cumulative_trapezoid(poles - connection, times, initial=0)
    frozen = 2 * (coefficient0 * np.exp(integral)[:, None] * shapes[0]).real
    moving = 2 * (coefficient0 * np.exp(integral)[:, None] * shapes).real
    diagonal = 2 * (coefficient0 * np.exp(corrected_integral)[:, None] * shapes).real
    coefficient = np.einsum('nd,nd->n', lefts, state)
    if np.min(abs(coefficient)) <= abs(coefficient0) * 1e-8:
        raise RuntimeError('Critical coefficient collapses; log-growth undefined')
    pair = 2 * (shapes * coefficient[:, None]).real
    rest = reproduced - pair
    shape_gain = np.linalg.norm(shapes, axis=1)
    if np.min(shape_gain) <= np.max(shape_gain) * 1e-8:
        raise RuntimeError('Critical sensor projection collapses')
    modal_growth = (poles + np.einsum('nd,nd->n', left_dot, state) / coefficient).real
    gain_growth = np.gradient(np.log(shape_gain), dt, edge_order=2)
    sensor_growth = modal_growth + gain_growth
    centered = reproduced - reproduced.mean(axis=0)
    energy = float(np.sum(centered ** 2))
    contrasts = {'frozen_carrier': frozen, 'moving_shape': moving, 'diagonal_transport': diagonal}
    stats = dict(replay_relative_rms=replay_error, references=reference_errors,
        branch_continuous=all(branches), maximum_eigen_residual=max(residuals),
        maximum_biorthogonality_error=float(np.max(abs(np.einsum('nd,nd->n', lefts, vectors) - 1))),
        constant_shape_carrier_alpha_linear_residual_max=float(np.max(abs(
            poles.real - np.polyval(np.polyfit(times, poles.real, 1), times)))),
        constant_shape_carrier_frequency_linear_residual_max_hz=float(np.max(abs(
            poles.imag / (2*np.pi) - np.polyval(np.polyfit(times, poles.imag / (2*np.pi), 1), times)))),
        endpoint=dict(physical_alpha=float(poles[-1].real), modal_coefficient_growth=float(modal_growth[-1]),
            diagonal_connection_real=float(connection[-1].real), sensor_projection_log_growth=float(gain_growth[-1]),
            sensor_modal_envelope_growth=float(sensor_growth[-1])),
        noncritical_sensor_component_relative_rms=float(np.linalg.norm(rest) / np.linalg.norm(reproduced)),
        waveform_sse_fraction={name: float(np.sum((value - reproduced)**2) / energy)
                               for name, value in contrasts.items()})
    arrays = dict(times=times, poles=poles, vectors=vectors, lefts=lefts, states=state,
        coefficient=coefficient, connection=connection, shape_gain=shape_gain,
        sensor_growth=sensor_growth, reproduced=reproduced, pair=pair, **contrasts)
    return stats, arrays


def run():
    if (OUT / 'results.json').exists():
        raise RuntimeError('Completed diagnostics are immutable')
    contract = json.loads((OUT / 'protocol.json').read_text())
    old_protocol, source, result, analysis, provenance = E._verify_inputs()
    rows, _, physical, white = E._load_controls(source, result, analysis)
    old = json.loads((E.OUT / 'results.json').read_text())
    baseline = {E._key(r['condition']): r for r in old['critical_only_stage']['rows']}
    if len(baseline) != 48:
        raise RuntimeError('Expected all48 saved physical fits')
    with np.load(SCRIPT / 'remote_input.npz') as z:
        data = cloudpickle.loads(z['data'].tobytes())
        fingerprint = json.loads(str(z['fingerprint']))
    if fingerprint != result['fingerprint']:
        raise RuntimeError('Physical input fingerprint mismatch')
    manifest_path = E.ROOT / 'p4_sensor_candidates/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sensors = np.asarray(manifest['candidate_grid_yx'])[data['layout']]
    plates, identities, identity_hashes = {}, {}, {}
    for case in source['cases']:
        did = case['design']
        path = SCRIPT / 'o1_corrected' / f'trajectories_{did}.npz'
        with np.load(path) as z:
            identity = json.loads(str(z['identity']))
        if identity['fingerprint'] != fingerprint:
            raise RuntimeError(f'Identity fingerprint mismatch: {did}')
        audit = json.loads((path.parent / f'audit_{did}.json').read_text())
        if not audit['acquisition_supported'] or audit['fingerprint'] != fingerprint:
            raise RuntimeError(f'Unsupported physical control: {did}')
        plates[did] = E.M.G.O.TP.PlateDynamics(DesignSample(**identity['design']), identity['ucrit'], sensors)
        identities[did], identity_hashes[did] = identity, digest(path)
    chosen = [row for row in rows if row['samples'] == 128 and row['excitation'] == 'critical_only_oracle']
    if len(chosen) != 48:
        raise RuntimeError('Selection must retain all48 critical views')
    cache, records, arrays = {}, [], {}
    started = time.perf_counter()
    checks = contract['checks']
    for row in chosen:
        wave = row['wave_index']
        if wave not in cache:
            stats, traces = physical_contrasts(plates[row['design']], row, physical[wave, :, :128], source['dt_seconds'])
            if stats['replay_relative_rms'] > checks['cached_replay_relative_rms']:
                raise RuntimeError(f'Cached replay mismatch: {E._key(row)} {stats["replay_relative_rms"]}')
            if stats['maximum_eigen_residual'] > checks['physical_eigen_residual_relative']:
                raise RuntimeError('Physical eigensystem residual fails')
            for ref in stats['references'].values():
                if ref['alpha'] > checks['reference_alpha_error_per_second'] or ref['frequency'] > checks['reference_frequency_error_hz']:
                    raise RuntimeError('Physical references disagree with cached truth')
            cache[wave] = stats, traces
            for key, value in traces.items():
                arrays[f'w{wave}_{key}'] = value
        stats, traces = cache[wave]
        signal, measurement = E._prepare(row, physical, white)
        previous = baseline[E._key(row)]
        if measurement['fit_signal_sha256'] != previous['measurement']['fit_signal_sha256']:
            raise RuntimeError('Original measured calibration differs from pinned baseline')
        offset = np.asarray(measurement['calibration_offset'])[:, None]
        scale = np.asarray(measurement['calibration_scale'])[:, None]
        draw = white[row['noise_index'], :, :128]
        noise = np.zeros_like(draw) if row['noise'] == 'clean' else draw * (
            row['absolute_noise_rms_m'] / np.sqrt(np.mean(draw**2)))
        fits = {}
        for name in ('frozen_carrier', 'moving_shape', 'diagonal_transport'):
            measured = np.clip((traces[name].T + noise - offset) / scale, -50, 50).astype(np.float32).astype(float)
            stationary = fit_single_mode(measured, source['dt_seconds'], varying=False, settings=old_protocol['solver'])
            fitted = fit_single_mode(measured, source['dt_seconds'], varying=True, settings=old_protocol['solver'], stationary=stationary)
            fits[name] = dict(fit=fitted, score=E._score(fitted['alpha_end'], fitted['frequency_end'], row['references']['end']))
        records.append(dict(condition=previous['condition'], physical=stats,
            baseline=previous['varying_fit'], baseline_score=previous['varying_end_score'], counterfactuals=fits))
        print('TRANSPORT_VIEW_COMPLETE', row['design'], row['end_ratio'], row['ratio_rate_per_second'], row['noise'], flush=True)
    provenance.update(protocol_sha256=digest(OUT / 'protocol.json'), baseline_results_sha256=digest(E.OUT / 'results.json'),
        implementation_sha256=digest(__file__), input_archive_sha256=digest(SCRIPT / 'remote_input.npz'),
        sensor_manifest_sha256=digest(manifest_path), identity_archive_sha256=identity_hashes)
    output = dict(protocol=contract, provenance=provenance, identities=identities, rows=records,
        waveform_count=len(cache), view_count=len(records), elapsed_seconds=time.perf_counter() - started,
        execution='CPU FP64 oracle spectral analysis and independent replay of pinned CUDA controls; no production backend change')
    E.M.G.O.C.save_npz(OUT / 'traces.npz', **arrays)
    E.M.G.O.C.save_json(OUT / 'results.json', E.M.clean_json(output))
    print('MODAL_TRANSPORT_COMPLETE', len(cache), len(records), output['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
