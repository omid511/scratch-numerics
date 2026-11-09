"""Frozen absolute-velocity preflight: acquire/account only; never train."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(SCRIPT)]
import diagnostic_matrix as M
import gpu_execution as G
import multipoint_numerics_20261009 as N
from mechanics.p4_margin_estimation.modal_features import estimate_modes
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
OUT = SCRIPT / 'multipoint_preflight_20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    if path.exists():
        raise RuntimeError(f'immutable preflight artifact exists: {path}')
    path.write_text(json.dumps(M.clean_json(value), indent=2, allow_nan=False) + '\n')


def forecast_skill(signal, mode):
    """CPU transcription of frozen prefix64/suffix64 forecast qualification."""
    if not len(mode['alpha']):
        return None
    x = np.asarray(signal, dtype=float)
    scale = np.max(abs(x[:, :64]))
    if scale == 0:
        return None
    x = x / scale
    times = np.arange(128) * .001
    growth = mode['alpha'][:, None] * times
    envelope = np.exp(growth - growth.max(axis=1, keepdims=True))
    phase = 2 * np.pi * mode['frequency'][:, None] * times
    basis = np.column_stack((np.ones(128), (envelope * np.cos(phase)).T,
                             (envelope * np.sin(phase)).T))
    coefficients = np.linalg.lstsq(basis[:64], x[:, :64].T, rcond=None)[0]
    suffix = x[:, 64:].T
    centered = float(np.sum((suffix - suffix.mean(axis=0, keepdims=True))**2))
    energy = float(np.sum(suffix**2))
    if centered <= np.finfo(float).eps * energy:
        return None
    skill = 1 - float(np.sum((suffix - basis[64:] @ coefficients)**2)) / centered
    return skill if np.isfinite(skill) else None


def history(signal, d, ridge):
    accepted = np.zeros(3, dtype=bool)
    features = np.zeros((3, 174))
    skills = []
    saturated = []
    for slot, window_slice in enumerate(N.PLATEAU_SLICES.values()):
        window = signal[:, window_slice]
        saturated.append(bool((abs(window) >= 50).any()))
        skill = None
        if np.isfinite(window).all() and not saturated[-1]:
            prefix = estimate_modes(window[:, :64], .001)
            skill = forecast_skill(window, prefix)
            if skill is not None and skill >= .5:
                mode = estimate_modes(window, .001, include_shapes=True)
                if len(mode['alpha']):
                    row = np.r_[G.O.B.base_feature_row(window, .001, mode, ridge),
                                G.O.B.modal_spatial_row(mode, d['pairs']),
                                G.O.B.modal_spatial_addition_row(mode, d['pairs'])]
                    if row.shape != (174,) or not np.isfinite(row).all():
                        raise RuntimeError('accepted window produced invalid174-coordinate feature row')
                    features[slot] = row
                    accepted[slot] = True
        skills.append(skill)
    ages = np.empty(3)
    last = 0.
    for slot, timestamp in enumerate((.127, .287, .447)):
        if accepted[slot]:
            last = timestamp
        ages[slot] = timestamp - last
    return features, accepted, ages, skills, saturated


def missing_floor(records, observed, mask):
    """Optimistic L1 lower bound for the exactly equal all-missing input group.

    Weights give each represented design equal total contribution, including
    nonmissing rows in each design denominator. Optimize a common missing
    output only; assume every other output can be perfect. Not an SNR bound.
    """
    indices = np.flatnonzero(mask)
    if not len(indices):
        return dict(rows=0, designs=0, all_missing_rows=0, optimal_missing_output=None, mae_floor=None)
    designs = np.array([row['design'] for row in records])
    labels = np.array([row['margin'] for row in records])
    represented = np.unique(designs[indices])
    weights = np.zeros(len(records))
    for design in represented:
        selected = indices[designs[indices] == design]
        weights[selected] = 1. / (len(represented) * len(selected))
    missing = indices[~observed[indices].any(axis=1)]
    if not len(missing):
        return dict(rows=len(indices), designs=len(represented), all_missing_rows=0,
                    optimal_missing_output=None, mae_floor=0.)
    order = missing[np.argsort(labels[missing], kind='stable')]
    cumulative = np.cumsum(weights[order])
    median = labels[order[np.searchsorted(cumulative, .5 * cumulative[-1])]]
    floor = float(np.sum(weights[missing] * abs(labels[missing] - median)))
    return dict(rows=len(indices), designs=len(represented), all_missing_rows=len(missing),
                optimal_missing_output=float(median), mae_floor=floor)


def regions(labels):
    return dict(all=np.ones(len(labels), dtype=bool), boundary=(labels > -.02) & (labels < .05),
        stable_sign=(labels > .01) & (labels < .05), unstable_sign=(labels <= -.01) & (labels >= -.05),
        farther_stable15=labels >= .15)


def relative_rms(actual, reference):
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return None
    denominator = float(np.linalg.norm(reference))
    if denominator == 0:
        return 0. if np.array_equal(actual, reference) else None
    return float(np.linalg.norm(actual - reference) / denominator)


def numerical_checks(p, d, fp):
    results = []
    for did in p['designs']:
        _, plate = M.context(d, fp, dict(design=did, replicate=0))
        for endpoint in p['numerical_endpoints_mps']:
            report = N.numerical_report(plate, endpoint, p['delta_mps'])
            report['design'] = did
            report['adaptive_ode_replay'] = {}
            for arm in p['arms']:
                # Independent adaptive ODE integration, not a second midpoint implementation.
                initial = plate.initial(0)[0]
                oracle = solve_ivp(lambda t, state: N.absolute_matrix(plate,
                    N.velocity_at(t, endpoint, p['delta_mps'], arm)) @ state,
                    (0., .447), initial, method='DOP853', t_eval=N.OBS_TIMES,
                    rtol=1e-9, atol=1e-12, max_step=.0005)
                if not oracle.success or oracle.y.shape[1] != 448:
                    raise RuntimeError('independent ODE replay failed')
                actual = N.acquire(plate, endpoint, p['delta_mps'], arm, coefficient_step=.00025)
                report['adaptive_ode_replay'][arm] = dict(
                    response_relative_rms=relative_rms(actual.response, plate.reduced['projection'] @ oracle.y),
                    final_state_relative_error=relative_rms(actual.final_state, oracle.y[:, -1]))
            results.append(report)
            print('MULTIPOINT_NUMERICAL', did, endpoint,
                  report['numerical_pass'], report['rom_pass'], flush=True)
    save('numerics.json', dict(results=results, protocol_sha256=sha(OUT / 'protocol.json'),
        full_fits_authorized=False, physical_rate_validated=False))


def accounting(p, d, fp):
    records = []
    rows = []
    observed = []
    ages = []
    physical = []
    ridge = PhysicsFeatureRidge()
    started = time.perf_counter()
    for did in p['designs']:
        identity, plate = M.context(d, fp, dict(design=did, replicate=0))
        for endpoint in p['endpoints_mps']:
            roots = [eig(N.absolute_matrix(plate, speed), right=False) for speed in
                     (endpoint - p['delta_mps'], endpoint, endpoint + p['delta_mps'])]
            physical.append(dict(design=did, endpoint_mps=endpoint,
                maximum_growth_per_second=[float(values.real.max()) for values in roots],
                positive_frequency_hz=[sorted((values.imag[values.imag > 0] / (2 * np.pi)).tolist()) for values in roots],
                minimum_mach=(endpoint - p['delta_mps']) / plate.design.c_air))
            for replicate in p['replicates']:
                acquired = {arm: N.acquire(plate, endpoint, p['delta_mps'], arm, replicate,
                    coefficient_step=.0005) for arm in p['arms']}
                noises, sigma = M.noises(acquired['S'].response,
                    np.random.SeedSequence([20261009, int(did[1:]), endpoint, replicate]))
                # Reuse actual physical perturbations, not just shared seed/percentages.
                for arm, acquisition in acquired.items():
                    for noise_name in p['noise_views']:
                        signal, offset, scale = M.normalize(acquisition.response, noises[noise_name])
                        features, good, age, skills, saturated = history(signal, d, ridge)
                        records.append(dict(design=did, endpoint_mps=endpoint, arm=arm, noise=noise_name,
                            replicate=replicate, margin=1 - endpoint / identity['ucrit'],
                            valid_samples=int(acquisition.valid.sum()), termination=acquisition.termination,
                            accepted_slots=good.tolist(), forecast_skills=skills, saturated_slots=saturated,
                            sigma_reference_S=float(sigma), calibration_scale=scale.tolist()))
                        rows.append(features)
                        observed.append(good)
                        ages.append(age)
        print('MULTIPOINT_ACCOUNTED', did, 'records', len(records), flush=True)
    observed = np.stack(observed)
    labels = np.array([record['margin'] for record in records])
    masks = regions(labels)
    summaries = {}
    for noise in p['noise_views']:
        for arm in [*p['arms'], 'shared_all_arms']:
            population = np.array([row['noise'] == noise and (arm == 'shared_all_arms' or row['arm'] == arm)
                                   for row in records])
            summaries[f'{noise}/{arm}'] = dict(histories=int(population.sum()),
                accepted_slot_counts=observed[population].sum(axis=0).tolist(),
                all_missing_histories=int((~observed[population].any(axis=1)).sum()),
                saturated_slot_counts=np.array([row['saturated_slots'] for row in records])[population].sum(axis=0).tolist(),
                target_regions={name: missing_floor(records, observed, population & region) for name, region in masks.items()})
    save('accounting.json', dict(records=records, summaries=summaries, physical_contrast=physical,
        seconds=time.perf_counter() - started, protocol_sha256=sha(OUT / 'protocol.json'),
        limitation='Proper-training within-simulator accounting only; CPU transcription of forecast gate, not independent physical validation or endpoint qualification.',
        all_denominators_preserved=True, full_fits_authorized=False))
    dataset = OUT / 'dataset.npz'
    if dataset.exists():
        raise RuntimeError('immutable preflight dataset exists')
    np.savez_compressed(dataset, features=np.stack(rows), observed=observed, age=np.stack(ages),
                        truth=labels, records=json.dumps(records), protocol_sha256=sha(OUT / 'protocol.json'))
    print('MULTIPOINT_PREFLIGHT_ACCOUNTING', len(records), 'histories', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('numerics', 'account'))
    args = parser.parse_args()
    p = json.loads((OUT / 'protocol.json').read_text())
    if p['full_fits_authorized'] or p['designs'] != ['D000007', 'D000033', 'D000068', 'D000087']:
        raise RuntimeError('proper-training feasibility contract changed')
    d, fp = G.inputs()
    with threadpool_limits(limits=1):
        (numerical_checks if args.phase == 'numerics' else accounting)(p, d, fp)


if __name__ == '__main__':
    main()
