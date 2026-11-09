"""Independent calibrated mixture/slot checks before diagnostic model fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(SCRIPT.parent)]
import gpu_execution as G
from mechanics.p4_margin_estimation.modal_features import modal_feature_row, physics_feature_groups, modal_spatial_row, modal_spatial_addition_row


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def injected(raw, record):
    name = record['noise']; seed = 20261010
    if '_fresh' in name:
        name, suffix = name.split('_fresh'); seed = int(suffix)
    profile = ('cross', 'stable', 'retreat', 'farther_stable', 'unstable_exclusion').index(record['profile'])
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(record['design'][1:]), record['replicate'], profile]))
    white = rng.normal(size=raw.shape); shared = rng.normal(size=(1, raw.shape[1]))
    level = .05 if name.endswith('5') else .01
    sigma = np.sqrt(((raw[:, :51] - raw[:, :51].mean(axis=1, keepdims=True)) ** 2).mean())
    if name == 'clean':
        return np.zeros_like(raw)
    if name.startswith('iid'):
        return level * sigma * white
    correlated = np.sqrt(.2) * white + np.sqrt(.8) * shared
    return level * sigma * correlated * np.sqrt((white ** 2).mean(axis=1, keepdims=True) / (correlated ** 2).mean(axis=1, keepdims=True))


def physical_table(shard, record, state_index, protocol):
    ratios = np.interp(np.arange(8001) * .001, record['schedule']['knots'], record['schedule']['ratios'])
    grid = np.unique(ratios)
    extra = []
    for a, b in zip(grid[:-1], grid[1:]):
        if b - a > protocol['tracking']['max_ratio_step']:
            extra.extend(np.linspace(a, b, int(np.ceil((b - a) / protocol['tracking']['max_ratio_step'])) + 1)[1:-1])
    grid = np.unique(np.r_[grid, extra]); ids = np.searchsorted(grid, ratios)
    poles, shapes, residual = [], [], 0.
    previous_roots = previous_vectors = None
    identity = json.loads(str(shard['identity'])); count = shard['projection'].shape[1] // 2
    for ratio in grid:
        velocity = ratio * identity['ucrit']; mach = velocity / identity['design']['c_air']
        dynamic = identity['design']['rho_air'] * velocity**2 / np.sqrt(mach**2 - 1)
        damping = dynamic * (mach**2 - 2) / (mach**2 - 1) / velocity
        matrix = shard['base'] + dynamic * shard['stiffness'] + damping * shard['damping']
        roots, vectors = eig(matrix, check_finite=False)
        positive = np.flatnonzero(roots.imag > 1e-8); r, v = roots[positive], vectors[:, positive]
        if previous_roots is None:
            perm = np.argsort(r.imag)
        else:
            separation = abs(previous_roots[:, None] - previous_roots[None, :]); np.fill_diagonal(separation, np.inf)
            separation = np.maximum(separation.min(axis=1), 1e-8)
            displacement = abs(previous_roots[:, None] - r[None, :]) / separation[:, None]
            overlap = abs(previous_vectors.conj().T @ v) / (np.linalg.norm(previous_vectors, axis=0)[:, None] * np.linalg.norm(v, axis=0)[None])
            rows, perm = linear_sum_assignment(displacement + 1 - np.minimum(overlap, 1))
            assert np.array_equal(rows, np.arange(count))
        r, v = r[perm], v[:, perm]
        residual = max(residual, float((np.linalg.norm(matrix @ v - v * r, axis=0) / (np.linalg.norm(matrix) * np.linalg.norm(v, axis=0))).max()))
        poles.append(r); shapes.append(shard['projection'] @ v)
        previous_roots, previous_vectors = r, v
    return grid, ids, np.stack(poles), np.stack(shapes), residual


def expected_row(measured, physical, original, pairs):
    # Independently assemble the counterfactual, preserving measured selection identities.
    row = original.copy(); row[7:22] = modal_feature_row(physical)
    groups = physics_feature_groups(physical)
    visible = np.flatnonzero(measured['energy_fraction'] >= .01)
    a0, f0 = measured['alpha'], measured['frequency']
    order = visible[np.lexsort((f0[visible], -a0[visible]))]
    a, f, energy = physical['alpha'], physical['frequency'], physical['energy_fraction']
    stability = np.zeros(10); stability[-1] = len(order)
    for slot, index in enumerate(order[:3]):
        stability[3 * slot:3 * slot + 3] = a[index], f[index], energy[index]
    spacing = np.zeros(4)
    if len(order) > 1:
        lead = order[0]; others = order[1:]
        neighbor = others[np.argmin(abs(f0[others] - f0[lead]))]
        i, j = np.triu_indices(len(visible), 1)
        natural = np.hypot(a, 2 * np.pi * f)
        spacing[:] = 2 * abs(f[lead] - f[neighbor]) / (f[lead] + f[neighbor]), np.min(2 * abs(f[visible[i]] - f[visible[j]]) / (f[visible[i]] + f[visible[j]])), 2 * np.hypot(a[lead] - a[neighbor], 2 * np.pi * (f[lead] - f[neighbor])) / (natural[lead] + natural[neighbor]), 1
    row[22:41] = np.r_[groups['damping'], stability, spacing]
    routing = dict(physical, alpha=a0, frequency=f0)
    row[42:107] = modal_spatial_row(routing, pairs); row[107:] = modal_spatial_addition_row(routing, pairs)
    return row


def run():
    protocol = json.loads((SCRIPT / 'protocol.json').read_text()); d, _ = G.inputs()
    with np.load(SCRIPT / 'dataset.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files if k != 'records'}; records = json.loads(str(z['records']))
    q, oracle, observed = data['features'], data['oracle_features'], data['observed']
    np.testing.assert_array_equal(q[..., :7], oracle[..., :7]); np.testing.assert_array_equal(q[..., 41], oracle[..., 41])
    np.testing.assert_array_equal(q[~observed], oracle[~observed])
    unmatched = ~(data['matched_branch'] >= 0).any(axis=2)
    np.testing.assert_array_equal(q[unmatched], oracle[unmatched])
    assert {r['profile'] for r in records} == set(protocol['profiles'])
    checked = 0; sensor_error = 0.; row_error = 0.; eigen_error = 0.; design_checks = {}
    designs = protocol['split']['train'] + protocol['split']['validation'] + protocol['split']['held'] + protocol['additional_designs']
    for did in designs:
        with np.load(SCRIPT / 'shards' / f'{did}.npz', allow_pickle=False) as shard:
            physical_records = json.loads(str(shard['records']))
            validation = json.loads(str(shard['validation']))
            assert validation['minimum_mach'] >= 2
            assert {r['profile'] for r in physical_records} == set(protocol['applicable_profiles'][did])
            for item in validation['extreme']:
                assert item['critical_relative_pole_gap'] <= .005
                assert item['full_response_relative_rms'] <= .02
                assert item['step_halving_relative_rms'] <= .005
            selected = [i for i, r in enumerate(records) if r['design'] == did and r['noise'] == 'clean' and r['replicate'] == 0]
            design_checks[did] = 0
            for ri in selected:
                record = records[ri]
                frames = np.flatnonzero(observed[ri] & (data['matched_branch'][ri] >= 0).any(axis=1))
                if not len(frames):
                    continue
                pi = next(i for i, r in enumerate(physical_records) if r['replicate'] == record['replicate'] and r['profile'] == record['profile'])
                raw = shard['raw'][pi]; noise = injected(raw, record)
                prefix = raw[:, :51] + noise[:, :51]; offset = prefix.mean(axis=1)
                scale = np.maximum(np.sqrt(((prefix - offset[:, None]) ** 2).mean(axis=1)), 1e-8)
                grid, ids, poles, shapes, residual = physical_table(shard, record, pi, protocol)
                eigen_error = max(eigen_error, residual)
                for frame in frames[np.unique(np.linspace(0, len(frames) - 1, min(3, len(frames)), dtype=int))]:
                    end = 128 + 64 * int(frame); count = int(data['mode_count'][ri, frame])
                    mode = dict(alpha=data['measured_alpha'][ri, frame, :count].copy(), frequency=data['measured_frequency'][ri, frame, :count].copy(),
                        energy_fraction=data['measured_energy_fraction'][ri, frame, :count].copy(), mode_shapes=data['measured_mode_shapes'][ri, frame, :count].copy(),
                        relative_fit_error=float(data['measured_fit_error'][ri, frame]), usable_fraction=float(data['measured_usable_fraction'][ri, frame]))
                    physical = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in mode.items()}
                    index = ids[end - 1]
                    for slot, branch in enumerate(data['matched_branch'][ri, frame, :count]):
                        if branch >= 0:
                            assert data['match_scores'][ri, frame, slot, 0] >= .8
                            assert data['match_scores'][ri, frame, slot, 0] - data['match_scores'][ri, frame, slot, 1] >= .1
                            physical['alpha'][slot] = poles[index, branch].real
                            physical['frequency'][slot] = poles[index, branch].imag / (2 * np.pi)
                            physical['mode_shapes'][slot] = shapes[index, :, branch] / scale
                    expected = expected_row(mode, physical, q[ri, frame], d['pairs'])
                    error = float(abs(expected - oracle[ri, frame]).max()); row_error = max(row_error, error)
                    assert error <= 1e-8, (did, frame, error)
                    state = shard['states'][pi, end - 1]; actual = state @ shard['projection'].T
                    error = float(np.linalg.norm(actual - raw[:, end - 1]) / max(np.linalg.norm(actual), 1e-30))
                    sensor_error = max(sensor_error, error); assert error <= 1e-9
                    checked += 1; design_checks[did] += 1
    report = dict(dataset_sha256=sha(SCRIPT / 'dataset.npz'), protocol_sha256=sha(SCRIPT / 'protocol.json'),
        verification_source_sha256=sha(__file__), independent_oracle_windows=checked, checked_by_design=design_checks,
        maximum_counterfactual_feature_error=row_error, maximum_state_sensor_error=sensor_error, maximum_eigen_residual=eigen_error,
        original_gates_envelope_strength_routing_preserved=True, added_regimes_in_both_supports=True,
        limitation='Clean replicate0 matched-window spot checks; correspondence ambiguity/coverage remain explicit, not an oracle upper bound')
    (SCRIPT / 'acquisition_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SUPPORT_ACQUISITION_INDEPENDENTLY_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
