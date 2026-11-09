"""Paired canonical/proposed batching on a frozen label-free measured sample."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(SCRIPT)]
import experiment_support_modal_v2 as E
from cuda_backend import estimate_batch
from extract_gate_bypass_receipt import OUT, sha, valid_mode


def differences(left, right, scale, active):
    delta = abs(left - right)
    standardized = delta / scale
    standardized[~active] = 0
    return dict(raw_max=float(delta.max()), standardized_max=float(standardized.max()),
        raw_by_coordinate=delta.tolist(), standardized_by_coordinate=standardized.tolist(),
        top_raw_coordinates=np.argsort(delta)[-8:][::-1].tolist(),
        top_standardized_coordinates=np.argsort(standardized)[-8:][::-1].tolist())


def characterize(protocol_path=None, output_path=None):
    protocol_path = Path(protocol_path) if protocol_path is not None else OUT / 'batching_protocol.json'
    output_path = Path(output_path) if output_path is not None else OUT / 'batching_characterization.json'
    p = json.loads(protocol_path.read_text())
    assert p['source_protocol_sha256'] == sha(OUT / 'protocol.json')
    with np.load(E.OUT / 'dataset.npz', allow_pickle=False) as z:
        records = json.loads(str(z['records']))
        observed, cached = z['observed'].copy(), z['features'].copy()
    with np.load(E.SHARDS / 'D000007.npz', allow_pickle=False) as z:
        physical_records = json.loads(str(z['records']))
        index = next(i for i, r in enumerate(physical_records) if r['profile'] == 'cross' and r['replicate'] == 0)
        raw, command = z['raw'][index].copy(), z['command'][index].copy()
    payload, _ = E.G.inputs()
    d = {'pairs': payload['pairs']}
    del payload
    engine = SimpleNamespace(ridge=E.G.O.B.PhysicsFeatureRidge())
    with np.load(SCRIPT / 'bounded_gru_20261009_v2/fold0_content_gru_seed42.npz', allow_pickle=False) as z:
        mean, scale, active = [z[k].copy() for k in ('mean', 'scale', 'active')]
    names = E.BASE_NOISE
    row_ids = {name: next(i for i, r in enumerate(records) if r['design'] == 'D000007'
        and r['replicate'] == 0 and r['profile'] == 'cross' and r['noise'] == name) for name in names}
    signals = {name: E.M.normalize(raw, E.noise_for(raw, records[row_ids[name]]), cast=True)[0] for name in names}
    ends = np.arange(128, raw.shape[-1] + 1, 64)
    windows = {name: {} for name in names}
    proposed_modes, proposed_skills = {}, {}
    for name in names:
        rid = row_ids[name]
        for frame, end in enumerate(ends):
            w = signals[name][:, end - 128:end]
            if not command[end - 128:end].any() and np.isfinite(w).all() and not (abs(w) >= 50).any():
                windows[name][frame] = w
        frames = list(windows[name])
        prefix = estimate_batch([windows[name][f][:, :64] for f in frames])
        skill = E.P.forecast_skills([windows[name][f] for f in frames], prefix, 64, 64)
        proposed_skills[name] = dict(zip(frames, skill))
        accepted = [f for f in frames if observed[rid, f]]
        rejected = [f for f, s in zip(frames, skill) if not observed[rid, f] and s < .5]
        proposed_modes[name] = {}
        for group in (accepted, rejected):
            if group:
                modes = estimate_batch([windows[name][f] for f in group])
                proposed_modes[name].update(zip(group, modes))
    canonical_cache = {}
    for sample in p['samples']:
        frame, name = sample['frame'], sample['noise']
        source_group = ('clean', 'iid1') if name in ('clean', 'iid1') else ('iid5', 'correlated1', 'correlated5')
        key = (frame, source_group)
        if key in canonical_cache:
            continue
        candidates = [noise for noise in source_group if frame in windows[noise]]
        batch = [windows[noise][frame] for noise in candidates]
        prefix = estimate_batch([w[:, :64] for w in batch])
        skill = E.P.forecast_skills(batch, prefix, 64, 64)
        canonical = {}
        for group in ([noise for noise, s in zip(candidates, skill) if s >= .5],
                      [noise for noise, s in zip(candidates, skill) if s < .5 and not observed[row_ids[noise], frame]]):
            if group:
                modes = estimate_batch([windows[noise][frame] for noise in group])
                canonical.update(zip(group, modes))
        canonical_cache[key] = (canonical, dict(zip(candidates, skill)))
    reports = []
    for sample in p['samples']:
        frame, name = sample['frame'], sample['noise']
        rid = row_ids[name]
        assert rid == sample['row']
        source_group = ('clean', 'iid1') if name in ('clean', 'iid1') else ('iid5', 'correlated1', 'correlated5')
        canonical, canonical_skills = canonical_cache[(frame, source_group)]
        left, right = canonical.get(name), proposed_modes[name].get(frame)
        result = dict(sample=sample, end_exclusive=int(ends[frame]), identical_measured_samples=True,
            canonical_forecast_skill=float(canonical_skills[name]), proposed_forecast_skill=float(proposed_skills[name][frame]),
            canonical_forecast_pass=bool(canonical_skills[name] >= .5), proposed_forecast_pass=bool(proposed_skills[name][frame] >= .5),
            cached_observed=bool(observed[rid, frame]), canonical_mode_count=len(left['alpha']) if left else None,
            proposed_mode_count=len(right['alpha']) if right else None)
        if left is None or right is None or not valid_mode(left) or not valid_mode(right):
            result['comparison_unavailable'] = 'Missing or invalid mode estimate under one batching policy; not silently substituted'
            reports.append(result)
            continue
        window = windows[name][frame]
        canonical_row = E.M.features(window, left, d, engine)
        proposed_row = E.M.features(window, right, d, engine)
        repeated = E.M.features(window, left, d, engine)
        result['assembly_only_repeat_byte_identical'] = np.array_equal(canonical_row.view(np.uint8), repeated.view(np.uint8))
        result['canonical_vs_proposed_features'] = differences(canonical_row, proposed_row, scale, active)
        if observed[rid, frame]:
            result['canonical_vs_cached_accepted'] = differences(canonical_row, cached[rid, frame], scale, active)
            result['proposed_vs_cached_accepted'] = differences(proposed_row, cached[rid, frame], scale, active)
        spatial_orders = []
        for mode in (left, right):
            visible = np.flatnonzero(mode['energy_fraction'] >= .01)
            spatial_orders.append(visible[np.lexsort((mode['frequency'][visible], -mode['alpha'][visible]))].tolist())
        result['canonical_least_stable_visible_order'] = spatial_orders[0]
        result['proposed_least_stable_visible_order'] = spatial_orders[1]
        result['same_spatial_selection'] = spatial_orders[0] == spatial_orders[1]
        if len(left['alpha']) == len(right['alpha']):
            result['same_frequency_slot_order'] = bool(np.array_equal(np.argsort(left['frequency']), np.argsort(right['frequency'])))
            result['ordered_mode_max_abs_difference'] = {key: float(abs(left[key] - right[key]).max())
                for key in ('alpha', 'frequency', 'energy_fraction')}
            shape_delta = left['mode_shapes'] - right['mode_shapes']
            result['ordered_complex_coefficient_max_abs_difference'] = float(abs(shape_delta).max())
            result['ordered_complex_coefficient_relative_l2_difference'] = float(np.linalg.norm(shape_delta) / max(np.linalg.norm(left['mode_shapes']), np.finfo(float).tiny))
            if result['same_frequency_slot_order']:
                hybrid = dict(left, mode_shapes=right['mode_shapes'])
                hybrid_row = E.M.features(window, hybrid, d, engine)
                result['coefficients_only_feature_difference'] = differences(canonical_row, hybrid_row, scale, active)
                result['remaining_poles_energies_feature_difference'] = differences(hybrid_row, proposed_row, scale, active)
        reports.append(result)
    result = E.clean_json(dict(scope='Frozen paired batching diagnostic; not a population-wide reproducibility or accuracy proof',
        selection=p['selection'],
        samples=reports, protocol_sha256=sha(protocol_path), source_sha256=sha(__file__),
        accepted_cache_preserved=True, no_fits=True, no_tolerance_changes=True,
        limits='No true modes/branches or labels used. Ordered coefficient comparisons are not physical branch attribution. Sample cannot exclude batching effects elsewhere.'))
    E.save_json(output_path, result)
    print('GATE_BATCHING_CHARACTERIZED', len(reports), flush=True)
    return result


if __name__ == '__main__':
    characterize()
