"""Exact, original-denominator collision and leverage receipt; never fits."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(SCRIPT)]
from experiment_bounded_gru_v2 import make_slots
OUT = SCRIPT / 'gate_bypass_receipt_20261010'
SOURCE = SCRIPT / 'support_modal_20261009_v2'
BASELINE = SCRIPT / 'bounded_gru_20261009_v2'
SEEDS = (42, 43, 44)
ARMS = ('content_gru', 'ordered_concat', 'unordered_pool', 'framewise_reference')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def original_population_weights(mask, designs):
    weights = np.zeros(mask.shape, dtype=np.float64)
    represented = [d for d in np.unique(designs) if (mask & (designs == d)[:, None]).any()]
    for design in represented:
        selected = mask & (designs == design)[:, None]
        weights[selected] = 1.0 / (len(represented) * int(selected.sum()))
    if represented:
        assert abs(weights.sum() - 1) < 1e-12
    return weights


def weighted_collision_floor(inputs, truth, weights, designs=None):
    """Condition on exact bytes; minimize weighted absolute error per class."""
    x = np.ascontiguousarray(inputs.reshape(-1, inputs.shape[-1]))
    y, w = truth.reshape(-1), weights.reshape(-1)
    selected = w > 0
    x, y, w = x[selected], y[selected], w[selected]
    if not len(y):
        return dict(floor=None, frames=0, classes=0, collision_groups=0, collision_frames=0,
                    conflicting_groups=0, per_design={}, largest_contributing_classes=[])
    keys = x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()
    _, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(groups, kind='stable')
    contributions = np.zeros(len(y))
    class_reports = []
    boundaries = np.r_[0, np.cumsum(counts)]
    for group, count in enumerate(counts):
        if count == 1:
            continue
        indices = order[boundaries[group]:boundaries[group + 1]]
        ys, ws = y[indices], w[indices]
        value_order = np.argsort(ys, kind='stable')
        cumulative = np.cumsum(ws[value_order])
        median = float(ys[value_order[np.searchsorted(cumulative, ws.sum() / 2, side='left')]])
        contributions[indices] = ws * abs(ys - median)
        class_reports.append(dict(frames=int(count), weight=float(ws.sum()), median=median,
            minimum_truth=float(ys.min()), maximum_truth=float(ys.max()),
            contribution=float(contributions[indices].sum())))
    per_design = {}
    if designs is not None:
        d = np.broadcast_to(np.asarray(designs)[:, None], truth.shape).ravel()[selected]
        per_design = {design: float(contributions[d == design].sum()) for design in np.unique(d)}
    return dict(floor=float(contributions.sum()), frames=len(y), classes=len(counts),
        collision_groups=int((counts > 1).sum()), collision_frames=int(counts[counts > 1].sum()),
        conflicting_groups=sum(r['minimum_truth'] != r['maximum_truth'] for r in class_reports),
        total_weight=float(w.sum()), per_design=per_design,
        largest_contributing_classes=sorted(class_reports, key=lambda r: r['contribution'], reverse=True)[:8])


def model_inputs(features, observed, age, eligible, mean, scale, active):
    content = (features - mean) / scale
    content[..., ~active] = 0
    content[~(observed | eligible)] = 0
    current = np.concatenate((content, observed[..., None], age[..., None]), axis=-1).astype(np.float32)
    return make_slots(current).reshape(*observed.shape, -1)


def history_mask(mask):
    output = mask.copy()
    output[:, 2:] |= mask[:, :-2]
    output[:, 4:] |= mask[:, :-4]
    return output


def contribution(point, truth, weights, affected, designs):
    error = abs(point - truth[None])
    total = (error * weights[None]).sum(axis=(1, 2))
    changed = (error * (weights * affected)[None]).sum(axis=(1, 2))
    unchanged = (error * (weights * ~affected)[None]).sum(axis=(1, 2))
    np.testing.assert_allclose(total, changed + unchanged, rtol=0, atol=1e-12)
    per_design = {}
    for design in np.unique(designs):
        selected = (designs == design)[:, None]
        if (weights * selected).any():
            per_design[design] = dict(total_contribution_by_seed=(error * (weights * selected)[None]).sum(axis=(1, 2)).tolist(),
                changed_contribution_by_seed=(error * (weights * selected * affected)[None]).sum(axis=(1, 2)).tolist(),
                unchanged_contribution_by_seed=(error * (weights * selected * ~affected)[None]).sum(axis=(1, 2)).tolist())
    return dict(mae_by_seed=total.tolist(), changed_error_contribution_by_seed=changed.tolist(),
        unchanged_error_contribution_by_seed=unchanged.tolist(), perfect_affected_correction_mae_by_seed=unchanged.tolist(),
        mae=float(total.mean()), changed_error_contribution=float(changed.mean()),
        unchanged_error_contribution=float(unchanged.mean()), per_design=per_design,
        limitation='Perfect affected-history correction leaves old unaffected predictions fixed; NOT a bound on a globally refitted model')


def analyze():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for relative, digest in protocol['inputs'].items():
        assert sha(ROOT / relative) == digest, f'Pinned source changed: {relative}'
    p = json.loads((BASELINE / 'protocol.json').read_text())
    artifact_hashes = {}
    # Eligibility artifacts are fixed before labels are loaded for scoring.
    replay_by_design = {}
    for path in sorted(OUT.glob('replay_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            assert str(z['protocol_sha256']) == sha(OUT / 'protocol.json')
            replay_by_design[path.stem.removeprefix('replay_')] = {k: z[k].copy() for k in z.files}
        artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
    assert len(replay_by_design) == 20, 'Complete original20-design inventory required'
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        records = json.loads(str(z['records']))
        original, observed, age, truth = [z[k].copy() for k in ('features', 'observed', 'age', 'truth')]
    designs = np.array([r['design'] for r in records])
    noises = np.array([r['noise'] for r in records])
    assert set(designs) == set(replay_by_design)
    eligible, finite_only = np.zeros(observed.shape, bool), np.zeros(observed.shape, bool)
    retained = original.copy()
    rebuilt = np.zeros_like(original)
    reasons = np.zeros(observed.shape, np.int8)
    all_indices = []
    inventory = []
    for design, replay in replay_by_design.items():
        indices = replay['row_indices']
        all_indices.extend(indices.tolist())
        assert np.array_equal(designs[indices], np.repeat(design, len(indices)))
        assert json.loads(str(replay['records'])) == [records[i] for i in indices]
        e = replay['eligible']
        assert not (e & observed[indices]).any()
        assert np.isfinite(replay['retained_features']).all()
        assert np.array_equal(replay['retained_features'][observed[indices]].view(np.uint8), original[indices][observed[indices]].view(np.uint8))
        eligible[indices], finite_only[indices] = e, replay['finite_only_eligible']
        retained[indices], rebuilt[indices], reasons[indices] = replay['retained_features'], replay['replay_features'], replay['reason']
        receipt_path = OUT / f'replay_{design}.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['replay_sha256'] == sha(OUT / f'replay_{design}.npz')
        inventory.extend(receipt['records'])
    assert sorted(all_indices) == list(range(len(records)))
    assert np.array_equal(retained[~observed & ~eligible], original[~observed & ~eligible])
    sensitivity = original.copy()
    sensitivity[finite_only] = retained[finite_only]
    old_scores = json.loads((BASELINE / 'results.json').read_text())['metrics']
    reports, reproduction, aggregate = {}, {}, {}
    for fold, excluded in enumerate(p['evaluation_folds']):
        baseline_path = BASELINE / f'fold{fold}_content_gru_seed42.npz'
        with np.load(baseline_path, allow_pickle=False) as z:
            mean, scale, active = [z[k].copy() for k in ('mean', 'scale', 'active')]
        for seed in SEEDS[1:]:
            with np.load(BASELINE / f'fold{fold}_content_gru_seed{seed}.npz', allow_pickle=False) as z:
                for key, array in (('mean', mean), ('scale', scale), ('active', active)):
                    assert np.array_equal(z[key], array), 'Seed-dependent scaling requires separate equality accounting'
        successful = observed & (reasons == 0)
        difference = rebuilt[successful] - original[successful]
        standardized = difference / scale
        standardized[:, ~active] = 0
        original32 = ((original[successful] - mean) / scale).astype(np.float32)
        replay32 = ((rebuilt[successful] - mean) / scale).astype(np.float32)
        original32[:, ~active] = 0
        replay32[:, ~active] = 0
        count_mismatches = ordering_mismatches = availability_mismatches = 0
        max_mode_difference = {'alpha': 0.0, 'frequency': 0.0, 'energy': 0.0}
        for design, replay in replay_by_design.items():
            obs = observed[replay['row_indices']] & (replay['reason'] == 0)
            count_mismatches += int((replay['mode_count'][obs] != replay['reference_mode_count'][obs]).sum())
            for key in max_mode_difference:
                a, b = replay[f'replay_{key}'][obs], replay[f'reference_{key}'][obs]
                finite = np.isfinite(a) & np.isfinite(b)
                if finite.any():
                    max_mode_difference[key] = max(max_mode_difference[key], float(abs(a[finite] - b[finite]).max()))
            # Coordinate order is compared directly: no re-sorting to hide a slot switch.
            a, b = replay['replay_frequency'][obs], replay['reference_frequency'][obs]
            same_availability = (np.isfinite(a) == np.isfinite(b)).all(axis=1)
            availability_mismatches += int((~same_availability).sum())
            aa, bb = np.where(np.isfinite(a), a, np.inf), np.where(np.isfinite(b), b, np.inf)
            ordering_mismatches += int((same_availability & (np.argsort(aa, axis=1, kind='stable')
                != np.argsort(bb, axis=1, kind='stable')).any(axis=1)).sum())
        reproduction[str(fold)] = dict(accepted=int(observed.sum()), replay_successful=int(successful.sum()),
            replay_failures=int((observed & ~successful).sum()),
            maximum_absolute_difference_by_coordinate=abs(difference).max(axis=0).tolist(),
            maximum_standardized_difference_by_coordinate=abs(standardized).max(axis=0).tolist(),
            differing_float32_coordinates=int((original32.view(np.uint32) != replay32.view(np.uint32)).sum()),
            differing_float32_rows=int((original32.view(np.uint32) != replay32.view(np.uint32)).any(axis=1).sum()),
            mode_count_mismatches=count_mismatches, frequency_order_mismatches=ordering_mismatches,
            maximum_ordered_mode_coordinate_difference=max_mode_difference,
            mode_availability_mismatches=availability_mismatches,
            cached_feature_mode_count_mismatches=int((rebuilt[successful, 18] != original[successful, 18]).sum()),
            cached_feature_availability_mismatches=int((rebuilt[successful, 21] != original[successful, 21]).sum()),
            cached_first_three_frequency_order_mismatches=int((np.argsort(rebuilt[successful][:, [8, 11, 14]], axis=1)
                != np.argsort(original[successful][:, [8, 11, 14]], axis=1)).any(axis=1).sum()),
            implementation_difference='Canonical base views used per-frame noise-group batching in fitter_comparison_acquire._process; current replay uses per-noise-row batching like support_modal_v2. Same CUDA estimator and measured samples; cache is never replaced.',
            accepted_features_preserved=True, active_feature_indices=np.flatnonzero(active).tolist())
        fold_rows = np.flatnonzero(np.isin(designs, excluded))
        d, n, y, o, a = designs[fold_rows], noises[fold_rows], truth[fold_rows], observed[fold_rows], age[fold_rows]
        q0 = model_inputs(original[fold_rows], o, a, np.zeros_like(o), mean, scale, active)
        comparisons = {'combined': (retained[fold_rows], eligible[fold_rows]),
                       'finite_only_sensitivity': (sensitivity[fold_rows], finite_only[fold_rows])}
        points = {}
        for arm in ARMS:
            paths = [((SOURCE / f'fold{fold}_measured_seed{seed}.npz') if arm == 'framewise_reference'
                      else (BASELINE / f'fold{fold}_{arm}_seed{seed}.npz')) for seed in SEEDS]
            arrays = []
            for path in paths:
                artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
                with np.load(path, allow_pickle=False) as z:
                    arrays.append(z['predictions'][fold_rows].copy())
            points[arm] = np.stack(arrays)
        for variant, (q, e) in comparisons.items():
            inputs = model_inputs(q, o, a, e, mean, scale, active)
            affected = (inputs.view(np.uint32) != q0.view(np.uint32)).any(axis=-1)
            any_eligible = history_mask(e)
            assert not (affected & ~any_eligible).any()
            for noise in sorted(set(n)):
                population = (n == noise)[:, None]
                for stratum, mask in (('all', np.ones_like(o)), ('boundary', (y > -.02) & (y < .05))):
                    selected = mask & population
                    w = original_population_weights(selected, d)
                    floor = weighted_collision_floor(inputs, y, w, d)
                    floor0 = weighted_collision_floor(q0, y, w, d)
                    target = (.010 if noise == 'clean' else .015) if stratum == 'all' else (.005 if noise == 'clean' else .010)
                    is_primary = noise in protocol['primary_noise']
                    decomposition = {arm: contribution(point, y, w, affected, d) for arm, point in points.items()}
                    for arm, decomposed in decomposition.items():
                        expected = old_scores[f'fold{fold}/evaluation/{noise}'][arm][stratum]['equal_design_mae_by_seed']
                        np.testing.assert_allclose(decomposed['mae_by_seed'], expected, rtol=0, atol=1e-8)
                    key = f'fold{fold}/{variant}/{noise}/{stratum}'
                    reports[key] = dict(fold=fold, variant=variant, noise=noise, stratum=stratum,
                        evaluated_designs=excluded, original_frames=int(selected.sum()),
                        original_weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                        eligible_history_frames=int((selected & any_eligible).sum()),
                        changed_input_history_frames=int((selected & affected).sum()),
                        eligible_history_weight=float(w[any_eligible].sum()), affected_history_weight=float(w[affected].sum()),
                        remaining_no_permitted_content_history_frames=int((selected & ~history_mask(o | e)).sum()),
                        remaining_no_permitted_content_history_weight=float(w[~history_mask(o | e)].sum()),
                        prohibited_current_frames=int((selected & np.isin(reasons[fold_rows], (1, 2, 3))).sum()),
                        full_no_mode_current_frames=int((selected & (reasons[fold_rows] == 5)).sum()),
                        full_invalid_current_frames=int((selected & (reasons[fold_rows] == 6)).sum()),
                        remaining_no_content_restricted_collision_floor=weighted_collision_floor(inputs, y, w * ~history_mask(o | e), d),
                        exact_collision_floor=floor, original_zeroing_collision_floor=floor0,
                        primary_target=target if is_primary else None,
                        collision_forecloses_target=bool(floor['floor'] > target) if is_primary else None,
                        comparator_error_contributions=decomposition)
                    aggregate.setdefault(f'{variant}/{noise}/{stratum}', []).append(reports[key])
        print('GATE_ANALYSIS_FOLD', fold, flush=True)
    aggregate_report = {}
    for key, rows in aggregate.items():
        assert len(rows) == 3
        result = dict(evaluation_designs=12, original_frames=sum(r['original_frames'] for r in rows),
            floor=float(np.mean([r['exact_collision_floor']['floor'] for r in rows])),
            zeroing_floor=float(np.mean([r['original_zeroing_collision_floor']['floor'] for r in rows])),
            affected_weight=float(np.mean([r['affected_history_weight'] for r in rows])),
            eligible_history_weight=float(np.mean([r['eligible_history_weight'] for r in rows])),
            primary_target=rows[0]['primary_target'], comparators={})
        result['collision_forecloses_target'] = result['floor'] > result['primary_target'] if result['primary_target'] is not None else None
        for arm in ARMS:
            result['comparators'][arm] = {field: np.mean([r['comparator_error_contributions'][arm][field] for r in rows], axis=0).tolist()
                for field in ('mae_by_seed', 'changed_error_contribution_by_seed', 'unchanged_error_contribution_by_seed')}
        aggregate_report[key] = result
    total_reason = {str(code): int((reasons == code).sum()) for code in range(11)}
    receipt = dict(scope='Exact original12-design grouped evaluation;20-design extraction inventory kept separate',
        protocol_sha256=sha(OUT / 'protocol.json'), analysis_source_sha256=sha(__file__),
        artifact_hashes=artifact_hashes, inventory_designs=20, inventory_rows=len(records),
        inventory_frames=int(observed.size), inventory_cached_accepted=int(observed.sum()),
        inventory_eligible=int(eligible.sum()), inventory_finite_only_eligible=int(finite_only.sum()),
        inventory_reasons=total_reason, inventory_by_record=inventory,
        reproduction=reproduction, fold_reports=reports, aggregate=aggregate_report,
        limits=['Exact collision bounds condition on fold-specific float32 permitted full inputs incl flags/age/padding; no rounding',
                'Weights are original equal-design all/boundary denominators, not renormalized on affected/missing histories',
                'Perfect changed-history correction is diagnostic leverage, not a global-refit accuracy lower bound',
                'Auxiliary extraction runtime included; qualification unchanged; no fits or endpoint credit'])
    save_json(OUT / 'analysis.json', receipt)
    print('GATE_RECEIPT_ANALYZED', len(reports), receipt['inventory_eligible'], flush=True)
    return receipt


if __name__ == '__main__':
    analyze()
