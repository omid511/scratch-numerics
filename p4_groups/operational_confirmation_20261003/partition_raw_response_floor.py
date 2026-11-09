"""Partition frozen modal collisions by raw eligibility; never fits or extracts modes."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
import analyze_gate_bypass_receipt as A
ROOT, SOURCE, BASELINE = A.ROOT, A.SOURCE, A.BASELINE
OUT = A.OUT / 'raw_response_partition'
CATEGORIES = ('full128', 'shorter_only', 'no_samples')


def trailing_any(mask, lags):
    result = np.zeros_like(mask)
    for lag in lags:
        if lag:
            result[:, lag:] |= mask[:, :-lag]
        else:
            result |= mask
    return result


def class_contributions(x, y, w):
    """Full-population class medians; attribution never renormalizes subsets."""
    result = np.zeros_like(y, dtype=float)
    selected = w > 0
    keys = np.ascontiguousarray(x[selected]).view(np.dtype((np.void, x.shape[-1] * x.dtype.itemsize))).ravel()
    _, groups = np.unique(keys, return_inverse=True)
    values, weights = y[selected], w[selected]
    assigned = np.zeros(len(values))
    order = np.argsort(groups, kind='stable')
    splits = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1, len(order)]
    for start, end in zip(splits[:-1], splits[1:]):
        members = order[start:end]
        value_order = np.argsort(values[members], kind='stable')
        sorted_members = members[value_order]
        index = np.searchsorted(np.cumsum(weights[sorted_members]), weights[members].sum() / 2, side='left')
        median = values[sorted_members[index]]
        assigned[members] = weights[members] * abs(values[members] - median)
    result[selected] = assigned
    return result


def mask_information(packed, sample_count, ends, policy):
    """Known causal sample-presence/command/finite/cap/admissibility and timebase."""
    unpacked = {name: np.unpackbits(values, axis=1, count=sample_count, bitorder='little').astype(bool)
                for name, values in packed.items()}
    rows, frames = len(unpacked['cap']), len(ends)
    # Five separate 384-bit masks; absent warmup is explicitly not finite/present.
    information = np.zeros((rows, frames, 5 * 48 + 4), np.uint8)
    for row in range(rows):
        command, finite, cap = [unpacked[name][row] for name in ('command', 'finite', 'cap')]
        valid = ~command & finite & ~cap
        if policy == 'precap_sensitivity':
            first_cap = np.flatnonzero(cap)
            if len(first_cap):
                valid[first_cap[0]:] = False
        for frame, end in enumerate(ends):
            start = max(0, int(end) - 384)
            count = int(end) - start
            for index, values in enumerate((np.ones(sample_count, bool), command, finite, cap, valid)):
                span = np.zeros(384, bool)
                span[384-count:] = values[start:end]
                information[row, frame, index*48:(index+1)*48] = np.packbits(span, bitorder='little')
            information[row, frame, -4:] = np.array([end], dtype='<i4').view(np.uint8)
    return information


def subset_report(mask, w, y, designs, profiles, points, attributed, inputs, extended):
    positive = mask & (w > 0)
    ranges = {}
    for profile in np.unique(profiles):
        take = positive & (profiles == profile)[:, None]
        if take.any():
            ranges[str(profile)] = dict(frames=int(take.sum()), weight=float(w[take].sum()),
                margin_min=float(y[take].min()), margin_max=float(y[take].max()),
                modal_collision_contribution=float(attributed[take].sum()))
    old_errors = {arm: dict(error_contribution_by_seed=(abs(point-y[None]) * (w*mask)[None]).sum(axis=(1, 2)).tolist())
                  for arm, point in points.items()}
    for value in old_errors.values():
        value['mean_error_contribution'] = float(np.mean(value['error_contribution_by_seed']))
    return dict(frames=int(positive.sum()), weight=float(w[positive].sum()),
        margin_range=[float(y[positive].min()), float(y[positive].max())] if positive.any() else None,
        profiles=ranges, modal_collision_contribution=float(attributed[positive].sum()),
        restricted_modal_bound=A.weighted_collision_floor(inputs, y, w*mask, designs),
        mask_aware_restricted_bound=A.weighted_collision_floor(extended, y, w*mask, designs),
        comparator_error_contributions=old_errors)


def smoke():
    raw = np.zeros((1, 6), bool)
    raw[0, 1] = True
    assert not trailing_any(raw, (4, 2, 0))[0, 4]
    assert trailing_any(raw, (4, 3, 2, 1, 0))[0, 4]
    assert not trailing_any(raw, (4, 3, 2, 1, 0))[0, 0]
    x = np.array([[[0.], [0.], [1.]]], dtype=np.float32)
    y = np.array([[0., 1., 9.]])
    w = np.array([[.25, .125, .625]])
    attributed = class_contributions(x, y, w)
    assert np.array_equal(attributed, [[0., .125, 0.]])
    # Mask distinction removes an old equality without inventing waveform values.
    metadata = np.array([[[0], [1], [0]]], np.uint8)
    extended = np.concatenate((x.view(np.uint8).reshape(1, 3, 4), metadata), axis=-1)
    assert A.weighted_collision_floor(extended, y, w)['floor'] == 0
    print('RAW_RESPONSE_PARTITION_SMOKE_PASSED', flush=True)


def partition():
    prior = json.loads((A.OUT / 'analysis.json').read_text())
    protocol = json.loads((BASELINE / 'protocol.json').read_text())
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        records = json.loads(str(z['records']))
        truth, observed, age = [z[name].copy() for name in ('truth', 'observed', 'age')]
        times = z['times'].copy()
    designs = np.array([r['design'] for r in records])
    noises = np.array([r['noise'] for r in records])
    profiles = np.array([r['profile'] for r in records])
    ends = np.rint(times*1000).astype(int) + 1
    reports, artifacts = {}, {}
    for fold, evaluated in enumerate(protocol['evaluation_folds']):
        fold_rows = np.flatnonzero(np.isin(designs, evaluated))
        local_index = {int(global_index): i for i, global_index in enumerate(fold_rows)}
        shape = (len(fold_rows), len(ends))
        retained = np.empty((*shape, 174), dtype=float)
        eligible, reason = np.zeros(shape, bool), np.empty(shape, np.int8)
        counts = np.empty((len(fold_rows), 2, len(ends)), np.int16)
        lengths = np.empty_like(counts)
        packed = {}
        for design in evaluated:
            replay_path, mask_path = A.OUT / f'replay_{design}.npz', OUT / f'masks_{design}.npz'
            artifacts[str(mask_path.relative_to(ROOT))] = A.sha(mask_path)
            with np.load(replay_path, allow_pickle=False) as z:
                indices = z['row_indices'].copy()
                destinations = np.array([local_index[int(index)] for index in indices])
                retained[destinations] = z['retained_features']
                eligible[destinations], reason[destinations] = z['eligible'], z['reason']
            with np.load(mask_path, allow_pickle=False) as z:
                assert np.array_equal(z['row_indices'], indices)
                assert json.loads(str(z['records'])) == [records[index] for index in indices]
                assert np.array_equal(z['ends'], ends)
                count = int(z['sample_count'])
                counts[destinations], lengths[destinations] = z['sample_counts'], z['longest_runs']
                for name in ('command', 'finite', 'cap'):
                    if name not in packed:
                        packed[name] = np.empty((len(fold_rows), z[name].shape[1]), np.uint8)
                    packed[name][destinations] = z[name]
        d, n, p = designs[fold_rows], noises[fold_rows], profiles[fold_rows]
        y, o, a = truth[fold_rows], observed[fold_rows], age[fold_rows]
        with np.load(BASELINE / f'fold{fold}_content_gru_seed42.npz', allow_pickle=False) as z:
            mean, scale, active = [z[name].copy() for name in ('mean', 'scale', 'active')]
        inputs = A.model_inputs(retained, o, a, eligible, mean, scale, active)
        input_bytes = inputs.view(np.uint8).reshape(*shape, -1)
        no_modal = ~A.history_mask(o | eligible)
        raw_full = ~np.isin(reason, (1, 2, 3))
        availability = {'three_slots': trailing_any(raw_full, (4, 2, 0)),
                        'five_windows': trailing_any(raw_full, (4, 3, 2, 1, 0))}
        points = {}
        for arm in A.ARMS:
            paths = [SOURCE / f'fold{fold}_measured_seed{seed}.npz' if arm == 'framewise_reference'
                     else BASELINE / f'fold{fold}_{arm}_seed{seed}.npz' for seed in A.SEEDS]
            arrays = []
            for path in paths:
                with np.load(path, allow_pickle=False) as z:
                    arrays.append(z['predictions'][fold_rows].copy())
            points[arm] = np.stack(arrays)
        for policy_index, policy in enumerate(('native_sample_rule', 'precap_sensitivity')):
            information = mask_information(packed, count, ends, policy)
            extended = np.concatenate((input_bytes, information), axis=-1)
            category_masks = {'full128': lengths[:, policy_index] >= 128,
                              'shorter_only': (lengths[:, policy_index] > 0) & (lengths[:, policy_index] < 128),
                              'no_samples': lengths[:, policy_index] == 0}
            assert np.array_equal(sum(mask.astype(int) for mask in category_masks.values()), np.ones(shape, int))
            for noise in sorted(set(n)):
                for stratum in ('all', 'boundary'):
                    population = (n == noise)[:, None] & (np.ones(shape, bool) if stratum == 'all' else ((y > -.02) & (y < .05)))
                    w = A.original_population_weights(population, d)
                    attributed = class_contributions(inputs, y, w)
                    old = prior['fold_reports'][f'fold{fold}/combined/{noise}/{stratum}']
                    bins = {name: subset_report(mask, w, y, d, p, points, attributed, inputs, extended)
                            for name, mask in category_masks.items()}
                    modal_absent = {name: subset_report(mask & no_modal, w, y, d, p, points, attributed, inputs, extended)
                                    for name, mask in category_masks.items()}
                    full_windows = {name: {'without_window': subset_report(~mask, w, y, d, p, points, attributed, inputs, extended),
                                          'window_but_no_modal': subset_report(mask & no_modal, w, y, d, p, points, attributed, inputs, extended)}
                                    for name, mask in availability.items()}
                    # New partition invariants against existing, unrecomputed endpoint receipt.
                    assert abs(sum(v['weight'] for v in bins.values()) - 1) < 1e-12
                    assert abs(sum(v['modal_collision_contribution'] for v in bins.values()) - old['exact_collision_floor']['floor']) < 1e-12
                    for arm in A.ARMS:
                        total = np.sum([v['comparator_error_contributions'][arm]['error_contribution_by_seed'] for v in bins.values()], axis=0)
                        np.testing.assert_allclose(total, old['comparator_error_contributions'][arm]['mae_by_seed'], rtol=0, atol=1e-12)
                    key = f'fold{fold}/{policy}/{noise}/{stratum}'
                    reports[key] = dict(fold=fold, policy=policy, noise=noise, stratum=stratum,
                        original_frames=int(population.sum()), original_modal_floor=old['exact_collision_floor']['floor'],
                        target=old['primary_target'], categories=bins, modal_absent_categories=modal_absent,
                        aligned_full_windows=full_windows)
            print('RAW_PARTITION_FOLD_POLICY_COMPLETE', fold, policy, flush=True)
    aggregates = {}
    for key, report in reports.items():
        aggregate_key = '/'.join(key.split('/')[1:])
        aggregates.setdefault(aggregate_key, []).append(report)
    for key, rows in aggregates.items():
        aggregates[key] = dict(original_frames=sum(r['original_frames'] for r in rows), original_modal_floor=float(np.mean([r['original_modal_floor'] for r in rows])),
            target=rows[0]['target'], categories={name: {field: float(np.mean([r['categories'][name][field] for r in rows])) for field in ('weight', 'modal_collision_contribution')} | {
                'restricted_modal_floor': float(np.mean([r['categories'][name]['restricted_modal_bound']['floor'] or 0 for r in rows])),
                'mask_aware_restricted_floor': float(np.mean([r['categories'][name]['mask_aware_restricted_bound']['floor'] or 0 for r in rows])),
                'old_content_gru_error': float(np.mean([r['categories'][name]['comparator_error_contributions']['content_gru']['mean_error_contribution'] for r in rows]))}
                for name in CATEGORIES})
    result = dict(scope='Original12-design grouped populations; existing measurements only; no fits', reports=reports, aggregate=aggregates,
        source_analysis_sha256=A.sha(A.OUT / 'analysis.json'), mask_artifact_sha256=artifacts, partition_source_sha256=A.sha(__file__),
        progress_ledger_sha256=A.sha(ROOT / 'P4_PROGRESS.json'),
        equality='Original fold-normalized float32 three-slot174 modal values plus unchanged qualification flags and float32 ages, original warmup padding. Mask-aware keys append five384-bit causal masks (presence,command,finite,cap,admissibility) plus exclusive current sample end. No design/profile/target identifiers or waveform values inserted.',
        exclusions='Native sample handling preserves source command/nonfinite/cap exclusions. Precap sensitivity also discards all samples after first trajectory cap, a conservative hypothetical restriction NOT a source policy change. No capped value or proposed post-cap predictor input is used.',
        limits=['Category contributions use FULL original modal-class medians; restricted bounds reoptimize only subset members but keep ORIGINAL denominator weights.',
                'Restricted modal bounds are not immutable raw-information limits; mask-aware bounds condition on specified metadata only, not every permitted causal predictor.',
                'Offsets/scales from leading calibration are preserved in mask artifacts but NOT input coordinates in these bounds; adding them can change equality classes.',
                'Contiguous128 snippets may have noncanonical alignment; availability is not estimability or qualified availability.',
                'No new acquisition, modal extraction, prediction, fitting, qualification relaxation or endpoint credit.'])
    A.save_json(OUT / 'partition.json', result)
    print('RAW_RESPONSE_PARTITION_COMPLETE', len(reports), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    smoke() if args.smoke else partition()
