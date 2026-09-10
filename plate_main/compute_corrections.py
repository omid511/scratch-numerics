#!/usr/bin/env python3
"""Build audited, mode-matched P1 targets from versioned LF/HF run bundles.

Frequency and field differences share the same verified pairing. Rejected modes
never become zero-filled labels; each output revision is immutable.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from p1_pairing import (PARAMETERS, load_run, check_pair, assign, align_real,
                        normalized, boundary_mask, repeated, assignment_margin)

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_reference_run(runs, excluded=()):
    """Choose the normalized-parameter medoid from the available runs."""
    excluded = set(excluded)
    candidates = sorted(set(runs) - excluded)
    if not candidates:
        raise ValueError('No runs remain for reference selection')
    values = np.asarray([
        [float(runs[run]['meta']['parameters'][parameter]) for parameter in PARAMETERS]
        for run in candidates
    ])
    if not np.isfinite(values).all():
        raise ValueError('Reference parameters must be finite')
    spans = np.ptp(values, axis=0)
    scaled = (values - values.min(axis=0)) / np.where(spans > 0, spans, 1)
    distances = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=2))
    return candidates[int(np.argmin(distances.sum(axis=1)))]


def _reference_shapes(lf, reference_run, modes):
    reference = lf[reference_run]
    if len(reference['f']) <= modes:
        raise ValueError('Export guard modes above target count to detect a cutoff degeneracy')
    indices = np.argsort(reference['f'], kind='stable')[:modes]
    shapes = np.stack([align_real(reference['w'][index]) for index in indices])
    return reference, indices, shapes

def _reference_pool(lf, modes):
    """Prepare per-mode references and stability scores once."""
    run_ids = sorted(lf)
    prepared = {}
    for run_id in run_ids:
        reference, indices, shapes = _reference_shapes(lf, run_id, modes)
        prepared[run_id] = dict(reference=reference, indices=indices,
                                 shapes=shapes)
    # Compare each sorted LF mode against the same sorted mode in every run.
    # This gives a deterministic stability score without allowing a
    # near-degenerate reference pair to define the mode label.
    stability = {}
    for mode in range(modes):
        candidates = np.stack([
            normalized(lf[run_id]['w'][prepared[run_id]['indices'][mode]][None])[0]
            for run_id in run_ids
        ]).reshape(len(run_ids), -1)
        targets = candidates.copy()
        stability[mode] = np.clip(np.abs(candidates.conj() @ targets.T) ** 2,
                                  0, 1)
    return dict(run_ids=run_ids, prepared=prepared, stability=stability)


def select_reference_modes(lf, modes, degeneracy_gap, reference_run=None,
                           excluded=(), pool=None):
    """Select a coherent stable reference, then fall back per mode."""
    excluded = set(excluded)
    available = [run_id for run_id in sorted(lf) if run_id not in excluded]
    if not available:
        raise ValueError('No runs remain for per-mode reference selection')
    if reference_run is not None and reference_run not in available:
        raise ValueError(f'Reference run {reference_run} unavailable')
    pool = _reference_pool(lf, modes) if pool is None else pool
    position = {run_id: index for index, run_id in enumerate(pool['run_ids'])}
    available_positions = [position[run_id] for run_id in available]

    def stable_modes(run_id):
        item = pool['prepared'][run_id]
        return [
            mode for mode in range(modes)
            if not repeated(item['reference']['f'], item['indices'][mode],
                            degeneracy_gap)
        ]

    if reference_run is not None:
        selected_by_mode = [reference_run] * modes
        selection_mode = 'explicit'
    else:
        coherent = [
            run_id for run_id in available
            if len(stable_modes(run_id)) == modes
        ]
        if coherent:
            scored = []
            for run_id in coherent:
                values = np.concatenate([
                    pool['stability'][mode][position[run_id],
                                             available_positions]
                    for mode in range(modes)
                ])
                scored.append((
                    round(float(np.quantile(values, .05)), 12),
                    round(float(np.median(values)), 12),
                    -run_id,
                    run_id,
                ))
            selected_by_mode = [max(scored)[-1]] * modes
            selection_mode = 'global-coherent'
        else:
            selected_by_mode = [None] * modes
            selection_mode = 'per-mode-fallback'
            for mode in range(modes):
                stable = [
                    run_id for run_id in available
                    if mode in stable_modes(run_id)
                ]
                candidates = stable or available
                scores = pool['stability'][mode]
                scored = []
                for run_id in candidates:
                    row = scores[position[run_id], available_positions]
                    scored.append((
                        round(float(np.quantile(row, .05)), 12),
                        round(float(np.median(row)), 12),
                        -run_id,
                        run_id,
                    ))
                selected_by_mode[mode] = max(scored)[-1]

    reference_ids, reference_indices, shapes, quality = [], [], [], []
    for mode, selected in enumerate(selected_by_mode):
        item = pool['prepared'][selected]
        selected_scores = pool['stability'][mode][
            position[selected], available_positions]
        reference_ids.append(selected)
        reference_indices.append(int(item['indices'][mode]))
        shapes.append(item['shapes'][mode])
        quality.append(dict(
            mode=mode + 1, reference_run=selected,
            reference_lf_mode=int(item['indices'][mode]) + 1,
            stable=mode in stable_modes(selected),
            p05_mac=float(np.quantile(selected_scores, .05)),
            median_mac=float(np.median(selected_scores)),
        ))
    return dict(reference_ids=np.array(reference_ids, dtype=int),
                reference_indices=np.array(reference_indices, dtype=int),
                shapes=np.stack(shapes), quality=quality,
                selection_mode=selection_mode)


def _tracking_stats(values, min_mac):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, below_floor=0)
    quantiles = np.quantile(values, [.05, .5, .95])
    return dict(count=int(len(values)), below_floor=int(np.sum(values < min_mac)),
                minimum=float(values.min()), p05=float(quantiles[0]),
                median=float(quantiles[1]), p95=float(quantiles[2]),
                maximum=float(values.max()))


def leave_p_out_tracking(lf, modes, p, min_mac, degeneracy_gap, pool=None):
    """Audit LF tracking with deterministic run-level leave-p-out folds."""
    if not isinstance(p, int) or p < 1:
        raise ValueError('leave_p_out must be a positive integer')
    run_ids = sorted(lf)
    if len(run_ids) < 2:
        return dict(p=p, folds=0, held_out_runs=0, stats=_tracking_stats([], min_mac))
    pool = _reference_pool(lf, modes) if pool is None else pool
    rows, values, reference_ids = [], [], []
    for fold, start in enumerate(range(0, len(run_ids), p), 1):
        held_out = run_ids[start:start + p]
        references = select_reference_modes(
            lf, modes, degeneracy_gap, excluded=held_out, pool=pool)
        for run_id in held_out:
            mapping, scores = assign(references['shapes'], lf[run_id]['w'])
            for mode in range(modes):
                score = float(scores[mode, mapping[mode]])
                values.append(score)
                rows.append(dict(
                    fold=fold, run=run_id, mode=mode + 1,
                    reference_run=int(references['reference_ids'][mode]),
                    tracking_mac=score))
            reference_ids.extend(references['reference_ids'].tolist())
    return dict(p=p, folds=len(set(row['fold'] for row in rows)),
                held_out_runs=len(set(row['run'] for row in rows)),
                reference_runs=sorted(set(reference_ids)),
                stats=_tracking_stats(values, min_mac))


def _portable_key(path, root):
    return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def build_targets(root, name='corrections_v5', modes=10, reference_run=None,
                  min_mac=.8, min_margin=.05, degeneracy_gap=.005,
                  min_hf_transverse_fraction=.5, leave_p_out=1):
    root = Path(root).resolve()
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Output name must be a single directory name')
    if (modes < 1 or not 0 <= min_mac <= 1 or not 0 <= min_margin <= 1
            or not 0 <= degeneracy_gap < 1
            or (min_hf_transverse_fraction is not None
                and not 0 <= min_hf_transverse_fraction <= 1)):
        raise ValueError('Invalid mode count or quality thresholds')
    if not isinstance(leave_p_out, int) or leave_p_out < 1:
        raise ValueError('leave_p_out must be a positive integer')
    destination = root / name
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; use --name for a new dataset revision')
    lpaths, hpaths = sorted((root / 'lf').glob('run_*.npz')), sorted((root / 'hf').glob('run_*.npz'))
    if not lpaths or not hpaths:
        raise ValueError('Need version-2 NPZ runs in both lf/ and hf/')
    lf, hf = {}, {}
    for paths, target, fidelity in ((lpaths, lf, 'lf'), (hpaths, hf, 'hf')):
        for path in paths:
            run = load_run(path, fidelity)
            run_id = run['meta']['run_id']
            if run_id in target:
                raise ValueError(f'Duplicate {fidelity} run ID {run_id}')
            target[run_id] = run
    if lf.keys() != hf.keys():
        raise ValueError(f'Run sets differ: LF-only={sorted(lf.keys() - hf.keys())}; '
                         f'HF-only={sorted(hf.keys() - lf.keys())}')
    for run_id in lf:
        check_pair(lf[run_id], hf[run_id])
    pool = _reference_pool(lf, modes)
    reference_selection = select_reference_modes(
        lf, modes, degeneracy_gap, reference_run=reference_run, pool=pool)
    selected_reference_ids = reference_selection['reference_ids']
    if reference_run is not None:
        summary_ref_id = reference_run
    elif len(set(selected_reference_ids.tolist())) == 1:
        summary_ref_id = int(selected_reference_ids[0])
    else:
        summary_ref_id = select_reference_run(lf)
    if summary_ref_id not in lf:
        raise ValueError(f'Reference run {summary_ref_id} unavailable')
    ref = lf[summary_ref_id]
    refs = reference_selection['shapes']
    mask = boundary_mask(ref['x'], ref['y'])
    interior = mask > 0
    tracking_cv = leave_p_out_tracking(
        lf, modes, leave_p_out, min_mac, degeneracy_gap, pool=pool)
    accepted, audit = [], []
    for run_id in sorted(lf):
        low, high = lf[run_id], hf[run_id]
        if any(low[a].shape != ref[a].shape or not np.allclose(low[a], ref[a], rtol=0, atol=1e-12)
               for a in ('x', 'y')):
            raise ValueError('All runs must share the reference spatial grid')
        if min(len(low['f']), len(high['f'])) <= modes:
            raise ValueError(f'Run {run_id}: export extra guard modes for matching')
        low_map, track_scores = assign(refs, low['w'])
        selected_low = low['w'][low_map]
        high_map, pair_scores = assign(selected_low, high['w'])
        for label, (reference_run_id, ri, li) in enumerate(zip(
                reference_selection['reference_ids'],
                reference_selection['reference_indices'], low_map), 1):
            reference_run_id, ri, li = int(reference_run_id), int(ri), int(li)
            reference_data = lf[reference_run_id]
            hi = int(high_map[label - 1])
            hf_fraction = (None if high.get('transverse_fraction') is None
                           else float(high['transverse_fraction'][hi]))
            row = dict(
                run=run_id, mode=label, reference_run=reference_run_id,
                reference_lf_mode=ri + 1, lf_mode=li + 1, hf_mode=hi + 1,
                tracking_mac=float(track_scores[label - 1, li]),
                pair_mac=float(pair_scores[label - 1, hi]),
                hf_w_peak_abs=float(high['w_peak_abs'][hi]),
                hf_w_peak_ratio=float(high['w_peak_ratio'][hi]),
                hf_transverse_fraction=hf_fraction,
                hf_quality_source=('transverse_energy_fraction'
                                   if hf_fraction is not None
                                   else 'raw_w_peak_ratio_fallback'),
                f_fsdt=float(low['f'][li]), f_comsol=float(high['f'][hi]))
            reasons = []
            if row['tracking_mac'] < min_mac or row['pair_mac'] < min_mac:
                reasons.append('low_MAC')
            if (assignment_margin(track_scores, label - 1, li) < min_margin
                    or assignment_margin(pair_scores, label - 1, hi) < min_margin):
                reasons.append('ambiguous_assignment')
            if (repeated(reference_data['f'], ri, degeneracy_gap)
                    or repeated(low['f'], li, degeneracy_gap)
                    or repeated(high['f'], hi, degeneracy_gap)):
                reasons.append('near_repeated_eigenvalue_use_subspace_target')
            if (hf_fraction is not None and min_hf_transverse_fraction is not None
                    and hf_fraction < min_hf_transverse_fraction):
                reasons.append('low_HF_transverse_energy_fraction')
            if not reasons:
                try:
                    lshape = align_real(low['w'][li], refs[label - 1])
                    hshape = align_real(high['w'][hi], lshape)
                except ValueError as exc:
                    reasons.append(str(exc))
            row['status'] = 'quarantined' if reasons else 'accepted'
            row['reason'] = ';'.join(reasons)
            if not reasons:
                correction = hshape - lshape
                row['delta_f'] = row['f_comsol'] - row['f_fsdt']
                row['relative_frequency_error_pct'] = 100 * row['delta_f'] / row['f_comsol']
                row['field_rms'] = float(np.sqrt(np.mean(correction ** 2)))
                row['field_rms_interior'] = float(np.sqrt(np.mean(correction[interior] ** 2)))
                row['field_rms_edge'] = float(np.sqrt(np.mean(correction[~interior] ** 2)))
                accepted.append((row, lshape, hshape, correction, low['meta']['parameters']))
            audit.append(row)
    staging = Path(tempfile.mkdtemp(prefix='.corrections-', dir=root))
    try:
        fields = staging / 'correction_fields'
        fields.mkdir()
        xg, yg = np.meshgrid(ref['x'], ref['y'])
        for row, _, _, delta, _ in accepted:
            np.savetxt(fields / f"correction_run{row['run']:04d}_mode{row['mode']:02d}.csv",
                       np.column_stack((xg.ravel(), yg.ravel(), delta.ravel())),
                       delimiter=',', header='x,y,delta_w', comments='')
        keys = list(dict.fromkeys(key for row in audit for key in row))
        with (staging / 'pairing.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(audit)
        shape = (0, len(ref['y']), len(ref['x']))
        stack = lambda index: np.stack([a[index] for a in accepted]) if accepted else np.empty(shape)
        np.savez_compressed(
            staging / 'training.npz', x=ref['x'], y=ref['y'],
            boundary_mask=mask, interior_mask=interior,
            lf=stack(1), hf=stack(2), correction=stack(3),
            run_ids=np.array([a[0]['run'] for a in accepted], dtype=int),
            mode_ids=np.array([a[0]['mode'] for a in accepted], dtype=int),
            parameters=np.array([[a[4][parameter] for parameter in PARAMETERS]
                                 for a in accepted]).reshape(-1, len(PARAMETERS)),
            parameter_names=np.array(PARAMETERS),
            f_lf=np.array([a[0]['f_fsdt'] for a in accepted]),
            f_hf=np.array([a[0]['f_comsol'] for a in accepted]))
        manifest = dict(
            schema_version=3, model_version='p1-corrections-v5',
            reference_run=summary_ref_id,
            reference_run_by_mode=reference_selection['reference_ids'].tolist(),
            reference_mode_quality=reference_selection['quality'],
            reference_selection=dict(
                strategy=reference_selection['selection_mode'],
                policy=(
                    'prefer one globally coherent non-degenerate LF reference; '
                    'fall back to per-mode stable references; explicit override '
                    'forces one run')),
            requested_modes=modes, min_mac=min_mac,
            min_assignment_margin=min_margin, degeneracy_relative_gap=degeneracy_gap,
            hf_quality_policy=dict(
                raw_w_peak_ratio='diagnostic_only; not an acceptance criterion',
                min_transverse_energy_fraction=min_hf_transverse_fraction,
                missing_fraction='use raw w-peak ratio only as a warning in provenance'),
            accepted=len(accepted), quarantined=len(audit) - len(accepted),
            normalization='per-mode max(abs(w)), aligned phase; Euclidean MAC on common uniform grid',
            labels='per-mode stable LF reference identity; not per-run sorted frequency rank',
            degeneracy_policy=(
                'quarantine individual labels with unresolved near-repeated modes; '
                'use subspace-valued learning for those cases'),
            boundary_policy='prediction is multiplied by boundary_mask; loss and RMSE exclude perimeter',
            leave_p_out_tracking=tracking_cv,
            split_policy='split by run_ids, never by pixel or individual mode',
            source_sha256={_portable_key(path, root): file_hash(path) for path in lpaths + hpaths},
            code_sha256={_portable_key(path, root): file_hash(path)
                         for path in (Path(__file__), Path(__file__).with_name('p1_pairing.py'))})
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        staging.rename(destination)
    except BaseException:
        if staging.parent == root and staging.name.startswith('.corrections-'):
            shutil.rmtree(staging)
        raise
    return manifest, destination
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'p1_data')
    parser.add_argument('--name', default='corrections_v5')
    parser.add_argument('--modes', type=int, default=10)
    parser.add_argument('--reference-run', type=int)
    parser.add_argument('--min-mac', type=float, default=.8)
    parser.add_argument('--min-margin', type=float, default=.05)
    parser.add_argument('--degeneracy-gap', type=float, default=.005)
    parser.add_argument('--min-hf-transverse-fraction', type=float, default=.5,
                        help='Quarantine exported HF modes below this fraction when available')
    parser.add_argument('--leave-p-out', type=int, default=1,
                        help='Run-level tracking audit fold size')
    args = parser.parse_args(argv)
    manifest, directory = build_targets(
        args.output, args.name, args.modes, args.reference_run,
        args.min_mac, args.min_margin, args.degeneracy_gap,
        args.min_hf_transverse_fraction, args.leave_p_out)
    print(f"Accepted {manifest['accepted']}; quarantined {manifest['quarantined']}. Output: {directory}")
    if not manifest['accepted']:
        raise SystemExit('No training targets accepted; inspect pairing.csv. Do not train.')


if __name__ == '__main__':
    main()
