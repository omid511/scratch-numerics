"""Independent arithmetic, waveform-descriptor and saved-network replay checks."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT / 'additive_spectral_20261009'
ROOT = SCRIPT.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tolerance=1e-12):
    if a is None or b is None:
        assert a is b
    else:
        np.testing.assert_allclose(a, b, rtol=0, atol=tolerance)


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    result = json.loads((OUT / 'results.json').read_text())
    training = json.loads((OUT / 'training.json').read_text())
    for category in ('inputs', 'helpers', 'implementation'):
        for path, digest in protocol[category].items():
            assert sha(ROOT / path) == digest
    assert result['protocol_sha256'] == training['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert result['predictions_sha256'] == training['predictions_sha256'] == sha(OUT / 'predictions.npz')
    assert result['dataset_sha256'] == training['dataset_sha256'] == sha(OUT / 'dataset.npz')
    assert result['scoring_source_sha256'] == sha(SCRIPT / 'score_additive_spectral.py')
    with np.load(OUT / 'dataset.npz', allow_pickle=False) as z:
        q, y, observed, age, times, values, valid, bins = (z[name] for name in
            ('features', 'truth', 'observed', 'age', 'times', 'descriptors', 'descriptor_valid', 'bins'))
        records = json.loads(str(z['records']))
        assert str(z['protocol_sha256']) == sha(OUT / 'protocol.json')
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as z:
        predictions = z['predictions']; arms = json.loads(str(z['arms'])); seeds = z['seeds'].tolist()
    assert predictions.shape == (5, 3, 336, 124)
    assert not valid[~observed].any() and np.array_equal(np.isfinite(values), valid)
    assert np.all((values[valid] >= -1e-12) & (values[valid] <= 1 + 1e-12))
    assert len(set((r['design'], r['replicate'], r['profile']) for r in records)) == 48
    proper = np.array([r['split'] == 'train' and r['noise'] in ('clean', 'iid1') for r in records])
    validation = np.array([r['split'] == 'validation' and r['noise'] in ('clean', 'iid1') for r in records])
    assert proper.sum() == 48 and validation.sum() == 24
    for split in ('train', 'validation', 'held'):
        assert set(r['design'] for r in records if r['split'] == split) == set(protocol['split'][split])
    # Every original record remains. Canonical clean/IID1 rows are exact, including gap ages.
    def identity(r):
        return tuple(r[n] for n in ('split', 'design', 'replicate', 'profile', 'noise'))
    lookup = {identity(r): i for i, r in enumerate(records)}
    original_checks = 0
    for path, canonical in ((SCRIPT / 'fitter_margin_comparison_20261008/dataset.npz', False),
                             (SCRIPT / 'temporal_probe_20261007/data.npz', True)):
        with np.load(path, allow_pickle=False) as z:
            old_records = json.loads(str(z['records']))
            for j, record in enumerate(old_records):
                i = lookup[identity(record)]
                for name, array in (('truth', y), ('observed', observed), ('age', age)):
                    np.testing.assert_array_equal(array[i], z[name][j]); original_checks += 1
                np.testing.assert_allclose(q[i], z['features'][j], atol=0 if canonical else 1e-5, rtol=0)
                original_checks += 1
    # Reconstruct measured samples and descriptors without either extraction helper.
    with np.load(SCRIPT / 'fitter_margin_comparison_20261008/raw_signals.npz', allow_pickle=False) as z:
        physical, commands = z['physical'], z['command']
        raw_records = json.loads(str(z['records']))
    raw_lookup = {(r['design'], r['replicate'], r['profile']): i for i, r in enumerate(raw_records)}
    descriptor_checks, maximum_descriptor_error = 0, 0.
    for row, record in enumerate(records):
        raw_index = raw_lookup[(record['design'], record['replicate'], record['profile'])]
        raw = physical[raw_index]
        noise_name, *fresh = record['noise'].split('_fresh')
        draw = int(fresh[0]) if fresh else 20261010
        rng = np.random.default_rng(np.random.SeedSequence([draw, int(record['design'][1:]), record['replicate'],
                    ('cross', 'stable', 'retreat').index(record['profile'])]))
        white = rng.normal(size=raw.shape)
        shared = rng.normal(size=(1, raw.shape[1]))
        corr = np.sqrt(.2) * white + np.sqrt(.8) * shared
        corr *= np.sqrt(np.mean(white ** 2, axis=1, keepdims=True) / np.mean(corr ** 2, axis=1, keepdims=True))
        sigma = np.sqrt(np.mean((raw[:, :51] - raw[:, :51].mean(axis=1, keepdims=True)) ** 2))
        level = .05 if noise_name.endswith('5') else .01
        noise = np.zeros_like(raw) if noise_name == 'clean' else level * sigma * (corr if noise_name.startswith('correlated') else white)
        prefix = raw[:, :51] + noise[:, :51]
        offset = prefix.mean(axis=1)
        scale = np.maximum(np.sqrt(np.mean((prefix - offset[:, None]) ** 2, axis=1)), 1e-8)
        signal = np.clip((raw + noise - offset[:, None]) / scale[:, None], -50, 50).astype(np.float32)
        accepted = np.flatnonzero(observed[row])
        if not len(accepted):
            continue
        selected = np.unique(accepted[np.linspace(0, len(accepted) - 1, min(4, len(accepted))).astype(int)])
        for frame in selected:
            end = 128 + 64 * int(frame)
            assert not commands[raw_index, end - 128:end].any()
            window = signal[:, end - 128:end].astype(np.float64)
            assert np.isfinite(window).all() and not (abs(window) >= 50).any()
            window -= window.mean(axis=1, keepdims=True)
            window *= .5 - .5 * np.cos(2 * np.pi * np.arange(128) / 128)
            transform = np.fft.rfft(window, axis=1)[:, 1:64]
            p = np.einsum('ik,ik->k', transform.conj(), transform).real
            total = float(p.sum()); distribution = p / total
            nonzero = distribution > 0
            expected = np.full(3, np.nan)
            expected[0] = -np.sum(distribution[nonzero] * np.log(distribution[nonzero])) / np.log(63)
            union = set()
            centers = [-1, -1, -1]
            for j, f in enumerate(q[row, frame, [8, 11, 14]]):
                if np.isfinite(f) and 0 < f < 500:
                    center = min(range(1, 64), key=lambda k: (abs(k * 1000 / 128 - f), k))
                    centers[j] = center
                    union.update(range(max(1, center - 1), min(63, center + 1) + 1))
            if union:
                expected[1] = distribution[[k - 1 for k in sorted(union)]].sum()
            singular = np.linalg.svd(transform, compute_uv=False)
            expected[2] = np.sum(singular ** 4) / np.sum(singular ** 2) ** 2
            np.testing.assert_array_equal(bins[row, frame], centers)
            np.testing.assert_array_equal(valid[row, frame], np.isfinite(expected))
            close(values[row, frame][np.isfinite(expected)], expected[np.isfinite(expected)])
            maximum_descriptor_error = max(maximum_descriptor_error, float(np.max(abs(values[row, frame] - expected))))
            descriptor_checks += 1
    # Independently calculate all masks, per-design errors and event counts.
    slope = abs(np.gradient(y, times, axis=1))
    crossing = np.zeros(y.shape, dtype=bool)
    changed_set = np.zeros_like(crossing)
    for row in range(len(records)):
        for frame in range(1, len(times)):
            if observed[row, frame - 1] and observed[row, frame]:
                crossing[row, frame] = tuple(bins[row, frame - 1]) != tuple(bins[row, frame])
                changed_set[row, frame] = sorted(bins[row, frame - 1]) != sorted(bins[row, frame])
    boundary = (y > -.02) & (y < .05)
    mask_map = dict(all=np.ones(y.shape, dtype=bool), current=observed, gap=~observed,
        boundary=boundary, boundary_current=boundary & observed, boundary_gap=boundary & ~observed,
        stable_side=(y >= .01) & (y <= .05), unstable_exclusion=(y <= -.01) & (y >= -.05),
        farther_stable10=y >= .10, farther_stable15=y >= .15,
        stationary=slope <= 1e-10, slow_ramp=(slope > 1e-10) & (slope <= .04), fast_ramp=slope > .04,
        bin_transition=crossing, bin_set_change=changed_set, mode_slot_reordering=crossing & ~changed_set,
        current_no_bin_transition=observed & ~crossing)
    for low, high in ((-.02, 0), (0, .01), (.01, .02), (.02, .05), (.05, .10)):
        mask_map[f'margin_{low}_{high}'] = (y >= low) & (y < high)
    design_array = np.array([r['design'] for r in records])
    metric_checks = 0
    def verify_summary(report, point, selected):
        nonlocal metric_checks
        assert report['frames'] == int(selected.sum())
        rows = []
        for design in sorted(set(design_array.tolist())):
            mask = selected & (design_array == design)[:, None]
            if mask.any():
                errors = point[:, mask] - y[mask][None]
                expected = abs(errors).mean(axis=1)
                close(report['per_design'][design]['mae_by_seed'], expected)
                close(report['per_design'][design]['signed_error_by_seed'], errors.mean(axis=1))
                assert report['per_design'][design]['frames'] == int(mask.sum())
                rows.append(expected)
        assert report['designs'] == len(rows)
        close(report['equal_design_mae_by_seed'], np.mean(rows, axis=0) if rows else None)
        close(report['equal_design_mae'], float(np.mean(rows)) if rows else None)
        event_definitions = (
            ('optimism_over02', selected, point - y[None] > .02),
            ('unstable_positive_point', selected & (y < 0), point > 0),
            ('stable_wrong_sign', selected & (y >= .01) & (y <= .05), point <= 0),
            ('unstable_exclusion_wrong_sign', selected & (y <= -.01) & (y >= -.05), point > 0))
        for name, mask, event in event_definitions:
            expected = np.count_nonzero(event & mask[None], axis=(1, 2))
            assert report[name]['frames'] == int(mask.sum())
            np.testing.assert_array_equal(report[name]['events_by_seed'], expected)
            close(report[name]['rate_by_seed'], expected / mask.sum() if mask.any() else None)
        metric_checks += 1
    for condition, by_arm in result['metrics'].items():
        split, noise = condition.split('/')
        population = np.array([r['split'] == split and r['noise'] == noise for r in records])[:, None]
        for ai, arm in enumerate(arms):
            for stratum, report in by_arm[arm].items():
                verify_summary(report, predictions[ai], population & mask_map[stratum])
            for profile in ('cross', 'stable', 'retreat'):
                pop = population & np.array([r['profile'] == profile for r in records])[:, None]
                for stratum, report in result['profiles'][condition][profile][arm].items():
                    verify_summary(report, predictions[ai], pop & mask_map[stratum])
                for stratum, report in result['matched_margin_profiles'][condition][profile][arm].items():
                    verify_summary(report, predictions[ai], pop & mask_map[stratum])
            selected = population & crossing
            error = predictions[ai] - y[None]
            jumps = np.zeros_like(error); jumps[:, :, 1:] = error[:, :, 1:] - error[:, :, :-1]
            report = result['bin_transitions'][condition][arm]
            assert report['frames'] == int(selected.sum())
            close(report['mean_abs_error_jump_by_seed'], abs(jumps[:, selected]).mean(axis=1) if selected.any() else None)
            close(report['p95_abs_error_jump_by_seed'], np.percentile(abs(jumps[:, selected]), 95, axis=1) if selected.any() else None)
    # Saved-network replay uses independently built NumPy inputs and float64 linear algebra.
    train_rows = q[proper][observed[proper]]
    mean, std = train_rows.mean(axis=0), train_rows.std(axis=0)
    active = std > 1e-8 * np.maximum(1., abs(mean))
    normalized = (q - mean) / np.where(active, std + 1e-8, 1.)
    normalized[..., ~active] = 0; normalized[~observed] = 0
    samples = [(i, f) for i, r in enumerate(records) if r['split'] == 'held' for f in (0, 31, 62, 123)]
    replay_checks, maximum_prediction_error = 0, 0.
    for ai, arm in enumerate(arms):
        components = [normalized, observed[..., None], age[..., None]]
        columns = [0, 1] if 'spectral' in arm else [2] if 'spatial' in arm else []
        if columns:
            v, mask = values[..., columns], valid[..., columns]
            means = np.array([np.mean(v[proper, :, j][mask[proper, :, j]]) for j in range(len(columns))])
            imputed = np.where(mask, v, means)
            a = imputed[proper][observed[proper]]
            am, ast = a.mean(axis=0), a.std(axis=0)
            keep = ast > 1e-8 * np.maximum(1., abs(am))
            extra = (imputed - am) / np.where(keep, ast + 1e-8, 1.)
            extra[..., ~keep] = 0
            if arm.startswith('shuffled_'):
                rng = np.random.default_rng(20261009)
                for split in ('train', 'validation', 'held'):
                    groups = sorted(set((r['design'], r['replicate'], r['profile']) for r in records if r['split'] == split))
                    permutation = rng.permutation(len(groups))
                    group_index = {group: j for j, group in enumerate(groups)}
                    index = {(r['design'], r['replicate'], r['profile'], r['noise']): i for i, r in enumerate(records) if r['split'] == split}
                    old = extra.copy()
                    for i, r in enumerate(records):
                        if r['split'] == split:
                            donor = groups[permutation[group_index[(r['design'], r['replicate'], r['profile'])]]]
                            extra[i] = old[index[(*donor, r['noise'])]]
            extra[~observed] = 0
            components.extend((extra, mask))
        x = np.concatenate(components, axis=2).astype(np.float32)
        for si, seed in enumerate(seeds):
            checkpoint = torch.load(OUT / f'{arm}_seed{seed}.pt', map_location='cpu', weights_only=False)
            assert checkpoint['protocol_sha256'] == sha(OUT / 'protocol.json')
            state = checkpoint['state']
            a = np.array([x[i, frame] for i, frame in samples], dtype=np.float64)
            for layer in ('0', '2', '4'):
                a = a @ state[f'{layer}.weight'].numpy().T + state[f'{layer}.bias'].numpy()
                if layer != '4':
                    a = np.maximum(a, 0)
            expected = np.array([predictions[ai, si, i, frame] for i, frame in samples])
            close(a[:, 0], expected, 2e-6)
            maximum_prediction_error = max(maximum_prediction_error, float(abs(a[:, 0] - expected).max()))
            replay_checks += len(samples)
    verification = dict(results_sha256=sha(OUT / 'results.json'), dataset_sha256=sha(OUT / 'dataset.npz'),
        predictions_sha256=sha(OUT / 'predictions.npz'), protocol_sha256=sha(OUT / 'protocol.json'),
        verification_source_sha256=sha(__file__), original_population_checks=original_checks,
        independent_descriptor_windows=descriptor_checks, maximum_descriptor_error=maximum_descriptor_error,
        independent_metric_rows=metric_checks, independent_saved_network_frames=replay_checks,
        maximum_saved_network_prediction_error=maximum_prediction_error,
        baseline_seed42_reproduction_error=training['baseline_seed42_reproduction_error'],
        no_reserved_designs_or_fresh_views_used_in_fitting=True,
        noise_wording_erratum='Protocol says original-eight RMS; pinned executed M.noises and canonical temporal source use all16 current sensor channels. Numerical noise generation is unchanged and independently reconstructed; no results retuned.',
        limitations='Numerical/development verification only, no blind or independent physical validation')
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2, allow_nan=False) + '\n')
    print('ADDITIVE_INDEPENDENTLY_VERIFIED', json.dumps(verification), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
