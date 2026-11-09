"""Verify bounded history inputs, saved networks, all metrics and target-gate gaps."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUT = SCRIPT / 'multivel_history_20261009'
SOURCE = SCRIPT / 'additive_spectral_20261009'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, atol=1e-12):
    if a is None or b is None:
        assert a is b
    else:
        np.testing.assert_allclose(a, b, rtol=0, atol=atol)


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    result = json.loads((OUT / 'results.json').read_text())
    for section in ('inputs', 'helpers', 'implementation'):
        for path, digest in protocol[section].items():
            assert sha(ROOT / path) == digest
    assert result['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert result['implementation_sha256'] == sha(SCRIPT / 'experiment_multivel_history.py')
    assert result['predictions_sha256'] == sha(OUT / 'predictions.npz')
    assert protocol['progress_ledger_sha256'] == sha(ROOT / 'P4_PROGRESS.json')
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        q, y, observed, age, times = (z[name] for name in ('features', 'truth', 'observed', 'age', 'times'))
        records = json.loads(str(z['records']))
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as z:
        points = z['predictions']; arms = json.loads(str(z['arms'])); seeds = z['seeds'].tolist()
        scaling = [z[name] for name in ('mean', 'scale', 'active')]
    with np.load(SOURCE / 'predictions.npz', allow_pickle=False) as z:
        source_arms = json.loads(str(z['arms']))
        np.testing.assert_array_equal(points[0], z['predictions'][source_arms.index('baseline174')])
    proper = np.array([r['split'] == 'train' and r['noise'] in ('clean', 'iid1') for r in records])
    validation = np.array([r['split'] == 'validation' and r['noise'] in ('clean', 'iid1') for r in records])
    assert proper.sum() == 48 and validation.sum() == 24
    assert points.shape == (5, 3, *y.shape) and np.isfinite(points).all()
    for split in ('train', 'validation', 'held'):
        assert set(r['design'] for r in records if r['split'] == split) == set(protocol['split'][split])
    train = q[proper][observed[proper]]
    mean, std = train.mean(axis=0), train.std(axis=0)
    active = std > 1e-8 * np.maximum(1., abs(mean)); scale = np.where(active, std + 1e-8, 1.)
    for actual, expected in zip(scaling, (mean, scale, active)):
        np.testing.assert_array_equal(actual, expected)
    normalized = (q - mean) / scale; normalized[..., ~active] = 0; normalized[~observed] = 0
    current = np.concatenate((normalized, observed[..., None], age[..., None]), axis=2).astype(np.float32)
    # Independently determine a donor's physical trajectory, never its label.
    rng = np.random.default_rng(20261011); donors = list(range(len(records)))
    for split in ('train', 'validation', 'held'):
        physical = sorted(set((r['design'], r['replicate'], r['profile']) for r in records if r['split'] == split))
        order = rng.permutation(len(physical)); where = {p: i for i, p in enumerate(physical)}
        lookup = {(r['design'], r['replicate'], r['profile'], r['noise']): i for i, r in enumerate(records) if r['split'] == split}
        for row, record in enumerate(records):
            if record['split'] == split:
                other = physical[order[where[(record['design'], record['replicate'], record['profile'])]]]
                donors[row] = lookup[(*other, record['noise'])]
                assert records[donors[row]]['split'] == split and records[donors[row]]['noise'] == record['noise']
    # Raw support endpoints, not only tensor dimensions: 3 adjacent128 samples, <=.5s.
    support_checks = 0
    for frame in range(4, len(times)):
        end = 128 + 64 * frame
        intervals = [(end - lag * 64 - 128, end - lag * 64) for lag in (4, 2, 0)]
        assert intervals[0][1] == intervals[1][0] and intervals[1][1] == intervals[2][0]
        assert intervals[2][1] == end and (end - intervals[0][0]) / 1000 == .384
        assert intervals[0][0] >= 0 and all(b <= end for _, b in intervals)
        support_checks += 1
    def input_at(row, frame, arm):
        if arm == 'single128':
            return current[row, frame]
        if arm == 'repeated_current':
            return np.tile(current[row, frame], 3)
        slots = []
        for lag in (4, 2, 0):
            other = donors[row] if arm == 'shuffled_past' and lag else row
            slots.append(current[other, frame - lag] if frame >= lag else np.zeros(176, dtype=np.float32))
        if arm == 'unordered_history':
            average = (slots[0] + slots[1] + slots[2]) / 3
            slots = [average, average, average]
        return np.concatenate(slots)
    replay_checks, maximum_error = 0, 0.
    samples = [(row, frame) for row, r in enumerate(records) if r['split'] == 'held' for frame in (0, 1, 3, 4, 31, 62, 123)]
    for ai, arm in enumerate(arms):
        x = np.stack([input_at(row, frame, arm) for row, frame in samples]).astype(np.float64)
        for si, seed in enumerate(seeds):
            directory = SOURCE if arm == 'single128' else OUT
            name = 'baseline174' if arm == 'single128' else arm
            checkpoint = torch.load(directory / f'{name}_seed{seed}.pt', map_location='cpu', weights_only=False)
            state = checkpoint['state']; value = x
            for layer in ('0', '2', '4'):
                value = value @ state[f'{layer}.weight'].numpy().T + state[f'{layer}.bias'].numpy()
                if layer != '4':
                    value = np.maximum(value, 0)
            expected = np.array([points[ai, si, row, frame] for row, frame in samples])
            close(value[:, 0], expected, 2e-6)
            maximum_error = max(maximum_error, float(abs(value[:, 0] - expected).max()))
            replay_checks += len(samples)
    # All regime labels are reconstructed only for scoring. Nothing gates fitting by them.
    warm = np.broadcast_to(np.arange(len(times))[None] >= 4, y.shape)
    distinct = np.zeros_like(observed); dwell = np.zeros_like(observed)
    for row in range(len(records)):
        for frame in range(4, len(times)):
            v = [y[row, frame - 4], y[row, frame - 2], y[row, frame]]
            span = max(v) - min(v)
            distinct[row, frame] = span >= .005; dwell[row, frame] = span <= 1e-10
    boundary = (y > -.02) & (y < .05)
    masks = dict(all=np.ones_like(observed), current=observed, gap=~observed,
        boundary=boundary, boundary_current=boundary & observed, boundary_gap=boundary & ~observed,
        post_warmup=warm, warmup=~warm, distinct_regimes=distinct,
        distinct_regimes_boundary=distinct & boundary, distinct_regimes_current=distinct & observed,
        single_regime_dwell=dwell, single_regime_dwell_boundary=dwell & boundary,
        weak_regime_change=warm & ~distinct & ~dwell, farther_stable=y >= .10,
        unstable_exclusion=(y <= -.01) & (y >= -.05))
    designs = np.array([r['design'] for r in records]); metric_checks = 0
    def verify(report, point, selected):
        nonlocal metric_checks
        assert report['frames'] == int(selected.sum())
        errors_by_design = []
        for design in sorted(set(designs.tolist())):
            mask = selected & (designs == design)[:, None]
            if mask.any():
                error = point[:, mask] - y[mask][None]
                expected = abs(error).mean(axis=1); errors_by_design.append(expected)
                close(report['per_design'][design]['mae_by_seed'], expected)
                close(report['per_design'][design]['signed_error_by_seed'], error.mean(axis=1))
                assert report['per_design'][design]['frames'] == int(mask.sum())
        assert report['designs'] == len(errors_by_design)
        close(report['equal_design_mae_by_seed'], np.mean(errors_by_design, axis=0) if errors_by_design else None)
        close(report['equal_design_mae'], float(np.mean(errors_by_design)) if errors_by_design else None)
        definitions = (('optimism_over02', selected, point - y[None] > .02),
            ('unstable_positive_point', selected & (y < 0), point > 0),
            ('stable_wrong_sign', selected & (y >= .01) & (y <= .05), point <= 0),
            ('unstable_exclusion_wrong_sign', selected & (y <= -.01) & (y >= -.05), point > 0))
        for name, mask, event in definitions:
            n = int(mask.sum()); counts = np.count_nonzero(event & mask[None], axis=(1, 2))
            assert report[name]['frames'] == n
            np.testing.assert_array_equal(report[name]['events_by_seed'], counts)
            close(report[name]['rate_by_seed'], counts / n if n else None)
        metric_checks += 1
    for condition, by_arm in result['metrics'].items():
        split, noise = condition.split('/')
        pop = np.array([r['split'] == split and r['noise'] == noise for r in records])[:, None]
        for ai, arm in enumerate(arms):
            for stratum, report in by_arm[arm].items():
                verify(report, points[ai], pop & masks[stratum])
            for profile in ('cross', 'stable', 'retreat'):
                subset = pop & np.array([r['profile'] == profile for r in records])[:, None]
                for stratum, report in result['profiles'][condition][profile][arm].items():
                    verify(report, points[ai], subset & masks[stratum])
    verification = dict(results_sha256=sha(OUT / 'results.json'), predictions_sha256=sha(OUT / 'predictions.npz'),
        protocol_sha256=sha(OUT / 'protocol.json'), verifier_sha256=sha(__file__),
        independent_raw_support_checks=support_checks, union_acquisition_seconds=.384,
        independent_saved_network_frames=replay_checks, maximum_saved_network_error=maximum_error,
        independent_metric_rows=metric_checks, baseline_predictions_unchanged=True,
        no_velocity_labels_in_model_inputs=True, no_future_or_cross_split_history=True,
        progress_ledger_unchanged=True, qualification='Not assessed/passed; existing missing populations and no blind independent validation',
        interpretation='Tests causal within384ms multi-regime history. Cannot identify effects of independent known-speed sweeps or larger velocity span; current128ms source is already time varying.')
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print('MULTIVEL_HISTORY_INDEPENDENTLY_VERIFIED', json.dumps(verification), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
