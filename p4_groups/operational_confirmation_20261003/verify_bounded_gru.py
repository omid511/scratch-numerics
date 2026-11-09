"""Independent CPU gate/slot/GRU equations and saved-prediction metric replay."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
SOURCE = SCRIPT / 'support_modal_20261009'
OUT = SCRIPT / 'bounded_gru_20261009'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_current(q, observed, age, proper):
    rows = q[proper][observed[proper]]
    mean = rows.mean(axis=0); std = rows.std(axis=0)
    active = std > 1e-8 * np.maximum(1, abs(mean))
    scale = np.where(active, std + 1e-8, 1.0)
    result = np.zeros((*q.shape[:2], 176), dtype=np.float32)
    normalized = (q - mean) / scale
    normalized[..., ~active] = 0
    result[..., :174] = np.where(observed[..., None], normalized, 0)
    result[..., 174] = observed; result[..., 175] = age
    return result, mean, scale, active


def linear(x, state, prefix):
    return x @ state[f'{prefix}.weight'].T + state[f'{prefix}.bias']


def replay(current, arm, state, indices):
    # Construct each decision independently, never calling the experiment's helpers.
    slots = np.zeros((len(indices), 3, 176), dtype=np.float32)
    for j, (row, frame) in enumerate(indices):
        for k, lag in enumerate((4, 2, 0)):
            if frame >= lag:
                slots[j, k] = current[row, frame - lag]
    if arm.endswith('gru'):
        if arm == 'flags_age_gru':
            slots = slots[..., -2:]
        hidden = np.zeros((len(indices), 32), dtype=np.float32)
        for k in range(3):
            i = slots[:, k] @ state['gru.weight_ih_l0'].T + state['gru.bias_ih_l0']
            h = hidden @ state['gru.weight_hh_l0'].T + state['gru.bias_hh_l0']
            ir, iz, inc = np.split(i, 3, axis=1); hr, hz, hnc = np.split(h, 3, axis=1)
            reset, update = expit(ir + hr), expit(iz + hz)
            candidate = np.tanh(inc + reset * hnc)
            hidden = (1 - update) * candidate + update * hidden
        return linear(np.maximum(linear(hidden, state, 'head.0'), 0), state, 'head.2')[:, 0]
    if arm == 'unordered_pool':
        pooled = slots.mean(axis=1)
        x = np.concatenate((pooled, pooled, pooled), axis=1)
    elif arm == 'ordered_concat':
        x = slots.reshape(len(indices), -1)
    else:
        raise ValueError(arm)
    return linear(np.maximum(linear(np.maximum(linear(x, state, '0'), 0), state, '2'), 0), state, '4')[:, 0]


def independent_strata(y, observed, times, profiles):
    warm = np.broadcast_to(np.arange(y.shape[1])[None, :] >= 4, y.shape)
    past4 = np.zeros_like(y); past2 = np.zeros_like(y)
    past4[:, 4:] = y[:, :-4]; past2[:, 2:] = y[:, :-2]
    span = np.maximum.reduce((past4, past2, y)) - np.minimum.reduce((past4, past2, y))
    boundary = (y > -.02) & (y < .05)
    result = dict(all=np.ones_like(observed), current=observed, gap=~observed, boundary=boundary,
        boundary_current=boundary & observed, boundary_gap=boundary & ~observed, post_warmup=warm,
        warmup=~warm, distinct_regimes=warm & (span > .005), distinct_regimes_boundary=warm & (span > .005) & boundary,
        single_regime_dwell=warm & (span <= 1e-10), single_regime_dwell_boundary=warm & (span <= 1e-10) & boundary,
        weak_regime_change=warm & (span > 1e-10) & (span <= .005), farther_stable=y >= .10,
        farther_stable15=y >= .15, unstable_exclusion=(y <= -.01) & (y >= -.05))
    for profile in sorted(set(profiles)):
        pm = (profiles == profile)[:, None]
        result[f'profile_{profile}'] = pm & result['all']
        result[f'boundary_profile_{profile}'] = pm & boundary
    return result


def check_metric(metric, points, truth, selected, designs):
    assert metric['frames'] == int(selected.sum())
    design_values = []
    for did, item in metric['per_design'].items():
        mask = selected & (designs == did)[:, None]
        assert item['frames'] == int(mask.sum())
        error = points[:, mask].astype(np.float64) - truth[mask]
        mae = np.abs(error).mean(axis=1)
        signed = error.mean(axis=1)
        np.testing.assert_allclose(mae, item['mae_by_seed'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(signed, item['signed_error_by_seed'], atol=1e-12, rtol=1e-12)
        design_values.append(mae)
    if design_values:
        np.testing.assert_allclose(np.mean(design_values, axis=0), metric['equal_design_mae_by_seed'], atol=1e-12, rtol=1e-12)
        assert abs(float(np.mean(design_values)) - metric['equal_design_mae']) <= 1e-12
    else:
        assert metric['equal_design_mae'] is None
    for key, eligible, comparator in (
        ('optimism_over02', selected, lambda p, y: p - y > .02),
        ('unstable_positive_point', selected & (truth < 0), lambda p, y: p > 0),
        ('stable_wrong_sign', selected & (truth >= .01) & (truth <= .05), lambda p, y: p <= 0),
        ('unstable_exclusion_wrong_sign', selected & (truth <= -.01) & (truth >= -.05), lambda p, y: p > 0)):
        count = int(eligible.sum())
        events = comparator(points[:, eligible], truth[eligible]).sum(axis=1)
        assert metric[key]['frames'] == count
        np.testing.assert_array_equal(events, metric[key]['events_by_seed'])
        if count:
            np.testing.assert_allclose(events / count, metric[key]['rate_by_seed'], atol=1e-12, rtol=0)
        else:
            assert metric[key]['rate_by_seed'] is None
    return 1


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    results = json.loads((OUT / 'results.json').read_text())
    for relative, digest in protocol['inputs'].items():
        assert sha(ROOT / relative) == digest, relative
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        q, y, observed, age, times = [z[k].copy() for k in ('features', 'truth', 'observed', 'age', 'times')]
        records = json.loads(str(z['records']))
    designs = np.array([r['design'] for r in records]); profiles = np.array([r['profile'] for r in records])
    strata = independent_strata(y, observed, times, profiles)
    count = replays = 0; maximum = 0.; checkpoint_hashes = {}
    for fold, excluded in enumerate(protocol['evaluation_folds']):
        for arm in protocol['arms']:
            points = []
            for seed in protocol['training']['seeds']:
                stem = f'fold{fold}_{arm}_seed{seed}'
                path = OUT / f'{stem}.npz'
                assert sha(path) == results['artifact_hashes'][path.name]
                with np.load(path, allow_pickle=False) as z:
                    saved = z['predictions'].copy(); proper = z['proper'].copy(); validation = z['validation'].copy()
                    current, mean, scale, active = independent_current(q, observed, age, proper)
                    np.testing.assert_allclose(mean, z['mean'], atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(scale, z['scale'], atol=1e-12, rtol=1e-12)
                    np.testing.assert_array_equal(active, z['active'])
                assert np.isfinite(saved).all()
                assert not set(designs[proper]) & (set(excluded) | set(protocol['split']['validation']) | set(protocol['split']['held']))
                assert set(designs[validation]) == set(protocol['split']['validation'])
                checkpoint_path = OUT / f'{stem}.pt'; checkpoint_hashes[checkpoint_path.name] = sha(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                assert checkpoint['protocol_sha256'] == sha(OUT / 'protocol.json')
                state = {k: v.numpy() for k, v in checkpoint['state'].items()}
                # Across records, current/gap and startup; include all slot lags.
                indices = [(int(r), int(f)) for r in np.linspace(0, len(records) - 1, 12, dtype=int)
                    for f in (0, 1, 2, 3, 4, 5, y.shape[1] // 2, y.shape[1] - 1)]
                expected = replay(current, arm, state, indices)
                actual = np.array([saved[r, f] for r, f in indices])
                difference = float(abs(expected - actual).max())
                assert difference <= 2e-6, (stem, difference)
                maximum = max(maximum, difference); replays += len(indices); points.append(saved)
            points = np.stack(points)
            for condition, arms in results['metrics'].items():
                pieces = condition.split('/')
                if pieces[0] != f'fold{fold}':
                    continue
                ids = excluded if pieces[1] == 'evaluation' else protocol['split']['held']
                population = np.array([r['design'] in ids and r['noise'] == pieces[2] for r in records])[:, None]
                for name, metric in arms[arm].items():
                    count += check_metric(metric, points, y, population & strata[name], designs)
    result = dict(results_sha256=sha(OUT / 'results.json'), protocol_sha256=sha(OUT / 'protocol.json'),
        independent_metric_rows=count, independent_saved_network_decisions=replays, maximum_numpy_network_error=maximum,
        checkpoints_sha256=checkpoint_hashes, no_cross_design_or_future_inputs=True, all_original_denominators_preserved=True,
        progress_ledger_sha256=sha(ROOT / 'P4_PROGRESS.json'), qualification='Not established by development replay')
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('BOUNDED_GRU_INDEPENDENTLY_VERIFIED', json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
