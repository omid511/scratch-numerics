"""Independent scaling, saved-head, fixed-gap and factorial metric verification."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import verify_bounded_gru as V
SCRIPT = Path(__file__).resolve().parent
SOURCE = SCRIPT / 'additive_spectral_20261009'
OUT = SCRIPT / 'current_context_20261009'
sha = V.sha


def masks(y, observed, times, bins, profiles):
    boundary = (y > -.02) & (y < .05)
    rate = abs(np.gradient(y, times, axis=1))
    transitions = np.zeros_like(observed)
    transitions[:, 1:] = observed[:, 1:] & observed[:, :-1] & (bins[:, 1:] != bins[:, :-1]).any(axis=2)
    set_changes = np.zeros_like(observed)
    set_changes[:, 1:] = transitions[:, 1:] & (np.sort(bins[:, 1:], axis=2) != np.sort(bins[:, :-1], axis=2)).any(axis=2)
    result = dict(all=np.ones_like(observed), current=observed, gap=~observed, boundary=boundary,
        boundary_current=boundary & observed, boundary_gap=boundary & ~observed,
        stable_side=(y >= .01) & (y <= .05), unstable_exclusion=(y <= -.01) & (y >= -.05),
        farther_stable10=y >= .10, farther_stable15=y >= .15, stationary=rate <= 1e-10,
        slow_ramp=(rate > 1e-10) & (rate <= .04), fast_ramp=rate > .04, bin_transition=transitions,
        bin_set_change=set_changes, mode_slot_reordering=transitions & ~set_changes,
        current_no_bin_transition=observed & ~transitions)
    for low, high in ((-.02, 0), (0, .01), (.01, .02), (.02, .05), (.05, .10)):
        result[f'margin_{low}_{high}'] = (y >= low) & (y < high)
    for profile in ('cross', 'stable', 'retreat'):
        pm = (profiles == profile)[:, None]
        result[f'profile_{profile}'] = pm & result['all']
        result[f'boundary_profile_{profile}'] = pm & boundary
    return result


def network(current, state):
    z = np.maximum(current @ state['0.weight'].T + state['0.bias'], 0)
    z = np.maximum(z @ state['2.weight'].T + state['2.bias'], 0)
    return (z @ state['4.weight'].T + state['4.bias'])[..., 0]


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    results = json.loads((OUT / 'results.json').read_text())
    for relative, digest in protocol['inputs'].items():
        assert sha(V.ROOT / relative) == digest, relative
    assert results['predictions_sha256'] == sha(OUT / 'predictions.npz')
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        q, y, observed, age, times, bins = [z[k].copy() for k in ('features', 'truth', 'observed', 'age', 'times', 'bins')]
        records = json.loads(str(z['records']))
    with np.load(SOURCE / 'predictions.npz', allow_pickle=False) as z:
        baseline = z['predictions'][json.loads(str(z['arms'])).index('baseline174')].copy()
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as z:
        points, natives = z['predictions'].copy(), z['native'].copy()
        proper, validation = z['proper'].copy(), z['validation'].copy()
        current, mean, scale, active = V.independent_current(q, observed, age, proper)
        np.testing.assert_allclose(mean, z['mean'], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(scale, z['scale'], atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(active, z['active'])
    designs = np.array([r['design'] for r in records]); profiles = np.array([r['profile'] for r in records])
    assert not set(designs[proper]) & set(designs[validation])
    assert set(designs[proper]) == {'D000007', 'D000033', 'D000068', 'D000087'}
    assert set(designs[validation]) == {'D000042', 'D000045'}
    strata = masks(y, observed, times, bins, profiles)
    count = decisions = transitions_checked = 0; maximum = 0.; checkpoint_hashes = {}
    for ai, arm in enumerate(protocol['arms']):
        array = current.copy()
        if arm.startswith('first107'):
            array[..., 107:174] = 0
        for si, seed in enumerate(protocol['seeds']):
            path = OUT / f'{arm}_seed{seed}.pt'
            checkpoint_hashes[path.name] = sha(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint['protocol_sha256'] == sha(OUT / 'protocol.json')
            state = {k: v.numpy() for k, v in checkpoint['state'].items()}
            replay = network(array, state)
            difference = float(abs(replay - natives[ai, si]).max())
            assert difference <= 2e-6, (arm, seed, difference)
            np.testing.assert_array_equal(points[ai, si][~observed], baseline[si][~observed])
            np.testing.assert_array_equal(points[ai, si][observed], natives[ai, si][observed])
            maximum = max(maximum, difference); decisions += replay.size
        for condition, arms in results['metrics'].items():
            split, noise = condition.split('/')
            pop = np.array([r['split'] == split and r['noise'] == noise for r in records])[:, None]
            for name, metric in arms[arm].items():
                count += V.check_metric(metric, points[ai], y, pop & strata[name], designs)
            changed = np.zeros_like(observed)
            changed[:, 1:] = observed[:, 1:] != observed[:, :-1]
            selected = pop & changed
            jump = np.zeros_like(points[ai], dtype=np.float64)
            jump[:, :, 1:] = np.diff(points[ai].astype(np.float64) - y[None], axis=2)
            item = results['transitions'][condition][arm]
            assert item['frames'] == int(selected.sum())
            if selected.any():
                values = abs(jump[:, selected])
                np.testing.assert_allclose(item['mean_abs_error_jump_by_seed'], values.mean(axis=1), atol=1e-12, rtol=0)
                np.testing.assert_allclose(item['p95_abs_error_jump_by_seed'], np.quantile(values, .95, axis=1), atol=1e-12, rtol=0)
            else:
                assert item['mean_abs_error_jump_by_seed'] is None
            transitions_checked += 1
            for name, per_design in results['factorial'][condition].items():
                for did, effects in per_design.items():
                    value = {a: np.array(arms[a][name]['per_design'][did]['mae_by_seed']) for a in protocol['arms']}
                    expected = dict(current_loss_full=value['full_current'] - value['full_all'],
                        second_block_all=value['first107_all'] - value['full_all'],
                        current_loss_first107=value['first107_current'] - value['first107_all'],
                        interaction=value['first107_current'] - value['first107_all'] - value['full_current'] + value['full_all'])
                    for key, result in expected.items():
                        np.testing.assert_allclose(effects[key], result, atol=1e-12, rtol=0)
    for si, seed in enumerate(protocol['seeds']):
        path = SOURCE / f'baseline174_seed{seed}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = {k: v.numpy() for k, v in checkpoint['state'].items()}
        error = float(abs(network(current, state) - baseline[si]).max())
        assert error <= 2e-6
        maximum = max(maximum, error); decisions += baseline[si].size
    report = dict(results_sha256=sha(OUT / 'results.json'), predictions_sha256=sha(OUT / 'predictions.npz'),
        independent_saved_network_decisions=decisions, maximum_numpy_network_error=maximum,
        independent_metric_rows=count, independent_transition_rows=transitions_checked,
        exact_original_native_gaps_preserved=True, factorial_effects_recomputed=True,
        checkpoint_hashes=checkpoint_hashes, progress_ledger_sha256=sha(V.ROOT / 'P4_PROGRESS.json'),
        qualification='Missing farther-stable and unstable-exclusion populations remain unknown')
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CURRENT_CONTEXT_INDEPENDENTLY_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
