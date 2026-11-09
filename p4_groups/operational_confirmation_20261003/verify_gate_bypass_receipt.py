"""Independent byte/provenance and original-population floor/leverage checks."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUT = SCRIPT / 'gate_bypass_receipt_20261010'
SOURCE = SCRIPT / 'support_modal_20261009_v2'
BASELINE = SCRIPT / 'bounded_gru_20261009_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_floor(inputs, target, weights):
    buckets = defaultdict(list)
    for index, vector in enumerate(inputs):
        buckets[vector.tobytes()].append(index)
    total = 0.0
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda i: target[i])
        half = sum(weights[i] for i in ordered) / 2
        running = 0.0
        for i in ordered:
            running += weights[i]
            if running >= half:
                median = target[i]
                break
        total += sum(weights[i] * abs(target[i] - median) for i in indices)
    return float(total)


def main():
    analysis = json.loads((OUT / 'analysis.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for relative, digest in protocol['inputs'].items():
        assert sha(ROOT / relative) == digest, relative
    assert sha(SCRIPT / 'analyze_gate_bypass_receipt.py') == analysis['analysis_source_sha256']
    assert sha(OUT / 'protocol.json') == analysis['protocol_sha256']
    for relative, digest in analysis['artifact_hashes'].items():
        assert sha(ROOT / relative) == digest, relative
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        q, observed, age, y = [z[k].copy() for k in ('features', 'observed', 'age', 'truth')]
        records = json.loads(str(z['records']))
    original_accepted = int(observed.sum())
    retained = q.copy()
    eligibility = np.zeros_like(observed)
    indices_seen = []
    eligible_total = replay_failures = gate_mismatches = 0
    for path in sorted(OUT.glob('replay_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            idx, e = z['row_indices'], z['eligible']
            indices_seen.extend(idx.tolist())
            assert str(z['protocol_sha256']) == sha(OUT / 'protocol.json')
            assert not (e & observed[idx]).any()
            assert np.array_equal(z['retained_features'][observed[idx]].view(np.uint8), q[idx][observed[idx]].view(np.uint8))
            assert np.array_equal(z['retained_features'][~(observed[idx] | e)], q[idx][~(observed[idx] | e)])
            assert (z['finite_only_eligible'] <= e).all()
            assert np.array_equal(z['finite_only_eligible'], e & (z['forecast_type'] == 1))
            assert np.isin(z['forecast_type'][e], (1, 2, 3)).all()
            assert np.isin(z['reason'][e], (7, 8, 9)).all()
            assert np.isfinite(z['retained_features'][e]).all()
            assert (z['mode_count'][e] > 0).all()
            assert np.array_equal(z['retained_features'][e, 18], z['mode_count'][e])
            assert (z['retained_features'][e, 21] == 1).all()
            retained[idx] = z['retained_features']
            eligibility[idx] = e
            eligible_total += int(e.sum())
            replay_failures += int((observed[idx] & (z['reason'] != 0)).sum())
            gate_mismatches += int((observed[idx] & (z['forecast_type'] != 4)).sum())
    assert sorted(indices_seen) == list(range(len(records)))
    assert eligible_total == analysis['inventory_eligible']
    p = json.loads((BASELINE / 'protocol.json').read_text())
    designs = np.array([r['design'] for r in records])
    noises = np.array([r['noise'] for r in records])
    checks = 0
    max_floor_error = max_contribution_error = 0.0
    for fold, eval_ids in enumerate(p['evaluation_folds']):
        with np.load(BASELINE / f'fold{fold}_content_gru_seed42.npz', allow_pickle=False) as z:
            mean, scale, active = [z[k].copy() for k in ('mean', 'scale', 'active')]
            predictions = z['predictions'].copy()
        for noise in protocol['primary_noise']:
            rows = np.flatnonzero(np.isin(designs, eval_ids) & (noises == noise))
            d, t, o, a, e = designs[rows], y[rows], observed[rows], age[rows], eligibility[rows]
            slots = []
            for content_q, allowed in ((q[rows], o), (retained[rows], o | e)):
                content = np.zeros_like(content_q)
                content[allowed] = (content_q[allowed] - mean) / scale
                content[..., ~active] = 0
                current = np.concatenate((content, o[..., None], a[..., None]), axis=-1).astype(np.float32)
                assembled = np.zeros((*o.shape, 3, 176), np.float32)
                for slot, lag in enumerate((4, 2, 0)):
                    for frame in range(lag, o.shape[1]):
                        assembled[:, frame, slot] = current[:, frame - lag]
                slots.append(assembled)
            changed = (slots[0].view(np.uint32) != slots[1].view(np.uint32)).any(axis=(2, 3))
            for stratum, mask in (('all', np.ones_like(o)), ('boundary', (t > -.02) & (t < .05))):
                weights = np.zeros_like(t)
                represented = [design for design in sorted(set(d)) if (mask & (d == design)[:, None]).any()]
                for design in represented:
                    selected = mask & (d == design)[:, None]
                    weights[selected] = 1 / (len(represented) * int(selected.sum()))
                target = t[mask]
                matrix = slots[1][mask].reshape(-1, 528)
                floor = independent_floor(matrix, target, weights[mask])
                report = analysis['fold_reports'][f'fold{fold}/combined/{noise}/{stratum}']
                error = abs(floor - report['exact_collision_floor']['floor'])
                max_floor_error = max(max_floor_error, error)
                assert error < 1e-12, (fold, noise, stratum, error)
                absolute = abs(predictions[rows] - t)
                affected = float(np.sum(weights * absolute * changed))
                unaffected = float(np.sum(weights * absolute * ~changed))
                stored = report['comparator_error_contributions']['content_gru']
                for actual, field in ((affected, 'changed_error_contribution_by_seed'),
                                      (unaffected, 'unchanged_error_contribution_by_seed')):
                    delta = abs(actual - stored[field][0])
                    max_contribution_error = max(max_contribution_error, delta)
                    assert delta < 1e-12, (fold, noise, stratum, field, delta)
                assert report['changed_input_history_frames'] == int((changed & mask).sum())
                checks += 1
    fixture_inputs = np.array([[1, 2], [1, 2], [9, 2]], np.float32)
    assert abs(independent_floor(fixture_inputs, np.array([0., .4, .9]), np.array([.125, .375, .5])) - .05) < 1e-12
    fixture_inputs[1, 0] = np.nextafter(fixture_inputs[1, 0], np.float32(np.inf))
    assert independent_floor(fixture_inputs, np.array([0., .4, .9]), np.array([.125, .375, .5])) == 0
    receipt = dict(independent_primary_population_checks=checks, maximum_floor_discrepancy=max_floor_error,
        maximum_comparator_contribution_discrepancy=max_contribution_error,
        inventory_rows=len(records), inventory_frames=int(observed.size), eligible_frames=eligible_total,
        cached_accepted_frames=original_accepted, accepted_replay_failures=replay_failures,
        accepted_replay_gate_mismatches=gate_mismatches, all_source_rows_once=True,
        cached_accepted_byte_preserved=True, qualification_and_original_denominators_preserved=True,
        progress_ledger_sha256=sha(ROOT / 'P4_PROGRESS.json'), analysis_sha256=sha(OUT / 'analysis.json'),
        verification_source_sha256=sha(__file__), no_fits=True)
    (OUT / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('GATE_RECEIPT_INDEPENDENT_VERIFICATION', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
