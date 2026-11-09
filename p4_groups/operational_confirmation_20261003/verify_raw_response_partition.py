"""Independent dictionary/explicit-frame verification of the new partition only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from collections import defaultdict
import json
import numpy as np
import analyze_gate_bypass_receipt as A
import partition_raw_response_floor as P


def floor_by_dictionary(inputs, truth, weights):
    groups = defaultdict(list)
    for row, frame in zip(*np.nonzero(weights > 0)):
        groups[inputs[row, frame].tobytes()].append((float(truth[row, frame]), float(weights[row, frame])))
    total = 0.
    for members in groups.values():
        members.sort(key=lambda pair: pair[0])
        halfway = sum(weight for _, weight in members) / 2
        cumulative = 0.
        median = members[-1][0]
        for value, weight in members:
            cumulative += weight
            if cumulative >= halfway:
                median = value
                break
        total += sum(weight * abs(value-median) for value, weight in members)
    return total


def verify():
    result = json.loads((P.OUT / 'partition.json').read_text())
    protocol = json.loads((A.BASELINE / 'protocol.json').read_text())
    primary = json.loads((A.OUT / 'protocol.json').read_text())['primary_noise']
    with np.load(A.SOURCE / 'dataset.npz', allow_pickle=False) as z:
        records = json.loads(str(z['records']))
        truth, observed, age = [z[key].copy() for key in ('truth', 'observed', 'age')]
        ends = np.rint(z['times']*1000).astype(int)+1
    designs = np.array([r['design'] for r in records])
    noises = np.array([r['noise'] for r in records])
    checks, maximum = 0, 0.
    category_checks, metadata_checks = 0, 0
    for fold, evaluated in enumerate(protocol['evaluation_folds']):
        fold_rows = np.flatnonzero(np.isin(designs, evaluated))
        index = {int(value): row for row, value in enumerate(fold_rows)}
        shape = (len(fold_rows), len(ends))
        retained, eligible = np.empty((*shape, 174)), np.zeros(shape, bool)
        lengths, counts = np.empty((len(fold_rows), 2, len(ends)), np.int16), np.empty((len(fold_rows), 2, len(ends)), np.int16)
        packed = {}
        for design in evaluated:
            with np.load(A.OUT / f'replay_{design}.npz', allow_pickle=False) as z:
                indices = z['row_indices'].copy()
                local = np.array([index[int(i)] for i in indices])
                retained[local], eligible[local] = z['retained_features'], z['eligible']
            with np.load(P.OUT / f'masks_{design}.npz', allow_pickle=False) as z:
                assert np.array_equal(indices, z['row_indices'])
                lengths[local], counts[local] = z['longest_runs'], z['sample_counts']
                sample_count = int(z['sample_count'])
                for key in ('command', 'finite', 'cap'):
                    if key not in packed:
                        packed[key] = np.empty((len(fold_rows), z[key].shape[1]), np.uint8)
                    packed[key][local] = z[key]
        with np.load(A.BASELINE / f'fold{fold}_content_gru_seed42.npz', allow_pickle=False) as z:
            mean, scale, active = [z[key].copy() for key in ('mean', 'scale', 'active')]
        y, d, n = truth[fold_rows], designs[fold_rows], noises[fold_rows]
        inputs = A.model_inputs(retained, observed[fold_rows], age[fold_rows], eligible, mean, scale, active)
        qbytes = inputs.view(np.uint8).reshape(*shape, -1)
        unpacked = {key: np.unpackbits(value, axis=1, count=sample_count, bitorder='little').astype(bool) for key, value in packed.items()}
        for policy_index, policy in enumerate(('native_sample_rule', 'precap_sensitivity')):
            metadata = P.mask_information(packed, sample_count, ends, policy)
            extended = np.concatenate((qbytes, metadata), axis=-1)
            for row in range(len(fold_rows)):
                command, finite, cap = [unpacked[key][row] for key in ('command', 'finite', 'cap')]
                first_cap = next((i for i, value in enumerate(cap) if value), sample_count)
                positions = np.arange(sample_count)
                valid = ~command & finite & ~cap
                if policy_index:
                    valid &= positions < first_cap
                last_invalid = np.maximum.accumulate(np.where(valid, -1, positions))
                streak = positions - last_invalid
                for frame, end in enumerate(ends):
                    expected = np.zeros((5, 384), bool)
                    samples = np.arange(int(end)-384, int(end))
                    present = samples >= 0
                    actual = samples[present]
                    expected[0, present] = True
                    expected[1, present] = command[actual]
                    expected[2, present] = finite[actual]
                    expected[3, present] = cap[actual]
                    expected[4, present] = valid[actual]
                    decoded = np.unpackbits(metadata[row, frame, :-4].reshape(5, 48), axis=1, bitorder='little')
                    assert np.array_equal(decoded, expected)
                    assert int(np.frombuffer(metadata[row, frame, -4:].tobytes(), dtype='<i4')[0]) == end
                    start = max(0, int(end)-384)
                    maximum_run = int(np.minimum(streak[start:end], np.arange(1, int(end)-start+1)).max())
                    assert maximum_run == lengths[row, policy_index, frame]
                    assert int(expected[4].sum()) == counts[row, policy_index, frame]
                    metadata_checks += 1
            for noise in primary:
                for stratum in ('all', 'boundary'):
                    selected = (n == noise)[:, None] & (np.ones(shape, bool) if stratum == 'all' else ((y > -.02) & (y < .05)))
                    weights = np.zeros(shape)
                    represented = [design for design in np.unique(d) if (selected & (d == design)[:, None]).any()]
                    for design in represented:
                        take = selected & (d == design)[:, None]
                        weights[take] = 1/(len(represented)*int(take.sum()))
                    report = result['reports'][f'fold{fold}/{policy}/{noise}/{stratum}']
                    masks = {'full128': lengths[:, policy_index] >= 128,
                             'shorter_only': (lengths[:, policy_index] > 0) & (lengths[:, policy_index] < 128),
                             'no_samples': lengths[:, policy_index] == 0}
                    for name, mask in masks.items():
                        value = report['categories'][name]
                        positive = selected & mask
                        assert int(positive.sum()) == value['frames']
                        assert abs(float(weights[mask].sum()) - value['weight']) < 1e-12
                        for field, coordinates in (('restricted_modal_bound', inputs), ('mask_aware_restricted_bound', extended)):
                            independent = floor_by_dictionary(coordinates, y, weights*mask)
                            discrepancy = abs(independent - (value[field]['floor'] or 0))
                            assert discrepancy < 1e-12
                            maximum = max(maximum, discrepancy)
                            checks += 1
                        category_checks += 1
            print('RAW_PARTITION_VERIFIED_FOLD_POLICY', fold, policy, flush=True)
    assert result['progress_ledger_sha256'] == A.sha(A.ROOT / 'P4_PROGRESS.json')
    verification = dict(independent_restricted_floor_checks=checks, category_weight_frame_checks=category_checks,
        explicit_sample_metadata_checks=metadata_checks, maximum_floor_discrepancy=maximum,
        partition_sha256=A.sha(P.OUT / 'partition.json'), verifier_sha256=A.sha(__file__),
        protected_progress_ledger_sha256=result['progress_ledger_sha256'],
        no_fits=True, original_qualification_unchanged=True)
    A.save_json(P.OUT / 'verification.json', verification)
    print('RAW_RESPONSE_PARTITION_INDEPENDENT_VERIFICATION', json.dumps(verification), flush=True)


if __name__ == '__main__':
    verify()
