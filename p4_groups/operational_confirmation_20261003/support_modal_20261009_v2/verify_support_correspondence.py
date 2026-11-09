"""Recompute accepted branch overlap/frequency distances without acquisition code."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from verify_support_acquisition import physical_table, sha

SCRIPT = Path(__file__).resolve().parent


def run():
    protocol = json.loads((SCRIPT / 'protocol.json').read_text())
    criteria = protocol['correspondence']
    with np.load(SCRIPT / 'dataset.npz', allow_pickle=False) as z:
        records = json.loads(str(z['records']))
        data = {name: z[name].copy() for name in ('observed', 'mode_count', 'matched_branch',
            'match_scores', 'measured_frequency', 'measured_mode_shapes')}
    with np.load(SCRIPT / 'recovered_modal_shapes.npz', allow_pickle=False) as recovery:
        assert str(recovery['source_dataset_sha256']) == sha(SCRIPT / 'dataset.npz')
        recovered_shapes = recovery['measured_mode_shapes'].copy()
    assert np.iscomplexobj(recovered_shapes)
    np.testing.assert_array_equal(recovered_shapes.real, data['measured_mode_shapes'])
    data['measured_mode_shapes'] = recovered_shapes
    checks = 0; score_error = 0.; minimum_overlap = 1.; maximum_frequency_distance = 0.
    by_design = {}
    for did in protocol['split']['train'] + protocol['split']['validation'] + protocol['split']['held'] + protocol['additional_designs']:
        by_design[did] = 0
        with np.load(SCRIPT / 'shards' / f'{did}.npz', allow_pickle=False) as shard:
            physical_records = json.loads(str(shard['records']))
            for ri, record in enumerate(records):
                if record['design'] != did or record['noise'] != 'clean' or record['replicate'] != 0:
                    continue
                frames = np.flatnonzero(data['observed'][ri] & (data['matched_branch'][ri] >= 0).any(axis=1))
                if not len(frames):
                    continue
                pi = next(i for i, r in enumerate(physical_records) if r['replicate'] == 0 and r['profile'] == record['profile'])
                raw = shard['raw'][pi]
                scale = np.maximum(np.std(raw[:, :51], axis=1), 1e-8)
                _, ids, poles, shapes, _ = physical_table(shard, record, pi, protocol)
                for frame in frames[np.unique(np.linspace(0, len(frames) - 1, min(3, len(frames)), dtype=int))]:
                    end = 128 + 64 * int(frame)
                    indices = ids[end - 128:end]
                    center = indices[(len(indices) - 1) // 2]
                    branches = data['matched_branch'][ri, frame, :data['mode_count'][ri, frame]]
                    accepted = branches[branches >= 0]
                    assert len(set(accepted)) == len(accepted)
                    for slot, branch in enumerate(branches):
                        if branch < 0:
                            continue
                        measured = data['measured_mode_shapes'][ri, frame, slot]
                        physical = shapes[center, :, branch] / scale
                        overlap = float(abs(np.vdot(measured, physical)) ** 2 / (np.vdot(measured, measured).real * np.vdot(physical, physical).real))
                        distance = float(np.sqrt(np.mean((data['measured_frequency'][ri, frame, slot] - poles[indices, branch].imag / (2 * np.pi)) ** 2)))
                        difference = abs(overlap - data['match_scores'][ri, frame, slot, 0])
                        assert difference <= 1e-9, (did, frame, slot, difference)
                        assert overlap >= criteria['minimum_squared_overlap'] - 1e-12
                        assert distance <= criteria['maximum_frequency_distance_hz'] + 1e-12
                        checks += 1; by_design[did] += 1
                        score_error = max(score_error, difference); minimum_overlap = min(minimum_overlap, overlap)
                        maximum_frequency_distance = max(maximum_frequency_distance, distance)
    report = dict(dataset_sha256=sha(SCRIPT / 'dataset.npz'), protocol_sha256=sha(SCRIPT / 'protocol.json'),
        verification_source_sha256=sha(__file__), recovered_diagnostics_sha256=sha(SCRIPT / 'recovered_modal_shapes.npz'), independent_branch_checks=checks, checked_by_design=by_design,
        maximum_overlap_score_error=score_error, minimum_accepted_overlap=minimum_overlap if checks else None,
        maximum_accepted_frequency_distance_hz=maximum_frequency_distance,
        limitation='Clean replicate0 accepted-branch spot checks; not an independent completeness or uniqueness proof for unresolved/ambiguous correspondence')
    (SCRIPT / 'correspondence_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SUPPORT_CORRESPONDENCE_INDEPENDENTLY_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
