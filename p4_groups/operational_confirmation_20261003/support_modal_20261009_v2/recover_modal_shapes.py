"""Recover discarded sine coefficients from frozen poles; never alter feature rows."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT.parent))
import experiment_support_modal_v2 as A


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovered(signal, alpha, frequency):
    x = np.asarray(signal, dtype=np.float64)
    hits = np.flatnonzero(np.any(abs(x) >= 50, axis=0))
    n = int(hits[0]) if len(hits) else x.shape[1]
    x = x[:, :n]; x = x / abs(x).max()
    t = np.arange(n) * .001
    growth = alpha[:, None] * t
    envelope = np.exp(growth - growth.max(axis=1, keepdims=True))
    phase = 2 * np.pi * frequency[:, None] * t
    basis = np.column_stack((np.ones(n), (envelope * np.cos(phase)).T, (envelope * np.sin(phase)).T))
    coefficient = np.linalg.lstsq(basis, x.T, rcond=None)[0]
    count = len(alpha)
    return coefficient[1:1 + count] - 1j * coefficient[1 + count:]


def run(probe):
    source = SCRIPT / 'dataset.npz'; output = SCRIPT / 'recovered_modal_shapes.npz'
    if not probe and output.exists():
        raise RuntimeError('Recovered diagnostics are immutable')
    with np.load(source, allow_pickle=False) as z:
        records = json.loads(str(z['records']))
        data = {name: z[name].copy() for name in ('mode_count', 'measured_alpha', 'measured_frequency', 'measured_mode_shapes')}
    original = data['measured_mode_shapes']
    assert not np.iscomplexobj(original), 'Expected explicitly recorded float-cast defect'
    shapes = original.astype(np.complex128)
    last_did = None; checks = 0; maximum_real_relative_error = 0.; maximum_imaginary = 0.
    for ri, record in enumerate(records):
        did = record['design']
        if did != last_did:
            with np.load(SCRIPT / 'shards' / f'{did}.npz', allow_pickle=False) as z:
                raw_rows = z['raw'].copy(); physical_records = json.loads(str(z['records']))
            last_did = did
        pi = next(i for i, r in enumerate(physical_records) if r['replicate'] == record['replicate'] and r['profile'] == record['profile'])
        raw = raw_rows[pi]
        signal = A.M.normalize(raw, A.noise_for(raw, record), cast=True)[0]
        for frame in np.flatnonzero(data['mode_count'][ri] > 0):
            count = int(data['mode_count'][ri, frame]); end = 128 + 64 * int(frame)
            value = recovered(signal[:, end - 128:end], data['measured_alpha'][ri, frame, :count], data['measured_frequency'][ri, frame, :count])
            relative = float(np.linalg.norm(value.real - original[ri, frame, :count]) / max(np.linalg.norm(original[ri, frame, :count]), 1e-30))
            assert relative <= 1e-8, (ri, frame, relative)
            shapes[ri, frame, :count].imag = value.imag
            checks += 1; maximum_real_relative_error = max(maximum_real_relative_error, relative)
            maximum_imaginary = max(maximum_imaginary, float(abs(value.imag).max()))
            if probe:
                print('COMPLEX_DIAGNOSTIC_DEFECT_PROBE', json.dumps(dict(record=ri, frame=int(frame), stored_dtype=str(original.dtype), recovered_maximum_imaginary=maximum_imaginary, real_relative_error=relative)), flush=True)
                return
        if ri % 100 == 0:
            print('COMPLEX_DIAGNOSTIC_RECOVERY', ri, checks, flush=True)
    np.testing.assert_array_equal(shapes.real, original)
    report = dict(source_dataset_sha256=sha(source), recovery_source_sha256=sha(__file__),
        recovered_windows=checks, maximum_real_relative_error=maximum_real_relative_error,
        maximum_recovered_imaginary_component=maximum_imaginary,
        features_and_oracle_rows_unchanged=True, original_real_components_exactly_preserved=True,
        method='Refit cosine/sine coefficients against measured windows using frozen ordered growth/frequency coordinates; restore imaginary components only. No pole re-estimation or feature regeneration.')
    np.savez_compressed(output, measured_mode_shapes=shapes, source_dataset_sha256=report['source_dataset_sha256'], recovery=json.dumps(report))
    report['recovered_diagnostics_sha256'] = sha(output)
    (SCRIPT / 'diagnostic_recovery.json').write_text(json.dumps(report, indent=2) + '\n')
    print('COMPLEX_DIAGNOSTICS_RECOVERED', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--probe', action='store_true')
    with threadpool_limits(limits=1):
        run(parser.parse_args().probe)
