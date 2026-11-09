"""Independent finite-input ambiguity bound and CUDA gate parity; no fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(SCRIPT)]
import multipoint_preflight_20261009 as P
import multipoint_numerics_20261009 as N
import diagnostic_matrix as M
import gpu_execution as G
OUT = P.OUT


def fixture():
    d, fp = G.inputs()
    ridge = P.PhysicsFeatureRidge()
    p = json.loads((OUT / 'protocol.json').read_text())
    windows = []
    expected = []
    records = []
    for did in p['designs']:
        _, plate = M.context(d, fp, dict(design=did, replicate=0))
        # Fixed grid members cover quiet, low/high near-critical and clipped cases.
        for endpoint in (900, 1240, 2080, 2240):
            for replicate in p['replicates']:
                acquisitions = {arm: N.acquire(plate, endpoint, p['delta_mps'], arm,
                    replicate, coefficient_step=.0005) for arm in p['arms']}
                noises, _ = M.noises(acquisitions['S'].response,
                    np.random.SeedSequence([20261009, int(did[1:]), endpoint, replicate]))
                for arm in p['arms']:
                    for noise in p['noise_views']:
                        signal, _, _ = M.normalize(acquisitions[arm].response, noises[noise])
                        _, good, _, _, _ = P.history(signal, d, ridge)
                        for slot, window in enumerate(N.PLATEAU_SLICES.values()):
                            windows.append(signal[:, window])
                            expected.append(good[slot])
                            records.append(dict(design=did, endpoint_mps=endpoint,
                                replicate=replicate, arm=arm, noise=noise, slot=slot))
    path = OUT / 'gate_fixture.npz'
    if path.exists():
        raise RuntimeError('immutable gate fixture exists')
    np.savez_compressed(path, windows=np.stack(windows), expected=np.array(expected),
                        records=json.dumps(records), protocol_sha256=P.sha(OUT / 'protocol.json'))
    print('MULTIPOINT_GATE_FIXTURE', len(windows), flush=True)


def cuda():
    import torch
    import pulse_aware_probe as original
    from cuda_backend import estimate_batch
    if not torch.cuda.is_available():
        raise RuntimeError('original CUDA gate required; no CPU fallback')
    path = OUT / 'gate_fixture.npz'
    with np.load(path) as data:
        windows = data['windows']
        expected = data['expected']
    actual = np.zeros(len(windows), dtype=bool)
    eligible = np.flatnonzero(np.isfinite(windows).all(axis=(1, 2)) &
                             ~(abs(windows) >= 50).any(axis=(1, 2)))
    for start in range(0, len(eligible), 64):
        selected = eligible[start:start + 64]
        signals = list(windows[selected])
        prefix = estimate_batch([signal[:, :64] for signal in signals])
        skills = original.forecast_skills(signals, prefix, 64, 64)
        chosen = np.flatnonzero(skills >= .5)
        if len(chosen):
            modes = estimate_batch([signals[index] for index in chosen])
            for index, mode in zip(chosen, modes):
                actual[selected[index]] = bool(len(mode['alpha']))
    discrepancies = np.flatnonzero(actual != expected)
    result = dict(windows=len(windows), eligible_windows=len(eligible),
        accepted_cpu=int(expected.sum()), accepted_cuda=int(actual.sum()),
        gate_discrepancies=discrepancies.tolist(), fixture_sha256=P.sha(path),
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        limitation='Fixed1440-window parity sample, not full-dataset feature/inference replay.',
        passed=not len(discrepancies))
    P.save('cuda_gate_verification.json', result)
    if len(discrepancies):
        raise RuntimeError(f'original CUDA qualification disagrees: {result}')
    print('ORIGINAL_CUDA_GATE_PARITY', json.dumps(result), flush=True)


def independent():
    with np.load(OUT / 'dataset.npz') as data:
        features = data['features']
        observed = data['observed']
        age = data['age']
        assert str(data['protocol_sha256']) == P.sha(OUT / 'protocol.json')
        labels = data['truth']
        records = json.loads(str(data['records']))
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert {row['design'] for row in records} == set(protocol['designs'])
    assert {row['design'] for row in records} == {'D000007', 'D000033', 'D000068', 'D000087'}
    assert {row['arm'] for row in records} == {'S', 'M', 'R'}
    assert {row['endpoint_mps'] for row in records} == set(protocol['endpoints_mps'])
    assert features.shape == (8160, 3, 174)
    missing = ~observed.any(axis=1)
    assert np.array_equal(features[missing], np.zeros_like(features[missing]))
    assert np.array_equal(age[missing], np.broadcast_to([.127, .287, .447], age[missing].shape))
    original = json.loads((OUT / 'accounting.json').read_text())
    bounds = {}
    for noise in ('clean', 'iid1', 'correlated1'):
        population = np.array([row['noise'] == noise for row in records])
        # Each design has exactly408 rows here; direct enumeration is independent
        # of the accounting helper's weighted-median implementation.
        sizes = [sum(row['noise'] == noise and row['design'] == design for row in records)
                 for design in sorted({row['design'] for row in records})]
        assert len(set(sizes)) == 1
        ambiguous = population & missing
        values = labels[ambiguous]
        candidates = np.unique(values)
        brute = float(np.min(abs(values[:, None] - candidates[None, :]).sum(axis=0)) / population.sum())
        expected = original['summaries'][f'{noise}/shared_all_arms']['target_regions']['all']['mae_floor']
        assert abs(brute - expected) < 1e-12
        bounds[noise] = dict(floor=brute, histories=int(population.sum()),
            all_missing=int(ambiguous.sum()), missing_margin_range=[float(values.min()), float(values.max())],
            boundary_missing=int((ambiguous & (labels > -.02) & (labels < .05)).sum()),
            prior_acquisition_range_missing=int((ambiguous & (labels >= -.03) & (labels <= .22)).sum()))
    ledger = P.sha(ROOT / 'P4_PROGRESS.json')
    baseline = json.loads((SCRIPT / 'current_context_20261009' / 'decision.json').read_text())
    expected_ledger = baseline['progress_ledger_sha256']
    assert ledger == expected_ledger, 'protected research progress ledger changed'
    paths = [OUT / 'protocol.json', OUT / 'dataset.npz', OUT / 'accounting.json', OUT / 'numerics.json',
             OUT / 'contrast.json', SCRIPT / 'multipoint_preflight_20261009.py',
             SCRIPT / 'multipoint_numerics_20261009.py', Path(__file__)]
    result = dict(histories=len(labels), all_missing_feature_and_age_equality=True,
        independently_enumerated_bounds=bounds, progress_ledger_sha256=ledger,
        artifact_sha256={str(path.relative_to(ROOT)): P.sha(path) for path in paths},
        no_held_or_reserved_designs=True, no_fits=True,
        limitation='Proper-training simulator-only evidence; ambiguous margins all lie outside prior-.03 to+.22 acquisition population.')
    P.save('verification.json', result)
    print('MULTIPOINT_INDEPENDENT_VERIFICATION', json.dumps(bounds), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('fixture', 'cuda', 'independent'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        dict(fixture=fixture, cuda=cuda, independent=independent)[args.phase]()


if __name__ == '__main__':
    main()
