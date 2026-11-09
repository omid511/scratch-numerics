"""Frozen forecast-gate-bypass receipt; measured inputs only, no fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(SCRIPT)]
import experiment_support_modal_v2 as E
from cuda_backend import estimate_batch, tensor
OUT = SCRIPT / 'gate_bypass_receipt_20261010'
SOURCE = E.OUT
REASONS = ('cached_accepted', 'command', 'nonfinite_waveform', 'cap',
           'forecast_pass_originally_unobserved', 'full_no_modes', 'full_invalid',
           'eligible_finite_fail', 'eligible_prefix_no_modes', 'eligible_undefined_variance',
           'solver_failure')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def timed_modes(windows):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = estimate_batch(windows)
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def valid_mode(mode):
    return bool(len(mode['alpha']) and all(np.isfinite(mode[k]).all() for k in
                ('alpha', 'frequency', 'energy_fraction', 'mode_shapes'))
                and np.isfinite(mode['relative_fit_error'])
                and np.isfinite(mode['usable_fraction']))


def extract(design, smoke=False):
    assert torch.cuda.is_available(), 'Original CUDA extractor required; no CPU fallback'
    destination = OUT / ('smoke' if smoke else '')
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f'replay_{design}.npz'
    if path.exists():
        raise RuntimeError(f'Immutable replay already exists: {path}')
    protocol_path = OUT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for relative, digest in protocol['inputs'].items():
        assert sha(ROOT / relative) == digest, f'Pinned source changed: {relative}'
    source_path = SOURCE / 'dataset.npz'
    with np.load(source_path, allow_pickle=False) as z:
        records_all = json.loads(str(z['records']))
        indices = np.array([i for i, r in enumerate(records_all) if r['design'] == design], dtype=np.int64)
        if smoke:
            indices = indices[:3]
        assert len(indices), design
        records = [records_all[i] for i in indices]
        cached = z['features'][indices].copy()
        observed = z['observed'][indices].copy()
        times = z['times'].copy()
        reference_modes = {k: z[k][indices].copy() for k in
            ('mode_count', 'measured_alpha', 'measured_frequency', 'measured_energy_fraction')}
    shard_path = E.SHARDS / f'{design}.npz'
    with np.load(shard_path, allow_pickle=False) as z:
        assert str(z['protocol_sha256']) == sha(SOURCE / 'protocol.json')
        assert str(z['design']) == design
        raw, command = z['raw'].copy(), z['command'].copy()
        physical_records = json.loads(str(z['records']))
    lookup = {(int(r['replicate']), r['profile']): i for i, r in enumerate(physical_records)}
    payload, _ = E.G.inputs()
    descriptors = {'pairs': payload['pairs']}
    del payload
    engine = SimpleNamespace(ridge=E.G.O.B.PhysicsFeatureRidge())
    ends = np.rint(times * 1000).astype(int) + 1
    assert np.array_equal(ends, np.arange(128, raw.shape[-1] + 1, 64))
    shape = observed.shape
    retained, rebuilt = cached.copy(), np.zeros_like(cached)
    eligible = np.zeros(shape, bool)
    finite_only = np.zeros(shape, bool)
    reason = np.full(shape, 10, np.int8)
    forecast_type = np.zeros(shape, np.int8)
    skills_all = np.full(shape, -np.inf)
    mode_count = np.zeros(shape, np.int16)
    poles = {k: np.full((*shape, 8), np.nan) for k in ('alpha', 'frequency', 'energy')}
    row_receipts, failures = [], []
    started = time.perf_counter()
    for row, record in enumerate(records):
        physical_index = lookup[(int(record['replicate']), record['profile'])]
        physical = raw[physical_index]
        signal, _, _ = E.M.normalize(physical, E.noise_for(physical, record), cast=True)
        candidate_indices, windows = [], []
        for frame, end in enumerate(ends):
            window = signal[:, end - 128:end]
            if command[physical_index, end - 128:end].any():
                reason[row, frame] = 1
            elif not np.isfinite(window).all():
                reason[row, frame] = 2
            elif (abs(window) >= 50).any():
                reason[row, frame] = 3
            else:
                candidate_indices.append(frame)
                windows.append(window)
        prefix_seconds = forecast_seconds = accepted_seconds = rejected_seconds = assembly_seconds = 0.0
        prefix_modes = []
        skills = np.full(len(windows), -np.inf)
        types = np.full(len(windows), 5, np.int8)
        if windows:
            try:
                prefix_modes, prefix_seconds = timed_modes([w[:, :64] for w in windows])
                torch.cuda.synchronize()
                clock = time.perf_counter()
                skills = E.P.forecast_skills(windows, prefix_modes, 64, 64)
                torch.cuda.synchronize()
                forecast_seconds = time.perf_counter() - clock
                # Exactly the existing suffix-variance criterion; not a new quality gate.
                x = tensor(np.stack(windows))
                normalizer = abs(x[:, :, :64]).amax(dim=(1, 2), keepdim=True)
                x = x / normalizer
                suffix = x[:, :, 64:].transpose(1, 2)
                centered = ((suffix - suffix.mean(dim=1, keepdim=True)) ** 2).sum(dim=(1, 2))
                energy = (suffix ** 2).sum(dim=(1, 2))
                variance_ok = (centered > torch.finfo(torch.float64).eps * energy).cpu().numpy()
                for i, (mode, skill) in enumerate(zip(prefix_modes, skills)):
                    if not len(mode['alpha']):
                        types[i] = 2
                    elif not valid_mode(mode):
                        types[i] = 5
                    elif np.isfinite(skill):
                        types[i] = 4 if skill >= .5 else 1
                    elif not variance_ok[i]:
                        types[i] = 3
                    else:
                        types[i] = 5
                        failures.append(dict(row=int(indices[row]), frame=int(candidate_indices[i]),
                            stage='forecast', error='Unexpected nonfinite skill despite modes and nondegenerate suffix'))
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                failures.append(dict(row=int(indices[row]), stage='prefix_or_forecast', error=repr(error)))
            forecast_type[row, candidate_indices] = types
            skills_all[row, candidate_indices] = skills
        # Accepted replay is diagnostic only, even when refreshed gate differs.
        accepted_positions = [i for i, frame in enumerate(candidate_indices) if observed[row, frame]]
        rejected_positions = [i for i, frame in enumerate(candidate_indices)
                              if not observed[row, frame] and types[i] in (1, 2, 3)]
        for positions, accepted in ((accepted_positions, True), (rejected_positions, False)):
            full_modes = []
            elapsed = 0.0
            if positions:
                try:
                    full_modes, elapsed = timed_modes([windows[i] for i in positions])
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    failures.append(dict(row=int(indices[row]), stage='accepted_full' if accepted else 'rejected_full',
                        frames=[int(candidate_indices[i]) for i in positions], error=repr(error)))
                    reason[row, [candidate_indices[i] for i in positions]] = 10
            if accepted:
                accepted_seconds += elapsed
            else:
                rejected_seconds += elapsed
            for i, mode in zip(positions, full_modes):
                frame = candidate_indices[i]
                mode_count[row, frame] = len(mode['alpha'])
                if not len(mode['alpha']):
                    reason[row, frame] = 5
                    continue
                if not valid_mode(mode):
                    reason[row, frame] = 6
                    continue
                count = len(mode['alpha'])
                if count > 8:
                    raise RuntimeError('Frozen eight-mode extractor contract changed')
                for dest, key in (('alpha', 'alpha'), ('frequency', 'frequency'), ('energy', 'energy_fraction')):
                    poles[dest][row, frame, :count] = mode[key]
                clock = time.perf_counter()
                try:
                    q = E.M.features(windows[i], mode, descriptors, engine)
                    if q.shape != (174,) or not np.isfinite(q).all():
                        reason[row, frame] = 6
                        continue
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    reason[row, frame] = 6
                    failures.append(dict(row=int(indices[row]), frame=int(frame), stage='features', error=repr(error)))
                    continue
                finally:
                    assembly_seconds += time.perf_counter() - clock
                if accepted:
                    rebuilt[row, frame] = q
                    reason[row, frame] = 0
                else:
                    retained[row, frame] = q
                    eligible[row, frame] = True
                    finite_only[row, frame] = types[i] == 1
                    reason[row, frame] = {1: 7, 2: 8, 3: 9}[int(types[i])]
        for i, frame in enumerate(candidate_indices):
            if not observed[row, frame] and types[i] == 4:
                reason[row, frame] = 4
        row_receipts.append(dict(row=int(indices[row]), design=design, profile=record['profile'],
            replicate=int(record['replicate']), noise=record['noise'], source_kind=record['source_kind'],
            frames=int(shape[1]), cached_accepted=int(observed[row].sum()),
            eligible=int(eligible[row].sum()), finite_only_eligible=int(finite_only[row].sum()),
            reasons={name: int((reason[row] == code).sum()) for code, name in enumerate(REASONS)},
            forecast_types={str(code): int((forecast_type[row] == code).sum()) for code in range(6)},
            accepted_replay_gate_mismatches=int((observed[row] & (forecast_type[row] != 4)).sum()),
            accepted_full_trials=len(accepted_positions), rejected_full_trials=len(rejected_positions),
            prefix_seconds=prefix_seconds, forecast_seconds=forecast_seconds,
            accepted_full_seconds=accepted_seconds, additional_rejected_full_seconds=rejected_seconds,
            feature_assembly_seconds=assembly_seconds))
        print('GATE_REPLAY_ROW', design, int(indices[row]), record['noise'], record['profile'],
              int(eligible[row].sum()), flush=True)
    assert not (eligible & observed).any()
    assert np.array_equal(retained[observed].view(np.uint8), cached[observed].view(np.uint8))
    assert np.array_equal(retained[~observed & ~eligible], cached[~observed & ~eligible])
    receipt = dict(scope='SMOKE ONLY' if smoke else 'Population replay, no fits', design=design,
        records=row_receipts, failures=failures, runtime_seconds=time.perf_counter() - started,
        device=torch.cuda.get_device_name(), torch=torch.__version__, fitting_started=False,
        cached_accepted_preserved=True, qualification_unchanged=True,
        source_dataset_sha256=sha(source_path), shard_sha256=sha(shard_path),
        protocol_sha256=sha(protocol_path), extractor_sha256=sha(__file__),
        reasons=list(REASONS), source_protocol_sha256=sha(SOURCE / 'protocol.json'))
    temporary = destination / f'.replay_{design}.npz'
    np.savez_compressed(temporary, row_indices=indices, retained_features=retained,
        eligible=eligible, finite_only_eligible=finite_only, replay_features=rebuilt,
        forecast_type=forecast_type, reason=reason, forecast_skill=skills_all, mode_count=mode_count,
        replay_alpha=poles['alpha'], replay_frequency=poles['frequency'], replay_energy=poles['energy'],
        reference_mode_count=reference_modes['mode_count'], reference_alpha=reference_modes['measured_alpha'],
        reference_frequency=reference_modes['measured_frequency'], reference_energy=reference_modes['measured_energy_fraction'],
        records=np.array(json.dumps(records)), protocol_sha256=np.array(sha(protocol_path)))
    temporary.rename(path)
    receipt['replay_sha256'] = sha(path)
    save_json(destination / f'replay_{design}.json', receipt)
    print('GATE_REPLAY_COMPLETE', design, len(indices), int(eligible.sum()), receipt['runtime_seconds'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.design == 'all':
            assert not args.smoke
            with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
                designs = sorted({r['design'] for r in json.loads(str(z['records']))})
            for design in designs:
                extract(design)
        else:
            extract(args.design, args.smoke)


if __name__ == '__main__':
    main()
