"""Audit existing measured samples only; no acquisition, modal extraction or fits."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(SCRIPT)]
import experiment_support_modal_v2 as E
from analyze_gate_bypass_receipt import save_json, sha
REPLAY = SCRIPT / 'gate_bypass_receipt_20261010'
OUT = REPLAY / 'raw_response_partition'


def span_masks(command, finite, cap, ends):
    """Original 384-sample trailing aperture, clipped only by early warmup."""
    starts = np.maximum(0, ends - 384)
    native = ~command & finite & ~cap
    first_cap = int(np.flatnonzero(cap)[0]) if cap.any() else len(cap)
    # Conservative pre-cap sensitivity; do not invent a permanent source gate.
    precap = native & (np.arange(len(cap)) < first_cap)
    counts, lengths = [], []
    for valid in (native, precap):
        count = np.zeros(len(ends), np.int16)
        longest = np.zeros(len(ends), np.int16)
        for frame, (start, end) in enumerate(zip(starts, ends)):
            selected = valid[start:end]
            count[frame] = int(selected.sum())
            transitions = np.diff(np.r_[False, selected, False].astype(np.int8))
            runs = np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)
            longest[frame] = int(runs.max()) if len(runs) else 0
        counts.append(count)
        lengths.append(longest)
    return counts, lengths, first_cap


def smoke():
    ends = np.array([128, 192, 256, 320, 384, 448])
    command = np.zeros(448, bool)
    finite = np.ones(448, bool)
    cap = np.zeros(448, bool)
    command[64:384] = True
    cap[400] = True
    counts, lengths, first = span_masks(command, finite, cap, ends)
    assert first == 400
    # Early warmup contributes no fabricated negative-time samples.
    assert lengths[0][0] == 64 and counts[0][0] == 64
    assert lengths[0][4] == 64 and counts[0][4] == 64
    # Capped sample never enters either mask; post-cap recovery is sensitivity only.
    assert lengths[0][5] == 47 and counts[0][5] == 63
    assert lengths[1][5] == 16 and counts[1][5] == 16
    command[:] = True
    counts, lengths, _ = span_masks(command, finite, cap, ends)
    assert np.array_equal(counts, np.zeros((2, len(ends)), np.int16))
    assert np.array_equal(lengths, np.zeros((2, len(ends)), np.int16))
    print('RAW_ADMISSIBILITY_SMOKE_PASSED', flush=True)


def audit(design):
    with np.load(E.OUT / 'dataset.npz', allow_pickle=False) as z:
        all_records = json.loads(str(z['records']))
        times = z['times'].copy()
    with np.load(REPLAY / f'replay_{design}.npz', allow_pickle=False) as z:
        indices = z['row_indices'].copy()
        reasons = z['reason'].copy()
    records = [all_records[i] for i in indices]
    receipt = json.loads((REPLAY / f'replay_{design}.json').read_text())
    shard = E.SHARDS / f'{design}.npz'
    assert sha(shard) == receipt['shard_sha256'], 'Immutable measured source changed'
    with np.load(shard, allow_pickle=False) as z:
        raw = z['raw'].copy()
        commands = z['command'].copy()
        physical_records = json.loads(str(z['records']))
    lookup = {(int(r['replicate']), r['profile']): i for i, r in enumerate(physical_records)}
    assert times.ndim == 1
    ends = np.rint(times * 1000).astype(int) + 1
    assert np.array_equal(ends, np.arange(128, raw.shape[-1] + 1, 64))
    width = (raw.shape[-1] + 7) // 8
    packed = {k: np.empty((len(records), width), np.uint8) for k in ('command', 'finite', 'cap')}
    counts = np.empty((len(records), 2, len(ends)), np.int16)
    longest = np.empty_like(counts)
    first_caps = np.empty(len(records), np.int32)
    offsets, scales, row_reports = [], [], []
    for row, record in enumerate(records):
        physical_index = lookup[(int(record['replicate']), record['profile'])]
        waveform = raw[physical_index]
        signal, offset, scale = E.M.normalize(waveform, E.noise_for(waveform, record), cast=True)
        command = commands[physical_index]
        finite = np.isfinite(signal).all(axis=0)
        cap = (abs(signal) >= 50).any(axis=0)
        native_windows = np.array([not command[end-128:end].any() and finite[end-128:end].all()
                                   and not cap[end-128:end].any() for end in ends])
        # This is verification of newly recovered masks against saved exclusion metadata.
        assert np.array_equal(native_windows, ~np.isin(reasons[row], (1, 2, 3)))
        for name, values in (('command', command), ('finite', finite), ('cap', cap)):
            packed[name][row] = np.packbits(values, bitorder='little')
        c, length, first = span_masks(command, finite, cap, ends)
        counts[row], longest[row], first_caps[row] = c, length, first
        offsets.append(offset)
        scales.append(scale)
        row_reports.append(dict(row=int(indices[row]), design=design, noise=record['noise'],
            profile=record['profile'], replicate=int(record['replicate']),
            first_cap_sample=first if first < len(cap) else None,
            native_full_windows=int(native_windows.sum()),
            native_span_categories={name: int(mask.sum()) for name, mask in
                (('full128', length[0] >= 128), ('shorter_only', (length[0] > 0) & (length[0] < 128)), ('no_samples', length[0] == 0))},
            precap_span_categories={name: int(mask.sum()) for name, mask in
                (('full128', length[1] >= 128), ('shorter_only', (length[1] > 0) & (length[1] < 128)), ('no_samples', length[1] == 0))}))
    OUT.mkdir(exist_ok=True)
    destination = OUT / f'masks_{design}.npz'
    np.savez_compressed(destination, row_indices=indices, **packed, sample_count=np.array(raw.shape[-1]),
        ends=ends, sample_counts=counts, longest_runs=longest, first_cap=first_caps,
        offsets=np.stack(offsets), scales=np.stack(scales), records=np.array(json.dumps(records)))
    save_json(OUT / f'masks_{design}.json', dict(records=row_reports,
        source_shard_sha256=receipt['shard_sha256'], replay_sha256=receipt['replay_sha256'],
        mask_sha256=sha(destination), audit_source_sha256=sha(__file__),
        scope='Existing waveform/masks; no labels, new acquisition, modal extraction or fits',
        handling='Native excludes command/nonfinite/capped samples only. Global first-cap truncation is conservative sensitivity, NOT an invented permanent source exclusion.'))
    print('RAW_RESPONSE_MASKS_COMPLETE', design, len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--design')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.design == 'all':
        with np.load(E.OUT / 'dataset.npz', allow_pickle=False) as z:
            designs = sorted({r['design'] for r in json.loads(str(z['records']))})
        for design in designs:
            audit(design)
    else:
        assert args.design
        audit(args.design)
