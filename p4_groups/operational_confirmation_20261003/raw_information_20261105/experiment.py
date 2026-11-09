"""Raw-information development probe: provenance preflight and CUDA smoke only."""
import os
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != HERE]
sys.path[:0] = [str(ROOT / 'src'), str(BASE)]
import numpy as np
import torch
import raw_modal_data as D
import experiment_raw_modal as R
sys.path.insert(0, str(HERE))
from model import RawOnlyModel

PROTOCOL = HERE / 'protocol.json'
RECOVERY = BASE / 'consumer_recovery_20261103'
MANIFEST = RECOVERY / 'lossless_carriers' / 'manifest_only.npz'
NORMALIZED = RECOVERY / 'full_consumer_v1' / 'normalized'
SHARDS = RECOVERY / 'full_consumer_v1' / 'candidate_shards'
DATASET = BASE / 'support_modal_20261009_v2' / 'dataset.npz'
MICRO = 512


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def preflight():
    protocol = json.loads(PROTOCOL.read_text())
    errors = []
    for name, expected in protocol['inputs'].items():
        path = ROOT / name
        if not path.is_file():
            errors.append({'path': name, 'error': 'missing'})
            continue
        size = path.stat().st_size
        if 'bytes' in expected and size != expected['bytes']:
            errors.append({'path': name, 'error': 'size mismatch', 'actual_bytes': size})
        elif sha(path) != expected['sha256']:
            errors.append({'path': name, 'error': 'sha256 mismatch'})
    if not SHARDS.is_dir():
        errors.append({'path': str(SHARDS.relative_to(ROOT)), 'error': 'missing directory'})
    if errors:
        return protocol, errors
    old_dataset, old_out, old_shards = D.DATASET, D.OUT, D.E.SHARDS
    try:
        D.DATASET, D.OUT, D.E.SHARDS = MANIFEST, NORMALIZED, SHARDS
        waves, admissible = D.load_measured()
        if waves.flags.writeable or admissible.flags.writeable:
            raise RuntimeError('native measured maps are not read-only')
        del waves, admissible
    except Exception as exc:
        errors.append({'path': str(NORMALIZED.relative_to(ROOT)), 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        D.DATASET, D.OUT, D.E.SHARDS = old_dataset, old_out, old_shards
    return protocol, errors


def load_inputs(protocol):
    old_dataset, old_out, old_shards = D.DATASET, D.OUT, D.E.SHARDS
    try:
        D.DATASET, D.OUT, D.E.SHARDS = MANIFEST, NORMALIZED, SHARDS
        records, times, records_text, _ = D._load_manifest()
        waves, admissible = D.load_measured()
    finally:
        D.DATASET, D.OUT, D.E.SHARDS = old_dataset, old_out, old_shards
    with np.load(DATASET, allow_pickle=False) as archive:
        if str(archive['records']) != records_text or not np.array_equal(archive['times'], times):
            raise RuntimeError('label and raw manifest records/times mismatch')
        truth = archive['truth'].copy()
    if truth.dtype != np.float64 or truth.shape != (len(records), D.FRAME_COUNT):
        raise RuntimeError('unexpected truth array shape or dtype')
    designs = np.array([r['design'] for r in records])
    noises = np.array([r['noise'] for r in records])
    held = np.isin(designs, protocol['split']['held'])
    truth[held] = np.nan
    fitting = set(protocol['split']['train']) | (set(protocol['additional_designs']) - set(protocol['evaluation_folds'][0]))
    if fitting & (set(protocol['evaluation_folds'][0]) | set(protocol['split']['validation']) | set(protocol['split']['held'])):
        raise RuntimeError('overlapping design populations')
    proper = np.isin(designs, list(fitting)) & np.isin(noises, ('clean', 'iid1'))
    valid = np.isin(designs, protocol['split']['validation']) & np.isin(noises, ('clean', 'iid1'))
    if not proper.any() or not valid.any() or np.any((proper | valid) & held):
        raise RuntimeError('invalid training population')
    truth[~(proper | valid)] = np.nan
    return truth, waves, admissible, proper, valid, held


def blocks(count):
    for start in range(0, count, MICRO):
        yield slice(start, min(start + MICRO, count))


def cache_part(rows, truth, waves, admissible, arm):
    row_ids, frames = R._decision_indices(rows, D.FRAME_COUNT)
    raw = R._cached_raw(waves, admissible, row_ids, frames)
    if arm == 'mask':
        raw[:, :, :16] = 0
    target = torch.as_tensor(np.ascontiguousarray(truth[row_ids, frames], dtype=np.float32), device='cuda')
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError('nonfinite cache')
    return raw, target


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def epoch(model, optimizer, proper, valid):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = sync_time()
    forward_seconds = backward_seconds = 0.0
    for block in blocks(len(proper[1])):
        mark = sync_time()
        point = model(proper[0][block])
        if not bool(torch.isfinite(point).all()):
            raise FloatingPointError('nonfinite training output')
        loss = (point - proper[1][block]).square().sum() / len(proper[1])
        forward_seconds += sync_time() - mark
        mark = sync_time()
        loss.backward()
        backward_seconds += sync_time() - mark
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('nonfinite training gradient')
    optimizer.step()
    train_seconds = sync_time() - start
    model.eval()
    start = sync_time()
    error = torch.zeros((), device='cuda', dtype=torch.float32)
    with torch.no_grad():
        for block in blocks(len(valid[1])):
            point = model(valid[0][block])
            if not bool(torch.isfinite(point).all()):
                raise FloatingPointError('nonfinite validation output')
            error += (point - valid[1][block]).abs().sum()
    validation_seconds = sync_time() - start
    return {'train_seconds': train_seconds, 'forward_seconds': forward_seconds,
            'backward_seconds': backward_seconds, 'validation_seconds': validation_seconds,
            'validation_mae': float(error / len(valid[1]))}


def predict(model, waves, admissible, held, arm):
    row_ids, frames = R._decision_indices(~held, D.FRAME_COUNT)
    output = np.full((len(held), D.FRAME_COUNT), np.nan, dtype=np.float32)
    start = sync_time()
    model.eval()
    with torch.no_grad():
        for block in blocks(len(row_ids)):
            ri, fi = row_ids[block], frames[block]
            raw = D.raw_windows(waves, admissible, ri, fi, zero_waveform=(arm == 'mask'))
            if not np.isfinite(raw).all():
                raise FloatingPointError('nonfinite streaming inputs')
            point = model(torch.as_tensor(raw, device='cuda', dtype=torch.float32))
            if not bool(torch.isfinite(point).all()):
                raise FloatingPointError('nonfinite streaming output')
            output[ri, fi] = point.cpu().numpy()
    seconds = sync_time() - start
    if not np.isnan(output[held]).all() or not np.isfinite(output[~held]).all():
        raise RuntimeError('prediction protection failed')
    return seconds, len(row_ids)


def smoke(protocol):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU smoke fallback')
    receipt_path = HERE / 'smoke_receipt.json'
    if receipt_path.exists() or (HERE / 'smoke_checkpoint.json').exists():
        raise FileExistsError('refusing existing smoke attempt')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    start = sync_time()
    truth, waves, admissible, proper_rows, valid_rows, held = load_inputs(protocol)
    receipt = {'protocol_sha256': sha(PROTOCOL), 'device': torch.cuda.get_device_name(),
               'torch_version': torch.__version__, 'setup_seconds': sync_time() - start,
               'arms': {}, 'held_predictions_absent': True,
               'scope': 'two-epoch resource smoke only; no development evaluation or scientific gate credit'}
    atomic_json(HERE / 'smoke_checkpoint.json', receipt)
    for arm in ('waveform', 'mask'):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        start = sync_time()
        model = RawOnlyModel().to(device='cuda', dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
        proper = cache_part(proper_rows, truth, waves, admissible, arm)
        valid = cache_part(valid_rows, truth, waves, admissible, arm)
        setup_seconds = sync_time() - start
        epochs = [epoch(model, optimizer, proper, valid) for _ in range(2)]
        inference_seconds, inference_frames = predict(model, waves, admissible, held, arm)
        receipt['arms'][arm] = {'setup_seconds': setup_seconds, 'epochs': epochs,
            'proper_frames': len(proper[1]), 'validation_frames': len(valid[1]),
            'inference_frames': inference_frames, 'inference_seconds': inference_seconds,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved()}
        atomic_json(HERE / 'smoke_checkpoint.json', receipt)
        del model, optimizer, proper, valid
        torch.cuda.empty_cache()
    atomic_json(receipt_path, receipt)
    print(json.dumps({'status': 'smoke_complete', 'receipt': str(receipt_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'selfcheck', 'smoke'), default='preflight')
    args = parser.parse_args()
    if args.phase == 'selfcheck':
        from selfcheck import main as selfcheck
        selfcheck()
        return
    protocol, errors = preflight()
    if errors:
        print(json.dumps({'status': 'preflight_failed', 'errors': errors}, indent=2))
        raise SystemExit(1)
    if args.phase == 'preflight':
        print(json.dumps({'status': 'preflight_ok', 'protocol_sha256': sha(PROTOCOL)}))
    else:
        smoke(protocol)


if __name__ == '__main__':
    main()
