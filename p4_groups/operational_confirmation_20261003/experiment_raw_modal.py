"""Fixed raw+modal versus zero-waveform control on preregistered TRAIN folds.

Scope: ONE fixed R (raw+modal) versus M (zero-waveform control) comparison, 18
matched fits (3 folds x 2 arms x 3 seeds). No new acquisition, modal extraction,
search, qualification relaxation, or held/blind predictions. Parent owns
OUT/protocol.json + OUT/preflight.json, integration, floors, docs.

Architecture (identical R/M, reset every decision, no dropout/hidden carry):
  modal GRU 176->32 over 3 slots (lags 4,2,0); raw GRU 18->32 over 384 samples;
  concat 32+32=64 -> Linear(64,16) -> ReLU -> Linear(16,1) scalar.
Raw 18 channels = 16 causally calibrated waveforms + admissibility + relative
time (-0.383..0s fixed both arms). M zeroes ONLY waveform 16; masks/time/modal/
flag/age identical. Same seed+architecture between arms; only wave16 differs.

Loss: row-mean MSE over every native frame of proper rows (no target
whitening), Adam lr .003 wd .0001 max 2000 patience 100, checkpoint on original
clean/IID1 validation native all-frame MAE. Full-batch objective accumulated
over fixed 2048-decision microbatches with ONE optimizer step per epoch, exact
global-count weighting, fixed order, no shuffling/AMP/tuning.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(SCRIPT)]
import raw_modal_data as D
import score_additive_spectral as S
import experiment_multivel_history as H

SOURCE = SCRIPT / 'support_modal_20261009_v2'
OUT = SCRIPT / 'raw_modal_20261010'
BOUNDED_OUT = SCRIPT / 'bounded_gru_20261009_v2'
ARMS = ('raw', 'mask')
SEEDS = (42, 43, 44)
LAGS = (4, 2, 0)
MICROBATCH_DEFAULT = 2048
LR, WD, MAX_EPOCHS, PATIENCE = 0.003, 0.0001, 2000, 100
FITTING_NOISES = ('clean', 'iid1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
_CONTRACT_CACHE = None
_SOURCE_SIGNATURE_CACHE = None
_FOLD_CACHE_KEY = None
_FOLD_CACHE = None


def _array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(repr(tuple(array.shape)).encode('ascii'))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _source_signature():
    global _SOURCE_SIGNATURE_CACHE
    if _SOURCE_SIGNATURE_CACHE is not None:
        return dict(_SOURCE_SIGNATURE_CACHE)
    result = {}
    for path in (SOURCE / 'dataset.npz', SOURCE / 'protocol.json', OUT / 'acquisition.json',
                 Path(getattr(D, '__file__', ''))):
        if path and path.is_file():
            key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            result[key] = sha(path)
    _SOURCE_SIGNATURE_CACHE = result
    return dict(result)


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
    temporary.replace(path)


def _cpu_state(state):
    return {key: value.detach().cpu() for key, value in state.items()}


def _release_fold_cache():
    global _FOLD_CACHE_KEY, _FOLD_CACHE
    _FOLD_CACHE_KEY = None
    _FOLD_CACHE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RawModalModel(torch.nn.Module):
    """Reset-every-decision modal+raw GRU; no retained state, no dropout."""

    def __init__(self):
        super().__init__()
        self.modal_gru = torch.nn.GRU(176, 32, batch_first=True)
        self.raw_gru = torch.nn.GRU(18, 32, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(64, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, modal, raw):
        if modal.ndim != 3 or tuple(modal.shape[1:]) != (3, 176):
            raise ValueError(f'Expected modal [B,3,176], got {tuple(modal.shape)}')
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (384, 18):
            raise ValueError(f'Expected raw [B,384,18], got {tuple(raw.shape)}')
        if modal.shape[0] != raw.shape[0]:
            raise ValueError('Modal/raw batch sizes differ')
        # No h0 is supplied intentionally: torch creates fresh zero hidden
        # state for every decision call.
        m_resp, _ = self.modal_gru(modal)
        r_resp, _ = self.raw_gru(raw)
        feat = torch.cat((m_resp[:, -1], r_resp[:, -1]), dim=-1)
        return self.head(feat)[..., 0]


def make_model(device):
    return RawModalModel().to(device=device, dtype=torch.float32)


def mechanism_smoke():
    torch.manual_seed(0)
    model = make_model('cpu').eval()
    modal = torch.arange(2 * 3 * 176, dtype=torch.float32).reshape(2, 3, 176) / 1000.0
    raw = torch.arange(2 * 384 * 18, dtype=torch.float32).reshape(2, 384, 18) / 5000.0
    with torch.no_grad():
        out = model(modal, raw)
        assert out.shape == (2,) and out.dtype == torch.float32 and torch.isfinite(out).all()
        # Reset every decision: batched vs split vs reordered agree.
        a = model(modal[:1], raw[:1])
        b = model(modal[1:], raw[1:])
        assert torch.allclose(out, torch.cat((a, b)), atol=1e-6, rtol=0)
        rev = model(modal.flip(0), raw.flip(0))
        assert torch.allclose(rev, out.flip(0), atol=1e-6, rtol=0)
        # No hidden carry: repeat first decision after different input.
        _ = model(modal[1:], raw[1:])
        assert torch.allclose(model(modal[:1], raw[:1]), a, atol=0, rtol=0)
        # Zero-waveform control: zeroing wave16 only changes output, keeps shape.
        masked = raw.clone()
        masked[..., :16] = 0
        out_mask = model(modal, masked)
        assert out_mask.shape == (2,) and torch.isfinite(out_mask).all()
        assert not torch.allclose(out, out_mask, atol=1e-6, rtol=0)
        assert torch.allclose(model(modal, masked), out_mask, atol=0, rtol=0)
        # Mask/time channels preserved matter only via input; zeroing them differs.
        tampered = masked.clone()
        tampered[..., 16:] = 0
        assert not torch.allclose(model(modal, tampered), out_mask, atol=1e-6, rtol=0)
    params = sum(p.numel() for p in model.parameters())
    return dict(modal_gru_in=176, raw_gru_in=18, raw_steps=384, slots=list(LAGS),
        head='Linear64x16-ReLU-Linear16x1', reset_every_decision=True, no_dropout=True,
        shape_ok=True, reorder_invariant=True, zero_wave16_changes_output=True, parameters=params)


def _verify_hashes(mapping):
    for relative, digest in (mapping or {}).items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f'Pinned prerequisite changed: {relative}')


def contract():
    global _CONTRACT_CACHE
    proto_path, pre_path = OUT / 'protocol.json', OUT / 'preflight.json'
    if not proto_path.exists():
        raise RuntimeError('Missing OUT/protocol.json; parent must freeze it first')
    if not pre_path.exists():
        raise RuntimeError('Missing OUT/preflight.json; parent must pass preflight first')
    proto_sha, pre_sha = sha(proto_path), sha(pre_path)
    if _CONTRACT_CACHE is not None and _CONTRACT_CACHE[:2] == (proto_sha, pre_sha):
        return _CONTRACT_CACHE[2:]
    protocol = json.loads(proto_path.read_text())
    preflight = json.loads(pre_path.read_text())
    if preflight.get('passed') is not True:
        raise RuntimeError('Preflight did not pass; refusing all fits')
    if preflight.get('protocol_sha256') != proto_sha:
        raise RuntimeError('Preflight protocol hash does not match protocol.json')
    _verify_hashes(protocol.get('inputs'))
    if not protocol.get('inputs'):
        raise RuntimeError('Protocol carries no immutable input hashes; refusing')
    _verify_hashes(preflight.get('inputs'))
    for key in ('split', 'additional_designs', 'evaluation_folds', 'training'):
        if key not in protocol:
            raise RuntimeError(f'Protocol missing required key: {key}')
    if tuple(protocol.get('arms', ())) != ARMS:
        raise RuntimeError('Protocol arms must be exactly raw/mask')
    if (int(protocol.get('raw_width', 0)), int(protocol.get('raw_samples', 0))) != (18, 384):
        raise RuntimeError('Protocol raw shape must be 18 channels by 384 samples')
    if list(protocol.get('lags', ())) != list(LAGS):
        raise RuntimeError('Protocol modal lag slots changed')
    if len(protocol['split'].get('train', ())) != 4 or len(protocol['split'].get('validation', ())) != 2:
        raise RuntimeError('Protocol must retain the original 4 train/2 validation designs')
    if len(protocol['split'].get('held', ())) != 2:
        raise RuntimeError('Protocol held membership changed')
    if len(protocol['additional_designs']) != 12 or len(set(protocol['additional_designs'])) != 12:
        raise RuntimeError('Protocol must carry twelve additional designs')
    folds = protocol['evaluation_folds']
    if len(folds) != 3 or any(len(f) != 4 for f in folds):
        raise RuntimeError('Protocol must define 3 evaluation folds of 4 designs (18 fits)')
    if set().union(*(set(f) for f in folds)) != set(protocol['additional_designs']):
        raise RuntimeError('Protocol folds do not partition additional designs')
    split_sets = [set(protocol['split'][name]) for name in ('train', 'validation', 'held')]
    if any(left & right for index, left in enumerate(split_sets) for right in split_sets[index + 1:]):
        raise RuntimeError('Protocol split memberships overlap')
    if set(protocol['additional_designs']) & set().union(*split_sets):
        raise RuntimeError('Additional designs overlap source/validation/held split')
    tr = protocol['training']
    if list(tr.get('seeds', [])) != list(SEEDS):
        raise RuntimeError('Protocol seeds must be [42,43,44]')
    if (tr.get('maximum_epochs'), tr.get('patience')) != (MAX_EPOCHS, PATIENCE):
        raise RuntimeError('Protocol epochs/patience must be 2000/100')
    if float(tr.get('learning_rate', -1)) != LR or float(tr.get('weight_decay', -1)) != WD:
        raise RuntimeError('Protocol lr/wd must be 0.003/0.0001')
    if tuple(tr.get('fitting_noise', ())) != FITTING_NOISES:
        raise RuntimeError('Protocol fitting_noise must be clean+iid1')
    micro = int(tr.get('microbatch', tr.get('microbatch_decisions', MICROBATCH_DEFAULT)))
    if micro != MICROBATCH_DEFAULT:
        raise RuntimeError('Protocol microbatch must be fixed at 2048')
    if D.OUT.resolve() != OUT.resolve():
        raise RuntimeError(f'Data module OUT mismatch: {D.OUT} vs {OUT}')
    _CONTRACT_CACHE = (proto_sha, pre_sha, protocol, preflight, micro)
    return protocol, preflight, micro


def _decision_indices(rows, n_frames):
    indices = np.flatnonzero(rows).astype(np.int64, copy=False)
    if indices.size == 0 or n_frames <= 0:
        raise RuntimeError('Empty decision population')
    return np.repeat(indices, n_frames), np.tile(np.arange(n_frames, dtype=np.int64), indices.size)


def _iter_blocks(count):
    for start in range(0, int(count), MICROBATCH_DEFAULT):
        yield slice(start, min(start + MICROBATCH_DEFAULT, int(count)))


def _cached_raw(waveforms, admissible, row_ids, frame_ids):
    result = torch.empty((row_ids.size, 384, 18), device='cuda', dtype=torch.float32)
    for block in _iter_blocks(row_ids.size):
        values = np.ascontiguousarray(D.raw_windows(
            waveforms, admissible, row_ids[block], frame_ids[block], zero_waveform=False), dtype=np.float32)
        if values.shape != (block.stop - block.start, 384, 18) or not np.isfinite(values).all():
            raise FloatingPointError('Malformed/nonfinite raw cache')
        result[block] = torch.as_tensor(values, device='cuda', dtype=torch.float32)
    return result


def _prepare_fold_cache(fold, proper, validation, modal, truth, waveforms, admissible):
    global _FOLD_CACHE_KEY, _FOLD_CACHE
    key = (int(fold), id(modal), _array_sha(np.asarray(proper, dtype=np.uint8)),
           _array_sha(np.asarray(validation, dtype=np.uint8)), tuple(modal.shape), tuple(truth.shape))
    if _FOLD_CACHE is not None and _FOLD_CACHE_KEY == key:
        return _FOLD_CACHE
    _release_fold_cache()
    n_frames = truth.shape[1]
    proper_rows, proper_frames = _decision_indices(proper, n_frames)
    valid_rows, valid_frames = _decision_indices(validation, n_frames)
    cache = {
        'proper_rows': proper_rows,
        'proper_frames': proper_frames,
        'valid_rows': valid_rows,
        'valid_frames': valid_frames,
        'proper_modal': torch.as_tensor(np.ascontiguousarray(modal[proper_rows, proper_frames], dtype=np.float32),
                                        device='cuda', dtype=torch.float32),
        'valid_modal': torch.as_tensor(np.ascontiguousarray(modal[valid_rows, valid_frames], dtype=np.float32),
                                       device='cuda', dtype=torch.float32),
        'proper_target': torch.as_tensor(np.ascontiguousarray(truth[proper_rows, proper_frames], dtype=np.float32),
                                         device='cuda', dtype=torch.float32),
        'valid_target': torch.as_tensor(np.ascontiguousarray(truth[valid_rows, valid_frames], dtype=np.float32),
                                        device='cuda', dtype=torch.float32),
    }
    cache['proper_raw'] = _cached_raw(waveforms, admissible, proper_rows, proper_frames)
    cache['valid_raw'] = _cached_raw(waveforms, admissible, valid_rows, valid_frames)
    for name in ('proper_modal', 'valid_modal', 'proper_target', 'valid_target', 'proper_raw', 'valid_raw'):
        if not bool(torch.isfinite(cache[name]).all()):
            raise FloatingPointError(f'Nonfinite {name} cache')
    _FOLD_CACHE_KEY, _FOLD_CACHE = key, cache
    return cache


def _raw_slice(raw, block, zero_wave):
    if not zero_wave:
        return raw[block]
    masked = raw[block].clone()
    masked[..., :16] = 0
    return masked


def _cached_train(model, optimizer, cache, zero_wave):
    count = int(cache['proper_target'].shape[0])
    optimizer.zero_grad(set_to_none=True)
    train_sum = 0.0
    for block in _iter_blocks(count):
        pred = model(cache['proper_modal'][block], _raw_slice(cache['proper_raw'], block, zero_wave))
        residual = pred - cache['proper_target'][block]
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError('Nonfinite cached training residual')
        (residual.square().sum() / count).backward()
        train_sum += float(residual.square().sum().detach().cpu())
    optimizer.step()
    return train_sum / count


def _cached_validation(model, cache, zero_wave):
    count = int(cache['valid_target'].shape[0])
    absolute = 0.0
    with torch.no_grad():
        for block in _iter_blocks(count):
            pred = model(cache['valid_modal'][block], _raw_slice(cache['valid_raw'], block, zero_wave))
            if not bool(torch.isfinite(pred).all()):
                raise FloatingPointError('Nonfinite cached validation prediction')
            absolute += float(torch.abs(pred - cache['valid_target'][block]).sum().detach().cpu())
    return absolute / count




def _fingerprint_payload(protocol_sha, preflight_sha, fold, arm, seed, proper, validation,
                         prediction_rows, scaling, data_shape, modal_shape, raw_shapes, micro):
    mean, scale, active = scaling
    return {
        'protocol_sha256': protocol_sha,
        'preflight_sha256': preflight_sha,
        'fold': int(fold),
        'arm': str(arm),
        'seed': int(seed),
        'architecture': {
            'modal_gru': [176, 32], 'modal_slots': 3,
            'raw_gru': [18, 32], 'raw_samples': 384,
            'head': [64, 16, 1], 'hidden_reset': 'every_forward', 'dropout': False,
        },
        'optimizer': {
            'name': 'Adam', 'learning_rate': LR, 'weight_decay': WD,
            'maximum_epochs': MAX_EPOCHS, 'patience': PATIENCE,
            'microbatch_decisions': int(micro), 'optimizer_steps_per_epoch': 1,
            'loss': 'full_proper_frame_sum_divided_by_global_count',
            'validation': 'full_native_all_frame_MAE',
        },
        'proper_sha256': _array_sha(np.asarray(proper, dtype=np.uint8)),
        'validation_sha256': _array_sha(np.asarray(validation, dtype=np.uint8)),
        'prediction_rows_sha256': _array_sha(np.asarray(prediction_rows, dtype=np.uint8)),
        'scaling': {
            'mean_sha256': _array_sha(mean), 'scale_sha256': _array_sha(scale),
            'active_sha256': _array_sha(active),
        },
        'data_shape': {
            'truth': list(data_shape), 'modal': list(modal_shape),
            'waveforms': list(raw_shapes[0]), 'admissible': list(raw_shapes[1]),
        },
        'source': _source_signature(),
        'dtype': 'float32_cuda_deterministic',
    }


def _stream_predictions(model, modal, waveforms, admissible, truth_shape, rows, zero_wave):
    n_rows, n_frames = truth_shape
    row_ids, frame_ids = _decision_indices(rows, n_frames)
    point = np.full(truth_shape, np.nan, dtype=np.float32)
    with torch.no_grad():
        for block in _iter_blocks(row_ids.size):
            ri, fi = row_ids[block], frame_ids[block]
            modal_batch = np.ascontiguousarray(modal[ri, fi], dtype=np.float32)
            raw_batch = np.ascontiguousarray(
                D.raw_windows(waveforms, admissible, ri, fi, zero_waveform=zero_wave), dtype=np.float32)
            if not np.isfinite(modal_batch).all() or not np.isfinite(raw_batch).all():
                raise FloatingPointError('Nonfinite streamed prediction input')
            values = model(
                torch.as_tensor(modal_batch, device='cuda', dtype=torch.float32),
                torch.as_tensor(raw_batch, device='cuda', dtype=torch.float32),
            )
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('Nonfinite streamed prediction')
            point[ri, fi] = values.detach().cpu().numpy().astype(np.float32, copy=False)
    return point


def train_fit(data, protocol, fold, arm, seed, proper, validation, modal, scaling, waveforms, admissible):
    if arm not in ARMS:
        raise ValueError(arm)
    if seed not in SEEDS:
        raise ValueError(seed)
    fold = int(fold)
    if fold not in range(3):
        raise ValueError(fold)
    frozen, _, micro = contract()
    if json.dumps(protocol, sort_keys=True, separators=(',', ':')) != json.dumps(
            frozen, sort_keys=True, separators=(',', ':')):
        raise RuntimeError('Passed protocol differs from frozen protocol.json')
    records = data['records']
    truth = np.asarray(data['truth'])
    modal = np.asarray(modal, dtype=np.float32)
    proper = np.asarray(proper, dtype=bool)
    validation = np.asarray(validation, dtype=bool)
    waveforms = np.asarray(waveforms)
    admissible = np.asarray(admissible)
    n_rows, n_frames = truth.shape
    if truth.ndim != 2 or len(records) != n_rows:
        raise ValueError('Truth/records shape mismatch')
    if proper.shape != (n_rows,) or validation.shape != (n_rows,):
        raise ValueError('Proper/validation masks shape mismatch')
    if np.any(proper & validation) or not proper.any() or not validation.any():
        raise ValueError('Proper/validation populations are empty or overlap')
    if modal.shape != (n_rows, n_frames, 3, 176):
        raise ValueError(f'Invalid modal shape {modal.shape}')
    if waveforms.shape != (n_rows, 16, 8001) or admissible.shape != (n_rows, 8001):
        raise ValueError('Invalid measured raw array shape')
    if not np.isfinite(truth).all() or not np.isfinite(modal).all():
        raise FloatingPointError('Nonfinite truth/modal input')
    held = np.asarray([r['design'] in frozen['split']['held'] for r in records], dtype=bool)
    prediction_rows = ~held
    if not prediction_rows.any():
        raise ValueError('No non-held prediction rows')
    mean, scale, active = (np.asarray(scaling[0]), np.asarray(scaling[1]),
                           np.asarray(scaling[2], dtype=bool))
    scaling = (mean, scale, active)
    payload = _fingerprint_payload(
        sha(OUT / 'protocol.json'), sha(OUT / 'preflight.json'), fold, arm, seed,
        proper, validation, prediction_rows, scaling, truth.shape, modal.shape,
        (waveforms.shape, admissible.shape), micro)
    fingerprint = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    stem = f'fold{fold}_{arm}_seed{seed}'
    result_path, state_path, log_path = (
        OUT / f'{stem}.npz', OUT / f'{stem}.pt', OUT / f'{stem}.json')
    present = [path.exists() for path in (result_path, state_path, log_path)]
    if list(OUT.glob(f'{stem}*.tmp')):
        raise RuntimeError(f'Partial temporary artifact remains for {stem}')
    if any(present):
        if not all(present):
            raise RuntimeError(f'Partial artifact for {stem}; refusing overwrite')
        with np.load(result_path, allow_pickle=False) as archive:
            if str(archive['fingerprint']) != fingerprint:
                raise RuntimeError(f'Artifact fingerprint mismatch for {stem}')
            predictions = np.asarray(archive['predictions'], dtype=np.float32)
            if predictions.shape != truth.shape:
                raise RuntimeError(f'Artifact shape mismatch for {stem}')
            if not np.array_equal(archive['proper'], proper) or not np.array_equal(
                    archive['validation'], validation) or not np.array_equal(
                    archive['prediction_rows'], prediction_rows):
                raise RuntimeError(f'Artifact membership mismatch for {stem}')
            if not np.array_equal(archive['mean'], mean) or not np.array_equal(
                    archive['scale'], scale) or not np.array_equal(
                    archive['active'], active):
                raise RuntimeError(f'Artifact scaling mismatch for {stem}')
            if not np.isnan(predictions[held]).all() or not np.isfinite(predictions[prediction_rows]).all():
                raise RuntimeError(f'Artifact protected/nonfinite prediction mismatch for {stem}')
            if str(archive['protocol_sha256']) != payload['protocol_sha256'] or \
                    str(archive['preflight_sha256']) != payload['preflight_sha256']:
                raise RuntimeError(f'Artifact contract hash mismatch for {stem}')
            metadata = json.loads(str(archive['metadata']))
            if metadata.get('fingerprint') != fingerprint:
                raise RuntimeError(f'Artifact metadata fingerprint mismatch for {stem}')
        checkpoint = torch.load(state_path, map_location='cpu')
        if checkpoint.get('fingerprint') != fingerprint or \
                checkpoint.get('protocol_sha256') != payload['protocol_sha256'] or \
                checkpoint.get('preflight_sha256') != payload['preflight_sha256']:
            raise RuntimeError(f'Checkpoint fingerprint mismatch for {stem}')
        log = json.loads(log_path.read_text())
        if log.get('status') != 'complete' or log.get('fingerprint') != fingerprint:
            raise RuntimeError(f'Fit log fingerprint mismatch for {stem}')
        return predictions, metadata

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda')
    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    cache = _prepare_fold_cache(fold, proper, validation, modal, truth, waveforms, admissible)
    total_proper = int(cache['proper_target'].shape[0])
    total_valid = int(cache['valid_target'].shape[0])
    zero_wave = arm == 'mask'
    best, stale, best_epoch, best_state, history = np.inf, 0, 0, None, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_mse = _cached_train(model, optimizer, cache, zero_wave)
        model.eval()
        validation_mae = _cached_validation(model, cache, zero_wave)
        if not np.isfinite(train_mse) or not np.isfinite(validation_mae):
            raise FloatingPointError(f'Nonfinite raw-modal metric for {stem}')
        history.append(dict(epoch=epoch + 1, train_mse=float(train_mse),
                            validation_mae=float(validation_mae)))
        if validation_mae < best:
            best, stale, best_epoch = float(validation_mae), 0, epoch + 1
            best_state = _cpu_state(model.state_dict())
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f'No finite raw-modal checkpoint selected for {stem}')
    model.load_state_dict(best_state)
    model.eval()
    point = _stream_predictions(model, modal, waveforms, admissible, truth.shape,
                                prediction_rows, zero_wave)
    if not np.isnan(point[held]).all() or not np.isfinite(point[prediction_rows]).all():
        raise FloatingPointError(f'Invalid held/nonfinite predictions for {stem}')
    metadata = dict(
        fold=fold, arm=arm, seed=seed, epochs=len(history), best_epoch=best_epoch,
        validation_mae=float(best), history=history,
        parameters=sum(p.numel() for p in model.parameters()),
        training_records=int(proper.sum()), validation_records=int(validation.sum()),
        proper_frame_count=total_proper, validation_frame_count=total_valid,
        fitting_designs=sorted({r['design'] for r, keep in zip(records, proper) if keep}),
        microbatch_decisions=micro, optimizer_steps_per_epoch=1,
        zero_waveform_only_wave16=bool(zero_wave), matched_init=f'same-{seed}-both-arms',
        prediction_rows=int(prediction_rows.sum()), held_rows=int(held.sum()),
        fingerprint=fingerprint, protocol_sha256=payload['protocol_sha256'],
        preflight_sha256=payload['preflight_sha256'])
    checkpoint = dict(
        state=best_state, arm=arm, seed=seed, fold=fold, fingerprint=fingerprint,
        fingerprint_payload=payload, protocol_sha256=payload['protocol_sha256'],
        preflight_sha256=payload['preflight_sha256'])
    _atomic_torch(state_path, checkpoint)
    _atomic_npz(
        result_path, predictions=point, mean=mean, scale=scale, active=active,
        proper=proper, validation=validation, prediction_rows=prediction_rows,
        metadata=json.dumps(metadata, allow_nan=False), fingerprint=fingerprint,
        protocol_sha256=np.asarray(payload['protocol_sha256']),
        preflight_sha256=np.asarray(payload['preflight_sha256']))
    save_json(log_path, dict(status='complete', fit_id=stem, fold=fold, arm=arm, seed=seed,
                             epochs=len(history), best_epoch=best_epoch,
                             validation_mae=float(best), fingerprint=fingerprint,
                             protocol_sha256=payload['protocol_sha256'],
                             preflight_sha256=payload['preflight_sha256']))
    print('RAW_MODAL_FIT', fold, arm, seed, best_epoch, best, flush=True)
    return point, metadata


def train():
    protocol, _, micro = contract()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = D.load_data()
    records = data['records']
    truth = np.asarray(data['truth'])
    waveforms, admissible = D.load_measured()
    held = np.asarray([r['design'] in protocol['split']['held'] for r in records], dtype=bool)
    prediction_rows = ~held
    validation_global = np.asarray([
        r['design'] in protocol['split']['validation'] and r['noise'] in FITTING_NOISES
        for r in records
    ], dtype=bool)
    if set(r['design'] for r, keep in zip(records, validation_global) if keep) != set(protocol['split']['validation']):
        raise RuntimeError('Validation population is not exactly validation clean/IID1')
    points, fits, fit_ids = [], [], []
    for fold, excluded in enumerate(protocol['evaluation_folds']):
        fitting_ids = set(protocol['split']['train']) | (set(protocol['additional_designs']) - set(excluded))
        if fitting_ids & (set(excluded) | set(protocol['split']['validation']) | set(protocol['split']['held'])):
            raise RuntimeError(f'Fold {fold} fitting population overlaps evaluation/validation/held')
        proper = np.asarray([
            r['design'] in fitting_ids and r['noise'] in FITTING_NOISES for r in records
        ], dtype=bool)
        if set(r['design'] for r, keep in zip(records, proper) if keep) != fitting_ids:
            raise RuntimeError(f'Fold {fold} proper population is not exact')
        modal, scaling = D.modal_inputs(data, proper)
        modal = np.asarray(modal, dtype=np.float32)
        if modal.shape != (*truth.shape, 3, 176):
            raise RuntimeError(f'Invalid modal input shape {modal.shape}')
        for arm in ARMS:
            for seed in SEEDS:
                point, metadata = train_fit(
                    data, protocol, fold, arm, seed, proper, validation_global,
                    modal, scaling, waveforms, admissible)
                points.append(point)
                fits.append(metadata)
                fit_ids.append(f'fold{fold}_{arm}_seed{seed}')
    if len(fits) != 18:
        raise RuntimeError(f'Expected 18 fits, got {len(fits)}')
    global_predictions = np.stack(points).astype(np.float32, copy=False)
    global_rows = np.broadcast_to(prediction_rows, (len(fits), prediction_rows.size)).copy()
    if global_predictions.shape != (18, *truth.shape):
        raise RuntimeError(f'Invalid global prediction shape {global_predictions.shape}')
    if not np.isnan(global_predictions[:, held]).all() or not np.isfinite(global_predictions[:, prediction_rows]).all():
        raise RuntimeError('Global predictions violate held-row protection')
    predictions_path = OUT / 'predictions.npz'
    if predictions_path.exists():
        with np.load(predictions_path, allow_pickle=False) as archive:
            if str(archive['protocol_sha256']) != sha(OUT / 'protocol.json') or \
                    str(archive['preflight_sha256']) != sha(OUT / 'preflight.json'):
                raise RuntimeError('Global prediction contract hash mismatch')
            if [str(value) for value in archive['fit_ids'].tolist()] != fit_ids:
                raise RuntimeError('Global prediction fit order mismatch')
            old_predictions = np.asarray(archive['predictions'], dtype=np.float32)
            old_rows = np.asarray(archive['prediction_rows'], dtype=bool)
            if old_predictions.shape != global_predictions.shape or not np.array_equal(old_rows, global_rows):
                raise RuntimeError('Global prediction shape/mask mismatch')
            if not np.array_equal(old_predictions, global_predictions, equal_nan=True):
                raise RuntimeError('Global predictions disagree with fit artifacts')
    else:
        _atomic_npz(
            predictions_path, predictions=global_predictions, prediction_rows=global_rows,
            fit_ids=np.asarray(fit_ids, dtype='U64'),
            folds=np.asarray([item['fold'] for item in fits], dtype=np.int64),
            arms=np.asarray([item['arm'] for item in fits], dtype='U16'),
            seeds=np.asarray([item['seed'] for item in fits], dtype=np.int64),
            protocol_sha256=np.asarray(sha(OUT / 'protocol.json')),
            preflight_sha256=np.asarray(sha(OUT / 'preflight.json')))
    _release_fold_cache()
    training = dict(
        fit_count=len(fits), expected_fit_count=18, fit_ids=fit_ids, fits=fits,
        prediction_shape=list(global_predictions.shape),
        prediction_rows_count=int(prediction_rows.sum()), held_rows_count=int(held.sum()),
        microbatch_decisions=micro, optimizer_steps_per_epoch=1,
        arms=list(ARMS), seeds=list(SEEDS), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, deterministic=True,
        protocol_sha256=sha(OUT / 'protocol.json'),
        preflight_sha256=sha(OUT / 'preflight.json'),
        predictions_sha256=sha(predictions_path), dataset_sha256=sha(SOURCE / 'dataset.npz'),
        objective='full native all-frame MSE; global 2048-decision accumulation; one Adam step per epoch',
        validation='full native all-frame MAE on clean/IID1 validation designs',
        no_held_predictions=True)
    training_path = OUT / 'training.json'
    if training_path.exists():
        prior = json.loads(training_path.read_text())
        for key in ('fit_count', 'fit_ids', 'prediction_shape', 'protocol_sha256', 'preflight_sha256',
                    'predictions_sha256', 'dataset_sha256'):
            if prior.get(key) != training[key]:
                raise RuntimeError(f'training.json fingerprint mismatch for {key}')
    save_json(training_path, training)
    print('RAW_MODAL_TRAINING_COMPLETE', len(fits), flush=True)


def score():
    protocol, _, _ = contract()
    if (OUT / 'verification.json').exists():
        raise RuntimeError('Verified scores are immutable')
    data = D.load_data()
    records = data['records']
    strata = H.strata(data)
    strata['farther_stable15'] = data['truth'] >= 0.15
    _, admissible = D.load_measured()
    prefix_counts = np.zeros((len(records), admissible.shape[1]+1), dtype=np.int32)
    np.cumsum(admissible, axis=1, dtype=np.int32, out=prefix_counts[:,1:])
    ends = 128+64*np.arange(data['truth'].shape[1])
    sample_counts = prefix_counts[:,ends]-prefix_counts[:,np.maximum(0,ends-384)]
    del prefix_counts
    raw_strata = dict(raw_no_samples=sample_counts == 0,
        raw_partial_window=(sample_counts > 0) & (sample_counts < 384),
        raw_full_window=sample_counts == 384,
        raw_present_no_modal=(sample_counts > 0) & ~data['observed'])
    strata.update(raw_strata)
    strata.update({f'boundary_{name}': mask & strata['boundary'] for name, mask in raw_strata.items()})
    frozen_arms = ('content_gru', 'ordered_concat', 'unordered_pool')
    score_arms = ('raw', 'mask', *frozen_arms, 'framewise_reference')
    metrics, paired, artifact_hashes = {}, {}, {}
    designs = np.array([r['design'] for r in records])
    for fold, excluded in enumerate(protocol['evaluation_folds']):
        points = []
        for arm in ('raw', 'mask'):
            rows = []
            for seed in SEEDS:
                path = OUT / f'fold{fold}_{arm}_seed{seed}.npz'
                artifact_hashes[path.name] = sha(path)
                with np.load(path, allow_pickle=False) as z:
                    rows.append(z['predictions'].copy())
            points.append(np.stack(rows))
        for arm in frozen_arms:
            rows = []
            for seed in SEEDS:
                path = BOUNDED_OUT / f'fold{fold}_{arm}_seed{seed}.npz'
                artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
                with np.load(path, allow_pickle=False) as z:
                    rows.append(z['predictions'].copy())
            points.append(np.stack(rows))
        ref = []
        for seed in SEEDS:
            path = SOURCE / f'fold{fold}_measured_seed{seed}.npz'
            artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                ref.append(z['predictions'].copy())
        points.append(np.stack(ref))
        for noise in sorted({r['noise'] for r in records if r['design'] in excluded}):
            condition = f'fold{fold}/evaluation/{noise}'
            pop = np.array([r['design'] in excluded and r['noise'] == noise for r in records])[:, None]
            masks = dict(strata)
            for profile in sorted({r['profile'] for r in records}):
                pm = np.array([r['profile'] == profile for r in records])[:, None]
                masks[f'profile_{profile}'] = pm & strata['all']
                masks[f'boundary_profile_{profile}'] = pm & strata['boundary']
            metrics[condition] = {arm: {name: S.summary(point, data['truth'], pop & mask, designs)
                for name, mask in masks.items()} for arm, point in zip(score_arms, points)}
            paired[condition] = {}
            for mask_name in ('all', 'boundary', 'distinct_regimes', 'boundary_current', 'profile_retreat',
                    'farther_stable15', *raw_strata, *(f'boundary_{name}' for name in raw_strata)):
                if mask_name not in metrics[condition]['raw']:
                    continue
                base = metrics[condition]['raw'][mask_name]
                paired[condition][mask_name] = {}
                for control in score_arms[1:]:
                    refm = metrics[condition][control][mask_name]
                    delta = {did: (np.array(g['mae_by_seed']) - refm['per_design'][did]['mae_by_seed']).tolist()
                        for did, g in base['per_design'].items() if did in refm['per_design']}
                    paired[condition][mask_name][control] = dict(delta_by_design_seed=delta,
                        mean_delta=float(np.mean(list(delta.values()))) if delta else None,
                        improved_every_design_seed=bool(delta and np.all(np.array(list(delta.values())) < 0)))
    save_json(OUT / 'results.json', dict(scope='Fixed raw+modal vs zero-waveform control; TRAIN folds only, never held',
        arms=list(score_arms), seeds=list(SEEDS), metrics=metrics, paired=paired,
        artifact_hashes=artifact_hashes, protocol_sha256=sha(OUT / 'protocol.json'),
        preflight_sha256=sha(OUT / 'preflight.json'), operational_predictor_changed=False,
        gaps='Native unqualified extrapolation; no stale/current substitution; held rows unscored',
        interpretation='Grouped TRAIN folds are development evidence, not blind validation. 384 samples span383ms; reset every decision.'))
    print('RAW_MODAL_SCORED', len(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'train', 'score'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == 'smoke':
            print('RAW_MODAL_MECHANISM_SMOKE', json.dumps(mechanism_smoke()))
        else:
            globals()[args.phase]()
