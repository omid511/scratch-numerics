"""First-fold raw-information fit; no evaluation scoring or qualification claim."""
import os
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import shutil
import signal
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != HERE]
spec = importlib.util.spec_from_file_location('_raw_information_smoke', HERE / 'experiment.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
import numpy as np
import torch

PROTOCOL = HERE / 'fit_protocol.json'


def array_sha(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode('ascii'))
    h.update(repr(value.shape).encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def preflight():
    p = json.loads(PROTOCOL.read_text())
    errors = []
    inherited = json.loads(E.PROTOCOL.read_text())['inputs']
    if any(p['inputs'].get(name) != expected for name, expected in inherited.items()):
        errors.append({'path': str(PROTOCOL), 'error': 'inherited input pins mismatch'})
    for name, expected in p['inputs'].items():
        if name in inherited and expected == inherited[name]:
            continue
        path = E.ROOT / name
        if not path.is_file():
            errors.append({'path': name, 'error': 'missing'})
        elif path.stat().st_size != expected['bytes'] or E.sha(path) != expected['sha256']:
            errors.append({'path': name, 'error': 'fingerprint mismatch'})
    smoke, inherited_errors = E.preflight()
    errors.extend(inherited_errors)
    if any(p.get(key) != smoke.get(key) for key in ('split', 'additional_designs', 'evaluation_folds', 'primary_noise')):
        errors.append({'path': str(PROTOCOL), 'error': 'frozen split/noise mismatch'})
    if not errors:
        evidence = json.loads((E.ROOT / p['label_alignment']).read_text())
        receipt = json.loads((E.ROOT / p['smoke_receipt']).read_text())
        if evidence['evidence']['status'] != 'nonheld_label_alignment_pass':
            errors.append({'path': p['label_alignment'], 'error': 'alignment not accepted'})
        if receipt.get('protocol_sha256') != E.sha(E.PROTOCOL) or not receipt.get('held_predictions_absent'):
            errors.append({'path': p['smoke_receipt'], 'error': 'smoke receipt mismatch'})
    return p, smoke, errors


def identity(protocol_sha, arm, proper, valid, held):
    return {'protocol_sha256': protocol_sha, 'arm': arm, 'fold': 0, 'seed': 42,
            'proper_sha256': array_sha(proper), 'validation_sha256': array_sha(valid),
            'held_sha256': array_sha(held)}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise RuntimeError('checkpoint requires CUDA RNG')
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, identity_value, model, optimizer, epoch, best_state, best_epoch, best, stale, history):
    E.R._atomic_torch(path, {'identity': identity_value, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best_state': best_state,
        'best_epoch': best_epoch, 'best': best, 'stale': stale, 'history': history,
        'rng': rng_state()})


def restore_checkpoint(path, expected, model, optimizer):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['identity'] != expected:
        raise RuntimeError('checkpoint identity mismatch')
    if not isinstance(state['epoch'], int) or not 0 <= state['epoch'] <= 2000:
        raise RuntimeError('invalid checkpoint epoch')
    if len(state['history']) != state['epoch'] or not 0 <= state['stale'] <= state['epoch']:
        raise RuntimeError('invalid checkpoint history')
    if state['epoch'] and (state['best_state'] is None or not np.isfinite(state['best'])):
        raise RuntimeError('invalid checkpoint best state')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    set_rng_state(state['rng'])
    return state


def save_json(path, value):
    E.R.save_json(path, value)


def snapshots(directory, protocol):
    target = directory / 'sources'
    target.mkdir()
    paths = [PROTOCOL, HERE / 'fit_runner.py', E.PROTOCOL, HERE / 'experiment.py', HERE / 'model.py',
             E.ROOT / protocol['label_alignment'], E.ROOT / protocol['smoke_receipt']]
    records = {}
    for index, source in enumerate(paths):
        destination = target / f'{index:02d}_{source.name}'
        shutil.copyfile(source, destination)
        digest = E.sha(source)
        if E.sha(destination) != digest:
            raise RuntimeError('source snapshot mismatch')
        records[str(source.relative_to(E.ROOT))] = {'snapshot': str(destination.relative_to(directory)), 'sha256': digest}
    provenance = {'protocol_sha256': E.sha(PROTOCOL), 'sources': records,
                  'fit_inputs': protocol['inputs'], 'smoke_inputs': json.loads(E.PROTOCOL.read_text())['inputs']}
    save_json(directory / 'provenance.json', provenance)
    return provenance


def verify_snapshots(directory):
    provenance = json.loads((directory / 'provenance.json').read_text())
    if provenance['protocol_sha256'] != E.sha(PROTOCOL):
        raise RuntimeError('resume protocol mismatch')
    for name, entry in provenance['sources'].items():
        if E.sha(E.ROOT / name) != entry['sha256'] or E.sha(directory / entry['snapshot']) != entry['sha256']:
            raise RuntimeError(f'resume source mismatch: {name}')
    return provenance


def stream_predictions(model, waves, admissible, held, arm):
    rows, frames = E.R._decision_indices(~held, E.D.FRAME_COUNT)
    prediction = np.full((len(held), E.D.FRAME_COUNT), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for block in E.blocks(len(rows)):
            ri, fi = rows[block], frames[block]
            raw = E.D.raw_windows(waves, admissible, ri, fi, zero_waveform=(arm == 'mask'))
            if not np.isfinite(raw).all():
                raise FloatingPointError('nonfinite inference input')
            values = model(torch.as_tensor(raw, device='cuda', dtype=torch.float32))
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('nonfinite prediction')
            prediction[ri, fi] = values.cpu().numpy()
    if not np.isnan(prediction[held]).all() or not np.isfinite(prediction[~held]).all():
        raise RuntimeError('held or nonheld prediction protection failure')
    return prediction


def fit(protocol, smoke_protocol, arm, resume):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fit fallback')
    directory = HERE / ('fit_' + arm)
    if resume:
        if not directory.is_dir():
            raise FileNotFoundError('resume namespace does not exist')
        if (directory / 'final_receipt.json').exists():
            raise RuntimeError('completed fit cannot be resumed')
        provenance = verify_snapshots(directory)
    else:
        directory.mkdir()
        provenance = snapshots(directory, protocol)
    started = time.monotonic()
    def timeout_handler(signum, frame):
        raise TimeoutError('fit budget exhausted; checkpoint retained')
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, protocol['fit_total_cap_seconds'])
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        truth, waves, admissible, proper_rows, valid_rows, held = E.load_inputs(smoke_protocol)
        smoke_arm = json.loads((E.ROOT / protocol['smoke_receipt']).read_text())['arms'][arm]
        if (len(held) != 2094 or int(proper_rows.sum()) * E.D.FRAME_COUNT != smoke_arm['proper_frames']
                or int(valid_rows.sum()) * E.D.FRAME_COUNT != smoke_arm['validation_frames']
                or int((~held).sum()) * E.D.FRAME_COUNT != smoke_arm['inference_frames']):
            raise RuntimeError('fit population differs from accepted smoke')
        current_identity = identity(E.sha(PROTOCOL), arm, proper_rows, valid_rows, held)
        model = E.RawOnlyModel().to(device='cuda', dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
        proper = E.cache_part(proper_rows, truth, waves, admissible, arm)
        valid = E.cache_part(valid_rows, truth, waves, admissible, arm)
        checkpoint = directory / 'checkpoint.pt'
        if resume:
            state = restore_checkpoint(checkpoint, current_identity, model, optimizer)
            epoch_number, best_state, best_epoch = state['epoch'], state['best_state'], state['best_epoch']
            best, stale, history = state['best'], state['stale'], state['history']
        else:
            epoch_number, best_state, best_epoch, best, stale, history = 0, None, 0, None, 0, []
            save_checkpoint(checkpoint, current_identity, model, optimizer, epoch_number, best_state, best_epoch, best, stale, history)
        save_json(directory / 'status.json', {'status': 'incomplete', 'identity': current_identity, 'epoch': epoch_number})
        while epoch_number < protocol['max_epochs'] and stale < protocol['patience']:
            result = E.epoch(model, optimizer, proper, valid)
            value = result['validation_mae']
            if not np.isfinite(value):
                raise FloatingPointError('nonfinite validation MAE')
            epoch_number += 1
            history.append(result)
            if best is None or value < best:
                best, best_epoch, stale = value, epoch_number, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch_number % protocol['checkpoint_interval'] == 0:
                save_checkpoint(checkpoint, current_identity, model, optimizer, epoch_number, best_state, best_epoch, best, stale, history)
                save_json(directory / 'status.json', {'status': 'incomplete', 'identity': current_identity, 'epoch': epoch_number})
        save_checkpoint(checkpoint, current_identity, model, optimizer, epoch_number, best_state, best_epoch, best, stale, history)
        model.load_state_dict(best_state, strict=True)
        E.R._atomic_torch(directory / 'best_model.pt', {'state': best_state, 'identity': current_identity, 'best_epoch': best_epoch})
        prediction = stream_predictions(model, waves, admissible, held, arm)
        metadata = {'fold': 0, 'seed': 42, 'arm': arm, 'protocol_sha256': E.sha(PROTOCOL),
                    'best_epoch': best_epoch, 'epochs': epoch_number, 'best_validation_mae': best,
                    'identity': current_identity, 'source_and_input_provenance': provenance}
        E.R._atomic_npz(directory / 'final_predictions.npz', predictions=prediction,
                        proper=proper_rows, validation=valid_rows, prediction_rows=~held,
                        protocol_sha256=np.array(E.sha(PROTOCOL)), metadata=np.array(json.dumps(metadata, allow_nan=False)))
        if E.sha(E.ROOT / 'P4_PROGRESS.json') != smoke_protocol['inputs']['P4_PROGRESS.json']['sha256']:
            raise RuntimeError('protected progress ledger changed')
        with np.load(directory / 'final_predictions.npz', allow_pickle=False) as archive:
            saved = archive['predictions']
            if saved.dtype != np.float32 or saved.shape != (len(held), E.D.FRAME_COUNT) or not np.isnan(saved[held]).all() or not np.isfinite(saved[~held]).all():
                raise RuntimeError('saved prediction integrity failure')
            if not np.array_equal(archive['proper'], proper_rows) or not np.array_equal(archive['validation'], valid_rows) or not np.array_equal(archive['prediction_rows'], ~held):
                raise RuntimeError('saved population integrity failure')
            if str(archive['protocol_sha256']) != current_identity['protocol_sha256'] or json.loads(str(archive['metadata'])) != metadata:
                raise RuntimeError('saved metadata integrity failure')
        saved_checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
        saved_best = torch.load(directory / 'best_model.pt', map_location='cpu', weights_only=False)
        if (saved_checkpoint['identity'] != current_identity or saved_checkpoint['epoch'] != epoch_number
                or saved_best['identity'] != current_identity or saved_best['best_epoch'] != best_epoch
                or any(not torch.equal(saved_best['state'][key], value) for key, value in best_state.items())):
            raise RuntimeError('saved checkpoint or best model integrity failure')
        artifacts = {name: E.sha(directory / name) for name in ('checkpoint.pt', 'best_model.pt', 'final_predictions.npz')}
        receipt = {'status': 'fit_complete', 'protocol_sha256': current_identity['protocol_sha256'],
                   'predictions_sha256': artifacts['final_predictions.npz'], 'model_sha256': artifacts['best_model.pt'],
                   'identity': current_identity, 'elapsed_seconds': time.monotonic()-started,
                   'epochs': epoch_number, 'history': history, 'best_epoch': best_epoch, 'best_validation_mae': best,
                   'parameter_count': sum(p.numel() for p in model.parameters()), 'provenance': provenance,
                   'artifacts': artifacts, 'scientific_gate_credit': False}
        save_json(directory / 'final_receipt.json', receipt)
        save_json(directory / 'status.json', {'status': 'fit_complete', 'identity': current_identity, 'epoch': epoch_number})
        print(json.dumps({'status': 'fit_complete', 'receipt': str(directory / 'final_receipt.json')}))
    except BaseException as exc:
        try:
            progress_unchanged = E.sha(E.ROOT / 'P4_PROGRESS.json') == smoke_protocol['inputs']['P4_PROGRESS.json']['sha256']
        except OSError:
            progress_unchanged = False
        save_json(directory / 'status.json', {'status': 'incomplete' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) and progress_unchanged else 'error',
                  'error_type': type(exc).__name__, 'message': str(exc), 'elapsed_seconds': time.monotonic()-started,
                  'protected_progress_unchanged': progress_unchanged})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def selfcheck():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    inputs = torch.tensor([[1., 2.], [3., 1.], [-1., 4.]], dtype=torch.float32)
    targets = torch.tensor([[1.], [0.], [2.]], dtype=torch.float32)
    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        (model(inputs)-targets).square().mean().backward()
        optimizer.step()
    model = torch.nn.Linear(2, 1).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
    step(model, optimizer)
    ident = identity('selfcheck', 'waveform', np.array([True, False]), np.array([False, True]), np.array([False, False]))
    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'checkpoint.pt'
        save_checkpoint(path, ident, model, optimizer, 1, best, 1, .2, 0, [{'validation_mae': .2}])
        step(model, optimizer)
        expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
        restored = torch.nn.Linear(2, 1).float()
        resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=.003, weight_decay=.0001)
        state = restore_checkpoint(path, ident, restored, resumed_optimizer)
        assert state['epoch'] == 1
        step(restored, resumed_optimizer)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        for key, changed in (('arm', 'mask'), ('protocol_sha256', 'other'), ('fold', 1),
                             ('seed', 43), ('proper_sha256', 'other'),
                             ('validation_sha256', 'other'), ('held_sha256', 'other')):
            wrong = dict(ident, **{key: changed})
            try:
                restore_checkpoint(path, wrong, restored, resumed_optimizer)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'mismatched {key} accepted')
    print('CPU_CHECKPOINT_SELFCHECK_OK save restore identity optimizer_continuation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'selfcheck', 'fit'), default='preflight')
    parser.add_argument('--arm', choices=('waveform', 'mask'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'selfcheck':
        if args.arm or args.resume:
            parser.error('selfcheck takes no arm or resume')
        selfcheck()
        return
    if args.phase == 'fit' and not args.arm:
        parser.error('fit requires --arm')
    if args.phase != 'fit' and (args.arm or args.resume):
        parser.error('arm and resume require --phase fit')
    protocol, smoke_protocol, errors = preflight()
    if errors:
        print(json.dumps({'status': 'preflight_failed', 'errors': errors}, indent=2))
        raise SystemExit(1)
    if args.phase == 'preflight':
        print(json.dumps({'status': 'preflight_ok', 'protocol_sha256': E.sha(PROTOCOL)}))
    else:
        fit(protocol, smoke_protocol, args.arm, args.resume)


if __name__ == '__main__':
    main()
