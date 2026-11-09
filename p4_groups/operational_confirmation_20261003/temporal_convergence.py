"""Extend the frozen temporal comparison using cached inputs, never new waveforms."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import temporal_probe as T
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling

SCRIPT = Path(__file__).resolve().parent
BASE = SCRIPT / 'temporal_probe_20261007'
OUT = SCRIPT / 'temporal_convergence_20261007'
KINDS = ('framewise', 'gru', 'time_only')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(kind, x):
    if kind == 'framewise':
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[-1], 32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    else:
        model = T.Recurrent(x.shape[-1])
    return model.to(device=x.device, dtype=x.dtype)


def predict(model, kind, x):
    point = model(x)
    return point[..., 0] if kind == 'framewise' else point


def train(kind, x, y, proper, validation, original, original_point, protocol):
    torch.manual_seed(42)
    if x.device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    model = make_model(kind, x)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
    best = np.inf
    best_state = None
    best_epoch = None
    stale = 0
    history = []
    prefix_check = None
    settings = protocol['training']
    for epoch in range(settings['maximum_epochs']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = ((predict(model, kind, x[proper]) - y[proper]) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite temporal loss; do not suppress numerical failure')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            mae = float(abs(predict(model, kind, x[validation]) - y[validation]).mean())
        history.append(dict(epoch=epoch + 1, train_mse=loss.detach().item(), validation_mae=mae))
        if mae < best:
            best = mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if epoch + 1 == 200:
            old_history = np.array([[r['train_mse'], r['validation_mae']] for r in original['history']])
            new_history = np.array([[r['train_mse'], r['validation_mae']] for r in history])
            np.testing.assert_allclose(new_history, old_history, rtol=1e-5, atol=1e-8)
            current_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            with torch.no_grad():
                prefix_point = predict(model, kind, x).cpu().numpy()
            np.testing.assert_allclose(prefix_point, original_point, rtol=0, atol=2e-6)
            model.load_state_dict(current_state)
            prefix_check = dict(history_max_error=float(abs(new_history - old_history).max()),
                                prediction_max_error=float(abs(prefix_point - original_point).max()))
            print('CONVERGENCE_PREFIX_VERIFIED', kind, json.dumps(prefix_check), flush=True)
        if stale >= settings['patience']:
            break
    assert prefix_check is not None, 'Original200-epoch comparison must be reproduced'
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        point = predict(model, kind, x).cpu().numpy()
    if not np.isfinite(point).all():
        raise RuntimeError('Nonfinite predictions')
    torch.save(dict(state_dict=best_state, kind=kind, protocol=protocol), OUT / f'{kind}.pt')
    return point, dict(best_epoch=best_epoch, epochs=len(history), validation_mae=best,
                       stopping='validation_plateau' if stale >= settings['patience'] else 'epoch_limit',
                       stale_epochs=stale, history=history, original200_reproduction=prefix_check,
                       parameters=sum(p.numel() for p in model.parameters()))


def summary(point, truth, mask):
    error = point - truth
    count = int(mask.sum())
    return dict(frames=count, mae=float(abs(error[mask]).mean()) if count else None,
                signed_error=float(error[mask].mean()) if count else None,
                over_0_01_rate=float((error[mask] > .01).mean()) if count else None,
                over_0_03_rate=float((error[mask] > .03).mean()) if count else None,
                false_stable_rate=float((point[mask] > 0).mean()) if count and np.all(truth[mask] < 0) else None)


def evaluate(point, dataset):
    truth, observed, age = (dataset[k] for k in ('truth', 'observed', 'age'))
    seen = np.maximum.accumulate(observed, axis=1)
    strata = dict(all=np.ones(truth.shape, bool), observed=observed, missing=~observed,
                  never_observed=~seen, gap_over_half_second=seen & (age > .5),
                  stable_last2pct=(truth > 0) & (truth < .02),
                  observed_stable_last2pct=observed & (truth > 0) & (truth < .02),
                  unstable=truth < 0)
    for low, high in ((0, .128), (.128, .5), (.5, 1), (1, np.inf)):
        strata[f'gap_age_{low}_{high}'] = seen & ~observed & (age > low) & (age <= high)
    records = dataset['records']
    result = {}
    for split in ('train', 'validation', 'held'):
        rows = np.array([r['split'] == split for r in records])[:, None]
        result[split] = {key: summary(point, truth, rows & mask) for key, mask in strata.items()}
        result[split]['by_profile_noise'] = {
            f'{profile}/{noise}': summary(point, truth, np.broadcast_to(
                np.array([r['split'] == split and r['profile'] == profile and r['noise'] == noise
                          for r in records])[:, None], truth.shape))
            for profile in T.PROTOCOL['profiles'] for noise in T.PROTOCOL['noise']}
        result[split]['by_design'] = {
            design: summary(point, truth, np.broadcast_to(
                np.array([r['split'] == split and r['design'] == design for r in records])[:, None], truth.shape))
            for design in T.PROTOCOL['split'][split]}
    return result


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    original = json.loads((BASE / 'results.json').read_text())
    assert protocol['base_protocol'] == T.PROTOCOL == original['protocol']
    assert sha256(SCRIPT / 'temporal_probe.py') == protocol['base_source_sha256'] == original['source_sha256']
    for name, expected in protocol['inputs'].items():
        assert sha256(BASE / name) == expected
    for name, expected in original['helper_source_sha256'].items():
        assert sha256(SCRIPT / name) == expected
    assert not (OUT / 'results.json').exists(), 'Keep completed convergence artifacts immutable'
    with np.load(BASE / 'data.npz') as z:
        assert json.loads(str(z['protocol'])) == T.PROTOCOL
        assert json.loads(str(z['fingerprint'])) == original['fingerprint']
        assert str(z['source_sha256']) == original['source_sha256']
        dataset = {k: z[k].copy() for k in ('features', 'truth', 'observed', 'age', 'times')}
        dataset['records'] = json.loads(str(z['records']))
    with np.load(BASE / 'predictions.npz') as z:
        original_point = z['predictions'].copy()
        old_scaling = {k: z[k].copy() for k in ('mean', 'scale', 'active')}
        for key in ('truth', 'observed', 'age', 'times'):
            np.testing.assert_array_equal(z[key], dataset[key])
        assert json.loads(str(z['records'])) == dataset['records'] == original['records']
    proper = np.array([r['split'] == 'train' for r in dataset['records']])
    validation = np.array([r['split'] == 'validation' for r in dataset['records']])
    q, observed = dataset['features'], dataset['observed']
    mean, scale, active = fit_ridge_scaling(q[proper][observed[proper]])
    for key, value in (('mean', mean), ('scale', scale), ('active', active)):
        np.testing.assert_array_equal(value, old_scaling[key])
    normalized = (q - mean) / scale
    normalized[..., ~active] = 0
    normalized[~observed] = 0
    if not torch.cuda.is_available():
        raise RuntimeError('Run the matched comparison on the existing GPU; no silent CPU fallback')
    device = torch.device('cuda')
    x = torch.as_tensor(np.concatenate((normalized, observed[..., None].astype(float),
                                       dataset['age'][..., None]), axis=2), device=device, dtype=torch.float32)
    target = torch.as_tensor(dataset['truth'], device=device, dtype=x.dtype)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    result = dict(protocol=protocol, source_sha256=sha256(Path(__file__)),
                  helper_source_sha256={p.name: sha256(p) for p in (Path(T.__file__), Path(T.M.__file__), Path(T.P.__file__))},
                  backend=dict(device=str(device), gpu=torch.cuda.get_device_name(device), torch=torch.__version__),
                  fingerprint=original['fingerprint'], records=dataset['records'], models={}, original200={})
    points = []
    for index, kind in enumerate(KINDS):
        inputs = x if kind != 'time_only' else torch.as_tensor(np.broadcast_to(
            dataset['times'][None, :, None], (len(dataset['records']), len(dataset['times']), 1)).copy(),
            device=device, dtype=x.dtype)
        point, training = train(kind, inputs, target, proper, validation,
                                original['models'][kind]['training'], original_point[index], protocol)
        points.append(point)
        result['models'][kind] = dict(training=training, evaluation=evaluate(point, dataset))
        result['original200'][kind] = evaluate(original_point[index], dataset)
        T.M.G.O.C.save_json(OUT / 'progress.json', T.M.clean_json(result))
        print('CONVERGENCE_TRAINED', kind, json.dumps({k: training[k] for k in
              ('best_epoch', 'epochs', 'validation_mae', 'stopping')}), flush=True)
        print('CONVERGENCE_HELD', kind, json.dumps(result['models'][kind]['evaluation']['held']), flush=True)
    T.M.G.O.C.save_npz(OUT / 'predictions.npz', predictions=np.stack(points), mean=mean, scale=scale, active=active,
                      **{k: dataset[k] for k in ('truth', 'observed', 'age', 'times')},
                      records=json.dumps(dataset['records']), protocol=json.dumps(protocol))
    result['elapsed_seconds'] = time.perf_counter() - started
    T.M.G.O.C.save_json(OUT / 'results.json', T.M.clean_json(result))
    print('CONVERGENCE_COMPLETE', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
