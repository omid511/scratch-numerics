"""Reset-at-decision response-history comparison on preregistered TRAIN folds."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
import score_additive_spectral as S
import experiment_multivel_history as H
SOURCE = SCRIPT / 'support_modal_20261009'
OUT = SCRIPT / 'bounded_gru_20261009'
ARMS = ('content_gru', 'ordered_concat', 'unordered_pool', 'flags_age_gru')
LAGS = (4, 2, 0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_data():
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ('features', 'truth', 'observed', 'age', 'times')}
        data['records'] = json.loads(str(z['records']))
    return data


def make_slots(current):
    slots = np.zeros((*current.shape[:2], 3, current.shape[-1]), dtype=np.float32)
    for index, lag in enumerate(LAGS):
        if lag:
            slots[:, lag:, index] = current[:, :-lag]
        else:
            slots[:, :, index] = current
    return slots


def arm_input(slots, arm):
    if arm == 'flags_age_gru':
        return slots[..., -2:].copy()
    if arm == 'content_gru':
        return slots
    if arm == 'ordered_concat':
        return slots.reshape(*slots.shape[:2], -1)
    if arm == 'unordered_pool':
        average = slots.mean(axis=2)
        return np.concatenate((average, average, average), axis=-1)
    raise ValueError(arm)


class ResetGRU(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gru = torch.nn.GRU(width, 32, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, x):
        # Each flattened decision is an independent sequence, including warmup.
        sequence = x.reshape(-1, 3, x.shape[-1])
        response, _ = self.gru(sequence)
        return self.head(response[:, -1])[..., 0].reshape(x.shape[:2])


def make_model(arm, width, device):
    if arm.endswith('gru'):
        model = ResetGRU(width)
    else:
        model = torch.nn.Sequential(torch.nn.Linear(width, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    return model.to(device=device, dtype=torch.float32)


def predict(model, arm, x):
    return model(x) if arm.endswith('gru') else model(x)[..., 0]


def mechanism_smoke():
    current = np.arange(1 * 9 * 176, dtype=np.float32).reshape(1, 9, 176)
    slots = make_slots(current)
    np.testing.assert_array_equal(slots[0, 4], current[0, [0, 2, 4]])
    assert np.count_nonzero(slots[:, :4, 0]) == 0
    changed = current.copy(); changed[:, 6:] *= -1
    np.testing.assert_array_equal(make_slots(changed)[:, :6], slots[:, :6])
    torch.manual_seed(42)
    model = make_model('content_gru', 176, 'cpu').eval()
    with torch.no_grad():
        reference = model(torch.from_numpy(slots)).numpy()
        shuffled = slots[:, ::-1].copy()
        actual = model(torch.from_numpy(shuffled)).numpy()[:, ::-1]
    np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=0)
    return dict(lags=list(LAGS), absent_warmup_zero=True, future_invariant=True,
        decision_reordering_invariant=True, oldest_waveform_sample_age_ms=383)


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Frozen bounded-GRU protocol already exists')
    source = json.loads((SOURCE / 'protocol.json').read_text())
    prereqs = [SOURCE / 'dataset.npz', SOURCE / 'protocol.json', SOURCE / 'results.json', SOURCE / 'verification.json',
        SCRIPT / 'current_context_20261009/results.json', SCRIPT / 'current_context_20261009/verification.json',
        ROOT / 'P4_PROGRESS.json', ROOT / 'src/mechanics/p4_margin_estimation/baselines.py',
        SCRIPT / 'score_additive_spectral.py', SCRIPT / 'experiment_multivel_history.py', Path(__file__)]
    protocol = dict(scope='Conditional bounded response-GRU comparison; TRAIN-design folds, not blind or deployment qualification',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in prereqs},
        split=source['split'], additional_designs=source['additional_designs'], evaluation_folds=source['evaluation_folds'],
        arms=list(ARMS), lags=list(LAGS), samples=128, stride=64, dt=.001,
        reference='Expanded TRAIN support, unchanged measured174 all-frame row-mean objective and original validation checkpoint designs. No result-based head/loss retuning.',
        training=dict(seeds=[42, 43, 44], maximum_epochs=2000, patience=100, learning_rate=.003, weight_decay=.0001,
            fitting_noise=['clean', 'iid1'], loss='Row-mean squared error on every native frame',
            selection='Original clean/IID1 checkpoint-validation native all-frame MAE'),
        gap_policy='Native outputs at every decision; missing-current outputs are explicitly unqualified extrapolation, not stale holds',
        mechanism=mechanism_smoke(), endpoint_limit='Runtime, uncertainty, warnings and independent validation not qualified')
    save_json(OUT / 'protocol.json', protocol)
    print('BOUNDED_GRU_FROZEN', sha(OUT / 'protocol.json'), flush=True)


def contract():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for relative, digest in protocol['inputs'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f'Pinned prerequisite changed: {relative}')
    return protocol


def standardize(data, proper):
    q, observed = data['features'], data['observed']
    mean, scale, active = fit_ridge_scaling(q[proper][observed[proper]])
    content = (q - mean) / scale
    content[..., ~active] = 0
    content[~observed] = 0
    current = np.concatenate((content, observed[..., None], data['age'][..., None]), axis=-1).astype(np.float32)
    return current, (mean, scale, active)


def train_fit(data, protocol, fold, arm, seed, proper, validation, current, scaling):
    stem = f'fold{fold}_{arm}_seed{seed}'
    result_path = OUT / f'{stem}.npz'
    if result_path.exists():
        with np.load(result_path, allow_pickle=False) as z:
            assert str(z['protocol_sha256']) == sha(OUT / 'protocol.json')
            return z['predictions'].copy(), json.loads(str(z['metadata']))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x = torch.as_tensor(arm_input(make_slots(current), arm), device='cuda')
    model = make_model(arm, x.shape[-1], 'cuda')
    y = torch.as_tensor(data['truth'], dtype=torch.float32, device='cuda')
    xt, yt, xv, yv = x[proper], y[proper], x[validation], y[validation]
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
    best, stale, best_state, best_epoch, history = np.inf, 0, None, 0, []
    for epoch in range(protocol['training']['maximum_epochs']):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = ((predict(model, arm, xt) - yt) ** 2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite bounded-GRU training loss')
        loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad():
            score = float(abs(predict(model, arm, xv) - yv).mean())
        history.append(dict(epoch=epoch + 1, train_mse=float(loss), validation_mae=score))
        if score < best:
            best, stale, best_epoch = score, 0, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= protocol['training']['patience']:
            break
    if best_state is None:
        raise RuntimeError('No finite bounded-GRU checkpoint selected')
    model.load_state_dict(best_state); model.eval()
    point = np.empty(data['truth'].shape, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), 32):
            point[start:start + 32] = predict(model, arm, x[start:start + 32]).cpu().numpy()
    if not np.isfinite(point).all():
        raise FloatingPointError('Nonfinite bounded-GRU predictions')
    metadata = dict(fold=fold, arm=arm, seed=seed, epochs=len(history), best_epoch=best_epoch,
        validation_mae=best, history=history, parameters=sum(p.numel() for p in model.parameters()),
        training_records=int(proper.sum()), fitting_designs=sorted({r['design'] for r, keep in zip(data['records'], proper) if keep}),
        fitting_rows=int(proper.sum() * data['truth'].shape[1]))
    torch.save(dict(state=best_state, arm=arm, seed=seed, protocol_sha256=sha(OUT / 'protocol.json')), OUT / f'{stem}.pt')
    np.savez_compressed(result_path, predictions=point, mean=scaling[0], scale=scaling[1], active=scaling[2],
        proper=proper, validation=validation, metadata=json.dumps(metadata), protocol_sha256=sha(OUT / 'protocol.json'))
    print('BOUNDED_GRU_FIT', fold, arm, seed, best_epoch, best, flush=True)
    return point, metadata


def train():
    protocol = contract()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    data = load_data(); records = data['records']
    noise = np.array([r['noise'] in ('clean', 'iid1') for r in records])
    validation = np.array([r['design'] in protocol['split']['validation'] for r in records]) & noise
    fits = []
    for fold, excluded in enumerate(protocol['evaluation_folds']):
        fitting_ids = set(protocol['split']['train']) | (set(protocol['additional_designs']) - set(excluded))
        proper = np.array([r['design'] in fitting_ids for r in records]) & noise
        assert not fitting_ids & (set(excluded) | set(protocol['split']['validation']) | set(protocol['split']['held']))
        current, scaling = standardize(data, proper)
        for arm in ARMS:
            for seed in protocol['training']['seeds']:
                _, metadata = train_fit(data, protocol, fold, arm, seed, proper, validation, current, scaling)
                fits.append(metadata)
    save_json(OUT / 'training.json', dict(fits=fits, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, protocol_sha256=sha(OUT / 'protocol.json'), dataset_sha256=sha(SOURCE / 'dataset.npz')))
    print('BOUNDED_GRU_TRAINING_COMPLETE', len(fits), flush=True)


def score():
    protocol = contract(); data = load_data(); records = data['records']
    if (OUT / 'verification.json').exists():
        raise RuntimeError('Verified scores are immutable')
    strata = H.strata(data)
    strata['farther_stable15'] = data['truth'] >= .15
    metrics, paired, artifact_hashes = {}, {}, {}
    for fold, excluded in enumerate(protocol['evaluation_folds']):
        points = []
        for arm in ARMS:
            rows = []
            for seed in protocol['training']['seeds']:
                path = OUT / f'fold{fold}_{arm}_seed{seed}.npz'
                artifact_hashes[path.name] = sha(path)
                with np.load(path, allow_pickle=False) as z:
                    rows.append(z['predictions'].copy())
            points.append(np.stack(rows))
        for population_name, ids in (('evaluation', excluded), ('secondary_held', protocol['split']['held'])):
            for noise in sorted({r['noise'] for r in records if r['design'] in ids}):
                condition = f'fold{fold}/{population_name}/{noise}'
                pop = np.array([r['design'] in ids and r['noise'] == noise for r in records])[:, None]
                designs = np.array([r['design'] for r in records])
                masks = dict(strata)
                for profile in sorted({r['profile'] for r in records}):
                    pm = np.array([r['profile'] == profile for r in records])[:, None]
                    masks[f'profile_{profile}'] = pm & strata['all']
                    masks[f'boundary_profile_{profile}'] = pm & strata['boundary']
                metrics[condition] = {arm: {name: S.summary(point, data['truth'], pop & mask, designs)
                    for name, mask in masks.items()} for arm, point in zip(ARMS, points)}
                paired[condition] = {}
                for mask_name in ('all', 'boundary', 'distinct_regimes', 'boundary_current', 'profile_retreat', 'farther_stable15'):
                    gru = metrics[condition]['content_gru'][mask_name]
                    paired[condition][mask_name] = {}
                    for control in ARMS[1:]:
                        ref = metrics[condition][control][mask_name]
                        delta = {did: (np.array(g['mae_by_seed']) - ref['per_design'][did]['mae_by_seed']).tolist()
                            for did, g in gru['per_design'].items()}
                        paired[condition][mask_name][control] = dict(delta_by_design_seed=delta,
                            mean_delta=float(np.mean(list(delta.values()))) if delta else None,
                            improved_every_design_seed=bool(delta and np.all(np.array(list(delta.values())) < 0)))
    save_json(OUT / 'results.json', dict(scope=protocol['scope'], arms=list(ARMS), seeds=protocol['training']['seeds'],
        metrics=metrics, paired=paired, artifact_hashes=artifact_hashes, protocol_sha256=sha(OUT / 'protocol.json'),
        operational_predictor_changed=False, gaps='Native unqualified extrapolation; no stale/current substitution',
        interpretation='Grouped TRAIN folds are development evidence, not blind validation.3 slots span383ms; no retained hidden state.'))
    print('BOUNDED_GRU_SCORED', len(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'freeze', 'train', 'score'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == 'smoke':
            print('BOUNDED_GRU_MECHANISM_SMOKE', json.dumps(mechanism_smoke()))
        else:
            globals()[args.phase]()
