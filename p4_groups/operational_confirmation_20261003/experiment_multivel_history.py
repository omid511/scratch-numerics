"""Within384ms multi-regime history; no supplied velocity or simulator truth inputs."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import experiment_additive_spectral as A
import temporal_convergence as T
import score_additive_spectral as SCORE

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUT = SCRIPT / 'multivel_history_20261009'
SOURCE = SCRIPT / 'additive_spectral_20261009'
LAGS = (4, 2, 0)  #64ms decision stride: three NONOVERLAPPING128ms apertures.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for section in ('inputs', 'helpers', 'implementation'):
        for relative, digest in protocol[section].items():
            if sha(ROOT / relative) != digest:
                raise RuntimeError(f'Pinned source changed: {relative}')
    return protocol


def histories(current, arm, records):
    """No borrowing before first frame or across acquisition views/structures."""
    if arm == 'repeated_current':
        return np.concatenate((current, current, current), axis=2)
    blocks = []
    for lag in LAGS:
        block = np.zeros_like(current)
        if lag:
            block[:, lag:] = current[:, :-lag]
        else:
            block[:] = current
        blocks.append(block)
    if arm == 'shuffled_past':
        rng = np.random.default_rng(20261011)
        donors = np.arange(len(records))
        for split in ('train', 'validation', 'held'):
            groups = sorted(set((r['design'], r['replicate'], r['profile']) for r in records if r['split'] == split))
            permutation = rng.permutation(len(groups))
            mapping = {group: groups[permutation[j]] for j, group in enumerate(groups)}
            index = {(r['design'], r['replicate'], r['profile'], r['noise']): j for j, r in enumerate(records) if r['split'] == split}
            for j, r in enumerate(records):
                if r['split'] == split:
                    donors[j] = index[(*mapping[(r['design'], r['replicate'], r['profile'])], r['noise'])]
        blocks[:2] = [block[donors] for block in blocks[:2]]
    elif arm == 'unordered_history':
        average = (blocks[0] + blocks[1] + blocks[2]) / 3
        blocks = [average, average, average]
    elif arm != 'ordered_history':
        raise ValueError(arm)
    return np.concatenate(blocks, axis=2)


def train_one(x, y, proper, validation, arm, seed, protocol):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = T.make_model('framewise', x)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
    xt, yt, xv, yv = x[proper], y[proper], x[validation], y[validation]
    best, stale, best_state, best_epoch, history = np.inf, 0, None, None, []
    for epoch in range(protocol['training']['maximum_epochs']):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(xt)[..., 0] - yt) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite history loss')
        loss.backward(); optimizer.step()
        with torch.no_grad():
            mae = float(abs(model(xv)[..., 0] - yv).mean())
        history.append(dict(epoch=epoch + 1, train_mse=float(loss.detach()), validation_mae=mae))
        if mae < best:
            best, best_epoch, stale = mae, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= protocol['training']['patience']:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        point = model(x)[..., 0].cpu().numpy()
    assert np.isfinite(point).all()
    torch.save(dict(state=best_state, arm=arm, seed=seed, protocol_sha256=sha(OUT / 'protocol.json')),
               OUT / f'{arm}_seed{seed}.pt')
    metadata = dict(seed=seed, best_epoch=best_epoch, epochs=len(history), validation_mae=best,
        input_width=x.shape[-1], parameters=sum(p.numel() for p in model.parameters()), history=history)
    print('MULTIVEL_HISTORY_TRAINED', arm, seed, best_epoch, best, flush=True)
    return point, metadata


def strata(data):
    y = data['truth']; observed = data['observed']
    warm = np.broadcast_to(np.arange(y.shape[1])[None] >= 4, y.shape)
    a = np.zeros_like(y); b = np.zeros_like(y)
    a[:, 4:] = y[:, :-4]; b[:, 2:] = y[:, :-2]
    span = np.maximum.reduce((a, b, y)) - np.minimum.reduce((a, b, y))
    moving = warm & (span >= .005)
    dwell = warm & (span <= 1e-10)
    boundary = (y > -.02) & (y < .05)
    return dict(all=np.ones_like(observed), current=observed, gap=~observed,
        boundary=boundary, boundary_current=boundary & observed, boundary_gap=boundary & ~observed,
        post_warmup=warm, warmup=~warm, distinct_regimes=moving,
        distinct_regimes_boundary=moving & boundary, distinct_regimes_current=moving & observed,
        single_regime_dwell=dwell, single_regime_dwell_boundary=dwell & boundary,
        weak_regime_change=warm & ~moving & ~dwell,
        farther_stable=y >= .10, unstable_exclusion=(y <= -.01) & (y >= -.05))


def score(points, data, arms):
    masks = strata(data); designs = np.array([r['design'] for r in data['records']])
    metrics, profiles = {}, {}
    for split in ('train', 'validation', 'held'):
        for noise in sorted(set(r['noise'] for r in data['records'] if r['split'] == split)):
            condition = f'{split}/{noise}'
            pop = np.array([r['split'] == split and r['noise'] == noise for r in data['records']])[:, None]
            metrics[condition] = {arm: {name: SCORE.summary(point, data['truth'], pop & mask, designs)
                for name, mask in masks.items()} for arm, point in zip(arms, points)}
            profiles[condition] = {}
            for profile in ('cross', 'stable', 'retreat'):
                profile_mask = pop & np.array([r['profile'] == profile for r in data['records']])[:, None]
                profiles[condition][profile] = {arm: {name: SCORE.summary(point, data['truth'], profile_mask & masks[name], designs)
                    for name in ('all', 'boundary', 'distinct_regimes', 'single_regime_dwell')} for arm, point in zip(arms, points)}
    return metrics, profiles


def smoke():
    # Behavioral checks of causal construction, including warmup and future-invariance.
    record = [dict(split='train', design='D000007', replicate=0, profile='cross', noise='clean')]
    x = np.arange(8, dtype=np.float32).reshape(1, 8, 1)
    h = histories(x, 'ordered_history', record)
    np.testing.assert_array_equal(h[0, 4], [0, 2, 4])
    np.testing.assert_array_equal(h[0, 1], [0, 0, 1])
    altered = x.copy(); altered[:, 5:] = -999
    np.testing.assert_array_equal(histories(altered, 'ordered_history', record)[:, :5], h[:, :5])
    return dict(nonoverlapping_causal_slots=True, absent_warmup_zero=True, future_changes_do_not_affect_past=True)


def run():
    protocol = contract(); construction = smoke()
    if (OUT / 'results.json').exists():
        raise RuntimeError('Completed history experiment is immutable')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required, no CPU fit fallback')
    torch.set_num_threads(1); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    data = A.load(SOURCE / 'dataset.npz')
    proper = np.array([r['split'] == 'train' and r['noise'] in ('clean', 'iid1') for r in data['records']])
    validation = np.array([r['split'] == 'validation' and r['noise'] in ('clean', 'iid1') for r in data['records']])
    current, scaling = A.input_matrix(data, None, None, [], proper)
    with np.load(SOURCE / 'predictions.npz', allow_pickle=False) as z:
        old_arms = json.loads(str(z['arms'])); baseline = z['predictions'][old_arms.index('baseline174')]
    # Exact saved baseline replay before any fits; no refitting or selecting baseline seeds.
    sample = torch.as_tensor(current, device='cuda')
    baseline_error = 0.
    for si, seed in enumerate(protocol['training']['seeds']):
        checkpoint = torch.load(SOURCE / f'baseline174_seed{seed}.pt', map_location='cuda', weights_only=False)
        model = T.make_model('framewise', sample); model.load_state_dict(checkpoint['state']); model.eval()
        with torch.no_grad():
            replay = model(sample)[..., 0].cpu().numpy()
        np.testing.assert_allclose(replay, baseline[si], rtol=0, atol=2e-6)
        baseline_error = max(baseline_error, float(abs(replay - baseline[si]).max()))
    del sample, model
    y = torch.as_tensor(data['truth'], device='cuda', dtype=torch.float32)
    prediction_rows, fits = [baseline], {}
    for arm in protocol['arms'][1:]:
        array = histories(current, arm, data['records'])
        x = torch.as_tensor(array, device='cuda')
        points, metadata = [], []
        for seed in protocol['training']['seeds']:
            point, fit = train_one(x, y, proper, validation, arm, seed, protocol)
            points.append(point); metadata.append(fit)
        prediction_rows.append(np.stack(points)); fits[arm] = metadata
        del x
    points = np.stack(prediction_rows)
    np.savez_compressed(OUT / 'predictions.npz', predictions=points, arms=json.dumps(protocol['arms']),
        seeds=protocol['training']['seeds'], protocol_sha256=sha(OUT / 'protocol.json'),
        **dict(zip(('mean', 'scale', 'active'), scaling)))
    metrics, profiles = score(points, data, protocol['arms'])
    result = dict(protocol=protocol, protocol_sha256=sha(OUT / 'protocol.json'),
        predictions_sha256=sha(OUT / 'predictions.npz'), implementation_sha256=sha(__file__),
        construction=construction, baseline_replay_max_error=baseline_error, fits=fits,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, metrics=metrics, profiles=profiles,
        no_velocity_or_truth_inputs=True, operational_predictor_changed=False,
        interpretation='Bounded causal multi-regime history, not a known-velocity sweep or identified growth-to-airspeed calibration. Current128ms windows already vary velocity slightly. Regime labels are scoring-only. Longer aperture, memory/capacity and schedule confounds need controls; two inspected held designs are not blind proof.')
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for condition in metrics:
        if condition.startswith('held/'):
            print('MULTIVEL_HISTORY_HELD', condition, json.dumps({arm: {s: metrics[condition][arm][s]['equal_design_mae']
                for s in ('all', 'boundary', 'distinct_regimes', 'single_regime_dwell')} for arm in protocol['arms']}), flush=True)
    print('MULTIVEL_HISTORY_COMPLETE', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
