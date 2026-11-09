"""Four-arm current-loss/context factorial with identical frozen native gaps."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import experiment_bounded_gru as B
import score_additive_spectral as S
SOURCE = B.SCRIPT / 'additive_spectral_20261009'
OUT = B.SCRIPT / 'current_context_20261009'
ARMS = ('full_all', 'full_current', 'first107_all', 'first107_current')
sha = B.sha


def load():
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ('features', 'truth', 'observed', 'age', 'times', 'bins')}
        data['records'] = json.loads(str(z['records']))
    with np.load(SOURCE / 'predictions.npz', allow_pickle=False) as z:
        baseline = z['predictions'][json.loads(str(z['arms'])).index('baseline174')].copy()
    return data, baseline


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Current-context protocol already frozen')
    inputs = [SOURCE / 'dataset.npz', SOURCE / 'predictions.npz', SOURCE / 'training.json',
        B.ROOT / 'P4_PROGRESS.json', Path(__file__), Path(B.__file__), Path(S.__file__),
        B.ROOT / 'src/mechanics/p4_margin_estimation/baselines.py']
    inputs.extend(SOURCE / f'baseline174_seed{seed}.pt' for seed in (42, 43, 44))
    B.save_json(OUT / 'protocol.json', dict(scope='Cached current-objective × second-mode context diagnostic',
        inputs={str(p.relative_to(B.ROOT)): sha(p) for p in inputs}, arms=list(ARMS), seeds=[42, 43, 44],
        training=dict(maximum_epochs=2000, patience=100, lr=.003, weight_decay=.0001),
        content='176 fixed width; first107 ablation zeros normalized174 content columns107:174 only',
        scaling='Common proper-training accepted clean/IID1 scaling across all four arms',
        selection='Original clean/IID1 validation all-frame MAE with same frozen baseline native gaps',
        loss='Mean square over all frames or accepted current frames; loss normalized by participating rows',
        gap_policy='Per-seed frozen baseline gap outputs; never stale/current qualification',
        reproduction='Archived checkpoint replay and full/all original-checkpoint-epoch training prefix checked. Composite selection may select another epoch.',
        endpoint='Diagnostic2 inspected held designs, missing farther-stable/unstable-exclusion guardrails remain unknown'))
    print('CURRENT_CONTEXT_FROZEN', flush=True)


def contract():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for relative, digest in protocol['inputs'].items():
        if sha(B.ROOT / relative) != digest:
            raise RuntimeError(f'Pinned current-context source changed: {relative}')
    return protocol


def train():
    protocol = contract()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(1); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    data, baseline = load(); records = data['records']
    noise = np.array([r['noise'] in ('clean', 'iid1') for r in records])
    proper = np.array([r['split'] == 'train' for r in records]) & noise
    validation = np.array([r['split'] == 'validation' for r in records]) & noise
    current, scaling = B.standardize(data, proper)
    y = torch.as_tensor(data['truth'], device='cuda', dtype=torch.float32)
    observed = torch.as_tensor(data['observed'], device='cuda')
    old_fits = json.loads((SOURCE / 'training.json').read_text())['fits']['baseline174']
    prediction_rows, native_rows, fits = [], [], []
    for arm in ARMS:
        array = current.copy()
        if arm.startswith('first107'):
            array[..., 107:174] = 0
        x = torch.as_tensor(array, device='cuda')
        xt, yt, xv, yv = x[proper], y[proper], x[validation], y[validation]
        train_current = observed[proper]; validation_current = observed[validation]
        native_seed, composite_seed = [], []
        for si, seed in enumerate(protocol['seeds']):
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = B.make_model('ordered_concat', 176, 'cuda')
            archived = torch.load(SOURCE / f'baseline174_seed{seed}.pt', map_location='cuda', weights_only=False)
            replay_model = B.make_model('ordered_concat', 176, 'cuda')
            replay_model.load_state_dict(archived['state']); replay_model.eval()
            with torch.no_grad():
                reproduced = B.predict(replay_model, 'ordered_concat', torch.as_tensor(current, device='cuda')).cpu().numpy()
            replay_error = float(abs(reproduced - baseline[si]).max())
            if replay_error > 2e-6:
                raise RuntimeError(f'Archived baseline replay failed: {replay_error}')
            del replay_model
            # The replay model was initialized after the fitted model; it must not perturb optimizer randomness (full batch, no dropout).
            optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
            frozen = torch.as_tensor(baseline[si], device='cuda')[validation]
            best, stale, best_epoch, best_state, history = np.inf, 0, 0, None, []
            original_epoch = int(old_fits[si]['best_epoch'])
            stopping_epoch = None; prefix_error = None
            for epoch in range(protocol['training']['maximum_epochs']):
                model.train(); optimizer.zero_grad(set_to_none=True)
                predicted = B.predict(model, 'ordered_concat', xt)
                error = predicted - yt
                loss = (error[train_current] ** 2).mean() if arm.endswith('current') else (error ** 2).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite current-context loss')
                loss.backward(); optimizer.step(); model.eval()
                with torch.no_grad():
                    candidate = B.predict(model, 'ordered_concat', xv)
                    composite = torch.where(validation_current, candidate, frozen)
                    score = float(abs(composite - yv).mean())
                if stopping_epoch is None:
                    history.append(dict(epoch=epoch + 1, train_mse=float(loss), validation_mae=score))
                    if score < best:
                        best, stale, best_epoch = score, 0, epoch + 1
                        best_state = copy.deepcopy(model.state_dict())
                    else:
                        stale += 1
                    if stale >= protocol['training']['patience']:
                        stopping_epoch = epoch + 1
                if arm == 'full_all' and epoch + 1 == original_epoch:
                    with torch.no_grad():
                        prefix = B.predict(model, 'ordered_concat', x).cpu().numpy()
                    prefix_error = float(abs(prefix - baseline[si]).max())
                    if prefix_error > 2e-6:
                        raise RuntimeError(f'Full/all original training prefix mismatch: {prefix_error}')
                if stopping_epoch is not None and (arm != 'full_all' or epoch + 1 >= original_epoch):
                    break
            if best_state is None:
                raise RuntimeError('No finite factorial checkpoint')
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                native = B.predict(model, 'ordered_concat', x).cpu().numpy()
            composite = np.where(data['observed'], native, baseline[si])
            if not np.isfinite(composite).all():
                raise FloatingPointError('Nonfinite composite output')
            torch.save(dict(state=best_state, arm=arm, seed=seed, protocol_sha256=sha(OUT / 'protocol.json')),
                OUT / f'{arm}_seed{seed}.pt')
            metadata = dict(arm=arm, seed=seed, best_epoch=best_epoch, epochs=len(history), validation_mae=best,
                history=history, archived_replay_error=replay_error, original_epoch=original_epoch,
                full_all_prefix_error=prefix_error, prefix_only_extra_updates=(epoch + 1 - len(history)),
                fitting_loss_rows=int(train_current.sum()) if arm.endswith('current') else int(yt.numel()))
            fits.append(metadata); native_seed.append(native); composite_seed.append(composite)
            print('CURRENT_CONTEXT_FIT', arm, seed, best_epoch, best, 'prefix', prefix_error, flush=True)
        native_rows.append(np.stack(native_seed)); prediction_rows.append(np.stack(composite_seed))
    np.savez_compressed(OUT / 'predictions.npz', predictions=np.stack(prediction_rows), native=np.stack(native_rows),
        mean=scaling[0], scale=scaling[1], active=scaling[2], proper=proper, validation=validation,
        arms=json.dumps(ARMS), seeds=protocol['seeds'], protocol_sha256=sha(OUT / 'protocol.json'))
    B.save_json(OUT / 'training.json', dict(fits=fits, gpu=torch.cuda.get_device_name(),
        predictions_sha256=sha(OUT / 'predictions.npz'), protocol_sha256=sha(OUT / 'protocol.json')))
    score()


def score():
    protocol = contract(); data, baseline = load(); records = data['records']
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as z:
        points = z['predictions'].copy()
    designs = np.array([r['design'] for r in records]); strata = S.masks(data['truth'], data['observed'], data['times'], data['bins'])
    original = {name: mask.copy() for name, mask in strata.items()}
    for profile in ('cross', 'stable', 'retreat'):
        pm = np.array([r['profile'] == profile for r in records])[:, None]
        strata[f'profile_{profile}'] = pm & original['all']
        strata[f'boundary_profile_{profile}'] = pm & original['boundary']
    metrics, factorial, transitions = {}, {}, {}
    for split in ('train', 'validation', 'held'):
        for noise in sorted({r['noise'] for r in records if r['split'] == split}):
            condition = f'{split}/{noise}'
            pop = np.array([r['split'] == split and r['noise'] == noise for r in records])[:, None]
            metrics[condition] = {arm: {name: S.summary(point, data['truth'], pop & mask, designs)
                for name, mask in strata.items()} for arm, point in zip(ARMS, points)}
            metrics[condition]['frozen_baseline'] = {name: S.summary(baseline, data['truth'], pop & mask, designs)
                for name, mask in strata.items()}
            factorial[condition] = {}
            for name in ('all', 'boundary', 'current', 'boundary_current', 'gap'):
                ds = metrics[condition]['full_all'][name]['per_design']
                effects = {}
                for did in ds:
                    value = {arm: np.array(metrics[condition][arm][name]['per_design'][did]['mae_by_seed']) for arm in ARMS}
                    effects[did] = dict(current_loss_full=(value['full_current'] - value['full_all']).tolist(),
                        second_block_all=(value['first107_all'] - value['full_all']).tolist(),
                        current_loss_first107=(value['first107_current'] - value['first107_all']).tolist(),
                        interaction=(value['first107_current'] - value['first107_all'] - value['full_current'] + value['full_all']).tolist())
                factorial[condition][name] = effects
            obs = data['observed']; changed = np.zeros_like(obs)
            changed[:, 1:] = obs[:, 1:] != obs[:, :-1]
            selected = pop & changed
            transitions[condition] = {}
            for arm, point in zip(ARMS, points):
                jump = np.zeros_like(point, dtype=np.float64)
                jump[:, :, 1:] = np.diff(point.astype(np.float64) - data['truth'][None], axis=2)
                values = abs(jump[:, selected])
                transitions[condition][arm] = dict(frames=int(selected.sum()),
                    mean_abs_error_jump_by_seed=values.mean(axis=1).tolist() if selected.any() else None,
                    p95_abs_error_jump_by_seed=np.quantile(values, .95, axis=1).tolist() if selected.any() else None)
    B.save_json(OUT / 'results.json', dict(scope=protocol['scope'], metrics=metrics, factorial=factorial,
        transitions=transitions, protocol_sha256=sha(OUT / 'protocol.json'), predictions_sha256=sha(OUT / 'predictions.npz'),
        operational_predictor_changed=False, qualification='No inferred guardrail pass from absent farther-stable/sign-exclusion populations'))
    print('CURRENT_CONTEXT_SCORED', len(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('freeze', 'train', 'score'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        globals()[args.phase]()
