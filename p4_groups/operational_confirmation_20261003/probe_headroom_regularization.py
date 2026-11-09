"""One fixed offline alpha=0 counterfactual; unchanged degree2 oracle features."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
import verify_headroom_oracle as V
import experiment_headroom_oracle as E

OUT = V.OUT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zero_penalty(x, y, query):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    active = std > 1e-8 * np.maximum(1, abs(mean))
    scale = np.where(active, std + 1e-8, 1)
    z = (x[:, active] - mean[active]) / scale[active]
    q = (query[:, active] - mean[active]) / scale[active]
    centered = y - y.mean()
    coefficient, _, rank, singular = np.linalg.lstsq(z, centered, rcond=None)
    prediction = q @ coefficient + y.mean()
    # Separate SciPy LAPACK invocation checks the rank convention and outputs.
    independent, _, other_rank, _ = lstsq(z, centered,
        cond=np.finfo(float).eps * max(z.shape), lapack_driver='gelsd')
    alternate = q @ independent + y.mean()
    relative = float(np.max(abs(prediction - alternate) / (1 + abs(prediction))))
    assert rank == other_rank and relative < 1e-7
    assert np.isfinite(prediction).all()
    return prediction, dict(rows=len(y), active_columns=int(active.sum()), rank=int(rank),
        smallest_singular_value=float(singular[-1]), independent_relative_error=relative)


def report(y, prediction, group, ids):
    return {name: E.metric(y, prediction, mask, ids) for name, mask in {
        'all': group, 'boundary': group & (y > -.02) & (y < .05),
        'unstable': group & (y < 0), 'farther_stable': group & (y >= .10)}.items()}


def run():
    target = OUT / 'regularization_probe.json'
    if target.exists():
        raise RuntimeError('Completed probe is immutable')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    result = json.loads((OUT / 'results.json').read_text())
    proof = json.loads((OUT / 'verification.json').read_text())
    assert proof['results_sha256'] == sha(OUT / 'results.json')
    assert proof['predictions_sha256'] == sha(OUT / 'predictions.npz')
    selection = result['selections']['critical_contrast/C']['selected']
    assert selection['degree'] == 2
    penalty = selection['alpha']
    with np.load(OUT / 'predictions.npz', allow_pickle=False) as archive:
        records = V.js(archive, 'records')
        ids = np.array([r['design'] for r in records])
        noise = np.array([r['noise'] for r in records])
        split = np.array([r['split'] for r in records])
        y = archive['truth'].copy()
        common = archive['critical_common'].copy()
        features = archive['critical_features'].copy()
        baseline = archive['predictions'][V.js(archive, 'names').index('critical_contrast/C')].copy()
    proper = common & ((split == 'train') & np.isin(noise, ('clean', 'iid1')))[:, None]
    allids = np.broadcast_to(ids[:, None], y.shape)
    assert set(allids[proper]) == set(protocol['split']['train'])
    assert int(proper.sum()) == result['selections']['critical_contrast/C']['tuning_rows']
    x = V.poly(features[proper], 2)
    full = np.full_like(y, np.nan)
    full[common], fit_info = zero_penalty(x, y[proper], V.poly(features[common], 2))
    oof_zero = np.full_like(y, np.nan)
    oof_penalty = np.full_like(y, np.nan)
    fold_info = {}
    for did in protocol['split']['train']:
        fit = proper & (ids != did)[:, None]
        query = common & (ids == did)[:, None]
        xp = V.poly(features[fit], 2)
        qp = V.poly(features[query], 2)
        oof_zero[query], fold_info[did] = zero_penalty(xp, y[fit], qp)
        oof_penalty[query] = V.independent_ridge(xp, y[fit], qp, [penalty])[:, 0]
    metrics = {}
    for name in ('train', 'held'):
        for condition in protocol['noise']:
            group = np.broadcast_to(((split == name) & (noise == condition))[:, None], y.shape)
            predictions = {'existing_penalty': baseline, 'zero_penalty': full}
            if name == 'train':
                predictions.update(lodo_existing_penalty=oof_penalty, lodo_zero_penalty=oof_zero)
            metrics[name + '/' + condition] = {
                arm: report(y, pred, group, ids) for arm, pred in predictions.items()}
    arrays = OUT / 'regularization_probe_predictions.npz'
    np.savez_compressed(arrays, zero_penalty=full, lodo_zero_penalty=oof_zero,
        lodo_existing_penalty=oof_penalty, critical_common=common, truth=y)
    payload = dict(scope='Offline single-change regularization diagnostic; not a deployable predictor',
        implementation_sha256=sha(__file__), protocol_sha256=sha(OUT / 'regularization_probe_protocol.json'),
        input_hashes={name: sha(OUT / name) for name in ('protocol.json', 'results.json', 'verification.json', 'predictions.npz')},
        prediction_arrays_sha256=sha(arrays), baseline_alpha=penalty, counterfactual_alpha=0,
        degree=2, fit_info=fit_info, fold_info=fold_info, metrics=metrics,
        limitations=['Four fitting designs and two repeatedly inspected held TRAIN designs; no blind evaluation',
            'LODO uses the already selected fixed baseline alpha, not nested performance estimation',
            'Unknown/stale rows remain unknown; no new physical observations or oracle deployment',
            'SciPy cross-check is numerical agreement, not independent physical validation',
            'No progress credit, promotion or gate clearance from an oracle probe'])
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print('FIXED_REGULARIZATION_PROBE_COMPLETE', json.dumps({'fit_info': fit_info,
        'held': {key: {arm: {'mae': row['all']['equal_design_mae'],
            'boundary': row['boundary']['equal_design_mae'],
            'unstable_positive': row['unstable']['unstable_false_positive_point']}
            for arm, row in arms.items()} for key, arms in metrics.items() if key.startswith('held/')}}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
