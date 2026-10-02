"""Design-isolated temporal models, standalone and additions to the frozen blend."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


E = module('temporal_fusion_contract', ROOT / 'p4_groups/blend_additions_20261002/run_experiment.py')
C, np = E.C, E.np
acquisition = module('temporal_acquisition', ROOT / 'p4_groups/local_observability_20261001/prepare_features.py')
import torch
from mechanics.p4_margin_estimation.temporal import TemporalMarginModel
from mechanics.p4_margin_estimation.optimization import fit_epochs
from mechanics.p4_margin_estimation.quantile_head import pinball_loss, SUPPORTED_QUANTILES
ARMS = ['blend_frozen', 'blend_control', 'gru', 'fixed_gru', 'learned_gru', 'tcn', 'fixed_tcn', 'learned_tcn']
DATA = None


def setup():
    global DATA
    device = torch.device(os.environ.get('P4_DEVICE', 'cpu'))
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('P4_DEVICE must select CPU or CUDA')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('P4_DEVICE requests CUDA but CUDA is unavailable')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = C.load_data()
    train_names = C.helpers.base.TRAIN_NAMES
    train_conditions = np.array([names.index(n) for n in train_names])
    np.testing.assert_array_equal(train, evaluation[train_conditions])
    protocol = json.loads((OUT / 'protocol.json').read_text())
    sources = [Path(__file__), OUT / 'protocol.json', ROOT / 'src/mechanics/p4_margin_estimation/temporal.py',
               ROOT / 'src/mechanics/p4_margin_estimation/optimization.py', ROOT / 'src/mechanics/p4_margin_estimation/tcn.py',
               ROOT / 'src/mechanics/p4_margin_estimation/baselines.py', ROOT / 'src/mechanics/p4_margin_estimation/transient.py',
               ROOT / 'p4_groups/blend_additions_20261002/run_experiment.py',
               ROOT / 'p4_groups/local_observability_20261001/prepare_features.py']
    reference_paths = [ROOT / 'p4_groups/blend_additions_20261002/predictions.npz',
                       ROOT / 'p4_groups/local_observability_20261001/features.npz']
    with np.load(reference_paths[1]) as z:
        raw_hashes = json.loads(str(z['raw_hashes']))
    fingerprint = {'sources': {str(p.relative_to(ROOT)): C.sha256(p) for p in sources},
                   'references': {str(p.relative_to(ROOT)): C.sha256(p) for p in reference_paths},
                   'raw': raw_hashes, 'versions': {'torch': torch.__version__, 'numpy': np.__version__},
                   'device': str(device)}
    side = np.concatenate((evaluation, np.broadcast_to(np.log(dt)[None, :, None], (len(names), len(rows), 1))), axis=2)
    for ci, name in enumerate(names):
        if name.startswith('dt_'):
            side[ci, :, -1] += np.log(.98 if name == 'dt_minus2pct' else 1.02)
    DATA = dict(train=train, evaluation=evaluation, y=y, ids=ids, rows=rows, names=names, folds=folds,
                layout=layout, dt=dt, train_conditions=train_conditions, fingerprint=fingerprint,
                side=side.astype(np.float32), settings=protocol['optimization'], device=device)
    return DATA


def prepare():
    d = setup()
    manifest_path = OUT / 'input_manifest.json'
    signal_path = OUT / 'signals.npy'
    if manifest_path.exists():
        record = json.loads(manifest_path.read_text())
        assert record['fingerprint'] == d['fingerprint'], 'stale temporal input cache'
        assert C.sha256(signal_path) == record['signals_hash'], 'changed temporal signals'
        assert C.sha256(OUT / 'lengths.npy') == record['lengths_hash']
        print('TEMPORAL_INPUT_REPLAY', record['shape'], flush=True)
        return
    signals = np.lib.format.open_memmap(OUT / 'signals.partial.npy', mode='w+', dtype=np.float32,
                                       shape=(len(d['names']), len(d['y']), 16, 512))
    lengths = np.empty(signals.shape[:2], dtype=np.int64)
    lookup = {int(row): i for i, row in enumerate(d['rows'])}
    gains = np.array(json.loads((ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    for did in np.unique(d['ids']):
        source = ROOT / 'p4_sensor_candidates' / f'{did}.npz'
        assert C.sha256(source) == d['fingerprint']['raw'][did]
        with np.load(source) as z:
            raw, sigma, source_rows = (z[k] for k in ('raw', 'noise_scale', 'source_indices'))
        for j, index in enumerate(source_rows):
            k = lookup[int(index)]
            assert d['ids'][k] == did
            white, shared = {}, {}
            for ci, name in enumerate(d['names']):
                signal, reported_dt = acquisition.measured_signal(raw[j], sigma[j], int(index), name,
                    d['layout'], gains, float(d['dt'][k]), white, shared)
                hits = np.flatnonzero(np.any(abs(signal) >= 50., axis=0))
                usable = int(hits[0]) if len(hits) else 512
                np.testing.assert_allclose(usable / 512, d['evaluation'][ci, k, 20], atol=1e-12, rtol=0)
                np.testing.assert_allclose(np.log(reported_dt), d['side'][ci, k, -1], atol=1e-6, rtol=0)
                signal[:, usable:] = 0
                signals[ci, k] = signal
                lengths[ci, k] = max(1, usable)
        print('TEMPORAL_ACQUISITION', did, len(source_rows), flush=True)
    signals.flush()
    shape = signals.shape
    del signals
    (OUT / 'signals.partial.npy').replace(signal_path)
    np.save(OUT / 'lengths.npy', lengths)
    C.save_json(manifest_path, dict(fingerprint=d['fingerprint'], shape=shape,
        signals_hash=C.sha256(signal_path), lengths_hash=C.sha256(OUT / 'lengths.npy'),
        prefix_lengths={'min': int(lengths.min()), 'median': float(np.median(lengths)), 'max': int(lengths.max())}))
    print('TEMPORAL_INPUTS_COMPLETE', shape, flush=True)


class Batches:
    """Vectorized memory-map batches: no per-clip Torch copies or worker processes."""
    def __init__(self, rows, conditions, mean, scale, active, *, seed=None):
        self.rows = np.tile(rows, len(conditions))
        self.conditions = np.repeat(conditions, len(rows))
        self.mean, self.scale, self.active = mean, scale, active
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def __iter__(self):
        d = DATA
        order = self.rng.permutation(len(self.rows)) if self.rng is not None else np.arange(len(self.rows))
        size = d['settings']['batch_size']
        for start in range(0, len(order), size):
            indices = order[start:start + size]
            rows, conditions = self.rows[indices], self.conditions[indices]
            lengths = d['lengths'][conditions, rows]
            x = d['signals'][conditions, rows, :, :int(lengths.max())]
            side = (d['side'][conditions, rows] - self.mean) / self.scale
            side[:, ~self.active] = 0
            yield torch.from_numpy(x), torch.from_numpy(lengths), torch.from_numpy(side), torch.from_numpy(d['y'][rows].astype(np.float32))


def loss_for_batch(model, batch):
    x, lengths, side, y = batch
    return pinball_loss(model(x, lengths, side), y, SUPPORTED_QUANTILES, weights=(2., 1., 1.))


def scaling(rows):
    d = DATA
    mean, scale, active = E.fit_ridge_scaling(d['side'][d['train_conditions']][:, rows].reshape(-1, 175))
    return mean.astype(np.float32), scale.astype(np.float32), active


def fit(kind, seed, fold, part, fit_rows, *, smoke=False):
    d = DATA
    path = OUT / f'{"smoke_" if smoke else ""}{kind}_seed{seed}_fold{fold}_part{part}.pt'
    if path.exists():
        saved = torch.load(path, weights_only=False, map_location='cpu')
        assert saved['fingerprint'] == d['fingerprint'], 'stale temporal model'
        np.testing.assert_array_equal(saved['fit_rows'], fit_rows)
    else:
        designs = np.unique(d['ids'][fit_rows]).copy()
        np.random.default_rng(2047 + fold + part).shuffle(designs)
        select_ids = designs[:int(np.ceil(.2 * len(designs)))]
        select_rows = fit_rows[np.isin(d['ids'][fit_rows], select_ids)]
        train_rows = fit_rows[~np.isin(d['ids'][fit_rows], select_ids)]
        assert not set(d['ids'][train_rows]) & set(d['ids'][select_rows])
        initial_scaling = scaling(train_rows)
        torch.manual_seed(seed)
        model = TemporalMarginModel(kind).to(d['device'])
        begin = time.perf_counter()
        maximum = 2 if smoke else d['settings']['epochs_max']
        select_history = fit_epochs(model, Batches(train_rows, d['train_conditions'], *initial_scaling, seed=seed),
            Batches(select_rows, d['train_conditions'], *initial_scaling), loss_for_batch,
            epochs=maximum, lr=.001, patience=12, weight_decay=.0001, schedule_epochs=80)
        chosen = int(np.argmin(select_history['val_loss'])) + 1
        final_scaling = scaling(fit_rows)
        torch.manual_seed(seed)
        model = TemporalMarginModel(kind).to(d['device'])
        history = fit_epochs(model, Batches(fit_rows, d['train_conditions'], *final_scaling, seed=seed),
            None, loss_for_batch, epochs=chosen, lr=.001, weight_decay=.0001, schedule_epochs=80)
        saved = dict(state=model.state_dict(), scaling=final_scaling, fingerprint=d['fingerprint'],
            fit_rows=fit_rows, training_rows=train_rows, selection_rows=select_rows,
            selected_epochs=chosen, selection_history=select_history, refit_history=history,
            seconds=time.perf_counter() - begin, parameters=sum(p.numel() for p in model.parameters()))
        temporary = path.with_suffix('.tmp')
        torch.save(saved, temporary)
        temporary.replace(path)
        print('TEMPORAL_FIT', kind, seed, fold, part, chosen, round(saved['seconds'], 2), flush=True)
    model = TemporalMarginModel(kind).to(d['device'])
    model.load_state_dict(saved['state'])
    model.eval()
    record = {key: saved[key] for key in ('selected_epochs', 'seconds', 'parameters', 'selection_history', 'refit_history')}
    record.update(path=str(path.relative_to(ROOT)), hash=C.sha256(path), fit_designs=np.unique(d['ids'][fit_rows]).tolist(),
                  selection_designs=np.unique(d['ids'][saved['selection_rows']]).tolist(),
                  training_designs=np.unique(d['ids'][saved['training_rows']]).tolist(), kind=kind, seed=seed, fold=fold, part=part)
    return model, saved['scaling'], record


def predict(model, rows, conditions, normalization):
    out = []
    with torch.inference_mode():
        for x, lengths, side, _ in Batches(rows, conditions, *normalization):
            out.append(model(x.to(DATA['device']), lengths.to(DATA['device']),
                             side.to(DATA['device'])).cpu().numpy())
    result = np.concatenate(out).reshape(len(conditions), len(rows), 3)
    assert np.isfinite(result).all() and np.all(result[:, :, 0] <= result[:, :, 1]) and np.all(result[:, :, 1] <= result[:, :, 2])
    return result


def inner_base(seed, fold, part, fit_rows, query_rows):
    d = DATA
    path = E.OLD / f'inner_seed{seed}_fold{fold}_part{part}.npz'
    with np.load(path) as z:
        np.testing.assert_array_equal(z['fit_rows'], fit_rows)
        np.testing.assert_array_equal(z['query_rows'], query_rows)
        meta = json.loads(str(z['metadata']))
        assert meta['one']['success'] and meta['full']['success']
        models = C.cloudpickle.loads(z['components'].tobytes())
        q = d['train'][:, query_rows].reshape(-1, 174)
        actual = np.column_stack((models['one'](q[:, :107]), models['full'](q))).reshape(4, len(query_rows), 2)
        np.testing.assert_allclose(actual, z['predictions'][:, :, :2], atol=1e-12, rtol=1e-12)
    return actual, dict(path=str(path.relative_to(ROOT)), hash=C.sha256(path), frozen_base=True,
                       fit_designs=meta['fit_designs'], query_designs=meta['query_designs'])


def fold_job(seed, fold, smoke=False):
    d = DATA
    torch.set_num_threads(1)
    d['signals'] = np.load(OUT / 'signals.npy', mmap_mode='r')
    d['lengths'] = np.load(OUT / 'lengths.npy', mmap_mode='r')
    path = OUT / f'{"smoke_" if smoke else ""}fold_seed{seed}_fold{fold}.npz'
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['fingerprint'])) == d['fingerprint']
        return path
    proper, cal, held = C.split(d['ids'], d['folds'], fold)
    designs = np.unique(d['ids'][proper]).copy()
    np.random.default_rng(1047 + fold).shuffle(designs)
    base_oof = np.full((4, len(proper), 2), np.nan)
    temporal_oof = {kind: np.full((4, len(proper)), np.nan) for kind in ('gru', 'tcn')}
    records = []
    for part, group in enumerate(np.array_split(designs, 3)):
        query = proper[np.isin(d['ids'][proper], group)]
        fit_rows = proper[~np.isin(d['ids'][proper], group)]
        slots = np.searchsorted(proper, query)
        assert not set(d['ids'][fit_rows]) & set(d['ids'][query])
        base_oof[:, slots], rec = inner_base(seed, fold, part, fit_rows, query)
        records.append(rec)
        for kind in ('gru', 'tcn'):
            model, normalization, rec = fit(kind, seed, fold, part, fit_rows, smoke=smoke)
            temporal_oof[kind][:, slots] = predict(model, query, d['train_conditions'], normalization)[:, :, 1]
            records.append(rec)
    one, full = C.reference_predictors(seed, fold, proper, cal, held)
    qcal, qheld = d['train'][:, cal].reshape(-1, 174), d['evaluation'][:, held].reshape(-1, 174)
    cm = np.column_stack((one(qcal[:, :107]), full(qcal)))
    hm = np.column_stack((one(qheld[:, :107]), full(qheld)))
    control = E.control_features(d['train'][:, proper].reshape(-1, 174), base_oof.reshape(-1, 2))
    cc, hc = E.control_features(qcal, cm), E.control_features(qheld, hm)
    target = np.tile(d['y'][proper], 4)
    cy, cids = np.tile(d['y'][cal], 4), np.tile(d['ids'][cal], 4)
    pcal, pheld = {'blend_frozen': cc[:, 0]}, {'blend_frozen': hc[:, 0]}
    head = E.head_fit(control, target - control[:, 0])
    heads = {'blend_control': head}
    pcal['blend_control'], pheld['blend_control'] = cc[:, 0] + E.head_predict(head, cc), hc[:, 0] + E.head_predict(head, hc)
    native = []
    for kind in ('gru', 'tcn'):
        model, normalization, rec = fit(kind, seed, fold, 3, proper, smoke=smoke)
        records.append(rec)
        cp = predict(model, cal, d['train_conditions'], normalization)[:, :, 1].reshape(-1)
        hp = predict(model, held, np.arange(len(d['names'])), normalization)
        native.append(hp)
        pcal[kind], pheld[kind] = cp, hp[:, :, 1].reshape(-1)
        pcal[f'fixed_{kind}'] = .5 * (cc[:, 0] + cp)
        pheld[f'fixed_{kind}'] = .5 * (hc[:, 0] + pheld[kind])
        tx = np.column_stack((control, temporal_oof[kind].reshape(-1) - control[:, 0]))
        cx, hx = np.column_stack((cc, cp - cc[:, 0])), np.column_stack((hc, pheld[kind] - hc[:, 0]))
        head = E.head_fit(tx, target - control[:, 0])
        heads[f'learned_{kind}'] = head
        pcal[f'learned_{kind}'] = cc[:, 0] + E.head_predict(head, cx)
        pheld[f'learned_{kind}'] = hc[:, 0] + E.head_predict(head, hx)
    predictions, thresholds, radii, rates = [], [], [], []
    for arm in ARMS:
        radius = C.calibration(pcal[arm], cy, cids)
        cp = C.distribution(pcal[arm], radius)
        predictions.append(C.distribution(pheld[arm], radius).reshape(len(d['names']), len(held), 6))
        levels = [C.threshold(cp[:, 3], cy, b) for b in C.BUDGETS]
        thresholds.append(levels)
        rates.append([float(np.mean(-cp[cy > 0, 3] <= t)) for t in levels])
        radii.append(radius)
    with np.load(ROOT / 'p4_groups/blend_additions_20261002/predictions.npz') as z:
        si = C.SEEDS.index(seed)
        np.testing.assert_allclose(predictions[0], z['predictions'][0, si, :, held].transpose(1, 0, 2), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(thresholds[0], z['thresholds'][0, si, fold], atol=1e-12, rtol=1e-12)
    C.save_npz(path, predictions=np.stack(predictions), thresholds=thresholds, native_quantiles=np.stack(native),
        records=json.dumps(records), heads=np.frombuffer(C.cloudpickle.dumps(heads), dtype=np.uint8),
        calibration=json.dumps(radii), calibration_false_warning=rates, proper_rows=proper, calibration_rows=cal,
        held_rows=held, fingerprint=json.dumps(d['fingerprint']), base_oof=base_oof,
        gru_oof=temporal_oof['gru'], tcn_oof=temporal_oof['tcn'])
    print('TEMPORAL_FOLD_COMPLETE', seed, fold, len(held), flush=True)
    return path


def run(smoke=False, workers=2):
    d = setup()
    record = json.loads((OUT / 'input_manifest.json').read_text())
    assert record['fingerprint'] == d['fingerprint']
    assert C.sha256(OUT / 'signals.npy') == record['signals_hash']
    assert C.sha256(OUT / 'lengths.npy') == record['lengths_hash']
    jobs = [(42, 0)] if smoke else [(s, f) for s in C.SEEDS for f in range(5)]
    if d['device'].type == 'cuda':
        if workers != 1:
            raise ValueError('CUDA comparison requires --workers 1; do not fork CUDA workers')
        for seed, fold in jobs:
            fold_job(seed, fold, smoke)
    else:
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            futures = {pool.submit(fold_job, s, f, smoke): (s, f) for s, f in jobs}
            for future in as_completed(futures):
                future.result()
    seeds = [42] if smoke else C.SEEDS
    selected = np.flatnonzero(d['folds'] == 0) if smoke else np.arange(len(d['y']))
    p = np.full((len(ARMS), len(seeds), len(d['names']), len(selected), 6), np.nan)
    t = np.full((len(ARMS), len(seeds), 5, len(C.BUDGETS)), np.nan)
    native = np.full((2, len(seeds), len(d['names']), len(selected), 3), np.nan)
    records, partitions, budgets = [], [], []
    for s, f in jobs:
        path = OUT / f'{"smoke_" if smoke else ""}fold_seed{s}_fold{f}.npz'
        with np.load(path) as z:
            held = z['held_rows']; si = seeds.index(s)
            p[:, si, :, np.searchsorted(selected, held)] = z['predictions'].transpose(2, 0, 1, 3)
            native[:, si, :, np.searchsorted(selected, held)] = z['native_quantiles'].transpose(2, 0, 1, 3)
            t[:, si, f] = z['thresholds']
            records += json.loads(str(z['records']))
            partitions.append(dict(seed=s, fold=f, **{k: z[k].tolist() for k in ('proper_rows', 'calibration_rows', 'held_rows')}))
            budgets.append(dict(seed=s, fold=f, rates=z['calibration_false_warning'].tolist()))
    assert np.isfinite(p).all() and np.isfinite(native).all()
    for _, f in jobs:
        assert np.isfinite(t[:, :, f]).all()
    prefix = 'smoke_' if smoke else ''
    C.save_npz(OUT / f'{prefix}predictions.npz', predictions=p, thresholds=t, native_quantiles=native,
        y=d['y'][selected], ids=d['ids'][selected], source_indices=d['rows'][selected], row_fold=d['folds'][selected],
        conditions=d['names'], arms=ARMS, seeds=seeds, fields=C.FIELDS, fingerprint=json.dumps(d['fingerprint']))
    C.save_json(OUT / f'{prefix}fits.json', dict(fingerprint=d['fingerprint'], fits=records,
        partitions=partitions, calibration_budgets=budgets, all_finite=True))
    print('TEMPORAL_COMPARISON_COMPLETE', p.shape, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    with E.threadpool_limits(limits=1):
        prepare() if args.prepare else run(args.smoke, args.workers)
