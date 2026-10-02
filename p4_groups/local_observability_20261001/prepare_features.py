"""Replay existing acquisitions and add label-free local-band descriptors."""
import re
from contract import *
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize


def measured_signal(raw, sigma, source_index, name, layout, gains, dt, white, shared):
    def draw(seed, common=False):
        cache = shared if common else white
        if seed not in cache:
            size = (1, raw.shape[1]) if common else raw.shape
            cache[seed] = np.random.default_rng(np.random.SeedSequence([seed, int(source_index)])).normal(size=size)
        return cache[seed]
    if name == 'clean':
        measured = raw
    elif name in ('gain_bias', 'dt_minus2pct', 'dt_plus2pct'):
        measured = raw + .01 * sigma * draw(929)
    elif name == 'correlated_train_draw':
        measured = raw + .01 * sigma * (np.sqrt(.2) * draw(929) + np.sqrt(.8) * draw(931, True))
    else:
        seeds = [int(v) for v in re.findall(r'\d+', name) if int(v) >= 900]
        level = .05 if name.startswith(('iid5_', 'correlated5_')) else .01
        if name.startswith('iid') and len(seeds) == 1:
            noise = draw(seeds[0])
        elif name.startswith('correlated') and len(seeds) == 2:
            noise = np.sqrt(.2) * draw(seeds[0]) + np.sqrt(.8) * draw(seeds[1], True)
        else:
            raise ValueError(f'unknown frozen acquisition: {name}')
        measured = raw + level * sigma * noise
    normalized = causal_calibration_normalize(measured, calibration_samples=51)[layout]
    if name == 'gain_bias':
        normalized = normalized * gains[:, None]
    reported_dt = dt * (.98 if name == 'dt_minus2pct' else 1.02 if name == 'dt_plus2pct' else 1.)
    return np.clip(normalized, -50., 50.).astype(np.float32), reported_dt


def main():
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = load_data()
    train_names = helpers.base.TRAIN_NAMES
    np.testing.assert_array_equal(train, evaluation[[names.index(n) for n in train_names]])
    gains = np.array(json.loads((ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    fingerprint = signature()
    lookup = {int(row): i for i, row in enumerate(rows)}
    control = np.full((len(names), len(rows), 12), np.nan)
    local = np.full_like(control, np.nan)
    raw_hashes = {}
    for did in np.unique(ids):
        source = ROOT / 'p4_sensor_candidates' / f'{did}.npz'
        raw_hashes[str(did)] = sha256(source)
        shard_signature = {**fingerprint, 'raw': raw_hashes[str(did)]}
        path = OUT / f'bands_{did}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint'])) == shard_signature, 'stale local descriptor shard'
                source_rows, c, q = z['source_indices'], z['control'], z['local']
        else:
            # Explicitly do not read poles, margins or oracle-mode identities.
            with np.load(source) as z:
                raw, sigma, source_rows = z['raw'], z['noise_scale'], z['source_indices']
            c = np.empty((len(names), len(source_rows), 12))
            q = np.empty_like(c)
            for j, index in enumerate(source_rows):
                k = lookup[int(index)]
                assert ids[k] == did
                white, shared = {}, {}
                for ci, name in enumerate(names):
                    signal, reported_dt = measured_signal(raw[j], sigma[j], int(index), name, layout, gains, float(dt[k]), white, shared)
                    hits = np.flatnonzero(np.any(np.abs(signal) >= 50., axis=0))
                    usable = int(hits[0]) if len(hits) else signal.shape[1]
                    np.testing.assert_allclose(usable / signal.shape[1], evaluation[ci, k, 20], atol=1e-12, rtol=0)
                    c[ci, j], q[ci, j] = feature_row(signal, reported_dt, evaluation[ci, k, FREQUENCY_COLUMNS])
            save_npz(path, control=c, local=q, source_indices=source_rows, fingerprint=json.dumps(shard_signature))
        target = np.array([lookup[int(row)] for row in source_rows])
        assert set(target) == set(np.flatnonzero(ids == did))
        control[:, target], local[:, target] = c, q
        print('Completed local acquisition', did, 'clips', len(source_rows), flush=True)
    assert np.isfinite(control).all() and np.isfinite(local).all()
    assert signature() == fingerprint, 'source/reference changed during extraction'
    names_global = [f'band{i}_{n}' for i in (1, 2) for n in GLOBAL_NAMES]
    names_local = [f'band{i}_{n}' for i in (1, 2) for n in LOCAL_NAMES]
    save_npz(OUT / 'features.npz', control=control, local=local,
             training_control=control[[names.index(n) for n in train_names]],
             training_local=local[[names.index(n) for n in train_names]],
             y=y, ids=ids, source_indices=rows, row_fold=folds, conditions=names,
             global_names=names_global, local_names=names_local, fingerprint=json.dumps(fingerprint), raw_hashes=json.dumps(raw_hashes))
    print('LOCAL_FEATURES_COMPLETE', control.shape, 'clean band availability', control[0, :, [2, 8]].mean(axis=1), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
