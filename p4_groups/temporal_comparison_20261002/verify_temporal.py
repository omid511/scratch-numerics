"""Pin cached tables before full scoring; replay real waveform inference afterward."""
import argparse
import json
import time
import experiment_temporal as T
C, np, torch, OUT = T.C, T.np, T.torch, T.OUT


def freeze_inputs():
    sources = C.signature()['sources']
    models = C.helpers.reference_manifest()
    manifest = {'sources': sources, 'frozen_base_models': models}
    path = OUT / 'frozen_table_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'frozen table/model provenance changed'
    else:
        C.save_json(path, manifest)
    print('TEMPORAL_TABLES_FROZEN', len(sources), len(models['models']), flush=True)


def verify(smoke=False):
    freeze_inputs()
    d = T.setup()
    torch.set_num_threads(1)
    d['signals'] = np.load(OUT / 'signals.npy', mmap_mode='r')
    d['lengths'] = np.load(OUT / 'lengths.npy', mmap_mode='r')
    manifest = json.loads((OUT / 'input_manifest.json').read_text())
    assert manifest['fingerprint'] == d['fingerprint']
    assert C.sha256(OUT / 'signals.npy') == manifest['signals_hash']
    assert C.sha256(OUT / 'lengths.npy') == manifest['lengths_hash']
    prefix = 'smoke_' if smoke else ''
    with np.load(OUT / f'{prefix}predictions.npz') as z:
        rows = z['source_indices']; y = z['y']; ids = z['ids']; folds = z['row_fold']
        names = z['conditions'].tolist(); seeds = z['seeds'].tolist()
        predictions = z['predictions']; native = z['native_quantiles']; thresholds = z['thresholds']
        assert z['arms'].tolist() == T.ARMS and names == d['names']
    selected = np.searchsorted(d['rows'], rows)
    np.testing.assert_array_equal(d['rows'][selected], rows)
    np.testing.assert_array_equal(d['y'][selected], y)
    np.testing.assert_array_equal(d['ids'][selected], ids)
    if not smoke:
        assert len(y) == 4454 and len(np.unique(ids)) == 77 and seeds == C.SEEDS
    samples = []
    # Boundary failures plus a stable, an unstable and an ordinary sample.
    wanted = [3696, 5198, int(rows[np.flatnonzero(y >= .15)[0]]),
              int(rows[np.flatnonzero((y < 0) & (y > -.01))[0]]), int(rows[len(rows) // 2])]
    for source in dict.fromkeys(wanted):
        match = np.flatnonzero(rows == source)
        if not len(match):
            continue
        k, local = int(selected[match[0]]), int(match[0])
        fold = int(folds[local]); seed = seeds[0]
        path = OUT / f'{prefix}fold_seed{seed}_fold{fold}.npz'
        with np.load(path) as z:
            proper, cal, held = (z[v] for v in ('proper_rows', 'calibration_rows', 'held_rows'))
            heads = C.cloudpickle.loads(z['heads'].tobytes())
            radii = json.loads(str(z['calibration']))
        assert not set(d['ids'][proper]) & set(d['ids'][held])
        one, full = C.reference_predictors(seed, fold, proper, cal, held)
        did = str(d['ids'][k])
        with np.load(T.ROOT / 'p4_sensor_candidates' / f'{did}.npz') as z:
            raw_index = int(np.flatnonzero(z['source_indices'] == source)[0])
            raw, sigma = z['raw'][raw_index], z['noise_scale'][raw_index]
        gains = np.array(json.loads((T.ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
        for ci in [names.index('clean'), names.index('iid1_new949'), names.index('correlated5_new949_950')]:
            signal, dt = T.acquisition.measured_signal(raw, sigma, source, names[ci], d['layout'], gains,
                float(d['dt'][k]), {}, {})
            hits = np.flatnonzero(np.any(abs(signal) >= 50., axis=0))
            usable = int(hits[0]) if len(hits) else 512
            signal[:, usable:] = 0
            np.testing.assert_array_equal(signal, d['signals'][ci, k])
            np.testing.assert_allclose(np.log(dt), d['side'][ci, k, -1], atol=1e-6, rtol=0)
            q = d['evaluation'][ci, k:k+1]
            members = np.column_stack((one(q[:, :107]), full(q)))
            control = T.E.control_features(q, members)
            points = {'blend_frozen': control[:, 0], 'blend_control': control[:, 0] + T.E.head_predict(heads['blend_control'], control)}
            for ki, kind in enumerate(('gru', 'tcn')):
                model, scaling, _ = T.fit(kind, seed, fold, 3, proper, smoke=smoke)
                mean, scale, active = scaling
                side = (d['side'][ci, k:k+1] - mean) / scale
                side[:, ~active] = 0
                x = torch.from_numpy(signal[None, :, :max(1, usable)]).to(d['device'])
                length = torch.tensor([max(1, usable)], dtype=torch.long, device=d['device'])
                with torch.inference_mode():
                    begin = time.perf_counter()
                    actual = model(x, length, torch.from_numpy(side).to(d['device'])).cpu().numpy()
                    elapsed = time.perf_counter() - begin
                np.testing.assert_allclose(actual[0], native[ki, 0, ci, local], atol=1e-5, rtol=1e-5)
                points[kind] = actual[:, 1]
                points[f'fixed_{kind}'] = .5 * (control[:, 0] + points[kind])
                hx = np.column_stack((control, points[kind] - control[:, 0]))
                points[f'learned_{kind}'] = control[:, 0] + T.E.head_predict(heads[f'learned_{kind}'], hx)
                samples.append(dict(source=source, condition=names[ci], kind=kind, seconds=elapsed))
            actual = np.stack([C.distribution(points[arm], radii[ai])[0] for ai, arm in enumerate(T.ARMS)])
            error = float(np.max(abs(actual - predictions[:, 0, ci, local])))
            np.testing.assert_allclose(actual, predictions[:, 0, ci, local], atol=2e-4, rtol=2e-4)
            print('TEMPORAL_WAVEFORM_REPLAY', source, names[ci], error, flush=True)
    assert samples
    proof = dict(real_waveform_replays=len(samples) // 2, temporal_inferences=len(samples),
        source_indices=sorted(set(s['source'] for s in samples)), timings=samples,
        fixed_tables_hash=C.sha256(OUT / 'frozen_table_manifest.json'), predictions_hash=C.sha256(OUT / f'{prefix}predictions.npz'),
        source_hash=C.sha256(T.ROOT / 'src/mechanics/p4_margin_estimation/temporal.py'),
        limitation='Cached measured side features replayed; timings exclude modal/spatial extraction. Single-row/batch numerical tolerance explicitly checked.')
    C.save_json(OUT / f'{prefix}verification.json', proof)
    print('TEMPORAL_WAVEFORM_VERIFICATION_COMPLETE', proof['real_waveform_replays'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-inputs', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with T.E.threadpool_limits(limits=1):
        freeze_inputs() if args.freeze_inputs else verify(args.smoke)
