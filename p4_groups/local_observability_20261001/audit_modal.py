"""Diagnostic-only solver comparisons for frozen TRAIN predictions."""
from audit_failures import *
from prepare_features import measured_signal
from scipy.optimize import linear_sum_assignment
from mechanics.p4_margin_estimation.modal_features import estimate_modes, modal_spatial_row, modal_spatial_addition_row

FIELDS = ['critical_recovered', 'critical_visible', 'critical_in_least_stable_pair',
          'lead_is_critical', 'critical_alpha_error', 'critical_frequency_error_hz',
          'critical_damping_error', 'critical_frequency_unique', 'label_represented',
          'label_matches_retained_critical', 'fit_residual', 'usable_fraction']


def diagnostic(modes, truth, label_omega, label_represented):
    a, f, e = modes['alpha'], modes['frequency'], modes['energy_fraction']
    freq = truth.imag / (2 * np.pi)
    tolerance = np.maximum(2., .05 * freq)
    visible = np.flatnonzero(e >= .01)
    order = visible[np.lexsort((f[visible], -a[visible]))]
    out = np.zeros(len(FIELDS))
    out[4:7] = np.nan
    out[7] = np.all(abs(freq[1:] - freq[0]) > 2 * tolerance[0])
    out[8] = label_represented
    out[9] = abs(abs(label_omega) / (2 * np.pi) - freq[0]) < 1e-3 * max(1., freq[0])
    out[10:] = modes['relative_fit_error'], modes['usable_fraction']
    if len(a):
        cost = abs(f[:, None] - freq[None]) / tolerance[None]
        estimated, actual = linear_sum_assignment(cost)
        hits = estimated[(actual == 0) & (cost[estimated, actual] <= 1.)]
        if len(hits):
            i = int(hits[0])
            out[:4] = 1, e[i] >= .01, i in order[:2], bool(len(order) and i == order[0])
            out[4:6] = a[i] - truth[0].real, f[i] - freq[0]
            out[6] = -a[i] / np.hypot(a[i], 2 * np.pi * f[i]) + truth[0].real / abs(truth[0])
    return out


def summarize(values, mask):
    x = values[mask]
    result = {'acquisitions': len(x)}
    for i, field in enumerate(FIELDS):
        finite = x[:, i][np.isfinite(x[:, i])]
        result[field] = {'n': len(finite), 'mean': float(finite.mean()) if len(finite) else None,
                         'median': float(np.median(finite)) if len(finite) else None,
                         'abs_p95': float(np.quantile(abs(finite), .95)) if len(finite) else None}
    return result


def run(data, additions, recorded):
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    severe = np.any(recorded - y > .03, axis=(0, 1))
    primary = np.array([name in ['clean', *helpers.FRESH] for name in names])
    selected = severe | primary[:, None]
    fingerprint = {'script': sha256(Path(__file__)), 'protocol': sha256(OUT / 'audit_protocol.json'),
                   'predictions': sha256(OUT / 'predictions.npz'), 'features': sha256(OUT / 'features.npz'),
                   'source': signature()}
    with np.load(ROOT / 'p4_dataset_saturation/metadata_arrays.npz') as z:
        label_omega, represented = z['label_omegas'][rows], z['label_represented'][rows]
    gains = np.array(json.loads((ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    values = np.full((*selected.shape, len(FIELDS)), np.nan)
    named, max_delta = {}, 0.
    ridge = helpers.PhysicsFeatureRidge()
    for did in np.unique(ids):
        source = ROOT / 'p4_sensor_candidates' / f'{did}.npz'
        digest = sha256(source)
        path = OUT / f'audit_modal_{did}.npz'
        sig = {**fingerprint, 'raw': digest}
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint'])) == sig, 'stale audit modal shard'
                target, block, delta = z['target'], z['values'], float(z['replay_delta'])
                named.update(json.loads(str(z['named'])))
        else:
            target = np.flatnonzero(ids == did)
            block = np.full((len(names), len(target), len(FIELDS)), np.nan)
            delta, cases = 0., {}
            with np.load(source) as z:
                raw, scale, source_rows, truth = z['raw'], z['noise_scale'], z['source_indices'], z['poles']
            np.testing.assert_array_equal(source_rows, rows[target])
            for j, k in enumerate(target):
                white, shared = {}, {}
                for ci in np.flatnonzero(selected[:, k]):
                    signal, reported_dt = measured_signal(raw[j], scale[j], int(rows[k]), names[ci], layout, gains, float(dt[k]), white, shared)
                    modes = estimate_modes(signal, reported_dt, include_shapes=True)
                    row = np.r_[helpers.base.base_feature_row(signal, reported_dt, modes, ridge),
                                modal_spatial_row(modes, pairs), modal_spatial_addition_row(modes, pairs)]
                    np.testing.assert_allclose(row, evaluation[ci, k], atol=1e-10, rtol=1e-10)
                    delta = max(delta, float(abs(row - evaluation[ci, k]).max()))
                    block[ci, j] = diagnostic(modes, truth[j], label_omega[k], represented[k])
                    if rows[k] in (3696, 5198) and names[ci] in ('clean', 'iid1_new949', 'iid1_new951'):
                        cases[f'{rows[k]}:{names[ci]}'] = {'diagnostics': {field: float(value) if np.isfinite(value) else None for field, value in zip(FIELDS, block[ci, j])},
                            'truth_poles': [[float(p.real), float(p.imag / (2*np.pi))] for p in truth[j]],
                            'estimated_modes': np.column_stack([modes['alpha'], modes['frequency'], modes['energy_fraction']]).tolist()}
            save_npz(path, target=target, values=block, replay_delta=delta, named=json.dumps(cases), fingerprint=json.dumps(sig))
            named.update(cases)
        values[:, target] = block
        max_delta = max(max_delta, delta)
        print('AUDIT_MODAL_DESIGN', did, int(selected[:, target].sum()), flush=True)
    assert np.isfinite(values[selected][:, :4]).all()
    assert signature() == fingerprint['source']
    result = {'fields': FIELDS, 'feature_replay_max_delta': max_delta, 'selected_acquisitions': int(selected.sum()),
              'named': named, 'populations': {}, 'fingerprint': fingerprint}
    for scope, condition_mask in [('all37', np.ones(len(names), bool)), ('primary9', primary)]:
        for label, margin_mask in [('all', np.ones(len(y), bool)), ('unstable_last1pct', (y < 0) & (y > -.01)),
                                  ('stable_last1pct', (y > 0) & (y < .01)), ('far_stable', y >= .15)]:
            mask = condition_mask[:, None] & margin_mask[None]
            result['populations'][f'{scope}:{label}:severe_union'] = summarize(values, mask & severe)
            if scope == 'primary9':
                result['populations'][f'{scope}:{label}:nonsevere'] = summarize(values, mask & ~severe)
    for ai, arm in enumerate(ARMS[:3]):
        for scope, conditions in [('all37', np.ones(len(names), bool)), ('primary9', primary)]:
            mask = conditions[:, None] & np.any(recorded[ai] - y > .03, axis=0)
            result['populations'][f'{scope}:{arm}:severe'] = summarize(values, mask)
    save_npz(OUT / 'audit_modal_values.npz', values=values, selected=selected, severe_union=severe, fields=FIELDS,
             conditions=names, y=y, ids=ids, source_indices=rows, fingerprint=json.dumps(fingerprint))
    save_json(OUT / 'audit_modal_results.json', result)
    print('AUDIT_MODAL_COMPLETE', result['selected_acquisitions'], 'replay_max_delta', max_delta, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(*load_audit_data())
