"""Secondary joint-pole proximity diagnostic; not a confidence rule."""
from audit_failures import *


def run():
    data, additions, recorded = load_audit_data()
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    truth = np.empty((len(rows), 8), dtype=complex)
    for did in np.unique(ids):
        idx = np.flatnonzero(ids == did)
        with np.load(ROOT / 'p4_sensor_candidates' / f'{did}.npz') as z:
            np.testing.assert_array_equal(z['source_indices'], rows[idx])
            truth[idx] = z['poles']
    # Modal energy-top3 and the two least-stable visible modes, from existing encoding.
    columns = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15], [27, 28, 29], [30, 31, 32]])
    encoded = evaluation[:, :, columns]
    poles = encoded[..., 0] + 2j * np.pi * encoded[..., 1]
    available = encoded[..., 2] > 0
    distances = abs(poles[..., None] - truth[None, :, None])
    nearest_truth = distances.argmin(axis=-1)
    critical_distance = distances[..., 0]
    critical_distance[~available] = np.inf
    best = critical_distance.argmin(axis=-1)
    best_distance = np.take_along_axis(critical_distance, best[..., None], axis=-1)[..., 0]
    closest_id = np.take_along_axis(nearest_truth, best[..., None], axis=-1)[..., 0]
    selected_pole = np.take_along_axis(poles, best[..., None], axis=-1)[..., 0]
    true_gap = abs(truth[:, 1:] - truth[:, :1]).min(axis=1)
    ratio = best_distance / true_gap[None]
    severe = np.any(recorded - y > .03, axis=(0, 1))
    primary = np.array([name in ['clean', *helpers.FRESH] for name in names])
    result = {'method': 'Nearest encoded complex pole to retained critical pole, compared with separation of the true critical pole from the other seven. Uses growth AND angular frequency; energy-top3 plus least-stable visible pair only, not all estimated modes. No reliability threshold selected.',
              'reason': 'Frequency-only separation is absent in every primary unstable-last1% severe acquisition; retain frequency matching but do not mistake a broad frequency gate for unique identity.',
              'populations': {}, 'named': {}}
    for scope, cm in [('all37', np.ones(len(names), bool)), ('primary9', primary)]:
        for label, ym in [('all', np.ones(len(y), bool)), ('unstable_last1pct', (y < 0) & (y > -.01))]:
            for outcome in ('severe', 'nonsevere'):
                mask = cm[:, None] & ym[None] & (severe if outcome == 'severe' else ~severe)
                finite = mask & np.isfinite(best_distance)
                r, a = ratio[finite], (selected_pole.real - truth[None, :, 0].real)[finite]
                result['populations'][f'{scope}:{label}:{outcome}'] = {'acquisitions': int(mask.sum()),
                    'encoded_pole_available': int(finite.sum()), 'closest_true_index0_rate': float((closest_id[finite] == 0).mean()) if finite.any() else None,
                    'critical_distance_over_true_gap_median': float(np.median(r)) if len(r) else None,
                    'critical_distance_over_true_gap_p95': float(np.quantile(r, .95)) if len(r) else None,
                    'critical_distance_over_true_gap_max': float(np.max(r)) if len(r) else None,
                    'abs_alpha_error_median': float(np.median(abs(a))) if len(a) else None,
                    'abs_alpha_error_p95': float(np.quantile(abs(a), .95)) if len(a) else None}
    for source in (3696, 5198):
        k = int(np.flatnonzero(rows == source)[0])
        ci = names.index('iid1_new949')
        result['named'][str(source)] = {'true_critical_alpha': float(truth[k, 0].real),
            'true_critical_frequency_hz': float(truth[k, 0].imag/(2*np.pi)),
            'true_nearest_complex_pole_gap_per_s': float(true_gap[k]),
            'encoded_critical_alpha': float(selected_pole[ci, k].real),
            'encoded_critical_frequency_hz': float(selected_pole[ci, k].imag/(2*np.pi)),
            'critical_distance_per_s': float(best_distance[ci, k]), 'distance_over_gap': float(ratio[ci, k]),
            'closest_true_pole_index': int(closest_id[ci, k]), 'encoded_slot': int(best[ci, k]),
            'measured_lead_closest_true_index': int(nearest_truth[ci, k, 3])}
    save_json(OUT / 'audit_complex_mode_results.json', result)
    print('AUDIT_COMPLEX_MODE_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
