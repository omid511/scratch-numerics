"""Consolidate finished audit artifacts and verify diagnostic isolation."""
from audit_failures import *


def run():
    data, additions, recorded = load_audit_data()
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    fitting = json.loads((OUT / 'audit_fitting_results.json').read_text())
    modal = json.loads((OUT / 'audit_modal_results.json').read_text())
    neighbor = json.loads((OUT / 'audit_neighborhood_results.json').read_text())
    result = {'cohort': json.loads((OUT / 'audit_cohort.json').read_text()),
              'named_replay': replay(data, additions, recorded),
              'fitting': fitting['aggregate'], 'modal_unique_frequency': {},
              'named_modal': modal['named'], 'named_neighbors': neighbor['named'], 'named_seen_design': {},
              'limitations': json.loads((OUT / 'audit_protocol.json').read_text())['limits']}
    result['joint_complex_pole_diagnostic'] = json.loads((OUT / 'audit_complex_mode_results.json').read_text())
    with np.load(OUT / 'audit_modal_values.npz') as z:
        values, selected, severe = z['values'], z['selected'], z['severe_union']
        np.testing.assert_array_equal(severe, np.any(recorded - y > .03, axis=(0, 1)))
        fields = z['fields'].tolist()
    primary = np.array([n in ['clean', *helpers.FRESH] for n in names])
    for scope, cm in [('all37', np.ones(len(names), bool)), ('primary9', primary)]:
        for label, ym in [('all', np.ones(len(y), bool)), ('unstable_last1pct', (y < 0) & (y > -.01))]:
            for outcome in ('severe', 'nonsevere'):
                if scope == 'all37' and outcome == 'nonsevere':
                    continue
                mask = cm[:, None] & ym[None] & (severe if outcome == 'severe' else ~severe)
                assert selected[mask].all()
                x = values[mask]
                unique = x[:, fields.index('critical_frequency_unique')] == 1
                matched = unique & (x[:, fields.index('critical_recovered')] == 1)
                alpha = x[matched, fields.index('critical_alpha_error')]
                result['modal_unique_frequency'][f'{scope}:{label}:{outcome}'] = {
                    'acquisitions': len(x), 'unique_frequency': int(unique.sum()), 'unique_and_recovered': int(matched.sum()),
                    'unique_recovery_rate': float(matched.sum()/unique.sum()) if unique.any() else None,
                    'matched_abs_alpha_median': float(np.median(abs(alpha))) if len(alpha) else None,
                    'matched_abs_alpha_p95': float(np.quantile(abs(alpha), .95)) if len(alpha) else None,
                    'unique_recovered_in_pair_rate': float(x[matched, fields.index('critical_in_least_stable_pair')].mean()) if matched.any() else None}
    with np.load(OUT / 'audit_neighborhoods.npz') as z:
        refs, ci, ki = z['references'], z['condition_indices'], z['clip_indices']
        assert len(ki) == int(severe.sum())
        assert severe[ci, ki].all()
        index = np.searchsorted(rows, refs)
        np.testing.assert_array_equal(rows[index], refs)
        for fold in range(5):
            proper, cal, held = split(ids, folds, fold)
            selected_queries = folds[ki] == fold
            block = index[:, :, selected_queries]
            assert np.isin(block, proper).all(), 'neighbor crosses proper-fit boundary'
            reference_ids = ids[block]
            assert np.all([len(set(v)) == 5 for v in reference_ids.reshape(-1, 5)])
    for source, held_seed in ((3696, 43), (5198, 42)):
        result['named_seen_design'][str(source)] = {}
        for arm in ARMS[:3]:
            cases = [r for r in fitting['named_seen_design'] if r['source'] == source and r['seed'] == held_seed and r['arm'] == arm]
            assert len(cases) > 0
            errors = np.array([r['error'] for r in cases])
            result['named_seen_design'][str(source)][arm] = {'seed': held_seed, 'proper_fit_model_folds': [r['model_fold'] for r in cases],
                'mean_error': float(errors.mean()), 'min_error': float(errors.min()), 'max_error': float(errors.max())}
    result['verification'] = {'modal_waveform_feature_replay_max_delta': modal['feature_replay_max_delta'],
        'modal_acquisitions_replayed': modal['selected_acquisitions'], 'neighbor_queries': len(ki),
        'all_neighbor_references_proper_fit_only': True, 'five_distinct_design_neighbors': True,
        'original_four_view_held_predictions_replayed': True, 'named_predictions_replayed': True,
        'trained_new_models': 0, 'reserved_designs_scored': 0}
    artifacts = ['audit_protocol.json', 'audit_failures.py', 'audit_mapping.py', 'audit_modal.py', 'audit_cohort.json',
                 'audit_fitting_results.json', 'audit_modal_results.json', 'audit_modal_values.npz',
                 'audit_neighborhood_results.json', 'audit_neighborhoods.npz', 'audit_evidence.py',
                 'audit_complex_modes.py', 'audit_complex_mode_results.json',
                 'predictions.npz', 'features.npz']
    result['hashes'] = {f: sha256(OUT / f) for f in artifacts}
    result['frozen_reference_signature'] = signature()
    result['model_hashes'] = {p.name: sha256(p) for arm in ARMS[1:3] for seed in SEEDS for fold in range(5)
                              for width in (107, 174) for p in [OUT / f'{arm}_{width}_seed{seed}_fold{fold}.npz']}
    save_json(OUT / 'audit_results.json', result)
    print('AUDIT_EVIDENCE_COMPLETE', json.dumps(result['verification']), flush=True)
    print('ACTUAL_FITTING_SUMMARY', json.dumps(result['fitting']), flush=True)
    print('NAMED_SEEN_DESIGN_ERRORS', json.dumps(result['named_seen_design']), flush=True)
    print('UNIQUE_FREQUENCY_MODAL_SUMMARY', json.dumps(result['modal_unique_frequency']), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
