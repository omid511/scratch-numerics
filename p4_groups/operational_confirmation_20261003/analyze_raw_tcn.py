"""Report fixed TCN trials against unchanged development targets and references.

No fits, new predictions, feature diagnostics or held-design scoring.
"""
import json
import numpy as np
import experiment_raw_tcn as T


def gates(values, comparisons, references, noise):
    overall, boundary = (.010, .005) if noise == 'clean' else (.015, .010)
    checks = dict(overall_absolute=values['slices']['all']['mean'] <= overall,
        boundary_absolute=values['slices']['boundary']['mean'] <= boundary,
        optimism_absolute=values['optimism']['mean'] <= .05,
        stable_sign_absolute=values['stable_sign']['mean'] <= .10 if values['stable_sign']['mean'] is not None else None,
        unstable_sign_absolute=values['unstable_sign']['mean'] <= .10 if values['unstable_sign']['mean'] is not None else None)
    relative = {}
    for name, reference in references.items():
        guard = {part + '_improved': comparisons[part][name]['mean'] < 0 for part in ('all', 'boundary')}
        guard['clear_paired_improvement'] = all(comparisons[part][name]['ci95'][1] < 0 for part in ('all', 'boundary'))
        for part in ('boundary_current', 'profile_retreat', 'farther_stable', 'farther_stable15'):
            guard[part] = comparisons[part][name]['mean'] <= (.005 if part.startswith('farther_stable') else 0)
        for part in ('optimism', 'stable_sign', 'unstable_sign'):
            candidate, baseline = values[part]['mean'], reference[part]['mean']
            guard[part] = candidate <= baseline if candidate is not None and baseline is not None else None
        relative[name] = guard
    return dict(targets=dict(all=overall, boundary=boundary), absolute=checks,
        by_reference=relative, all_descriptive_checks_pass=all(value is True for value in checks.values()) and
        all(value is True for guard in relative.values() for key, value in guard.items() if key != 'clear_paired_improvement'))


def run():
    protocol = T.contract()
    training = json.loads((T.OUT/'training.json').read_text())
    results = json.loads((T.OUT/'results.json').read_text())
    summary = json.loads((T.OUT/'comparison_summary.json').read_text())
    digest = T.R.sha(T.OUT/'protocol.json')
    if training['fit_count'] != 18 or any(payload['protocol_sha256'] != digest for payload in (training, results)):
        raise RuntimeError('Incomplete or mismatched TCN trial artifacts')
    if training['predictions_sha256'] != T.R.sha(T.OUT/'predictions.npz'):
        raise RuntimeError('TCN prediction artifact changed')
    for name, expected in summary['sources'].items():
        if T.R.sha(T.OUT/name) != expected:
            raise RuntimeError(f'TCN report source changed: {name}')
    data = T.D.load_data()
    records = data['records']
    designs = np.array([row['design'] for row in records])
    noises = np.array([row['noise'] for row in records])
    held = np.isin(designs, protocol['split']['held'])
    with np.load(T.OUT/'predictions.npz', allow_pickle=False) as archive:
        points = archive['predictions'].copy()
        folds, arms, seeds = (archive[key].copy() for key in ('folds', 'arms', 'seeds'))
    if not np.isnan(points[:,held]).all() or not np.isfinite(points[:,~held]).all():
        raise RuntimeError('TCN held protection or allowed coverage changed')
    checks = 0
    for fold, ids in enumerate(protocol['evaluation_folds']):
        for noise in protocol['primary_noise']:
            population = np.isin(designs, ids) & (noises == noise)
            for arm in T.ARMS:
                index = [np.flatnonzero((folds == fold) & (arms == arm) & (seeds == seed)).item() for seed in T.SEEDS]
                for part, mask in (('all', np.ones(data['truth'].shape, bool)),
                        ('boundary', (data['truth'] > -.02) & (data['truth'] < .05))):
                    saved = results['metrics'][f'fold{fold}/evaluation/{noise}'][arm][part]['per_design']
                    for design in ids:
                        selected = population[:,None] & mask & (designs == design)[:,None]
                        direct = abs(points[index][:,selected] - data['truth'][selected]).mean(axis=1)
                        np.testing.assert_allclose(direct, saved[design]['mae_by_seed'], rtol=0, atol=1e-12)
                        if int(selected.sum()) != saved[design]['frames']:
                            raise RuntimeError('TCN original scoring denominator changed')
                        checks += 1
    output = {}
    for noise in protocol['primary_noise']:
        source = summary['primary'][noise]
        output[noise] = {}
        for arm in T.ARMS:
            controls = [*T.A.ARMS, *[name for name in T.ARMS if name != arm]]
            comparisons = {part:{control:T.A.paired(T.A.joined(results['metrics'], noise, arm, part),
                T.A.joined(results['metrics'], noise, control, part)) for control in controls}
                for part in ('all', 'boundary', 'boundary_current', 'profile_retreat', 'farther_stable', 'farther_stable15')}
            values = source['arms'][arm]
            output[noise][arm] = dict(metrics=values, comparisons=comparisons,
                gates=gates(values, comparisons, {control:source['arms'][control] for control in controls}, noise))
            print('RAW_TCN_GATES', noise, arm, json.dumps(output[noise][arm]['gates']), flush=True)
    T.R.save_json(T.OUT/'guarded_comparison.json', dict(primary=output,
        independent_original_denominator_checks=checks, held_predictions_absent=True,
        sources={name:T.R.sha(T.OUT/name) for name in ('protocol.json', 'training.json', 'results.json', 'comparison_summary.json', 'predictions.npz')},
        analysis_sha256=T.R.sha(__file__),
        limits='Descriptive development gates only; no promotion, qualification, calibrated warning or independent physical-transfer credit. Paired intervals resample 12 represented designs, retain all 3 seeds, and exclude refit/seed uncertainty and multiplicity correction.'))
    print('RAW_TCN_GUARDED_COMPARISON_COMPLETE', checks, flush=True)


if __name__ == '__main__':
    run()
