"""Report the fixed R/M comparison on original development denominators; no fits."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
OUT = SCRIPT / 'raw_modal_20261010'
SOURCE = SCRIPT / 'support_modal_20261009_v2'
ARMS = ('raw', 'mask', 'content_gru', 'ordered_concat', 'unordered_pool', 'framewise_reference')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def joined(metrics, noise, arm, stratum):
    records = {}
    for fold in range(3):
        part = metrics[f'fold{fold}/evaluation/{noise}'][arm][stratum]['per_design']
        if records.keys() & part.keys():
            raise RuntimeError('Evaluation design occurs in multiple folds')
        records.update(part)
    return records


def mae(records):
    values = np.array([r['mae_by_seed'] for r in records.values()], dtype=float)
    return dict(designs=len(records), frames=sum(r['frames'] for r in records.values()),
        by_seed=values.mean(axis=0).tolist() if len(values) else None,
        mean=float(values.mean()) if len(values) else None)


def contribution(whole, subset):
    """Keep each design's original stratum count; absent subsets contribute zero."""
    values, weight = np.zeros(3), 0.0
    for design, base in whole.items():
        if design in subset:
            record = subset[design]
            fraction = record['frames'] / base['frames']
            values += fraction * np.array(record['mae_by_seed']) / len(whole)
            weight += fraction / len(whole)
    return dict(original_weight=float(weight), by_seed=values.tolist(), mean=float(values.mean()))


def paired(a, b):
    if set(a) != set(b):
        raise RuntimeError('Paired design populations differ')
    differences = np.array([np.array(a[d]['mae_by_seed']) - b[d]['mae_by_seed'] for d in sorted(a)])
    draws = np.random.default_rng(42).integers(0, len(differences), (2000, len(differences)))
    bootstrap = differences.mean(axis=1)[draws].mean(axis=1)
    return dict(mean=float(differences.mean()), by_seed=differences.mean(axis=0).tolist(),
        ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
        improved_designs=int((differences.mean(axis=1) < 0).sum()),
        design_seed_delta={d: differences[i].tolist() for i, d in enumerate(sorted(a))},
        improved_every_design_seed=bool((differences < 0).all()))


def event(metrics, noise, arm, stratum, name):
    groups = [metrics[f'fold{fold}/evaluation/{noise}'][arm][stratum][name] for fold in range(3)]
    frames = sum(g['frames'] for g in groups)
    counts = np.sum([g['events_by_seed'] for g in groups], axis=0)
    rates = counts / frames if frames else None
    return dict(frames=frames, events_by_seed=counts.tolist(),
        rate_by_seed=rates.tolist() if rates is not None else None,
        mean=float(rates.mean()) if rates is not None else None)


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    training = json.loads((OUT/'training.json').read_text())
    results = json.loads((OUT/'results.json').read_text())
    if training['fit_count'] != 18 or results['protocol_sha256'] != sha(OUT/'protocol.json'):
        raise RuntimeError('Incomplete or mismatched fixed comparison')
    if training['predictions_sha256'] != sha(OUT/'predictions.npz'):
        raise RuntimeError('Prediction artifact changed')
    with np.load(SOURCE/'dataset.npz', allow_pickle=False) as z:
        truth, observed = z['truth'].copy(), z['observed'].copy()
        records = json.loads(str(z['records']))
    designs = np.array([r['design'] for r in records]); noises = np.array([r['noise'] for r in records])
    held = np.isin(designs, protocol['split']['held'])
    with np.load(OUT/'predictions.npz', allow_pickle=False) as z:
        predictions = z['predictions'].copy()
        folds, arms, seeds = z['folds'].copy(), z['arms'].copy(), z['seeds'].copy()
        if not np.isnan(predictions[:,held]).all() or not np.isfinite(predictions[:,~held]).all():
            raise RuntimeError('Held protection or allowed prediction coverage failed')
        if not np.array_equal(z['prediction_rows'], np.broadcast_to(~held, (18, len(held)))):
            raise RuntimeError('Prediction scope changed')
    checks = 0
    for fold, ids in enumerate(protocol['evaluation_folds']):
        for noise in protocol['primary_noise']:
            rows = np.isin(designs, ids) & (noises == noise)
            for arm in ('raw', 'mask'):
                indices = [np.flatnonzero((folds == fold) & (arms == arm) & (seeds == seed)).item() for seed in (42,43,44)]
                point = predictions[indices]
                for stratum, mask in (('all', np.ones(truth.shape, bool)),
                        ('boundary', (truth > -.02) & (truth < .05))):
                    expected = results['metrics'][f'fold{fold}/evaluation/{noise}'][arm][stratum]['per_design']
                    for design in ids:
                        selected = rows[:,None] & mask & (designs == design)[:,None]
                        direct = np.mean(abs(point[:,selected] - truth[selected]), axis=1)
                        np.testing.assert_allclose(direct, expected[design]['mae_by_seed'], rtol=0, atol=1e-12)
                        if int(selected.sum()) != expected[design]['frames']:
                            raise RuntimeError('Original scoring denominator changed')
                        checks += 1
    if sha(ROOT/'P4_PROGRESS.json') != protocol['protected_progress_ledger_sha256']:
        raise RuntimeError('Protected progress ledger changed')
    summaries = {}
    for noise in protocol['primary_noise']:
        by_arm = {}
        for arm in ARMS:
            whole = joined(results['metrics'], noise, arm, 'all')
            boundary = joined(results['metrics'], noise, arm, 'boundary')
            slices = {name: mae(joined(results['metrics'], noise, arm, name)) for name in
                ('all', 'boundary', 'boundary_current', 'raw_present_no_modal', 'raw_no_samples',
                 'raw_full_window', 'raw_partial_window', 'profile_retreat', 'farther_stable', 'farther_stable15')}
            by_arm[arm] = dict(slices=slices,
                no_sample_all=contribution(whole, joined(results['metrics'], noise, arm, 'raw_no_samples')),
                no_sample_boundary=contribution(boundary, joined(results['metrics'], noise, arm, 'boundary_raw_no_samples')),
                optimism=event(results['metrics'], noise, arm, 'boundary', 'optimism_over02'),
                stable_wrong_sign=event(results['metrics'], noise, arm, 'current', 'stable_wrong_sign'),
                unstable_wrong_sign=event(results['metrics'], noise, arm, 'current', 'unstable_exclusion_wrong_sign'))
        comparisons = {}
        for name in ('all', 'boundary', 'raw_present_no_modal', 'boundary_raw_present_no_modal',
                'boundary_current', 'profile_retreat', 'farther_stable', 'farther_stable15'):
            a = joined(results['metrics'], noise, 'raw', name)
            comparisons[name] = {control: paired(a, joined(results['metrics'], noise, control, name))
                for control in ARMS[1:] if a}
        raw = by_arm['raw']
        all_target, boundary_target = (.010, .005) if noise == 'clean' else (.015, .010)
        references = ARMS[2:]
        guarded = {
            'overall_absolute': raw['slices']['all']['mean'] <= all_target,
            'boundary_absolute': raw['slices']['boundary']['mean'] <= boundary_target,
            'optimism_absolute': raw['optimism']['mean'] <= .05,
            'stable_sign_absolute': raw['stable_wrong_sign']['mean'] <= .10 if raw['stable_wrong_sign']['mean'] is not None else None,
            'unstable_sign_absolute': raw['unstable_wrong_sign']['mean'] <= .10 if raw['unstable_wrong_sign']['mean'] is not None else None,
            'beats_mask_all': comparisons['all']['mask']['mean'] < 0,
            'beats_mask_boundary': comparisons['boundary']['mask']['mean'] < 0,
        }
        reference_guards = {}
        for reference in references:
            checks_for_reference = dict(
                overall_improved=comparisons['all'][reference]['mean'] < 0,
                boundary_improved=comparisons['boundary'][reference]['mean'] < 0)
            for name in ('boundary_current', 'profile_retreat', 'farther_stable', 'farther_stable15'):
                checks_for_reference[name] = (comparisons[name][reference]['mean'] <=
                    (.005 if name.startswith('farther_stable') else 0)) if comparisons[name] else None
            for guard in ('optimism', 'stable_wrong_sign', 'unstable_wrong_sign'):
                observed_rate, reference_rate = raw[guard]['mean'], by_arm[reference][guard]['mean']
                checks_for_reference[guard] = (observed_rate <= reference_rate
                    if observed_rate is not None and reference_rate is not None else None)
            reference_guards[reference] = checks_for_reference
        summaries[noise] = dict(arms=by_arm, raw_minus_controls=comparisons,
            targets=dict(all=all_target, boundary=boundary_target), descriptive_gates=guarded,
            guards_by_frozen_reference=reference_guards)
    report = dict(scope='Fixed 18-fit grouped development comparison; not blind validation or a promoted predictor',
        independent_original_denominator_checks=checks, held_predictions_absent=True,
        qualification_and_progress_ledger_unchanged=True, primary=summaries,
        sources={name:sha(OUT/name) for name in ('protocol.json','preflight.json','training.json','results.json','predictions.npz')},
        bootstrap='2000 seed42 design resamples; keep all3 seeds paired within each design; no refit/seed uncertainty or multiplicity correction',
        no_sample_contributions='Equal original design weights and original all/boundary frame counts; never renormalized on no-sample histories',
        limits='Point estimates only. Uncertainty, qualification, run-risk, warning, runtime and independent physical validation remain unpassed.')
    (OUT/'comparison_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('RAW_MODAL_COMPARISON', checks, 'original-denominator checks', flush=True)
    for noise, values in summaries.items():
        print(noise, {arm: {s: values['arms'][arm]['slices'][s]['mean'] for s in ('all','boundary')}
            for arm in ('raw','mask','content_gru','ordered_concat','unordered_pool','framewise_reference')}, flush=True)


if __name__ == '__main__':
    run()
