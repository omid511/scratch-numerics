"""Conditional train/OOF gaps and descriptive P4 feature-neighborhood audit.

Reuses fixed predictions and acquisition shards. No new prediction rule is fitted.
"""
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'p4_groups/correlated_training_20260929'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(PARENT)]
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from experiment_p4_sensor_layout import recovery_metrics
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
from mechanics.p4_margin_estimation.modal_features import estimate_modes, modal_feature_row
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
from mechanics.p4_margin_estimation.decision_metrics import error_exceedance
from run_augmentation_experiment import sha256

THRESHOLDS = [0., .005, .01, .02, .03, .05]


def distribution(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {'finite_count': int(len(finite)), 'missing_count': int(values.size-len(finite)),
            'mean': float(finite.mean()) if len(finite) else None,
            'p10_median_p90': np.quantile(finite, [.1, .5, .9]).tolist() if len(finite) else None}


def conditional(y, predictions, ids, prediction_kind):
    repeats = predictions.reshape(-1, len(y))
    over = error_exceedance(y, repeats, ids, THRESHOLDS)
    under = error_exceedance(y, repeats, ids, THRESHOLDS, direction='under')
    for record in (over, under):
        record['n_prediction_exposures_per_clip'] = record.pop('n_seeds')
        record['mean_exposure_median_error'] = record.pop('mean_seed_median_error')
        for row in record['curve']:
            row['per_exposure_rates'] = row.pop('per_seed_rates')
            row['per_exposure_counts'] = row.pop('per_seed_counts')
    error = predictions-y
    return {'over': over, 'under': under, 'prediction_kind': prediction_kind,
            'clip_mae': float(abs(error).mean()),
            'per_seed_bias': error.reshape(3, -1).mean(axis=1).tolist(),
            'per_seed_clip_mae': abs(error).reshape(3, -1).mean(axis=1).tolist()}


def paired_gaps(y, training, held, ids):
    # Average metrics across prediction exposures BEFORE clustering physical rows.
    train_error = training.reshape(-1, len(y))-y
    held_error = held.reshape(-1, len(y))-y
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = np.random.default_rng(42).multinomial(
        len(unique), np.full(len(unique), 1/len(unique)), size=2000)
    denominator = weights @ counts
    changes = {'mean_signed_error': held_error.mean(axis=0)-train_error.mean(axis=0),
               'clip_mae': abs(held_error).mean(axis=0)-abs(train_error).mean(axis=0)}
    for delta in THRESHOLDS:
        changes[f'over_{delta}'] = (held_error>delta).mean(axis=0)-(train_error>delta).mean(axis=0)
        changes[f'under_{delta}'] = (held_error < -delta).mean(axis=0)-(train_error < -delta).mean(axis=0)
    result = {}
    for name, difference in changes.items():
        totals = np.bincount(inverse, weights=difference)
        bootstrap = (weights @ totals)/denominator
        result[name] = {'held_minus_train': float(difference.mean()),
                        'design_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist()}
    macro = np.bincount(inverse, weights=changes['clip_mae'])/counts
    result['design_mae'] = {'held_minus_train': float(macro.mean()),
        'design_bootstrap_95': np.quantile((weights @ macro)/len(unique), [.025, .975]).tolist()}
    return result


def neighbors(views, evaluation, ids, folds):
    n = len(ids)
    indices = np.empty((2, 2, n, 10), dtype=np.int64)
    distances = np.empty_like(indices, dtype=float)
    for fi in range(5):
        query_rows = np.flatnonzero(folds == fi)
        fit_rows = np.flatnonzero(folds != fi)
        mean, scale, active = fit_ridge_scaling(np.concatenate([v[fit_rows] for v in views]))
        assert not set(ids[query_rows]) & set(ids[fit_rows])
        for variant in range(2):
            supported = active.copy()
            if variant:
                supported[18:22] = False
            for ci in range(2):
                q = ((evaluation[ci, query_rows]-mean)/scale)[:, supported]
                reference = ((evaluation[ci, fit_rows]-mean)/scale)[:, supported]
                d = cdist(q, reference)/np.sqrt(supported.sum())
                nearest = np.argpartition(d, 9, axis=1)[:, :10]
                nearest = np.take_along_axis(nearest, np.argsort(np.take_along_axis(d, nearest, axis=1), axis=1), axis=1)
                indices[variant, ci, query_rows] = fit_rows[nearest]
                distances[variant, ci, query_rows] = np.take_along_axis(d, nearest, axis=1)
    return indices, distances


def modal_diagnostics(rows, ids, evaluation):
    with np.load(ROOT / 'p4_dataset_saturation/metadata_arrays.npz') as z:
        dts = z['dts']
    protocol = json.loads((ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())
    layout = np.asarray(protocol['selected_sensor_indices'])
    diagnostics = np.full((2, len(rows), 10), np.nan)
    rowmap = {int(row): i for i, row in enumerate(rows)}
    for di, did in enumerate(np.unique(ids)):
        with np.load(ROOT / 'p4_sensor_candidates' / f'{did}.npz') as z:
            raw, source, scale, poles = z['raw'], z['source_indices'], z['noise_scale'], z['poles']
        for j, source_index in enumerate(source):
            k = rowmap[int(source_index)]
            noise = np.random.default_rng(np.random.SeedSequence([932, int(source_index)])).normal(size=raw[j].shape)
            for ci in range(2):
                measured = raw[j] if ci == 0 else raw[j]+.01*scale[j]*noise
                signal = np.clip(causal_calibration_normalize(measured, calibration_samples=51), -50, 50).astype(np.float32)[layout]
                modes = estimate_modes(signal, float(dts[source_index]))
                np.testing.assert_allclose(modal_feature_row(modes), evaluation[ci, k, 7:22], atol=1e-10, rtol=1e-10)
                diagnostics[ci, k, :4] = recovery_metrics(modes, poles[j])
                true_alpha = float(poles[j, 0].real)
                diagnostics[ci, k, 7] = true_alpha
                diagnostics[ci, k, 8] = evaluation[ci, k, 16]-true_alpha
                if len(modes['alpha']):
                    diagnostics[ci, k, 9] = float(max(modes['alpha'])-true_alpha)
                    frequency = poles[j].imag/(2*np.pi)
                    costs = abs(modes['frequency'][:, None]-frequency[None])/np.maximum(2., .05*frequency)[None]
                    estimated, actual = linear_sum_assignment(costs)
                    match = np.flatnonzero((actual == 0)&(costs[estimated, actual] <= 1.))
                    if len(match):
                        index = estimated[match[0]]
                        diagnostics[ci, k, 4:7] = (
                            modes['energy_fraction'][index], index < 3,
                            modes['energy_fraction'][index] >= .01)
        if (di+1) % 20 == 0:
            print('Modal diagnostic designs', di+1, flush=True)
    return diagnostics


def cohort(mask, y, ids, x, neighbors_index, distances, recovery):
    part = {'clips': int(mask.sum()), 'designs': int(len(np.unique(ids[mask])))}
    part['true_margin'] = distribution(y[mask])
    part['margin_bin_counts'] = {
        f'{lo}:{hi}': int((mask & (y >= lo) & (y < hi)).sum())
        for lo, hi in [(0., .01), (.01, .02), (.02, .03), (.03, .04), (.04, .05)]}
    columns = [16, 18, 19, 20, 21, 22, 25, 26, 36, 37, 38, 39, 40, 41]
    with np.load(ROOT / 'p4_groups/robustness_symmetric_spacing_20260929/features.npz') as z:
        names = z['original_feature_names'].tolist()+['xcorr']
    names[37] = 'symmetric_lead_neighbor_frequency_gap'
    names[39] = 'symmetric_lead_neighbor_pole_distance'
    part['features'] = {names[c]: distribution(x[mask, c]) for c in columns}
    part['modal_recovery'] = {name: distribution(recovery[mask, c]) for c, name in enumerate(
        ['all_mode_recovery', 'matched_alpha_error', 'critical_alpha_error', 'critical_recovered',
         'matched_critical_energy_fraction', 'matched_critical_in_energy_top3',
         'matched_critical_visible', 'true_critical_alpha',
         'max_visible_alpha_minus_true_critical', 'max_estimated_alpha_minus_true_critical'])}
    part['neighborhoods'] = {}
    for vi, name in enumerate(('all_supported_features', 'without_explicit_quality')):
        ix, dist = neighbors_index[vi, mask], distances[vi, mask]
        item = {'nearest_rms_standardized_distance': distribution(dist[:, 0])}
        for count in (3, 10):
            target = y[ix[:, :count]]
            item[f'k{count}'] = {
                'target_std': distribution(target.std(axis=1)),
                'target_range': distribution(np.ptp(target, axis=1)),
                'target_mean_minus_query': distribution(target.mean(axis=1)-y[mask]),
                'target_median_minus_query': distribution(np.median(target, axis=1)-y[mask]),
                'neighbor_design_count': distribution(np.array([len(np.unique(ids[r])) for r in ix[:, :count]])),
                'fraction_of_neighbors_margin_ge_0.15': distribution((target >= .15).mean(axis=1)),
            }
        part['neighborhoods'][name] = item
    return part


def main():
    with np.load(OUT / 'predictions.npz') as z:
        training, held = z['training_predictions'], z['heldout_predictions']
        y, ids, rows, folds = z['y'], z['ids'], z['source_indices'], z['row_fold']
        conditions = z['conditions'].tolist()
    with np.load(PARENT / 'shared.npz') as z:
        views = np.concatenate((z['train3'], z['independent_extra'][None]))
        evaluation = z['evaluation'][[z['evaluation_names'].tolist().index(c) for c in conditions]]
        np.testing.assert_array_equal(rows, z['source_indices'])
        np.testing.assert_array_equal(ids, z['ids'])
    neighbor_indices, neighbor_distances = neighbors(views, evaluation, ids, folds)
    recovery = modal_diagnostics(rows, ids, evaluation)
    np.savez_compressed(OUT / 'diagnostics.npz', neighbor_indices=neighbor_indices,
                        neighbor_distances=neighbor_distances, recovery=recovery,
                        source_indices=rows, ids=ids, y=y, conditions=conditions)
    slices = {'all': np.ones(len(y), dtype=bool), 'safer_at_least_0.15': y >= .15}
    for epsilon in (.01, .02, .05):
        slices[f'stable_below_{epsilon}'] = (y>0)&(y<epsilon)
        slices[f'unstable_within_{epsilon}'] = (y<0)&(y>-epsilon)
    payload = {'protocol': json.loads((OUT / 'protocol.json').read_text()),
               'comparison': {}, 'failure_cohorts': {}, 'failure_cases': {}}
    for ci, condition in enumerate(conditions):
        payload['comparison'][condition] = {}
        for name, mask in slices.items():
            a, b = training[:, :, ci, :][:, :, mask], held[:, ci, :][:, mask]
            payload['comparison'][condition][name] = {
                'training': conditional(y[mask], a, ids[mask], 'three seeds x four in-sample fit exposures'),
                'heldout': conditional(y[mask], b, ids[mask], 'three seeds x one design-held-out prediction'),
                'paired_gap': paired_gaps(y[mask], a, b, ids[mask])}
        stable = slices['stable_below_0.05']
        errors = held[:, ci]-y
        severe = stable & ((errors>.03).sum(axis=0)>=2)
        payload['failure_cohorts'][condition] = {
            name: cohort(mask, y, ids, evaluation[ci], neighbor_indices[:, ci], neighbor_distances[:, ci], recovery[ci])
            for name, mask in [('severe', severe), ('other_stable_below_0.05', stable&~severe)]}
        cases = []
        for k in np.flatnonzero(severe):
            ix = neighbor_indices[0, ci, k]
            cases.append({'source_index': int(rows[k]), 'design': str(ids[k]), 'true_margin': float(y[k]),
                          'errors_by_seed': errors[:, k].tolist(),
                          'mean_in_sample_error': float((training[:, :, ci, k]-y[k]).mean()),
                          'features': evaluation[ci, k].tolist(), 'recovery': [float(v) if np.isfinite(v) else None for v in recovery[ci, k]],
                          'neighbor_source_indices': rows[ix].tolist(), 'neighbor_designs': ids[ix].tolist(),
                          'neighbor_targets': y[ix].tolist(), 'neighbor_distances': neighbor_distances[0, ci, k].tolist()})
        payload['failure_cases'][condition] = sorted(cases, key=lambda c: -np.mean(c['errors_by_seed']))
        print(condition, 'severe cases', int(severe.sum()), 'designs', len(np.unique(ids[severe])), flush=True)
    clean_rows = {c['source_index'] for c in payload['failure_cases']['clean']}
    noisy_rows = {c['source_index'] for c in payload['failure_cases']['iid1_fresh932']}
    payload['clean_noisy_severe_overlap'] = len(clean_rows & noisy_rows)
    sources = ['experiment_p4_tabular.py', 'experiment_p4_sensor_layout.py',
               'src/mechanics/p4_margin_estimation/modal_features.py',
               'src/mechanics/p4_margin_estimation/baselines.py',
               'src/mechanics/p4_margin_estimation/decision_metrics.py']
    payload['hashes'] = {'analysis': sha256(Path(__file__)), 'predictions': sha256(OUT / 'predictions.npz'),
                         'shared_features': sha256(PARENT / 'shared.npz'),
                         'sources': {p: sha256(ROOT / p) for p in sources},
                         'acquisition_shards': {str(d): sha256(ROOT / 'p4_sensor_candidates' / f'{d}.npz') for d in np.unique(ids)}}
    (OUT / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print('Saved conditional gaps, neighborhoods and modal diagnostics', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
