#!/usr/bin/env python3
"""Score the frozen fitter/margin comparison; never train neural checkpoints."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

import experiment_p4_ridge_cv as ridge
import temporal_convergence as convergence
import temporal_probe as probe

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT / 'fitter_margin_comparison_20261008'
BASE = SCRIPT / 'temporal_probe_20261007'
CONVERGENCE = SCRIPT / 'temporal_convergence_20261007'
KINDS = ('framewise', 'gru', 'time_only')
MARGIN_NAMES = ('ridge6', 'ridge174', 'framewise', 'gru', 'time_only')
BUDGETS = (.01, .05, .10)
NOISE_FOR_TUNING = ('clean', 'iid1')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scalar_json(archive, key):
    return json.loads(str(archive[key]))


def record_key(record):
    return tuple(record[k] for k in ('design', 'split', 'replicate', 'profile', 'noise'))


def validate_inputs(protocol):
    hashes = {}
    for name, expected in protocol['inputs'].items():
        path = SCRIPT / name
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f'frozen input hash mismatch: {name}: {actual} != {expected}')
        hashes[name] = actual
    return hashes


def expected_records(protocol, base_protocol, smoke):
    rows = []
    for split, designs in protocol['split'].items():
        for design in (designs[:1] if smoke else designs):
            for replicate in range(1 if smoke else base_protocol['schedule_replicates']):
                for profile in base_protocol['profiles']:
                    for noise in protocol['noise']:
                        rows.append(dict(design=design, split=split, replicate=replicate,
                                         profile=profile, noise=noise))
    return rows


def load_dataset(path):
    keys = ('features', 'truth', 'observed', 'age', 'times', 'fitter_features',
            'fitter_observed', 'fitter_age')
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in keys}
        data['records'] = read_scalar_json(z, 'records')
        data['protocol'] = read_scalar_json(z, 'protocol')
        data['provenance'] = read_scalar_json(z, 'provenance')
        data['fit_records'] = read_scalar_json(z, 'fit_records')
    return data


def validate_dataset(data, protocol, base_protocol, smoke):
    records = data['records']
    expected = expected_records(protocol, base_protocol, smoke)
    if [record_key(r) for r in records] != [record_key(r) for r in expected]:
        raise ValueError('dataset records are not in frozen split/design/replicate/profile/noise order')
    splits = protocol['split']
    designs = [d for names in splits.values() for d in names]
    if len(designs) != len(set(designs)):
        raise ValueError('frozen design partitions overlap')
    if not smoke and len(splits['train']) != 4:
        raise ValueError('full comparison requires all four proper training designs')
    n, t = len(records), len(data['times'])
    shapes = {'features': (n, t, 174), 'truth': (n, t), 'observed': (n, t),
              'age': (n, t), 'fitter_features': (n, t, 6),
              'fitter_observed': (n, t), 'fitter_age': (n, t)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f'{key} shape {data[key].shape} != {shape}')
    if t != 124 or not np.isfinite(data['truth']).all() or not np.isfinite(data['times']).all():
        raise ValueError('unexpected frame count or nonfinite truth/time')
    if not np.all(np.diff(data['times']) > 0):
        raise ValueError('frame times must increase')
    if data['protocol'] != protocol:
        raise ValueError('acquisition protocol differs from the on-disk frozen protocol')
    if not np.isfinite(data['features']).all() or not np.isfinite(data['fitter_features']).all():
        raise ValueError('feature arrays must be finite; unavailable rows are represented by zeros')
    if not np.isfinite(data['age']).all() or not np.isfinite(data['fitter_age']).all():
        raise ValueError('observation ages must be finite')
    if data['provenance']['input_hashes'] != protocol['inputs']:
        raise ValueError('acquisition inputs disagree with frozen hashes')
    if data['provenance']['comparison_protocol_sha256'] != sha256(OUT / 'protocol.json'):
        raise ValueError('acquisition protocol hash disagrees with frozen protocol')
    if np.any(data['features'][~data['observed']] != 0):
        raise ValueError('original unavailable features must be zero')
    if np.any(data['fitter_features'][~data['fitter_observed']] != 0):
        raise ValueError('unavailable fitter features must be zero')
    if np.any(data['age'] < 0) or np.any(data['fitter_age'] < 0):
        raise ValueError('observation ages cannot be negative')
    return designs


def validate_fits(data):
    attempts = {}
    for fit in data['fit_records']:
        key = (int(fit['record_index']), int(fit['frame_index']))
        if key in attempts or not (0 <= key[0] < len(data['records']) and 0 <= key[1] < len(data['times'])):
            raise ValueError(f'duplicate or out-of-range fit attempt: {key}')
        if 'stationary' not in fit or 'varying' not in fit:
            raise ValueError(f'incomplete fit attempt: {key}')
        attempts[key] = fit
    expected = {(r, f) for r in range(len(data['records']))
                for f in np.flatnonzero(data['observed'][r])}
    if set(attempts) != expected:
        raise ValueError('fit_records must retain every and only shared-gate fit attempt')
    valid = np.zeros_like(data['fitter_observed'])
    for (r, f), fit in attempts.items():
        results = (fit['stationary'], fit['varying'])
        usable = all(bool(x['success']) and not x['bound_hits'] for x in results)
        valid[r, f] = usable
        if usable:
            varying = fit['varying']
            p = varying['parameters']
            alpha, freq = float(varying['alpha_end']), float(varying['frequency_end'])
            vals = np.asarray([alpha, freq, p['alpha_slope'], p['frequency_slope'],
                               alpha / (2 * np.pi * freq), varying['relative_fit_error']], float)
            if not np.isfinite(vals).all() or freq == 0:
                raise ValueError(f'nonfinite successful varying fit: {(r, f)}')
            if not np.allclose(data['fitter_features'][r, f], vals, rtol=1e-5, atol=1e-7):
                raise ValueError(f'fitter feature/result mismatch at {(r, f)}')
    if not np.array_equal(valid, data['fitter_observed']):
        raise ValueError('fitter availability must require both successful interior fits')
    return attempts


def load_archived(data, smoke=False):
    with np.load(BASE / 'data.npz', allow_pickle=False) as z:
        old = {k: z[k].copy() for k in ('features', 'truth', 'observed', 'age', 'times')}
        old['records'] = read_scalar_json(z, 'records')
    with np.load(CONVERGENCE / 'predictions.npz', allow_pickle=False) as z:
        points = z['predictions'].copy()
        mean, scale, active = (z[k].copy() for k in ('mean', 'scale', 'active'))
        point_records = read_scalar_json(z, 'records')
        for key in ('truth', 'observed', 'age', 'times'):
            if not np.array_equal(z[key], old[key]):
                raise ValueError(f'archived prediction {key} differs from original dataset')
    if point_records != old['records'] or points.shape != (3, len(old['records']), len(old['times'])):
        raise ValueError('unexpected archived convergence prediction schema')
    old_index = {record_key(rec): i for i, rec in enumerate(old['records'])}
    new_index = {record_key(rec): i for i, rec in enumerate(data['records'])}
    if len(old_index) != len(old['records']) or len(new_index) != len(data['records']):
        raise ValueError('duplicate trajectory/noise records')
    for key, oi in old_index.items():
        ni = new_index.get(key)
        if ni is None:
            if smoke:
                continue
            raise ValueError(f'missing original clean/iid1 record {key}')
        if not np.allclose(data['features'][ni], old['features'][oi], rtol=0, atol=1e-5):
            raise ValueError(f'original features changed beyond acquisition tolerance for {key}')
        for name in ('truth', 'observed', 'age'):
            if not np.array_equal(data[name][ni], old[name][oi]):
                raise ValueError(f'original {name} changed for {key}')
    if not (mean.shape == scale.shape == active.shape == (174,)):
        raise ValueError('unexpected archived feature scaling')
    return old, points, (mean, scale, active), old_index, new_index


def select_ridge(x, y, ids, design_order, degrees, smoke, label):
    if smoke:
        return dict(label=label, mode='mechanical smoke; fixed degree=1 alpha=1, no CV',
                    selected=dict(degree=1, alpha=1.0), candidates=[], folds=[],
                    tuning_rows=len(y), tuning_designs=sorted(set(ids.tolist())))
    if set(ids.tolist()) != set(design_order):
        raise ValueError(f'{label} LODO requires valid rows from every proper design')
    candidates, folds = [], []
    for degree in degrees:
        oof = np.full((len(y), len(ridge.ALPHAS)), np.nan)
        for held in design_order:
            test = ids == held
            train = ~test
            if not test.any() or not train.any():
                raise ValueError(f'{label} LODO fold has an empty side: {held}')
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            xtrain = poly.fit_transform(x[train])
            xtest = poly.transform(x[test])
            pred = ridge.ridge_predictions(xtrain, y[train], xtest, ridge.ALPHAS)
            oof[test] = pred
            folds.append(dict(degree=degree, held_design=held, train_rows=int(train.sum()),
                              validation_rows=int(test.sum()), alpha_mae=np.abs(pred-y[test, None]).mean(axis=0).tolist()))
        per_design = ridge.per_design_mae(y, oof, ids)
        ordered = np.unique(ids)
        for i, did in enumerate(ordered):
            for j, alpha in enumerate(ridge.ALPHAS):
                candidates.append(dict(degree=int(degree), alpha=float(alpha), design=did,
                                       design_mae=float(per_design[i, j])))
        scores = per_design.mean(axis=0)
        for j, alpha in enumerate(ridge.ALPHAS):
            candidates.append(dict(degree=int(degree), alpha=float(alpha), design='mean_equal_design',
                                   design_mae=float(scores[j])))
    means = [r for r in candidates if r['design'] == 'mean_equal_design']
    winner = min(means, key=lambda r: (r['design_mae'], r['degree'], -r['alpha']))
    return dict(label=label, mode='leave-one-design-out; clean/iid1 valid proper rows only',
                selected=dict(degree=winner['degree'], alpha=winner['alpha']),
                candidates=candidates, folds=folds, tuning_rows=len(y),
                tuning_designs=sorted(set(ids.tolist())))


def causal_hold(raw, fresh, times):
    held = np.full(raw.shape, np.nan)
    age = np.full(raw.shape, np.nan)
    for r in range(len(raw)):
        value = np.nan
        since = np.nan
        for f, time in enumerate(times):
            if fresh[r, f]:
                value, since = raw[r, f], time
            if np.isfinite(value):
                held[r, f], age[r, f] = value, time - since
    return held, age


def apply_ridge(x, truth, train_mask, fresh, times, degree, alpha):
    train_x, train_y = x[train_mask], truth[train_mask]
    if len(train_y) == 0 or not np.isfinite(train_x).all() or not np.isfinite(train_y).all():
        raise ValueError('ridge final map has no finite training rows')
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    train_poly = poly.fit_transform(train_x)
    raw = np.full(fresh.shape, np.nan)
    raw[fresh] = ridge.ridge_predictions(train_poly, train_y, poly.transform(x[fresh]),
                                          np.asarray([alpha]))[:, 0]
    if not np.isfinite(raw[fresh]).all():
        raise ValueError('ridge produced nonfinite fresh predictions')
    held, age = causal_hold(raw, fresh, times)
    return raw, held, age


def infer_learned(data, scaling):
    mean, scale, active = scaling
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('invalid frozen neural input scaling')
    normalized = (data['features'] - mean) / scale
    normalized[..., ~active] = 0
    normalized[~data['observed']] = 0
    inp = np.concatenate((normalized, data['observed'][..., None], data['age'][..., None]), axis=-1).astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    outputs = {}
    for kind in KINDS:
        width = 1 if kind == 'time_only' else inp.shape[-1]
        sample = torch.empty((1, inp.shape[1], width), device=device, dtype=torch.float32)
        model = convergence.make_model(kind, sample).to(device=device, dtype=torch.float32)
        checkpoint = torch.load(CONVERGENCE / f'{kind}.pt', map_location=device, weights_only=False)
        if checkpoint.get('kind') != kind:
            raise ValueError(f'wrong checkpoint kind for {kind}')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        if kind == 'time_only':
            values = np.broadcast_to(data['times'][None, :, None],
                                     (len(data['records']), len(data['times']), 1)).copy().astype(np.float32)
        else:
            values = inp
        with torch.no_grad():
            tensor = torch.as_tensor(values, device=device, dtype=torch.float32)
            point = convergence.predict(model, kind, tensor).cpu().numpy()
        if not np.isfinite(point).all():
            raise ValueError(f'{kind} returned nonfinite margin predictions')
        outputs[kind] = point
    return outputs, str(device)


def summary(y, point, mask):
    population = int(mask.sum())
    valid = mask & np.isfinite(y) & np.isfinite(point)
    n = int(valid.sum())
    return dict(population_frames=population, scored_frames=n,
                unknown_frames=population-n,
                mae=float(np.abs(point[valid]-y[valid]).mean()) if n else None)


def count_rate(count, denominator):
    return dict(count=int(count), denominator=int(denominator),
                rate=float(count/denominator) if denominator else None)


def event_report(y, score, threshold, group):
    positive, safe = group & (y <= .02), group & (y > .02)
    finite = np.isfinite(score)
    if threshold is None:
        return dict(status='no_calibration', threshold=None,
                    positive=dict(population_frames=int(positive.sum()), finite_frames=int((positive&finite).sum()),
                                  unknown_frames=int((positive&~finite).sum()), true_positives=None, misses=None, recall=None),
                    safe=dict(population_frames=int(safe.sum()), finite_frames=int((safe&finite).sum()),
                              unknown_frames=int((safe&~finite).sum()), false_warnings=None, false_warning_rate=None))
    fired = finite & (score > threshold)
    tp, safe_warn = int((fired&positive).sum()), int((fired&safe).sum())
    pos_known, safe_known = int((positive&finite).sum()), int((safe&finite).sum())
    return dict(status='scored', threshold=float(threshold),
                positive=dict(population_frames=int(positive.sum()), finite_frames=pos_known,
                              unknown_frames=int(positive.sum())-pos_known, true_positives=tp,
                              misses=int(positive.sum())-tp, recall=tp/positive.sum() if positive.any() else None,
                              known_recall=tp/pos_known if pos_known else None),
                safe=dict(population_frames=int(safe.sum()), finite_frames=safe_known,
                          unknown_frames=int(safe.sum())-safe_known, false_warnings=safe_warn,
                          false_warning_rate=safe_warn/safe.sum() if safe.any() else None,
                          known_false_warning_rate=safe_warn/safe_known if safe_known else None))


def gap_strata(y, point, age, fresh, group):
    gap = group & ~fresh
    bins = (('gap_0_128ms', age >= 0, age <= .128),
            ('gap_128_256ms', age > .128, age <= .256),
            ('gap_over_256ms', age > .256, np.ones(age.shape, bool)))
    out = {}
    for name, lower, upper in bins:
        mask = gap & lower & upper
        out[name] = summary(y, point, mask)
    out['unknown_gap_output'] = dict(frames=int((gap & ~np.isfinite(point)).sum()),
                                     denominator=int(gap.sum()))
    return out


def fit_counts(record_mask, data, attempts):
    indices = np.flatnonzero(record_mask)
    keys = [(r, f) for (r, f) in attempts if record_mask[r]]
    bad_stationary = sum(not bool(attempts[k]['stationary']['success']) for k in keys)
    bad_varying = sum(not bool(attempts[k]['varying']['success']) for k in keys)
    bounds = sum(bool(x['bound_hits']) for k in keys for x in attempts[k].values()
                 if isinstance(x, dict) and 'bound_hits' in x)
    return dict(shared_gate_frames=int(data['observed'][indices].sum()),
                fit_attempts=len(keys), stationary_failures=int(bad_stationary),
                varying_failures=int(bad_varying), optimizer_or_bound_unavailable=len(keys)-int(data['fitter_observed'][indices].sum()),
                fitter_fresh=int(data['fitter_observed'][indices].sum()))


def paired_differences(data, points, record_mask):
    result = []
    designs = sorted({r['design'] for i, r in enumerate(data['records']) if record_mask[i]})
    for design in designs:
        select = record_mask & np.asarray([r['design'] == design for r in data['records']])
        mask = np.broadcast_to(select[:, None], data['truth'].shape) & data['observed'] & data['fitter_observed']
        row = dict(design=design, common_fresh_frames=int(mask.sum()), differences={})
        for baseline in ('ridge6', 'ridge174'):
            for model in ('framewise', 'gru'):
                valid = mask & np.isfinite(points[baseline]) & np.isfinite(points[model])
                denom = int(mask.sum())
                if int(valid.sum()) != denom:
                    raise ValueError('nonfinite output on paired common-fresh mask')
                a = float(np.abs(points[baseline][valid]-data['truth'][valid]).mean()) if denom else None
                b = float(np.abs(points[model][valid]-data['truth'][valid]).mean()) if denom else None
                row['differences'][f'{baseline}_minus_{model}'] = a-b if denom else None
        result.append(row)
    vals = {key: [r['differences'][key] for r in result if r['differences'][key] is not None]
            for key in ('ridge6_minus_framewise', 'ridge6_minus_gru',
                        'ridge174_minus_framewise', 'ridge174_minus_gru')}
    return dict(by_design=result, equal_design_mean={k: float(np.mean(v)) if v else None for k, v in vals.items()},
                inference='descriptive paired design differences; no significance claim')


def summarize_group(data, points, scores, thresholds, attempts, record_mask):
    group = np.broadcast_to(record_mask[:, None], data['truth'].shape)
    shared = group & data['observed'] & data['fitter_observed']
    finite_all = group & np.isfinite(data['truth'])
    for name in MARGIN_NAMES:
        finite_all &= np.isfinite(points[name])
    if not all(np.isfinite(points[name][shared]).all() for name in MARGIN_NAMES):
        raise ValueError('nonfinite margin output on common-fresh denominator')
    coverage = dict(total_frames=int(group.sum()),
        original_fresh=count_rate((group&data['observed']).sum(), group.sum()),
        fitter_fresh=count_rate((group&data['fitter_observed']).sum(), group.sum()),
        shared_fresh=count_rate(shared.sum(), group.sum()))
    metrics = {}
    for name in MARGIN_NAMES:
        fresh = data['fitter_observed'] if name == 'ridge6' else data['observed']
        point = points[name]
        near = group & (data['truth'] > 0) & (data['truth'] < .02)
        unstable = group & (data['truth'] < 0)
        metrics[name] = dict(available_only=summary(data['truth'], point, group&fresh),
            common_fresh=summary(data['truth'], point, shared),
            all_finite_output=summary(data['truth'], point, group),
            common_finite_all_frame=summary(data['truth'], point, finite_all),
            optimistic_errors={str(e): dict(population_frames=int(near.sum()),
                scored_frames=int((near&np.isfinite(point)).sum()),
                unknown_frames=int((near&~np.isfinite(point)).sum()),
                count=int((near&np.isfinite(point)&(point-data['truth']>e)).sum()),
                rate=(float((near&np.isfinite(point)&(point-data['truth']>e)).sum()/(near&np.isfinite(point)).sum())
                      if (near&np.isfinite(point)).any() else None)) for e in (.01, .03)},
            false_stable=dict(population_frames=int(unstable.sum()),
                scored_frames=int((unstable&np.isfinite(point)).sum()),
                unknown_frames=int((unstable&~np.isfinite(point)).sum()),
                count=int((unstable&np.isfinite(point)&(point>0)).sum()),
                rate=(float((unstable&np.isfinite(point)&(point>0)).sum()/(unstable&np.isfinite(point)).sum())
                      if (unstable&np.isfinite(point)).any() else None)),
            gap_age=gap_strata(data['truth'], point,
                data['fitter_age'] if name=='ridge6' else data['age'], fresh, group))
    warning_metrics = {}
    for key, score in scores.items():
        warning_metrics[key] = {budget: event_report(data['truth'], score,
            thresholds[key][budget]['threshold'], group) for budget in thresholds[key]}
    warning_metrics['growth_sign_alpha_gt_0'] = event_report(
        data['truth'], scores['alpha_end'], 0.0, group)
    return dict(coverage=coverage, fit=fit_counts(record_mask, data, attempts),
                margins=metrics, warnings=warning_metrics,
                common_fresh_paired_design_differences=paired_differences(data, points, record_mask))


def calibrate_threshold(score, safe_mask, budget):
    population = int(safe_mask.sum())
    vals = score[safe_mask & np.isfinite(score)]
    if not len(vals):
        reason = 'no_calibration_population' if population == 0 else 'no_finite_calibration_scores'
        return dict(threshold=None, status=reason, population_denominator=population,
                    finite_scores=0, allowed_false_warnings=int(np.floor(budget*population)))
    allowed = int(np.floor(budget*population + 1e-12))
    descending = np.sort(vals)[::-1]
    threshold = float(descending[allowed]) if allowed < len(descending) else float(np.nextafter(descending[-1], -np.inf))
    actual = int(np.sum(vals > threshold))
    if actual > allowed:
        raise ValueError('tie-safe strict threshold exceeds the empirical false-warning budget')
    return dict(threshold=threshold, status='calibrated', population_denominator=population,
                finite_scores=len(vals), unknown_scores=population-len(vals),
                allowed_false_warnings=allowed, calibration_false_warnings=actual,
                calibration_false_warning_rate=actual/population if population else None,
                comparison='strict score > threshold; unknown safe windows remain in denominator')


def temporal_alerts(data, scores, thresholds):
    rows = []
    specs = [(name, budget, thresholds[name][budget]['threshold'])
             for name in scores for budget in thresholds[name]]
    specs.append(('growth_sign_alpha_gt_0', 'uncalibrated', 0.0))
    for r, record in enumerate(data['records']):
        if record['profile'] != 'cross':
            continue
        y, times = data['truth'][r], data['times']
        for name, budget, threshold in specs:
            score = scores['alpha_end' if name == 'growth_sign_alpha_gt_0' else name][r]
            if threshold is None:
                rows.append(dict(readout=name, budget=budget, record_index=r, record=record,
                                 status='no_calibration', near_boundary=None, flutter=None))
                continue
            alarm = np.isfinite(score) & (score > threshold)
            near_i = np.flatnonzero(y <= .02)
            flutter_i = np.flatnonzero(y <= 0)
            def event(indices, useful_start=None):
                if not len(indices):
                    return dict(entry_absent=True, entry_time=None, pre_entry_warning_frames=int(alarm.sum()),
                                pre_entry_alarm=bool(alarm.any()), first_warning_before_entry_time=(float(times[np.flatnonzero(alarm)[0]]) if alarm.any() else None),
                                first_useful_alert_absent=True, first_useful_alert_time=None,
                                lead_seconds=None, delay_seconds=None)
                entry = int(indices[0]); pre = np.flatnonzero(alarm[:entry])
                start = entry if useful_start is None else useful_start
                useful = np.flatnonzero(alarm[start:])
                alert = int(start+useful[0]) if len(useful) else None
                return dict(entry_absent=False, entry_time=float(times[entry]),
                    pre_entry_warning_frames=int(len(pre)), pre_entry_alarm=bool(len(pre)),
                    first_warning_before_entry_time=float(times[pre[0]]) if len(pre) else None,
                    first_useful_alert_absent=alert is None,
                    first_useful_alert_time=float(times[alert]) if alert is not None else None,
                    lead_seconds=float(times[entry]-times[alert]) if alert is not None else None,
                    delay_seconds=float(times[alert]-times[entry]) if alert is not None else None)
            rows.append(dict(readout=name, budget=budget, record_index=r, record=record,
                             status='scored', near_boundary=event(near_i),
                             flutter=event(flutter_i, int(near_i[0]) if len(near_i) else None)))
    return rows


def run(smoke=False):
    protocol_path = OUT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    input_hashes = validate_inputs(protocol)
    base_protocol = json.loads((BASE / 'protocol.json').read_text())
    data_path = OUT / ('smoke_dataset.npz' if smoke else 'dataset.npz')
    pred_path = OUT / ('smoke_predictions.npz' if smoke else 'predictions.npz')
    result_path = OUT / ('smoke_results.json' if smoke else 'results.json')
    if pred_path.exists() or result_path.exists():
        raise FileExistsError(f'refusing to overwrite completed comparison artifacts in {OUT}')
    data = load_dataset(data_path)
    design_order = validate_dataset(data, protocol, base_protocol, smoke)
    attempts = validate_fits(data)
    old, original_points, scaling, old_index, new_index = load_archived(data, smoke)
    tuning_records = np.asarray([r['split']=='train' and r['noise'] in NOISE_FOR_TUNING
                                 for r in data['records']])
    train_designs = protocol['split']['train']
    train6 = np.broadcast_to(tuning_records[:, None], data['truth'].shape) & data['fitter_observed']
    rows6 = np.nonzero(train6)
    sel6 = select_ridge(data['fitter_features'][rows6], data['truth'][rows6],
                        np.asarray([data['records'][r]['design'] for r in rows6[0]]),
                        train_designs, (1, 2), smoke, 'six physical fitter features')
    train174 = np.broadcast_to(tuning_records[:, None], data['truth'].shape) & data['observed']
    rows174 = np.nonzero(train174)
    sel174 = select_ridge(data['features'][rows174], data['truth'][rows174],
                          np.asarray([data['records'][r]['design'] for r in rows174[0]]),
                          train_designs, (1,), smoke, 'original174 linear representation control')
    chosen6, chosen174 = sel6['selected'], sel174['selected']
    final6 = np.broadcast_to(np.asarray([r['split']=='train' and r['noise'] in NOISE_FOR_TUNING
                                         for r in data['records']])[:, None], data['truth'].shape) & data['fitter_observed']
    raw6, point6, age6 = apply_ridge(data['fitter_features'], data['truth'], final6,
        data['fitter_observed'], data['times'], chosen6['degree'], chosen6['alpha'])
    final174 = np.broadcast_to(np.asarray([r['split']=='train' and r['noise'] in NOISE_FOR_TUNING
                                           for r in data['records']])[:, None], data['truth'].shape) & data['observed']
    raw174, point174, age174 = apply_ridge(data['features'], data['truth'], final174,
        data['observed'], data['times'], chosen174['degree'], chosen174['alpha'])
    learned, device = infer_learned(data, scaling)
    points = dict(ridge6=point6, ridge174=point174, **learned)
    # Compare frozen clean/iid1 inference to archived converged points before publishing.
    reproduction = {}
    for kind_index, kind in enumerate(KINDS):
        errors = []
        for key, oi in old_index.items():
            if key not in new_index:
                continue
            ni = new_index[key]
            if key[-1] in NOISE_FOR_TUNING:
                errors.append(np.max(np.abs(learned[kind][ni]-original_points[kind_index, oi])))
        maximum = float(max(errors)) if errors else None
        if maximum is None or maximum > 2e-6:
            raise RuntimeError(f'{kind} archived prediction reproduction failed: max error {maximum}')
        reproduction[kind] = dict(records=len(errors), max_abs_difference=maximum, tolerance=2e-6)
    alpha_end = np.full(data['truth'].shape, np.nan)
    alpha_end[data['fitter_observed']] = data['fitter_features'][..., 0][data['fitter_observed']]
    scores = {name: -points[name] for name in MARGIN_NAMES}
    scores['alpha_end'] = alpha_end
    val = np.asarray([r['split']=='validation' and r['noise'] in NOISE_FOR_TUNING
                      for r in data['records']])[:, None]
    safe = np.broadcast_to(val, data['truth'].shape) & (data['truth'] > .02)
    thresholds = {name: {str(b): calibrate_threshold(score, safe, b) for b in BUDGETS}
                  for name, score in scores.items()}
    split_noise, design_noise = {}, {}
    for split in protocol['split']:
        for noise in protocol['noise']:
            group = np.asarray([r['split']==split and r['noise']==noise for r in data['records']])
            report = summarize_group(data, points, scores, thresholds, attempts, group)
            report['held_designs_are_exploratory'] = split == 'held'
            split_noise[f'{split}/{noise}'] = report
            for did in protocol['split'][split]:
                mask = group & np.asarray([r['design']==did for r in data['records']])
                one = summarize_group(data, points, scores, thresholds, attempts, mask)
                one['held_design_is_exploratory'] = split == 'held'
                design_noise[f'{split}/{did}/{noise}'] = one
    thresholds['growth_sign_alpha_gt_0'] = dict(calibrated=False, threshold=0.0,
        rule='literal physical growth-sign alarm alpha_end > 0; fresh valid fitter outputs only')
    temporal = temporal_alerts(data, scores, thresholds)
    provenance = dict(protocol_sha256=sha256(protocol_path), dataset_sha256=sha256(data_path),
        frozen_input_hashes=input_hashes, dataset_provenance=data['provenance'],
        implementation_sha256={name: sha256(path) for name, path in
            (('fitter_comparison_score.py', Path(__file__)),
             ('experiment_p4_ridge_cv.py', SCRIPT.parents[1] / 'experiment_p4_ridge_cv.py'))},
        checkpoint_inference_device=device, old_inputs_reproduced=True,
        learned_reproduction=reproduction, mode='mechanical_smoke' if smoke else 'full_frozen_comparison')
    selection = dict(smoke=bool(smoke), fitter6=sel6, representation_control174=sel174,
        fitter6_final_training_rows=int(final6.sum()), representation_control174_final_training_rows=int(final174.sum()),
        final_fit_population='proper train designs, clean/iid1, respective valid observations only')
    result = dict(scope=protocol['scope'], mode='mechanical smoke only; not evidence' if smoke else protocol['scope'],
        protocol=protocol, provenance=provenance, selection=selection,
        margin_readouts=list(MARGIN_NAMES), warning_calibration=dict(
            population='validation safe windows m>0.02, clean/iid1 pooled; unknown scores remain in population denominator',
            budgets=list(BUDGETS), strict_comparison='score > threshold', thresholds=thresholds,
            reference='alpha_end > 0 is an uncalibrated physical growth-sign alarm'),
        metrics=dict(per_split_noise=split_noise, per_design_noise=design_noise),
        temporal_cross_warning=temporal,
        limitations=['Exploratory comparison; validation previously participated in checkpoint selection.',
            'Two held designs are previously inspected and are descriptive only; no significance claim.',
            'Fitter qualification gate is unchanged; this comparison does not qualify or promote it.'])
    names = list(MARGIN_NAMES)
    warning_names = list(scores)
    T = probe.M.clean_json
    serialized_thresholds = json.dumps(T(thresholds), allow_nan=False)
    probe.M.G.O.C.save_npz(pred_path, predictions=np.stack([points[n] for n in names]),
        prediction_names=json.dumps(names), fresh_ridge6=raw6, fresh_ridge174=raw174,
        ridge6_prediction_age=age6, ridge174_prediction_age=age174,
        raw_alpha_end=alpha_end, warning_scores=np.stack([scores[n] for n in warning_names]),
        warning_score_names=json.dumps(warning_names), warning_thresholds=serialized_thresholds,
        truth=data['truth'], times=data['times'], observed=data['observed'], age=data['age'],
        fitter_features=data['fitter_features'], fitter_observed=data['fitter_observed'],
        fitter_age=data['fitter_age'], records=json.dumps(data['records']),
        protocol=json.dumps(protocol))
    probe.M.G.O.C.save_json(result_path, T(result))
    print('FITTER_COMPARISON_COMPLETE', len(data['records']), 'records', result_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='score smoke_dataset.npz mechanically; does not validate the full four-design contract')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(smoke=args.smoke)
