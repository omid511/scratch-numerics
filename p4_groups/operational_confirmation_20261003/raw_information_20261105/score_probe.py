"""Score completed first-probe fits; never score held designs or claim qualification."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_native():
    spec = importlib.util.spec_from_file_location('_raw_information_fit', HERE / 'fit_runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rate(mask, events):
    n = int(mask.sum())
    k = int((mask & events).sum())
    return {'count': n, 'errors': k, 'rate': k / n if n else None}


def design_metrics(truth, prediction, counts):
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    counts = np.asarray(counts).reshape(-1)
    if not len(truth) or truth.shape != prediction.shape or truth.shape != counts.shape:
        raise ValueError('empty or mismatched design arrays')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or np.any((counts < 0) | (counts > 384)):
        raise ValueError('invalid design data')
    error = prediction - truth
    absolute = np.abs(error)
    boundary = (truth > -.02) & (truth < .05)
    strata = {}
    for name, mask in [('none', counts == 0), ('partial', (counts > 0) & (counts < 384)), ('full', counts == 384)]:
        n = int(mask.sum())
        total = float(absolute[mask].sum())
        strata[name] = {'count': n, 'conditional_mae': total / n if n else None,
                        'contribution': total / len(truth)}
    all_mae = float(absolute.mean())
    if not np.isclose(sum(x['contribution'] for x in strata.values()), all_mae, rtol=1e-12, atol=1e-15):
        raise RuntimeError('strata do not partition all-frame MAE')
    return {'frames': len(truth), 'all_mae': all_mae,
            'boundary_count': int(boundary.sum()),
            'boundary_mae': float(absolute[boundary].mean()) if boundary.any() else None,
            'optimism': rate(boundary, error > .02),
            'stable_sign': rate((truth >= .01) & (truth <= .05), prediction <= 0),
            'unstable_sign': rate((truth >= -.05) & (truth <= -.01), prediction > 0),
            'strata': strata}


def equal_mean(values):
    return float(np.mean(values, dtype=np.float64)) if all(v is not None for v in values) and values else None


def summarize(designs, noise):
    out = {'designs': designs}
    for metric in ('all_mae', 'boundary_mae'):
        out[metric] = equal_mean([d[metric] for d in designs.values()])
    for metric in ('optimism', 'stable_sign', 'unstable_sign'):
        out[metric] = equal_mean([d[metric]['rate'] for d in designs.values()])
    out['strata'] = {name: {'count': sum(d['strata'][name]['count'] for d in designs.values()),
                            'contribution': equal_mean([d['strata'][name]['contribution'] for d in designs.values()]),
                            'conditional_mae_by_design': {k: d['strata'][name]['conditional_mae'] for k, d in designs.items()}}
                     for name in ('none', 'partial', 'full')}
    limits = {'all_mae': .010 if noise == 'clean' else .015,
              'boundary_mae': .005 if noise == 'clean' else .010,
              'optimism': .05, 'stable_sign': .10, 'unstable_sign': .10}
    out['diagnostic_targets'] = {key: {'limit': limit, 'passed': out[key] is not None and out[key] <= limit}
                                 for key, limit in limits.items()}
    return out


def compare(wave, mask):
    result = {}
    for key in ('all_mae', 'boundary_mae'):
        differences = {d: (None if wave['designs'][d][key] is None or mask['designs'][d][key] is None else
                           wave['designs'][d][key] - mask['designs'][d][key]) for d in wave['designs']}
        mean = equal_mean(list(differences.values()))
        baseline = mask[key]
        result[key] = {'per_design_waveform_minus_mask': differences, 'mean_difference': mean,
                       'relative_mean_change': mean / baseline if mean is not None and baseline is not None and baseline != 0 else None}
    return result


def window_counts(admissible, rows, frames):
    ends = 128 + 64 * np.asarray(frames)
    return np.array([np.count_nonzero(admissible[row, max(0, int(end) - 384):int(end)])
                     for row, end in zip(rows, ends)], dtype=np.int16)


def check_artifacts(F, smoke, fit, records):
    E = F.E
    digest = E.sha(F.PROTOCOL)
    designs = np.array([r['design'] for r in records])
    noises = np.array([r['noise'] for r in records])
    held = np.isin(designs, smoke['split']['held'])
    fitting = set(smoke['split']['train']) | (set(smoke['additional_designs']) - set(smoke['evaluation_folds'][0]))
    proper = np.isin(designs, list(fitting)) & np.isin(noises, ('clean', 'iid1'))
    validation = np.isin(designs, smoke['split']['validation']) & np.isin(noises, ('clean', 'iid1'))
    if len(records) != 2094 or not held.any() or not proper.any() or not validation.any() or np.any((proper | validation) & held):
        raise RuntimeError('invalid reconstructed cohorts')
    predictions = {}
    for arm in ('waveform', 'mask'):
        directory = HERE / ('fit_' + arm)
        receipt = json.loads((directory / 'final_receipt.json').read_text())
        identity = F.identity(digest, arm, proper, validation, held)
        if receipt['status'] != 'fit_complete' or receipt['protocol_sha256'] != digest or receipt['identity'] != identity:
            raise RuntimeError(f'{arm}: receipt identity or completion mismatch')
        for filename, field in [('final_predictions.npz', 'predictions_sha256'), ('best_model.pt', 'model_sha256')]:
            actual = E.sha(directory / filename)
            if receipt[field] != actual or receipt['artifacts'][filename] != actual:
                raise RuntimeError(f'{arm}: {filename} fingerprint mismatch')
        if E.sha(directory / 'checkpoint.pt') != receipt['artifacts']['checkpoint.pt']:
            raise RuntimeError(f'{arm}: checkpoint fingerprint mismatch')
        provenance = F.verify_snapshots(directory)
        if receipt['provenance'] != provenance or provenance['fit_inputs'] != fit['inputs'] or provenance['smoke_inputs'] != smoke['inputs']:
            raise RuntimeError(f'{arm}: provenance mismatch')
        with np.load(directory / 'final_predictions.npz', allow_pickle=False) as archive:
            pred = archive['predictions']
            if pred.dtype != np.float32 or pred.shape != (2094, 124) or not np.isnan(pred[held]).all() or not np.isfinite(pred[~held]).all():
                raise RuntimeError(f'{arm}: prediction shape, dtype or protection mismatch')
            for key, expected in [('proper', proper), ('validation', validation), ('prediction_rows', ~held)]:
                array = archive[key]
                if array.dtype != np.bool_ or not np.array_equal(array, expected):
                    raise RuntimeError(f'{arm}: {key} cohort mismatch')
            if archive['protocol_sha256'].shape != () or str(archive['protocol_sha256']) != digest:
                raise RuntimeError(f'{arm}: prediction protocol mismatch')
            meta = json.loads(str(archive['metadata']))
            if any(meta[k] != v for k, v in [('protocol_sha256', digest), ('arm', arm), ('fold', 0), ('seed', 42), ('identity', identity), ('source_and_input_provenance', provenance)]):
                raise RuntimeError(f'{arm}: prediction metadata mismatch')
            for key in ('best_epoch', 'epochs', 'best_validation_mae'):
                if meta[key] != receipt[key]:
                    raise RuntimeError(f'{arm}: {key} mismatch')
            predictions[arm] = pred.copy()
    return predictions, designs, noises, held


def preflight(F):
    missing = [str(HERE / ('fit_' + arm) / name) for arm in ('waveform', 'mask')
               for name in ('final_receipt.json', 'final_predictions.npz', 'best_model.pt', 'checkpoint.pt', 'provenance.json')
               if not (HERE / ('fit_' + arm) / name).is_file()]
    if missing:
        raise FileNotFoundError('completed fit artifacts missing: ' + ', '.join(missing))
    fit, smoke, errors = F.preflight()
    if errors:
        raise RuntimeError('input preflight failed: ' + json.dumps(errors))
    for key in ('split', 'additional_designs', 'evaluation_folds', 'primary_noise'):
        if fit[key] != smoke[key]:
            raise RuntimeError(f'fit/smoke {key} mismatch')
    if fit['fold'] != 0 or fit['seed'] != 42 or fit['arms'] != ['waveform', 'mask'] or len(smoke['evaluation_folds'][0]) != 4 or len(smoke['primary_noise']) != 7:
        raise RuntimeError('unexpected frozen study identity')
    E = F.E
    old_dataset = E.D.DATASET
    try:
        E.D.DATASET = E.MANIFEST
        records, times, records_text, _ = E.D._load_manifest()
    finally:
        E.D.DATASET = old_dataset
    checked = check_artifacts(F, smoke, fit, records)
    return fit, smoke, (records, times, records_text, checked)

def execute(F, fit, smoke, checked_inputs):
    E = F.E
    records, times, records_text, checked = checked_inputs
    old_dataset, old_out, old_shards = E.D.DATASET, E.D.OUT, E.D.E.SHARDS
    try:
        E.D.DATASET, E.D.OUT, E.D.E.SHARDS = E.MANIFEST, E.NORMALIZED, E.SHARDS
        _, admissible = E.D.load_measured()
    finally:
        E.D.DATASET, E.D.OUT, E.D.E.SHARDS = old_dataset, old_out, old_shards
    predictions, designs, noises, held = checked
    with np.load(E.DATASET, allow_pickle=False) as archive:
        if str(archive['records']) != records_text or not np.array_equal(archive['times'], times):
            raise RuntimeError('label and manifest records/times mismatch')
        truth = archive['truth'].copy()
    truth[held] = np.nan
    if truth.dtype != np.float64 or truth.shape != (2094, 124):
        raise RuntimeError('unexpected truth dtype or shape')
    results = {}
    for noise in smoke['primary_noise']:
        arms = {arm: {} for arm in predictions}
        for design in smoke['evaluation_folds'][0]:
            rows = np.flatnonzero((designs == design) & (noises == noise) & ~held)
            if not len(rows):
                raise RuntimeError(f'missing evaluation population: {design} {noise}')
            row_ids = np.repeat(rows, 124)
            frames = np.tile(np.arange(124), len(rows))
            counts = window_counts(admissible, row_ids, frames)
            for arm in arms:
                arms[arm][design] = design_metrics(truth[rows], predictions[arm][rows], counts)
        wave, mask = (summarize(arms[arm], noise) for arm in ('waveform', 'mask'))
        results[noise] = {'waveform': wave, 'mask': mask, 'paired': compare(wave, mask)}
    output = {'scope': 'First development probe, four previously inspected designs and one seed; no independent confidence intervals, blind validation or qualification.',
              'fit_protocol_sha256': E.sha(F.PROTOCOL), 'scorer_sha256': E.sha(Path(__file__)),
              'fit_receipt_sha256': {arm: E.sha(HERE / ('fit_' + arm) / 'final_receipt.json') for arm in predictions},
              'evaluation_designs': smoke['evaluation_folds'][0], 'noise_views': results}
    destination = HERE / 'comparison.json'
    if destination.exists():
        raise FileExistsError('refusing to replace existing comparison.json')
    E.atomic_json(destination, output)
    print(json.dumps({'status': 'scored', 'output': str(destination)}))


def selfcheck():
    y = np.array([-.05, -.02, -.01, .01, .05, 0.0], dtype=np.float64)
    p = np.array([.01, 0, .01, 0, 0, .03], dtype=np.float64)
    a = design_metrics(y, p, [0, 1, 384, 0, 384, 1])
    assert a['boundary_count'] == 3 and a['optimism']['count'] == 3 and a['optimism']['errors'] == 1
    assert a['stable_sign'] == {'count': 2, 'errors': 2, 'rate': 1.0}
    assert a['unstable_sign'] == {'count': 3, 'errors': 2, 'rate': 2 / 3}
    assert np.isclose(a['strata']['none']['contribution'], (.06 + .01) / 6)
    assert np.isclose(a['strata']['partial']['contribution'], (.02 + .03) / 6)
    assert np.isclose(a['strata']['full']['contribution'], (.02 + .05) / 6)
    b = design_metrics([.02], [.12], [384])
    assert b['unstable_sign']['rate'] is None and b['strata']['none']['contribution'] == 0
    summary = summarize({'A': a, 'B': b}, 'clean')
    assert np.isclose(summary['all_mae'], (a['all_mae'] + .1) / 2)
    assert not np.isclose(summary['all_mae'], (a['all_mae'] * 6 + .1) / 7)
    assert summary['unstable_sign'] is None and not summary['diagnostic_targets']['unstable_sign']['passed']
    assert np.isclose(sum(x['contribution'] for x in summary['strata'].values()), summary['all_mae'])
    admissible = np.ones((1, 512), dtype=bool)
    assert window_counts(admissible, [0, 0], [0, 4]).tolist() == [128, 384]
    evidence = {'status': 'passed', 'checks': ['strict boundary and closed sign sides', 'null eligibility fails target', 'equal design weighting with unequal frames', 'original denominator stratum contributions and empty stratum', 'native causal window count'], 'synthetic_only': True}
    (HERE / 'score_selfcheck.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    print(json.dumps(evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['selfcheck'], default=None)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.phase == 'selfcheck':
        if args.execute:
            parser.error('--execute cannot be combined with --phase selfcheck')
        selfcheck()
        return
    F = load_native()
    fit, smoke, checked = preflight(F)
    if args.execute:
        execute(F, fit, smoke, checked)
    else:
        print(json.dumps({'status': 'preflight_passed', 'scoring_executed': False}))


if __name__ == '__main__':
    main()
