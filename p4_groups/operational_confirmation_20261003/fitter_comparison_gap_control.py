"""Post-hoc gap-policy control; no new fitting, model selection or noise tuning."""
import json
from pathlib import Path
import numpy as np
import fitter_comparison_score as S


def run():
    out = S.OUT
    target = out / 'gap_policy_control.json'
    if target.exists():
        raise FileExistsError('Keep completed gap-policy control immutable')
    data = S.load_dataset(out / 'dataset.npz')
    original = json.loads((out / 'results.json').read_text())
    with np.load(out / 'predictions.npz') as archive:
        points = dict(zip(json.loads(str(archive['prediction_names'])), archive['predictions']))
        alpha = archive['raw_alpha_end'].copy()
    held_points = {}
    for name in ('framewise', 'gru'):
        held_points[name + '_hold'], _ = S.causal_hold(points[name], data['observed'], data['times'])
    held_alpha, _ = S.causal_hold(alpha, data['fitter_observed'], data['times'])
    scores = {name: -point for name, point in held_points.items()}
    scores['alpha_hold'] = held_alpha
    validation = np.asarray([r['split'] == 'validation' and r['noise'] in ('clean', 'iid1')
                             for r in data['records']])[:, None]
    safe = np.broadcast_to(validation, data['truth'].shape) & (data['truth'] > .02)
    thresholds = {name: {str(b): S.calibrate_threshold(score, safe, b) for b in S.BUDGETS}
                  for name, score in scores.items()}
    groups = {}
    for split in ('validation', 'held'):
        for noise in original['protocol']['noise']:
            selected = np.asarray([r['split'] == split and r['noise'] == noise
                                   for r in data['records']])[:, None]
            group = np.broadcast_to(selected, data['truth'].shape)
            common = group.copy()
            for point in points.values():
                common &= np.isfinite(point)
            for point in held_points.values():
                common &= np.isfinite(point)
            fresh = group & data['observed'] & data['fitter_observed']
            margins = {name: {'common_finite_all_frame': S.summary(data['truth'], point, common),
                              'common_fresh': S.summary(data['truth'], point, fresh)}
                       for name, point in {**points, **held_points}.items()}
            warnings = {name: {str(b): S.event_report(data['truth'], score,
                                                   thresholds[name][str(b)]['threshold'], group)
                               for b in S.BUDGETS} for name, score in scores.items()}
            warnings['alpha_hold_sign'] = S.event_report(data['truth'], held_alpha, 0., group)
            groups[split + '/' + noise] = {'margins': margins, 'warnings': warnings}
    result = {'posthoc_diagnostic': True,
              'purpose': 'Separate new estimator/readout from causal hold-last versus native neural gap output; no new training or winner selection',
              'provenance': {'dataset_sha256': S.sha256(out / 'dataset.npz'),
                             'predictions_sha256': S.sha256(out / 'predictions.npz'),
                             'original_results_sha256': S.sha256(out / 'results.json'),
                             'source_sha256': S.sha256(Path(__file__)),
                             'score_helper_sha256': S.sha256(Path(S.__file__))},
              'warning_thresholds': thresholds, 'per_split_noise': groups,
              'limitations': ['Control added after first result, not a new predeclared primary comparison',
                              'Two held designs; no significance or deployment claim',
                              'Hold-last outputs remain stale, not measured availability or safety certification']}
    S.probe.M.G.O.C.save_json(target, S.probe.M.clean_json(result))
    for noise in original['protocol']['noise']:
        row = groups['held/' + noise]
        print('GAP_CONTROL', noise,
              {n: m['common_finite_all_frame']['mae'] for n, m in row['margins'].items()},
              {n: (w['0.05']['positive']['recall'], w['0.05']['safe']['false_warning_rate'])
               for n, w in row['warnings'].items() if n != 'alpha_hold_sign'}, flush=True)


if __name__ == '__main__':
    run()
