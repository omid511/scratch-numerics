"""Paired warning gains and held false alarms; no probability interpretation."""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ranking as R
from threadpoolctl import threadpool_limits
spec = importlib.util.spec_from_file_location('ranking_scoring_helpers', R.E.BASE / 'analyze_results.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
C, np, OUT = R.C, R.np, R.OUT


def analyze():
    fits = json.loads((OUT / 'fits.json').read_text())
    assert fits['all_converged'] and len(fits['fits']) == 15
    with np.load(OUT / 'predictions.npz') as z:
        p, t = z['predictions'][:, 0], z['thresholds'][:, 0]
        y, ids, folds = z['y'], z['ids'], z['row_fold']
        names, arms = z['conditions'].tolist(), z['arms'].tolist()
        assert json.loads(str(z['fingerprint'])) == R.signature()
        assert arms == R.ARMS and z['fields'].tolist() == R.FIELDS and z['seeds'].tolist() == [42]
    assert np.isfinite(p).all() and np.isfinite(t).all()
    for candidate in p:
        np.testing.assert_array_equal(candidate[..., [0,1,2,4,5]], p[0][..., [0,1,2,4,5]])
    comparisons = [(arm, 'blend_frozen') for arm in arms[1:]]
    comparisons += [('rank_products', 'rank_control'), ('rank_hurwitz', 'rank_control'), ('rank_hurwitz', 'rank_products')]
    output = dict(protocol=json.loads((OUT / 'protocol.json').read_text()),
        hashes={k: C.sha256(OUT / k) for k in ('protocol.json', 'predictions.npz', 'fits.json', 'analyze_ranking.py')},
        scope=dict(physical_clips=len(y), designs=len(np.unique(ids)), folds=5, seeds=[42], conditions=names),
        frozen_margin_and_intervals_exact=True, warning_score_is_uncalibrated=True, conditions={}, families={})
    groups = [(name, [name], output['conditions']) for name in names]
    groups += [(family, members, output['families']) for family, members in A.FAMILIES.items()]
    for label, members, destination in groups:
        indices = [names.index(n) for n in members]
        value = dict(acquisition_views=members, arms={a: A.scores(p[i, indices], t[i], y, ids, folds)
                                                    for i, a in enumerate(arms)}, paired={})
        masks = A.slices(y)
        # Every accuracy comparison is exactly identical, compute once per group.
        same = {name: C.helpers.paired_metrics(y[mask], p[0, indices][:, mask, 0],
                                               p[0, indices][:, mask, 0], ids[mask]) for name, mask in masks.items()}
        for candidate, reference in comparisons:
            ai, ri = arms.index(candidate), arms.index(reference)
            pair = dict(slices=same, warning={})
            for budget in C.BUDGETS:
                wc, wp = A.warnings(p[ri, indices], t[ri], folds, budget), A.warnings(p[ai, indices], t[ai], folds, budget)
                result = {}
                for name, mask, missed in [('stable_false_warning', y>0, False), ('unstable_miss', y<=0, True),
                                           ('unstable_last1_miss', masks['unstable0.01'], True)]:
                    a, b = (~wc[:, mask], ~wp[:, mask]) if missed else (wc[:, mask], wp[:, mask])
                    result[name] = C.helpers.paired_rate_difference(a, b, ids[mask])
                pair['warning'][str(budget)] = result
            value['paired'].setdefault(candidate, {})[reference] = pair
        destination[label] = value
        if label == 'clean' or label in A.FAMILIES:
            print('WARNING_RANK_ENDPOINTS', label, json.dumps({arm: {
                'optimism03': v['slices']['unstable0.01']['over03'],
                'miss5': v['warning']['0.05']['unstable_last1_miss']['rate'],
                'false5': v['warning']['0.05']['stable_false_warning']['rate'],
                'miss1': v['warning']['0.01']['unstable_last1_miss']['rate'],
                'false1': v['warning']['0.01']['stable_false_warning']['rate']}
                for arm, v in value['arms'].items()}), flush=True)
    C.save_json(OUT / 'results.json', output)
    print('WARNING_RANK_EVALUATION_COMPLETE', len(names), len(A.FAMILIES), 'all budgets, primary endpoints, matched controls and guardrails', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        analyze()
