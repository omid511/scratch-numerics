"""Primary endpoints plus matched-control/constraint side-effect comparisons."""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_correction as E
from threadpoolctl import threadpool_limits
spec = importlib.util.spec_from_file_location('coordinate_scoring_helpers', E.BASE / 'analyze_results.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
C, np, OUT = E.C, E.np, E.OUT


def analyze():
    fits = json.loads((OUT / 'fits.json').read_text())
    assert fits['all_converged'] and len(fits['fits']) == 20
    for record in fits['fits']:
        assert record['success']
        if record['constrained']:
            assert min(record['constraint_slacks']) >= -1e-7
    with np.load(OUT / 'predictions.npz') as z:
        p, t = z['predictions'][:, 0], z['thresholds'][:, 0]
        y, ids, folds = z['y'], z['ids'], z['row_fold']
        names, arms = z['conditions'].tolist(), z['arms'].tolist()
        fingerprint = json.loads(str(z['fingerprint']))
        assert arms == E.ARMS and z['seeds'].tolist() == [42]
    assert fingerprint == E.signature() and np.isfinite(p).all()
    comparisons = [('blend_control', 'blend_frozen')]
    for arm in arms[2:]:
        comparisons.extend((arm, ref) for ref in ('blend_frozen', 'blend_control'))
    comparisons.extend([('hurwitz_unconstrained', 'products_unconstrained'),
                        ('hurwitz_constrained', 'products_constrained'),
                        ('hurwitz_constrained', 'hurwitz_unconstrained'),
                        ('products_constrained', 'products_unconstrained')])
    output = dict(protocol=json.loads((OUT / 'protocol.json').read_text()),
        hashes={k: C.sha256(OUT / k) for k in ('protocol.json', 'predictions.npz', 'fits.json', 'analyze_correction.py')},
        scope=dict(physical_clips=len(y), designs=len(np.unique(ids)), folds=5, seeds=[42], conditions=names),
        conditions={}, families={})
    groups = [(name, [name], output['conditions']) for name in names]
    groups += [(family, members, output['families']) for family, members in A.FAMILIES.items()]
    for label, members, destination in groups:
        indices = [names.index(n) for n in members]
        value = dict(acquisition_views=members, arms={a: A.scores(p[i, indices], t[i], y, ids, folds)
                                                    for i, a in enumerate(arms)}, paired={})
        for candidate, reference in comparisons:
            ai, ri = arms.index(candidate), arms.index(reference)
            value['paired'].setdefault(candidate, {})[reference] = A.compare(p[ri, indices], p[ai, indices],
                                                                            t[ri], t[ai], y, ids, folds)
        destination[label] = value
        if label == 'clean' or label in A.FAMILIES:
            print('CORRECTION_ENDPOINTS', label, json.dumps({a: {
                'optimism03_unstable_last1': v['slices']['unstable0.01']['over03'],
                'optimism03_stable_last1': v['slices']['stable0.01']['over03'],
                'warning5_miss': v['warning']['0.05']['unstable_last1_miss']['rate'],
                'warning5_false': v['warning']['0.05']['stable_false_warning']['rate'],
                'warning1_miss': v['warning']['0.01']['unstable_last1_miss']['rate'],
                'warning1_false': v['warning']['0.01']['stable_false_warning']['rate'],
                'design_mae': v['slices']['all']['design_mae'], 'far_under05': v['slices']['far_stable']['under05']}
                for a, v in value['arms'].items()}), flush=True)
    C.save_json(OUT / 'results.json', output)
    print('CORRECTION_EVALUATION_COMPLETE', len(names), len(A.FAMILIES), 'all endpoints, budgets and matched controls', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        analyze()
