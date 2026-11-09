"""Complete paired boundary/warning evaluation; no outcome-driven tuning."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as E
from threadpoolctl import threadpool_limits
C, np, OUT = E.C, E.np, E.OUT
FAMILIES = {
    'iid1': ['iid1_new949', 'iid1_new951'],
    'correlated1': ['correlated1_new949_950', 'correlated1_new951_952'],
    'iid5': ['iid5_new949', 'iid5_new951'],
    'correlated5': ['correlated5_new949_950', 'correlated5_new951_952'],
    'replication_iid1': ['iid1_new943', 'iid1_new945'],
    'replication_correlated1': ['correlated1_new943_944', 'correlated1_new945_946'],
    'replication_iid5': ['iid5_new943', 'iid5_new945'],
    'replication_correlated5': ['correlated5_new943_944', 'correlated5_new945_946']}


def slices(y):
    masks = {'all': np.ones(len(y), bool), 'far_stable': y >= .15}
    for eps in (.01, .02, .05):
        masks[f'stable{eps}'] = (y > 0) & (y < eps)
        masks[f'unstable{eps}'] = (y < 0) & (y > -eps)
    return masks


def warnings(p, thresholds, folds, budget):
    if budget == 0:
        return p[..., 0] <= 0
    return -p[..., 3] <= thresholds[:, C.BUDGETS.index(budget)][folds]


def scores(p, thresholds, y, ids, folds):
    masks = slices(y)
    output = dict(slices={}, warning={}, per_fold={})
    error = p[..., 0] - y
    for label, mask in masks.items():
        output['slices'][label] = dict(physical_clips=int(mask.sum()), designs=len(np.unique(ids[mask])),
            **C.helpers.point(y[mask], p[:, mask, 0], ids[mask]),
            signed_p95=float(np.quantile(error[:, mask], .95)), signed_p99=float(np.quantile(error[:, mask], .99)),
            over02=float((error[:, mask] > .02).mean()),
            over03_count_per_view=(error[:, mask] > .03).sum(axis=1).tolist())
    for budget in [0., *C.BUDGETS]:
        w = warnings(p, thresholds, folds, budget)
        output['warning'][str(budget)] = {}
        for label, mask, missed in [('stable_false_warning', y > 0, False), ('unstable_miss', y <= 0, True),
                                   ('stable_last1_warning', masks['stable0.01'], False),
                                   ('unstable_last1_miss', masks['unstable0.01'], True)]:
            event = ~w[:, mask] if missed else w[:, mask]
            output['warning'][str(budget)][label] = dict(rate=float(event.mean()),
                physical_clips=int(mask.sum()), designs=len(np.unique(ids[mask])),
                counts_per_view=event.sum(axis=1).tolist(), rates_per_view=event.mean(axis=1).tolist())
    for fold in range(5):
        m = folds == fold
        near = m & masks['unstable0.01']
        stable = m & (y > 0)
        output['per_fold'][str(fold)] = dict(
            design_mae=C.helpers.point(y[m], p[:, m, 0], ids[m])['design_mae'],
            near_unstable_physical_clips=int(near.sum()),
            near_unstable_over03=float((error[:, near] > .03).mean()),
            warning={str(b): dict(misses=float((~warnings(p, thresholds, folds, b)[:, near]).mean()),
                                  false_warnings=float(warnings(p, thresholds, folds, b)[:, stable].mean()))
                     for b in C.BUDGETS})
    return output


def compare(control, candidate, ct, pt, y, ids, folds):
    masks = slices(y)
    output = {'slices': {label: C.helpers.paired_metrics(y[mask], control[:, mask, 0], candidate[:, mask, 0], ids[mask])
                         for label, mask in masks.items()}, 'warning': {}}
    for budget in C.BUDGETS:
        wc, wp = warnings(control, ct, folds, budget), warnings(candidate, pt, folds, budget)
        output['warning'][str(budget)] = {}
        for label, mask, missed in [('stable_false_warning', y > 0, False), ('unstable_miss', y <= 0, True),
                                   ('unstable_last1_miss', masks['unstable0.01'], True)]:
            a, b = (wc[:, mask], wp[:, mask]) if not missed else (~wc[:, mask], ~wp[:, mask])
            output['warning'][str(budget)][label] = C.helpers.paired_rate_difference(a, b, ids[mask])
    return output


def analyze():
    fits = json.loads((OUT / 'fits.json').read_text())
    assert fits['all_converged'] and len(fits['fits']) == 10
    for record in fits['fits']:
        assert C.sha256(E.ROOT / record['path']) == record['hash']
    with np.load(OUT / 'predictions.npz') as z:
        p, t = z['predictions'][:, 0], z['thresholds'][:, 0]
        y, ids, folds = z['y'], z['ids'], z['row_fold']
        names, arms = z['conditions'].tolist(), z['arms'].tolist()
        fingerprint = json.loads(str(z['fingerprint']))
        assert z['seeds'].tolist() == [42] and arms == E.ARMS and len(y) == 4454
    assert fingerprint == E.signature() and np.isfinite(p).all() and np.isfinite(t).all()
    output = dict(protocol=json.loads((OUT / 'protocol.json').read_text()),
        hashes={k: C.sha256(OUT / k) for k in ('protocol.json', 'predictions.npz', 'fits.json', 'analyze_results.py')},
        scope=dict(physical_clips=len(y), designs=len(np.unique(ids)), folds=5, seeds=[42], conditions=names),
        conditions={}, families={})
    comparisons = [('blend_products', 'blend_frozen'), ('blend_hurwitz', 'blend_frozen'),
                   ('blend_hurwitz', 'blend_products'), ('full_products', 'full_frozen'),
                   ('full_hurwitz', 'full_frozen'), ('full_hurwitz', 'full_products')]
    groups = [(name, [name], output['conditions']) for name in names]
    groups += [(family, members, output['families']) for family, members in FAMILIES.items()]
    for label, members, destination in groups:
        indices = [names.index(n) for n in members]
        value = dict(acquisition_views=members, arms={a: scores(p[i, indices], t[i], y, ids, folds) for i, a in enumerate(arms)}, paired={})
        for candidate, reference in comparisons:
            ai, ri = arms.index(candidate), arms.index(reference)
            value['paired'].setdefault(candidate, {})[reference] = compare(p[ri, indices], p[ai, indices], t[ri], t[ai], y, ids, folds)
        destination[label] = value
        if label == 'clean' or label in FAMILIES:
            print('COORDINATE_ENDPOINTS', label, json.dumps({a: {
                'optimism03_unstable_last1': v['slices']['unstable0.01']['over03'],
                'optimism03_stable_last1': v['slices']['stable0.01']['over03'],
                'warning5_miss': v['warning']['0.05']['unstable_last1_miss']['rate'],
                'warning5_false': v['warning']['0.05']['stable_false_warning']['rate'],
                'warning1_miss': v['warning']['0.01']['unstable_last1_miss']['rate'],
                'warning1_false': v['warning']['0.01']['stable_false_warning']['rate'],
                'design_mae': v['slices']['all']['design_mae'],
                'far_under05': v['slices']['far_stable']['under05']}
                for a, v in value['arms'].items()}), flush=True)
    C.save_json(OUT / 'results.json', output)
    print('COORDINATE_EVALUATION_COMPLETE', len(names), len(FAMILIES), 'all endpoints, budgets, folds and guardrails', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        analyze()
