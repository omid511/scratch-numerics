"""Score retained additions jointly only after independent retention is immutable."""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as E
spec = importlib.util.spec_from_file_location('integration_evaluation', Path(__file__).with_name('evaluate_results.py'))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
C, np, OUT = E.C, E.np, E.OUT


def combine():
    retained_path = OUT / 'retention.json'
    selection = json.loads(retained_path.read_text())
    assert selection['independent_results_hash'] == E.sha256(OUT / 'results.json')
    assert selection['protocol_hash'] == E.sha256(OUT / 'protocol.json')
    retained = selection['retained']
    record = dict(retention_hash=E.sha256(retained_path), independent_results_hash=selection['independent_results_hash'],
                  retained=retained, source=E.sha256(Path(__file__)),
                  interpretation='Same previously scored TRAIN acquisitions. Post-selection combination is exploratory, not independent confirmation.')
    if not retained:
        record.update(status='empty_retained_set', final_candidate='blend_frozen',
                      joint_fit_performed=False, reason='No independently eligible addition; no rejected methods mixed.')
        E.save_json(OUT / 'combination.json', record)
        print('RETAINED_COMBINATION_COMPLETE: empty retained set; frozen blend remains final candidate', flush=True)
        return
    with np.load(OUT / 'predictions.npz') as z:
        p, t, y, ids, folds = [z[k] for k in ('predictions', 'thresholds', 'y', 'ids', 'row_fold')]
        names, seeds, old_arms = z['conditions'].tolist(), z['seeds'].tolist(), z['arms'].tolist()
    arms = ['blend_frozen', 'blend_control', *retained]
    if len(retained) >= 2:
        arms += ['joint_retained', 'equal_retained']
    predictions = np.full((len(arms), len(seeds), len(names), len(y), 6), np.nan)
    thresholds = np.full((len(arms), len(seeds), 5, len(C.BUDGETS)), np.nan)
    for ai, arm in enumerate(arms[:2 + len(retained)]):
        predictions[ai] = p[old_arms.index(arm)]
        thresholds[ai] = t[old_arms.index(arm)]
    del p, t
    heads, splits = [], []
    ai = [E.ADDONS.index(a.split('_', 1)[1]) for a in retained]
    if len(retained) >= 2:
        for si, seed in enumerate(seeds):
            for fold in range(5):
                source = OUT / f'fusion_seed{seed}_fold{fold}.npz'
                with np.load(source) as z:
                    proper, cal, held = [z[k] for k in ('proper_rows', 'calibration_rows', 'held_rows')]
                    assert all(not set(ids[x]) & set(ids[v]) for x, v in ((proper, cal), (proper, held), (cal, held)))
                    tx, cx, hx = [z[k].reshape(-1, 7) for k in ('train_control', 'calibration_control', 'held_control')]
                    ta, ca, ha = [z[k].reshape(-1, len(E.ADDONS))[:, ai] for k in ('train_addons', 'calibration_addons', 'held_addons')]
                blocks = [np.column_stack([x, a - x[:, :1]]) for x, a in ((tx, ta), (cx, ca), (hx, ha))]
                head = E.head_fit(blocks[0], np.tile(y[proper], 4) - tx[:, 0])
                points = {
                    'joint_retained': (cx[:, 0] + E.head_predict(head, blocks[1]), hx[:, 0] + E.head_predict(head, blocks[2])),
                    'equal_retained': ((cx[:, 0] + ca.sum(axis=1)) / (1 + len(ai)),
                                       (hx[:, 0] + ha.sum(axis=1)) / (1 + len(ai)))}
                radii = {}
                for arm, (cp, hp) in points.items():
                    index = arms.index(arm)
                    radius = C.calibration(cp, np.tile(y[cal], 4), np.tile(ids[cal], 4))
                    cdist = C.distribution(cp, radius)
                    predictions[index, si, :, held] = C.distribution(hp, radius).reshape(len(names), len(held), 6).transpose(1, 0, 2)
                    thresholds[index, si, fold] = [C.threshold(cdist[:, 3], np.tile(y[cal], 4), b) for b in C.BUDGETS]
                    radii[arm] = radius
                path = OUT / f'joint_seed{seed}_fold{fold}.npz'
                E.save_npz(path, head=np.frombuffer(E.cloudpickle.dumps(head), dtype=np.uint8), calibration=json.dumps(radii),
                           proper_rows=proper, calibration_rows=cal, held_rows=held,
                           retained=retained, retention_hash=record['retention_hash'], independent_fold_hash=E.sha256(source))
                heads.append(dict(seed=seed, fold=fold, path=path.name, hash=E.sha256(path)))
                splits.append(dict(seed=seed, fold=fold, fit_designs=np.unique(ids[proper]).tolist(),
                                   calibration_designs=np.unique(ids[cal]).tolist(), held_designs=np.unique(ids[held]).tolist()))
    assert np.isfinite(predictions).all() and np.isfinite(thresholds).all()
    E.save_npz(OUT / 'combination_predictions.npz', predictions=predictions, thresholds=thresholds,
               y=y, ids=ids, row_fold=folds, conditions=names, seeds=seeds, arms=arms, retention_hash=record['retention_hash'])
    result = dict(arms=arms, retained=retained, conditions={}, families={}, hashes=record)
    candidates = arms[2:]
    for ci, name in enumerate(names):
        result['conditions'][name] = A.describe(predictions[:, :, ci], thresholds, y, ids, folds, arms)
        result['conditions'][name]['prediction_rows'] = [{'condition': name, 'seed': s} for s in seeds]
        if name in ['clean', *C.helpers.FRESH]:
            result['conditions'][name]['paired'] = {a: {
                ref: A.comparison(predictions[arms.index(ref), :, ci], predictions[arms.index(a), :, ci],
                                  thresholds[arms.index(ref)], thresholds[arms.index(a)], y, ids, folds)
                for ref in ['blend_frozen', 'blend_control', *[r for r in retained if r != a]]}
                for a in candidates}
    for family, members in A.FAMILIES.items():
        fp = np.stack([np.concatenate([predictions[i, :, names.index(n)] for n in members]) for i in range(len(arms))])
        ft = np.concatenate([thresholds, thresholds], axis=1)
        result['families'][family] = A.describe(fp, ft, y, ids, folds, arms)
        result['families'][family]['prediction_rows'] = [{'condition': n, 'seed': s} for n in members for s in seeds]
        result['families'][family]['row_note'] = 'Per-seed arrays contain one row per acquisition/seed pair in prediction_rows order; draws are not new physical designs.'
        result['families'][family]['paired'] = {a: {
            ref: A.comparison(fp[arms.index(ref)], fp[arms.index(a)], ft[arms.index(ref)], ft[arms.index(a)], y, ids, folds)
            for ref in ['blend_frozen', 'blend_control', *[r for r in retained if r != a]]}
            for a in candidates}
        print('COMBINATION_FAMILY', family, {a: round(result['families'][family]['slices']['all']['arms'][a]['design_mae'], 6) for a in arms}, flush=True)
    E.save_json(OUT / 'combination_results.json', result)
    record.update(status='scored_joint_retained_set' if len(retained) >= 2 else 'single_retained_route_replayed',
                  joint_fit_performed=len(retained) >= 2, final_candidates=arms[2:], fits=heads, partitions=splits,
                  prediction_hash=E.sha256(OUT / 'combination_predictions.npz'), results_hash=E.sha256(OUT / 'combination_results.json'))
    E.save_json(OUT / 'combination.json', record)
    assert E.sha256(retained_path) == record['retention_hash'], 'retention changed during combination'
    print('RETAINED_COMBINATION_COMPLETE', record['status'], retained, flush=True)


if __name__ == '__main__':
    with E.threadpool_limits(limits=1):
        combine()
