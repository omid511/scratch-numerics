"""Physically parameterized affine corrections; no new temporal/member models."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'p4_groups/stability_coordinates_20261003'
sys.path.insert(0, str(OUT))
spec = importlib.util.spec_from_file_location('physical_coordinate_base', BASE / 'run_experiment.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
C, np = H.C, H.np
from corrective_head import correction, fit_head, gradient_check
from threadpoolctl import threadpool_limits
ARMS = ['blend_frozen', 'blend_control', 'products_unconstrained', 'products_constrained',
        'hurwitz_unconstrained', 'hurwitz_constrained']
OLD = ROOT / 'p4_groups/blend_additions_20261002'
INNER = ROOT / 'p4_groups/blend_physical_integration_20261001'
DATA = None


def control_features(q, members):
    return np.column_stack((.5*(members[:, 0]+members[:, 1]), members[:, 0]-members[:, 1],
                            q[:, 27], q[:, 28], q[:, 20], q[:, 19], q[:, 29]))


def old_head_predict(head, x):
    z = (x - head['mean'])/head['scale']
    z[:, ~head['active']] = 0.
    return z @ head['coef'] + head['intercept']


def signature():
    paths = [OUT / 'protocol.json', Path(__file__), OUT / 'corrective_head.py']
    paths += [OLD / f'fusion_seed42_fold{f}.npz' for f in range(5)]
    paths += [INNER / f'inner_seed42_fold{f}_part{i}.npz' for f in range(5) for i in range(3)]
    return {'sources': {str(p.relative_to(ROOT)): C.sha256(p) for p in paths}, 'coordinates': H.signature()}


def setup():
    global DATA
    d = H.setup()
    DATA = {**d, 'fingerprint': signature()}
    return DATA


def _verify_archived_contract(archived, current):
    assert archived['versions'] == current['versions']
    assert archived['references'] == current['references']
    baseline = 'src/mechanics/p4_margin_estimation/baselines.py'
    sources = current['sources'].copy()
    proof = {}
    if archived['sources'][baseline] != sources[baseline]:
        # Reconstruct the exact archived bytes, not a broad hash exemption.
        now = (ROOT / baseline).read_text()
        prior = subprocess.check_output(
            ['rtk', 'proxy', 'git', 'show', 'd662243^:' + baseline], cwd=ROOT, text=True)
        def training_function(source):
            node = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'train_gru')
            return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
        present, previous = training_function(now), training_function(prior)
        assert now.count(present) == 1
        reconstructed = now.replace(present, previous)
        digest = hashlib.sha256(reconstructed.encode()).hexdigest()
        assert digest == archived['sources'][baseline], 'unexplained baseline source change'
        proof = dict(file=baseline, archived_sha256=digest, current_sha256=sources[baseline],
                     exact_difference='train_gru only; not called by this experiment',
                     reconstruction_commit='d662243^')
        sources[baseline] = digest
    assert archived['sources'] == sources
    return proof


def previous_fold(fold, proper, cal, held):
    d = DATA
    path = OLD / f'fusion_seed42_fold{fold}.npz'
    with np.load(path) as z:
        for key, value in [('proper_rows', proper), ('calibration_rows', cal), ('held_rows', held)]:
            np.testing.assert_array_equal(z[key], value)
        provenance = json.loads(str(z['fingerprint']))
        compatibility = _verify_archived_contract(provenance['frozen'], d['fingerprint']['coordinates']['contract'])
        if compatibility:
            d['baseline_compatibility'] = compatibility
            print('ARCHIVED_CONTRACT_VERIFIED', json.dumps(compatibility), flush=True)
        for file, digest in provenance['sources'].items():
            assert C.sha256(ROOT / file) == digest
        tx, cx, hx = z['train_control'], z['calibration_control'], z['held_control']
        heads = C.cloudpickle.loads(z['heads'].tobytes())
        ai = z['arms'].tolist().index('blend_control')
        control_prediction, control_thresholds = z['predictions'][ai], z['thresholds'][ai]
        records = json.loads(str(z['records']))
    designs = np.unique(d['ids'][proper]).copy()
    np.random.default_rng(1047+fold).shuffle(designs)
    replay_max, covered = 0., np.zeros(len(proper), int)
    for part, group in enumerate(np.array_split(designs, 3)):
        query = proper[np.isin(d['ids'][proper], group)]
        fit = proper[~np.isin(d['ids'][proper], group)]
        assert not set(d['ids'][fit]) & set(d['ids'][query])
        inner_path = INNER / f'inner_seed42_fold{fold}_part{part}.npz'
        recorded = next(r for r in records if r.get('path') == str(inner_path.relative_to(ROOT)))
        assert C.sha256(inner_path) == recorded['hash']
        with np.load(inner_path) as z:
            np.testing.assert_array_equal(z['fit_rows'], fit)
            np.testing.assert_array_equal(z['query_rows'], query)
            meta = json.loads(str(z['metadata']))
            assert meta['one']['success'] and meta['full']['success']
            assert meta['fit_designs'] == np.unique(d['ids'][fit]).tolist()
            assert meta['query_designs'] == np.unique(d['ids'][query]).tolist()
            # Read only the observed one/full predictions; no oracle-prior output.
            members = z['predictions'][:, :, :2].copy()
            objects = C.cloudpickle.loads(z['components'].tobytes())
        q = d['train'][:, query].reshape(-1, 174)
        actual = np.column_stack((objects['one'](q[:, :107]), objects['full'](q)))
        np.testing.assert_allclose(actual.reshape(4, len(query), 2), members, atol=1e-12, rtol=1e-12)
        slots = np.searchsorted(proper, query)
        controls = control_features(q, actual).reshape(4, len(query), 7)
        np.testing.assert_allclose(controls, tx[:, slots], atol=1e-12, rtol=1e-12)
        replay_max = max(replay_max, float(abs(controls-tx[:, slots]).max()))
        covered[slots] += 1
    assert np.all(covered == 1)
    one, full = C.reference_predictors(42, fold, proper, cal, held)
    for rows, base, recorded in [(cal, d['train'], cx), (held, d['evaluation'], hx)]:
        q = base[:, rows].reshape(-1, 174)
        controls = control_features(q, np.column_stack((one(q[:, :107]), full(q))))
        np.testing.assert_allclose(controls.reshape(recorded.shape), recorded, atol=1e-12, rtol=1e-12)
    return tx, cx, hx, heads['blend_control'], control_prediction, control_thresholds, replay_max


def run():
    d = setup()
    fingerprint = d['fingerprint']
    predictions = np.full((len(ARMS), 1, 37, len(d['y']), 6), np.nan)
    thresholds = np.full((len(ARMS), 1, 5, len(C.BUDGETS)), np.nan)
    records, partitions, gradient_proofs = [], [], []
    for fold in range(5):
        proper, cal, held = C.split(d['ids'], d['folds'], fold)
        tx, cx, hx, control_head, archived_control, archived_thresholds, replay_max = previous_fold(fold, proper, cal, held)
        designs = np.unique(d['ids'][proper]).copy()
        np.random.default_rng(2047+fold).shuffle(designs)
        objective_ids, validation_ids = np.array_split(designs, 2)
        update = np.flatnonzero(np.isin(d['ids'][proper], objective_ids))
        validation = np.flatnonzero(np.isin(d['ids'][proper], validation_ids))
        assert not set(objective_ids) & set(validation_ids)
        cp = [cx[:, :, 0].reshape(-1), cx[:, :, 0].reshape(-1)+old_head_predict(control_head, cx.reshape(-1, 7))]
        hp = [hx[:, :, 0].reshape(-1), hx[:, :, 0].reshape(-1)+old_head_predict(control_head, hx.reshape(-1, 7))]
        target, ctarget, cids = np.tile(d['y'][proper], 4), np.tile(d['y'][cal], 4), np.tile(d['ids'][cal], 4)
        heads = {'blend_control': control_head}
        fold_records = []
        for kind in ('products', 'hurwitz'):
            trainx = np.concatenate((tx, H.additions(d['train'][:, proper], kind)), axis=-1)
            calx = np.concatenate((cx, H.additions(d['train'][:, cal], kind)), axis=-1).reshape(-1, 28)
            heldx = np.concatenate((hx, H.additions(d['evaluation'][:, held], kind)), axis=-1).reshape(-1, 28)
            x, vx = trainx[:, update].reshape(-1, 28), trainx[:, validation].reshape(-1, 28)
            base, vbase = tx[:, update, 0].reshape(-1), tx[:, validation, 0].reshape(-1)
            y, vy = np.tile(d['y'][proper[update]], 4), np.tile(d['y'][proper[validation]], 4)
            if fold == 0:
                check = gradient_check(x, base, y, vx, vbase, vy)
                gradient_proofs.append(dict(kind=kind, **check))
                print('CORRECTION_GRADIENT_CHECK', kind, json.dumps(check), flush=True)
            for constrained in (False, True):
                arm = f'{kind}_{"constrained" if constrained else "unconstrained"}'
                head, meta = fit_head(x, base, y, vx, vbase, vy, constrained)
                heads[arm] = head
                cp.append(cx[:, :, 0].reshape(-1)+correction(head, calx))
                hp.append(hx[:, :, 0].reshape(-1)+correction(head, heldx))
                record = dict(arm=arm, fold=fold, seed=42, **meta)
                fold_records.append(record)
                print('CORRECTION_FIT', json.dumps(record), flush=True)
        radii, fold_predictions, fold_thresholds, budgets = {}, [], [], []
        for ai, arm in enumerate(ARMS):
            radius = C.calibration(cp[ai], ctarget, cids)
            cdist, hdist = C.distribution(cp[ai], radius), C.distribution(hp[ai], radius)
            levels = [C.threshold(cdist[:, 3], ctarget, b) for b in C.BUDGETS]
            rates = [float(np.mean(-cdist[ctarget>0, 3] <= level)) for level in levels]
            assert np.all(np.array(rates) <= C.BUDGETS)
            radii[arm] = radius
            fold_predictions.append(hdist.reshape(37, len(held), 6))
            fold_thresholds.append(levels); budgets.append(rates)
        fp, ft = np.array(fold_predictions), np.array(fold_thresholds)
        np.testing.assert_allclose(fp[0], d['baseline'][:, held], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fp[1], archived_control, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(ft[0], d['baseline_thresholds'][fold])
        np.testing.assert_array_equal(ft[1], archived_thresholds)
        assert np.isfinite(fp).all() and np.isfinite(ft).all()
        predictions[:, 0][:, :, held] = fp; thresholds[:, 0, fold] = ft
        partition = dict(fold=fold, proper_rows=proper.tolist(), calibration_rows=cal.tolist(), held_rows=held.tolist(),
                         objective_rows=proper[update].tolist(), constraint_rows=proper[validation].tolist(),
                         objective_designs=objective_ids.tolist(), constraint_designs=validation_ids.tolist())
        C.save_npz(OUT / f'fold{fold}.npz', predictions=fp, thresholds=ft,
            calibration_points=np.array(cp).reshape(len(ARMS), 4, len(cal)), calibration=json.dumps(radii),
            calibration_false_warning_rates=budgets, records=json.dumps(fold_records),
            heads=np.frombuffer(C.cloudpickle.dumps(heads), dtype=np.uint8), partitions=json.dumps(partition),
            fingerprint=json.dumps(fingerprint), old_oof_replay_max_error=replay_max)
        records.extend(fold_records); partitions.append(partition)
        print('CORRECTION_FOLD_COMPLETE', fold, 'OOF/control/frozen replay and calibration budgets passed', flush=True)
    assert len(records) == 20 and all(r['success'] for r in records)
    assert np.isfinite(predictions).all() and signature() == fingerprint
    C.save_npz(OUT / 'predictions.npz', predictions=predictions, thresholds=thresholds,
        y=d['y'], ids=d['ids'], source_indices=d['rows'], row_fold=d['folds'], conditions=d['names'],
        arms=ARMS, seeds=[42], budgets=C.BUDGETS, fields=C.FIELDS, fingerprint=json.dumps(fingerprint))
    C.save_json(OUT / 'fits.json', dict(all_converged=True, fits=records, partitions=partitions,
                                      gradient_checks=gradient_proofs, fingerprint=fingerprint,
                                      baseline_compatibility=d.get('baseline_compatibility', {})))
    print('CORRECTION_EXPERIMENT_COMPLETE', predictions.shape, 'twenty fitted affine heads; no member/temporal training', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
