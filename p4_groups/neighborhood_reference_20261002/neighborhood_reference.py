"""Fixed TRAIN-only neighborhood mapping against the unchanged frozen blend."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_observability_20261001'))
import contract as frozen
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
np, helpers = frozen.np, frozen.helpers
ROOT, OUT = frozen.ROOT, Path(__file__).resolve().parent
save_json, save_npz, sha256 = frozen.save_json, frozen.save_npz, frozen.sha256
K = 5


class DesignNeighborhood:
    def __init__(self, views, targets, ids, source_rows):
        if views.ndim != 3 or views.shape[:2] != (4, len(targets)):
            raise ValueError('expected four acquisition views per physical reference clip')
        if len(ids) != len(targets) or len(source_rows) != len(targets):
            raise ValueError('unaligned physical reference metadata')
        if not np.isfinite(views).all() or not np.isfinite(targets).all():
            raise ValueError('nonfinite reference data')
        self.targets, self.ids, self.source_rows = targets, ids, source_rows
        self.groups = [np.flatnonzero(ids == d) for d in np.unique(ids)]
        if len(self.groups) < K:
            raise ValueError('need at least five distinct proper-fit designs')
        x = views.reshape(-1, views.shape[-1])
        self.mean, self.scale, self.active = fit_ridge_scaling(x)
        self.reference = np.ascontiguousarray(((x - self.mean) / self.scale)[:, self.active])
        if not self.reference.shape[1]:
            raise ValueError('no active reference features')
        self.norm = np.einsum('ij,ij->i', self.reference, self.reference)

    def distances(self, q):
        z = ((q - self.mean) / self.scale)[:, self.active]
        distance = z @ self.reference.T
        distance *= -2
        distance += self.norm[None]
        distance += np.einsum('ij,ij->i', z, z)[:, None]
        np.maximum(distance, 0., out=distance)
        return distance.reshape(len(q), 4, len(self.targets)).min(axis=1)

    def predict(self, q):
        if q.ndim != 2 or q.shape[1] != len(self.mean) or not np.isfinite(q).all():
            raise ValueError('expected finite aligned query features')
        points = np.empty(len(q))
        rows = np.empty((len(q), K), dtype=np.int64)
        rms = np.empty((len(q), K))
        for start in range(0, len(q), 128):
            block = q[start:start + 128]
            d = self.distances(block)
            nearest = np.stack([g[d[:, g].argmin(axis=1)] for g in self.groups], axis=1)
            design_distance = np.take_along_axis(d, nearest, axis=1)
            chosen = np.argsort(design_distance, axis=1, kind='stable')[:, :K]
            indices = np.take_along_axis(nearest, chosen, axis=1)
            end = start + len(block)
            points[start:end] = np.median(self.targets[indices], axis=1)
            rows[start:end] = self.source_rows[indices]
            rms[start:end] = np.sqrt(np.take_along_axis(d, indices, axis=1) / self.reference.shape[1])
        return points, rows, rms


def signature():
    return {'execution': sha256(Path(__file__)), 'protocol': sha256(OUT / 'protocol.json'),
            'frozen_sources': frozen.signature(), 'baseline_predictions': sha256(frozen.OUT / 'predictions.npz')}


def load_data():
    data = frozen.load_data()
    with np.load(frozen.OUT / 'predictions.npz') as z:
        for key, value in [('y', data[2]), ('ids', data[3]), ('source_indices', data[4]), ('row_fold', data[6])]:
            np.testing.assert_array_equal(z[key], value)
        assert z['conditions'].tolist() == data[5] and z['seeds'].tolist() == frozen.SEEDS
        assert z['arms'][0] == 'blend_frozen'
        baseline, thresholds = z['predictions'][0], z['thresholds'][0]
    return data, baseline, thresholds


def verify_neighbors(refs, proper, ids, source_rows):
    lookup = {int(r): i for i, r in enumerate(source_rows)}
    idx = np.array([lookup[int(r)] for r in refs.ravel()]).reshape(refs.shape)
    assert np.isin(idx, proper).all(), 'neighbor outside proper-fit partition'
    assert all(len(set(v)) == K for v in ids[idx].reshape(-1, K)), 'duplicate design votes'


def numerical_smoke():
    # Unequal design sizes, duplicate acquisition copies, target-blind selection and ties.
    ids = np.array(['A', 'A', 'B', 'C', 'D', 'E', 'F'])
    targets = np.array([.1, .9, .2, .3, .4, .5, .6])
    x = np.array([0., .01, 1., 2., 3., 4., 5.])[None, :, None]
    views = np.repeat(x, 4, axis=0)
    model = DesignNeighborhood(views, targets, ids, np.arange(7))
    q = np.array([[0.], [2.5], [5.]])
    z = ((q - model.mean) / model.scale)[:, model.active]
    direct = ((z[:, None] - model.reference[None]) ** 2).sum(axis=-1).reshape(3, 4, 7).min(axis=1)
    np.testing.assert_allclose(model.distances(q), direct, atol=1e-12, rtol=1e-12)
    point, refs, _ = model.predict(q)
    np.testing.assert_allclose(point, [.3, .4, .4])
    np.testing.assert_array_equal(refs[0], [0, 2, 3, 4, 5])
    changed = DesignNeighborhood(views, -targets, ids, np.arange(7))
    np.testing.assert_array_equal(changed.predict(q)[1], refs)
    np.testing.assert_allclose(changed.predict(q)[0], -point)
    tied = DesignNeighborhood(np.repeat(np.arange(6.)[None, :, None], 4, axis=0), np.arange(6.), np.array(list('ABCDEF')), np.arange(6))
    # With odd k, distances to0 and5 tie at2.5; lexical designA wins the fifth vote.
    np.testing.assert_array_equal(tied.predict(np.array([[2.5]]))[1][0], [2, 3, 1, 4, 0])
    print('NEIGHBOR_NUMERICAL_SMOKE_PASSED: direct distances, distinct designs, targets and stable ties', flush=True)


def run(smoke=False):
    data, baseline, baseline_thresholds = load_data()
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    fingerprint = signature()
    predictions = np.full((len(names), len(y), len(frozen.FIELDS)), np.nan)
    thresholds = np.full((5, len(frozen.BUDGETS)), np.nan)
    manifests = []
    for fold in range(1 if smoke else 5):
        proper, cal, held = frozen.split(ids, folds, fold)
        path = OUT / f'fold{fold}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint'])) == fingerprint, 'stale neighborhood fold'
                for key, value in [('proper_rows', proper), ('calibration_rows', cal), ('held_rows', held)]:
                    np.testing.assert_array_equal(z[key], value)
                hp, levels, neighbors = z['predictions'], z['thresholds'], z['neighbors']
                manifest = json.loads(str(z['manifest']))
        else:
            member_cal, member_held, neighbors, distances, scaling = [], [], [], [], []
            for width in (107, 174):
                model = DesignNeighborhood(train[:, proper, :width], y[proper], ids[proper], rows[proper])
                cp, cref, _ = model.predict(train[:, cal, :width].reshape(-1, width))
                hp, href, distance = model.predict(evaluation[:, held, :width].reshape(-1, width))
                verify_neighbors(cref, proper, ids, rows)
                verify_neighbors(href, proper, ids, rows)
                member_cal.append(cp)
                member_held.append(hp)
                neighbors.append(href.reshape(len(names), len(held), K))
                distances.append(distance.reshape(len(names), len(held), K))
                scaling.append({'width': width, 'active_features': int(model.active.sum())})
                print('NEIGHBOR_MEMBER_COMPLETE', fold, width, len(hp), flush=True)
            cp = .5 * (member_cal[0] + member_cal[1])
            hp = .5 * (member_held[0] + member_held[1])
            target, design = np.tile(y[cal], 4), np.tile(ids[cal], 4)
            radius = frozen.calibration(cp, target, design)
            cp_dist = frozen.distribution(cp, radius)
            hp = frozen.distribution(hp, radius).reshape(len(names), len(held), -1)
            levels = np.array([frozen.threshold(cp_dist[:, 3], target, b) for b in frozen.BUDGETS])
            manifest = {'fold': fold, 'proper_designs': np.unique(ids[proper]).tolist(),
                        'calibration_designs': np.unique(ids[cal]).tolist(), 'held_designs': np.unique(ids[held]).tolist(),
                        'members': scaling, 'calibration': radius,
                        'calibration_false_warning': [float(np.mean(-cp_dist[target > 0, 3] <= t)) for t in levels]}
            neighbors = np.stack(neighbors)
            save_npz(path, predictions=hp, thresholds=levels, neighbors=neighbors,
                     neighbor_distances=np.stack(distances), member_points=np.stack(member_held).reshape(2, len(names), len(held)),
                     calibration_points=cp, proper_rows=proper, calibration_rows=cal, held_rows=held,
                     manifest=json.dumps(manifest), fingerprint=json.dumps(fingerprint))
        verify_neighbors(neighbors, proper, ids, rows)
        # Exercise both actual frozen members; saved seed predictions remain separate.
        for si, seed in enumerate(frozen.SEEDS):
            one, full = frozen.reference_predictors(seed, fold, proper, cal, held)
            q = evaluation[:, held].reshape(-1, 174)
            actual = .5 * (one(q[:, :107]) + full(q))
            np.testing.assert_allclose(actual.reshape(len(names), -1), baseline[si, :, held, 0].T, atol=1e-12, rtol=1e-12)
        predictions[:, held] = hp
        thresholds[fold] = levels
        manifests.append(manifest)
        print('NEIGHBOR_FOLD_COMPLETE', fold, len(held), flush=True)
    selected = np.flatnonzero(folds == 0) if smoke else np.arange(len(y))
    assert np.isfinite(predictions[:, selected]).all()
    assert np.isfinite(thresholds[:1] if smoke else thresholds).all()
    assert signature() == fingerprint, 'source changed during experiment'
    save_npz(OUT / ('smoke_predictions.npz' if smoke else 'predictions.npz'), candidate=predictions[:, selected],
             candidate_thresholds=thresholds, baseline=baseline[:, :, selected], baseline_thresholds=baseline_thresholds,
             budgets=frozen.BUDGETS, fields=frozen.FIELDS, y=y[selected], ids=ids[selected], source_indices=rows[selected],
             row_fold=folds[selected], conditions=names, baseline_seeds=frozen.SEEDS, fingerprint=json.dumps(fingerprint))
    save_json(OUT / ('smoke_manifest.json' if smoke else 'manifest.json'), {'folds': manifests,
              'partial_smoke': smoke, 'deterministic_candidate': True, 'new_mlp_fits': 0, 'fingerprint': fingerprint})
    print('NEIGHBOR_SMOKE_PASSED' if smoke else 'NEIGHBOR_EXPERIMENT_COMPLETE', predictions[:, selected].shape, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with frozen.threadpool_limits(limits=1):
        if args.smoke:
            numerical_smoke()
        run(args.smoke)
