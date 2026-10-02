"""Read-only TRAIN failure audit; no fitting or production-policy changes."""
import argparse
from contract import *
from run_experiment import load_features, query


def load_audit_data():
    data, additions = load_features()
    with np.load(OUT / 'predictions.npz') as z:
        recorded = z['predictions'][:3, ..., 0]
        for key, value in [('y', data[2]), ('ids', data[3]), ('source_indices', data[4]), ('row_fold', data[6])]:
            np.testing.assert_array_equal(z[key], value)
    return data, additions, recorded


def predictors(arm, seed, fold, proper, cal, held):
    if arm == 'blend_frozen':
        return reference_predictors(seed, fold, proper, cal, held)
    models = []
    for width in (107, 174):
        path = OUT / f'{arm}_{width}_seed{seed}_fold{fold}.npz'
        with np.load(path) as z:
            for key, value in [('proper_rows', proper), ('calibration_rows', cal), ('held_rows', held)]:
                np.testing.assert_array_equal(z[key], value)
        models.append(helpers.base.trusted_predictor(path))
    return models


def inputs(base, controls, local, selected, width, arm):
    if arm == 'blend_frozen':
        return base[:, selected, :width].reshape(-1, width)
    return query(base, controls, local, selected, width, arm, np.arange(len(selected)))


def predict(models, base, controls, local, selected, arm):
    members = [model(inputs(base, controls, local, selected, width, arm))
               for model, width in zip(models, (107, 174))]
    return .5 * (members[0] + members[1]), np.stack(members)


def replay(data, additions, recorded, require_accurate=False):
    train, evaluation, y, ids, rows, names, folds, layout, pairs, dt = data
    tc, tl, ec, el = additions
    records = []
    for source, seed in ((3696, 43), (5198, 42)):
        k = int(np.flatnonzero(rows == source)[0])
        fold, ci = int(folds[k]), names.index('iid1_new949')
        proper, cal, held = split(ids, folds, fold)
        assert ids[k] not in ids[proper] and ids[k] not in ids[cal]
        record = {'source_index': source, 'design': str(ids[k]), 'seed': seed, 'fold': fold, 'margin': float(y[k]), 'predictions': {}}
        for ai, arm in enumerate(ARMS[:3]):
            models = predictors(arm, seed, fold, proper, cal, held)
            value, members = predict(models, evaluation[ci:ci+1], ec[ci:ci+1], el[ci:ci+1], np.array([k]), arm)
            np.testing.assert_allclose(value[0], recorded[ai, SEEDS.index(seed), ci, k], atol=1e-12, rtol=1e-12)
            record['predictions'][arm] = {'point': float(value[0]), 'error': float(value[0] - y[k]), 'members': members[:, 0].tolist()}
        records.append(record)
        print('ACTUAL_FAILURE_REPLAY', json.dumps(record), flush=True)
    if require_accurate and any(r['predictions']['blend_local']['error'] > .03 for r in records):
        raise AssertionError('Local model overestimates true margin by more than .03 on named held-design clips')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--require-accurate', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        data, additions, recorded = load_audit_data()
        replay(data, additions, recorded, args.require_accurate)
