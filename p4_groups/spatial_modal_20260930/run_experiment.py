"""Spatial versus equally wide shuffled-spatial MLP; fixed three-seed protocol."""
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'p4_groups/correlated_training_20260929'
sys.path[:0] = [str(ROOT),str(PARENT)]
import numpy as np
from threadpoolctl import threadpool_limits
from experiment_p4_tabular import fit_network
from run_augmentation_experiment import sha256


def main():
    filenames = ['augmentation_results.json','augmentation_seed43.json','augmentation_seed44.json']
    parents = [json.loads((PARENT/name).read_text()) for name in filenames]
    fingerprint = {
        'protocol': sha256(OUT/'protocol.json'), 'execution': sha256(Path(__file__)),
        'shared_features': sha256(PARENT/'shared.npz'), 'spatial_features': sha256(OUT/'spatial_features.npz'),
        'weighted_reference':sha256(ROOT/'p4_groups/local_weighting_20260929/predictions.npz'),
        'parents': {name:sha256(PARENT/name) for name in filenames},
        'sources': {name:sha256(ROOT/name) for name in (
            'experiment_p4_tabular.py','src/mechanics/p4_margin_estimation/baselines.py')}}
    with np.load(PARENT/'shared.npz') as z:
        base_train = np.concatenate((z['train3'],z['independent_extra'][None]))
        base_eval,y,ids,rows,conditions = z['evaluation'],z['y'],z['ids'],z['source_indices'],z['evaluation_names']
    with np.load(OUT/'spatial_features.npz') as z:
        extra_train,extra_eval = z['training_spatial'],z['evaluation_spatial']
        for key,reference in [('source_indices',rows),('conditions',conditions),('ids',ids),('y',y)]:
            np.testing.assert_array_equal(z[key],reference)
    assert extra_train.shape == (4,len(y),65) and extra_eval.shape == (len(conditions),len(y),65)
    seeds,arms = [42,43,44],['spatial','shuffled_spatial']
    n,nc = len(y),len(conditions)
    row_fold = np.asarray(parents[0]['design_assignments']['row_fold'])
    training = np.full((2,3,4,nc,n),np.nan)
    held = np.full((2,3,nc,n),np.nan)
    baseline = np.stack([[run['arms']['extra_independent4']['records'][str(c)]['predictions']
                          for c in conditions] for run in parents])
    with np.load(ROOT/'p4_groups/local_weighting_20260929/predictions.npz') as z:
        np.testing.assert_array_equal(z['source_indices'],rows)
        np.testing.assert_array_equal(z['conditions'],conditions)
        np.testing.assert_array_equal(z['y'],y)
        np.testing.assert_array_equal(z['ids'],ids)
        np.testing.assert_array_equal(z['row_fold'],row_fold)
        np.testing.assert_array_equal(z['seeds'],seeds)
        weighted_reference = z['heldout_predictions'][1]
    fits = []
    for si,(seed,parent) in enumerate(zip(seeds,parents)):
        np.testing.assert_array_equal(parent['design_assignments']['source_indices'],rows)
        np.testing.assert_array_equal(parent['design_assignments']['row_fold'],row_fold)
        for fold in parent['arms']['extra_independent4']['folds']:
            fi = fold['fold']; test = np.asarray(fold['test_row_indices']); fit = np.flatnonzero(row_fold != fi)
            assert not set(ids[fit])&set(ids[test])
            rng = np.random.default_rng(934+fi)
            shuffled_fit,shuffled_test = rng.permutation(fit),rng.permutation(test)
            for ai,arm in enumerate(arms):
                spatial_fit = fit if arm == 'spatial' else shuffled_fit
                spatial_test = test if arm == 'spatial' else shuffled_test
                assert set(spatial_fit)==set(fit) and set(spatial_test)==set(test)
                cache = OUT/f'{arm}_seed{seed}_fold{fi}.npz'
                if cache.exists():
                    with np.load(cache) as z:
                        assert json.loads(str(z['fingerprint'])) == fingerprint,'stale fit cache'
                        for key,reference in [('train_rows',fit),('held_rows',test),
                                              ('spatial_train_rows',spatial_fit),('spatial_held_rows',spatial_test)]:
                            np.testing.assert_array_equal(z[key],reference)
                        tr,ho,metadata = z['training_predictions'],z['heldout_predictions'],json.loads(str(z['metadata']))
                    print('Resume',arm,seed,fi,flush=True)
                else:
                    x = np.concatenate((base_train[:,fit],extra_train[:,spatial_fit]),axis=2).reshape(-1,107)
                    target = np.tile(y[fit],4)
                    predict,metadata = fit_network(x,target,seed=seed,alpha=1.,max_iter=10000,max_fun=200000)
                    tr = np.stack([predict(np.column_stack((q[fit],e[spatial_fit]))) for q,e in zip(base_eval,extra_eval)])
                    ho = np.stack([predict(np.column_stack((q[test],e[spatial_test]))) for q,e in zip(base_eval,extra_eval)])
                    if not np.isfinite(tr).all() or not np.isfinite(ho).all():
                        raise ValueError('nonfinite spatial candidate prediction')
                    metadata.update({'features':107,'training_rows':len(target)})
                    temporary = cache.with_suffix('.tmp')
                    with temporary.open('wb') as handle:
                        np.savez_compressed(handle,train_rows=fit,held_rows=test,
                                            spatial_train_rows=spatial_fit,spatial_held_rows=spatial_test,
                                            training_predictions=tr,heldout_predictions=ho,
                                            metadata=json.dumps(metadata),fingerprint=json.dumps(fingerprint))
                    temporary.replace(cache)
                    print('Fit',arm,seed,fi,json.dumps(metadata),flush=True)
                slots = fi-(row_fold[fit]<fi).astype(int)
                for ci in range(nc):
                    training[ai,si,slots,ci,fit] = tr[ci]
                    held[ai,si,ci,test] = ho[ci]
                fits.append({'arm':arm,'seed':seed,'fold':fi,**metadata})
    assert np.isfinite(training).all() and np.isfinite(held).all()
    path = OUT/'predictions.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle,training_predictions=training,heldout_predictions=held,
                            baseline_predictions=baseline,local2_reference_predictions=weighted_reference,
                            y=y,ids=ids,source_indices=rows,row_fold=row_fold,conditions=conditions,seeds=seeds,arms=arms)
    path.with_suffix('.tmp').replace(path)
    (OUT/'fits.json').write_text(json.dumps({'fingerprint':fingerprint,'fits':fits,
        'all_fits_converged':all(f['success'] for f in fits)},indent=2,allow_nan=False)+'\n')
    print('Completed',len(fits),'fits; all converged:',all(f['success'] for f in fits),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
