"""Three matched custom objectives, unchanged supervised views and frozen references."""
import argparse
from experiment_helpers import *
from threadpoolctl import threadpool_limits


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arms',default=','.join(ARMS));args=ap.parse_args()
    arms=args.arms.split(',');assert all(a in ARMS for a in arms)
    train,evaluation,y,ids,rows,names,folds,_,_=load_data();manifest=reference_manifest()
    sig={**signature(),'execution':sha256(Path(__file__)),'references':manifest}
    tr=np.full((len(arms),3,4,len(names),len(y)),np.nan);ho=np.full((len(arms),3,len(names),len(y)),np.nan);fits=[]
    for ai,arm in enumerate(arms):
        for si,seed in enumerate(SEEDS):
            for fi in range(5):
                fit,test=np.flatnonzero(folds!=fi),np.flatnonzero(folds==fi);n=len(fit)
                assert not set(ids[fit])&set(ids[test])
                x=train[:,fit].reshape(-1,174);target=np.tile(y[fit],4)
                pair=np.column_stack([n+np.arange(n),3*n+np.arange(n)])
                np.testing.assert_array_equal(target[pair[:,0]],target[pair[:,1]])
                weights=(abs(y[fit])<.05).astype(float) if arm=='localized' else np.ones(n)
                assert weights.sum()>0
                weight=0. if arm=='supervised' else 1.
                path=OUT/f'{arm}_seed{seed}_fold{fi}.npz'
                arrays={'train_rows':fit,'held_rows':test,'pairs':pair,'pair_weights':weights,'train_source_indices':rows[fit]}
                if path.exists():
                    with np.load(path) as z:
                        assert json.loads(str(z['fingerprint']))==sig,'stale objective fit'
                        for k,v in arrays.items():np.testing.assert_array_equal(z[k],v)
                        predict=cloudpickle.loads(z['predictor'].tobytes());meta=json.loads(str(z['metadata']))
                    print('Resume',arm,seed,fi,flush=True)
                else:
                    predict,meta=fit_network(x,target,seed=seed,alpha=1.,custom=True,
                        consistency_pairs=pair,consistency_weight=weight,
                        consistency_pair_weights=weights if arm=='localized' else None)
                    p=predict(x);gap=p[pair[:,0]]-p[pair[:,1]]
                    near=abs(y[fit])<.05
                    meta.update(supervised_half_mse=float(np.mean((p-target)**2)/2),
                        global_pair_half_mse=float(np.mean(gap**2)/2),
                        near_pair_half_mse=float(np.mean(gap[near]**2)/2),
                        consistency_lambda=weight,pairs=n,weighted_pairs=int(weights.sum()),
                        optimizer='matched custom L-BFGS-B')
                    save_npz(path,predictor=np.frombuffer(cloudpickle.dumps(predict),dtype=np.uint8),metadata=json.dumps(meta),
                             fingerprint=json.dumps(sig),**arrays)
                    print('Fit',arm,seed,fi,json.dumps(meta),flush=True)
                a=np.stack([predict(q[fit]) for q in evaluation]);b=np.stack([predict(q[test]) for q in evaluation])
                assert np.isfinite(a).all() and np.isfinite(b).all()
                slots=fi-(folds[fit]<fi).astype(int)
                for ci in range(len(names)):tr[ai,si,slots,ci,fit]=a[ci];ho[ai,si,ci,test]=b[ci]
                fits.append({'arm':arm,'seed':seed,'fold':fi,'model_hash':sha256(path),**meta})
    suffix='_'.join(arms)
    assert np.isfinite(tr).all() and np.isfinite(ho).all()
    save_npz(OUT/f'predictions_{suffix}.npz',training_predictions=tr,heldout_predictions=ho,y=y,ids=ids,
             source_indices=rows,row_fold=folds,conditions=names,arms=arms,seeds=SEEDS,fingerprint=json.dumps(sig))
    save_json(OUT/f'fits_{suffix}.json',{'fingerprint':sig,'fits':fits,'all_converged':all(f['success'] for f in fits)})
    print('Completed matched objectives',arms,len(fits),'fits; converged',all(f['success'] for f in fits),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
