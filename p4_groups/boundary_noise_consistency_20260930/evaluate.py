"""Evaluate matched loss arms and hash-pinned native references on fresh draws."""
from experiment_helpers import *
from threadpoolctl import threadpool_limits


def main():
    manifest=reference_manifest();sig=signature()
    with np.load(OUT/'fresh_features.npz') as z:
        x,y,ids,rows,folds,names=z['features'],z['y'],z['ids'],z['source_indices'],z['row_fold'],z['conditions'].tolist()
    arms=ARMS+['one','PCS','blend'];tr=np.full((6,3,4,8,len(y)),np.nan);ho=np.full((6,3,8,len(y)),np.nan);hashes={}
    for si,seed in enumerate(SEEDS):
        for fi in range(5):
            fit,test=np.flatnonzero(folds!=fi),np.flatnonzero(folds==fi)
            assert not set(ids[fit])&set(ids[test]);predictions={}
            for ai,a in enumerate(arms):
                if a=='blend':
                    f,h=tuple(.5*(u+v) for u,v in zip(predictions['one'],predictions['PCS']))
                else:
                    path=base.model_path(a,seed,fi) if a in ('one','PCS') else OUT/f'{a}_seed{seed}_fold{fi}.npz'
                    hashes[path.name]=sha256(path)
                    if a in ('one','PCS'):assert hashes[path.name]==manifest['models'][path.name]
                    with np.load(path) as z:
                        meta=json.loads(str(z['metadata']))
                        if a in ARMS:assert json.loads(str(z['fingerprint']))['sources']==sig['sources']
                        predict=cloudpickle.loads(z['predictor'].tobytes())  # Own source-pinned artifacts; failures marked below.
                    width=107 if a=='one' else 174
                    f=np.stack([predict(q[fit,:width]) for q in x]);h=np.stack([predict(q[test,:width]) for q in x])
                predictions[a]=(f,h);slots=fi-(folds[fit]<fi).astype(int)
                for ci in range(8):tr[ai,si,slots,ci,fit]=f[ci];ho[ai,si,ci,test]=h[ci]
    assert np.isfinite(tr).all() and np.isfinite(ho).all()
    np.testing.assert_array_equal(ho[5],.5*(ho[3]+ho[4]));np.testing.assert_array_equal(tr[5],.5*(tr[3]+tr[4]))
    save_npz(OUT/'fresh_predictions.npz',training_predictions=tr,heldout_predictions=ho,y=y,ids=ids,
             source_indices=rows,row_fold=folds,conditions=names,arms=arms,seeds=SEEDS)
    save_json(OUT/'evaluation_manifest.json',{'models':hashes,'source':sha256(Path(__file__)),
        'features':sha256(OUT/'fresh_features.npz'),'predictions':sha256(OUT/'fresh_predictions.npz')})
    print('Completed fresh evaluation:45 matched loss models,30 frozen references; no refitting',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
