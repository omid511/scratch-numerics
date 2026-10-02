"""Fixed reduced ingredient fits and finalist matched-width controls."""
import argparse
import json
from common import *
from threadpoolctl import threadpool_limits


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arms',default='F,P,C,S,PC,PS,CS');ap.add_argument('--controls',action='store_true');args=ap.parse_args()
    train,evaluation,y,ids,rows,names,folds,_,_=load_data()
    arms=['sham_'+a for a in json.loads((OUT/'selection.json').read_text())['selected']] if args.controls else args.arms.split(',')
    assert all(a.removeprefix('sham_') in MASKS[:-1] for a in arms)
    signature={**source_signature(),'execution':sha256(Path(__file__))}
    if args.controls: signature['selection']=sha256(OUT/'selection.json')
    tr=np.full((len(arms),3,4,len(names),len(y)),np.nan);ho=np.full((len(arms),3,len(names),len(y)),np.nan)
    records=[]
    for ai,arm in enumerate(arms):
        cols=columns(arm.removeprefix('sham_'))
        for si,seed in enumerate((42,43,44)):
            for fi in range(5):
                fit,test=np.flatnonzero(folds!=fi),np.flatnonzero(folds==fi)
                assert not set(ids[fit])&set(ids[test])
                rng=np.random.default_rng(934+fi)
                pf,pt=(rng.permutation(fit),rng.permutation(test)) if arm.startswith('sham_') else (None,None)
                path=model_path(arm,seed,fi)
                arrays={'train_rows':fit,'held_rows':test,'columns':cols}
                if pf is not None: arrays.update(addition_train_rows=pf,addition_held_rows=pt)
                x=np.stack([query(q,fit,arm,pf) for q in train]).reshape(-1,len(cols))
                predict,meta=fitted_model(x,np.tile(y[fit],4),seed,path,signature,**arrays)
                a=np.stack([predict(query(q,fit,arm,pf)) for q in evaluation])
                b=np.stack([predict(query(q,test,arm,pt)) for q in evaluation])
                assert np.isfinite(a).all() and np.isfinite(b).all()
                slots=fi-(folds[fit]<fi).astype(int)
                for ci in range(len(names)):
                    tr[ai,si,slots,ci,fit]=a[ci];ho[ai,si,ci,test]=b[ci]
                records.append({'arm':arm,'seed':seed,'fold':fi,'features':len(cols),**meta})
        print('Completed arm',arm,flush=True)
    suffix='controls' if args.controls else '_'.join(arms)
    save_npz(OUT/f'predictions_{suffix}.npz',training_predictions=tr,heldout_predictions=ho,y=y,ids=ids,
             source_indices=rows,row_fold=folds,conditions=names,arms=arms,seeds=[42,43,44],fingerprint=json.dumps(signature))
    save_json(OUT/f'fits_{suffix}.json',{'fingerprint':signature,'fits':records,'all_converged':all(r['success'] for r in records)})
    print('Completed matrix batch',arms,'fits',len(records),'all converged',all(r['success'] for r in records),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
