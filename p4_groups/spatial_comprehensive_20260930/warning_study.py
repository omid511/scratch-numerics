"""Proper-fit predictors and design-disjoint calibration; no threshold transfer."""
from common import *
from threadpoolctl import threadpool_limits


def threshold(scores,budget):
    scores=np.sort(np.asarray(scores));k=int(np.floor(budget*len(scores)))
    value=float(np.nextafter(scores[k],-np.inf))
    assert np.mean(scores<=value)<=budget
    return value


def main():
    selection=json.loads((OUT/'selection.json').read_text());base=['one','PCS']+selection['selected'];arms=base+['blend']
    train,_,y,ids,rows,_,folds,_,_=load_data()
    with np.load(OUT/'confirmation_features.npz') as z:
        x,names=z['features'],z['conditions'].tolist();np.testing.assert_array_equal(z['source_indices'],rows)
    budgets=np.array([.01,.05,.10,.20]);pred=np.full((len(arms),3,8,len(y)),np.nan)
    levels=np.full((len(arms),3,5,len(budgets)),np.nan);fits=[];splits=[]
    calibration_rates=np.full_like(levels,np.nan)
    signature={**source_signature(),'execution':sha256(Path(__file__)),'selection':sha256(OUT/'selection.json')}
    for fi in range(5):
        outer=np.flatnonzero(folds!=fi);test=np.flatnonzero(folds==fi)
        designs=np.unique(ids[outer]);np.random.default_rng(947+fi).shuffle(designs)
        cal_ids=designs[:int(np.ceil(.2*len(designs)))];fit_ids=designs[len(cal_ids):]
        fit=outer[np.isin(ids[outer],fit_ids)];cal=outer[np.isin(ids[outer],cal_ids)]
        assert not set(ids[fit])&set(ids[cal]) and not set(ids[outer])&set(ids[test])
        splits.append({'fold':fi,'proper_designs':fit_ids.tolist(),'calibration_designs':cal_ids.tolist(),'heldout_designs':np.unique(ids[test]).tolist()})
        for si,seed in enumerate((42,43,44)):
            calibrations={};held={}
            for ai,arm in enumerate(arms):
                if arm=='blend':
                    score=.5*(calibrations['one']+calibrations['PCS']);out=.5*(held['one']+held['PCS'])
                else:
                    cols=columns(arm);path=OUT/f'warning_{arm}_seed{seed}_fold{fi}.npz'
                    tx=np.stack([query(q,fit,arm) for q in train]).reshape(-1,len(cols))
                    predict,meta=fitted_model(tx,np.tile(y[fit],4),seed,path,signature,proper_rows=fit,calibration_rows=cal,held_rows=test,columns=cols)
                    if not meta['success']:raise RuntimeError(f'warning fit failed: {path}')
                    score=np.concatenate([predict(query(q,cal,arm)) for q in train]);out=np.stack([predict(query(q,test,arm)) for q in x])
                    calibrations[arm]=score;held[arm]=out;fits.append({'arm':arm,'seed':seed,'fold':fi,'model_hash':sha256(path),**meta})
                stable=score[np.tile(y[cal]>0,4)]
                for bi,budget in enumerate(budgets):
                    levels[ai,si,fi,bi]=threshold(stable,budget)
                    calibration_rates[ai,si,fi,bi]=np.mean(stable<=levels[ai,si,fi,bi])
                for ci in range(8):pred[ai,si,ci,test]=out[ci]
    assert np.isfinite(pred).all() and np.isfinite(levels).all()
    save_npz(OUT/'warning_predictions.npz',predictions=pred,thresholds=levels,calibration_rates=calibration_rates,
             budgets=budgets,arms=arms,conditions=names,y=y,ids=ids,source_indices=rows,row_fold=folds)
    result={'splits':splits,'fits':fits,'conditions':{},'hashes':{'source':sha256(Path(__file__)),
        'protocol':sha256(OUT/'protocol.json'),'predictions':sha256(OUT/'warning_predictions.npz')}}
    for ci,c in enumerate(names):
        result['conditions'][c]={}
        for ai,arm in enumerate(arms):
            records={}
            for bi,label in enumerate(['zero']+[str(b) for b in budgets]):
                t=0. if bi==0 else levels[ai,:,folds,bi-1].T
                warned=pred[ai,:,ci,:]<=t
                record={}
                for side,m in [('stable_all',y>0),('unstable_all',y<=0),('stable_last1',(y>0)&(y<.01)),('unstable_last1',(y<=0)&(y>-.01))]:
                    events=warned[:,m] if side.startswith('stable') else ~warned[:,m]
                    record[side]={'clips':int(m.sum()),'designs':int(len(np.unique(ids[m]))),'rate':float(events.mean()),
                                  'per_seed':events.mean(axis=1).tolist(),
                                  'design_bootstrap_95':paired_rate_difference(np.zeros_like(events),events,ids[m])['design_bootstrap_95']}
                records[label]=record
            result['conditions'][c][arm]=records
        print('Warning tradeoffs',c,{a:{b:(r['stable_all']['rate'],r['unstable_all']['rate']) for b,r in records.items()} for a,records in result['conditions'][c].items()},flush=True)
    save_json(OUT/'warning_results.json',result)
    print('Completed design-disjoint warning calibration',len(fits),'fits',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
