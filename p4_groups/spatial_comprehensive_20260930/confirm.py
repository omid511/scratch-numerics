"""Frozen finalist/reference/sham evaluation and strict preregistered acceptance."""
from common import *
from assess_matrix import point
from threadpoolctl import threadpool_limits


def main():
    selection=json.loads((OUT/'selection.json').read_text());selected=selection['selected']
    with np.load(OUT/'confirmation_features.npz') as z:
        x,y,ids,rows,folds,names=z['features'],z['y'],z['ids'],z['source_indices'],z['row_fold'],z['conditions'].tolist()
    arms=['one','PCS','blend']+selected+['sham_'+a for a in selected]
    tr=np.full((len(arms),3,4,8,len(y)),np.nan);ho=np.full((len(arms),3,8,len(y)),np.nan);hashes={}
    for si,seed in enumerate((42,43,44)):
        for fi in range(5):
            fit,test=np.flatnonzero(folds!=fi),np.flatnonzero(folds==fi)
            assert not set(ids[fit])&set(ids[test])
            predictions={}
            for ai,arm in enumerate(arms):
                if arm=='blend':a,b=tuple(.5*(a+b) for a,b in zip(predictions['one'],predictions['PCS']))
                else:
                    path=model_path(arm,seed,fi);hashes[path.name]=sha256(path)
                    if arm in selected:assert hashes[path.name]==selection['model_hashes'][arm][f'{seed}_{fi}']
                    predict=trusted_predictor(path);rng=np.random.default_rng(934+fi)
                    pf,pt=(rng.permutation(fit),rng.permutation(test)) if arm.startswith('sham_') else (None,None)
                    a=np.stack([predict(query(q,fit,arm,pf)) for q in x]);b=np.stack([predict(query(q,test,arm,pt)) for q in x])
                predictions[arm]=(a,b);slots=fi-(folds[fit]<fi).astype(int)
                for ci in range(8):tr[ai,si,slots,ci,fit]=a[ci];ho[ai,si,ci,test]=b[ci]
    np.testing.assert_array_equal(ho[2],.5*(ho[0]+ho[1]));assert np.isfinite(ho).all() and np.isfinite(tr).all()
    save_npz(OUT/'confirmation_predictions.npz',training_predictions=tr,heldout_predictions=ho,
             y=y,ids=ids,source_indices=rows,row_fold=folds,conditions=names,arms=arms)
    masks={'all':np.ones(len(y),bool),'far':y>=.15}
    for eps in (.01,.02,.05):masks[f'stable{eps}']=(y>0)&(y<eps);masks[f'unstable{eps}']=(y<0)&(y>-eps)
    result={'conditions':{},'acceptance':{},'seed_fold_stability':{},'model_hashes':hashes,'selection_hash':sha256(OUT/'selection.json'),
            'predictions_hash':sha256(OUT/'confirmation_predictions.npz'),'analysis_hash':sha256(Path(__file__))}
    for ci,c in enumerate(names):
        records={}
        for label,m in masks.items():
            target,groups=y[m],ids[m];a,b=tr[:,:,:,ci,:][...,m],ho[:,:,ci,:][...,m]
            records[label]={'clips':int(m.sum()),'designs':int(len(np.unique(groups))),
                'arms':{arm:{'training':summarize(target,a[ai],groups,'three seeds x four in-sample exposures'),
                              'heldout':summarize(target,b[ai],groups,'three paired seeds x one OOF')} for ai,arm in enumerate(arms)},
                'differences':{},'wrong_side':{}}
            for ai,arm in enumerate(arms):
                if arm.startswith('sham_') or arm=='one':continue
                refs=['one','PCS','blend'] if arm in selected else ['one']
                if arm in selected:refs+=['sham_'+arm]
                records[label]['differences'][arm]={ref:arm_difference(target,b[arms.index(ref)],b[ai],groups,'candidate_minus_reference') for ref in refs if ref!=arm}
                for side,sm in [('stable',target>0),('unstable',target<0)]:
                    if not sm.any():continue
                    wrong=b[...,sm]<=0 if side=='stable' else b[...,sm]>0
                    records[label]['wrong_side'].setdefault(arm,{})[side]=paired_rate_difference(wrong[0],wrong[ai],groups[sm])
        result['conditions'][c]=records
        result['seed_fold_stability'][c]={}
        for ai,arm in enumerate(arms):
            result['seed_fold_stability'][c][arm]={
                'per_seed_mae':[point(y,p,ids)['design_mae'] for p in ho[ai,:,ci]],
                'per_seed_fold_delta_vs_one':[[point(y[folds==f],p[folds==f],ids[folds==f])['design_mae']-
                    point(y[folds==f],base[folds==f],ids[folds==f])['design_mae'] for f in range(5)]
                    for p,base in zip(ho[ai,:,ci],ho[0,:,ci])]}
        result['acceptance'][c]={}
        for arm in selected:
            delta=lambda label,key:records[label]['differences'][arm]['one'][key]['design_bootstrap_95'][1]
            upper=lambda label,side:records[label]['wrong_side'][arm][side]['design_bootstrap_95'][1]
            checks={'global_gain':delta('all','design_mae')<0,'stable_last1_noninferior':delta('stable0.01','design_mae')<=0,
                    'stable_sign_noninferior':upper('stable0.01','stable')<=0,
                    'unstable_sign_noninferior':upper('unstable0.01','unstable')<=0,'far_under_noninferior':delta('far','under_0.05')<=0}
            if '1_' in c:checks['matched_control_gain']=records['all']['differences'][arm]['sham_'+arm]['design_mae']['design_bootstrap_95'][1]<0
            point_checks={
                'global_gain':records['all']['differences'][arm]['one']['design_mae']['candidate_minus_reference']<0,
                'stable_last1_nonworsening':records['stable0.01']['differences'][arm]['one']['design_mae']['candidate_minus_reference']<=0,
                'stable_sign_nonworsening':records['stable0.01']['wrong_side'][arm]['stable']['candidate_minus_reference']<=0,
                'unstable_sign_nonworsening':records['unstable0.01']['wrong_side'][arm]['unstable']['candidate_minus_reference']<=0,
                'far_under_nonworsening':records['far']['differences'][arm]['one']['under_0.05']['candidate_minus_reference']<=0}
            result['acceptance'][c][arm]={'checks':checks,'all_pass':all(checks.values()),
                                        'point_checks':point_checks,'all_point_pass':all(point_checks.values())}
        print(c,'MAE',{a:records['all']['arms'][a]['heldout']['over']['design_mae'] for a in arms},
              'accepted',{a:v['all_pass'] for a,v in result['acceptance'][c].items()},flush=True)
    result['qualified_finalists']=[a for a in selected if selection['qualified'] and all(result['acceptance'][c][a]['all_pass'] for c in names)]
    save_json(OUT/'confirmation_results.json',result)
    print('Strict zero-deterioration acceptance',result['qualified_finalists'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
