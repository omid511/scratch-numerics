"""Failure associations, repeated-draw variability and ingredient interactions."""
from common import *
from assess_matrix import load_predictions,point
from threadpoolctl import threadpool_limits


def distribution(x):
    x=np.asarray(x);v=x[np.isfinite(x)]
    return {'finite':int(len(v)),'missing':int(x.size-len(v)),
            'mean':float(v.mean()) if len(v) else None,
            'p10_median_p90':np.quantile(v,[.1,.5,.9]).tolist() if len(v) else None}


def main():
    _,_,y,ids,rows,names,folds,_,_=load_data();preds=load_predictions()
    with np.load(OUT/'diagnostics.npz') as z:
        np.testing.assert_array_equal(z['source_indices'],rows)
        d,fields,dnames=z['diagnostics'],z['fields'].tolist(),z['conditions'].tolist()
    result={'conditions':{},'repeated_noise':{},'factorial_interactions':{},
            'hashes':{'diagnostics':sha256(OUT/'diagnostics.npz'),'analysis':sha256(Path(__file__))}}
    for ci,c in enumerate(dnames):
        row=d[ci];boundary=(abs(y)<.01)&(y!=0)
        cohorts={'stable_last1':(y>0)&(y<.01),'unstable_last1':(y<0)&(y>-.01),
                 'far_stable':y>=.15,'boundary_critical_recovered':boundary&(row[:,11]==1),
                 'boundary_critical_missing':boundary&(row[:,11]==0),
                 'boundary_lead_is_critical':boundary&(row[:,13]==1),
                 'boundary_lead_not_critical':boundary&(row[:,13]==0)}
        cut=np.quantile(row[boundary,2],[.25,.75]);index=names.index(c)
        cohorts.update(boundary_second_available=boundary&(row[:,15]==1),
                       boundary_second_missing=boundary&(row[:,15]==0),
                       boundary_low_fit_residual=boundary&(row[:,2]<=cut[0]),
                       boundary_high_fit_residual=boundary&(row[:,2]>=cut[1]),
                       baseline_severe_stable=(y>0)&(y<.05)&(((preds['one'][1][:,index,:]-y)>.03).sum(axis=0)>=2))
        records={}
        for label,m in cohorts.items():
            if not m.any():records[label]={'clips':0};continue
            records[label]={'clips':int(m.sum()),'designs':int(len(np.unique(ids[m]))),
                            'diagnostics':{f:distribution(row[m,i]) for i,f in enumerate(fields)},'models':{}}
            records[label]['source_indices']=rows[m].tolist()
            for arm,(tr,ho) in preds.items():
                index=names.index(c)
                records[label]['models'][arm]={'training':point(y[m],tr[:,:,index,:][...,m],ids[m]),
                                               'heldout':point(y[m],ho[:,index,:][...,m],ids[m])}
        result['conditions'][c]=records
    for label,pair in [('iid1',['iid1_fresh939','iid1_fresh941']),('iid5',['iid5_fresh939','iid5_fresh941']),
                       ('correlated1',['correlated_fresh939_940','correlated_fresh941_942'])]:
        ix=[names.index(c) for c in pair];result['repeated_noise'][label]={}
        m=(abs(y)<.01)&(y!=0);d1,d2=d[[dnames.index(c) for c in pair]]
        result['repeated_noise'][label]['modal_identity']={}
        for field in (7,8):
            matched=m&(d1[:,field]>=0)&(d2[:,field]>=0)
            result['repeated_noise'][label]['modal_identity'][fields[field]]={
                'matched_both':int(matched.sum()),'unresolved':int(m.sum()-matched.sum()),
                'switch_rate_matched':float((d1[matched,field]!=d2[matched,field]).mean()) if matched.any() else None}
        for arm,(_,ho) in preds.items():
            p=ho[:,ix,:];m=(abs(y)<.01)&(y!=0)
            systematic=(p.mean(axis=1)-y)[:,m]
            result['repeated_noise'][label][arm]={
                'boundary_mean_signed_error':float(systematic.mean()),
                'boundary_mean_squared_bias_per_seed_clip':float((systematic**2).mean()),
                'boundary_noise_variance_per_seed_clip':float(p[:,:,m].var(axis=1).mean()),
                'boundary_sign_disagreement':float(((p[:,0,m]<=0)!=(p[:,1,m]<=0)).mean())}
    matrix=json.loads((OUT/'matrix_results.json').read_text())
    for c,arms in matrix['conditions'].items():
        result['factorial_interactions'][c]={}
        for label in ('all','stable0.01','unstable0.01','far'):
            losses={a:arms[a][label]['heldout']['design_mae'] for a in MASKS}
            effects={}
            for block in 'PCS':
                differences=[]
                for a in MASKS:
                    active=set() if a=='F' else set(a)
                    if block not in active:
                        target=''.join(b for b in 'PCS' if b in active|{block})
                        differences.append(losses[target]-losses[a])
                effects[block]=float(np.mean(differences))
            result['factorial_interactions'][c][label]={'block_average_mae_effect':effects,
                'pair_PC':losses['PC']-losses['P']-losses['C']+losses['F'],
                'pair_PS':losses['PS']-losses['P']-losses['S']+losses['F'],
                'pair_CS':losses['CS']-losses['C']-losses['S']+losses['F'],
                'triple':losses['PCS']-losses['PC']-losses['PS']-losses['CS']+losses['P']+losses['C']+losses['S']-losses['F']}
    save_json(OUT/'diagnostic_results.json',result)
    print('Saved recovery cohorts, training/OOF comparisons, repeated-noise variance and factorial interactions',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
