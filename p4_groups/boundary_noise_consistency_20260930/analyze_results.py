"""Separate prediction stability from boundary accuracy and frozen-reference guardrails."""
from experiment_helpers import *
from threadpoolctl import threadpool_limits


def cached_predictions():
    train,e,y,ids,rows,names,folds,layout,pairs=load_data();data={};fits=[]
    for suffix in ('supervised_global','localized'):
        with np.load(OUT/f'predictions_{suffix}.npz') as z:
            for k,v in [('y',y),('ids',ids),('source_indices',rows),('row_fold',folds),('conditions',names)]:np.testing.assert_array_equal(z[k],v)
            for i,a in enumerate(z['arms'].tolist()):data[a]=(z['training_predictions'][i],z['heldout_predictions'][i])
        fits+=json.loads((OUT/f'fits_{suffix}.json').read_text())['fits']
    parent=[]
    for path in (base.PRIOR/'predictions.npz',base.VERIFY/'predictions.npz',BASE/'confirmation_predictions.npz'):
        with np.load(path) as z:
            for k,v in [('y',y),('ids',ids),('source_indices',rows),('row_fold',folds)]:np.testing.assert_array_equal(z[k],v)
            parent.append((z['training_predictions'][:2],z['heldout_predictions'][:2],z['conditions'].tolist()))
    assert sum((p[2] for p in parent),[])==names
    for i,a in enumerate(('one','PCS')):
        data[a]=(np.concatenate([p[0][i] for p in parent],axis=2),np.concatenate([p[1][i] for p in parent],axis=1))
    data['blend']=tuple(.5*(a+b) for a,b in zip(data['one'],data['PCS']))
    return data,fits,y,ids,rows,names,folds


def analyze(data,y,ids,names,folds):
    result={'conditions':{},'seed_fold':{}};masks=slices(y)
    for ci,c in enumerate(names):
        record={}
        for label,m in masks.items():
            target,groups=y[m],ids[m];record[label]={'clips':int(m.sum()),'designs':int(len(np.unique(groups))),'arms':{},'differences':{},'wrong_side':{}}
            for a,(tr,ho) in data.items():
                t,h=tr[:,:,ci,:][...,m],ho[:,ci,:][...,m]
                record[label]['arms'][a]={'training':point(target,t,groups),'heldout':point(target,h,groups),
                    'held_minus_train':paired_metrics(target,t,h,groups),
                    'heldout_curves':{k:float(v.mean()) for k,v in metrics.per_clip(target,h).items()},
                    'per_seed_heldout':[point(target,p,groups) for p in h],
                    'per_seed_signed_p99':np.quantile((h-target).reshape(3,-1),.99,axis=1).tolist()}
            for a in ('supervised','global','localized'):
                refs=[r for r in ('supervised','global','one','PCS','blend') if r!=a and (a=='localized' or r!='global')]
                h=data[a][1][:,ci,:][...,m]
                record[label]['differences'][a]={r:paired_metrics(target,data[r][1][:,ci,:][...,m],h,groups) for r in refs}
                for side,sm in [('stable',target>0),('unstable',target<0)]:
                    if not sm.any():continue
                    def wrong(p):return p[...,sm]<=0 if side=='stable' else p[...,sm]>0
                    record[label]['wrong_side'].setdefault(a,{})[side]={r:paired_rate_difference(
                        wrong(data[r][1][:,ci,:][...,m]),wrong(h),groups[sm]) for r in refs}
        result['conditions'][c]=record;result['seed_fold'][c]={}
        for a,(_,ho) in data.items():
            result['seed_fold'][c][a]=[[point(y[folds==f],p[folds==f],ids[folds==f])['design_mae']-
                point(y[folds==f],basep[folds==f],ids[folds==f])['design_mae'] for f in range(5)]
                for p,basep in zip(ho[:,ci],data['supervised'][1][:,ci])]
        print(c,'global/stable1 MAE',{a:[round(record[s]['arms'][a]['heldout']['design_mae'],6) for s in ('all','stable0.01')] for a in data},flush=True)
    return result


def stability(data,y,names):
    result={};m=(abs(y)<.01)&(y!=0)
    for kind in ('iid1','correlated1','iid5','correlated5'):
        ix=[i for i,c in enumerate(names) if c.startswith(kind+'_')]
        assert len(ix)==2
        result[kind]={}
        for a,(_,ho) in data.items():
            p=ho[:,ix,:][:,:,m];error=p.mean(axis=1)-y[m]
            result[kind][a]={'squared_two_draw_mean_error':float((error**2).mean()),
                'two_draw_variance':float(p.var(axis=1).mean()),'sign_disagreement':float(((p[:,0]<=0)!=(p[:,1]<=0)).mean())}
    return result


def acceptance(records,converged):
    out={}
    for c,record in records.items():
        out[c]={}
        for ref in ('supervised','one','blend'):
            specifications=[('stable_mae','stable0.01','design_mae'),('unstable_mae','unstable0.01','design_mae'),
                            ('global_mae','all','design_mae'),('far_under05','far','under_0.05')]
            checks={};points={}
            for key,label,metric in specifications:
                value=record[label]['differences']['localized'][ref][metric]
                strict=ref=='supervised' and key=='stable_mae' and c.startswith(('iid1_','correlated1_'))
                upper=value['design_bootstrap_95'][1];delta=value['candidate_minus_reference']
                checks[key]=upper<0 if strict else upper<=0;points[key]=delta<0 if strict else delta<=0
            for side,label in [('stable','stable0.01'),('unstable','unstable0.01')]:
                value=record[label]['wrong_side']['localized'][side][ref]
                checks[side+'_sign']=value['design_bootstrap_95'][1]<=0;points[side+'_sign']=value['candidate_minus_reference']<=0
            out[c][ref]={'checks':checks,'all_pass':all(checks.values()) and converged,
                        'point_checks':points,'all_point_pass':all(points.values())}
    return out


def main():
    data,fits,y,ids,rows,names,folds=cached_predictions();cached=analyze(data,y,ids,names,folds)
    cached.update(fits=fits,all_converged=all(f['success'] for f in fits),
                  hashes={'analysis':sha256(Path(__file__)),'protocol':sha256(OUT/'protocol.json')})
    save_json(OUT/'cached_results.json',cached)
    with np.load(OUT/'fresh_predictions.npz') as z:
        for k,v in [('y',y),('ids',ids),('source_indices',rows),('row_fold',folds)]:np.testing.assert_array_equal(z[k],v)
        fresh={a:(z['training_predictions'][i],z['heldout_predictions'][i]) for i,a in enumerate(z['arms'].tolist())}
        names=z['conditions'].tolist()
    result=analyze(fresh,y,ids,names,folds);result['stability']=stability(fresh,y,names)
    result['all_converged']=cached['all_converged'];result['acceptance']=acceptance(result['conditions'],result['all_converged'])
    result['qualified_against_matched_control']=all(v['supervised']['all_pass'] for v in result['acceptance'].values())
    result['qualified_replacement']=result['qualified_against_matched_control'] and all(v[r]['all_pass'] for v in result['acceptance'].values() for r in ('one','blend'))
    result['hashes']={'analysis':sha256(Path(__file__)),'protocol':sha256(OUT/'protocol.json'),
                     'predictions':sha256(OUT/'fresh_predictions.npz'),'fits_supervised_global':sha256(OUT/'fits_supervised_global.json'),
                     'fits_localized':sha256(OUT/'fits_localized.json')}
    save_json(OUT/'fresh_results.json',result)
    print('Localized qualified versus matched control:',result['qualified_against_matched_control'],
          '; qualified replacement:',result['qualified_replacement'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
