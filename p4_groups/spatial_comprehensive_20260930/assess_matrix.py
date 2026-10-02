"""Exploratory ingredient comparisons; lock finalists before confirmation outcomes."""
from common import *
from threadpoolctl import threadpool_limits


def point(y,p,ids):
    error=p.reshape(-1,len(y))-y; mae=abs(error).mean(axis=0)
    out={'design_mae':float(np.mean([mae[ids==d].mean() for d in np.unique(ids)])),
         'bias':float(error.mean()),'over01':float((error>.01).mean()),'over03':float((error>.03).mean()),
         'over05':float((error>.05).mean()),'under05':float((error<-.05).mean())}
    for side,m in [('stable',y>0),('unstable',y<0)]:
        out['wrong_'+side]=float((p[...,m]<=0 if side=='stable' else p[...,m]>0).mean()) if m.any() else None
    return out


def per_clip(y,p):
    error=p.reshape(-1,len(y))-y
    metrics={'clip_mae':abs(error).mean(axis=0),'mean_signed_error':error.mean(axis=0)}
    for delta in (0.,.005,.01,.02,.03,.05):
        metrics[f'over_{delta}']=(error>delta).mean(axis=0)
        metrics[f'under_{delta}']=(error<-delta).mean(axis=0)
    return metrics


def paired_metrics(y,control,candidate,ids):
    a,b=per_clip(y,control),per_clip(y,candidate)
    unique,inverse,counts=np.unique(ids,return_inverse=True,return_counts=True)
    weights=np.random.default_rng(42).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)
    keys=list(a);totals=np.stack([np.bincount(inverse,weights=b[k]-a[k]) for k in keys],axis=1)
    macro=totals[:,0]/counts
    samples=(weights@totals)/(weights@counts)[:,None]
    result={k:{'candidate_minus_reference':float((b[k]-a[k]).mean()),
               'design_bootstrap_95':np.quantile(samples[:,i],[.025,.975]).tolist()} for i,k in enumerate(keys)}
    result['design_mae']={'candidate_minus_reference':float(macro.mean()),
                         'design_bootstrap_95':np.quantile(weights@macro/len(unique),[.025,.975]).tolist()}
    return result


def converged_models(arm):
    for seed in (42,43,44):
        for fold in range(5):
            with np.load(model_path(arm,seed,fold)) as z:
                if not json.loads(str(z['metadata']))['success']:return False
    return True


def load_predictions():
    data={}
    for file in ('predictions_F_P_C_S.npz','predictions_PC_PS_CS.npz'):
        with np.load(OUT/file) as z:
            for i,a in enumerate(z['arms'].tolist()):data[a]=(z['training_predictions'][i],z['heldout_predictions'][i])
    with np.load(PRIOR/'predictions.npz') as a,np.load(VERIFY/'predictions.npz') as b:
        for i,name in [(0,'one'),(1,'PCS')]:
            data[name]=(np.concatenate([a['training_predictions'][i],b['training_predictions'][i]],axis=2),
                        np.concatenate([a['heldout_predictions'][i],b['heldout_predictions'][i]],axis=1))
    data['blend']=tuple(.5*(a+b) for a,b in zip(data['one'],data['PCS']))
    return data


def main():
    _,_,y,ids,rows,names,folds,_,_=load_data();data=load_predictions()
    masks={'all':np.ones(len(y),bool),'far':y>=.15}
    for eps in (.01,.02,.05):
        masks[f'stable{eps}']=(y>0)&(y<eps);masks[f'unstable{eps}']=(y<0)&(y>-eps)
    results={'conditions':{},'stability':{},'model_hashes':{},'protocol_hash':sha256(OUT/'protocol.json')}
    for a,(tr,ho) in data.items():
        results['model_hashes'][a]={f'{s}_{f}':sha256(model_path(a,s,f)) for s in (42,43,44) for f in range(5)} if a!='blend' else {}
        for ci,c in enumerate(names):
            result={}
            for label,m in masks.items():
                train=point(y[m],tr[:,:,ci,:][...,m],ids[m]);held=point(y[m],ho[:,ci,:][...,m],ids[m])
                result[label]={'training':train,'heldout':held,'held_minus_train':{k:held[k]-train[k] for k in held if held[k] is not None}}
                target,groups=y[m],ids[m];a_tr,a_ho=tr[:,:,ci,:][...,m],ho[:,ci,:][...,m]
                result[label].update(
                    held_minus_train_ci=paired_metrics(target,a_tr,a_ho,groups),
                    paired_vs_one=paired_metrics(target,data['one'][1][:,ci,:][...,m],a_ho,groups),
                    per_seed_heldout=[point(target,p,groups) for p in a_ho],
                    per_seed_signed_p99=np.quantile((a_ho-target).reshape(3,-1),.99,axis=1).tolist())
            results['conditions'].setdefault(c,{})[a]=result
            results['stability'].setdefault(c,{})[a]=[float(np.mean([abs(p[ids==d]-y[ids==d]).mean() for d in np.unique(ids)])) for p in ho[:,ci]]
            results['stability'][c][a]={'per_seed':results['stability'][c][a],
                'per_seed_outer_fold':[[point(y[folds==f],p[folds==f],ids[folds==f])['design_mae'] for f in range(5)] for p in ho[:,ci]]}
    latest=names[-6:];ranks=[]
    for arm in MASKS[:-1]:
        def violations(conditions):
            total=0
            for c in conditions:
                q,b=results['conditions'][c][arm],results['conditions'][c]['one']
                for label,k in [('stable0.01','design_mae'),('stable0.01','wrong_stable'),('unstable0.01','wrong_unstable'),('far','under05')]:
                    total+=q[label]['heldout'][k]>b[label]['heldout'][k]
            return int(total)
        global_mae=float(np.mean([results['conditions'][c][arm]['all']['heldout']['design_mae'] for c in latest]))
        base=float(np.mean([results['conditions'][c]['one']['all']['heldout']['design_mae'] for c in latest]))
        converged=converged_models(arm)
        ranks.append({'arm':arm,'eligible':bool(global_mae<=base and converged),'primary_violations':violations(latest[:4]),
                      'stress_violations':violations(latest[4:]),'mean_global_mae':global_mae,'converged':converged})
    eligible=sorted([r for r in ranks if r['eligible']],key=lambda r:(r['primary_violations'],r['stress_violations'],r['mean_global_mae'],r['arm']))
    chosen=eligible[:2]
    if not chosen:
        available=[r for r in ranks if r['converged']]
        if not available:raise RuntimeError('No converged reduced variant available for diagnostic confirmation')
        chosen=[min(available,key=lambda r:(r['mean_global_mae'],r['arm']))]
    selection={'selected':[r['arm'] for r in chosen],'qualified':bool(eligible),'ranking':ranks,
               'protocol_hash':results['protocol_hash'],'assessment_source':sha256(Path(__file__)),
               'model_hashes':{r['arm']:results['model_hashes'][r['arm']] for r in chosen}}
    save_json(OUT/'matrix_results.json',results)
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==selection,'locked finalist selection changed'
    else:save_json(path,selection)
    print('Ingredient rankings',json.dumps(ranks),flush=True);print('Locked finalists',selection['selected'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
