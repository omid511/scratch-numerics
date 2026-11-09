"""W1: objective/capacity controls; immutable continuous margin fields."""
from gap_common import *
import copy
import torch.nn.functional as F


def warning_score(head, control, q):
    x=R.features(control,q,'rank_products')
    return -control[:,0]/.01+neural_predict(head['model'],head['scaling'],x)


def train(x,control,y,ids,views,kind,budget,seed,validation=None,epochs=300):
    scaling=scale_fit(x); z=tensor(scaled(scaling,x)); prior=tensor(-control[:,0]/.01)
    model=net(x.shape[1],kind!='affine',seed)
    stable=y>0; near=(y<0)&(y>-.01)
    if not stable.any() or not near.any(): raise ValueError('empty warning cohort')
    nearw=tensor(weights(ids[near])); classw=np.zeros(len(y))
    for mask in (stable,~stable): classw[mask]=.5*weights(ids[mask])
    classw=tensor(classw); target=tensor((~stable).astype(float))
    thresholds=nn.Parameter(tensor([float(np.quantile(-control[(views==v)&stable,0]/.01,1-budget)) for v in range(4)]))
    optimizer=torch.optim.Adam(list(model.parameters())+[thresholds],lr=.003)
    dual=torch.ones(4)
    masks=[(views==v)&stable for v in range(4)]
    groupweights=[tensor(weights(ids[m])) for m in masks]
    best=None; history=[]; feasible=0; start=time.perf_counter()
    for epoch in range(1,epochs+1):
        optimizer.zero_grad(); s=prior+model(z)[:,0]
        if kind=='classification':
            loss=(F.binary_cross_entropy_with_logits(s,target,reduction='none')*classw).sum()
            violation=torch.zeros(4)
        else:
            missed=(F.softplus(thresholds[tensor(views).long()][near]-s[near])*nearw).sum()
            rates=torch.stack([(torch.sigmoid(s[m]-thresholds[v])*w).sum() for v,(m,w) in enumerate(zip(masks,groupweights))])
            violation=rates-budget
            loss=missed+(dual*violation).sum()
        loss=loss+.005*sum(v.square().sum() for v in model.parameters())
        if not torch.isfinite(loss): raise RuntimeError('nonfinite warning loss')
        loss.backward(); optimizer.step()
        if kind!='classification': dual=(dual+.5*violation.detach()).clamp(min=0)
        if epoch%10==0 or epoch==epochs:
            if validation is None: continue
            vx,vc,vy,vi,vviews=validation
            with torch.no_grad():
                train_score=(prior+model(z)[:,0]).numpy()
                val_score=-vc[:,0]/.01+model(tensor(scaled(scaling,vx)))[:,0].numpy()
            threshold=C.threshold(train_score,y,budget)
            flags=-val_score<=threshold
            rates=[float(flags[(vviews==v)&(vy>0)].mean()) for v in range(4)]
            miss=float((~flags[(vy<0)&(vy>-.01)]).mean())
            violation_actual=max(max(rates)-budget,0.)
            bce=float((F.binary_cross_entropy_with_logits(tensor(val_score),tensor((vy<=0).astype(float)),reduction='none')*tensor(weights(vi))).sum())
            feasible+=int(violation_actual==0)
            key=(bce,) if kind=='classification' else (violation_actual,miss,bce)
            record=dict(epoch=epoch,loss=float(loss.detach()),inner_miss=miss,inner_view_false_rates=rates,
                        inner_indicator_feasible=violation_actual==0,dual=dual.tolist())
            history.append(record)
            if best is None or key<best[0]: best=(key,copy.deepcopy(model),record)
    head=dict(model=model,scaling=scaling)
    meta=dict(kind=kind,budget=budget,epochs=epochs,seconds=time.perf_counter()-start,
              training_thresholds=thresholds.detach().tolist(),dual=dual.tolist(),history=history,
              inner_feasible_checkpoints=feasible)
    if validation is not None:
        head['model']=best[1]; meta['selected']=best[2]
    with torch.no_grad():
        final_score=prior+head['model'](z)[:,0]
        meta['training_smooth_false_rates']=[float((torch.sigmoid(final_score[m]-thresholds[v])*w).sum()) for v,(m,w) in enumerate(zip(masks,groupweights))]
    return head,meta


def run():
    d=E.setup(); fp=fingerprint('warning_screen.py')
    for screen in PROTOCOL['screens']:
        fold,seed=screen['fold'],screen['seed']; proper,cal,held=C.split(d['ids'],d['folds'],fold)
        tx,cx,hx,*_=E.previous_fold(fold,proper,cal,held)
        tq,cq,hq=[base[:,rows].reshape(-1,174) for base,rows in ((d['train'],proper),(d['train'],cal),(d['evaluation'],held))]
        tc,cc,hc=[a.reshape(-1,7) for a in (tx,cx,hx)]
        x=R.features(tc,tq,'rank_products'); y=np.tile(d['y'][proper],4); ids=np.tile(d['ids'][proper],4); views=np.repeat(np.arange(4),len(proper))
        fit,query,*_=inner_members(fold,proper)
        inner=np.isin(ids,d['ids'][query]); assert not set(ids[inner])&set(ids[~inner])
        validation=(x[inner],tc[inner],y[inner],ids[inner],views[inner])
        with np.load(R.OUT/f'fold{fold}.npz') as z:
            old=C.cloudpickle.loads(z['heads'].tobytes())['rank_products']
            oldt=z['thresholds'][R.ARMS.index('rank_products')].copy()
        arms=['blend_frozen','rank_products']; predictions=[d['baseline'][:,held].copy() for _ in arms]
        predictions[1][...,3]=R.score(old,hc,hq,'rank_products').reshape(len(d['names']),len(held))
        thresholds=[d['baseline_thresholds'][fold],oldt]; heads={}; records=[]
        cy=np.tile(d['y'][cal],4)
        for budget in (.01,.05):
            for kind in ('affine','nonlinear','classification'):
                arm=f'{kind}_b{int(budget*100)}'
                _,inner_meta=train(x[~inner],tc[~inner],y[~inner],ids[~inner],views[~inner],kind,budget,seed,validation)
                epoch=inner_meta['selected']['epoch']
                head,meta=train(x,tc,y,ids,views,kind,budget,seed,epochs=epoch)
                cs=warning_score(head,cc,cq); hs=warning_score(head,hc,hq)
                levels=np.array([C.threshold(cs,cy,b) for b in C.BUDGETS])
                rates=[float((-cs[cy>0]<=t).mean()) for t in levels]
                assert np.all(np.array(rates)<=C.BUDGETS)
                hp=d['baseline'][:,held].copy(); hp[...,3]=hs.reshape(len(d['names']),len(held))
                np.testing.assert_array_equal(hp[...,[0,1,2,4,5]],d['baseline'][:,held][...,[0,1,2,4,5]])
                arms.append(arm); predictions.append(hp); thresholds.append(levels); heads[arm]=head
                records.append(dict(arm=arm,inner=inner_meta,final=meta,calibration_false_rates=rates))
                print('W1_FIT_COMPLETE',fold,arm,'selected_epoch',epoch,'inner_feasible',inner_meta['inner_feasible_checkpoints'],flush=True)
        predictions=np.array(predictions); thresholds=np.array(thresholds)
        assert np.isfinite(predictions).all()
        C.save_npz(OUT/f'w1_fold{fold}.npz',predictions=predictions,thresholds=thresholds,arms=arms,held_rows=held,
                   proper_rows=proper,calibration_rows=cal,heads=np.frombuffer(C.cloudpickle.dumps(heads),dtype=np.uint8),
                   fingerprint=json.dumps(fp))
        C.save_json(OUT/f'w1_fits_fold{fold}.json',records)
        evaluate(OUT/f'w1_results_fold{fold}.json',d,held,arms,predictions,thresholds,screen)
    assert fingerprint('warning_screen.py')==fp
    print('W1_SCREEN_COMPLETE',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): run()
