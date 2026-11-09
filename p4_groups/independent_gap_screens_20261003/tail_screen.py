"""S1: warm-start nonlinear blended-output tail fitting, honest inner designs."""
from gap_common import *
import copy


def cvar90(values,w):
    order=torch.argsort(values,descending=True,stable=True)
    vw,ww=values[order],w[order]
    before=torch.cumsum(ww,0)-ww
    mass=torch.minimum(ww,(.1-before).clamp(min=0))
    return (vw*mass).sum()/.1


def diagnostics(pred,y,ids,views):
    e=pred-y; stable=(y>0)&(y<.01); stable5=(y>0)&(y<.05)
    unstable=(y<0)&(y>-.01); far=y>=.15; clean=views==0
    return dict(stable_over03=float((e[stable]>.03).mean()),stable5_over03=float((e[stable5]>.03).mean()),
                unstable_over03=float((e[unstable]>.03).mean()),far_under05=float((e[far]<-.05).mean()),
                clean_mae=float(np.dot(weights(ids[clean]),abs(e[clean]))),
                stable_p99=float(np.quantile(e[stable],.99)),mae=float(np.dot(weights(ids),abs(e))))


def train(full,one,x,y,ids,views,kind,validation=None,epochs=300):
    model,scaling=warm_net(full); z=tensor(scaled(scaling,x)); target=tensor(y)
    frozen=tensor(one(x[:,:107])); w=tensor(weights(ids))
    initial=[p.detach().clone() for p in model.parameters()]
    blended0=.5*(full(x)+one(x[:,:107]))
    np.testing.assert_allclose(.5*(neural_predict(model,scaling,x)+frozen.numpy()),blended0,atol=1e-12,rtol=1e-12)
    groups=[]; support=[]
    for v in range(4):
        for label,mask in [('stable_last1',(y>0)&(y<.01)),('stable_1to5',(y>=.01)&(y<.05)),
                           ('unstable_last1',(y<0)&(y>-.01)),('unstable_1to5',(y<=-.01)&(y>-.05))]:
            mask=mask&(views==v); count=len(np.unique(ids[mask]))
            if count<3: raise ValueError(f'insufficient tail group support: {v}/{label}/{count}')
            groups.append((mask,tensor(weights(ids[mask])))); support.append(dict(view=v,regime=label,designs=count,rows=int(mask.sum())))
    optimizer=torch.optim.Adam(model.parameters(),lr=.00003)
    history=[]; best=None; start=time.perf_counter()
    if validation is not None:
        vx,vy,vi,vviews=validation
        base=diagnostics(.5*(full(vx)+one(vx[:,:107])),vy,vi,vviews)
    for epoch in range(1,epochs+1):
        optimizer.zero_grad(); blended=.5*(model(z)[:,0]+frozen); error=(blended-target)/.01
        loss=(w*error.square()).sum()
        tails=torch.stack([cvar90(error[m].clamp(min=0),gw) for m,gw in groups])
        if kind=='average_tail': loss=loss+.25*tails.mean()
        elif kind=='worst_tail': loss=loss+.25*tails.max()
        loss=loss+.005*sum(p.square().sum() for p in model.parameters())
        loss=loss+.05*sum((p-q).square().sum() for p,q in zip(model.parameters(),initial))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite nonlinear tail objective')
        loss.backward(); optimizer.step()
        if epoch%10==0 or epoch==epochs:
            if validation is None: continue
            pred=.5*(neural_predict(model,scaling,vx)+one(vx[:,:107]))
            diag=diagnostics(pred,vy,vi,vviews)
            violations=sum(diag[k]>base[k]+1e-12 for k in ('unstable_over03','far_under05','clean_mae'))
            key=(diag['stable_over03'],diag['stable5_over03'],violations,diag['stable_p99'],diag['mae'])
            record=dict(epoch=epoch,loss=float(loss.detach()),guardrail_violations=violations,**diag)
            history.append(record)
            if best is None or key<best[0]: best=(key,copy.deepcopy(model),record)
    meta=dict(kind=kind,epochs=epochs,seconds=time.perf_counter()-start,groups=support,history=history)
    if validation is not None:
        model=best[1]; meta.update(selected=best[2],inner_frozen=base)
    return dict(model=model,scaling=scaling),meta


def run():
    d=E.setup(); fp=fingerprint('tail_screen.py')
    for screen in PROTOCOL['screens']:
        fold,seed=screen['fold'],screen['seed']; torch.manual_seed(seed)
        proper,cal,held=C.split(d['ids'],d['folds'],fold)
        E.previous_fold(fold,proper,cal,held)
        fit,query,ione,ifull=inner_members(fold,proper)
        one,full=C.reference_predictors(42,fold,proper,cal,held)
        ix=d['train'][:,fit].reshape(-1,174); iy=np.tile(d['y'][fit],4); ii=np.tile(d['ids'][fit],4); iv=np.repeat(np.arange(4),len(fit))
        vx=d['train'][:,query].reshape(-1,174); vy=np.tile(d['y'][query],4); vi=np.tile(d['ids'][query],4); vv=np.repeat(np.arange(4),len(query))
        assert not set(ii)&set(vi)
        x=d['train'][:,proper].reshape(-1,174); y=np.tile(d['y'][proper],4); ids=np.tile(d['ids'][proper],4); views=np.repeat(np.arange(4),len(proper))
        cx=d['train'][:,cal].reshape(-1,174); cy=np.tile(d['y'][cal],4); cids=np.tile(d['ids'][cal],4)
        hx=d['evaluation'][:,held].reshape(-1,174)
        arms=['blend_frozen']; predictions=[d['baseline'][:,held]]; thresholds=[d['baseline_thresholds'][fold]]; records=[]; heads={}
        for kind in ('mse','average_tail','worst_tail'):
            _,inner=train(ifull,ione,ix,iy,ii,iv,kind,(vx,vy,vi,vv))
            epochs=inner['selected']['epoch']
            head,meta=train(full,one,x,y,ids,views,kind,epochs=epochs)
            cp=.5*(one(cx[:,:107])+neural_predict(head['model'],head['scaling'],cx))
            hp=.5*(one(hx[:,:107])+neural_predict(head['model'],head['scaling'],hx))
            distribution,levels,calmeta=calibrated(cp,cy,cids,hp)
            arms.append(kind); predictions.append(distribution.reshape(len(d['names']),len(held),6)); thresholds.append(levels); heads[kind]=head
            records.append(dict(arm=kind,inner=inner,final=meta,calibration=calmeta))
            print('S1_FIT_COMPLETE',fold,kind,'selected_epoch',epochs,flush=True)
        predictions=np.array(predictions); thresholds=np.array(thresholds)
        assert np.isfinite(predictions).all()
        C.save_npz(OUT/f's1_fold{fold}.npz',predictions=predictions,thresholds=thresholds,arms=arms,held_rows=held,
                   proper_rows=proper,calibration_rows=cal,heads=np.frombuffer(C.cloudpickle.dumps(heads),dtype=np.uint8),fingerprint=json.dumps(fp))
        C.save_json(OUT/f's1_fits_fold{fold}.json',records)
        evaluate(OUT/f's1_results_fold{fold}.json',d,held,arms,predictions,thresholds,screen)
    assert fingerprint('tail_screen.py')==fp
    print('S1_SCREEN_COMPLETE',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): run()
