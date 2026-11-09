"""Continue two fixed candidates independently, preserving the original screen."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OLD=ROOT/'p4_groups/independent_gap_screens_20261003'
sys.path.insert(0,str(OLD))
import gap_common as G
import warning_screen as W
import noise_screen as N
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
C,E,R,np=G.C,G.E,G.R,G.np
P=json.loads((OUT/'protocol.json').read_text())
SEEDS=P['confirmation']['fold_seeds']
ARMS=['blend_frozen','rank_products','warning_nonlinear','unweighted_own','covariance_full']


def signature():
    return dict(parent=E.signature(),sources={str(p.relative_to(ROOT)):C.sha256(p) for p in
        (OUT/'protocol.json',Path(__file__),OLD/'gap_common.py',OLD/'warning_screen.py',OLD/'noise_screen.py')})


def extract_remaining(payload):
    did,rows,dts,names,layout,pairs,gains,fp=payload
    rawpath=ROOT/'p4_sensor_candidates'/f'{did}.npz'; identity=dict(fingerprint=fp,raw_hash=C.sha256(rawpath),sources=rows)
    path=OUT/f'covariance_features_{did}.npz'
    if path.exists():
        with np.load(path) as z: assert json.loads(str(z['identity']))==identity
        return did
    with np.load(rawpath) as z: raw,sigma,sources=z['raw'],z['noise_scale'],z['source_indices']
    ridge=N.B.PhysicsFeatureRidge(); features=np.empty((len(names),len(rows),174)); seconds=[]; available=[]
    for j,(source,dt) in enumerate(zip(rows,dts)):
        k=int(np.flatnonzero(sources==source)[0]); white,shared={},{}; covariance={}
        for ci,name in enumerate(names):
            start=G.time.perf_counter()
            signal,reported_dt,scales,rho=N.measured(raw[k],sigma[k],source,name,layout,gains,dt,white,shared)
            if name=='clean': modes=N.estimate_modes(signal,reported_dt,include_shapes=True)
            else:
                if rho not in covariance: covariance[rho]=N.covariance_from_quiet(source,len(layout),rho)[0]
                modes=N.weighted_modes(signal,reported_dt,covariance[rho]/scales[:,None]/scales[None,:])
            features[ci,j]=np.r_[N.B.base_feature_row(signal,reported_dt,modes,ridge),N.B.modal_spatial_row(modes,pairs),N.B.modal_spatial_addition_row(modes,pairs)]
            seconds.append(G.time.perf_counter()-start); available.append(bool(len(modes['alpha'])))
    assert np.isfinite(features).all()
    C.save_npz(path,features=features,source_indices=rows,seconds=seconds,available=available,identity=json.dumps(identity))
    print('CONFIRM_COVARIANCE_ACQUIRED',did,len(rows)*len(names),flush=True)
    return did


def covariance_data(d,fp):
    with np.load(OLD/'n1_features.npz') as z:
        assert json.loads(str(z['fingerprint']))==G.fingerprint('noise_screen.py')
        training=z['training'][N.KINDS.index('full')].copy()
        evaluation=np.full(d['evaluation'].shape,np.nan)
        evaluation[:,z['held_rows']]=z['evaluation'][N.KINDS.index('full')]
    gains=np.array(json.loads((ROOT/'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    remaining=np.isin(d['folds'],[2,3,4]); payloads=[]
    for did in np.unique(d['ids'][remaining]):
        rows=np.flatnonzero((d['ids']==did)&remaining)
        payloads.append((str(did),d['rows'][rows].tolist(),d['dt'][rows].tolist(),d['names'],d['layout'],d['pairs'],gains,fp))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for task in as_completed([pool.submit(extract_remaining,p) for p in payloads]): task.result()
    lookup={int(v):k for k,v in enumerate(d['rows'])}; timing=[]; availability=[]
    for did in np.unique(d['ids'][remaining]):
        with np.load(OUT/f'covariance_features_{did}.npz') as z:
            slots=np.array([lookup[int(v)] for v in z['source_indices']]); evaluation[:,slots]=z['features']
            timing.extend(z['seconds']); availability.extend(z['available'])
    assert np.isfinite(training).all() and np.isfinite(evaluation).all()
    C.save_json(OUT/'covariance_acquisition.json',dict(new_acquisitions=len(timing),observed_availability=float(np.mean(availability)),
        seconds_p95=float(np.quantile(timing,.95)),seconds_p99=float(np.quantile(timing,.99)),seconds_max=float(max(timing)),
        limitation='Four-worker synthetic quiet calibration identification+feature timings, not sensor-to-output latency.'))
    C.save_npz(OUT/'covariance_features.npz',training=training,evaluation=evaluation,fingerprint=json.dumps(fp))
    return training,evaluation


def warning_head(d,fold,proper,cal,held,seed,fp):
    if fold<2:
        with np.load(OLD/f'w1_fold{fold}.npz') as z:
            assert json.loads(str(z['fingerprint']))==G.fingerprint('warning_screen.py')
            for key,rows in [('proper_rows',proper),('calibration_rows',cal),('held_rows',held)]: np.testing.assert_array_equal(z[key],rows)
            head=C.cloudpickle.loads(z['heads'].tobytes())['classification_b5']
        record=next(v for v in json.loads((OLD/f'w1_fits_fold{fold}.json').read_text()) if v['arm']=='classification_b5')
        return head,dict(reused_original_screen=True,original=record)
    path=OUT/f'warning_fold{fold}.npz'
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['fingerprint']))==fp
            return C.cloudpickle.loads(z['head'].tobytes()),json.loads(str(z['metadata']))
    tx,_,_,*_=E.previous_fold(fold,proper,cal,held)
    q=d['train'][:,proper].reshape(-1,174); control=tx.reshape(-1,7); x=R.features(control,q,'rank_products')
    y=np.tile(d['y'][proper],4); ids=np.tile(d['ids'][proper],4); views=np.repeat(np.arange(4),len(proper))
    _,query,*_=G.inner_members(fold,proper); inner=np.isin(ids,d['ids'][query])
    assert not set(ids[inner])&set(ids[~inner])
    validation=(x[inner],control[inner],y[inner],ids[inner],views[inner])
    _,imeta=W.train(x[~inner],control[~inner],y[~inner],ids[~inner],views[~inner],'classification',.05,seed,validation)
    head,meta=W.train(x,control,y,ids,views,'classification',.05,seed,epochs=imeta['selected']['epoch'])
    record=dict(seed=seed,inner=imeta,final=meta)
    C.save_npz(path,head=np.frombuffer(C.cloudpickle.dumps(head),dtype=np.uint8),metadata=json.dumps(record),fingerprint=json.dumps(fp),proper_rows=proper)
    print('CONFIRM_WARNING_FIT',fold,seed,imeta['selected']['epoch'],flush=True)
    return head,record


def covariance_models(training,d,fold,proper,seed,fp):
    if fold<2:
        with np.load(OLD/f'n1_fold{fold}.npz') as z:
            assert json.loads(str(z['fingerprint']))==G.fingerprint('noise_screen.py')
            np.testing.assert_array_equal(z['proper_rows'],proper)
            models=C.cloudpickle.loads(z['models'].tobytes())['full']
        record=next(v for v in json.loads((OLD/f'n1_fits_fold{fold}.json').read_text()) if v['arm']=='full_own')
        return models,dict(reused_original_screen=True,original=record)
    x=training[:,proper].reshape(-1,174); y=np.tile(d['y'][proper],4); models=[]; records=[]
    for width in (107,174):
        path=OUT/f'covariance_{width}_fold{fold}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint']))==fp
                model=C.cloudpickle.loads(z['predictor'].tobytes()); meta=json.loads(str(z['metadata']))
        else:
            model,meta=C.fit_network(x[:,:width],y,seed=seed,alpha=1.,max_iter=20000,max_fun=400000)
            C.save_npz(path,predictor=np.frombuffer(C.cloudpickle.dumps(model),dtype=np.uint8),metadata=json.dumps(meta),fingerprint=json.dumps(fp),proper_rows=proper)
        if not meta['success']: raise RuntimeError(f'nonconverged covariance confirmation {fold}/{width}: {meta}')
        models.append(model); records.append(dict(width=width,**meta))
        print('CONFIRM_COVARIANCE_FIT',fold,width,json.dumps(meta),flush=True)
    return models,dict(seed=seed,fits=records)


def run():
    d=E.setup(); fp=signature(); warning=[]
    # Warning continuation can finish before the orthogonal extraction/refits.
    for fold,seed in enumerate(SEEDS):
        proper,cal,held=C.split(d['ids'],d['folds'],fold)
        E.previous_fold(fold,proper,cal,held)
        head,record=warning_head(d,fold,proper,cal,held,seed,fp); warning.append((head,record))
    training,evaluation=covariance_data(d,fp)
    predictions=np.repeat(d['baseline'][None],len(ARMS),axis=0); thresholds=np.full((len(ARMS),5,len(C.BUDGETS)),np.nan); fits=[]
    for fold,seed in enumerate(SEEDS):
        proper,cal,held=C.split(d['ids'],d['folds'],fold); one,full=C.reference_predictors(42,fold,proper,cal,held)
        unweighted=C.reference_predictors(seed,fold,proper,cal,held)
        cq=d['train'][:,cal].reshape(-1,174); hq=d['evaluation'][:,held].reshape(-1,174)
        cc=E.control_features(cq,np.column_stack((one(cq[:,:107]),full(cq))))
        hc=E.control_features(hq,np.column_stack((one(hq[:,:107]),full(hq))))
        cy=np.tile(d['y'][cal],4); cids=np.tile(d['ids'][cal],4)
        with np.load(R.OUT/f'fold{fold}.npz') as z:
            products=C.cloudpickle.loads(z['heads'].tobytes())['rank_products']; pt=z['thresholds'][R.ARMS.index('rank_products')]
        covariance,crecord=covariance_models(training,d,fold,proper,seed,fp)
        whead,wrecord=warning[fold]; calmeta={}
        thresholds[0,fold]=d['baseline_thresholds'][fold]
        thresholds[1,fold]=pt
        predictions[1,:,held,3]=R.score(products,hc,hq,'rank_products').reshape(len(d['names']),len(held)).T
        cs=W.warning_score(whead,cc,cq); hs=W.warning_score(whead,hc,hq)
        wt=np.array([C.threshold(cs,cy,b) for b in C.BUDGETS]); thresholds[2,fold]=wt
        predictions[2,:,held,3]=hs.reshape(len(d['names']),len(held)).T
        calmeta['warning_nonlinear']=dict(false_rates=[float((-cs[cy>0]<=t).mean()) for t in wt])
        for ai,arm,models,tx,hx in [(3,'unweighted_own',unweighted,cq,hq),
            (4,'covariance_full',covariance,training[:,cal].reshape(-1,174),evaluation[:,held].reshape(-1,174))]:
            a,b=models; cp=.5*(a(tx[:,:107])+b(tx)); hp=.5*(a(hx[:,:107])+b(hx))
            dist,levels,record=G.calibrated(cp,cy,cids,hp)
            predictions[ai,:,held]=dist.reshape(len(d['names']),len(held),6).transpose(1,0,2)
            thresholds[ai,fold]=levels; calmeta[arm]=record
        for ai in (0,1,2): np.testing.assert_array_equal(predictions[ai,:,held][...,[0,1,2,4,5]],d['baseline'][:,held].transpose(1,0,2)[...,[0,1,2,4,5]])
        if fold<2:
            with np.load(OLD/f'w1_fold{fold}.npz') as z:
                np.testing.assert_allclose(predictions[2,:,held],z['predictions'][z['arms'].tolist().index('classification_b5')].transpose(1,0,2),atol=1e-12)
            with np.load(OLD/f'n1_fold{fold}.npz') as z:
                np.testing.assert_allclose(predictions[4,:,held],z['predictions'][z['arms'].tolist().index('full_own')].transpose(1,0,2),atol=1e-12)
        models=dict(warning=whead,covariance=covariance)
        C.save_npz(OUT/f'confirmed_fold{fold}.npz',models=np.frombuffer(C.cloudpickle.dumps(models),dtype=np.uint8),proper_rows=proper,calibration_rows=cal,held_rows=held,
            thresholds=thresholds[:,fold],fingerprint=json.dumps(fp))
        fits.append(dict(fold=fold,seed=seed,warning=wrecord,covariance=crecord,calibration=calmeta))
    assert np.isfinite(predictions).all() and np.isfinite(thresholds).all() and signature()==fp
    C.save_npz(OUT/'confirmed_predictions.npz',predictions=predictions,thresholds=thresholds,arms=ARMS,y=d['y'],ids=d['ids'],row_fold=d['folds'],source_indices=d['rows'],conditions=d['names'],fingerprint=json.dumps(fp))
    C.save_json(OUT/'confirmation_fits.json',dict(fits=fits,fingerprint=fp,protocol=P))
    result=dict(scope='Five TRAIN folds, mixed prescribed seeds42/43/44; original two screens reused, not blind confirmation.',conditions={},families={})
    for label,members,dest in [(n,[n],result['conditions']) for n in d['names']]+[(f,v,result['families']) for f,v in G.A.FAMILIES.items()]:
        ci=[d['names'].index(n) for n in members]; dest[label]=dict(arms={arm:G.A.scores(predictions[i,ci],thresholds[i],d['y'],d['ids'],d['folds']) for i,arm in enumerate(ARMS)},paired={})
        for candidate,reference in [('warning_nonlinear','blend_frozen'),('warning_nonlinear','rank_products'),('covariance_full','blend_frozen'),('covariance_full','unweighted_own')]:
            ai,ri=ARMS.index(candidate),ARMS.index(reference)
            dest[label]['paired'].setdefault(candidate,{})[reference]=G.A.compare(predictions[ri,ci],predictions[ai,ci],thresholds[ri],thresholds[ai],d['y'],d['ids'],d['folds'])
        if label in ('clean',*G.A.FAMILIES): print('CONFIRM_ENDPOINTS',label,json.dumps({arm:dict(miss5=v['warning']['0.05']['unstable_last1_miss']['rate'],false5=v['warning']['0.05']['stable_false_warning']['rate'],stable_over03=v['slices']['stable0.01']['over03']) for arm,v in dest[label]['arms'].items()}),flush=True)
    C.save_json(OUT/'confirmation_results.json',result)
    print('INDEPENDENT_FIVE_FOLD_CONFIRMATION_COMPLETE',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): run()
