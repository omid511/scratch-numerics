"""N1: independent synthetic quiet covariance; original-channel modal refits."""
from gap_common import *
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from mechanics.p4_margin_estimation.modal_features import estimate_modes
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
B=C.helpers.base
KINDS=['diagonal','full','oracle']


def covariance_from_quiet(source,channels,rho):
    rng=np.random.default_rng(np.random.SeedSequence([1147,int(source)]))
    quiet=np.sqrt(1-rho)*rng.normal(size=(channels,4096))+np.sqrt(rho)*rng.normal(size=(1,4096))
    quiet-=quiet.mean(axis=1,keepdims=True)
    estimated=quiet@quiet.T/(quiet.shape[1]-1)
    estimated=.95*estimated+.05*np.diag(np.diag(estimated))
    oracle=(1-rho)*np.eye(channels)+rho*np.ones((channels,channels))
    return estimated,oracle


def weighted_modes(signal,dt,covariance):
    x=np.asarray(signal,dtype=float); n_total=x.shape[1]
    hits=np.flatnonzero(np.any(abs(x)>=50.,axis=0)); n=int(hits[0]) if hits.size else n_total
    if n<32:
        return estimate_modes(signal,dt,include_shapes=True)
    values,vectors=np.linalg.eigh(covariance)
    if not np.isfinite(values).all() or values.min()<=0: raise ValueError('noise covariance must be positive definite')
    whitening=(vectors/np.sqrt(values))@vectors.T
    whitened=whitening@x[:,:n]
    # Prefix already determined from ORIGINAL physical calibrated channels.
    poles=estimate_modes(whitened,dt,cap=np.finfo(float).max,include_shapes=True)
    alpha,frequency=poles['alpha'],poles['frequency']
    if not len(alpha):
        poles['usable_fraction']=n/n_total
        return poles
    x=x[:,:n]; scale=np.max(abs(x)); x=x/scale
    times=np.arange(n)*dt; growth=alpha[:,None]*times
    envelope=np.exp(growth-growth.max(axis=1,keepdims=True)); phase=2*np.pi*frequency[:,None]*times
    basis=np.column_stack((np.ones(n),(envelope*np.cos(phase)).T,(envelope*np.sin(phase)).T))
    coefficients=np.linalg.lstsq(basis,x.T,rcond=None)[0]
    count=len(alpha); centered=np.sum((x-x.mean(axis=1,keepdims=True))**2)
    if centered<=np.finfo(float).eps*np.sum(x*x):
        return estimate_modes(np.zeros_like(signal),dt,include_shapes=True)
    energy=np.array([np.sum((basis[:,1+i,None]*coefficients[1+i]+basis[:,1+count+i,None]*coefficients[1+count+i])**2) for i in range(count)])
    if not np.isfinite(energy).all() or energy.sum()<=0: raise FloatingPointError('invalid original-channel modal energy')
    order=np.argsort(-energy,kind='stable')
    return dict(alpha=alpha[order],frequency=frequency[order],energy_fraction=energy[order]/energy.sum(),
                relative_fit_error=float(np.sum((x.T-basis@coefficients)**2)/centered),usable_fraction=n/n_total,
                mode_shapes=(coefficients[1:1+count]-1j*coefficients[1+count:])[order])


def measured(raw,sigma,source,name,layout,gains,dt,white,shared):
    def draw(seed,common=False):
        cache=shared if common else white
        if seed not in cache:
            cache[seed]=np.random.default_rng(np.random.SeedSequence([seed,int(source)])).normal(size=(1,raw.shape[1]) if common else raw.shape)
        return cache[seed]
    rho=0.
    if name=='clean': response=raw
    elif name in ('gain_bias','dt_minus2pct','dt_plus2pct'): response=raw+.01*sigma*draw(929)
    elif name=='correlated_train_draw':
        rho=.8; response=raw+.01*sigma*(np.sqrt(.2)*draw(929)+np.sqrt(.8)*draw(931,True))
    else:
        seeds=[int(v) for v in re.findall(r'\d+',name) if int(v)>=900]
        level=.05 if name.startswith(('iid5_','correlated5_')) else .01
        if name.startswith('iid') and len(seeds)==1: noise=draw(seeds[0])
        elif name.startswith('correlated') and len(seeds)==2:
            rho=.8; noise=np.sqrt(.2)*draw(seeds[0])+np.sqrt(.8)*draw(seeds[1],True)
        else: raise ValueError(name)
        response=raw+level*sigma*noise
    prefix=response[:,:51]; scales=np.maximum(np.sqrt(np.mean((prefix-prefix.mean(axis=1,keepdims=True))**2,axis=1)),1e-8)[layout]
    signal=causal_calibration_normalize(response,calibration_samples=51)[layout]
    if name=='gain_bias': signal*=gains[:,None]; scales=scales/gains
    return np.clip(signal,-50,50).astype(np.float32),dt*(.98 if name=='dt_minus2pct' else 1.02 if name=='dt_plus2pct' else 1.),scales,rho


def extract_design(payload):
    did,metadata,names,layout,pairs,gains,fp=payload
    path=ROOT/'p4_sensor_candidates'/f'{did}.npz'; raw_hash=C.sha256(path)
    shard=OUT/f'n1_features_{did}.npz'; identity=dict(fingerprint=fp,raw_hash=raw_hash,metadata=metadata)
    if shard.exists():
        with np.load(shard) as z:
            assert json.loads(str(z['identity']))==identity,'stale extraction shard'
        return str(did)
    with np.load(path) as z:
        source_rows=z['source_indices']; raw=z['raw']; sigma=z['noise_scale']
    ridge=B.PhysicsFeatureRidge(); features=[]; rows=[]; conditions=[]; timing=[]; available=[]
    for item in metadata:
        source,dt,requested=item['source'],item['dt'],item['conditions']
        j=int(np.flatnonzero(source_rows==source)[0]); white,shared={},{}; covariances={}
        for ci in requested:
            name=names[ci]; signal,reported_dt,scales,rho=measured(raw[j],sigma[j],source,name,layout,gains,dt,white,shared)
            if rho not in covariances: covariances[rho]=covariance_from_quiet(source,len(layout),rho)
            estimated,oracle=covariances[rho]; perkind=[]; durations=[]; availability=[]
            for kind in KINDS:
                start=time.perf_counter()
                if name=='clean': modes=estimate_modes(signal,reported_dt,include_shapes=True)
                else:
                    covariance=oracle if kind=='oracle' else (np.diag(np.diag(estimated)) if kind=='diagonal' else estimated)
                    covariance=covariance/scales[:,None]/scales[None,:]
                    modes=weighted_modes(signal,reported_dt,covariance)
                q=np.r_[B.base_feature_row(signal,reported_dt,modes,ridge),B.modal_spatial_row(modes,pairs),B.modal_spatial_addition_row(modes,pairs)]
                perkind.append(q); durations.append(time.perf_counter()-start); availability.append(bool(len(modes['alpha'])))
            features.append(perkind); rows.append(source); conditions.append(ci); timing.append(durations); available.append(availability)
    features=np.array(features); assert np.isfinite(features).all()
    C.save_npz(shard,features=features,source_indices=rows,conditions=conditions,seconds=timing,available=available,identity=json.dumps(identity))
    print('N1_EXTRACTION_COMPLETE',did,len(rows),'acquisitions',flush=True)
    return str(did)


def run():
    d=E.setup(); fp=fingerprint('noise_screen.py'); names=d['names']; trainci=[names.index(n) for n in B.TRAIN_NAMES]
    heldunion=np.isin(d['folds'],[s['fold'] for s in PROTOCOL['screens']])
    gains=np.array(json.loads((ROOT/'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    payloads=[]
    for did in np.unique(d['ids']):
        metadata=[dict(source=int(d['rows'][k]),dt=float(d['dt'][k]),conditions=list(range(len(names))) if heldunion[k] else trainci) for k in np.flatnonzero(d['ids']==did)]
        payloads.append((str(did),metadata,names,d['layout'],d['pairs'],gains,fp))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in as_completed([pool.submit(extract_design,p) for p in payloads]): result.result()
    train=np.full((3,4,len(d['y']),174),np.nan); evaluation=np.full((3,len(names),int(heldunion.sum()),174),np.nan)
    lookup={int(v):k for k,v in enumerate(d['rows'])}; heldrows=np.flatnonzero(heldunion); heldlookup={int(v):k for k,v in enumerate(heldrows)}
    timing=[]; availability=[]
    for did in np.unique(d['ids']):
        with np.load(OUT/f'n1_features_{did}.npz') as z:
            for source,ci,features in zip(z['source_indices'],z['conditions'],z['features']):
                row=lookup[int(source)]
                if int(ci) in trainci: train[:,trainci.index(int(ci)),row]=features
                if heldunion[row]: evaluation[:,int(ci),heldlookup[row]]=features
            timing.extend(z['seconds']); availability.extend(z['available'])
    assert np.isfinite(train).all() and np.isfinite(evaluation).all()
    timing=np.array(timing); availability=np.array(availability)
    C.save_json(OUT/'n1_extraction_summary.json',dict(acquisitions=len(timing),kinds={kind:dict(availability=float(availability[:,i].mean()),
                identification_feature_seconds_p95=float(np.quantile(timing[:,i],.95)),p99=float(np.quantile(timing[:,i],.99)),maximum=float(timing[:,i].max())) for i,kind in enumerate(KINDS)},
                limitation='Synthetic quiet calibration; extraction timing includes modal feature assembly but not raw acquisition normalization/I/O; four concurrent CPU workers.'))
    C.save_npz(OUT/'n1_features.npz',training=train,evaluation=evaluation,held_rows=heldrows,fingerprint=json.dumps(fp))
    for screen in PROTOCOL['screens']:
        fold,seed=screen['fold'],screen['seed']; proper,cal,held=C.split(d['ids'],d['folds'],fold)
        E.previous_fold(fold,proper,cal,held)
        cy=np.tile(d['y'][cal],4); cids=np.tile(d['ids'][cal],4); ty=np.tile(d['y'][proper],4)
        slots=np.array([heldlookup[int(k)] for k in held]); arms=['blend_frozen','unweighted_own']; predictions=[d['baseline'][:,held]]; thresholds=[d['baseline_thresholds'][fold]]; records=[]; models={}
        one,full=C.reference_predictors(seed,fold,proper,cal,held)
        cq=d['train'][:,cal].reshape(-1,174); hq=d['evaluation'][:,held].reshape(-1,174)
        hp,levels,calmeta=calibrated(.5*(one(cq[:,:107])+full(cq)),cy,cids,.5*(one(hq[:,:107])+full(hq)))
        predictions.append(hp.reshape(len(names),len(held),6)); thresholds.append(levels)
        records.append(dict(arm='unweighted_own',seed=seed,reused_converged_archived_models=True,calibration=calmeta))
        for ki,kind in enumerate(KINDS):
            tq=train[ki,:,proper].transpose(1,0,2).reshape(-1,174)
            cq=train[ki,:,cal].transpose(1,0,2).reshape(-1,174)
            hq=evaluation[ki,:,slots].transpose(1,0,2).reshape(-1,174)
            for replay in (True,False):
                arm=f'{kind}_fixed' if replay else f'{kind}_own'
                if replay: a,b=one,full; fitmeta=[]
                else:
                    components=[]; fitmeta=[]
                    for width in (107,174):
                        path=OUT/f'n1_{kind}_{width}_fold{fold}.npz'
                        if path.exists():
                            with np.load(path) as z:
                                assert json.loads(str(z['fingerprint']))==fp
                                predict=C.cloudpickle.loads(z['predictor'].tobytes()); meta=json.loads(str(z['metadata']))
                        else:
                            predict,meta=C.fit_network(tq[:,:width],ty,seed=seed,alpha=1.,max_iter=20000,max_fun=400000)
                            C.save_npz(path,predictor=np.frombuffer(C.cloudpickle.dumps(predict),dtype=np.uint8),metadata=json.dumps(meta),fingerprint=json.dumps(fp),proper_rows=proper)
                        if not meta['success']: raise RuntimeError(f'nonconverged N1 refit: {kind}/{width}/{fold}: {meta}')
                        components.append(predict); fitmeta.append(dict(width=width,**meta))
                        print('N1_REFIT_COMPLETE',fold,kind,width,json.dumps(meta),flush=True)
                    a,b=components; models[kind]=(a,b)
                cp=.5*(a(cq[:,:107])+b(cq)); hp=.5*(a(hq[:,:107])+b(hq))
                distribution,levels,calmeta=calibrated(cp,cy,cids,hp)
                arms.append(arm); predictions.append(distribution.reshape(len(names),len(held),6)); thresholds.append(levels)
                records.append(dict(arm=arm,fits=fitmeta,calibration=calmeta))
        predictions=np.array(predictions); thresholds=np.array(thresholds)
        C.save_npz(OUT/f'n1_fold{fold}.npz',predictions=predictions,thresholds=thresholds,arms=arms,held_rows=held,proper_rows=proper,calibration_rows=cal,
                   models=np.frombuffer(C.cloudpickle.dumps(models),dtype=np.uint8),fingerprint=json.dumps(fp))
        C.save_json(OUT/f'n1_fits_fold{fold}.json',records)
        evaluate(OUT/f'n1_results_fold{fold}.json',d,held,arms,predictions,thresholds,screen)
    assert fingerprint('noise_screen.py')==fp
    print('N1_SCREEN_COMPLETE',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): run()
