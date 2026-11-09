"""Exercise actual raw acquisitions, saved models and calibration for all screens."""
from gap_common import *
import warning_screen as W
import noise_screen as N
spec=importlib.util.spec_from_file_location('gap_raw_acquisition',C.OUT/'prepare_features.py')
acquisition=importlib.util.module_from_spec(spec); spec.loader.exec_module(acquisition)


def verify():
    d=E.setup(); gains=np.array(json.loads((ROOT/'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    with np.load(C.OUT/'features.npz') as z: raw_hashes=json.loads(str(z['raw_hashes']))
    with np.load(OUT/'n1_features.npz') as z:
        ntrain=z['training'].copy(); neval=z['evaluation'].copy(); nheld=z['held_rows'].copy()
    heldlookup={int(v):k for k,v in enumerate(nheld)}
    replays=[]; max_error=0.; cal_proofs=[]; invariants=[]
    for screen in PROTOCOL['screens']:
        fold,seed=screen['fold'],screen['seed']; proper,cal,held=C.split(d['ids'],d['folds'],fold)
        one,full=C.reference_predictors(42,fold,proper,cal,held)
        none,nfull=C.reference_predictors(seed,fold,proper,cal,held)
        artifacts={}
        for name,script in [('w1','warning_screen.py'),('s1','tail_screen.py'),('n1','noise_screen.py')]:
            with np.load(OUT/f'{name}_fold{fold}.npz') as z:
                assert json.loads(str(z['fingerprint']))==fingerprint(script)
                for key,rows in [('proper_rows',proper),('calibration_rows',cal),('held_rows',held)]: np.testing.assert_array_equal(z[key],rows)
                artifacts[name]=dict(p=z['predictions'].copy(),t=z['thresholds'].copy(),arms=z['arms'].tolist(),
                    heads=C.cloudpickle.loads(z['heads' if name!='n1' else 'models'].tobytes()))
            a=artifacts[name]; assert np.isfinite(a['p']).all() and np.isfinite(a['t']).all()
            np.testing.assert_array_equal(a['p'][0],d['baseline'][:,held])
        for arm_prediction in artifacts['w1']['p']:
            np.testing.assert_array_equal(arm_prediction[...,[0,1,2,4,5]],d['baseline'][:,held][...,[0,1,2,4,5]])
        cq=d['train'][:,cal].reshape(-1,174); cy=np.tile(d['y'][cal],4); cids=np.tile(d['ids'][cal],4)
        control=E.control_features(cq,np.column_stack((one(cq[:,:107]),full(cq))))
        for arm,head in artifacts['w1']['heads'].items():
            cs=W.warning_score(head,control,cq); ai=artifacts['w1']['arms'].index(arm)
            levels=[C.threshold(cs,cy,b) for b in C.BUDGETS]
            np.testing.assert_array_equal(levels,artifacts['w1']['t'][ai])
            cal_proofs.append(dict(screen='w1',fold=fold,arm=arm,exact_threshold_replay=True))
        slevels={}
        for arm,head in artifacts['s1']['heads'].items():
            cp=.5*(one(cq[:,:107])+neural_predict(head['model'],head['scaling'],cq)); levels=C.calibration(cp,cy,cids)
            cs=C.distribution(cp,levels)[:,3]; ai=artifacts['s1']['arms'].index(arm)
            np.testing.assert_array_equal([C.threshold(cs,cy,b) for b in C.BUDGETS],artifacts['s1']['t'][ai])
            slevels[arm]=levels; cal_proofs.append(dict(screen='s1',fold=fold,arm=arm,exact_threshold_replay=True))
        nlevels={}
        for arm in artifacts['n1']['arms'][1:]:
            if arm=='unweighted_own': nx=cq; a,b=none,nfull
            else:
                kind,mode=arm.split('_'); ki=N.KINDS.index(kind); nx=ntrain[ki,:,cal].transpose(1,0,2).reshape(-1,174)
                a,b=(none,nfull) if mode=='fixed' else artifacts['n1']['heads'][kind]
            cp=.5*(a(nx[:,:107])+b(nx)); levels=C.calibration(cp,cy,cids); cs=C.distribution(cp,levels)[:,3]
            ai=artifacts['n1']['arms'].index(arm)
            np.testing.assert_array_equal([C.threshold(cs,cy,budget) for budget in C.BUDGETS],artifacts['n1']['t'][ai])
            nlevels[arm]=levels; cal_proofs.append(dict(screen='n1',fold=fold,arm=arm,exact_threshold_replay=True))
        requested=[int(held[np.flatnonzero(mask[held])[0]]) for mask in ((d['y']>0)&(d['y']<.01),(d['y']<0)&(d['y']>-.01),d['y']>=.15)]
        for k in requested:
            did=str(d['ids'][k]); source=int(d['rows'][k]); path=ROOT/'p4_sensor_candidates'/f'{did}.npz'
            assert C.sha256(path)==raw_hashes[did]
            with np.load(path) as z:
                j=int(np.flatnonzero(z['source_indices']==source)[0]); raw,sigma=z['raw'][j],z['noise_scale'][j]
            hslot=int(np.flatnonzero(held==k)[0])
            for condition in ('clean','iid5_new949','correlated5_new949_950'):
                ci=d['names'].index(condition)
                signal,dt=acquisition.measured_signal(raw,sigma,source,condition,d['layout'],gains,float(d['dt'][k]),{},{})
                modes=N.estimate_modes(signal,dt,include_shapes=True)
                ridge=N.B.PhysicsFeatureRidge()
                q=np.r_[N.B.base_feature_row(signal,dt,modes,ridge),N.B.modal_spatial_row(modes,d['pairs']),N.B.modal_spatial_addition_row(modes,d['pairs'])][None]
                np.testing.assert_allclose(q[0],d['evaluation'][ci,k],atol=1e-9,rtol=1e-9)
                control=E.control_features(q,np.column_stack((one(q[:,:107]),full(q))))
                errors={}
                for arm,head in artifacts['w1']['heads'].items():
                    ai=artifacts['w1']['arms'].index(arm); actual=W.warning_score(head,control,q)[0]; expected=artifacts['w1']['p'][ai,ci,hslot,3]
                    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8); errors['w1/'+arm]=float(abs(actual-expected))
                for arm,head in artifacts['s1']['heads'].items():
                    ai=artifacts['s1']['arms'].index(arm); point=.5*(one(q[:,:107])+neural_predict(head['model'],head['scaling'],q))
                    actual=C.distribution(point,slevels[arm])[0]; expected=artifacts['s1']['p'][ai,ci,hslot]
                    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8); errors['s1/'+arm]=float(abs(actual-expected).max())
                signal2,dt2,scales,rho=N.measured(raw,sigma,source,condition,d['layout'],gains,float(d['dt'][k]),{},{})
                np.testing.assert_array_equal(signal2,signal); assert dt==dt2
                estimated,oracle=N.covariance_from_quiet(source,len(d['layout']),rho)
                for ki,kind in enumerate(N.KINDS):
                    cov=oracle if kind=='oracle' else (np.diag(np.diag(estimated)) if kind=='diagonal' else estimated)
                    modes=N.estimate_modes(signal,dt,include_shapes=True) if condition=='clean' else N.weighted_modes(signal,dt,cov/scales[:,None]/scales[None,:])
                    nx=np.r_[N.B.base_feature_row(signal,dt,modes,ridge),N.B.modal_spatial_row(modes,d['pairs']),N.B.modal_spatial_addition_row(modes,d['pairs'])][None]
                    np.testing.assert_allclose(nx[0],neval[ki,ci,heldlookup[k]],atol=1e-9,rtol=1e-9)
                    for mode in ('fixed','own'):
                        arm=kind+'_'+mode; ai=artifacts['n1']['arms'].index(arm)
                        a,b=(none,nfull) if mode=='fixed' else artifacts['n1']['heads'][kind]
                        point=.5*(a(nx[:,:107])+b(nx)); actual=C.distribution(point,nlevels[arm])[0]; expected=artifacts['n1']['p'][ai,ci,hslot]
                        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8); errors['n1/'+arm]=float(abs(actual-expected).max())
                maximum=max(errors.values()); max_error=max(max_error,maximum)
                replays.append(dict(fold=fold,source_index=source,design=did,condition=condition,max_output_error=maximum))
                print('GAP_RAW_REPLAY',fold,source,condition,maximum,flush=True)
        invariants.append(dict(fold=fold,disjoint_designs=True,frozen_warning_point_intervals_exact=True,all_outputs_finite=True))
    C.save_json(OUT/'verification.json',dict(actual_waveform_replays=replays,maximum_output_error=max_error,
                calibration_replays=cal_proofs,invariants=invariants,scope='Two TRAIN fold screens; not operational verification.'))
    print('GAP_VERIFICATION_COMPLETE',len(replays),'waveforms',len(cal_proofs),'exact calibration replays',flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): verify()
