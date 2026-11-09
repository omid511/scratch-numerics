"""Raw measured-waveform and calibration replay of independently confirmed arms."""
import confirm_candidates as F
G,C,E,R,np=F.G,F.C,F.E,F.R,F.np
N,W=F.N,F.W
import importlib.util,json
spec=importlib.util.spec_from_file_location('confirmation_raw_acquisition',C.OUT/'prepare_features.py')
acquisition=importlib.util.module_from_spec(spec); spec.loader.exec_module(acquisition)


def verify():
    d=E.setup(); fp=F.signature()
    with np.load(F.OUT/'confirmed_predictions.npz') as z:
        assert json.loads(str(z['fingerprint']))==fp
        predictions=z['predictions'].copy(); thresholds=z['thresholds'].copy(); arms=z['arms'].tolist()
        for key,value in [('y',d['y']),('ids',d['ids']),('row_fold',d['folds']),('source_indices',d['rows']),('conditions',d['names'])]: np.testing.assert_array_equal(z[key],value)
    with np.load(F.OUT/'covariance_features.npz') as z:
        assert json.loads(str(z['fingerprint']))==fp
        ntrain=z['training'].copy(); neval=z['evaluation'].copy()
    with np.load(C.OUT/'features.npz') as z: raw_hashes=json.loads(str(z['raw_hashes']))
    gains=np.array(json.loads((F.ROOT/'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    replays=[]; calibration=[]; invariants=[]; maximum=0.
    for fold,seed in enumerate(F.SEEDS):
        proper,cal,held=C.split(d['ids'],d['folds'],fold)
        assert not(set(d['ids'][proper])&set(d['ids'][held]) or set(d['ids'][cal])&set(d['ids'][held]) or set(d['ids'][proper])&set(d['ids'][cal]))
        with np.load(F.OUT/f'confirmed_fold{fold}.npz') as z:
            assert json.loads(str(z['fingerprint']))==fp
            for key,value in [('proper_rows',proper),('calibration_rows',cal),('held_rows',held)]: np.testing.assert_array_equal(z[key],value)
            models=C.cloudpickle.loads(z['models'].tobytes()); np.testing.assert_array_equal(z['thresholds'],thresholds[:,fold])
        one,full=C.reference_predictors(42,fold,proper,cal,held)
        unweighted=C.reference_predictors(seed,fold,proper,cal,held)
        cq=d['train'][:,cal].reshape(-1,174); cy=np.tile(d['y'][cal],4); ids=np.tile(d['ids'][cal],4)
        control=E.control_features(cq,np.column_stack((one(cq[:,:107]),full(cq))))
        warning=W.warning_score(models['warning'],control,cq)
        np.testing.assert_array_equal([C.threshold(warning,cy,b) for b in C.BUDGETS],thresholds[arms.index('warning_nonlinear'),fold])
        calibration.append(dict(fold=fold,arm='warning_nonlinear',exact_threshold_replay=True))
        levels={}
        for arm,fit,features in [('unweighted_own',unweighted,cq),('covariance_full',models['covariance'],ntrain[:,cal].reshape(-1,174))]:
            a,b=fit; point=.5*(a(features[:,:107])+b(features)); level=C.calibration(point,cy,ids); score=C.distribution(point,level)[:,3]
            np.testing.assert_array_equal([C.threshold(score,cy,budget) for budget in C.BUDGETS],thresholds[arms.index(arm),fold])
            levels[arm]=level; calibration.append(dict(fold=fold,arm=arm,exact_threshold_replay=True))
        for arm in ('blend_frozen','rank_products','warning_nonlinear'):
            np.testing.assert_array_equal(predictions[arms.index(arm),:,held][...,[0,1,2,4,5]],d['baseline'][:,held].transpose(1,0,2)[...,[0,1,2,4,5]])
        requests=[int(held[np.flatnonzero(mask[held])[0]]) for mask in ((d['y']>0)&(d['y']<.01),(d['y']<0)&(d['y']>-.01),d['y']>=.15)]
        conditions=['clean','iid5_new949','correlated5_new949_950',G.A.FAMILIES['replication_correlated5'][0]]
        for row in requests:
            source=int(d['rows'][row]); did=str(d['ids'][row]); path=F.ROOT/'p4_sensor_candidates'/f'{did}.npz'
            assert C.sha256(path)==raw_hashes[did]
            with np.load(path) as z:
                index=int(np.flatnonzero(z['source_indices']==source)[0]); raw,sigma=z['raw'][index],z['noise_scale'][index]
            for condition in conditions:
                ci=d['names'].index(condition)
                signal,dt=acquisition.measured_signal(raw,sigma,source,condition,d['layout'],gains,float(d['dt'][row]),{},{})
                modes=N.estimate_modes(signal,dt,include_shapes=True); ridge=N.B.PhysicsFeatureRidge()
                features=np.r_[N.B.base_feature_row(signal,dt,modes,ridge),N.B.modal_spatial_row(modes,d['pairs']),N.B.modal_spatial_addition_row(modes,d['pairs'])][None]
                np.testing.assert_allclose(features[0],d['evaluation'][ci,row],atol=1e-9,rtol=1e-9)
                control=E.control_features(features,np.column_stack((one(features[:,:107]),full(features))))
                actual=W.warning_score(models['warning'],control,features)[0]; expected=predictions[arms.index('warning_nonlinear'),ci,row,3]
                np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8); errors={'warning_nonlinear':float(abs(actual-expected))}
                measured,dt2,scales,rho=N.measured(raw,sigma,source,condition,d['layout'],gains,float(d['dt'][row]),{},{})
                np.testing.assert_array_equal(measured,signal); assert dt==dt2
                covariance=N.covariance_from_quiet(source,len(d['layout']),rho)[0]/scales[:,None]/scales[None,:]
                modes=N.estimate_modes(signal,dt,include_shapes=True) if condition=='clean' else N.weighted_modes(signal,dt,covariance)
                nx=np.r_[N.B.base_feature_row(signal,dt,modes,ridge),N.B.modal_spatial_row(modes,d['pairs']),N.B.modal_spatial_addition_row(modes,d['pairs'])][None]
                np.testing.assert_allclose(nx[0],neval[ci,row],atol=1e-9,rtol=1e-9)
                for arm,fit,q in [('unweighted_own',unweighted,features),('covariance_full',models['covariance'],nx)]:
                    a,b=fit; actual=C.distribution(.5*(a(q[:,:107])+b(q)),levels[arm])[0]; expected=predictions[arms.index(arm),ci,row]
                    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8); errors[arm]=float(abs(actual-expected).max())
                error=max(errors.values()); maximum=max(maximum,error)
                replays.append(dict(fold=fold,source_index=source,design=did,condition=condition,errors=errors))
        invariants.append(dict(fold=fold,partitions_disjoint=True,frozen_warning_margin_intervals_exact=True))
        print('CONFIRM_RAW_REPLAY',fold,len(requests)*len(conditions),'waveforms',flush=True)
    assert F.signature()==fp
    C.save_json(F.OUT/'confirmation_verification.json',dict(raw_waveform_replays=replays,maximum_output_error=maximum,
        exact_calibration_replays=calibration,invariants=invariants,fingerprint=fp,scope='Saved-model five-fold clip replay, not trajectory operational validation.'))
    print('CONFIRMATION_VERIFICATION_COMPLETE',len(replays),'waveforms',len(calibration),'exact calibration replays',maximum,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): verify()
