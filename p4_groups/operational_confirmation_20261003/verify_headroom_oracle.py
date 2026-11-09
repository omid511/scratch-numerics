"""Independent arithmetic, map, population and nonorthogonal-energy checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

SCRIPT=Path(__file__).resolve().parent
ROOT=SCRIPT.parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=SCRIPT/'headroom_oracle_20261008'
SOURCE=SCRIPT/'fitter_margin_comparison_20261008'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(archive,key):
    return json.loads(str(archive[key]))


def poly(x,degree):
    columns=[x[:,j] for j in range(x.shape[1])]
    if degree==2:
        columns.extend(x[:,j]*x[:,k] for j,k in itertools.combinations_with_replacement(range(x.shape[1]),2))
    return np.column_stack(columns)


def independent_ridge(x,y,query,alphas):
    mean=x.mean(axis=0);std=x.std(axis=0)
    keep=std>1e-8*np.maximum(1,abs(mean))
    scale=np.where(keep,std+1e-8,1)
    z=(x-mean)/scale;q=(query-mean)/scale
    z[:,~keep]=0;q[:,~keep]=0
    u,s,vh=np.linalg.svd(z,full_matrices=False)
    projected=u.T@(y-y.mean())
    return np.column_stack([q@vh.T@(s/(s*s+a)*projected)+y.mean() for a in alphas])


def equal_design_error(y,pred,ids):
    return np.stack([np.abs(pred[ids==did]-y[ids==did,None]).mean(axis=0) for did in sorted(set(ids))]).mean(axis=0)


def verify_maps(z,result,protocol):
    records=js(z,'records');ids=np.array([r['design'] for r in records]);noise=np.array([r['noise'] for r in records]);split=np.array([r['split'] for r in records])
    names=js(z,'names');truth=z['truth'];allids=np.broadcast_to(ids[:,None],truth.shape)
    candidates=np.logspace(-6,6,25);checks=0;max_error=0.
    for name,selection in result['selections'].items():
        family,arm=name.split('/')
        common=z['critical_common'] if family=='critical_contrast' else z['matched_common']
        fit=common&((split=='train')&np.isin(noise,('clean','iid1')))[:,None]
        feature=z[{'A':'measured_features','B':'matched_features','C':'critical_features'}[arm]]
        x=feature[fit];y=truth[fit];did=allids[fit]
        if selection['tuning_rows']!=len(y) or set(did)!=set(protocol['split']['train']):
            raise AssertionError('Proper-only tuning population changed')
        scored=[]
        for degree in (1,2):
            xp=poly(x,degree);oof=np.empty((len(y),len(candidates)))
            for held in protocol['split']['train']:
                test=did==held
                oof[test]=independent_ridge(xp[~test],y[~test],xp[test],candidates)
            errors=equal_design_error(y,oof,did)
            scored.extend((float(error),degree,-float(alpha)) for error,alpha in zip(errors,candidates))
        _,degree,minus_alpha=min(scored)
        if degree!=selection['selected']['degree'] or -minus_alpha!=selection['selected']['alpha']:
            raise AssertionError(f'Independent LODO selection differs for {name}')
        pred=independent_ridge(poly(x,degree),y,poly(feature[common],degree),[-minus_alpha])[:,0]
        saved=z['predictions'][names.index(name)]
        if not np.array_equal(np.isfinite(saved),common):
            raise AssertionError('Oracle acquired stale/extra outputs')
        error=float(np.max(abs(pred-saved[common])))
        if error>1e-8:
            raise AssertionError(f'Independent selected-map error {name}: {error}')
        max_error=max(max_error,error);checks+=1
    return checks,max_error


def verify_metrics(z,result):
    truth=z['truth'];records=js(z,'records');names=js(z,'names')
    ids=np.array([r['design'] for r in records]);noise=np.array([r['noise'] for r in records]);split=np.array([r['split'] for r in records])
    fresh=z['source_fresh'];checked=0
    for key,models in result['metrics'].items():
        sp,n=key.split('/');group=np.broadcast_to(((split==sp)&(noise==n))[:,None],truth.shape)
        masks={'all':group,'source_fresh':group&fresh,'critical_common':group&z['critical_common'],
               'matched_common':group&z['matched_common'],'gap':group&~fresh,
               'boundary':group&(truth>-.02)&(truth<.05),'unstable':group&(truth<0),'farther_stable':group&(truth>=.10)}
        for name,strata in models.items():
            pred=z['predictions'][names.index(name)]
            for label,reported in strata.items():
                population=masks[label];finite=population&np.isfinite(pred)
                assert reported['population']==int(population.sum())
                assert reported['scored']==int(finite.sum())
                assert reported['unknown']==int((population&~np.isfinite(pred)).sum())
                errors=[]
                for did,entry in reported['per_design'].items():
                    eligible=population&(ids==did)[:,None];scored=eligible&np.isfinite(pred)
                    assert entry['population']==int(eligible.sum()) and entry['scored']==int(scored.sum())
                    assert entry['unknown']==int((eligible&~np.isfinite(pred)).sum())
                    if scored.any():
                        error=float(np.mean([abs(float(a)-float(b)) for a,b in zip(pred[scored],truth[scored])]))
                        assert abs(entry['mae']-error)<1e-12;errors.append(error)
                    else:assert entry['mae'] is None
                if errors:assert abs(reported['equal_design_mae']-sum(errors)/len(errors))<1e-12
                else:assert reported['equal_design_mae'] is None
                eligibility={'optimism':finite&(truth>-.02)&(truth<.05),
                    'stable_sign':finite&(truth>=.01)&(truth<=.05),
                    'unstable_sign':finite&(truth<=-.01)&(truth>=-.05),
                    'unstable_false_positive_point':finite&(truth<0)}
                events={'optimism':pred-truth>.02,'stable_sign':np.sign(pred)!=np.sign(truth),
                    'unstable_sign':np.sign(pred)!=np.sign(truth),'unstable_false_positive_point':pred>0}
                for stat,eligible in eligibility.items():
                    r=reported[stat];den=int(eligible.sum());num=int((eligible&events[stat]).sum())
                    assert r['events']==num and r['denominator']==den
                    assert (r['rate'] is None) if den==0 else abs(r['rate']-num/den)<1e-12
                checked+=1
        pop=result['populations'][key]
        assert pop['total']==int(group.sum()) and pop['source_fresh']==int((group&fresh).sum())
        assert pop['source_fresh']+pop['gap']==pop['total']
        assert sum(pop['reasons'].values())==pop['total']
        assert pop['critical_valid']==int((group&z['critical_common']).sum())
        assert pop['matched_valid']==int((group&z['matched_common']).sum())
    return checked


def paired_reports(z):
    records=js(z,'records');ids=np.array([r['design'] for r in records]);noise=np.array([r['noise'] for r in records]);split=np.array([r['split'] for r in records]);names=js(z,'names');truth=z['truth']
    out={}
    for family,arms,maskname in [('critical_contrast',('A','C'),'critical_common'),('correspondence_contrast',('A','B','C'),'matched_common')]:
        labels=[family+'/'+arm for arm in arms]
        if not set(labels)<=set(names):
            out[family]={'status':'training support unresolved'};continue
        out[family]={}
        for condition in ('clean','iid1','correlated1','iid5','correlated5'):
            group=z[maskname]&((split=='held')&(noise==condition))[:,None]
            out[family][condition]={}
            for region,eligible in [('all',group),('boundary',group&(truth>-.02)&(truth<.05)),('unstable',group&(truth<0)),('farther_stable',group&(truth>=.10))]:
                rows={}
                for did in sorted(set(ids[(split=='held')&(noise==condition)])):
                    use=eligible&(ids==did)[:,None]
                    maes={arm:float(np.abs(z['predictions'][names.index(family+'/'+arm)][use]-truth[use]).mean()) if use.any() else None for arm in arms}
                    delta={b+'_minus_'+a:maes[b]-maes[a] if use.any() else None for a,b in itertools.combinations(arms,2)}
                    rows[did]={'frames':int(use.sum()),'mae':maes,'paired_differences':delta}
                out[family][condition][region]=rows
    return out


def energy_and_conditioning(z):
    """Endpoint conditioning and nonadditive window energies; no visibility gate."""
    import trajectory_physics as TP
    import diagnostic_matrix as M
    with np.load(SOURCE/'raw_signals.npz',allow_pickle=False) as rawz:
        raw=rawz['physical'].copy();raw_records=js(rawz,'records')
    lookup={(r['design'],r['replicate'],r['profile']):i for i,r in enumerate(raw_records)}
    records=js(z,'records');shape=z['truth'].shape
    interference=np.full(shape,np.nan);total_energy=np.full(shape,np.nan)
    conditioning=np.full(shape,np.nan);projector_norm=np.full(shape,np.nan)
    endpoint_pole_error=0.;matrix_cache={};shards={}
    for ri,r in enumerate(records):
        did=r['design']
        if did not in shards:
            with np.load(OUT/f'states_{did}.npz',allow_pickle=False) as s:
                shards[did]={k:s[k].copy() for k in ('base','stiffness','damping','physical_basis','projection','structural_dofs','identity')}
        s=shards[did];identity=js(s,'identity');design=identity['design']
        physical=raw[lookup[(did,r['replicate'],r['profile'])]]
        noises,_=M.noises(physical,np.random.SeedSequence([20261010,int(did[1:]),r['replicate'],('cross','stable','retreat').index(r['profile'])]))
        _,_,scale=M.normalize(physical,noises[r['noise']])
        for frame in np.flatnonzero(z['critical_common'][ri]):
            end=128+64*frame;start=end-128
            window=physical[:,start:end]/scale[:,None]
            total=float(np.sum((window-window.mean(axis=1,keepdims=True))**2))
            total_energy[ri,frame]=total
            interference[ri,frame]=total-z['support'][ri,frame,0]-z['support'][ri,frame,1]
            ratio=float(np.interp((end-1)*.001,r['schedule']['knots'],r['schedule']['ratios']))
            cache_key=(did,ratio)
            if cache_key not in matrix_cache:
                velocity=ratio*identity['ucrit'];mach=velocity/design['c_air']
                dynamic=design['rho_air']*velocity**2/np.sqrt(mach*mach-1)
                damping=dynamic*(mach*mach-2)/(mach*mach-1)/velocity
                matrix=s['base']+dynamic*s['stiffness']+damping*s['damping']
                roots,vectors=np.linalg.eig(matrix);positive=np.flatnonzero(roots.imag>TP.TRANSIENT_FILTER.omega_min)
                participation=TP._transverse_participation(s['physical_basis']@vectors[:matrix.shape[0]//2,positive],int(s['structural_dofs']))
                candidates=positive[participation>=TP.TRANSIENT_FILTER.eta_w_min]
                critical=int(candidates[np.argmax(roots[candidates].real)])
                left=np.linalg.solve(vectors,np.eye(len(roots)))[critical]
                matrix_cache[cache_key]=(roots[critical],float(np.linalg.cond(vectors)),float(np.linalg.norm(vectors[:,critical])*np.linalg.norm(left)))
            root,cond,norm=matrix_cache[cache_key]
            oracle=z['critical_features'][ri,frame]
            error=max(abs(root.real-oracle[0]),abs(root.imag/(2*np.pi)-oracle[1]))
            endpoint_pole_error=max(endpoint_pole_error,float(error))
            conditioning[ri,frame]=cond;projector_norm[ri,frame]=norm
    if endpoint_pole_error>1e-7:
        raise AssertionError(f'Independent endpoint critical pole mismatch: {endpoint_pole_error}')
    summaries={}
    for noise in ('clean','iid1','correlated1','iid5','correlated5'):
        selected=z['critical_common']&np.array([r['split']=='held' and r['noise']==noise for r in records])[:,None]
        n=int(selected.sum())
        def stats(values):
            v=values[selected]
            return {'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'max':float(np.max(v))} if n else None
        fraction=np.divide(interference,total_energy,out=np.full(shape,np.nan),where=total_energy>0)
        energy=z['support'][...,0]
        noise_energy=z['support'][...,2]
        snr=np.divide(energy,noise_energy,out=np.full(shape,np.nan),where=noise_energy>0)
        summaries[noise]={'frames':n,'critical_rest_interference_fraction':stats(fraction),
            'absolute_interference_fraction':stats(abs(fraction)),'full_eigenvector_condition':stats(conditioning),
            'critical_eigenprojector_norm':stats(projector_norm),
            'critical_to_injected_noise_energy_median':float(np.nanmedian(snr[selected])) if n and noise!='clean' else None}
    np.savez_compressed(OUT/'conditioning_and_interference.npz',interference=interference,total_energy=total_energy,
                        condition=conditioning,critical_projector_norm=projector_norm)
    return summaries,endpoint_pole_error,len(matrix_cache)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert sha(OUT/'protocol.json')==result['protocol_sha256']
    assert sha(SCRIPT/'experiment_headroom_oracle.py')==result['implementation_sha256']
    assert sha(OUT/'predictions.npz')==result['predictions_sha256']
    for path,expected in protocol['inputs'].items():assert sha(ROOT/path)==expected
    for did,expected in result['shard_hashes'].items():
        assert sha(OUT/f'oracle_{did}.npz')==expected
        with np.load(OUT/f'oracle_{did}.npz',allow_pickle=False) as shard:
            assert str(shard['state_sha256'])==sha(OUT/f'states_{did}.npz')
            assert float(shard['replay_max'])<=protocol['replay_relative_rms_tolerance']
            valid=shard['valid_b']
            scores=shard['match_score'][valid]
            assert np.array_equal(valid,shard['reason']=='resolved')
            assert np.all(scores[:,0]>=protocol['correspondence']['minimum_squared_overlap'])
            assert np.all(scores[:,0]-scores[:,1]>=protocol['correspondence']['minimum_best_minus_second_squared_overlap'])
            assert shard['critical'].shape==shard['matched'].shape
            assert np.array_equal(np.isfinite(shard['matched']).all(axis=-1),valid)
            assert np.array_equal(np.isfinite(shard['critical']).all(axis=-1),shard['valid_c'])
    with np.load(OUT/'predictions.npz',allow_pickle=False) as z:
        with np.load(SOURCE/'dataset.npz',allow_pickle=False) as original:
            assert np.array_equal(z['truth'],original['truth'])
            assert js(z,'records')==js(original,'records')
            assert np.array_equal(z['source_fresh'],original['observed']&original['fitter_observed'])
        assert not (z['matched_common']&~z['critical_common']).any()
        assert not (z['critical_common']&~z['source_fresh']).any()
        assert np.array_equal(z['critical_features'][z['critical_common'],5],z['measured_features'][z['critical_common'],5])
        assert np.array_equal(z['matched_features'][z['matched_common'],5],z['measured_features'][z['matched_common'],5])
        with np.load(SOURCE/'predictions.npz',allow_pickle=False) as original:
            oldnames=js(original,'prediction_names');names=js(z,'names')
            for old,new in (('ridge6','frozen_ridge6'),('framewise','frozen_framewise')):
                assert np.array_equal(original['predictions'][oldnames.index(old)],z['predictions'][names.index(new)],equal_nan=True)
        maps,error=verify_maps(z,result,protocol)
        metrics=verify_metrics(z,result)
        pairs=paired_reports(z)
        energy,pole_error,endpoints=energy_and_conditioning(z)
    (OUT/'paired_results.json').write_text(json.dumps(pairs,indent=2,allow_nan=False)+'\n')
    (OUT/'conditioning_and_interference.json').write_text(json.dumps({'summaries':energy,'independent_endpoint_pole_max_error':pole_error,
        'limitations':'Pair energies are not additive: include interference. Conditioning/projector norms are coordinate/gauge-dependent diagnostics, not a recoverability guarantee.'},indent=2,allow_nan=False)+'\n')
    report={'results_sha256':sha(OUT/'results.json'),'predictions_sha256':sha(OUT/'predictions.npz'),
        'verification_source_sha256':sha(__file__),'metric_checks':metrics,'independent_lodo_maps':maps,
        'independent_map_max_error':error,'independent_endpoint_pole_checks':endpoints,
        'independent_endpoint_pole_max_error':pole_error,'all_source_populations_preserved':True,
        'frozen_comparators_unchanged':True,'oracle_only_fresh_outputs':True,
        'paired_results_sha256':sha(OUT/'paired_results.json'),
        'conditioning_results_sha256':sha(OUT/'conditioning_and_interference.json'),
        'conditioning_arrays_sha256':sha(OUT/'conditioning_and_interference.npz'),
        'limitations':'Independent numerical checks, not blind/independent physical validation or performance improvement.'}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HEADROOM_ORACLE_VERIFIED',json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
