"""O1 baseline: separate design fit/policy/calibration/held trajectories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; SCRIPT=Path(__file__).resolve().parent
OUT=SCRIPT/'o1_corrected'
sys.path.insert(0,str(ROOT/'p4_groups/independent_gap_screens_20261003'))
import gap_common as G
sys.path.insert(0,str(SCRIPT))
import trajectory_physics as TP
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from mechanics.p4_margin_estimation.design_sampler import sample_designs
from mechanics.p4_margin_estimation.operational import causal_observation,trailing_windows,delivered_times,alarm_episodes,crossing_metrics
C,E,np=G.C,G.E,G.np
B=C.helpers.base
P=json.loads((SCRIPT/'protocol.json').read_text())
P['O1'].update(json.loads((SCRIPT/'trajectory_revision.json').read_text())['O1'])
EPISODE_BUDGETS=(.1,1.,10.,100.)


def signature():
    return dict(parent=E.signature(),sources={str(p.relative_to(ROOT)):C.sha256(p) for p in
        (SCRIPT/'protocol.json',SCRIPT/'trajectory_revision.json',Path(__file__),SCRIPT/'trajectory_physics.py',ROOT/'src/mechanics/p4_margin_estimation/operational.py',ROOT/'p4_dataset_saturation/metadata.json')})


def partitions(ids):
    ids=np.unique(ids).copy(); np.random.default_rng(1747).shuffle(ids)
    fit=int(np.floor(.6*len(ids))); policy=int(np.floor(.1*len(ids))); cal=int(np.ceil(.1*len(ids)))
    sets=dict(fit=ids[:fit],policy=ids[fit:fit+policy],calibration=ids[fit+policy:fit+policy+cal],held=ids[fit+policy+cal:])
    assert sum(len(v) for v in sets.values())==len(ids)
    assert all(not set(a)&set(b) for i,a in enumerate(sets.values()) for b in list(sets.values())[i+1:])
    return sets


def fit_baseline(d,sets,fp):
    rows=np.flatnonzero(np.isin(d['ids'],sets['fit'])); models=[]; records=[]
    x=d['train'][:,rows].reshape(-1,174); y=np.tile(d['y'][rows],4)
    for width in (107,174):
        path=OUT/f'trajectory_baseline{width}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint']))==fp
                np.testing.assert_array_equal(z['fit_rows'],rows)
                model=C.cloudpickle.loads(z['predictor'].tobytes()); meta=json.loads(str(z['metadata']))
        else:
            model,meta=C.fit_network(x[:,:width],y,seed=42,alpha=1.,max_iter=20000,max_fun=400000)
            C.save_npz(path,predictor=np.frombuffer(C.cloudpickle.dumps(model),dtype=np.uint8),metadata=json.dumps(meta),fit_rows=rows,fingerprint=json.dumps(fp))
        if not meta['success']: raise RuntimeError(f'nonconverged trajectory baseline {width}: {meta}')
        models.append(model); records.append(dict(width=width,**meta))
    calrows=np.flatnonzero(np.isin(d['ids'],sets['calibration'])); cq=d['train'][:,calrows].reshape(-1,174); cy=np.tile(d['y'][calrows],4)
    point=.5*(models[0](cq[:,:107])+models[1](cq)); score=-point/.01
    thresholds={f'legacy_clip{int(b*100)}':float(-C.threshold(score,cy,b)) for b in (.01,.05)}
    rates={key:float((score[cy>0]>=t).mean()) for key,t in thresholds.items()}
    assert rates['legacy_clip1']<=.01 and rates['legacy_clip5']<=.05
    C.save_json(OUT/'trajectory_fit.json',dict(partitions={k:v.tolist() for k,v in sets.items()},fits=records,legacy_score_thresholds=thresholds,calibration_false_flags=rates,fingerprint=fp))
    return models,thresholds


def infer_measured(raw,noise,models,pairs,*,supported=True):
    times=[]; points=[]; scores=[]; seconds=[]; identifiable=[]; quality=[]; reasons=[]
    ends=np.arange(512,raw.shape[1]+1,100); window_times=(ends-1)/1000.
    if not supported:
        count=len(ends)
        return dict(times=window_times,point=np.full(count,np.nan),score=np.full(count,np.nan),seconds=np.zeros(count),
            identified=np.zeros(count,bool),available=np.zeros(count,bool),delivery=window_times,fit_error=np.full(count,np.nan),reason=np.full(count,'unsupported_acquisition'))
    # Calibrate once from the observed prefix, never a future trajectory RMS.
    _,offset,scale=causal_observation(raw[:,:51],noise[:,:51])
    measured=raw+noise; ridge=B.PhysicsFeatureRidge()
    for end,window in trailing_windows(measured):
        start=G.time.perf_counter(); point=score=np.nan; identified=False; fit_error=np.nan; reason='missing_input'
        if np.isfinite(window).all():
            signal=np.clip((window-offset[:,None])/scale[:,None],-50.,50.).astype(np.float32)
            modes=B.estimate_modes(signal,.001,include_shapes=True); fit_error=modes['relative_fit_error']
            if len(modes['alpha']):
                q=np.r_[B.base_feature_row(signal,.001,modes,ridge),B.modal_spatial_row(modes,pairs),B.modal_spatial_addition_row(modes,pairs)][None]
                point=float(.5*(models[0](q[:,:107])[0]+models[1](q)[0])); score=-point/.01
                identified=bool(np.isfinite(point) and np.isfinite(q).all()); reason='available' if identified else 'nonfinite_estimate'
            else: reason='unidentified_or_saturated'
        elapsed=G.time.perf_counter()-start
        times.append((end-1)/1000.); points.append(point); scores.append(score); seconds.append(elapsed); identifiable.append(identified); quality.append(fit_error); reasons.append(reason)
    delivery=delivered_times(times,seconds); deadline=(delivery-np.array(times))<=.1
    available=np.array(identifiable)&deadline
    reasons=np.array(reasons,dtype='<U40'); reasons[np.array(identifiable)&~deadline]='deadline_miss'
    return dict(times=np.array(times),point=np.array(points),score=np.array(scores),seconds=np.array(seconds),identified=np.array(identifiable),
        available=available,delivery=delivery,fit_error=np.array(quality),reason=reasons)


def preflight_design(payload):
    design,ucrit,group,sensor_yx,fp=payload
    path=OUT/f'audit_{design.design_id}.json'
    if path.exists():
        audit=json.loads(path.read_text())
        assert audit['fingerprint']==fp and audit['design']==design.__dict__ and audit['group']==group
        return design.design_id,audit
    if .9*ucrit/design.c_air<2.:
        audit=dict(acquisition_supported=False,rom_valid=None,bandwidth_supported=None,
            minimum_mach=.9*ucrit/design.c_air,reason='initial_profile_outside_validated_Mach2_piston_domain')
    else:
        audit=TP.PlateDynamics(design,ucrit,sensor_yx).audit()
    audit.update(fingerprint=fp,design=design.__dict__,group=group)
    C.save_json(path,audit)
    print('O1_PREFLIGHT_DESIGN',design.design_id,group,'numerics',audit['rom_valid'],
        'bandwidth',audit['bandwidth_supported'],'supported',audit['acquisition_supported'],flush=True)
    return design.design_id,audit


def preflight(payloads,fp):
    audits={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for task in as_completed([pool.submit(preflight_design,p) for p in payloads]):
            did,audit=task.result(); audits[did]=audit
    failed=[did for did,a in audits.items() if a['rom_valid'] is False]
    supported={group:sum(a['acquisition_supported'] for a in audits.values() if a['group']==group)
        for group in ('policy','calibration','held')}
    C.save_json(OUT/'preflight.json',dict(audits=audits,numerical_failures=failed,
        supported_designs=supported,total_designs=len(audits),fingerprint=fp))
    if failed: raise RuntimeError(f'numerical preflight failed; no waveform extraction started: {failed}')
    if not supported['policy'] or not supported['calibration']:
        raise RuntimeError(f'acquisition preflight lacks policy/calibration coverage; no waveform extraction started: {supported}')
    print('O1_PREFLIGHT_COMPLETE',supported,flush=True)
    return audits


def trajectory_design(payload):
    design,ucrit,group,sensor_yx,pairs,model_bytes,fp=payload
    path=OUT/f'trajectories_{design.design_id}.npz'; auditpath=OUT/f'audit_{design.design_id}.json'
    identity=dict(fingerprint=fp,design=design.__dict__,ucrit=ucrit,group=group)
    if path.exists():
        with np.load(path) as z: assert json.loads(str(z['identity']))==identity
        return design.design_id
    models=C.cloudpickle.loads(model_bytes)
    audit=json.loads(auditpath.read_text())
    assert audit['fingerprint']==fp and audit['design']==design.__dict__ and audit['group']==group
    dynamics=TP.PlateDynamics(design,ucrit,sensor_yx) if audit['acquisition_supported'] else None
    frames=[]; descriptors=[]; offsets=[0]
    for profile in TP.PROFILES:
        for replicate in range(3):
            for forcing in TP.FORCING:
                if dynamics is not None and audit['acquisition_supported']:
                    acquired=dynamics.acquire(profile,replicate,forcing); raw=acquired.response
                    termination=acquired.termination
                else:
                    raw=np.full((len(sensor_yx),int(TP.PROFILES[profile][0][-1]*1000)+1),np.nan); termination='unsupported_acquisition'
                if dynamics is not None and audit['acquisition_supported'] and replicate==0 and forcing=='initial_only':
                    C.save_npz(OUT/f'raw_{design.design_id}_{profile}.npz',response=raw,valid=acquired.valid,times=acquired.times,
                        final_state=acquired.final_state,fingerprint=json.dumps(fp))
                for noise in TP.NOISE:
                    if audit['acquisition_supported']: injected,sigma=TP.continuous_noise(raw,profile,forcing,replicate,int(design.design_id[1:]),noise)
                    else: injected=np.zeros_like(raw); sigma=None
                    frame=infer_measured(raw,injected,models,pairs,supported=audit['acquisition_supported'])
                    frames.append(frame); offsets.append(offsets[-1]+len(frame['times']))
                    descriptors.append(dict(design=design.design_id,group=group,profile=profile,forcing=forcing,replicate=replicate,noise=noise,
                        duration=TP.PROFILES[profile][0][-1],crossing=TP.crossing_time(profile),approach_start=5. if profile.endswith('cross') else None,
                        acquisition_supported=bool(audit['acquisition_supported']),physical_noise_reference_m=sigma,termination=termination))
                print('O1_TRAJECTORY_ACQUIRED',design.design_id,profile,forcing,replicate,flush=True)
    arrays={key:np.concatenate([f[key] for f in frames]) for key in frames[0]}
    C.save_npz(path,**arrays,offsets=offsets,descriptors=json.dumps(descriptors),identity=json.dumps(identity))
    print('O1_DESIGN_COMPLETE',design.design_id,group,len(descriptors),'views',flush=True)
    return design.design_id


def load_trajectories(designs):
    rows=[]
    for did in designs:
        with np.load(OUT/f'trajectories_{did}.npz') as z:
            descriptors=json.loads(str(z['descriptors'])); offsets=z['offsets']; keys=('times','point','score','seconds','identified','available','delivery','fit_error','reason')
            for i,descriptor in enumerate(descriptors):
                a,b=offsets[i:i+2]; rows.append(dict(descriptor=descriptor,**{k:z[k][a:b].copy() for k in keys}))
    return rows


def episodes(row,threshold):
    flags=row['score']>=threshold
    return alarm_episodes(row['delivery'],flags,row['available'],operating_end=max(row['descriptor']['duration'],float(row['delivery'][-1])))


def operational_thresholds(policy,calibration):
    policy_safe=[r for r in policy if r['descriptor']['profile']=='safe_dwell']
    values=np.concatenate([r['score'][r['available']] for r in policy_safe])
    if not len(values): raise RuntimeError('no available benign policy observations; cannot select advisory policies')
    candidates=np.unique(np.nextafter(np.quantile(values,np.linspace(.5,1.,21)),np.inf))
    safe=[r for r in calibration if r['descriptor']['profile']=='safe_dwell']
    calvalues=np.concatenate([r['score'][r['available']] for r in safe])
    if not len(calvalues): raise RuntimeError('no available benign calibration observations')
    candidates=np.unique(np.r_[candidates,np.nextafter(calvalues.max(),np.inf)])
    groups={(noise,forcing):[r for r in safe if r['descriptor']['noise']==noise and r['descriptor']['forcing']==forcing] for noise in TP.NOISE for forcing in TP.FORCING}
    rates=[]
    for threshold in candidates:
        rates.append({f'{noise}/{forcing}':sum(len(episodes(r,threshold)) for r in rows)/sum(r['descriptor']['duration']/3600 for r in rows) for (noise,forcing),rows in groups.items()})
    thresholds={}; records={}
    for budget in EPISODE_BUDGETS:
        valid=[i for i,rate in enumerate(rates) if max(rate.values())<=budget]
        if not valid: raise RuntimeError('declared calibration-maximum threshold failed zero-observed episode check')
        selected=min(valid,key=lambda i:candidates[i]); name=f'episodes_per_hour_{budget}'
        thresholds[name]=float(candidates[selected]); records[name]=dict(maximum_calibration_episode_rate=max(rates[selected].values()),rates_by_group=rates[selected],
            available_policy_windows=len(values),available_calibration_windows=len(calvalues),benign_calibration_hours_by_group={f'{n}/{f}':sum(r['descriptor']['duration']/3600 for r in v) for (n,f),v in groups.items()},
            zero_observed_calibration_episodes=all(v==0 for v in rates[selected].values()))
    return thresholds,dict(candidate_thresholds=candidates.tolist(),policies=records,
        limitation='Empirical whole-trajectory calibration with short benign exposure; no population episode-rate bound. Selection has no crossing/held labels.')


def row_metrics(row,threshold):
    d=row['descriptor']; end=d['duration']; alarm=episodes(row,threshold); flags=row['score']>=threshold
    active=np.zeros(len(row['times']),dtype=bool)
    for begin,finish in alarm:
        active|=(row['delivery']>=begin)&(row['delivery']<finish)
    starts=np.minimum(row['delivery'],end); nexts=np.r_[row['delivery'][1:],end]
    stops=np.minimum(np.minimum(nexts,row['times']+.1),end); coverage=np.maximum(stops-starts,0)*row['available']
    available_seconds=float(coverage.sum()); false_safe=0.; positive_margin=0.
    crossing=d['crossing']
    if crossing is not None:
        unstable=np.maximum(stops-np.maximum(starts,crossing),0)*row['available']
        false_safe=float(unstable[~active].sum()); positive_margin=float(unstable[row['point']>0].sum())
    mask=row['available']; y=1-TP.ratio_at(d['profile'],row['times']); error=row['point']-y
    tails={}
    for label,selection in [('stable_last1',(y>0)&(y<.01)),('unstable_last1',(y<0)&(y>-.01)),('stable_last5',(y>0)&(y<.05))]:
        selected=selection&mask
        tails[label]=dict(available_windows=int(selected.sum()),over03_count=int((error[selected]>.03).sum()),
                         mean_abs_error=None if not selected.any() else float(abs(error[selected]).mean()))
    alarm_seconds=float(np.maximum(np.minimum(alarm[:,1],end)-np.minimum(alarm[:,0],end),0).sum())
    return dict(episodes=int(np.count_nonzero(alarm[:,0]<=end)),episode_rate_per_hour=float(np.count_nonzero(alarm[:,0]<=end)/(end/3600)),
        alarm_active_fraction=alarm_seconds/end,available_fraction=available_seconds/end,unavailable_seconds=end-available_seconds,
        false_safe_seconds=false_safe,supercritical_positive_margin_seconds=positive_margin,
        deadline_misses=int(np.count_nonzero(row['delivery']-row['times']>.1)),windows=len(row['times']),
        saturated_or_unidentified_windows=int(np.count_nonzero(row['reason']=='unidentified_or_saturated')),tails=tails,
        event=None if crossing is None else crossing_metrics(alarm,crossing,d['approach_start']))


def scorecard(held,thresholds,audits,fp):
    output=dict(scope='TRAIN-only disjoint trajectory-held nominal designs; frozen-time crossing events, not physical harmful-amplitude acceptance.',
        thresholds=thresholds,design_audits=audits,policies={},fingerprint=fp,
        limits='Synthetic sensor stresses; recovery/communication/harmful-amplitude limits unavailable. Algorithmic output availability is not calibrated correctness confidence. Latency measured under four concurrent CPU workers.')
    for policy,threshold in thresholds.items():
        result={}
        for noise in TP.NOISE:
            for forcing in TP.FORCING:
                groups={profile:[r for r in held if r['descriptor']['noise']==noise and r['descriptor']['forcing']==forcing and r['descriptor']['profile']==profile] for profile in TP.PROFILES}
                metrics={profile:[row_metrics(r,threshold) for r in rows] for profile,rows in groups.items()}
                value=dict(profiles={})
                for profile,rows in groups.items():
                    scores=metrics[profile]; hours=sum(r['descriptor']['duration']/3600 for r in rows)
                    all_seconds=np.concatenate([r['seconds'] for r in rows]); windows=sum(v['windows'] for v in scores)
                    v=dict(trajectories=len(rows),designs=len({r['descriptor']['design'] for r in rows}),operating_hours=hours,
                        episode_rate_per_hour=sum(r['episodes'] for r in scores)/hours,
                        alarm_active_fraction=float(np.mean([r['alarm_active_fraction'] for r in scores])),
                        available_fraction=float(np.mean([r['available_fraction'] for r in scores])),
                        unavailable_seconds=sum(r['unavailable_seconds'] for r in scores),false_safe_seconds=sum(r['false_safe_seconds'] for r in scores),
                        supercritical_positive_margin_seconds=sum(r['supercritical_positive_margin_seconds'] for r in scores),
                        compute_seconds_p95=float(np.quantile(all_seconds,.95)),compute_seconds_p99=float(np.quantile(all_seconds,.99)),compute_seconds_max=float(all_seconds.max()),
                        deadline_miss_fraction=sum(r['deadline_misses'] for r in scores)/windows,
                        tails={label:dict(available_windows=sum(s['tails'][label]['available_windows'] for s in scores),
                            over03_count=sum(s['tails'][label]['over03_count'] for s in scores)) for label in ('stable_last1','unstable_last1','stable_last5')})
                    if profile.endswith('cross'):
                        events=[s['event'] for s in scores]; leads=[e['delivered_lead'] for e in events if e['delivered_lead'] is not None]
                        v.update(timely_event_misses={str(h):float(np.mean([e['timely_miss'][str(h)] for e in events])) for h in (.5,1.,2.,5.)},
                            observed_lead_seconds=leads,events_without_warning=sum(e['first_effective_warning'] is None for e in events))
                    value['profiles'][profile]=v
                result[f'{noise}/{forcing}']=value
        output['policies'][policy]=result
    C.save_json(OUT/'trajectory_scorecard.json',output)
    print('O1_BASELINE_SCORECARD_COMPLETE',len(held),'held trajectory views',flush=True)
    return output


def run(*,preflight_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    d=E.setup(); fp=signature(); sets=partitions(d['ids'])
    metadata=json.loads((ROOT/'p4_dataset_saturation/metadata.json').read_text()); all_designs=sample_designs(metadata['n_designs'],seed=metadata['manifest']['seed'])
    designs={v.design_id:v for v in all_designs}; ucrit={v['design_id']:v['u_crit'] for v in metadata['designs'] if v['split']=='train'}
    manifest=json.loads((ROOT/'p4_sensor_candidates/manifest.json').read_text()); sensor=np.array(manifest['candidate_grid_yx'])[d['layout']]
    assert set(sets['fit'])|set(sets['policy'])|set(sets['calibration'])|set(sets['held'])==set(metadata['train_designs'])
    audit_payloads=[(designs[did],ucrit[did],group,sensor,fp) for group in ('policy','calibration','held') for did in sets[group]]
    preflight(audit_payloads,fp)
    if preflight_only: return
    models,legacy=fit_baseline(d,sets,fp)
    blob=C.cloudpickle.dumps(models)
    payloads=[(design,u,group,layout,d['pairs'],blob,fp) for design,u,group,layout,_ in audit_payloads]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for task in as_completed([pool.submit(trajectory_design,p) for p in payloads]): task.result()
    policy=load_trajectories(sets['policy']); calibration=load_trajectories(sets['calibration']); held=load_trajectories(sets['held'])
    thresholds,record=operational_thresholds(policy,calibration); thresholds.update(legacy)
    C.save_json(OUT/'trajectory_calibration.json',dict(thresholds=thresholds,calibration=record,fingerprint=fp))
    audits={did:json.loads((OUT/f'audit_{did}.json').read_text()) for did in sets['held']}
    assert signature()==fp
    scorecard(held,thresholds,audits,fp)
    print('O1_TRAJECTORY_BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':
    import argparse
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1): run(preflight_only=args.preflight_only)
