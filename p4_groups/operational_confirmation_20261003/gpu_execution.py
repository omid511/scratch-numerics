"""Resume O1 using CUDA propagation/identification/inference, never CPU fallback."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import trajectory_runner as O
from cuda_backend import CudaPlate,CudaInference,estimate_batch


def inputs():
    with np.load(O.SCRIPT/'remote_input.npz') as z:
        d=O.C.cloudpickle.loads(z['data'].tobytes()); fp=json.loads(str(z['fingerprint']))
    assert O.signature()==fp
    return d,fp


def backend_signature():
    return dict(device=torch.cuda.get_device_name(),torch=torch.__version__,dtype='float64',
        sources={p.name:O.C.sha256(p) for p in (Path(__file__),O.SCRIPT/'cuda_backend.py')})


def validate(d,models,signature):
    from mechanics.p4_margin_estimation.design_sampler import DesignSample
    engine=CudaInference(models,d['pairs']); differences=[]; mode_errors=[]
    with np.load(O.OUT/'raw_D000068_slow_cross.npz') as z: raw=z['response'].copy()
    noises=[O.TP.continuous_noise(raw,'slow_cross','initial_only',0,68,noise)[0] for noise in O.TP.NOISE]
    signals=[]
    for noise in noises:
        _,off,scl=O.causal_observation(raw[:,:51],noise[:,:51])
        for end in (512,612,712,1512,4612):
            signal=np.clip((raw[:,end-512:end]+noise[:,end-512:end]-off[:,None])/scl[:,None],-50.,50.).astype(np.float32)
            signals.append(signal)
    # Explicit rank-deficient, constant, zero and saturation-aperture cases.
    t=np.arange(512)*.001; shape=np.arange(1.,17.)[:,None]
    signals.extend([shape*np.exp(-20*t)*np.cos(2*np.pi*65*t),np.zeros((16,512)),np.ones((16,512))])
    short=signals[0].copy(); short[:,20]=50.; signals.append(short)
    reference=[O.B.estimate_modes(signal,.001,include_shapes=True) for signal in signals]
    actual=estimate_batch(signals)
    ridge=O.B.PhysicsFeatureRidge(); rows=[]; cpu_rows=[]
    for signal,a,b in zip(signals,actual,reference):
        assert len(a['alpha'])==len(b['alpha']), (a['alpha'],b['alpha'])
        assert a['usable_fraction']==b['usable_fraction']
        if len(a['alpha']):
            row=np.r_[O.B.base_feature_row(signal,.001,a,ridge),O.B.modal_spatial_row(a,d['pairs']),O.B.modal_spatial_addition_row(a,d['pairs'])]
            cpu=np.r_[O.B.base_feature_row(signal,.001,b,ridge),O.B.modal_spatial_row(b,d['pairs']),O.B.modal_spatial_addition_row(b,d['pairs'])]
            rows.append(row); cpu_rows.append(cpu)
            mode_errors.append(float(abs(row-cpu).max()))
    predicted=engine.predict(np.stack(rows)); expected=.5*(models[0](np.stack(cpu_rows)[:,:107])+models[1](np.stack(cpu_rows)))
    differences.extend(abs(predicted-expected).tolist())
    assert max(differences)<1e-6, ('CPU/CUDA margin disagreement',max(differences))
    with np.load(O.OUT/'trajectories_D000068.npz') as z: identity=json.loads(str(z['identity']))
    plate=O.TP.PlateDynamics(DesignSample(**identity['design']),identity['ucrit'],sensor_yx(d))
    gpu=CudaPlate(plate); propagation=[]
    for forcing in O.TP.FORCING:
        times=np.arange(5513)/1000.; initial=plate.initial(0,weak=forcing=='weak_critical_initial')[0]
        force=O.TP.physical_pulse if forcing=='intermittent_physical_pulses' else None
        cpu=O.TP.propagate_state(times,initial,lambda t:plate.matrix(O.TP.ratio_at('fast_cross',t)),plate.reduced['projection'],
            coefficient_step=.002,input_vector=plate.reduced['force'],forcing=force)
        result=gpu.acquire('fast_cross',0,forcing,duration=5.512)
        np.testing.assert_array_equal(result.valid,cpu.valid)
        relative=float(np.sqrt(np.mean((result.response-cpu.response)**2))/np.sqrt(np.mean(cpu.response**2)))
        assert relative<1e-7,relative
        propagation.append(dict(forcing=forcing,relative_rms=relative))
    raw=result.response
    noise,_=O.TP.continuous_noise(raw,'fast_cross','weak_critical_initial',0,68,'correlated5')
    first=engine.infer_views(raw,[noise])[0]; altered=raw.copy(); altered[:,800:]*=2.
    second=engine.infer_views(altered,[noise])[0]; past=first['times']<.8
    for field in ('point','score','fit_error','identified'): np.testing.assert_array_equal(first[field][past],second[field][past])
    missing=noise.copy(); missing[:,800:900]=np.nan
    dropped=engine.infer_views(raw,[missing])[0]; affected=(dropped['times']>=.8)&(dropped['times']<=1.3)
    assert not dropped['available'][affected].any() and np.isnan(dropped['point'][affected]).all()
    proof=dict(backend=signature,actual_and_boundary_signals=len(signals),maximum_margin_difference=max(differences),
        maximum_feature_difference=max(mode_errors),propagation=propagation,causal_past_windows_exact=int(past.sum()),
        missing_windows_unavailable=int(affected.sum()),device_allocated_bytes=torch.cuda.memory_allocated(),
        equivalence_limits='FP64 margin difference <1e-6; propagation relative RMS <1e-7. Original physical/Mach/bandwidth qualification unchanged.')
    O.C.save_json(O.OUT/'cuda_validation.json',proof)
    print('O1_CUDA_VALIDATION_COMPLETE',json.dumps(proof),flush=True)
    return engine


def sensor_yx(d):
    manifest=json.loads((O.ROOT/'p4_sensor_candidates/manifest.json').read_text())
    return np.array(manifest['candidate_grid_yx'])[d['layout']]


def design_run(design,ucrit,group,sensors,engine,fp,backend):
    path=O.OUT/f'trajectories_{design.design_id}.npz'
    identity=dict(fingerprint=fp,design=design.__dict__,ucrit=ucrit,group=group)
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['identity']))==identity
            if 'backend_fingerprint' in z: assert json.loads(str(z['backend_fingerprint']))==backend
        print('O1_CUDA_REUSED_DESIGN',design.design_id,flush=True); return
    audit=json.loads((O.OUT/f'audit_{design.design_id}.json').read_text()); assert audit['fingerprint']==fp
    plate=CudaPlate(O.TP.PlateDynamics(design,ucrit,sensors)) if audit['acquisition_supported'] else None
    frames=[]; descriptors=[]; offsets=[0]
    for profile in O.TP.PROFILES:
        for replicate in range(3):
            for forcing in O.TP.FORCING:
                if plate is None:
                    raw=np.full((len(sensors),int(O.TP.PROFILES[profile][0][-1]*1000)+1),np.nan); termination='unsupported_acquisition'
                else:
                    acquired=plate.acquire(profile,replicate,forcing); raw=acquired.response; termination=acquired.termination
                    if replicate==0 and forcing=='initial_only':
                        O.C.save_npz(O.OUT/f'raw_{design.design_id}_{profile}.npz',response=raw,valid=acquired.valid,times=acquired.times,
                            final_state=acquired.final_state,fingerprint=json.dumps(fp),backend_fingerprint=json.dumps(backend))
                noises=[]; sigmas=[]
                for noise in O.TP.NOISE:
                    injected,sigma=O.TP.continuous_noise(raw,profile,forcing,replicate,int(design.design_id[1:]),noise) if plate is not None else (np.zeros_like(raw),None)
                    noises.append(injected); sigmas.append(sigma)
                results=engine.infer_views(raw,noises,supported=plate is not None)
                for noise,sigma,frame in zip(O.TP.NOISE,sigmas,results):
                    frames.append(frame); offsets.append(offsets[-1]+len(frame['times']))
                    descriptors.append(dict(design=design.design_id,group=group,profile=profile,forcing=forcing,replicate=replicate,noise=noise,
                        duration=O.TP.PROFILES[profile][0][-1],crossing=O.TP.crossing_time(profile),approach_start=5. if profile.endswith('cross') else None,
                        acquisition_supported=bool(audit['acquisition_supported']),physical_noise_reference_m=sigma,termination=termination))
                print('O1_CUDA_TRAJECTORY_ACQUIRED',design.design_id,profile,forcing,replicate,flush=True)
    O.C.save_npz(path,**{key:np.concatenate([f[key] for f in frames]) for key in frames[0]},offsets=offsets,
        descriptors=json.dumps(descriptors),identity=json.dumps(identity),backend_fingerprint=json.dumps(backend))
    print('O1_CUDA_DESIGN_COMPLETE',design.design_id,group,len(descriptors),flush=True)


def run(validate_only=False):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU run is forbidden')
    d,fp=inputs(); sets=O.partitions(d['ids']); models,legacy=O.fit_baseline(d,sets,fp); backend=backend_signature()
    engine=validate(d,models,backend)
    if validate_only: return
    metadata=json.loads((O.ROOT/'p4_dataset_saturation/metadata.json').read_text())
    designs={v.design_id:v for v in O.sample_designs(metadata['n_designs'],seed=metadata['manifest']['seed'])}
    velocities={v['design_id']:v['u_crit'] for v in metadata['designs'] if v['split']=='train'}; sensors=sensor_yx(d)
    O.preflight([(designs[did],velocities[did],group,sensors,fp) for group in ('policy','calibration','held') for did in sets[group]],fp)
    reused=sorted(p.stem.removeprefix('trajectories_') for p in O.OUT.glob('trajectories_*.npz'))
    previous=json.loads((O.OUT/'execution.json').read_text()) if (O.OUT/'execution.json').exists() else {}
    origins=previous.get('inference_host_by_design',{})
    checkpoint_backends={}
    for did in reused:
        with np.load(O.OUT/f'trajectories_{did}.npz') as z:
            checkpoint_backends[did]='CUDA float64' if 'backend_fingerprint' in z else 'CPU NumPy/SciPy'
    for group in ('policy','calibration','held'):
        for did in sets[group]:
            origins.setdefault(did,platform.node())
            checkpoint_backends.setdefault(did,'CUDA float64')
    execution=dict(host=platform.node(),backend='CUDA float64 propagation/LS-ESPRIT/frozen MLP; CPU acquisition setup and small feature assembly',
        backend_fingerprint=backend,reused_designs=reused,workers=1,parallel_noise_views=9,fingerprint=fp,
        inference_host_by_design=origins,inference_backend_by_design=checkpoint_backends,
        timing_limit='CUDA batches only simultaneous noise views at one trailing-window timestamp, never future windows. Each view records full shared-batch elapsed time, not amortized throughput. CPU checkpoint timings are retained; no deployment deadline claim.')
    O.C.save_json(O.OUT/'execution.json',execution); print('O1_CUDA_RUN_STARTED',json.dumps(execution),flush=True)
    for group in ('policy','calibration','held'):
        for did in sets[group]: design_run(designs[did],velocities[did],group,sensors,engine,fp,backend)
    policy=O.load_trajectories(sets['policy']); calibration=O.load_trajectories(sets['calibration']); held=O.load_trajectories(sets['held'])
    thresholds,record=O.operational_thresholds(policy,calibration); thresholds.update(legacy)
    O.C.save_json(O.OUT/'trajectory_calibration.json',dict(thresholds=thresholds,calibration=record,fingerprint=fp))
    audits={did:json.loads((O.OUT/f'audit_{did}.json').read_text()) for did in sets['held']}
    O.scorecard(held,thresholds,audits,fp)
    import verify_trajectories
    verify_trajectories.verify(data=d,infer_views=engine.infer_views)
    print('O1_CUDA_BENCHMARK_AND_REPLAY_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--validate-only',action='store_true'); args=parser.parse_args()
    with threadpool_limits(limits=1): run(args.validate_only)
