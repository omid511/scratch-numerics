"""One randomized design-grouped framewise/GRU/time-only comparison."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import diagnostic_matrix as M
import pulse_aware_probe as P
from cuda_backend import tensor, advance, estimate_batch, CudaInference

OUT=M.G.O.SCRIPT/'temporal_probe_20261007'
PROTOCOL=json.loads((OUT/'protocol.json').read_text())


def schedule(design, replicate, profile):
    rng=np.random.default_rng(np.random.SeedSequence([20261009,int(design[1:]),replicate,PROTOCOL['profiles'].index(profile)]))
    initial=rng.uniform(.90,.98); start=rng.uniform(1.,5.); finish=min(start+rng.uniform(1.,2.),6.8)
    if profile=='cross':
        knots=np.array([0.,start,finish,8.]); values=np.array([initial,initial,1.005,1.005])
    elif profile=='stable':
        knots=np.array([0.,start,finish,8.]); values=np.array([initial,initial,rng.uniform(.93,.997),rng.uniform(.93,.997)])
    else:
        knots=np.array([0.,start,finish,8.]); values=np.array([initial,rng.uniform(.985,.999),rng.uniform(.90,.95),rng.uniform(.90,.95)])
    onsets=.6+np.arange(5)*1.4+rng.uniform(0.,.65,5)
    amplitudes=rng.uniform(.5,1.5,5)
    return dict(knots=knots,ratios=values,onsets=onsets,amplitudes=amplitudes)


def acquire(plate, description, replicate):
    times=np.arange(8001)/1000.; begins=np.arange(0,8000,2)
    ratios=np.interp((begins+1)/1000.,description['knots'],description['ratios'])
    unique,indices=np.unique(ratios,return_inverse=True)
    velocity=tensor(unique*plate.ucrit); mach=velocity/plate.design.c_air
    if bool((mach<2).any()):
        raise ValueError('Probe violates the validated supersonic acquisition domain')
    dynamic=plate.design.rho_air*velocity**2/torch.sqrt(mach**2-1)
    damping=dynamic*(mach**2-2)/(mach**2-1)/velocity
    n=len(plate.reduced['force']); force=tensor(plate.reduced['force'])
    transitions=[]; lefts=[]; rights=[]
    for begin in range(0,len(unique),128):
        a=tensor(plate.reduced['base'])+dynamic[begin:begin+128,None,None]*tensor(plate.reduced['stiffness'])+damping[begin:begin+128,None,None]*tensor(plate.reduced['damping'])
        augmented=torch.zeros((len(a),n+2,n+2),device='cuda',dtype=torch.float64)
        augmented[:,:n,:n]=a; augmented[:,:n,n]=force; augmented[:,n,n+1]=1.
        step=torch.linalg.matrix_exp(augmented*.001)
        right=step[:,:n,-1]/.001
        transitions.append(step[:,:n,:n].contiguous()); rights.append(right); lefts.append(step[:,:n,-2]-right)
    loads=np.zeros(len(times))
    for onset, amplitude in zip(description['onsets'],description['amplitudes']):
        phase=times-onset; active=(phase>0)&(phase<.020)
        loads[active]+=amplitude*np.sin(np.pi*phase[active]/.020)
    initial=plate.initial(replicate)[0]
    response,valid,state,terminated=advance(tensor(initial),torch.cat(transitions),torch.cat(lefts),torch.cat(rights),indices.tolist(),loads.tolist(),tensor(plate.reduced['projection']),2)
    if terminated or not bool(valid.all()):
        raise RuntimeError('Temporal physical trajectory exceeded numerical dynamic range')
    return times,response.cpu().numpy(),loads!=0


def data(d, fp, engine, smoke):
    records=[]; rows=[]; labels=[]; available=[]; ages=[]
    for split,designs in PROTOCOL['split'].items():
        for design in designs[:1] if smoke else designs:
            _,plate=M.context(d,fp,dict(design=design,replicate=0))
            for replicate in range(1 if smoke else PROTOCOL['schedule_replicates']):
                for profile in PROTOCOL['profiles']:
                    description=schedule(design,replicate,profile)
                    times,raw,command=acquire(plate,description,replicate)
                    sigma=float(np.sqrt(((raw[:,:51]-raw[:,:51].mean(axis=1,keepdims=True))**2).mean()))
                    rng=np.random.default_rng(np.random.SeedSequence([20261010,int(design[1:]),replicate,PROTOCOL['profiles'].index(profile)]))
                    injected=.01*sigma*rng.normal(size=raw.shape)
                    ends=np.arange(128,len(times)+1,64); frame_times=times[ends-1]
                    truth=1-np.interp(frame_times,description['knots'],description['ratios'])
                    signals=[M.normalize(raw,noise)[0] for noise in (np.zeros_like(raw),injected)]
                    q=np.zeros((2,len(ends),174)); good=np.zeros((2,len(ends)),bool)
                    age=np.zeros((2,len(ends)))
                    last=np.zeros(2)
                    for j,end in enumerate(ends):
                        windows=[]; indices=[]
                        for i,signal in enumerate(signals):
                            window=signal[:,end-128:end]
                            if not command[end-128:end].any() and np.isfinite(window).all() and not (abs(window)>=50).any():
                                windows.append(window); indices.append(i)
                        if windows:
                            prefix_modes=estimate_batch([window[:,:64] for window in windows])
                            skills=P.forecast_skills(windows,prefix_modes,64,64)
                            chosen=[k for k,skill in enumerate(skills) if skill>=.5]
                            if chosen:
                                modes=estimate_batch([windows[k] for k in chosen])
                                for k,mode in zip(chosen,modes):
                                    i=indices[k]
                                    if len(mode['alpha']):
                                        q[i,j]=M.features(windows[k],mode,d,engine)
                                        if np.isfinite(q[i,j]).all():
                                            good[i,j]=True; last[i]=frame_times[j]
                        age[:,j]=frame_times[j]-last
                    for i,noise in enumerate(PROTOCOL['noise']):
                        record=dict(design=design,split=split,profile=profile,replicate=replicate,noise=noise,
                            schedule=M.clean_json(description),accepted_windows=int(good[i].sum()),frames=len(ends))
                        records.append(record); rows.append(q[i]); labels.append(truth); available.append(good[i]); ages.append(age[i])
                    print('TEMPORAL_ACQUIRED',design,replicate,profile,'accepted',good.sum(axis=1).tolist(),flush=True)
    return dict(features=np.stack(rows),truth=np.stack(labels),observed=np.stack(available),age=np.stack(ages),
                times=frame_times,records=records)


class Recurrent(torch.nn.Module):
    def __init__(self,width):
        super().__init__()
        self.gru=torch.nn.GRU(width,32,batch_first=True)
        self.head=torch.nn.Sequential(torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1))
    def forward(self,x):
        hidden,_=self.gru(x)
        return self.head(hidden)[...,0]


def train(kind,x,y,proper,validation):
    torch.manual_seed(42)
    if x.device.type=='cuda':
        torch.cuda.manual_seed_all(42)
    if kind=='framewise':
        model=torch.nn.Sequential(torch.nn.Linear(x.shape[-1],32),torch.nn.ReLU(),torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1)).to(device=x.device,dtype=x.dtype)
    else:
        model=Recurrent(x.shape[-1]).to(device=x.device,dtype=x.dtype)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0001)
    best=np.inf; best_state=None; best_epoch=None; stale=0; history=[]
    for epoch in range(200):
        model.train(); optimizer.zero_grad(set_to_none=True)
        point=model(x[proper]); point=point[...,0] if kind=='framewise' else point
        loss=((point-y[proper])**2).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite temporal loss; do not suppress numerical failure')
        loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad():
            point=model(x[validation]); point=point[...,0] if kind=='framewise' else point
            mae=float(abs(point-y[validation]).mean())
        history.append(dict(epoch=epoch+1,train_mse=loss.detach().item(),validation_mae=mae))
        if mae<best:
            best=mae; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1; stale=0
        else:
            stale+=1
        if stale>=30:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        point=model(x); point=point[...,0] if kind=='framewise' else point
    torch.save(dict(state_dict=best_state,kind=kind,protocol=PROTOCOL),OUT/f'{kind}.pt')
    return point.cpu().numpy(),dict(best_epoch=best_epoch,epochs=len(history),validation_mae=best,history=history,
                                   parameters=sum(p.numel() for p in model.parameters()))


def metrics(point,y,observed):
    error=point-y
    def condition(mask):
        count=int(mask.sum())
        return dict(frames=count,mae=float(abs(error[mask]).mean()) if count else None,
            signed_error=float(error[mask].mean()) if count else None,
            over_0_01_rate=float((error[mask]>.01).mean()) if count else None,
            over_0_03_rate=float((error[mask]>.03).mean()) if count else None,
            optimistic_positive_margin_rate=float((point[mask]>0).mean()) if count else None)
    return dict(all=condition(np.ones(y.shape,bool)),observed=condition(observed),missing=condition(~observed),
                stable_last2pct=condition((y>0)&(y<.02)),unstable=condition(y<0))


def supplements(d, fp, engine):
    """Cached frozen-baseline residuals and true near-boundary waveform queries."""
    partitions=M.G.O.partitions(d['ids']); output=dict(clip_residuals={},boundary_queries=[])
    point=engine.predict(d['train'].reshape(-1,174)).reshape(4,-1)
    for split in ('fit','held'):
        indices=np.isin(d['ids'],partitions[split]); y=np.broadcast_to(d['y'][indices],point[:,indices].shape)
        output['clip_residuals'][split]=metrics(point[:,indices],y,np.ones(y.shape,bool))
    for case in M.PROTOCOL['cases']:
        _,plate=M.context(d,fp,case)
        rawpath=P.OUT/f"raw_{case['design']}_fast_cross_intermittent_physical_pulses_rep{case['replicate']}.npz"
        with np.load(rawpath) as z:
            raw=z['response'].copy()
        signal,_,_=M.normalize(raw,np.zeros_like(raw))
        with np.load(P.OUT/f"gate_{case['design']}_fast_cross_intermittent_physical_pulses_rep{case['replicate']}_clean.npz") as z:
            times=z['times']; accepted=z['baseline_available']&z['predictive_guard']
        mean,std,active=[v.cpu().numpy() for v in engine.models[1][1:]]
        fit=np.isin(d['ids'],partitions['fit']); train=(d['train'][0,fit]-mean)/std; train[:,~active]=0
        for t in times[accepted]:
            margin=1-float(M.G.O.TP.ratio_at('fast_cross',t))
            if not -.02<margin<.05:
                continue
            end=int(round(t*1000))+1; window=signal[:,end-512:end]; mode=estimate_batch([window])[0]
            ref=M.truth(plate,float(M.G.O.TP.ratio_at('fast_cross',t))); quality=M.quality(mode,ref)
            q=M.features(window,mode,d,engine); prediction=float(engine.predict(q[None])[0])
            query=(q-mean)/std; query[~active]=0
            distance=np.sqrt(((train-query)**2).sum(axis=1)/int(active.sum())); nearest=np.argsort(distance)[:7]
            perturb=(d['train'][1,fit]-d['train'][0,fit])/std; perturb[:,~active]=0
            output['boundary_queries'].append(dict(design=case['design'],time=float(t),margin=margin,prediction=prediction,
                max_abs_standardized=float(abs(query[active]).max()),physically_credible=quality['recovered'],pole_quality=quality,
                nearest_distances=distance[nearest],neighbor_margins=d['y'][fit][nearest],neighbor_designs=d['ids'][fit][nearest],
                median_same_clip_noise_distance=float(np.median(np.sqrt((perturb**2).sum(axis=1)/int(active.sum()))))))
    return output


def run(smoke):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    d,fp=M.G.inputs(); models=[]
    for width in (107,174):
        with np.load(M.G.O.OUT/f'trajectory_baseline{width}.npz') as z:
            assert json.loads(str(z['fingerprint']))==fp
            models.append(M.G.O.C.cloudpickle.loads(z['predictor'].tobytes()))
    engine=CudaInference(models,d['pairs']); start=time.perf_counter(); cache=OUT/('smoke_data.npz' if smoke else 'data.npz')
    if cache.exists():
        with np.load(cache) as z:
            assert json.loads(str(z['protocol']))==PROTOCOL
            assert json.loads(str(z['fingerprint']))==fp
            assert str(z['source_sha256'])==M.G.O.C.sha256(Path(__file__))
            dataset={k:z[k].copy() for k in ('features','truth','observed','age','times')}; dataset['records']=json.loads(str(z['records']))
    else:
        dataset=data(d,fp,engine,smoke)
        M.G.O.C.save_npz(cache,**{k:v for k,v in dataset.items() if k!='records'},records=json.dumps(dataset['records']),
            protocol=json.dumps(PROTOCOL),fingerprint=json.dumps(fp),source_sha256=M.G.O.C.sha256(Path(__file__)))
    q,y,observed=dataset['features'],dataset['truth'],dataset['observed']; records=dataset['records']
    proper=np.array([r['split']=='train' for r in records]); validation=np.array([r['split']=='validation' for r in records]); held=np.array([r['split']=='held' for r in records])
    assert all(not set(r['design'] for r in records if r['split']==a)&set(r['design'] for r in records if r['split']==b) for a,b in [('train','held'),('train','validation'),('validation','held')])
    # Only accepted training inputs determine feature scaling; missing rows are explicit, never simulator imputation.
    if int(observed[proper].sum())<20:
        M.save('temporal_prerequisite.json',dict(accepted_training_windows=int(observed[proper].sum()),status='Insufficient accepted inputs; no conditional model training'))
        raise RuntimeError('Insufficient accepted training inputs for the declared conditional comparison')
    from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
    mean,scale,active=fit_ridge_scaling(q[proper][observed[proper]])
    normalized=(q-mean)/scale; normalized[...,~active]=0; normalized[~observed]=0
    x=np.concatenate((normalized,observed[...,None].astype(float),dataset['age'][...,None]),axis=2)
    x=torch.as_tensor(x,device='cuda',dtype=torch.float32); target=torch.as_tensor(y,device='cuda',dtype=torch.float32)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    output=dict(protocol=PROTOCOL,fingerprint=fp,source_sha256=M.G.O.C.sha256(Path(__file__)),backend=M.G.backend_signature(),
        records=records,models={},accepted_training_windows=int(observed[proper].sum()),elapsed_seconds=None,
        training_dtype='float32',helper_source_sha256={p.name:M.G.O.C.sha256(p) for p in (Path(M.__file__),Path(P.__file__))})
    predictions=[]
    for kind in ('framewise','gru','time_only'):
        inputs=x if kind!='time_only' else torch.as_tensor(np.broadcast_to(dataset['times'][None,:,None],(len(records),len(dataset['times']),1)).copy(),device='cuda',dtype=torch.float32)
        point,training=train(kind,inputs,target,proper,validation); predictions.append(point)
        output['models'][kind]=dict(training=training,held=metrics(point[held],y[held],observed[held]),
            by_profile_noise={f'{profile}/{noise}':metrics(point[selection],y[selection],observed[selection])
                for profile in PROTOCOL['profiles'] for noise in PROTOCOL['noise']
                for selection in [held & np.array([r['profile']==profile and r['noise']==noise for r in records])]})
        print('TEMPORAL_TRAINED',kind,'epoch',training['best_epoch'],'held',json.dumps(output['models'][kind]['held']),flush=True)
    M.G.O.C.save_npz(OUT/('smoke_predictions.npz' if smoke else 'predictions.npz'),predictions=np.stack(predictions),mean=mean,scale=scale,active=active,truth=y,
        observed=observed,age=dataset['age'],times=dataset['times'],records=json.dumps(records),protocol=json.dumps(PROTOCOL))
    if not smoke:
        output['frozen_baseline_diagnostics']=supplements(d,fp,engine)
    output['elapsed_seconds']=time.perf_counter()-start
    M.G.O.C.save_json(OUT/('smoke_results.json' if smoke else 'results.json'),M.clean_json(output))
    print('TEMPORAL_PROBE_COMPLETE',len(records),'trajectories',output['elapsed_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.smoke)
