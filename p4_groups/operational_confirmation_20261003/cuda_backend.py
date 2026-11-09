"""Float64 CUDA propagation, LS-ESPRIT and frozen MLP inference for O1.

CPU work is limited to acquisition setup, unchanged small feature assembly,
and result serialization. No numerical CPU fallback is permitted.
"""
import time
from typing import List
import numpy as np
import torch
import trajectory_runner as O
from mechanics.p4_margin_estimation.operational import StateAcquisition


def tensor(value):
    return torch.as_tensor(value,dtype=torch.float64,device='cuda')


def least_squares(a,b):
    # CUDA gels assumes full rank. SVD retains NumPy's rank-deficient contract.
    u,s,vh=torch.linalg.svd(a,full_matrices=False,driver='gesvd')
    threshold=s[...,:1]*(max(a.shape[-2:])*torch.finfo(a.dtype).eps)
    inverse=torch.where(s>threshold,1/s,0.)
    return (vh.transpose(-2,-1)*inverse.unsqueeze(-2))@(u.transpose(-2,-1)@b)


def estimate_batch(signals,dt=.001):
    if not np.isfinite(dt) or dt<=0: raise ValueError('positive physical dt required')
    results=[]; groups={}
    for i,signal in enumerate(signals):
        x=np.asarray(signal,dtype=np.float64)
        if x.ndim!=2 or not x.shape[0] or x.shape[1]<8 or not np.isfinite(x).all():
            raise ValueError('finite physical channels with at least eight samples required')
        hits=np.flatnonzero(np.any(abs(x)>=50.,axis=0)); n=int(hits[0]) if len(hits) else x.shape[1]
        results.append(dict(alpha=np.empty(0),frequency=np.empty(0),energy_fraction=np.empty(0),
            relative_fit_error=1.,usable_fraction=n/x.shape[1],mode_shapes=np.empty((0,x.shape[0]),complex)))
        if n>=32 and abs(x[:,:n]).max()>0:
            groups.setdefault((n,x.shape[0]),[]).append((i,x[:,:n]))
    for (n,_),rows in groups.items():
        x=tensor(np.stack([row[1] for row in rows])); x=x/torch.amax(abs(x),dim=(-2,-1),keepdim=True)
        width=min(64,n//2); h=x.unfold(2,width,1).reshape(len(rows),-1,width)
        eigenvalues,vectors=torch.linalg.eigh(h.transpose(1,2)@h)
        ranks=torch.clamp((eigenvalues>eigenvalues[:,-1:]*1e-12).sum(dim=1),max=min(17,width-1)).cpu().tolist()
        by_rank={}
        for k,rank in enumerate(ranks):
            if rank>=2 and float(eigenvalues[k,-1])>0: by_rank.setdefault(rank,[]).append(k)
        for rank,indices in by_rank.items():
            subspace=vectors[indices,:,-rank:]
            shift=least_squares(subspace[:,:-1],subspace[:,1:])
            roots=torch.linalg.eigvals(shift)
            by_count={}
            for j,k in enumerate(indices):
                poles=roots[j][(roots[j].imag>1e-8)&(abs(roots[j])>1e-12)]
                if len(poles): by_count.setdefault(len(poles),[]).append((k,poles))
            for count,items in by_count.items():
                positions=[item[0] for item in items]; poles=torch.stack([item[1] for item in items])
                alpha=torch.log(abs(poles))/dt; frequency=torch.angle(poles)/(2*np.pi*dt)
                t=torch.arange(n,dtype=torch.float64,device='cuda')*dt
                growth=alpha[:,:,None]*t; envelope=torch.exp(growth-growth.amax(dim=2,keepdim=True))
                phase=2*np.pi*frequency[:,:,None]*t
                basis=torch.cat((torch.ones((len(items),n,1),device='cuda',dtype=torch.float64),
                    (envelope*torch.cos(phase)).transpose(1,2),(envelope*torch.sin(phase)).transpose(1,2)),dim=2)
                selected=x[positions]; coefficients=least_squares(basis,selected.transpose(1,2))
                reconstructed=basis@coefficients
                centered=((selected-selected.mean(dim=2,keepdim=True))**2).sum(dim=(1,2))
                relative=((selected.transpose(1,2)-reconstructed)**2).sum(dim=(1,2))/centered
                components=(basis[:,:,1:1+count,None]*coefficients[:,None,1:1+count,:]
                    +basis[:,:,1+count:,None]*coefficients[:,None,1+count:,:])
                energies=(components**2).sum(dim=(1,3)); total=energies.sum(dim=1)
                for j,k in enumerate(positions):
                    if bool(centered[j]<=torch.finfo(torch.float64).eps*(selected[j]**2).sum()) or not bool(torch.isfinite(energies[j]).all()) or float(total[j])<=0: continue
                    order=torch.argsort(energies[j],descending=True,stable=True)
                    shape=torch.complex(coefficients[j,1:1+count],-coefficients[j,1+count:])[order]
                    results[rows[k][0]].update(alpha=alpha[j,order].cpu().numpy(),frequency=frequency[j,order].cpu().numpy(),
                        energy_fraction=(energies[j,order]/total[j]).cpu().numpy(),relative_fit_error=float(relative[j]),mode_shapes=shape.cpu().numpy())
    return results


@torch.jit.script
def advance(state:torch.Tensor,transition:torch.Tensor,left:torch.Tensor,right:torch.Tensor,
            indices:List[int],loads:List[float],projection:torch.Tensor,block:int):
    response=torch.full((projection.size(0),len(loads)),float('nan'),dtype=state.dtype,device=state.device)
    valid=torch.zeros(len(loads),dtype=torch.bool,device=state.device)
    response[:,0]=projection@state; valid[0]=True
    terminated=False
    for i in range(len(loads)-1):
        k=indices[i//block]
        state=transition[k]@state
        if loads[i]!=0. or loads[i+1]!=0.:
            state=state+left[k]*loads[i]+right[k]*loads[i+1]
        if not bool(torch.isfinite(state).all()) or bool(torch.max(torch.abs(state))>1e100):
            terminated=True
            break
        response[:,i+1]=projection@state; valid[i+1]=True
    return response,valid,state,terminated


class CudaPlate:
    def __init__(self,plate):
        self.plate=plate; self.profile=None; self.cache={}
        self.base=tensor(plate.reduced['base']); self.stiffness=tensor(plate.reduced['stiffness'])
        self.damping=tensor(plate.reduced['damping']); self.projection=tensor(plate.reduced['projection'])
        self.force=tensor(plate.reduced['force'])

    def acquire(self,profile,replicate,forcing,*,duration=None,coefficient_step=.002):
        if self.profile!=profile: self.cache.clear(); self.profile=profile
        end=O.TP.PROFILES[profile][0][-1] if duration is None else duration
        times=np.arange(int(round(end*1000))+1)/1000.; dt=.001
        block=int(round(coefficient_step/dt))
        if block<1 or not np.isfinite(coefficient_step): raise ValueError('invalid integration step')
        forced=forcing=='intermittent_physical_pulses'; key=(len(times),block,forced)
        if key not in self.cache:
            begins=np.arange(0,len(times)-1,block); ends=np.minimum(begins+block,len(times)-1)
            ratios=O.TP.ratio_at(profile,.5*(times[begins]+times[ends]))
            unique,indices=np.unique(ratios,return_inverse=True)
            velocity=tensor(unique*self.plate.ucrit); mach=velocity/self.plate.design.c_air
            if bool((mach<=1).any()): raise ValueError('trajectory left supersonic piston domain')
            dynamic=self.plate.design.rho_air*velocity**2/torch.sqrt(mach**2-1)
            damping=dynamic*(mach**2-2)/(mach**2-1)/velocity
            count=len(self.force); transitions=[]; lefts=[]; rights=[]
            for begin in range(0,len(unique),128):
                d=dynamic[begin:begin+128]; c=damping[begin:begin+128]
                matrices=self.base+d[:,None,None]*self.stiffness+c[:,None,None]*self.damping
                if forced:
                    augmented=torch.zeros((len(d),count+2,count+2),device='cuda',dtype=torch.float64)
                    augmented[:,:count,:count]=matrices; augmented[:,:count,count]=self.force
                    augmented[:,count,count+1]=1.
                    step=torch.linalg.matrix_exp(augmented*dt); transition=step[:,:count,:count].contiguous()
                    right=step[:,:count,-1]/dt; left=step[:,:count,-2]-right
                else:
                    transition=torch.linalg.matrix_exp(matrices*dt)
                    left=right=torch.zeros((len(d),count),device='cuda',dtype=torch.float64)
                transitions.append(transition); lefts.append(left); rights.append(right)
            self.cache[key]=(torch.cat(transitions),torch.cat(lefts),torch.cat(rights),indices.tolist())
        transition,left,right,indices=self.cache[key]
        initial=tensor(self.plate.initial(replicate,weak=forcing=='weak_critical_initial')[0])
        loads=[O.TP.physical_pulse(float(t)) for t in times] if forced else [0.]*len(times)
        response,valid,state,terminated=advance(initial,transition,left,right,indices,loads,self.projection,block)
        return StateAcquisition(times,response.cpu().numpy(),state.cpu().numpy(),valid.cpu().numpy(),
            'numerical_dynamic_range' if terminated else None)


class CudaInference:
    def __init__(self,models,pairs):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback forbidden')
        torch.set_num_threads(1); self.models=[]; self.pairs=pairs
        for predictor in models:
            network,scaling=O.G.warm_net(predictor)
            self.models.append((network.to(device='cuda',dtype=torch.float64).eval(),
                tensor(scaling['mean']),tensor(scaling['scale']),torch.as_tensor(scaling['active'],device='cuda')))
        self.ridge=O.B.PhysicsFeatureRidge()

    def predict(self,rows):
        value=tensor(rows); outputs=[]
        with torch.no_grad():
            for network,mean,scale,active in self.models:
                z=(value[:,:len(mean)]-mean)/scale; z[:,~active]=0.
                outputs.append(network(z)[:,0])
        return (.5*(outputs[0]+outputs[1])).cpu().numpy()

    def infer_views(self,raw,noises,*,supported=True):
        ends=np.arange(512,raw.shape[1]+1,100); window_times=(ends-1)/1000.
        if not supported:
            return [O.infer_measured(raw,noise,[],self.pairs,supported=False) for noise in noises]
        frames=[dict(times=window_times.copy(),point=np.full(len(ends),np.nan),score=np.full(len(ends),np.nan),
            seconds=np.zeros(len(ends)),identified=np.zeros(len(ends),bool),fit_error=np.full(len(ends),np.nan),
            reason=np.full(len(ends),'missing_input',dtype='<U40')) for _ in noises]
        measured=[]; offsets=[]; scales=[]
        for noise in noises:
            _,offset,scale=O.causal_observation(raw[:,:51],noise[:,:51])
            measured.append(raw+noise); offsets.append(offset); scales.append(scale)
        for j,end in enumerate(ends):
            torch.cuda.synchronize(); start=time.perf_counter(); signals=[]; indices=[]
            for i,value in enumerate(measured):
                window=value[:,end-512:end]
                if np.isfinite(window).all():
                    signals.append(np.clip((window-offsets[i][:,None])/scales[i][:,None],-50.,50.).astype(np.float32)); indices.append(i)
            modes=estimate_batch(signals); rows=[]; present=[]
            for signal,mode,i in zip(signals,modes,indices):
                frames[i]['fit_error'][j]=mode['relative_fit_error']
                if len(mode['alpha']):
                    row=np.r_[O.B.base_feature_row(signal,.001,mode,self.ridge),O.B.modal_spatial_row(mode,self.pairs),O.B.modal_spatial_addition_row(mode,self.pairs)]
                    if not np.isfinite(row).all(): raise FloatingPointError('nonfinite CUDA-derived features')
                    rows.append(row); present.append(i)
                else: frames[i]['reason'][j]='unidentified_or_saturated'
            if rows:
                for point,i in zip(self.predict(np.stack(rows)),present):
                    identified=bool(np.isfinite(point)); frames[i]['point'][j]=point; frames[i]['score'][j]=-point/.01
                    frames[i]['identified'][j]=identified; frames[i]['reason'][j]='available' if identified else 'nonfinite_estimate'
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            for frame in frames: frame['seconds'][j]=elapsed
        for frame in frames:
            frame['delivery']=O.delivered_times(frame['times'],frame['seconds'])
            deadline=frame['delivery']-frame['times']<=.1
            frame['available']=frame['identified']&deadline
            frame['reason'][frame['identified']&~deadline]='deadline_miss'
        return frames
