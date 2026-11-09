"""Continuous linear-state acquisition and causal advisory episode metrics.

Simulator truth specifies dynamics/events, never estimator inputs. Flags on
stable windows retain their instantaneous labels; event anticipation is a
separate contract. No missing observation is converted into a safe decision.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
from scipy.linalg import expm


@dataclass
class StateAcquisition:
    times: np.ndarray
    response: np.ndarray
    final_state: np.ndarray
    valid: np.ndarray
    termination: str | None


def propagate_state(times, initial_state, matrix_at: Callable, projection, *,
                    coefficient_step=.01, input_vector=None, forcing=None,
                    maximum_state=1e100):
    """Exponential-midpoint integration with continuous, never-reset state.

    Uniform physical output sampling. Coefficients are frozen only within
    numerical integration blocks. Forcing is linearly interpolated between
    output-step endpoints and integrated with the frozen matrix exponential.
    Constant coefficients reproduce matrix-exponential evolution.
    Nonfinite/dynamic-range termination leaves remaining acquisition missing.
    """
    times=np.asarray(times,dtype=float); state=np.array(initial_state,dtype=float,copy=True)
    projection=np.asarray(projection,dtype=float)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all(): raise ValueError('finite sample times required')
    increments=np.diff(times); dt=float(increments[0])
    if dt<=0 or not np.allclose(increments,dt,rtol=1e-9,atol=1e-12): raise ValueError('strictly increasing uniform times required')
    if state.ndim!=1 or projection.ndim!=2 or projection.shape[1]!=len(state) or not np.isfinite(state).all() or not np.isfinite(projection).all(): raise ValueError('incompatible finite state/projection')
    if not np.isfinite(coefficient_step) or coefficient_step<dt or not np.isfinite(maximum_state) or maximum_state<=0: raise ValueError('invalid integration bounds')
    block=max(1,int(round(coefficient_step/dt)))
    input_vector=np.zeros(len(state)) if input_vector is None else np.asarray(input_vector,dtype=float)
    if input_vector.shape!=state.shape or not np.isfinite(input_vector).all(): raise ValueError('invalid input vector')
    previous_load=0. if forcing is None else float(forcing(times[0]))
    if not np.isfinite(previous_load): raise ValueError('nonfinite physical forcing')
    response=np.full((projection.shape[0],len(times)),np.nan); valid=np.zeros(len(times),bool)
    response[:,0]=projection@state; valid[0]=True; previous=None; transition=None; termination=None
    for begin in range(0,len(times)-1,block):
        end=min(begin+block,len(times)-1); midpoint=.5*(times[begin]+times[end])
        matrix=np.asarray(matrix_at(midpoint),dtype=float)
        if matrix.shape!=(len(state),len(state)) or not np.isfinite(matrix).all(): raise ValueError('invalid dynamics matrix')
        if previous is None or not np.array_equal(previous,matrix):
            if forcing is None:
                transition=expm(matrix*dt)
            else:
                # Augment with u'=slope: exact first-order hold for each step.
                augmented=np.zeros((len(state)+2,len(state)+2))
                augmented[:-2,:-2]=matrix; augmented[:-2,-2]=input_vector
                augmented[-2,-1]=1.
                step=expm(augmented*dt); transition=step[:-2,:-2]
                right=step[:-2,-1]/dt; left=step[:-2,-2]-right
            previous=matrix.copy()
        for index in range(begin,end):
            state=transition@state
            if forcing is not None:
                load=float(forcing(times[index+1]))
                if not np.isfinite(load): raise ValueError('nonfinite physical forcing')
                state+=left*previous_load+right*load
                previous_load=load
            if not np.isfinite(state).all() or np.max(abs(state))>maximum_state:
                termination='numerical_dynamic_range'; break
            response[:,index+1]=projection@state; valid[index+1]=True
        if termination is not None: break
    return StateAcquisition(times,response,state,valid,termination)


def causal_observation(raw, noise, *, calibration_samples=51, cap=50.):
    """One fixed leading calibration; shared measured samples across windows.

    Missing data stay NaN. Calibration must itself be available. The original
    calibrated physical amplitudes determine saturation, not modal transforms.
    """
    raw=np.asarray(raw,dtype=float); noise=np.asarray(noise,dtype=float)
    if raw.ndim!=2 or noise.shape!=raw.shape or not 1<=calibration_samples<=raw.shape[1] or not np.isfinite(cap) or cap<=0: raise ValueError('invalid acquisition/calibration')
    measured=raw+noise; leading=measured[:,:calibration_samples]
    if not np.isfinite(leading).all(): raise ValueError('leading calibration unavailable')
    offset=leading.mean(axis=1,keepdims=True); scale=np.maximum(np.sqrt(np.mean((leading-offset)**2,axis=1,keepdims=True)),1e-8)
    calibrated=(measured-offset)/scale
    return np.clip(calibrated,-cap,cap).astype(np.float32),offset[:,0],scale[:,0]


def trailing_windows(samples, *, window=512, stride=100):
    """Yield (exclusive sample end, shared trailing view); never future data."""
    samples=np.asarray(samples)
    if samples.ndim!=2 or window<1 or stride<1: raise ValueError('invalid trailing-window settings')
    for end in range(window,samples.shape[1]+1,stride): yield end,samples[:,end-window:end]


def delivered_times(window_end, computation_seconds, *, communication_seconds=0.):
    """Single causal monitor queue; include compute/backlog/communication delay."""
    end=np.asarray(window_end,dtype=float); compute=np.asarray(computation_seconds,dtype=float)
    if end.ndim!=1 or compute.shape!=end.shape or not np.isfinite(end).all() or not np.isfinite(compute).all() or np.any(np.diff(end)<=0) or np.any(compute<0) or not np.isfinite(communication_seconds) or communication_seconds<0: raise ValueError('invalid delivery timing')
    delivered=np.empty_like(end); previous=-np.inf
    for i,(when,duration) in enumerate(zip(end,compute)):
        finished=max(when,previous)+duration; delivered[i]=finished+communication_seconds; previous=finished
    return delivered


def alarm_episodes(times, flags, available, *, clear_updates=5, operating_end=None):
    """Unavailable updates neither begin nor clear an alarm, and never mean safe.

    An episode starts with the first available positive. Exactly the declared
    number of consecutive available negatives clears it; a missing update
    breaks that consecutive-negative run. Returned end is the clearing delivery
    time, or operating_end for an episode still active at acquisition end.
    """
    times=np.asarray(times,dtype=float); flags=np.asarray(flags,dtype=bool); available=np.asarray(available,dtype=bool)
    if times.ndim!=1 or flags.shape!=times.shape or available.shape!=times.shape or not np.isfinite(times).all() or np.any(np.diff(times)<0) or clear_updates<1: raise ValueError('invalid alarm observations')
    if operating_end is None: operating_end=float(times[-1]) if len(times) else 0.
    if not np.isfinite(operating_end) or (len(times) and operating_end<times[-1]): raise ValueError('invalid operating end')
    episodes=[]; start=None; negatives=0
    for when,positive,usable in zip(times,flags,available):
        if not usable: negatives=0; continue
        if positive:
            if start is None: start=float(when)
            negatives=0
        elif start is not None:
            negatives+=1
            if negatives>=clear_updates:
                episodes.append((start,float(when))); start=None; negatives=0
    if start is not None: episodes.append((start,float(operating_end)))
    return np.array(episodes,dtype=float).reshape(-1,2)


def crossing_metrics(episodes, crossing, approach_start, horizons=(.5,1.,2.,5.)):
    """An alarm must be active during the approach, not merely flagged once."""
    episodes=np.asarray(episodes,dtype=float).reshape(-1,2)
    if not np.isfinite(crossing) or not np.isfinite(approach_start) or crossing<approach_start: raise ValueError('invalid crossing event')
    eligible=episodes[(episodes[:,1]>=approach_start)&(episodes[:,0]<=crossing)]
    delivered=np.maximum(eligible[:,0],approach_start)
    first=float(delivered.min()) if len(delivered) else None
    return dict(first_effective_warning=first,delivered_lead=None if first is None else float(crossing-first),
        timely_miss={str(float(h)):bool(not len(eligible) or not np.any((delivered<=crossing-h)&(eligible[:,1]>=approach_start))) for h in horizons})
