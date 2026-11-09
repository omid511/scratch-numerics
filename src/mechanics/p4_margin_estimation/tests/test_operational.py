"""Operational invariants: state, causality, missing data and delivered alarms."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from mechanics.p4_margin_estimation.operational import (
    propagate_state,causal_observation,trailing_windows,delivered_times,
    alarm_episodes,crossing_metrics,
)


def test_variable_dynamics_converge_and_state_continues_without_reset():
    times=np.arange(1001)*.001
    def matrix(t): return np.array([[0.,1.],[-4.-3.*t,-.3]])
    reference=solve_ivp(lambda t,x:matrix(t)@x,(0.,1.),[1.,.2],t_eval=times,rtol=1e-11,atol=1e-13,method='DOP853').y
    coarse=propagate_state(times,[1.,.2],matrix,np.eye(2),coefficient_step=.02)
    fine=propagate_state(times,[1.,.2],matrix,np.eye(2),coefficient_step=.01)
    assert np.max(abs(fine.response-reference))<np.max(abs(coarse.response-reference))/3
    prefix=propagate_state(times[:501],[1.,.2],matrix,np.eye(2),coefficient_step=.01)
    suffix=propagate_state(times[500:],prefix.final_state,matrix,np.eye(2),coefficient_step=.01)
    np.testing.assert_allclose(suffix.response,fine.response[:,500:],atol=1e-12)


def test_constant_force_accumulates_state_instead_of_reexciting_windows():
    times=np.arange(1001)*.001
    result=propagate_state(times,[0.,0.],lambda t:np.array([[0.,1.],[0.,0.]]),np.eye(2),input_vector=[0.,1.],forcing=lambda t:2.)
    np.testing.assert_allclose(result.response[0],times**2,atol=1e-12)
    np.testing.assert_allclose(result.response[1],2*times,atol=1e-12)


def test_observation_is_causal_and_dropout_remains_missing():
    t=np.arange(1200)*.001; raw=np.array([np.sin(31*t),np.cos(23*t)])
    noise=np.zeros_like(raw); altered=raw.copy(); altered[:,800:]*=1000
    original,_,_=causal_observation(raw,noise); changed,_,_=causal_observation(altered,noise)
    windows=list(trailing_windows(original)); changed_windows=list(trailing_windows(changed))
    np.testing.assert_array_equal(windows[0][1],changed_windows[0][1])
    dropout=raw.copy(); dropout[:,600:620]=np.nan
    measured,_,_=causal_observation(dropout,noise)
    assert np.isnan(measured[:,600:620]).all()
    with pytest.raises(ValueError,match='calibration unavailable'): causal_observation(dropout,noise,calibration_samples=650)


def test_unavailable_updates_neither_clear_nor_create_alarms():
    times=np.arange(12)*.1
    flags=np.array([1,0,0,0,0,0,0,0,0,0,1,0],bool)
    available=np.ones(12,bool); available[3]=False
    episodes=alarm_episodes(times,flags,available,clear_updates=5,operating_end=1.2)
    np.testing.assert_allclose(episodes,[[0.,.8],[1.,1.2]])
    np.testing.assert_array_equal(alarm_episodes(times,flags,np.zeros(12,bool)),np.empty((0,2)))


def test_backlog_and_communication_reduce_usable_lead():
    delivery=delivered_times([1.,1.1,1.2],[.15,.15,.01],communication_seconds=.05)
    np.testing.assert_allclose(delivery,[1.2,1.35,1.36])
    result=crossing_metrics([[1.35,2.]],1.8,1.,horizons=(.4,.5))
    assert result['timely_miss']=={'0.4':False,'0.5':True}
    assert result['delivered_lead']==pytest.approx(.45)


def test_expired_early_alarm_does_not_cover_later_hazardous_approach():
    result=crossing_metrics([[.5,1.],[3.,4.]],4.,2.,horizons=(.5,2.))
    assert result['first_effective_warning']==3.
    assert result['timely_miss']=={'0.5':False,'2.0':True}
    missed=crossing_metrics([[.5,1.]],4.,2.)
    assert missed['first_effective_warning'] is None
    assert all(missed['timely_miss'].values())


@pytest.fixture
def trajectory_benchmark():
    import os
    import sys
    import torch
    from pathlib import Path
    previous_path=sys.path.copy()
    dtype,threads=torch.get_default_dtype(),torch.get_num_threads()
    keys=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    previous_env={key:os.environ.get(key) for key in keys}
    try:
        sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'p4_groups/operational_confirmation_20261003'))
        import trajectory_runner
        yield trajectory_runner
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
        sys.path[:]=previous_path
        for key,value in previous_env.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value


@pytest.mark.parametrize('missing_index,expected_false_safe,expected_positive',[
    (None,.3,.7),(3,0.,.6),
])
def test_false_safe_duration_uses_latched_alarm_not_negative_window_flags(
    trajectory_benchmark,missing_index,expected_false_safe,expected_positive,
):
    benchmark=trajectory_benchmark
    crossing=benchmark.TP.crossing_time('fast_cross')
    times=crossing-.1+np.arange(8)*.1
    available=np.ones(8,dtype=bool)
    if missing_index is not None: available[missing_index]=False
    point=np.full(8,.02); point[0]=-.1
    row=dict(descriptor=dict(duration=20.,crossing=crossing,approach_start=5.,profile='fast_cross'),
        times=times,delivery=times,point=point,score=-point/.01,available=available,
        reason=np.where(available,'available','missing_input'))
    result=benchmark.row_metrics(row,5.)
    assert result['false_safe_seconds']==pytest.approx(expected_false_safe)
    assert result['supercritical_positive_margin_seconds']==pytest.approx(expected_positive)


def test_short_physical_pulse_preserves_oscillatory_forced_response():
    times=np.arange(121)*.001
    matrix=np.array([[0.,900.],[-900.,-2.]])
    load=np.array([0.,1.])
    def pulse(t):
        return np.sin(np.pi*(t-.02)/.02) if .02<=t<=.04 else 0.
    reference=solve_ivp(lambda t,x:matrix@x+load*pulse(t),(0.,.12),[0.,0.],
        t_eval=times,rtol=1e-10,atol=1e-13,max_step=.0005,method='DOP853').y
    actual=propagate_state(times,[0.,0.],lambda t:matrix,np.eye(2),
        input_vector=load,forcing=pulse).response
    relative_rms=np.sqrt(np.mean((actual-reference)**2)/np.mean(reference**2))
    assert relative_rms<.005
