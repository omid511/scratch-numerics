"""Independent analytic-Jacobian/QR checks and live fitter replay for the checkpoint."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import experiment_time_varying_fit as E
from noise_aware_carrier import fit_noise_aware

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT / 'noise_aware_carrier_20261008'
TRANSPORT = SCRIPT / 'modal_transport_20261008'


def analytic_statistic(x, dt, fit):
    n, channels = x.shape[1], x.shape[0]
    duration = (n-1)*dt
    u = (np.arange(n)-.5*(n-1))*dt/duration
    names = ('alpha_mid', 'alpha_slope', 'alpha_curvature',
             'frequency_mid', 'frequency_slope', 'frequency_curvature')
    theta = np.array([fit['parameters'][k] for k in names])*np.array([duration,duration**2,duration**3]*2)
    am, ad, ac, fm, fd, fc = theta
    envelope = np.exp(am*u+.5*ad*u**2+ac*u**3/6)
    phase = 2*np.pi*(fm*u+.5*fd*u**2+fc*u**3/6)
    cos, sin = envelope*np.cos(phase), envelope*np.sin(phase)
    basis = np.column_stack((np.ones(n), cos, sin))
    coefficients = np.linalg.lstsq(basis, x.T, rcond=None)[0]
    residual = basis@coefficients-x.T
    pinverse = np.linalg.solve(basis.T@basis, basis.T)
    columns = []
    for j, factor in enumerate((u,.5*u**2,u**3/6,2*np.pi*u,np.pi*u**2,np.pi*u**3/3)):
        derivative = np.column_stack((np.zeros(n),factor*(cos if j<3 else -sin),
                                      factor*(sin if j<3 else cos)))
        response = derivative@coefficients
        jac = response-basis@(pinverse@response)-pinverse.T@(derivative.T@residual)
        columns.append(jac.T.ravel())
    jacobian = np.column_stack(columns)
    _, upper = np.linalg.qr(jacobian, mode='reduced')
    inverse_r = np.linalg.solve(upper, np.eye(6))
    inverse_gram = inverse_r@inverse_r.T
    variance = np.maximum(np.sum(residual**2,axis=0)/(n-3-6/channels),
                          np.finfo(float).eps**2*np.mean(x**2,axis=1))
    weighted = np.repeat(variance,n)[:,None]*jacobian
    covariance = inverse_gram@(jacobian.T@weighted)@inverse_gram
    curve = covariance[np.ix_([2,5],[2,5])]
    curve = .5*(curve+curve.T)
    return float(theta[[2,5]]@np.linalg.solve(curve,theta[[2,5]]))


def run():
    results = json.loads((OUT/'results.json').read_text())
    protocol, source, cached, analysis, _ = E._verify_inputs()
    raw_rows, _, physical, white = E._load_controls(source,cached,analysis)
    raw = {E._key(r):r for r in raw_rows}
    curved = {E._key(r['condition']):r for r in json.loads((TRANSPORT/'curvature_results.json').read_text())['rows']}
    for name,digest in results['provenance']['inputs_sha256'].items():
        assert E.M.G.O.C.sha256(E.ROOT/name)==digest,name
    for name,digest in results['provenance']['implementation_sha256'].items():
        assert E.M.G.O.C.sha256(SCRIPT/name)==digest,name
    assert len(results['rows'])==len(results['frozen_carrier_rows'])==48
    assert {E._key(r['condition']) for r in results['rows']}==set(curved)
    deltas=[]
    with np.load(TRANSPORT/'traces.npz') as traces:
        for rows,arm in ((results['rows'],'physical'),(results['frozen_carrier_rows'],'frozen_carrier')):
            for record in rows:
                key=E._key(record['condition']);original=raw[key]
                x,measurement=E._prepare(original,physical,white)
                if arm=='physical':
                    assert measurement['fit_signal_sha256']==record['measurement']['fit_signal_sha256']
                else:
                    draw=white[original['noise_index'],:,:128]
                    noise=np.zeros_like(draw) if original['noise']=='clean' else draw*(original['absolute_noise_rms_m']/np.sqrt(np.mean(draw**2)))
                    x=np.clip((traces[f'w{original["wave_index"]}_frozen_carrier'].T+noise-np.array(measurement['calibration_offset'])[:,None])/
                        np.array(measurement['calibration_scale'])[:,None],-50,50).astype(np.float32).astype(float)
                selection=record['carrier_selection']
                statistic=analytic_statistic(x,source['dt_seconds'],curved[key]['fits'][arm]['fit'])
                assert selection['status'] in ('curvature_supported','curvature_unsupported'),selection
                delta=abs(statistic-selection['statistic'])/max(1.,abs(statistic))
                assert delta<1e-4,(key,arm,delta)
                expected='curved' if statistic>2*np.log(x.size) else 'linear'
                assert selection['chosen']==expected,(key,arm)
                deltas.append(delta)
                fit=record['varying_fit'] if arm=='physical' else record['fit']
                score=record['varying_end_score'] if arm=='physical' else record['score']
                truth=original['references']['end']
                ae=abs(fit['alpha_end']-truth['alpha']);fe=abs(fit['frequency_end']/truth['frequency']-1)
                assert score['end_pole_matches_reference']==(ae<=max(1.,.1*abs(truth['alpha'])) and fe<=.05)
                assert score['growth_sign_correct']==(np.sign(fit['alpha_end'])==np.sign(truth['alpha']))
                assert abs(score['alpha_absolute_error_per_second']-ae)<1e-10
    rows=results['rows'];gates=results['critical_pass_checks']['checks']
    matched=sum(r['varying_end_score']['end_pole_matches_reference'] for r in rows)
    stationary=lambda noise:[r for r in rows if r['condition']['noise']==noise and r['condition']['ratio_rate_per_second']==0]
    noisy_median=float(np.median([r['varying_end_score']['alpha_absolute_error_per_second'] for r in stationary('iid1')]))
    clean_max=max(r['varying_end_score']['alpha_absolute_error_per_second'] for r in stationary('clean'))
    assert len(stationary('iid1'))==len(stationary('clean'))==8
    independent=dict(
        all48_critical_fits_successful_and_interior=all(r['varying_fit']['success'] and not r['varying_fit']['bound_hits'] for r in rows),
        all48_end_poles_match=matched==48,
        all48_end_growth_signs_correct=all(r['varying_end_score']['growth_sign_correct'] for r in rows),
        clean_stationary_max_alpha_error_per_second=clean_max<=.001,
        noisy_stationary_median_alpha_error_allowance_over_esprit_per_second=noisy_median<=gates['noisy_stationary_median_alpha_error_allowance_over_esprit_per_second']['threshold'])
    ramp_pass=True
    for noise in ('clean','iid1'):
        group=[r for r in rows if r['condition']['noise']==noise and r['condition']['ratio_rate_per_second']!=0]
        mae=float(np.mean([r['varying_end_score']['alpha_absolute_error_per_second'] for r in group]))
        sm=float(np.mean([r['stationary_end_score']['alpha_absolute_error_per_second'] for r in group]))
        lm=float(np.mean([r['archived_ls_esprit']['nearest_end_frequency']['end_reference_score']['alpha_absolute_error_per_second'] for r in group]))
        ramp_pass &= len(group)==16 and mae<sm and mae<lm
    independent['ramping_alpha_mae']=bool(ramp_pass)
    assert independent==results['critical_pass_checks']['booleans']
    assert results['progress']['percentage']==100*sum(independent.values())/6
    replays=[]
    for design,noise,rate in (('D000042','iid1',0.),('D000007','clean',.15625)):
        record=next(r for r in rows if r['condition']['design']==design and r['condition']['noise']==noise
                    and r['condition']['ratio_rate_per_second']==rate and r['condition']['end_ratio']==.99)
        x,_=E._prepare(raw[E._key(record['condition'])],physical,white)
        live=fit_noise_aware(x,source['dt_seconds'],settings=protocol['solver'])
        assert live['selection']['chosen']==record['carrier_selection']['chosen']
        assert abs(live['fit']['alpha_end']-record['varying_fit']['alpha_end'])<1e-8
        replays.append(dict(condition=record['condition'],chosen=live['selection']['chosen'],alpha_end=live['fit']['alpha_end']))
    verification=dict(result_sha256=E.M.G.O.C.sha256(OUT/'results.json'),verification_source_sha256=E.M.G.O.C.sha256(Path(__file__)),
        analytic_jacobian_qr_checks=len(deltas),maximum_relative_statistic_difference=max(deltas),
        independent_gate_checks=independent,progress=results['progress'],live_pinned_fitter_replays=replays,
        limitation='Independent numerical verification of the same model/data, not independent physical-model validation')
    E.M.G.O.C.save_json(OUT/'verification.json',E.M.clean_json(verification))
    print('INDEPENDENT_NOISE_AWARE_VERIFICATION',json.dumps(verification),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
