"""TRAIN-only diagnostic matrix; frozen O1 results are read-only inputs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits
import gpu_execution as G
import pulse_aware_probe as P
from cuda_backend import CudaPlate, CudaInference, estimate_batch, tensor, least_squares, advance
from mechanics.p4_margin_estimation.design_sampler import DesignSample

OUT = G.O.SCRIPT / 'diagnostic_matrix_20261007'
PROTOCOL = json.loads((OUT / 'protocol.json').read_text())


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def save(name, value):
    G.O.C.save_json(OUT / name, clean_json(value))


def variant(signal, dt=.001, maximum=8, direct=False, dc=False, lag=64):
    """One-variable LS-ESPRIT variants; preserve cap-before-transformation."""
    if maximum == 8 and not direct and not dc and lag == 64:
        return estimate_batch([signal], dt)[0]
    source = np.asarray(signal, float)
    hits = np.flatnonzero(np.any(abs(source) >= 50., axis=0))
    n = int(hits[0]) if len(hits) else source.shape[1]
    empty = dict(alpha=np.empty(0), frequency=np.empty(0), energy_fraction=np.empty(0),
                 relative_fit_error=1., usable_fraction=n/source.shape[1])
    if n < max(32, 2*maximum+4) or not np.isfinite(source[:, :n]).all():
        return empty
    x = tensor(source[:, :n])
    if dc:
        x = x-x.mean(dim=1, keepdim=True)
    if float(x.abs().max()) == 0:
        return empty
    x = x/x.abs().max()
    width = min(lag, n//2)
    h = x.unfold(1, width, 1).reshape(-1, width)
    if direct:
        _, s, vh = torch.linalg.svd(h, full_matrices=False, driver='gesvd')
        rank = min(2*maximum+1, width-1, int((s > s[0]*1e-6).sum()))
        space = vh.T[:, :rank]
    else:
        values, vectors = torch.linalg.eigh(h.T@h)
        rank = min(2*maximum+1, width-1, int((values > values[-1]*1e-12).sum()))
        space = vectors[:, -rank:] if rank else vectors[:, :0]
    if rank < 2:
        return empty
    roots = torch.linalg.eigvals(least_squares(space[:-1], space[1:]))
    roots = roots[(roots.imag > 1e-8) & (roots.abs() > 1e-12)]
    if not len(roots):
        return empty
    alpha, frequency = torch.log(roots.abs())/dt, torch.angle(roots)/(2*np.pi*dt)
    t = torch.arange(n, device='cuda', dtype=torch.float64)*dt
    growth = alpha[:, None]*t
    envelope = torch.exp(growth-growth.amax(dim=1, keepdim=True))
    phase = 2*np.pi*frequency[:, None]*t
    basis = torch.cat((torch.ones((n, 1), device='cuda', dtype=torch.float64),
                       (envelope*torch.cos(phase)).T, (envelope*torch.sin(phase)).T), dim=1)
    coefficients = least_squares(basis, x.T)
    count = len(roots)
    components = (basis[:, 1:1+count, None]*coefficients[None, 1:1+count]
                  + basis[:, 1+count:, None]*coefficients[None, 1+count:])
    energies = (components**2).sum(dim=(0, 2))
    centered = ((x-x.mean(dim=1, keepdim=True))**2).sum()
    if float(centered) <= torch.finfo(torch.float64).eps*float((x**2).sum()) or not bool(torch.isfinite(energies).all()) or float(energies.sum()) <= 0:
        return empty
    return dict(alpha=alpha.cpu().numpy(), frequency=frequency.cpu().numpy(),
                energy_fraction=(energies/energies.sum()).cpu().numpy(),
                relative_fit_error=float(((basis@coefficients-x.T)**2).sum()/centered),
                usable_fraction=n/source.shape[1])


def eigensystem(plate, ratio):
    roots, vectors = torch.linalg.eig(tensor(plate.matrix(ratio)))
    q = (tensor(plate.basis[:, :plate.count]).to(torch.complex128)
         @ (vectors[:plate.count]/tensor(plate.omega[:plate.count, None])))
    participation = G.O.TP._transverse_participation(q.cpu().numpy(), len(plate.omega))
    visible = (roots.imag.cpu().numpy() > G.O.TP.TRANSIENT_FILTER.omega_min) & (participation >= G.O.TP.TRANSIENT_FILTER.eta_w_min)
    indices = np.flatnonzero(visible)
    if not len(indices):
        raise RuntimeError('No physically transverse oscillatory reference modes')
    critical = int(indices[np.argmax(roots.real.cpu().numpy()[indices])])
    return roots, vectors, critical


def truth(plate, ratio):
    roots, _, critical = eigensystem(plate, float(ratio))
    pole = complex(roots[critical].item())
    return dict(alpha=pole.real, frequency=pole.imag/(2*np.pi))


def quality(mode, reference):
    a, f = mode['alpha'], mode['frequency']
    result = dict(identified=bool(len(a)), recovered=False, modes=len(a),
                  false_positive_growth=bool(len(a) and reference['alpha'] < 0 and np.max(a) > 0),
                  alpha_error=None, frequency_relative_error=None,
                  fit_error=mode['relative_fit_error'], usable_fraction=mode['usable_fraction'])
    if len(a):
        k = int(np.argmin(abs(f-reference['frequency'])))
        ae = abs(a[k]-reference['alpha'])
        fe = abs(f[k]-reference['frequency'])/reference['frequency']
        result.update(alpha_error=float(ae), frequency_relative_error=float(fe),
                      estimated_alpha=float(a[k]), estimated_frequency=float(f[k]),
                      recovered=bool(fe <= .05 and ae <= max(1., .1*abs(reference['alpha']))))
    return result


def oracle_skill(physical, measured):
    p = physical-physical.mean(axis=1, keepdims=True)
    m = measured-measured.mean(axis=1, keepdims=True)
    pscale, mscale = float(abs(p).max()), float(abs(m).max())
    if pscale == 0 or mscale == 0:
        return 0.
    p, m = p/pscale, m/mscale
    pp, mm = float((p*p).sum()), float((m*m).sum())
    # Diagnostic best scalar fit of the exact physical waveform; never a gate.
    return 0. if pp == 0 or mm == 0 else float((p*m).sum()**2/(pp*mm))


def normalize(raw, noise, cast=True):
    prefix = raw[:, :51]+noise[:, :51]
    offset = prefix.mean(axis=1)
    scale = np.maximum(np.sqrt(((prefix-offset[:, None])**2).mean(axis=1)), 1e-8)
    signal = np.clip((raw+noise-offset[:, None])/scale[:, None], -50., 50.)
    return (signal.astype(np.float32) if cast else signal), offset, scale


def features(signal, mode, d, engine, dt=.001):
    row = np.r_[G.O.B.base_feature_row(signal, dt, mode, engine.ridge),
                G.O.B.modal_spatial_row(mode, d['pairs']),
                G.O.B.modal_spatial_addition_row(mode, d['pairs'])]
    return row


def noises(raw, seed):
    rng = np.random.default_rng(seed)
    white = rng.normal(size=raw.shape)
    shared = rng.normal(size=(1, raw.shape[1]))
    correlated = np.sqrt(.2)*white+np.sqrt(.8)*shared
    # Match realized marginal RMS per channel, not percentage of decaying response.
    correlated *= np.sqrt((white**2).mean(axis=1, keepdims=True)/(correlated**2).mean(axis=1, keepdims=True))
    sigma = float(np.sqrt(((raw[:, :51]-raw[:, :51].mean(axis=1, keepdims=True))**2).mean()))
    return {'clean': np.zeros_like(raw), 'iid1': .01*sigma*white,
            'correlated1': .01*sigma*correlated, 'iid5': .05*sigma*white,
            'correlated5': .05*sigma*correlated}, sigma


def context(d, fp, case):
    with np.load(G.O.OUT/f"trajectories_{case['design']}.npz") as z:
        identity = json.loads(str(z['identity']))
    assert identity['fingerprint'] == fp
    audit = json.loads((G.O.OUT/f"audit_{case['design']}.json").read_text())
    assert audit['acquisition_supported'] and audit['fingerprint'] == fp
    plate = G.O.TP.PlateDynamics(DesignSample(**identity['design']), identity['ucrit'], G.sensor_yx(d))
    return identity, plate


def cached(d, fp, engine, smoke):
    result = dict(information=[], aperture=[], amplitude=[], conditioning=[], order=[], contract=[], ambiguity=[], noise_floor=[], detector=[])
    reference_cache = {}
    def reference(plate, profile, end):
        ratio = float(G.O.TP.ratio_at(profile, end))
        key = (plate.design.design_id, ratio)
        if key not in reference_cache:
            reference_cache[key] = truth(plate, ratio)
        return reference_cache[key]
    for case in PROTOCOL['cases'][:1] if smoke else PROTOCOL['cases']:
        _, plate = context(d, fp, case)
        rawsets = {}
        for profile, duration in PROTOCOL['profiles'].items():
            if smoke and profile != 'safe_dwell':
                continue
            for forcing in PROTOCOL['forcing']:
                rawpath = P.OUT/f"raw_{case['design']}_{profile}_{forcing}_rep{case['replicate']}.npz"
                with np.load(rawpath) as z:
                    raw = z['response'].copy()
                    assert z['valid'].all()
                rawsets[(profile, forcing)] = raw
                _, archived = P.archive_rows(case['design'], profile, forcing, case['replicate'], duration)
                original_count = int(archived['clean']['descriptor']['duration']*1000)+1
                fixture = np.zeros((raw.shape[0], original_count)); fixture[:, :51] = raw[:, :51]
                command = np.array([G.O.TP.physical_pulse(t/1000) != 0 for t in range(raw.shape[1])]) if forcing == 'intermittent_physical_pulses' else np.zeros(raw.shape[1], bool)
                for noise in P.PROTOCOL['noise']:
                    injected, _ = G.O.TP.continuous_noise(fixture, profile, forcing, case['replicate'], int(case['design'][1:]), noise)
                    injected = injected[:, :raw.shape[1]]
                    measured, _, scale = normalize(raw, injected)
                    with np.load(P.OUT/f"gate_{case['design']}_{profile}_{forcing}_rep{case['replicate']}_{noise}.npz") as z:
                        gate = {key: z[key].copy() for key in z.files}
                    for j, t in enumerate(gate['times']):
                        end = int(round(t*1000))+1
                        physical = raw[:, end-128:end]/scale[:, None]
                        observation = (raw[:, end-128:end]+injected[:, end-128:end])/scale[:, None]
                        snr = float(np.sqrt(((physical-physical.mean(axis=1, keepdims=True))**2).mean()))
                        nrms = float(np.sqrt(((injected[:, end-128:end]/scale[:, None])**2).mean()))
                        margin = 1-float(G.O.TP.ratio_at(profile, t))
                        result['information'].append(dict(design=case['design'], profile=profile, forcing=forcing, noise=noise,
                            time=t, pulse_age=t%2-1 if t%2 >= 1 else t%2+1,
                            margin=margin, snr=snr/nrms if nrms else None,
                            oracle_skill=oracle_skill(physical, observation), baseline_available=gate['baseline_available'][j],
                            guard=gate['pulse_guard'][j], predictive=gate['predictive_guard'][j], forecast_skills=gate['forecast_skills'][j],
                            prediction_error=float(gate['point'][j]-margin) if np.isfinite(gate['point'][j]) else None))
                        if noise == 'clean' and gate['predictive_guard'][j] and not smoke and sum(r['design'] == case['design'] and r['profile'] == profile for r in result['ambiguity']) < 6:
                            signal = measured[:, end-512:end]
                            mode = estimate_batch([signal])[0]
                            ref = reference(plate, profile, t)
                            if quality(mode, ref)['recovered']:
                                q = features(signal, mode, d, engine)
                                point = float(engine.predict(q[None])[0])
                                mean, std, active = [v.cpu().numpy() for v in engine.models[1][1:]]
                                fit = np.isin(d['ids'], G.O.partitions(d['ids'])['fit']) & (d['ids'] != case['design'])
                                train = (d['train'][0, fit]-mean)/std
                                query = (q-mean)/std; train[:, ~active] = 0; query[~active] = 0
                                distance = np.sqrt(((train-query)**2).sum(axis=1)/int(active.sum()))
                                order = np.argsort(distance)[:7]
                                perturb = (d['train'][1, fit]-d['train'][0, fit])/std; perturb[:, ~active] = 0
                                result['ambiguity'].append(dict(design=case['design'], profile=profile, time=t, truth=margin, prediction=point,
                                    max_abs_standardized=float(abs(query[active]).max()),
                                    outside_training_range=int(((q < d['train'][:, fit].min(axis=(0, 1))) | (q > d['train'][:, fit].max(axis=(0, 1))))[active].sum()),
                                    nearest_distances=distance[order], neighbor_margins=d['y'][fit][order], neighbor_designs=d['ids'][fit][order],
                                    median_same_clip_noise_distance=float(np.median(np.sqrt((perturb**2).sum(axis=1)/int(active.sum()))))))
                noise_map, sigma = noises(raw, int(case['design'][1:])+20261007)
                for pulse_start in (1., 3., 5., 7., 9.):
                    if pulse_start+.512 > duration:
                        continue
                    if smoke and pulse_start != 3.:
                        continue
                    pulse_end = pulse_start+.020
                    for placement in ('start_aligned', 'end_aligned'):
                        for length in PROTOCOL['apertures']:
                            stop = int(round((pulse_end+length/1000 if placement == 'start_aligned' else pulse_end+.512)*1000))+1
                            start = stop-length
                            free = not command[start:stop].any()
                            ref = reference(plate, profile, (stop-1)/1000)
                            for noise, injected in noise_map.items():
                                measured, _, scale = normalize(raw, injected)
                                window = measured[:, start:stop]
                                mode = estimate_batch([window])[0]
                                result['aperture'].append(dict(design=case['design'], profile=profile, forcing=forcing, noise=noise,
                                    placement=placement, samples=length, time=(stop-1)/1000, command_free=free,
                                    true_alpha=ref['alpha'], true_frequency=ref['frequency'], physical_noise_reference_m=sigma,
                                    oracle_skill=oracle_skill(raw[:, start:stop]/scale[:, None], (raw[:, start:stop]+injected[:, start:stop])/scale[:, None]),
                                    pre_crossing=profile == 'fast_cross' and (stop-1)/1000 <= archived['clean']['descriptor']['crossing']-1,
                                    **quality(mode, ref)))
                if forcing == 'intermittent_physical_pulses':
                    for endtime in (3.011, 3.127, 3.255, 3.611, 4.611):
                        if endtime > duration:
                            continue
                        end = int(round(endtime*1000))+1
                        ref = reference(plate, profile, endtime)
                        for noise in ('clean', 'iid1', 'correlated5'):
                            signal64, _, _ = normalize(raw, noise_map[noise], cast=False)
                            for name, signal, opts in [('float32', signal64.astype(np.float32), {}), ('float64', signal64, {}),
                                                       ('dc_removed', signal64.astype(np.float32), {'dc': True}),
                                                       ('direct_svd', signal64.astype(np.float32), {'direct': True})]:
                                mode = variant(signal[:, end-512:end], **opts)
                                result['conditioning'].append(dict(design=case['design'], profile=profile, noise=noise, time=endtime,
                                    variant=name, **quality(mode, ref)))
                            for maximum in PROTOCOL['maximum_mode_pairs']:
                                mode = variant(signal64.astype(np.float32)[:, end-512:end], maximum=maximum)
                                result['order'].append(dict(design=case['design'], profile=profile, noise=noise, time=endtime,
                                    maximum=maximum, **quality(mode, ref)))
                    # Full-aperture, fixed physical floor sweep at early/late free decay.
                    for t in (3.532, 3.611, 4.611):
                        end = int(round(t*1000))+1; ref = reference(plate, profile, t)
                        for level in PROTOCOL['noise_floor_levels']:
                            for family in ('iid', 'correlated'):
                                injected = noise_map[family+'1']*(level/.01)
                                measured, _, _ = normalize(raw, injected)
                                result['noise_floor'].append(dict(design=case['design'], profile=profile, time=t, family=family,
                                    level=level, absolute_noise_rms_m=float(np.sqrt((injected**2).mean())),
                                    **quality(estimate_batch([measured[:, end-512:end]])[0], ref)))
                # Measurement-only detector; raw simulator truth is used only to score detections.
                for noise, injected in noise_map.items():
                    signal = raw+injected
                    increments = np.diff(signal, axis=1, prepend=signal[:, :1])
                    energy = (increments**2).mean(axis=0); sums = np.r_[0., np.cumsum(energy)]
                    detections = []; last = -1000; previously_high = False
                    for end in range(120, len(energy)):
                        recent = (sums[end+1]-sums[end-19])/20
                        background = (sums[end-19]-sums[end-119])/100
                        high = recent > 4*max(background, 1e-30)
                        if high and not previously_high and end-last >= 100:
                            detections.append(end/1000); last = end
                        previously_high = high
                    onsets = np.flatnonzero(command & ~np.r_[False, command[:-1]])/1000
                    delays = [min([t-onset for t in detections if 0 <= t-onset <= .05], default=None) for onset in onsets]
                    false = sum(not any(0 <= t-onset <= .05 for onset in onsets) for t in detections)
                    result['detector'].append(dict(design=case['design'], profile=profile, forcing=forcing, noise=noise,
                        detections=detections, command_onsets=onsets, delays=delays, false_detections=false,
                        missed_onsets=sum(v is None for v in delays)))
                print('MATRIX_CACHE_COMPLETE', case['design'], profile, forcing, flush=True)
            if (profile, 'initial_only') not in rawsets:
                continue
            initial = rawsets[(profile, 'initial_only')]; forced = rawsets[(profile, 'intermittent_physical_pulses')]
            contribution = forced-initial
            for amplitude in PROTOCOL['pulse_amplitudes']:
                raw = initial+amplitude*contribution
                # Freeze original calibration and noise; pulse amplitude cannot change leading prefix.
                np.testing.assert_array_equal(raw[:, :51], initial[:, :51])
                noise_map, _ = noises(initial, int(case['design'][1:])+20261007)
                for endtime in (3.011, 3.111, 3.611):
                    end = int(round(endtime*1000))+1; ref = reference(plate, profile, endtime)
                    for noise in ('clean', 'iid1'):
                        signal, _, _ = normalize(raw, noise_map[noise]); window = signal[:, end-512:end]
                        mode = estimate_batch([window])[0]
                        point = float(engine.predict(features(window, mode, d, engine)[None])[0]) if len(mode['alpha']) else None
                        result['amplitude'].append(dict(design=case['design'], profile=profile, amplitude=amplitude,
                            noise=noise, time=endtime, cap_hits=int((abs(window) >= 50.).sum()), prediction=point, **quality(mode, ref)))
            # Explicit noise-only negatives retain the real acquisition calibration prefix.
            quiet = np.zeros_like(initial); quiet[:, :51] = initial[:, :51]
            ref = reference(plate, profile, 4.611); end = 4612
            for noise in ('iid1', 'correlated5'):
                signal, _, _ = normalize(quiet, noise_map[noise])
                window = signal[:, end-512:end]
                for name, opts in [('float32', {}), ('dc_removed', {'dc': True}), ('direct_svd', {'direct': True})]:
                    mode = variant(window, **opts)
                    result['conditioning'].append(dict(design=case['design'], profile=profile, noise=noise, time=4.611,
                        control='noise_only', variant=name, false_physical_estimate=bool(len(mode['alpha'])),
                        **{**quality(mode, ref), 'recovered': False}))
                for maximum in PROTOCOL['maximum_mode_pairs']:
                    mode = variant(window, maximum=maximum)
                    result['order'].append(dict(design=case['design'], profile=profile, noise=noise, time=4.611,
                        control='noise_only', maximum=maximum, false_physical_estimate=bool(len(mode['alpha'])),
                        **{**quality(mode, ref), 'recovered': False}))
        # Same physical clip through clip and trajectory paths, with the same calibration samples.
        with np.load(G.O.ROOT/'p4_sensor_candidates'/f"{case['design']}.npz") as z:
            rawclip = z['raw'][0].copy(); source_index = int(z['source_indices'][0])
        dtmeta = np.load(G.O.ROOT/'p4_dataset_saturation/metadata_arrays.npz')
        dt = float(dtmeta['dts'][source_index]); dtmeta.close()
        clip, cm = G.O.B.extract(rawclip, dt, d['layout'], d['pairs'], engine.ridge)
        signal, _, _ = G.O.causal_observation(rawclip[d['layout']], np.zeros_like(rawclip[d['layout']]))
        modes = estimate_batch([signal], dt)[0]; replay = features(signal, modes, d, engine, dt=dt)
        np.testing.assert_allclose(replay, clip, rtol=1e-6, atol=1e-6)
        canonical = G.O.B.causal_calibration_normalize(rawclip, calibration_samples=51)[d['layout']]
        canonical = np.clip(canonical, -50, 50).astype(np.float32)
        np.testing.assert_array_equal(canonical, signal)
        result['contract'].append(dict(design=case['design'], source_index=source_index, clip_dt=dt, trajectory_dt=.001,
            normalized_samples_exact=True, maximum_feature_difference=float(abs(replay-clip).max()), feature_columns=len(clip),
            note='Identical leading calibration and segment match. Recalibrating each later trajectory clip would change the acquisition contract.'))
    return result

def propagate(plate, times, initial, ratio, loads):
    """CUDA exact linear step with piecewise-linear physical input, no CPU propagation."""
    dt = float(times[1]-times[0]); block = max(1, int(round(.002/dt)))
    begins = np.arange(0, len(times)-1, block); ends = np.minimum(begins+block, len(times)-1)
    ratios = ratio(.5*(times[begins]+times[ends]))
    unique, indices = np.unique(ratios, return_inverse=True)
    transitions=[]; lefts=[]; rights=[]
    for value in unique:
        a = tensor(plate.matrix(float(value)))
        augmented = torch.zeros((len(initial)+2, len(initial)+2), device='cuda', dtype=torch.float64)
        augmented[:-2, :-2] = a; augmented[:-2, -2] = tensor(plate.reduced['force']); augmented[-2, -1] = 1.
        step = torch.linalg.matrix_exp(augmented*dt)
        transitions.append(step[:-2, :-2]); rights.append(step[:-2, -1]/dt); lefts.append(step[:-2, -2]-rights[-1])
    response, valid, state, terminated = advance(tensor(initial), torch.stack(transitions), torch.stack(lefts),
        torch.stack(rights), indices.tolist(), [float(v) for v in loads], tensor(plate.reduced['projection']), block)
    if terminated or not bool(valid.all()):
        raise RuntimeError('Physical control exceeded numerical dynamic range')
    return response.cpu().numpy(), state.cpu().numpy()


def physical(d, fp, smoke):
    output = dict(excitation=[], nonstationarity=[], sampling=[], amplitude_replay=[])
    cases = PROTOCOL['cases'][:1] if smoke else PROTOCOL['cases']+[dict(design=did, replicate=0) for did in PROTOCOL['confirmation_designs']]
    for case in cases:
        _, plate = context(d, fp, case)
        initial = plate.initial(case['replicate'])[0]
        for ratio in PROTOCOL['fixed_ratios']:
            roots, vectors, critical = eigensystem(plate, ratio); ref = truth(plate, ratio)
            coefficients = torch.linalg.solve(vectors, tensor(initial).to(torch.complex128))
            weak = coefficients.clone(); weak[critical] *= .01
            partner = int(torch.argmin(abs(roots-roots[critical].conj())))
            weak[partner] *= .01
            weak_initial = (vectors@weak).real.cpu().numpy()
            critical_initial = vectors[:, critical].real.cpu().numpy()
            projection = plate.reduced['projection']
            critical_initial *= max(float(np.sqrt(np.mean((projection@initial)**2))), 1e-30)/max(float(np.sqrt(np.mean((projection@critical_initial)**2))), 1e-30)
            times = np.arange(513)/1000
            reference_scale = max(float(np.sqrt(np.mean((projection@initial)**2))), 1e-30)
            for excitation, state in [('ordinary', initial), ('weak_critical', weak_initial), ('critical_only_oracle', critical_initial)]:
                raw, _ = propagate(plate, times, state, lambda t: np.full_like(t, ratio), np.zeros(len(times)))
                # Fixed acquisition reference across excitation arms; no SNR renormalization.
                white = np.random.default_rng(20261007+int(case['design'][1:])).normal(size=raw.shape)
                noise = .01*reference_scale*white
                for family, injected in [('clean', np.zeros_like(raw)), ('iid1', noise)]:
                    signal, _, _ = normalize(raw, injected)
                    for length in PROTOCOL['apertures']:
                        output['excitation'].append(dict(design=case['design'], confirmation=case['design'] in PROTOCOL['confirmation_designs'], ratio=ratio,
                            excitation=excitation, noise=family, samples=length, true_alpha=ref['alpha'], true_frequency=ref['frequency'],
                            initial_sensor_rms_m=float(np.sqrt(np.mean((projection@state)**2))),
                            critical_sensor_projection_gain=float(torch.linalg.vector_norm(tensor(projection).to(torch.complex128)@vectors[:, critical])),
                            ordinary_critical_modal_coefficient=float(abs(coefficients[critical])),
                            state_norm=float(np.linalg.norm(state)), **quality(estimate_batch([signal[:, :length]])[0], ref)))
            if smoke:
                break
        # Same state/input/noise under a stationary counterpart versus a 256ms velocity ramp.
        times = np.arange(257)/1000
        white = np.random.default_rng(20261008+int(case['design'][1:])).normal(size=(len(d['layout']), len(times)))
        for condition, ratio in [('fixed', lambda t: np.full_like(t, .99)), ('ramp', lambda t: .97+.04*t/.256)]:
            raw, _ = propagate(plate, times, initial, ratio, np.zeros(len(times)))
            injected = .01*max(float(np.sqrt(np.mean((plate.reduced['projection']@initial)**2))), 1e-30)*white
            for noise, value in [('clean', np.zeros_like(raw)), ('iid1', injected)]:
                signal, _, _ = normalize(raw, value)
                output['nonstationarity'].append(dict(design=case['design'], condition=condition, noise=noise,
                    initial_ratio=float(ratio(np.array([0.]))[0]), midpoint_ratio=float(ratio(np.array([.128]))[0]), end_ratio=float(ratio(np.array([.256]))[0]),
                    end_reference=truth(plate, float(ratio(np.array([.256]))[0])), midpoint_reference=truth(plate, float(ratio(np.array([.128]))[0])),
                    **quality(estimate_batch([signal])[0], truth(plate, float(ratio(np.array([.256]))[0])))))
        # Higher rate is newly simulated, not interpolated. Filter delay excluded from any warning claim.
        raw_rates={}
        for rate in (1000, 4000):
            times = np.arange(int(.512*rate)+1)/rate
            loads = np.array([G.O.TP.physical_pulse(float(t+1)) for t in times])
            raw, _ = propagate(plate, times, initial, lambda t: np.full_like(t, .99), loads)
            raw_rates[rate] = raw
        down = resample_poly(raw_rates[4000], 1, 4, axis=1)
        ref = truth(plate, .99)
        for name, raw, rate in [('native1k', raw_rates[1000], 1000), ('native4k', raw_rates[4000], 4000), ('antialiased1k_offline', down, 1000)]:
            # Same 128/256ms post-input intervals and 64ms Hankel lag at both native rates.
            signal = np.clip(raw/np.maximum(np.sqrt((raw[:, :int(.051*rate)]**2).mean(axis=1, keepdims=True)), 1e-8), -50, 50).astype(np.float32)
            for duration in (.128, .256):
                begin=int(.020*rate); end=begin+int(duration*rate)
                output['sampling'].append(dict(design=case['design'], condition=name, rate=rate, duration=duration,
                    temporal_lag_ms=64, filter_delay_seconds=0 if name == 'native1k' else None,
                    limitation='Matched 64ms Hankel lag; offline zero-phase anti-aliasing is not causal latency evidence.',
                    **quality(variant(signal[:, begin:end], dt=1/rate, lag=int(.064*rate)), ref)))
        if case['design'] == PROTOCOL['cases'][0]['design']:
            # One physical replay validates amplitude superposition against an independently propagated input.
            times=np.arange(3501)/1000; ratio=lambda t: G.O.TP.ratio_at('safe_dwell', t)
            base, _=propagate(plate, times, initial, ratio, np.zeros(len(times)))
            unit, _=propagate(plate, times, initial, ratio, np.array([G.O.TP.physical_pulse(float(t)) for t in times]))
            doubled, _=propagate(plate, times, initial, ratio, np.array([2*G.O.TP.physical_pulse(float(t)) for t in times]))
            reconstructed=base+2*(unit-base)
            relative=float(np.linalg.norm(doubled-reconstructed)/np.linalg.norm(doubled))
            np.testing.assert_allclose(doubled, reconstructed, rtol=1e-9, atol=1e-16)
            output['amplitude_replay'].append(dict(design=case['design'], amplitude=2, relative_rms_error=relative))
        print('MATRIX_PHYSICS_COMPLETE', case['design'], flush=True)
    return output


def policy(fp):
    fit=json.loads((G.O.OUT/'trajectory_fit.json').read_text())
    calibration=json.loads((G.O.OUT/'trajectory_calibration.json').read_text())
    assert calibration['fingerprint'] == fp
    rows=G.O.load_trajectories(fit['partitions']['calibration'])
    safe=[r for r in rows if r['descriptor']['profile'] == 'safe_dwell']
    result=dict(attribution=[], candidate_audit=[])
    for row in safe:
        d=row['descriptor']; mask=row['available']; scores=row['score'][mask]; points=row['point'][mask]
        if not len(scores):
            continue
        phases=row['times'][mask]%2
        onset=((phases>=1)&(phases<=1.020+.512)) if d['forcing'] == 'intermittent_physical_pulses' else np.zeros(len(scores), bool)
        for category, chosen in [('onset_overlap', onset), ('onset_free', ~onset)]:
            for threshold in (calibration['thresholds']['episodes_per_hour_1.0'], calibration['thresholds']['legacy_clip1'], 350.):
                altered={**row, 'available': mask.copy()}; altered['available'][mask] &= chosen
                metric=G.O.row_metrics(altered, threshold)
                result['attribution'].append(dict(design=d['design'], noise=d['noise'], forcing=d['forcing'], category=category,
                    threshold=threshold, windows=int(chosen.sum()), margin_below_minus_one=int((points[chosen]<-1).sum()),
                    max_score=float(scores[chosen].max()) if chosen.any() else None, episodes=metric['episodes'],
                    alarm_active_fraction=metric['alarm_active_fraction'],
                    limitation='Category-isolated replay; original hysteresis holds alarms across unavailable updates. Episodes are not additive across categories.'))
    # Dense thresholds evaluated ONLY on calibration; no held labels or selection changes.
    thresholds=np.unique(np.r_[calibration['calibration']['candidate_thresholds'], np.linspace(0, 360, 181)])
    for threshold in thresholds:
        groups={}
        active=[]
        for row in safe:
            key=row['descriptor']['noise']+'/'+row['descriptor']['forcing']
            count,hours=groups.get(key,(0,0.))
            groups[key]=(count+len(G.O.episodes(row, float(threshold))), hours+row['descriptor']['duration']/3600)
            active.append(G.O.row_metrics(row, float(threshold))['alarm_active_fraction'])
        result['candidate_audit'].append(dict(threshold=threshold, maximum_episode_rate=max(n/h for n,h in groups.values()),
                                              mean_alarm_active_fraction=float(np.mean(active))))
    return result


def run(smoke, phase):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU numerical fallback')
    d, fp=G.inputs(); models=[]
    for width in (107,174):
        with np.load(G.O.OUT/f'trajectory_baseline{width}.npz') as z:
            assert json.loads(str(z['fingerprint'])) == fp
            models.append(G.O.C.cloudpickle.loads(z['predictor'].tobytes()))
    engine=CudaInference(models,d['pairs']); start=time.perf_counter()
    result=dict(protocol=PROTOCOL, source_sha256=G.O.C.sha256(Path(__file__)), backend=G.backend_signature(), fingerprint=fp)
    if phase in ('cached','all'):
        result['cached']=cached(d, fp, engine, smoke)
    if phase in ('physical','all'):
        result['physical']=physical(d, fp, smoke)
    if phase in ('policy','all') and not smoke:
        result['policy']=policy(fp)
    result['elapsed_seconds']=time.perf_counter()-start
    filename=('smoke_' if smoke else '')+phase+'_results.json'
    save(filename,result)
    print('DIAGNOSTIC_MATRIX_COMPLETE', filename, result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--phase',choices=('cached','physical','policy','all'),default='all')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.smoke,args.phase)
