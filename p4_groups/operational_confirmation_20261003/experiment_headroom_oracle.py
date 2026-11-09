"""Offline pole-estimation/branch-choice/headroom-conversion diagnostic.

Oracle values never enter an operational predictor. Completed source artifacts
are read-only; all unresolved rows stay in population accounting.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
OUT = SCRIPT / 'headroom_oracle_20261008'
SOURCE = SCRIPT / 'fitter_margin_comparison_20261008'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(archive, key):
    return json.loads(str(archive[key]))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)


def contract():
    p = json.loads((OUT / 'protocol.json').read_text())
    for name, expected in p['inputs'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'Pinned input changed: {name}')
    if 'implementation_sha256' in p and sha(__file__) != p['implementation_sha256']:
        raise RuntimeError('Frozen diagnostic implementation changed')
    return p


def source_data():
    with np.load(SOURCE / 'dataset.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ('truth', 'observed', 'fitter_observed', 'fitter_features', 'times')}
        data['records'] = scalar(z, 'records')
        data['fits'] = scalar(z, 'fit_records')
    return data


def replay(design=None):
    """Record the original CUDA state via an augmented observation matrix."""
    p = contract()
    import gpu_execution as G
    import diagnostic_matrix as M
    import temporal_probe as T
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('State replay requires CUDA, no CPU propagation fallback')
    d, fp = G.inputs()
    with np.load(SOURCE / 'raw_signals.npz', allow_pickle=False) as z:
        raw = z['physical'].copy()
        records = scalar(z, 'records')
    designs = sorted({r['design'] for r in records})
    for did in designs:
        if design is not None and did != design:
            continue
        path = OUT / f'states_{did}.npz'
        rows = [i for i, r in enumerate(records) if r['design'] == did]
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z['protocol_sha256']) != sha(OUT / 'protocol.json') or scalar(z, 'records') != [records[i] for i in rows]:
                    raise RuntimeError('State shard does not match frozen contract')
            print('ORACLE_STATE_REUSED', did, flush=True)
            continue
        identity, plate = M.context(d, fp, dict(design=did, replicate=0))
        observer = copy.copy(plate)
        observer.reduced = dict(plate.reduced)
        observer.reduced['projection'] = np.vstack((plate.reduced['projection'], np.eye(2 * plate.count)))
        states, errors = [], []
        for i in rows:
            r = records[i]
            _, response, _ = T.acquire(observer, r['schedule'], r['replicate'])
            sensor_response = response[:len(plate.sensor)]
            error = float(np.linalg.norm(sensor_response - raw[i]) / np.linalg.norm(raw[i]))
            if error > p['replay_relative_rms_tolerance']:
                raise RuntimeError(f'Original waveform replay failed {did}/{i}: {error}')
            states.append(response[len(plate.sensor):].T)
            errors.append(error)
        matrices = plate.reduced
        save_npz(path, states=np.stack(states), raw_indices=np.array(rows),
                 records=json.dumps([records[i] for i in rows]), identity=json.dumps(identity),
                 base=matrices['base'], stiffness=matrices['stiffness'], damping=matrices['damping'],
                 projection=matrices['projection'], physical_basis=plate.basis[:, :plate.count] / plate.omega[:plate.count],
                 structural_dofs=np.array(len(plate.omega)), replay_errors=np.array(errors),
                 protocol_sha256=np.array(sha(OUT / 'protocol.json')))
        print('ORACLE_STATE_REPLAYED', did, len(rows), 'max_relative_rms', max(errors), flush=True)


def pole_table(state_archive, ratios, p):
    """Exact eigensystems at requested sample ratios; extra points bound tracking steps."""
    import trajectory_physics as TP
    requested, reverse = np.unique(ratios, return_inverse=True)
    extra = []
    step = p['tracking']['max_ratio_step']
    for a, b in zip(requested[:-1], requested[1:]):
        if b - a > step:
            extra.extend(np.linspace(a, b, int(np.ceil((b-a)/step)) + 1)[1:-1])
    grid = np.unique(np.r_[requested, extra])
    requested_grid = np.searchsorted(grid, requested)
    sample_grid = requested_grid[reverse].reshape(ratios.shape)
    identity = scalar(state_archive, 'identity')
    rho = identity['design']['rho_air']
    sound = identity['design']['c_air']
    ucrit = identity['ucrit']
    basis = state_archive['physical_basis']
    projection = state_archive['projection']
    count = projection.shape[1] // 2
    samples = np.prod(ratios.shape)
    poles = np.empty((len(grid), count), complex)
    shapes = np.empty((len(grid), projection.shape[0], count), complex)
    eligible = np.zeros((len(grid), count), bool)
    quality = np.ones((len(grid), count), bool)
    pairs = np.empty((samples, projection.shape[0], count))
    flat_states = state_archive['states'].reshape(samples, 2*count)
    order = np.argsort(sample_grid.ravel(), kind='stable')
    sorted_grid = sample_grid.ravel()[order]
    starts = np.searchsorted(sorted_grid, np.arange(len(grid)), side='left')
    ends = np.searchsorted(sorted_grid, np.arange(len(grid)), side='right')
    max_residual = max_state_error = max_sensor_error = 0.
    previous = previous_vectors = None
    for k, ratio in enumerate(grid):
        velocity = ratio * ucrit
        mach = velocity / sound
        if mach < 2:
            raise RuntimeError('Oracle left original supported Mach domain')
        dynamic = rho * velocity**2 / np.sqrt(mach**2-1)
        damping = dynamic * (mach**2-2)/(mach**2-1)/velocity
        matrix = state_archive['base'] + dynamic*state_archive['stiffness'] + damping*state_archive['damping']
        roots, vectors = eig(matrix, check_finite=False)
        positive = np.flatnonzero(roots.imag > 1e-8)
        if len(positive) != count or np.any(abs(roots.imag) <= 1e-8):
            raise RuntimeError('Oscillatory-pair topology changed; cannot infer branch identity')
        r, v = roots[positive], vectors[:, positive]
        if previous is None:
            perm = np.argsort(r.imag)
        else:
            separation = abs(previous[:, None] - previous[None, :])
            np.fill_diagonal(separation, np.inf)
            separation = np.maximum(separation.min(axis=1), 1e-8)
            displacement = abs(previous[:, None]-r[None, :]) / separation[:, None]
            overlap = abs(previous_vectors.conj().T @ v) / (np.linalg.norm(previous_vectors, axis=0)[:, None]*np.linalg.norm(v, axis=0)[None, :])
            cost = displacement + 1 - np.minimum(overlap, 1)
            row, perm = linear_sum_assignment(cost)
            if not np.array_equal(row, np.arange(count)):
                raise RuntimeError('Non-complete pole assignment')
            for branch, column in enumerate(perm):
                alternatives = np.delete(cost[branch], column)
                quality[k, branch] = (overlap[branch, column] >= p['tracking']['minimum_overlap'] and
                    displacement[branch, column] <= p['tracking']['maximum_displacement_in_previous_nearest_pole_separation'] and
                    alternatives.min()-cost[branch, column] >= p['tracking']['minimum_assignment_runner_up_margin'])
        r, v = r[perm], v[:, perm]
        positive = positive[perm]
        poles[k] = r
        shapes[k] = projection @ v
        participation = TP._transverse_participation(basis @ v[:count], int(state_archive['structural_dofs']))
        eligible[k] = (r.imag > TP.TRANSIENT_FILTER.omega_min) & (participation >= TP.TRANSIENT_FILTER.eta_w_min)
        if not eligible[k].any():
            raise RuntimeError('No physically transverse critical candidate')
        residual = np.linalg.norm(matrix @ v - v*r[None, :], axis=0) / (np.linalg.norm(matrix)*np.linalg.norm(v, axis=0))
        max_residual = max(max_residual, float(residual.max()))
        quality[k] &= residual <= p['tracking']['maximum_relative_eigen_residual']
        positions = order[starts[k]:ends[k]]
        if len(positions):
            coefficients = solve(vectors, flat_states[positions].T, check_finite=False)
            reproduced_state = (vectors @ coefficients).T
            state_error = np.linalg.norm(reproduced_state-flat_states[positions]) / max(np.linalg.norm(flat_states[positions]), 1e-30)
            components = 2*np.real(shapes[k][:, :, None] * coefficients[positive][None, :, :])
            pairs[positions] = components.transpose(2, 0, 1)
            sensor = flat_states[positions] @ projection.T
            sensor_error = np.linalg.norm(pairs[positions].sum(axis=2)-sensor) / max(np.linalg.norm(sensor), 1e-30)
            max_state_error = max(max_state_error, float(state_error))
            max_sensor_error = max(max_sensor_error, float(sensor_error))
        previous, previous_vectors = r, v
    if max_state_error > 1e-9 or max_sensor_error > 1e-9:
        raise RuntimeError('Full eigensystem state/sensor reconstruction failed')
    dominant = np.argmax(np.where(eligible, poles.real, -np.inf), axis=1)
    return dict(grid=grid, indices=sample_grid, poles=poles, shapes=shapes, quality=quality,
                dominant=dominant, pairs=pairs.reshape(*ratios.shape, projection.shape[0], count),
                maximum_eigen_residual=max_residual, maximum_state_error=max_state_error,
                maximum_sensor_error=max_sensor_error, failed_tracking_edges=int((~quality).sum()))


def oracle_row(poles, residual):
    t = (np.arange(128)-63.5)*.001
    slopes = np.sum(t[:, None]*np.c_[poles.real, poles.imag/(2*np.pi)], axis=0) / np.sum(t*t)
    a, f = poles[-1].real, poles[-1].imag/(2*np.pi)
    return np.r_[a, f, slopes, a/(2*np.pi*f), residual]


def extract_design(did):
    p = contract()
    path = OUT / f'oracle_{did}.npz'
    if path.exists():
        raise RuntimeError(f'Completed oracle shard is immutable: {did}')
    data = source_data()
    import diagnostic_matrix as M
    with np.load(SOURCE / 'raw_signals.npz', allow_pickle=False) as z:
        raw = z['physical'].copy()
    with np.load(OUT / f'states_{did}.npz', allow_pickle=False) as s:
        if str(s['protocol_sha256']) != sha(OUT / 'protocol.json'):
            raise RuntimeError('State shard protocol changed')
        physical_records = scalar(s, 'records')
        times = np.arange(8001)*.001
        ratios = np.array([np.interp(times, r['schedule']['knots'], r['schedule']['ratios']) for r in physical_records])
        table = pole_table(s, ratios, p)
        raw_indices = s['raw_indices'].copy()
        state_hash = sha(OUT / f'states_{did}.npz')
        replay_max = float(s['replay_errors'].max())
    physical_index = {(r['design'], r['replicate'], r['profile']):i for i,r in enumerate(physical_records)}
    fits = {(r['record_index'],r['frame_index']):r['varying'] for r in data['fits']}
    record_indices = np.array([i for i,r in enumerate(data['records']) if r['design']==did])
    shape = (len(record_indices), len(data['times']))
    critical = np.full((*shape,6),np.nan)
    matched = np.full_like(critical,np.nan)
    valid_c = np.zeros(shape,bool)
    valid_b = np.zeros(shape,bool)
    critical_id = np.full(shape,-1,int)
    matched_id = np.full(shape,-1,int)
    matches_critical = np.zeros(shape,bool)
    switches = np.zeros(shape,int)
    support = np.full((*shape,3),np.nan)
    reason = np.full(shape,'source_unavailable',dtype='U40')
    match_score = np.full((*shape,2),np.nan)
    for local, ri in enumerate(record_indices):
        record = data['records'][ri]
        pi = physical_index[(did,record['replicate'],record['profile'])]
        physical = raw[raw_indices[pi]]
        seed=np.random.SeedSequence([20261010,int(did[1:]),record['replicate'],('cross','stable','retreat').index(record['profile'])])
        noises,_=M.noises(physical,seed)
        noise=noises[record['noise']]
        signal,_,scale=M.normalize(physical,noise)
        for frame in np.flatnonzero(data['observed'][ri] & data['fitter_observed'][ri]):
            end=128+64*frame; start=end-128
            ix=table['indices'][pi,start:end]
            cid=int(table['dominant'][ix[-1]])
            critical_id[local,frame]=cid
            switches[local,frame]=int(np.sum(np.diff(table['dominant'][ix])!=0))
            lo,hi=int(ix.min()),int(ix.max())
            if not table['quality'][lo+1:hi+1,cid].all():
                reason[local,frame]='critical_tracking_ambiguous'
                continue
            poles=table['poles'][ix]
            critical[local,frame]=oracle_row(poles[:,cid],data['fitter_features'][ri,frame,5])
            valid_c[local,frame]=True
            component=table['pairs'][pi,start:end,:,cid].T/scale[:,None]
            rest=(physical[:,start:end]/scale[:,None])-component
            injected=noise[:,start:end]/scale[:,None]
            energy=lambda x:float(np.sum((x-x.mean(axis=1,keepdims=True))**2))
            support[local,frame]=[energy(component),energy(rest),energy(injected)]
            fit=fits[(int(ri),int(frame))]
            parameters=fit['parameters']; t=(np.arange(128)-63.5)*.001
            envelope=np.exp(parameters['alpha_mid']*t+.5*parameters['alpha_slope']*t*t)
            phase=2*np.pi*(parameters['frequency_mid']*t+.5*parameters['frequency_slope']*t*t)
            basis=np.column_stack((np.ones(128),envelope*np.cos(phase),envelope*np.sin(phase)))
            coefficients,_,rank,_=np.linalg.lstsq(basis,signal[:,start:end].T,rcond=None)
            if rank!=3:
                raise RuntimeError('Archived fitter basis cannot be reconstructed')
            measured_shape=coefficients[1]-1j*coefficients[2]
            model_f=parameters['frequency_mid']+parameters['frequency_slope']*t
            frequency_error=np.sqrt(np.mean((poles.imag/(2*np.pi)-model_f[:,None])**2,axis=0))
            candidates=np.flatnonzero(frequency_error<=1/.127)
            physical_shapes=table['shapes'][ix[63]]/scale[:,None]
            norm=np.linalg.norm(measured_shape)*np.linalg.norm(physical_shapes,axis=0)
            overlap=np.divide(abs(measured_shape.conj()@physical_shapes)**2,norm**2,out=np.zeros_like(norm),where=norm>0)
            ranked=candidates[np.argsort(-overlap[candidates],kind='stable')]
            if not len(ranked):
                reason[local,frame]='no_temporal_correspondence'
                continue
            bid=int(ranked[0]); best=float(overlap[bid]); second=float(overlap[ranked[1]]) if len(ranked)>1 else 0.
            match_score[local,frame]=[best,second]
            if best<p['correspondence']['minimum_squared_overlap'] or best-second<p['correspondence']['minimum_best_minus_second_squared_overlap']:
                reason[local,frame]='spatial_correspondence_ambiguous'
                continue
            if not table['quality'][lo+1:hi+1,bid].all():
                reason[local,frame]='matched_tracking_ambiguous'
                continue
            matched[local,frame]=oracle_row(poles[:,bid],data['fitter_features'][ri,frame,5])
            valid_b[local,frame]=True; matched_id[local,frame]=bid
            matches_critical[local,frame]=bid==cid; reason[local,frame]='resolved'
    stats={k:table[k] for k in ('maximum_eigen_residual','maximum_state_error','maximum_sensor_error','failed_tracking_edges')}
    save_npz(path,record_indices=record_indices,critical=critical,matched=matched,valid_c=valid_c,valid_b=valid_b,
             critical_id=critical_id,matched_id=matched_id,matches_critical=matches_critical,switches=switches,
             support=support,reason=reason,match_score=match_score,stats=json.dumps(stats),
             replay_max=np.array(replay_max),state_sha256=np.array(state_hash),protocol_sha256=np.array(sha(OUT/'protocol.json')))
    print('ORACLE_EXTRACTED',did,'critical',int(valid_c.sum()),'matched',int(valid_b.sum()),stats,flush=True)
    return did


def extract(workers):
    p=contract()
    designs=sum(p['split'].values(),[])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(extract_design,did) for did in designs]
        for future in as_completed(futures):
            future.result()


def metric(y,pred,mask,ids):
    finite=mask & np.isfinite(pred)
    per_design={}
    for did in sorted(set(ids[mask.any(axis=1)].tolist())):
        selected=mask & (ids==did)[:,None]
        scored=selected & np.isfinite(pred)
        per_design[did]=dict(population=int(selected.sum()),scored=int(scored.sum()),unknown=int((selected&~np.isfinite(pred)).sum()),
            mae=float(np.abs(pred[scored]-y[scored]).mean()) if scored.any() else None)
    errors=[r['mae'] for r in per_design.values() if r['mae'] is not None]
    boundary=finite & (y>-.02)&(y<.05)
    sign=finite & (abs(y)>=.01)&(abs(y)<=.05)
    def rate(event,eligible):
        return dict(events=int((event&eligible).sum()),denominator=int(eligible.sum()),rate=float(event[eligible].mean()) if eligible.any() else None)
    return dict(population=int(mask.sum()),scored=int(finite.sum()),unknown=int((mask&~np.isfinite(pred)).sum()),
                equal_design_mae=float(np.mean(errors)) if errors else None,per_design=per_design,
                optimism=rate(pred-y>.02,boundary),
                stable_sign=rate(np.sign(pred)!=np.sign(y),sign&(y>0)),
                unstable_sign=rate(np.sign(pred)!=np.sign(y),sign&(y<0)),
                unstable_false_positive_point=rate(pred>0,finite&(y<0)))


def score():
    p=contract(); data=source_data()
    import fitter_comparison_score as F
    with np.load(SOURCE/'predictions.npz',allow_pickle=False) as z:
        names=scalar(z,'prediction_names'); frozen=z['predictions'].copy()
    shape=data['truth'].shape
    arrays={k:np.full((*shape,6),np.nan) for k in ('critical','matched')}
    masks={k:np.zeros(shape,bool) for k in ('valid_c','valid_b')}
    reasons=np.full(shape,'not_loaded',dtype='U40'); support=np.full((*shape,3),np.nan)
    matches=np.zeros(shape,bool); switches=np.zeros(shape,int); shard_hashes={}; shard_stats={}
    for did in sum(p['split'].values(),[]):
        path=OUT/f'oracle_{did}.npz'
        with np.load(path,allow_pickle=False) as z:
            rows=z['record_indices']
            if str(z['protocol_sha256'])!=sha(OUT/'protocol.json') or str(z['state_sha256'])!=sha(OUT/f'states_{did}.npz'):
                raise RuntimeError('Oracle shard provenance changed')
            for k in arrays: arrays[k][rows]=z[k]
            for k in masks: masks[k][rows]=z[k]
            reasons[rows]=z['reason']; support[rows]=z['support'];matches[rows]=z['matches_critical'];switches[rows]=z['switches']
            shard_stats[did]=scalar(z,'stats')
        shard_hashes[did]=sha(path)
    records=data['records']; ids=np.array([r['design'] for r in records])
    noises=np.array([r['noise'] for r in records]); splits=np.array([r['split'] for r in records])
    fresh=data['observed']&data['fitter_observed']
    if (masks['valid_b']&~masks['valid_c']).any() or (masks['valid_c']&~fresh).any():
        raise RuntimeError('Oracle masks escape original source population')
    families={'critical_contrast':masks['valid_c'],'correspondence_contrast':masks['valid_c']&masks['valid_b']}
    predictions={'frozen_ridge6':frozen[names.index('ridge6')], 'frozen_framewise':frozen[names.index('framewise')]}
    selections={}; statuses={}
    for family, common in families.items():
        fit=common & ((splits=='train')&np.isin(noises,('clean','iid1')))[:,None]
        if set(ids[np.any(fit,axis=1)].tolist())!=set(p['split']['train']):
            statuses[family]='unresolved: a proper training design has no common valid rows'
            continue
        statuses[family]='fitted'
        feature_sets={'A':data['fitter_features'],'C':arrays['critical']}
        if family=='correspondence_contrast':feature_sets['B']=arrays['matched']
        for arm,x in feature_sets.items():
            label=family+'/'+arm
            selection=F.select_ridge(x[fit],data['truth'][fit],np.broadcast_to(ids[:,None],shape)[fit],p['split']['train'],(1,2),False,label)
            degree=selection['selected']['degree'];alpha=selection['selected']['alpha']
            raw,_,_=F.apply_ridge(x,data['truth'],fit,common,data['times'],degree,alpha)
            predictions[label]=raw;selections[label]=selection
    reports={}; populations={}
    for split in ('train','validation','held'):
        for noise in p['noise']:
            key=split+'/'+noise;group=((splits==split)&(noises==noise))[:,None]*np.ones(shape,bool)
            strata={'all':group,'source_fresh':group&fresh,'critical_common':group&families['critical_contrast'],
                    'matched_common':group&families['correspondence_contrast'],'gap':group&~fresh,
                    'boundary':group&(data['truth']>-.02)&(data['truth']<.05),
                    'unstable':group&(data['truth']<0),'farther_stable':group&(data['truth']>=.10)}
            reports[key]={name:{s:metric(data['truth'],pred,mask,ids) for s,mask in strata.items()} for name,pred in predictions.items()}
            unique,counts=np.unique(reasons[group],return_counts=True)
            common=group&families['correspondence_contrast']
            cpop=group&masks['valid_c']
            populations[key]=dict(total=int(group.sum()),source_fresh=int((group&fresh).sum()),gap=int((group&~fresh).sum()),
                critical_valid=int(cpop.sum()),matched_valid=int(common.sum()),reasons=dict(zip(unique.tolist(),counts.tolist())),
                matched_is_critical=int((common&matches).sum()),dominance_switch_windows=int((cpop&(switches>0)).sum()),
                critical_vs_rest_energy_median=float(np.median(support[cpop,0]/np.maximum(support[cpop,1],1e-30))) if cpop.any() else None,
                critical_vs_noise_energy_median=float(np.median(support[cpop,0]/np.maximum(support[cpop,2],1e-30))) if cpop.any() else None)
    ordered=list(predictions)
    save_npz(OUT/'predictions.npz',predictions=np.stack([predictions[k] for k in ordered]),names=json.dumps(ordered),
             truth=data['truth'],records=json.dumps(records),source_fresh=fresh,critical_common=families['critical_contrast'],
             matched_common=families['correspondence_contrast'],reason=reasons,support=support,matches_critical=matches,switches=switches,
             critical_features=arrays['critical'],matched_features=arrays['matched'],measured_features=data['fitter_features'])
    result=dict(scope=p['scope'],protocol_sha256=sha(OUT/'protocol.json'),implementation_sha256=sha(__file__),
                input_hashes=p['inputs'],shard_hashes=shard_hashes,shard_stats=shard_stats,statuses=statuses,
                selections=selections,populations=populations,metrics=reports,
                predictions_sha256=sha(OUT/'predictions.npz'),limits=p['limits'],operational_model_changed=False)
    save_json(OUT/'results.json',result)
    print('HEADROOM_ORACLE_COMPLETE',json.dumps(statuses),flush=True)
    for noise in ('clean','iid1','correlated1'):
        print('HELD_ORACLE',noise,json.dumps({name:reports['held/'+noise][name]['source_fresh']['equal_design_mae'] for name in ordered}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('replay','extract','score'));ap.add_argument('--design');ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    if args.phase=='replay':replay(args.design)
    elif args.phase=='extract':extract(args.workers)
    else:
        if (OUT/'results.json').exists():raise RuntimeError('Completed results are immutable')
        score()


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
