"""Report temporal comparison, sparse input coverage and extrapolation errors."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).resolve().parent
OUT=SCRIPT/'temporal_probe_20261007'


def summary(prediction,truth,selected):
    error=prediction-truth
    if not selected.any():
        return dict(frames=0,mae=None,signed_error=None,over_0_03_rate=None)
    return dict(frames=int(selected.sum()),mae=float(abs(error[selected]).mean()),
                signed_error=float(error[selected].mean()),over_0_03_rate=float((error[selected]>.03).mean()))


def run():
    result=json.loads((OUT/'results.json').read_text())
    assert result['source_sha256']==hashlib.sha256((SCRIPT/'temporal_probe.py').read_bytes()).hexdigest()
    assert result['protocol']==json.loads((OUT/'protocol.json').read_text())
    for name,expected in result['helper_source_sha256'].items():
        assert hashlib.sha256((SCRIPT/name).read_bytes()).hexdigest()==expected
    with np.load(OUT/'predictions.npz') as z:
        prediction=z['predictions'].copy(); truth=z['truth'].copy(); observed=z['observed'].copy()
        age=z['age'].copy(); times=z['times'].copy(); records=json.loads(str(z['records']))
        assert json.loads(str(z['protocol']))==result['protocol']
    assert np.isfinite(prediction).all()
    assert prediction.shape==(3,len(records),len(times))
    held=np.array([r['split']=='held' for r in records])
    seen=np.maximum.accumulate(observed,axis=1)
    analysis=dict(scope=result['protocol']['scope'],source_hash_exact=True,models={},coverage={},crossing_input_events=[],
        result_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest())
    for model,point in zip(('framewise','gru','time_only'),prediction):
        analysis['models'][model]={
            'all_held':summary(point,truth,np.broadcast_to(held[:,None],truth.shape)),
            'observed_held':summary(point,truth,held[:,None]&observed),
            'never_observed_held':summary(point,truth,held[:,None]&~seen),
            'gap_over_half_second_held':summary(point,truth,held[:,None]&seen&(age>.5)),
            'observed_stable_last2pct':summary(point,truth,held[:,None]&observed&(truth>0)&(truth<.02)),
            'unstable':summary(point,truth,held[:,None]&(truth<0)),
            'per_held_design':{design:summary(point,truth,np.broadcast_to(np.array([r['split']=='held' and r['design']==design for r in records])[:,None],truth.shape))
                for design in result['protocol']['split']['held']}}
    for noise in ('clean','iid1'):
        rows=np.array([r['split']=='held' and r['noise']==noise for r in records])
        analysis['coverage'][noise]=dict(frames=int(observed[rows].size),observed_frames=int(observed[rows].sum()),
            observation_fraction=float(observed[rows].mean()),never_observed_frames=int((~seen[rows]).sum()),
            gap_over_half_second_frames=int((seen[rows]&(age[rows]>.5)).sum()))
    for i,record in enumerate(records):
        if record['split']!='held' or record['profile']!='cross':
            continue
        schedule=record['schedule']; knots=np.array(schedule['knots']); ratios=np.array(schedule['ratios'])
        index=int(np.flatnonzero(ratios>=1)[0]); left=index-1
        crossing=float(knots[left]+(1-ratios[left])/(ratios[index]-ratios[left])*(knots[index]-knots[left]))
        approach=(times>=max(0.,crossing-3))&(times<=crossing-1)
        accepted=observed[i]&approach
        analysis['crossing_input_events'].append(dict(design=record['design'],replicate=record['replicate'],noise=record['noise'],
            crossing=crossing,at_least_one_second_pre_crossing_windows=int(accepted.sum()),
            most_recent_observation_before_crossing=float(times[observed[i]&(times<=crossing)][-1]) if (observed[i]&(times<=crossing)).any() else None))
    diagnostics=result['frozen_baseline_diagnostics']; queries=diagnostics['boundary_queries']
    credible=[r for r in queries if r['physically_credible']]; in_range=[r for r in credible if r['max_abs_standardized']<=5]
    analysis['frozen_baseline_diagnostics']=dict(clip_residuals=diagnostics['clip_residuals'],boundary_queries=len(queries),
        credible_boundary_queries=len(credible),max5sigma_credible_boundary_queries=len(in_range),
        credible_boundary_signed_errors=[r['prediction']-r['margin'] for r in credible],
        max5sigma_credible_boundary_signed_errors=[r['prediction']-r['margin'] for r in in_range],
        note='Frozen baseline supplements use original512ms features. Reference critical poles and training-support descriptions are evaluation-only; no predictor changed.')
    analysis['paired_gru_minus_framewise']={key:analysis['models']['gru'][key]['mae']-analysis['models']['framewise'][key]['mae']
        for key in ('all_held','observed_held','gap_over_half_second_held') if analysis['models']['gru'][key]['mae'] is not None and analysis['models']['framewise'][key]['mae'] is not None}
    analysis['limitations']=['Two held designs, one initialization, one fixed design split; no robust architecture superiority claim',
        'Repeated original TRAIN use, not reserved/blind evaluation',
        'All three selected checkpoints are at the200-epoch limit; convergence is not established',
        'Raw optimistic_positive_margin_rate is point>0; it denotes false-stable prediction only within the unstable subset',
        'Models output predictions through missing-input gaps; this is extrapolation, not measured availability or confidence',
        'No calibrated warning policy, no gap-filling safety certification, no physical harmful-amplitude or deployment qualification',
        'Randomized schedules and time-only control reduce fixed-clock confounding but cannot prove deployment transfer']
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    print('TEMPORAL_ANALYSIS_COMPLETE',json.dumps({k:{m:v[m] for m in ('all_held','observed_held','gap_over_half_second_held')} for k,v in analysis['models'].items()}))
    print('TEMPORAL_INPUT_COVERAGE',json.dumps(analysis['coverage']))
    print('TEMPORAL_CROSSING_INPUTS',json.dumps(analysis['crossing_input_events']))
    print('FROZEN_BASELINE_SUPPLEMENTS',json.dumps(analysis['frozen_baseline_diagnostics']))


if __name__=='__main__':
    run()
