"""Summarize paired diagnostic evidence without changing model or policy."""
import json
from pathlib import Path
import hashlib
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from mechanics.p4_margin_estimation.operational import alarm_episodes

SCRIPT = Path(__file__).resolve().parent
OUT = SCRIPT/'diagnostic_matrix_20261007'


def groups(rows, keys, predicate=lambda row: True):
    result = {}
    for row in rows:
        if predicate(row):
            key = '/'.join(str(row[k]) for k in keys)
            result.setdefault(key, []).append(row)
    return result


def estimates(rows):
    def median(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(values)) if values else None
    return dict(windows=len(rows), recovered=sum(r['recovered'] for r in rows),
                identified=sum(r['identified'] for r in rows),
                abstentions=sum(not r['identified'] for r in rows),
                false_positive_growth=sum(r['false_positive_growth'] for r in rows),
                median_alpha_error=median('alpha_error'), median_frequency_relative_error=median('frequency_relative_error'))


def hysteresis():
    fit=json.loads((SCRIPT/'o1_corrected/trajectory_fit.json').read_text())
    thresholds=json.loads((SCRIPT/'o1_corrected/trajectory_calibration.json').read_text())['thresholds']
    rows=[]
    for design in fit['partitions']['calibration']:
        with np.load(SCRIPT/'o1_corrected'/f'trajectories_{design}.npz') as z:
            descriptions=json.loads(str(z['descriptors']))
            for i,description in enumerate(descriptions):
                if description['profile']!='safe_dwell':
                    continue
                start,stop=z['offsets'][i:i+2]
                delivery=z['delivery'][start:stop]
                available=z['available'][start:stop]
                score=z['score'][start:stop]
                for name in ('episodes_per_hour_1.0','legacy_clip1'):
                    for clears in (1,5):
                        duration=description['duration']
                        episodes=alarm_episodes(delivery,score>=thresholds[name],available,clear_updates=clears,
                            operating_end=max(duration,float(delivery[-1])))
                        seconds=float(np.maximum(np.minimum(episodes[:,1],duration)-np.minimum(episodes[:,0],duration),0).sum())
                        rows.append(dict(noise=description['noise'],forcing=description['forcing'],policy=name,clear_updates=clears,
                            episodes=int((episodes[:,0]<=duration).sum()),alarm_seconds=seconds))
    return {key:dict(episodes=sum(r['episodes'] for r in selected),alarm_seconds=sum(r['alarm_seconds'] for r in selected))
        for key,selected in groups(rows,['noise','forcing','policy','clear_updates']).items()}


def run():
    path = OUT/'all_results.json'
    result = json.loads(path.read_text()); cached = result['cached']; physical = result['physical']
    source = SCRIPT/'diagnostic_matrix.py'
    assert result['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result['protocol'] == json.loads((OUT/'protocol.json').read_text())
    assert len(cached['contract']) == 2
    assert {r['design'] for r in physical['excitation']} == {'D000042','D000053','D000007','D000033'}
    analysis = dict(scope=result['protocol']['scope'], source_hash_exact=True,
                    evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), hypotheses={})
    hypotheses = analysis['hypotheses']
    information = {}
    for name, rows in groups(cached['information'], ['profile','forcing','noise']).items():
        eligible = [r for r in rows if r['guard'] and r['baseline_available']]
        rejected = [r for r in eligible if not r['predictive']]
        oracle_good = [r for r in rejected if r['oracle_skill'] >= .5]
        accepted = [r for r in eligible if r['predictive']]
        information[name] = dict(windows=len(rows), guarded_identified=len(eligible), predictive_accepted=len(accepted),
            rejected=len(rejected), rejected_oracle_good=len(oracle_good),
            rejected_oracle_good_prediction_mae=float(np.mean([abs(r['prediction_error']) for r in oracle_good])) if oracle_good else None,
            accepted_oracle_bad=sum(r['oracle_skill'] < .5 for r in accepted),
            rejected_oracle_median=float(np.median([r['oracle_skill'] for r in rejected])) if rejected else None,
            oracle_good_by_pulse_age={label:sum(r['oracle_skill'] >= .5 for r in rejected if lower <= r['pulse_age'] < upper)
                for label,lower,upper in [('early',0,.2),('middle',.2,.6),('late',.6,2)]},
            oracle_good_last_5pct=sum(r['oracle_skill'] >= .5 and 0 < r['margin'] < .05 for r in rejected))
    hypotheses['1_information_vs_gate'] = information
    aperture = {key:estimates(rows) for key,rows in groups(cached['aperture'], ['profile','forcing','noise','placement','samples'], lambda r:r['command_free']).items()}
    pre_crossing = {key:estimates(rows) for key,rows in groups(cached['aperture'], ['noise','placement','samples'], lambda r:r['command_free'] and r['pre_crossing'] and r['forcing']=='intermittent_physical_pulses').items()}
    hypotheses['2_aperture'] = dict(all_free_windows=aperture, pre_crossing_at_least_one_second=pre_crossing,
                                    limitation='Start-aligned and end-aligned comparisons keep lag64. Not a calibrated warning policy or compute-latency qualification.')
    hypotheses['3_forcing_vs_cap'] = dict(amplitude_replay=physical['amplitude_replay'],
        unsaturated_onset_positive_growth=sum(r['time']==3.011 and r['cap_hits']==0 and r['false_positive_growth'] for r in cached['amplitude']),
        unsaturated_onset_windows=sum(r['time']==3.011 and r['cap_hits']==0 for r in cached['amplitude']),
        onset_examples=[r for r in cached['amplitude'] if r['time']==3.011 and r['noise']=='clean'],
        free_decay={key:estimates(rows) for key,rows in groups(cached['amplitude'], ['amplitude','noise'], lambda r:r['time']==3.611).items()})
    hypotheses['4_noise'] = dict(matched_aperture_comparisons={key:estimates(rows) for key,rows in groups(cached['aperture'], ['noise','placement','samples'], lambda r:r['command_free'] and r['forcing']=='intermittent_physical_pulses').items()},
        floor={key:estimates(rows) for key,rows in groups(cached['noise_floor'], ['profile','family','level','time']).items()},
        absolute_noise_rms_m=sorted(set(r['absolute_noise_rms_m'] for r in cached['noise_floor'])),
        limitation='Paired iid/correlated draws have matched realized per-channel marginal RMS, correlation target0.8. Levels are fixed fractions of actual initial calibration response, never instantaneous shrinking response.')
    hypotheses['6_conditioning'] = {key:estimates(rows) for key,rows in groups(cached['conditioning'], ['noise','variant'], lambda r:'control' not in r).items()}
    hypotheses['7_order'] = {key:estimates(rows) for key,rows in groups(cached['order'], ['noise','maximum'], lambda r:'control' not in r).items()}
    hypotheses['negative_controls'] = {family:{key:dict(windows=len(rows), false_physical_estimates=sum(r['false_physical_estimate'] for r in rows),
        false_positive_growth=sum(r['false_positive_growth'] for r in rows)) for key,rows in groups(cached[family], ['noise', 'variant' if family=='conditioning' else 'maximum'], lambda r:r.get('control')=='noise_only').items()}
        for family in ('conditioning','order')}
    hypotheses['10_contract'] = cached['contract']
    # Safe/fast profiles share identical initial waveform segments; do not count those twice.
    unique={}
    for row in cached['ambiguity']:
        key=(row['design'],row['time'],row['truth'],row['prediction'],tuple(row['nearest_distances']))
        unique.setdefault(key,row)
    neighbors=list(unique.values())
    # Declare a simple descriptive subset, not a confidence/calibration criterion.
    reasonably_in_range = [r for r in neighbors if r['max_abs_standardized'] <= 5.]
    hypotheses['11_representation_vs_readout'] = dict(raw_queries=len(cached['ambiguity']),queries=len(neighbors), max5sigma_queries=len(reasonably_in_range),
        all_queries=neighbors, max5sigma_signed_errors=[r['prediction']-r['truth'] for r in reasonably_in_range],
        max5sigma_neighbor_label_spans=[max(r['neighbor_margins'])-min(r['neighbor_margins']) for r in reasonably_in_range],
        limitation='Cross-design nearest neighbors use fit designs only. Frequency/damping recovery and predictive gate restrict query credibility; max5sigma is descriptive, not a calibrated support test. Nearby labels do not prove non-identifiability.')
    attribution = result['policy']['attribution']; dense = result['policy']['candidate_audit']
    best = {str(b):min((r for r in dense if r['maximum_episode_rate']<=b), key=lambda r:r['threshold']) for b in (1.,10.,100.)}
    hypotheses['12_policy'] = dict(dense_calibration_minima=best,
        extremes_by_family={key:dict(windows=sum(r['windows'] for r in rows), below_minus_one=sum(r['margin_below_minus_one'] for r in rows),
            max_score=max((r['max_score'] for r in rows if r['max_score'] is not None), default=None),
            total_episodes=sum(r['episodes'] for r in rows), mean_alarm_active_fraction=float(np.mean([r['alarm_active_fraction'] for r in rows])))
            for key,rows in groups(attribution,['noise','forcing','category','threshold']).items()},
        limitation='Dense grid uses calibration only, leaves policy unchanged. Category-isolated alarms retain original hysteresis; missing updates hold episodes, and counts are not additive.')
    hypotheses['12_policy']['hysteresis_clearing_counterfactual']=hysteresis()
    hypotheses['12_policy']['waveform_attribution_subset']={
        key:dict(windows=len(rows),oracle_good=sum(r['oracle_skill']>=.5 for r in rows),
            noise_dominated=sum(r['noise']!='clean' and r['oracle_skill']<.01 for r in rows),
            predictive_accepted=sum(r['predictive'] for r in rows))
        for key,rows in groups(cached['information'],['profile','forcing','noise'],lambda r:r['guard'] and r['baseline_available']).items()}
    hypotheses['5_excitation'] = {key:estimates(rows) for key,rows in groups(physical['excitation'], ['confirmation','ratio','excitation','noise','samples']).items()}
    hypotheses['8_nonstationarity'] = dict(rows=physical['nonstationarity'], summary={key:estimates(rows) for key,rows in groups(physical['nonstationarity'], ['condition','noise']).items()})
    hypotheses['9_sampling'] = dict(rows=physical['sampling'], summary={key:estimates(rows) for key,rows in groups(physical['sampling'], ['condition','duration']).items()},
        limitation='New physical4k acquisitions, matched64ms Hankel lag. Offline zero-phase polyphase filtering is not a deployable causal monitor; no filter/inference delay qualification.')
    detector = cached['detector']
    hypotheses['waveform_detector'] = {key:dict(onsets=sum(len(r['command_onsets']) for r in rows), detected_onsets=sum(len(r['command_onsets'])-r['missed_onsets'] for r in rows),
        false_detections=sum(r['false_detections'] for r in rows), delays=[v for r in rows for v in r['delays'] if v is not None])
        for key,rows in groups(detector,['noise','forcing']).items()}
    coverage = pre_crossing['iid1/start_aligned/128']
    analysis['temporal_decision'] = dict(noisy_128ms_pre_crossing_recovered=coverage['recovered'], noisy_128ms_pre_crossing_windows=coverage['windows'],
        status='Requires randomized continuous trajectory experiment' if coverage['recovered'] else 'Prerequisite fails on checked noisy windows',
        limitation='Pole recovery in isolated pulse windows is not sufficient evidence for informative continuous inputs or margin separation. No GRU trained on these fixed schedules.')
    analysis['deployment_missing'] = ['Measured sensor noise PSD, marginal RMS, correlation, drift and units', 'Representative actuator availability/log timestamps or independent measured input',
        'ADC sampling/anti-aliasing transfer function', 'Harmful-amplitude limit and required crossing lead', 'Causal filter, inference, acquisition and communication latency']
    analysis['counts'] = {key:len(rows) for key,rows in cached.items()}
    analysis['physical_counts'] = {key:len(rows) for key,rows in physical.items()}
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    print('DIAGNOSTIC_ANALYSIS_COMPLETE', json.dumps(analysis['counts']), json.dumps(analysis['physical_counts']))
    print('PRE_CROSSING_POLE_RECOVERY', json.dumps(pre_crossing))
    print('POLICY_DENSE_MINIMA', json.dumps(best))
    print('CONDITIONING', json.dumps(hypotheses['6_conditioning']))
    print('NEGATIVE_CONTROLS', json.dumps(hypotheses['negative_controls']))
    print('TEMPORAL_DECISION', json.dumps(analysis['temporal_decision']))
    print('HYSTERESIS_COUNTERFACTUAL',json.dumps(hypotheses['12_policy']['hysteresis_clearing_counterfactual']))


if __name__ == '__main__':
    run()
